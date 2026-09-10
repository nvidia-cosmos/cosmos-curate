# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract tests for the direct-``clips.lance`` embedding schemas and group batch builders.

The builders emit what a fill worker hands to ``update_columns``: exactly one column
group's fields and NO identity column, one row per input row in input order. That
key-free, cardinality-preserving shape is what lets the worker attach ``clip_id``
positionally, so these tests pin row count, column set, and per-row null masking.
Image and action carry a per-row ``valid`` mask so a decode / geometry failure
persists as an all-NULL (pending / retryable) group instead of dropping the row;
text has no mask (it never fails per row).
"""

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    ACTION_DIM,
    EMBED_SOURCE_COLUMNS,
    EMBED_SOURCE_ROW,
    EMBEDDING_COLUMN_GROUPS,
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
    _as_matrix,
    action_columns_batch,
    descriptor_batch,
    image_columns_batch,
    text_columns_batch,
)

_DESCRIPTOR_DIM = 600


def test_text_columns_batch_carries_two_vectors_and_no_key() -> None:
    """The text group batch is exactly its group's fields: two vectors + the model id."""
    subtask = np.zeros((2, TEXT_DIM), dtype=np.float32)
    task = np.ones((2, TEXT_DIM), dtype=np.float32)
    table = text_columns_batch(2, subtask, task, "BAAI/bge-small-en-v1.5")

    assert table.schema.names == list(TEXT_COLUMN_GROUP.field_names)
    assert table.num_rows == 2
    assert table.column("embedding_text_subtask").type == pa.list_(pa.float32(), TEXT_DIM)
    assert table.column("embedding_text_model_id").to_pylist() == ["BAAI/bge-small-en-v1.5"] * 2


def test_image_columns_batch_all_valid_fills_every_row() -> None:
    """With every row valid, the image batch carries a vector and the model id per row."""
    vectors = np.ones((3, IMAGE_DIM), dtype=np.float32)
    valid = np.ones(3, dtype=np.bool_)
    table = image_columns_batch(3, vectors, "facebook/dinov2-small", valid)

    assert table.schema.names == list(IMAGE_COLUMN_GROUP.field_names)
    assert table.column("embedding_image").null_count == 0
    assert table.column("embedding_image_model_id").to_pylist() == ["facebook/dinov2-small"] * 3


def test_image_columns_batch_invalid_row_is_all_null_but_kept() -> None:
    """An invalid row is kept in place, with BOTH its vector and its model id nulled.

    This is the cardinality-preserving contract the positional key attachment relies
    on: a decode failure persists as the pending / retryable NULL group at the same
    position rather than dropping the row and shifting every row after it.
    """
    vectors = np.zeros((2, IMAGE_DIM), dtype=np.float32)
    valid = np.array([True, False], dtype=np.bool_)
    table = image_columns_batch(2, vectors, "m", valid)

    assert table.num_rows == 2
    assert table.column("embedding_image").is_valid().to_pylist() == [True, False]
    assert table.column("embedding_image_model_id").to_pylist() == ["m", None]


def test_action_columns_batch_invalid_row_nulls_vector_and_provenance() -> None:
    """An invalid action row nulls its vector, descriptor version, AND fingerprint together."""
    vectors = np.zeros((2, ACTION_DIM), dtype=np.float32)
    valid = np.array([False, True], dtype=np.bool_)
    table = action_columns_batch(2, vectors, "descv1", "fp-abc", valid)

    assert table.schema.names == list(ACTION_COLUMN_GROUP.field_names)
    assert table.column("embedding_action").is_valid().to_pylist() == [False, True]
    assert table.column("embedding_action_descriptor_version").to_pylist() == [None, "descv1"]
    assert table.column("embedding_action_pca_fingerprint").to_pylist() == [None, "fp-abc"]


def test_descriptor_batch_matches_descriptor_row_and_masks_invalid() -> None:
    """The descriptor transport is ``action_data_uri`` + a per-row-nullable descriptor."""
    descriptors = np.zeros((2, _DESCRIPTOR_DIM), dtype=np.float32)
    valid = np.array([True, False], dtype=np.bool_)
    table = descriptor_batch(["s3://x.bin", "s3://y.bin"], descriptors, _DESCRIPTOR_DIM, valid)

    assert table.schema.names == ["action_data_uri", "descriptor"]
    assert table.column("action_data_uri").to_pylist() == ["s3://x.bin", "s3://y.bin"]
    assert table.column("descriptor").is_valid().to_pylist() == [True, False]


def test_empty_image_batch_yields_zero_rows_with_the_group_schema() -> None:
    """A zero-row batch is still a correctly-typed group batch, not an error."""
    empty = np.zeros((0, IMAGE_DIM), dtype=np.float32)
    table = image_columns_batch(0, empty, "m", np.zeros(0, dtype=np.bool_))

    assert table.num_rows == 0
    assert table.schema.names == list(IMAGE_COLUMN_GROUP.field_names)


def test_empty_descriptor_batch_yields_zero_rows() -> None:
    """A zero-row descriptor batch builds a valid empty list column (offsets need one entry)."""
    table = descriptor_batch(
        [], np.zeros((0, _DESCRIPTOR_DIM), dtype=np.float32), _DESCRIPTOR_DIM, np.zeros(0, np.bool_)
    )

    assert table.num_rows == 0
    assert table.schema.names == ["action_data_uri", "descriptor"]


def test_image_columns_batch_rejects_wrong_width_matrix() -> None:
    """A wrong-width vector matrix is rejected by the shared shape guard, not deep in Arrow."""
    valid = np.ones(2, dtype=np.bool_)
    with pytest.raises(ValueError, match="shape"):
        image_columns_batch(2, np.zeros((2, IMAGE_DIM + 1), dtype=np.float32), "m", valid)


def test_image_columns_batch_rejects_misaligned_validity_mask() -> None:
    """A mask that is not one entry per row is rejected: it would misalign every NULL.

    Without a key column the mask is the only thing saying which rows stayed pending,
    so a short mask must fail here rather than silently marking the wrong clips
    complete.
    """
    vectors = np.zeros((3, IMAGE_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="validity mask"):
        image_columns_batch(3, vectors, "m", np.ones(2, dtype=np.bool_))


def test_group_vectors_are_nullable_fixed_size_lists_with_expected_widths() -> None:
    """Each group's vector column is a nullable fixed-size list at the group's stored width."""
    assert TEXT_COLUMN_GROUP.schema.field("embedding_text_subtask").type == pa.list_(pa.float32(), TEXT_DIM)
    assert IMAGE_COLUMN_GROUP.schema.field("embedding_image").type == pa.list_(pa.float32(), IMAGE_DIM)
    assert ACTION_COLUMN_GROUP.schema.field("embedding_action").type == pa.list_(pa.float32(), ACTION_DIM)
    for group in EMBEDDING_COLUMN_GROUPS:
        for field in group.schema:
            assert field.nullable


def test_group_vector_list_child_is_nullable() -> None:
    """The fixed-size-list child is nullable so an all-NULL column survives ``add_columns``."""
    image_type = IMAGE_COLUMN_GROUP.schema.field("embedding_image").type
    assert image_type.field(0).nullable


def test_registered_groups_own_disjoint_namespaces() -> None:
    """No column name is claimed by two registered groups (private ``embedding_<name>`` prefix)."""
    seen: set[str] = set()
    for group in EMBEDDING_COLUMN_GROUPS:
        for name in group.field_names:
            assert name not in seen
            assert name.startswith(f"embedding_{group.name}")
            seen.add(name)


def test_as_matrix_rejects_transposed_matrix() -> None:
    """A (dim, rows) matrix has the right element count but the wrong shape."""
    with pytest.raises(ValueError, match="shape"):
        _as_matrix(np.zeros((IMAGE_DIM, 3), dtype=np.float32), 3, IMAGE_DIM)


def test_as_matrix_rejects_higher_rank_stack() -> None:
    """A (rows, 1, dim) stack is rejected rather than silently flattened."""
    with pytest.raises(ValueError, match="matrix or a flat"):
        _as_matrix(np.zeros((3, 1, IMAGE_DIM), dtype=np.float32), 3, IMAGE_DIM)


def test_as_matrix_rejects_flat_vector_for_multiple_rows() -> None:
    """A flat buffer is accepted only for a single row, never for several."""
    with pytest.raises(ValueError, match="single row"):
        _as_matrix(np.zeros(2 * IMAGE_DIM, dtype=np.float32), 2, IMAGE_DIM)


def test_as_matrix_accepts_flat_vector_for_single_row() -> None:
    """A flat 1-D vector reshapes to (1, dim)."""
    out = _as_matrix(np.arange(IMAGE_DIM, dtype=np.float32), 1, IMAGE_DIM)
    assert out.shape == (1, IMAGE_DIM)


def test_embed_source_columns_derive_from_source_row() -> None:
    """The projected columns are exactly the source schema's field names, so they cannot drift."""
    assert tuple(EMBED_SOURCE_ROW.names) == EMBED_SOURCE_COLUMNS
