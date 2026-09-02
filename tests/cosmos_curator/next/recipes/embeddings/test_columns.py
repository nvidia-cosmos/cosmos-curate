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

"""Behavioural tests for the driver-side schema, state, and read functions over a clips table.

Every function under test takes a dataset the CALLER opened, so these tests are
pure Lance and run everywhere - nothing here touches Ray. Three concerns share the
file because they share the same fixture shape:

- SCHEMA: widening one group in (``ensure_embedding_columns``) and detaching one
  back out (``drop_embedding_group``), each a single metadata commit;
- STATE: the at-most-one-producer and staleness rules
  (``validate_embedding_group``);
- READ: the pending predicate (``pending_filter``), the filled count, and the
  one-column stream.

The bounded distinct-value read those state rules are built on belongs to
``lance_utils`` and is covered by that module's own suite, because the curation
leg reads the same columns through it.

The pending predicate is asserted by COUNTING the rows it matches against a real
table rather than by comparing SQL text, so a rewrite that preserves the meaning
does not redden the file.
"""

import pathlib
from collections.abc import Sequence

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_DIM,
    ACTION_GROUP_SCHEMA,
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    IMAGE_GROUP_SCHEMA,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
    TEXT_GROUP_SCHEMA,
)
from cosmos_curator.next.recipes.embeddings.columns import (
    count_filled,
    drop_embedding_group,
    ensure_embedding_columns,
    pending_filter,
    scan_column,
    validate_embedding_group,
)
from cosmos_curator.next.recipes.embeddings.modalities import _IMAGE_APPLICABILITY_FILTER
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

from .conftest import CLIPS_BASE_SCHEMA, ClipsTableFactory, add_group_columns


def _fixed_size_list(values: Sequence[list[float] | None], dim: int) -> pa.Array:
    """Build a ``fixed_size_list<float32, dim>`` array; a ``None`` entry is a null row."""
    return pa.array(values, type=pa.list_(pa.float32(), dim))


def _vector(seed: int, dim: int) -> list[float]:
    """Return a deterministic non-null vector of width ``dim``."""
    return [float(seed)] * dim


def _write_clips_dataset(
    tmp_path: pathlib.Path,
    *,
    rows: int,
    clip_uris: Sequence[str | None] | None = None,
    text: dict[str, list] | None = None,
    image: dict[str, list] | None = None,
    action: dict[str, list] | None = None,
    rows_per_file: int = 3,
    name: str = "clips.lance",
) -> str:
    """Write a ``clips.lance`` with base columns and optionally populated embedding groups.

    Each of ``text`` / ``image`` / ``action`` (when given) carries per-row column
    value lists (a vector list or ``None`` per row, and the provenance strings),
    so a test can model empty, complete, corrupt, or stale group rows directly.
    Unlike ``add_group_columns`` (metadata-only, all NULL), this writes the values
    into the base table so validation and selection have data to read.
    """
    clip_ids = [f"c{i}" for i in range(rows)]
    uris = list(clip_uris) if clip_uris is not None else [f"clip{i}.bin" for i in range(rows)]
    arrays: dict[str, pa.Array] = {
        "clip_id": pa.array(clip_ids, pa.string()),
        "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
        "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
        "clip_uri": pa.array(uris, pa.large_string()),
        "action_data_uri": pa.array([f"act{i}.bin" for i in range(rows)], pa.large_string()),
        "source_dataset": pa.array(["ds_under_test"] * rows, pa.string()),
    }
    fields = list(CLIPS_BASE_SCHEMA)
    if text is not None:
        arrays["embedding_text_subtask"] = _fixed_size_list(text["subtask"], TEXT_DIM)
        arrays["embedding_text_task"] = _fixed_size_list(text["task"], TEXT_DIM)
        arrays["embedding_text_model_id"] = pa.array(text["model_id"], pa.string())
        fields += list(TEXT_GROUP_SCHEMA)
    if image is not None:
        arrays["embedding_image"] = _fixed_size_list(image["image"], IMAGE_DIM)
        arrays["embedding_image_model_id"] = pa.array(image["model_id"], pa.string())
        fields += list(IMAGE_GROUP_SCHEMA)
    if action is not None:
        arrays["embedding_action"] = _fixed_size_list(action["action"], ACTION_DIM)
        arrays["embedding_action_descriptor_version"] = pa.array(action["descriptor_version"], pa.string())
        arrays["embedding_action_pca_fingerprint"] = pa.array(action["fingerprint"], pa.string())
        fields += list(ACTION_GROUP_SCHEMA)
    schema = pa.schema(fields)
    table = pa.table({column: arrays[column] for column in schema.names}, schema=schema)
    uri = str(tmp_path / name)
    lance.write_dataset(table, uri, max_rows_per_file=rows_per_file, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri


def test_ensure_adds_only_enabled_group_in_one_version(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring one group adds exactly that group's fields in a single metadata version."""
    uri = make_clips_table()
    before = lance.dataset(uri).version

    added, commit_version = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    after = lance.dataset(uri)
    assert added == len(TEXT_COLUMN_GROUP.field_names)
    assert commit_version == before + 1
    assert after.version == before + 1
    names = set(after.schema.names)
    assert set(TEXT_COLUMN_GROUP.field_names) <= names
    assert not (set(IMAGE_COLUMN_GROUP.field_names) & names)


def test_text_only_ensure_leaves_other_groups_absent(make_clips_table: ClipsTableFactory) -> None:
    """A text-only ensure creates no image or action columns."""
    uri = make_clips_table()
    ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])
    names = set(lance.dataset(uri).schema.names)
    assert not any(name.startswith("embedding_image") for name in names)
    assert not any(name.startswith("embedding_action") for name in names)


def test_ensure_present_group_is_idempotent_noop(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring an already-present, exactly-matching group adds nothing and no new version."""
    uri = make_clips_table()
    add_group_columns(uri, TEXT_GROUP_SCHEMA)
    before = lance.dataset(uri).version

    added, commit_version = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    assert added == 0
    assert commit_version is None
    assert lance.dataset(uri).version == before


def test_ensure_one_group_present_another_absent_is_valid(make_clips_table: ClipsTableFactory) -> None:
    """Ensuring [text, image] with text already present adds only the image fields."""
    uri = make_clips_table()
    add_group_columns(uri, TEXT_GROUP_SCHEMA)

    added, _ = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP, IMAGE_COLUMN_GROUP])

    assert added == len(IMAGE_COLUMN_GROUP.field_names)
    names = set(lance.dataset(uri).schema.names)
    assert set(IMAGE_COLUMN_GROUP.field_names) <= names


def test_ensure_partial_group_fails(make_clips_table: ClipsTableFactory) -> None:
    """A group with only some of its fields present is a corrupt schema and fails."""
    uri = make_clips_table()
    add_group_columns(uri, pa.schema([TEXT_GROUP_SCHEMA.field("embedding_text_subtask")]))
    with pytest.raises(ValueError, match="partially present"):
        ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])


def test_ensure_wrong_dimension_fails(make_clips_table: ClipsTableFactory) -> None:
    """A present image vector of the wrong width fails the exact-match check."""
    uri = make_clips_table()
    wrong = pa.schema(
        [
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM + 1), nullable=True),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, wrong)
    with pytest.raises(ValueError, match="type"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_wrong_type_fails(make_clips_table: ClipsTableFactory) -> None:
    """A present image vector stored as a variable list (not fixed-size) fails."""
    uri = make_clips_table()
    wrong = pa.schema(
        [
            pa.field("embedding_image", pa.list_(pa.float32()), nullable=True),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, wrong)
    with pytest.raises(ValueError, match="type"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_wrong_nullability_fails(tmp_path: pathlib.Path) -> None:
    """A present image vector declared non-nullable fails the exact-match check."""
    rows = 2
    schema = pa.schema(
        [
            *list(CLIPS_BASE_SCHEMA),
            pa.field("embedding_image", pa.list_(pa.float32(), IMAGE_DIM), nullable=False),
            pa.field("embedding_image_model_id", pa.string(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "clip_id": pa.array([f"c{i}" for i in range(rows)], pa.string()),
            "task_name": pa.array([f"t{i}" for i in range(rows)], pa.string()),
            "subtask_name": pa.array([f"s{i}" for i in range(rows)], pa.string()),
            "clip_uri": pa.array([f"u{i}" for i in range(rows)], pa.large_string()),
            "action_data_uri": pa.array([f"a{i}" for i in range(rows)], pa.large_string()),
            "source_dataset": pa.array(["ds"] * rows, pa.string()),
            "embedding_image": _fixed_size_list([_vector(i, IMAGE_DIM) for i in range(rows)], IMAGE_DIM),
            "embedding_image_model_id": pa.array(["m"] * rows, pa.string()),
        },
        schema=schema,
    )
    uri = str(tmp_path / "clips.lance")
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    with pytest.raises(ValueError, match="nullable"):
        ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])


def test_ensure_tolerates_unknown_future_group(make_clips_table: ClipsTableFactory) -> None:
    """An unrelated future embedding group on the table does not block ensuring a known one."""
    uri = make_clips_table()
    future = pa.schema(
        [
            pa.field("embedding_future_x", pa.list_(pa.float32(), 8), nullable=True),
            pa.field("embedding_future_model_id", pa.string(), nullable=True),
        ]
    )
    add_group_columns(uri, future)

    added, _ = ensure_embedding_columns(lance.dataset(uri), [TEXT_COLUMN_GROUP])

    names = set(lance.dataset(uri).schema.names)
    assert added == len(TEXT_COLUMN_GROUP.field_names)
    assert {"embedding_future_x", "embedding_future_model_id"} <= names


def test_dropping_a_group_detaches_every_field_and_reports_the_count(make_clips_table: ClipsTableFactory) -> None:
    """Dropping a present group removes exactly its own columns and returns how many."""
    uri = make_clips_table()
    add_group_columns(uri, pa.schema([*list(IMAGE_GROUP_SCHEMA), *list(TEXT_GROUP_SCHEMA)]))

    dropped = drop_embedding_group(lance.dataset(uri), IMAGE_COLUMN_GROUP)

    names = set(lance.dataset(uri).schema.names)
    assert dropped == len(IMAGE_COLUMN_GROUP.field_names)
    assert not (set(IMAGE_COLUMN_GROUP.field_names) & names)
    assert set(TEXT_COLUMN_GROUP.field_names) <= names


def test_dropping_a_partially_present_group_clears_the_field_that_exists(
    make_clips_table: ClipsTableFactory,
) -> None:
    """A half-installed group is still fully cleared rather than failing the whole reset.

    Lance's ``drop_columns`` rejects a name the schema does not carry and then drops
    NOTHING, so a group narrowed to its present fields is the only way a corrupt
    half-installed group can be reset back to absent.
    """
    uri = make_clips_table()
    add_group_columns(uri, pa.schema([IMAGE_GROUP_SCHEMA.field("embedding_image")]))

    dropped = drop_embedding_group(lance.dataset(uri), IMAGE_COLUMN_GROUP)

    assert dropped == 1
    assert "embedding_image" not in set(lance.dataset(uri).schema.names)


def test_dropping_an_absent_group_reports_zero_and_commits_no_version(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Resetting a group that was never added is a no-op that leaves no version behind.

    ``drop_columns([])`` does not raise but does commit, so a reset of an unused
    modality would otherwise litter the table's history with empty versions.
    """
    uri = make_clips_table()
    dataset = lance.dataset(uri)
    before = dataset.version

    dropped = drop_embedding_group(dataset, IMAGE_COLUMN_GROUP)

    assert dropped == 0
    assert lance.dataset(uri).version == before


def test_a_dropped_group_can_be_re_added_with_its_exact_types(tmp_path: pathlib.Path) -> None:
    """Re-adding a dropped group restores each field's stored type and nullability."""
    rows = 2
    uri = _write_clips_dataset(
        tmp_path,
        rows=rows,
        image={"image": [_vector(i, IMAGE_DIM) for i in range(rows)], "model_id": ["m"] * rows},
    )
    drop_embedding_group(lance.dataset(uri), IMAGE_COLUMN_GROUP)

    ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])

    schema = lance.dataset(uri).schema
    for expected in IMAGE_GROUP_SCHEMA:
        stored = schema.field(expected.name)
        assert stored.type == expected.type
        assert stored.nullable


def test_a_re_added_group_reads_null_rather_than_its_pre_drop_values(tmp_path: pathlib.Path) -> None:
    """The drop detaches the data too, so a refill starts from NULL and cannot resurrect old vectors.

    This is what makes a PARTIAL refill safe: the fragments a later run never
    reaches read NULL rather than the values the dropped basis or model produced.
    """
    rows = 2
    uri = _write_clips_dataset(
        tmp_path,
        rows=rows,
        image={"image": [_vector(i, IMAGE_DIM) for i in range(rows)], "model_id": ["m"] * rows},
    )
    drop_embedding_group(lance.dataset(uri), IMAGE_COLUMN_GROUP)
    ensure_embedding_columns(lance.dataset(uri), [IMAGE_COLUMN_GROUP])

    values = lance.dataset(uri).to_table(columns=list(IMAGE_COLUMN_GROUP.field_names))

    for column in values.itercolumns():
        assert column.null_count == rows


def test_validate_absent_group_fails(make_clips_table: ClipsTableFactory) -> None:
    """Validating a group whose columns were never added fails clearly."""
    uri = make_clips_table()
    with pytest.raises(ValueError, match="not present"):
        validate_embedding_group(lance.dataset(uri), TEXT_COLUMN_GROUP, expected_provenance=None)


def test_validate_empty_group_passes(tmp_path: pathlib.Path) -> None:
    """An all-NULL group has no complete rows and passes validation."""
    uri = _write_clips_dataset(
        tmp_path,
        rows=3,
        text={"subtask": [None, None, None], "task": [None, None, None], "model_id": [None, None, None]},
    )
    validate_embedding_group(
        lance.dataset(uri), TEXT_COLUMN_GROUP, expected_provenance={"embedding_text_model_id": "m"}
    )


def test_validate_complete_matching_producer_passes(tmp_path: pathlib.Path) -> None:
    """A complete group whose provenance matches the configured producer passes."""
    rows = 3
    uri = _write_clips_dataset(
        tmp_path,
        rows=rows,
        text={
            "subtask": [_vector(i, TEXT_DIM) for i in range(rows)],
            "task": [_vector(i, TEXT_DIM) for i in range(rows)],
            "model_id": ["m"] * rows,
        },
    )
    validate_embedding_group(
        lance.dataset(uri), TEXT_COLUMN_GROUP, expected_provenance={"embedding_text_model_id": "m"}
    )


def test_validate_stale_model_id_fails(tmp_path: pathlib.Path) -> None:
    """A complete group whose model id differs from the configured one is refused as stale.

    Filling the remaining rows would leave one group holding two models' vectors,
    which no consumer can compare, so the run stops and directs the operator to
    reset the group instead.
    """
    rows = 2
    uri = _write_clips_dataset(
        tmp_path,
        rows=rows,
        image={"image": [_vector(i, IMAGE_DIM) for i in range(rows)], "model_id": ["old-model"] * rows},
    )
    with pytest.raises(ValueError, match="stale"):
        validate_embedding_group(
            lance.dataset(uri), IMAGE_COLUMN_GROUP, expected_provenance={"embedding_image_model_id": "new-model"}
        )


def test_validate_multiple_producers_fails(tmp_path: pathlib.Path) -> None:
    """A group carrying two distinct model ids (two producers) fails validation."""
    uri = _write_clips_dataset(
        tmp_path,
        rows=2,
        image={"image": [_vector(0, IMAGE_DIM), _vector(1, IMAGE_DIM)], "model_id": ["m-a", "m-b"]},
    )
    with pytest.raises(ValueError, match="multiple producers"):
        validate_embedding_group(lance.dataset(uri), IMAGE_COLUMN_GROUP, expected_provenance=None)


def test_count_filled_counts_non_null_primary(tmp_path: pathlib.Path) -> None:
    """count_filled returns the number of rows whose primary vector is non-NULL."""
    uri = _write_clips_dataset(
        tmp_path,
        rows=3,
        image={"image": [_vector(0, IMAGE_DIM), None, _vector(2, IMAGE_DIM)], "model_id": ["m", None, "m"]},
    )
    assert count_filled(lance.dataset(uri), IMAGE_COLUMN_GROUP) == 2


def test_count_filled_is_zero_for_a_group_whose_columns_are_absent(make_clips_table: ClipsTableFactory) -> None:
    """Asking how much of a not-yet-added group is filled reads zero rather than failing.

    The action leg's outcome check counts a group that a failed run may never have
    widened, so an unresolvable column must answer "nothing is filled" instead of
    raising out of the SQL planner.
    """
    uri = make_clips_table()
    assert count_filled(lance.dataset(uri), IMAGE_COLUMN_GROUP) == 0


def test_pending_filter_selects_applicable_rows_whose_primary_vector_is_null(tmp_path: pathlib.Path) -> None:
    """A row is pending only when the modality can embed it and it holds no vector yet.

    Both undecodable-by-construction shapes are excluded so a future edit to
    ``_IMAGE_APPLICABILITY_FILTER`` cannot readmit them: an empty-string
    ``clip_uri`` (c1) and a NULL ``clip_uri`` (c2). c3 is already embedded, so only
    the applicable, vector-less c0 is owed work.
    """
    uri = _write_clips_dataset(
        tmp_path,
        rows=4,
        clip_uris=["clip0.bin", "", None, "clip3.bin"],
        image={
            "image": [None, None, None, _vector(3, IMAGE_DIM)],
            "model_id": [None, None, None, "m"],
        },
    )
    row_filter = pending_filter(IMAGE_COLUMN_GROUP, _IMAGE_APPLICABILITY_FILTER)

    matched = lance.dataset(uri).to_table(columns=["clip_id"], filter=row_filter)

    assert matched.column("clip_id").to_pylist() == ["c0"]


def test_pending_filter_without_an_applicability_predicate_matches_every_unembedded_row(
    tmp_path: pathlib.Path,
) -> None:
    """Text applies to every row, so its predicate degrades to the bare primary-vector-IS-NULL clause."""
    rows = 3
    uri = _write_clips_dataset(
        tmp_path,
        rows=rows,
        clip_uris=[""] * rows,
        text={"subtask": [None] * rows, "task": [None] * rows, "model_id": [None] * rows},
    )
    row_filter = pending_filter(TEXT_COLUMN_GROUP, None)

    # Every row lacks media, which must not exclude it from the text modality.
    assert lance.dataset(uri).count_rows(filter=row_filter) == rows


def test_scan_column_streams_only_the_requested_column_of_matching_rows(
    make_clips_table: ClipsTableFactory,
) -> None:
    """``scan_column`` projects one column and pushes the predicate into the scan."""
    uri = make_clips_table(rows=4, rows_per_file=2, action_uris=["a0.bin", "", "a2.bin", None])

    batches = list(
        scan_column(
            lance.dataset(uri),
            "action_data_uri",
            row_filter="action_data_uri IS NOT NULL AND action_data_uri != ''",
            batch_size=1,
        )
    )

    assert [batch.schema.names for batch in batches] == [["action_data_uri"]] * len(batches)
    values = [value for batch in batches for value in batch.column("action_data_uri").to_pylist()]
    assert values == ["a0.bin", "a2.bin"]


def test_scan_column_yields_nothing_when_no_row_matches(make_clips_table: ClipsTableFactory) -> None:
    """An empty match yields no batches at all, so a caller needs no zero-row special case."""
    uri = make_clips_table(rows=2, rows_per_file=2)

    batches = list(
        scan_column(lance.dataset(uri), "action_data_uri", row_filter="action_data_uri = 'absent'", batch_size=8)
    )

    assert batches == []
