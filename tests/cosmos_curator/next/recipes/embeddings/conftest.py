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

"""Fixtures for the direct-``clips.lance`` embedding recipe tests.

Builds a tiny base-schema ``clips.lance`` (the columns the embedding legs read
from the source: key, media/action URIs, and the text label columns) so a test
can exercise schema widening, selection, and column publication against a real
Lance table without the robot-action-split producer. The base schema is a
faithful subset of that producer's ``OUTCOME_SCHEMA`` (``clip_uri`` /
``action_data_uri`` are ``large_string``); embedding columns are never seeded
here - a test adds them with :func:`add_group_columns` to model the
"column absent until its modality runs" contract.
"""

import pathlib
from collections.abc import Callable, Sequence

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# Base ``clips.lance`` schema used by the recipe tests: the exact columns the
# embedding source projection reads (see EMBED_SOURCE_ROW). Mirrors the producer
# OUTCOME_SCHEMA field types so the append-after-widening behaviour a test models
# here matches the real narrow-append path (clip_uri / action_data_uri are
# large_string; the label columns are non-null string).
CLIPS_BASE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string()),
        pa.field("action_data_uri", pa.large_string()),
        pa.field("source_dataset", pa.string(), nullable=False),
    ]
)

# Factory that returns a ``clips.lance`` URI. Keyword-only; see ``make_clips_table``.
ClipsTableFactory = Callable[..., str]


def _sentinel_uri(row_index: int, kind: str) -> str:
    """Return a deterministic non-empty media/action URI for a row."""
    return f"{kind}{row_index}.bin"


def add_group_columns(uri: str, group_schema: pa.Schema) -> None:
    """Add an all-nullable column group to ``uri`` as one metadata-only version."""
    dataset = lance.dataset(uri)
    dataset.add_columns(group_schema)


@pytest.fixture
def make_clips_table(tmp_path: pathlib.Path) -> ClipsTableFactory:
    """Return a builder for a base-schema ``clips.lance`` under ``tmp_path``.

    Keyword arguments (all optional):

    - ``rows``: number of clip rows (default 6);
    - ``rows_per_file``: ``max_rows_per_file`` so a test can force multiple
      fragments (default 3, i.e. two fragments at the default row count);
    - ``clip_ids``: per-row key override, so a test can model the duplicate
      ``clip_id`` the producer does not rule out. Defaults to ``c0..c{rows-1}``;
    - ``clip_uris`` / ``action_uris``: per-row override lists; an entry may be
      ``None`` or ``""`` to model a clip with no media / no action artifact.
      Defaults populate every row with a distinct non-empty URI;
    - ``name``: dataset directory name (default ``clips.lance``).

    Returns the dataset URI (a local path string). Cleanup is automatic through
    ``tmp_path``.
    """

    def build(  # noqa: PLR0913 -- a test data builder; each argument is one independent per-row column override
        *,
        rows: int = 6,
        rows_per_file: int = 3,
        clip_ids: Sequence[str] | None = None,
        clip_uris: Sequence[str | None] | None = None,
        action_uris: Sequence[str | None] | None = None,
        name: str = "clips.lance",
    ) -> str:
        keys = list(clip_ids) if clip_ids is not None else [f"c{i}" for i in range(rows)]
        uris = list(clip_uris) if clip_uris is not None else [_sentinel_uri(i, "clip") for i in range(rows)]
        actions = list(action_uris) if action_uris is not None else [_sentinel_uri(i, "act") for i in range(rows)]
        table = pa.table(
            {
                "clip_id": pa.array(keys, pa.string()),
                "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
                "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
                "clip_uri": pa.array(uris, pa.large_string()),
                "action_data_uri": pa.array(actions, pa.large_string()),
                "source_dataset": pa.array(["ds_under_test"] * rows, pa.string()),
            },
            schema=CLIPS_BASE_SCHEMA,
        )
        uri = str(tmp_path / name)
        lance.write_dataset(
            table,
            uri,
            max_rows_per_file=rows_per_file,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        return uri

    return build
