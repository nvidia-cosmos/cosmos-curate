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

"""Atomic Lance publication for robot-action-split outcomes.

Uses the ``write_fragments`` + ``LanceDataset.commit`` pattern for crash-safe,
idempotent writes — the same approach as ``video_split/lance_sink.py``.
"""

from typing import Any

import lance
import pyarrow as pa
from lance.fragment import write_fragments

from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION, open_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target rows per Lance fragment. Downstream Ray Data stages (embedding, dedup)
# get one parallel task per fragment, so fragment count bounds GPU parallelism.
# This write is driver-side, so GPU count is unknown here — size fragments so
# total count comfortably exceeds any expected GPU fleet (e.g. 32-64 GPUs).
# 8k rows ≈ 16 fragments for a 130k-row dataset; expose via execution config
# if callers need to match a specific cluster size.
_ROWS_PER_FRAGMENT = 8_000

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

OUTCOME_SCHEMA = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("span_group_id", pa.string(), nullable=False),
        pa.field("view_name", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("episode_index", pa.int32(), nullable=False),
        pa.field("subtask_index", pa.int32(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("task_index", pa.int32(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("frame_start", pa.int32(), nullable=False),
        pa.field("frame_end", pa.int32(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("native_fps", pa.float64(), nullable=False),
        pa.field("episode_from_timestamp", pa.float64(), nullable=False),
        pa.field("clip_uri", pa.large_string()),
        pa.field("action_data_uri", pa.large_string()),
        pa.field("camera_motion_annotation", pa.large_string()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.large_string()),
    ]
)

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_outcomes_to_lance(
    outcomes: list[dict[str, Any]],
    *,
    lance_uri: str,
    storage_options: dict[str, str] | None = None,
) -> int:
    """Write successful outcomes as Lance fragments and commit.

    Only rows where ``status == 'success'`` are written.  Uses
    ``write_fragments`` + ``LanceDataset.commit`` for an atomic, crash-safe
    write that mirrors the ``video_split`` pattern.

    Args:
        outcomes: List of outcome dicts as returned by ``process_batch``.
        lance_uri: Target Lance dataset URI (local path or ``s3://...``).
        storage_options: Lance storage options (AWS credentials, endpoint, …).
            Obtain via ``get_lance_storage_options(lance_uri, profile_name=...)``.

    Returns:
        The Lance dataset version after the write.

    Raises:
        ValueError: When no successful outcome rows are provided.

    """
    success_rows = [o for o in outcomes if o.get("status") == "success"]
    if not success_rows:
        msg = "write_outcomes_to_lance: no successful outcomes to write"
        raise ValueError(msg)

    table = _build_table(success_rows)

    # Determine create vs append from the driver's current view of the dataset.
    dataset = open_dataset(lance_uri, storage_options=storage_options)
    if dataset is None:
        fragment_mode = "create"
        read_version = None
    else:
        fragment_mode = "append"
        read_version = dataset.version

    # Write fragments (uncommitted). max_rows_per_file causes write_fragments
    # to split the table into ceil(rows / N) fragments internally, giving
    # downstream Ray Data scans (embedding, dedup) enough partitions to
    # saturate available GPU parallelism.
    reader = pa.RecordBatchReader.from_batches(OUTCOME_SCHEMA, table.to_batches())
    fragments = write_fragments(
        reader,
        lance_uri,
        schema=OUTCOME_SCHEMA,
        mode=fragment_mode,
        max_rows_per_file=_ROWS_PER_FRAGMENT,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )

    # Commit all fragments in one atomic transaction.
    properties = {"kind": "robot-action-split", "snapshot": "clips"}
    if read_version is None:
        operation: lance.LanceOperation.BaseOperation = lance.LanceOperation.Overwrite(OUTCOME_SCHEMA, fragments)
        transaction = lance.Transaction(
            read_version=0,
            operation=operation,
            transaction_properties=properties,
        )
    else:
        operation = lance.LanceOperation.Append(fragments)
        transaction = lance.Transaction(
            read_version=read_version,
            operation=operation,
            transaction_properties=properties,
        )

    # TODO(Ray migration): handle the concurrent-create race.
    # When Ray flat_map is wired in, multiple workers may simultaneously see
    # dataset=None and each attempt an Overwrite.  Only one will win; the rest
    # will raise a conflict error that propagates uncaught here.  One fix is to
    # wrap this commit in try/except, detect the conflict, reopen the dataset,
    # and retry as Append.  The other is video_split's shape, where workers only
    # write fragments and a single driver-side commit owns the transaction.
    # Intentionally deferred: the current sequential pipeline makes this race
    # impossible in practice.
    committed = lance.LanceDataset.commit(
        lance_uri,
        transaction,
        storage_options=storage_options,
        enable_v2_manifest_paths=True,
    )
    return int(committed.version)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert a list of outcome dicts to a PyArrow Table conforming to OUTCOME_SCHEMA."""
    arrays: dict[str, list[Any]] = {field.name: [] for field in OUTCOME_SCHEMA}

    for row in rows:
        arrays["clip_id"].append(str(row["clip_id"]))
        arrays["span_group_id"].append(str(row["span_group_id"]))
        arrays["view_name"].append(str(row["view_name"]))
        arrays["source_id"].append(str(row["source_id"]))
        arrays["source_dataset"].append(str(row["source_dataset"]))
        arrays["episode_id"].append(str(row["episode_id"]))
        arrays["episode_index"].append(int(row["episode_index"]))
        arrays["subtask_index"].append(int(row["subtask_index"]))
        arrays["subtask_name"].append(str(row["subtask_name"]))
        arrays["task_index"].append(int(row["task_index"]))
        arrays["task_name"].append(str(row["task_name"]))
        arrays["frame_start"].append(int(row["frame_start"]))
        arrays["frame_end"].append(int(row["frame_end"]))
        arrays["start_ns"].append(int(row["start_ns"]))
        arrays["end_ns"].append(int(row["end_ns"]))
        arrays["native_fps"].append(float(row["native_fps"]))
        arrays["episode_from_timestamp"].append(float(row["episode_from_timestamp"]))
        arrays["clip_uri"].append(str(row["clip_uri"]) if row.get("clip_uri") is not None else None)
        arrays["action_data_uri"].append(
            str(row["action_data_uri"]) if row.get("action_data_uri") is not None else None
        )
        arrays["camera_motion_annotation"].append(
            str(row["camera_motion_annotation"]) if row.get("camera_motion_annotation") is not None else None
        )
        arrays["status"].append(str(row["status"]))
        arrays["error_stage"].append(str(row["error_stage"]) if row.get("error_stage") is not None else None)
        arrays["error_message"].append(str(row["error_message"]) if row.get("error_message") is not None else None)

    return pa.table(arrays, schema=OUTCOME_SCHEMA)
