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

"""Arrow contracts for published clips and operational errors."""

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.recipes.video_split.contracts import UNKNOWN_FRAME_COUNT

SOURCE_MEDIA_FIELDS = (
    "source_size_bytes",
    "source_duration_ns",
    "source_width",
    "source_height",
    "source_frame_rate",
    "source_frame_count",
    "source_video_codec",
)

CLIP_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.large_string(), nullable=False),
        pa.field("source_size_bytes", pa.int64(), nullable=False),
        pa.field("source_duration_ns", pa.int64(), nullable=False),
        pa.field("source_width", pa.int32(), nullable=False),
        pa.field("source_height", pa.int32(), nullable=False),
        pa.field("source_frame_rate", pa.float64(), nullable=False),
        pa.field("source_frame_count", pa.int64()),
        pa.field("source_video_codec", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("clip_size_bytes", pa.int64(), nullable=False),
        pa.field("clip_duration_ns", pa.int64(), nullable=False),
        pa.field("clip_width", pa.int32(), nullable=False),
        pa.field("clip_height", pa.int32(), nullable=False),
        pa.field("clip_frame_rate", pa.float64(), nullable=False),
        pa.field("clip_frame_count", pa.int64()),
        pa.field("clip_video_codec", pa.string(), nullable=False),
    ]
)

ERROR_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("scope", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.large_string(), nullable=False),
        pa.field("clip_id", pa.string()),
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("error_stage", pa.string(), nullable=False),
        pa.field("error_message", pa.large_string(), nullable=False),
    ]
)

_NULLABLE_FRAME_COUNTS = ("source_frame_count", "clip_frame_count")
_TERMINAL_RECORD_TYPES = frozenset({"clip", "error"})


def validate_terminal_record_types(work_records: pa.Table) -> None:
    """Reject internal work rows instead of silently omitting them."""
    record_types = {str(value) for value in work_records["record_type"].to_pylist()}
    unexpected = sorted(record_types - _TERMINAL_RECORD_TYPES)
    if unexpected:
        msg = f"Publication received unexpected record type(s): {', '.join(unexpected)}"
        raise ValueError(msg)


def clip_table(work_records: pa.Table) -> pa.Table:
    """Project uploaded clip records onto the canonical clip schema."""
    clips = work_records.filter(pc.equal(work_records["record_type"], "clip"))
    table = clips.select([field.name for field in CLIP_SCHEMA])
    for name in _NULLABLE_FRAME_COUNTS:
        index = table.schema.get_field_index(name)
        counts = table.column(index)
        table = table.set_column(
            index,
            name,
            pc.if_else(pc.equal(counts, UNKNOWN_FRAME_COUNT), pa.scalar(None, counts.type), counts),
        )
    return table.cast(CLIP_SCHEMA)


def error_table(work_records: pa.Table) -> pa.Table:
    """Project sparse source/clip failures onto the JSON report schema."""
    failures = work_records.filter(pc.equal(work_records["record_type"], "error"))
    rows: list[dict[str, Any]] = []
    for failure in failures.to_pylist():
        clip_id = str(failure["clip_id"])
        rows.append(
            {
                "record_schema_version": failure["record_schema_version"],
                "media_contract_version": failure["media_contract_version"],
                "scope": "clip" if clip_id else "source",
                "source_id": failure["source_id"],
                "source_uri": failure["source_uri"],
                "clip_id": clip_id or None,
                "start_ns": int(failure["start_ns"]) if clip_id else None,
                "end_ns": int(failure["end_ns"]) if clip_id else None,
                "error_stage": failure["error_stage"],
                "error_message": failure["error_message"],
            }
        )
    return pa.Table.from_pylist(rows, schema=ERROR_SCHEMA)
