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

"""Arrow row contracts for clip and source snapshots."""

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.recipes.video_split.contracts import UNKNOWN_FRAME_COUNT

WORK_RECORD_SCHEMA = pa.schema(
    [
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.large_string(), nullable=False),
        pa.field("source_media_known", pa.bool_(), nullable=False),
        pa.field("source_size_bytes", pa.int64(), nullable=False),
        pa.field("source_duration_ns", pa.int64(), nullable=False),
        pa.field("source_width", pa.int32(), nullable=False),
        pa.field("source_height", pa.int32(), nullable=False),
        pa.field("source_frame_rate", pa.float64(), nullable=False),
        pa.field("source_frame_count", pa.int64(), nullable=False),
        pa.field("source_video_codec", pa.string(), nullable=False),
        pa.field("planned_clip_count", pa.int64(), nullable=False),
        pa.field("published_clip_count", pa.int64(), nullable=False),
        pa.field("failed_clip_count", pa.int64(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("clip_size_bytes", pa.int64(), nullable=False),
        pa.field("clip_duration_ns", pa.int64(), nullable=False),
        pa.field("clip_width", pa.int32(), nullable=False),
        pa.field("clip_height", pa.int32(), nullable=False),
        pa.field("clip_frame_rate", pa.float64(), nullable=False),
        pa.field("clip_frame_count", pa.int64(), nullable=False),
        pa.field("clip_video_codec", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_stage", pa.string(), nullable=False),
        pa.field("error_message", pa.large_string(), nullable=False),
    ]
)

CLIP_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.large_string(), nullable=False),
        # A clip row only exists because its source probed cleanly, so source
        # media is known here. It is nullable on the source row, where a failed
        # probe is one of the outcomes being recorded.
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

SOURCE_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.large_string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        # Null only when the source never probed. Carrying these here is what
        # gives a failed or zero-clip source a record of what it was; on clip
        # rows the same values are denormalized for clip-only consumers.
        pa.field("source_size_bytes", pa.int64()),
        pa.field("source_duration_ns", pa.int64()),
        pa.field("source_width", pa.int32()),
        pa.field("source_height", pa.int32()),
        pa.field("source_frame_rate", pa.float64()),
        pa.field("source_frame_count", pa.int64()),
        pa.field("source_video_codec", pa.string()),
        pa.field("planned_clip_count", pa.int64(), nullable=False),
        pa.field("published_clip_count", pa.int64(), nullable=False),
        pa.field("failed_clip_count", pa.int64(), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.large_string()),
        pa.field("clips_lance_uri", pa.large_string(), nullable=False),
        pa.field("clips_lance_version", pa.int64(), nullable=False),
    ]
)

# The worker emits complete source outcomes before the clip snapshot version is
# known. The driver binds these two publication fields after committing clips.
_SOURCE_BINDING_FIELDS = frozenset({"clips_lance_uri", "clips_lance_version"})
SOURCE_OUTCOME_SCHEMA = pa.schema([field for field in SOURCE_SCHEMA if field.name not in _SOURCE_BINDING_FIELDS])

# Media properties carried from source processing into both published schemas.
# Named once so the work record and published projections cannot drift apart.
SOURCE_MEDIA_FIELDS = (
    "source_size_bytes",
    "source_duration_ns",
    "source_width",
    "source_height",
    "source_frame_rate",
    "source_frame_count",
    "source_video_codec",
)


_NULLABLE_FRAME_COUNTS = ("source_frame_count", "clip_frame_count")


def clip_table(work_records: pa.Table) -> pa.Table:
    """Project published clip work records onto the canonical clip schema."""
    table = work_records.select([field.name for field in CLIP_SCHEMA])
    for name in _NULLABLE_FRAME_COUNTS:
        index = table.schema.get_field_index(name)
        counts = table.column(index)
        table = table.set_column(
            index,
            name,
            pc.if_else(pc.equal(counts, UNKNOWN_FRAME_COUNT), pa.scalar(None, counts.type), counts),
        )
    return table.cast(CLIP_SCHEMA)


def source_outcome_table(work_records: pa.Table) -> pa.Table:
    """Project worker-owned source outcomes onto the unbound source schema."""
    outcomes = work_records.filter(pc.equal(work_records["record_type"], "source_outcome"))
    rows: list[dict[str, Any]] = []
    for outcome in outcomes.to_pylist():
        row = {field.name: outcome[field.name] for field in SOURCE_OUTCOME_SCHEMA}
        if not outcome["source_media_known"]:
            row.update(dict.fromkeys(SOURCE_MEDIA_FIELDS))
        elif row["source_frame_count"] == UNKNOWN_FRAME_COUNT:
            row["source_frame_count"] = None
        row["error_stage"] = row["error_stage"] or None
        row["error_message"] = row["error_message"] or None
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=SOURCE_OUTCOME_SCHEMA)


def work_record_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a table with the stable schema shared by Ray work and recovery results."""
    return pa.Table.from_pylist(rows, schema=WORK_RECORD_SCHEMA)


def source_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build the canonical source snapshot from bound driver-side outcomes."""
    projected = [{field.name: row.get(field.name) for field in SOURCE_SCHEMA} for row in rows]
    return pa.Table.from_pylist(projected, schema=SOURCE_SCHEMA)
