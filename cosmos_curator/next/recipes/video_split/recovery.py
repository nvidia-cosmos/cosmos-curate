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

"""Driver-side reconciliation of selected sources with canonical clip rows."""

import json
import sys
from dataclasses import dataclass
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig
from cosmos_curator.next.recipes.video_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
    UNKNOWN_FRAME_COUNT,
)
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.processing import plan_clip_work
from cosmos_curator.next.recipes.video_split.records import SOURCE_MEDIA_FIELDS

_RECOVERY_COLUMNS = (
    "record_schema_version",
    "media_contract_version",
    "source_id",
    "source_uri",
    *SOURCE_MEDIA_FIELDS,
    "start_ns",
    "end_ns",
    "clip_id",
    "clip_uri",
)


@dataclass(frozen=True, slots=True)
class ReconciledSources:
    """Source work remaining after comparison with one canonical Lance version."""

    source_items: tuple[dict[str, Any], ...]
    complete_sources: int
    partial_sources: int
    unknown_sources: int
    committed_clip_rows: int


@dataclass(frozen=True, slots=True)
class _CommittedClip:
    start_ns: int
    end_ns: int
    clip_uri: str
    record_schema_version: int
    media_contract_version: int


@dataclass(slots=True)
class _CommittedSource:
    fields: dict[str, Any]
    clips: dict[str, _CommittedClip]


def reconcile_sources(
    scheduled_source_uris: tuple[str, ...],
    *,
    dataset: lance.LanceDataset,
    config: ResolvedVideoSplitConfig,
) -> ReconciledSources:
    """Return unknown and clip-partial sources in their requested scheduling order."""
    uri_by_source_id = _source_uri_index(scheduled_source_uris)
    committed_by_source: dict[str, _CommittedSource] = {}
    seen_clip_ids: set[str] = set()
    committed_clip_rows = 0

    source_ids = tuple(uri_by_source_id)
    if source_ids:
        source_filter = pc.field("source_id").isin(pa.array(source_ids, type=pa.string()))
        scanner = dataset.scanner(columns=list(_RECOVERY_COLUMNS), filter=source_filter)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                _record_committed_row(
                    row,
                    uri_by_source_id=uri_by_source_id,
                    committed_by_source=committed_by_source,
                    seen_clip_ids=seen_clip_ids,
                )
                committed_clip_rows += 1

    source_items: list[dict[str, Any]] = []
    complete_sources = 0
    partial_sources = 0
    unknown_sources = 0
    for source_uri in scheduled_source_uris:
        source_id = make_source_id(source_uri)
        committed = committed_by_source.get(source_id)
        if committed is None:
            source_items.append(_source_item(source_uri, source_id=source_id))
            unknown_sources += 1
            continue

        source_fields = _work_source_fields(committed.fields)
        expected = plan_clip_work(source_fields, config=config)
        missing_spans: list[list[int]] = []
        for clip_work in expected:
            clip_id = str(clip_work["clip_id"])
            canonical = committed.clips.get(clip_id)
            if canonical is None:
                missing_spans.append([int(clip_work["start_ns"]), int(clip_work["end_ns"])])
            else:
                _validate_expected_clip(canonical, clip_work)

        if not missing_spans:
            complete_sources += 1
            continue

        source_items.append(
            _source_item(
                source_uri,
                source_id=source_id,
                source_fields=source_fields,
                missing_spans=missing_spans,
            )
        )
        partial_sources += 1

    return ReconciledSources(
        source_items=tuple(source_items),
        complete_sources=complete_sources,
        partial_sources=partial_sources,
        unknown_sources=unknown_sources,
        committed_clip_rows=committed_clip_rows,
    )


def _source_uri_index(source_uris: tuple[str, ...]) -> dict[str, str]:
    uri_by_source_id: dict[str, str] = {}
    for source_uri in source_uris:
        source_id = make_source_id(source_uri)
        previous = uri_by_source_id.setdefault(source_id, source_uri)
        if previous != source_uri:
            msg = f"Selected source URIs have the same source_id: {previous!r} and {source_uri!r}"
            raise ValueError(msg)
    return uri_by_source_id


def _record_committed_row(
    row: dict[str, Any],
    *,
    uri_by_source_id: dict[str, str],
    committed_by_source: dict[str, _CommittedSource],
    seen_clip_ids: set[str],
) -> None:
    source_id = sys.intern(str(row["source_id"]))
    source_uri = sys.intern(str(row["source_uri"]))
    expected_source_uri = uri_by_source_id.get(source_id)
    if expected_source_uri is None:
        msg = f"Lance source_id {source_id!r} was returned outside the selected source filter"
        raise ValueError(msg)
    if source_uri != expected_source_uri or make_source_id(source_uri) != source_id:
        msg = f"Canonical source identity mismatch for source_id {source_id}"
        raise ValueError(msg)

    source_fields = {
        "source_id": source_id,
        "source_uri": source_uri,
        **{field: row[field] for field in SOURCE_MEDIA_FIELDS},
    }
    committed_source = committed_by_source.get(source_id)
    if committed_source is None:
        committed_source = _CommittedSource(fields=source_fields, clips={})
        committed_by_source[source_id] = committed_source
    elif committed_source.fields != source_fields:
        msg = f"Canonical source metadata is inconsistent for {source_uri}"
        raise ValueError(msg)

    clip_id = str(row["clip_id"])
    if clip_id in seen_clip_ids:
        msg = f"Canonical table contains duplicate clip_id {clip_id}"
        raise ValueError(msg)
    seen_clip_ids.add(clip_id)
    committed_source.clips[clip_id] = _CommittedClip(
        start_ns=int(row["start_ns"]),
        end_ns=int(row["end_ns"]),
        clip_uri=str(row["clip_uri"]),
        record_schema_version=int(row["record_schema_version"]),
        media_contract_version=int(row["media_contract_version"]),
    )


def _work_source_fields(fields: dict[str, Any]) -> dict[str, Any]:
    work_fields = dict(fields)
    if work_fields["source_frame_count"] is None:
        work_fields["source_frame_count"] = UNKNOWN_FRAME_COUNT
    return work_fields


def _source_item(
    source_uri: str,
    *,
    source_id: str,
    source_fields: dict[str, Any] | None = None,
    missing_spans: list[list[int]] | None = None,
) -> dict[str, Any]:
    known = source_fields is not None
    return {
        "source_uri": source_uri,
        "source_id": source_id,
        "source_known": known,
        "source_size_bytes": 0 if source_fields is None else int(source_fields["source_size_bytes"]),
        "source_duration_ns": 0 if source_fields is None else int(source_fields["source_duration_ns"]),
        "source_width": 0 if source_fields is None else int(source_fields["source_width"]),
        "source_height": 0 if source_fields is None else int(source_fields["source_height"]),
        "source_frame_rate": 0.0 if source_fields is None else float(source_fields["source_frame_rate"]),
        "source_frame_count": (
            UNKNOWN_FRAME_COUNT if source_fields is None else int(source_fields["source_frame_count"])
        ),
        "source_video_codec": "" if source_fields is None else str(source_fields["source_video_codec"]),
        "missing_spans_json": json.dumps(missing_spans or [], separators=(",", ":")),
    }


def _validate_expected_clip(canonical: _CommittedClip, expected: dict[str, Any]) -> None:
    clip_id = str(expected["clip_id"])
    if canonical.record_schema_version != CLIP_RECORD_SCHEMA_VERSION:
        msg = (
            f"Canonical clip {clip_id} has record_schema_version={canonical.record_schema_version}; "
            f"expected {CLIP_RECORD_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    if canonical.media_contract_version != MEDIA_CONTRACT_VERSION:
        msg = (
            f"Canonical clip {clip_id} has media_contract_version={canonical.media_contract_version}; "
            f"expected {MEDIA_CONTRACT_VERSION}"
        )
        raise ValueError(msg)
    if (
        canonical.start_ns != int(expected["start_ns"])
        or canonical.end_ns != int(expected["end_ns"])
        or canonical.clip_uri != str(expected["clip_uri"])
    ):
        msg = f"Canonical geometry or media URI does not match deterministic clip_id {clip_id}"
        raise ValueError(msg)
