# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for clip-granular source reconciliation across invocations."""

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_split import lance_sink
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config_data
from cosmos_curator.next.recipes.video_split.contracts import UNKNOWN_FRAME_COUNT
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.processing import plan_clip_work
from cosmos_curator.next.recipes.video_split.records import clip_table
from cosmos_curator.next.recipes.video_split.recovery import reconcile_sources

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"


def _config(tmp_path: Path, *source_uris: str) -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": list(source_uris or (_SOURCE_URI,))},
            "split": {"duration_s": 10.0, "stride_s": 10.0, "min_duration_s": 2.0},
            "output": {
                "media_root": "s3://example-bucket/output",
                "clips_lance_uri": str(tmp_path / "lance"),
            },
        }
    )


def _source_fields(
    source_uri: str = _SOURCE_URI,
    *,
    duration_ns: int = 25_000_000_000,
    frame_count: int = 750,
) -> dict[str, Any]:
    return {
        "source_id": make_source_id(source_uri),
        "source_uri": source_uri,
        "source_size_bytes": 1024,
        "source_duration_ns": duration_ns,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": frame_count,
        "source_video_codec": "h264",
    }


def _planned(config: ResolvedVideoSplitConfig, fields: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return plan_clip_work(_source_fields() if fields is None else fields, config=config)


def _publish(uri: str, rows: list[dict[str, Any]]) -> None:
    clips = clip_table(pa.Table.from_pylist(rows))
    candidate = lance_sink.write_clip_fragment(clips, uri=uri, storage_profile="default")
    assert candidate is not None
    lance_sink.append_clip_fragment(candidate, uri=uri, storage_profile="default", attempts=1)


def test_absent_sources_are_unknown_and_keep_scheduling_order(tmp_path: Path) -> None:
    """Sources without rows retain discovery priority and require full probing."""
    source_uris = (
        "s3://example-bucket/raw/large.mp4",
        "s3://example-bucket/raw/small.mp4",
    )
    config = _config(tmp_path, *source_uris)
    dataset = lance_sink.open_or_create_clip_table(uri=config.output.clips_lance_uri, storage_profile="default")

    result = reconcile_sources(tuple(reversed(source_uris)), dataset=dataset, config=config)

    assert [item["source_uri"] for item in result.source_items] == list(reversed(source_uris))
    assert all(item["source_known"] is False for item in result.source_items)
    assert all(item["missing_spans_json"] == "[]" for item in result.source_items)
    assert result.complete_sources == result.partial_sources == result.committed_clip_rows == 0
    assert result.unknown_sources == 2


def test_partial_source_recreates_only_a_clip_missing_across_fragment_boundary(tmp_path: Path) -> None:
    """Committed sibling clips survive while one uncommitted span is replayed."""
    config = _config(tmp_path)
    uri = config.output.clips_lance_uri
    lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")
    planned = _planned(config)
    _publish(uri, [planned[0]])
    _publish(uri, [planned[2]])

    result = reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)

    assert result.complete_sources == result.unknown_sources == 0
    assert result.partial_sources == 1
    assert result.committed_clip_rows == 2
    assert len(result.source_items) == 1
    item = result.source_items[0]
    assert item["source_known"] is True
    assert json.loads(item["missing_spans_json"]) == [[10_000_000_000, 20_000_000_000]]

    _publish(uri, [planned[1]])
    resumed = reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)

    assert resumed.source_items == ()
    assert resumed.complete_sources == 1
    assert resumed.committed_clip_rows == 3
    assert len(lance.dataset(uri).get_fragments()) == 3


def test_nullable_source_frame_count_becomes_worker_sentinel(tmp_path: Path) -> None:
    """Canonical nulls use the schema-stable sentinel during Ray execution."""
    config = _config(tmp_path)
    uri = config.output.clips_lance_uri
    lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")
    fields = _source_fields(frame_count=UNKNOWN_FRAME_COUNT)
    planned = _planned(config, fields)
    _publish(uri, [planned[0]])

    result = reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)

    assert result.source_items[0]["source_frame_count"] == UNKNOWN_FRAME_COUNT


def test_duplicate_canonical_clip_ids_fail_reconciliation(tmp_path: Path) -> None:
    """Membership checks cannot hide a preexisting uniqueness violation."""
    config = _config(tmp_path)
    uri = config.output.clips_lance_uri
    lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")
    row = _planned(config)[0]
    table = clip_table(pa.Table.from_pylist([row, row]))
    lance.write_dataset(table, uri, mode="append")

    with pytest.raises(ValueError, match="duplicate clip_id"):
        reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)


def test_inconsistent_canonical_source_metadata_fails_reconciliation(tmp_path: Path) -> None:
    """A known source needs one unambiguous durable plan geometry."""
    config = _config(tmp_path)
    uri = config.output.clips_lance_uri
    lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")
    planned = _planned(config)
    inconsistent = dict(planned[1]) | {"source_width": 1280}
    _publish(uri, [planned[0], inconsistent])

    with pytest.raises(ValueError, match="source metadata is inconsistent"):
        reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)


def test_expected_clip_with_different_media_uri_fails_reconciliation(tmp_path: Path) -> None:
    """An expected identity cannot silently point at a different media root."""
    config = _config(tmp_path)
    uri = config.output.clips_lance_uri
    lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")
    row = _planned(config)[0] | {"clip_uri": "s3://different-bucket/clips/wrong.mp4"}
    _publish(uri, [row])

    with pytest.raises(ValueError, match="geometry or media URI"):
        reconcile_sources((_SOURCE_URI,), dataset=lance.dataset(uri), config=config)
