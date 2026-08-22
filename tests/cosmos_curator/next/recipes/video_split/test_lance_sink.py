# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distributed clip fragment writes and snapshot commits."""

from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_split import lance_sink
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA, clip_table

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"
_UNKNOWN_FRAME_COUNT = -1


def _record(clip_id: str, *, record_type: str = "clip", frame_count: int = 300) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "record_schema_version": 1,
        "media_contract_version": 1,
        "source_id": make_source_id(_SOURCE_URI),
        "source_uri": _SOURCE_URI,
        "source_size_bytes": 1024,
        "source_duration_ns": 30_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": frame_count,
        "source_video_codec": "h264",
        "start_ns": 0,
        "end_ns": 10_000_000_000,
        "clip_id": clip_id,
        "clip_uri": f"s3://example-bucket/output/clips/{clip_id}.mp4",
        "clip_size_bytes": 512,
        "clip_duration_ns": 10_000_000_000,
        "clip_width": 1920,
        "clip_height": 1080,
        "clip_frame_rate": 30.0,
        "clip_frame_count": frame_count,
        "clip_video_codec": "h264",
    }


def _write(records: list[dict[str, Any]], *, uri: str) -> list[str]:
    work = pa.Table.from_pylist(records)
    return lance_sink.write_clip_fragments(clip_table(work), uri=uri, storage_profile="default")


def test_fragments_from_independent_workers_commit_as_one_snapshot(tmp_path: Path) -> None:
    """Workers assign colliding fragment IDs, so the commit must reassign them."""
    uri = str(tmp_path / "clips.lance")
    first = _write([_record("a"), _record("b")], uri=uri)
    second = _write([_record("c")], uri=uri)

    version = lance_sink.commit_clip_snapshot(first + second, uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert len(first + second) == len(dataset.get_fragments()) == 2
    assert sorted(row["clip_id"] for row in dataset.to_table().to_pylist()) == ["a", "b", "c"]


def test_one_batch_writes_exactly_one_fragment(tmp_path: Path) -> None:
    """A publication batch is not subdivided below Lance's native limits."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write([_record(f"c{i}") for i in range(10)], uri=uri)

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert len(fragments) == len(dataset.get_fragments()) == 1
    assert dataset.count_rows() == 10


def test_error_records_are_not_clip_rows(tmp_path: Path) -> None:
    """Operational errors are excluded before the canonical fragment write."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write([_record("published"), _record("failed", record_type="error")], uri=uri)

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    assert [row["clip_id"] for row in lance.dataset(uri).to_table().to_pylist()] == ["published"]


def test_unknown_frame_counts_are_published_as_null(tmp_path: Path) -> None:
    """The work-record sentinel becomes a real null in the published schema."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write([_record("a", frame_count=_UNKNOWN_FRAME_COUNT)], uri=uri)

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    published = lance.dataset(uri).to_table().to_pylist()[0]
    assert published["clip_frame_count"] is None
    assert published["source_frame_count"] is None


def test_empty_run_commits_an_empty_clip_snapshot(tmp_path: Path) -> None:
    """Zero published clips still produce a complete canonical snapshot."""
    uri = str(tmp_path / "clips.lance")

    version = lance_sink.commit_clip_snapshot([], uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert dataset.count_rows() == 0
    assert dataset.schema == CLIP_SCHEMA


def test_rerun_overwrites_the_complete_snapshot(tmp_path: Path) -> None:
    """Each run owns a complete snapshot rather than appending to the last one."""
    uri = str(tmp_path / "clips.lance")
    lance_sink.commit_clip_snapshot(_write([_record("stale")], uri=uri), uri=uri, storage_profile="default")

    version = lance_sink.commit_clip_snapshot(
        _write([_record("fresh")], uri=uri),
        uri=uri,
        storage_profile="default",
    )

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert [row["clip_id"] for row in dataset.to_table().to_pylist()] == ["fresh"]


def test_fragment_writer_requires_the_canonical_schema(tmp_path: Path) -> None:
    """Internal work tables cannot accidentally become a public contract."""
    with pytest.raises(ValueError, match="canonical clip schema"):
        lance_sink.write_clip_fragments(
            pa.table({"clip_id": ["a"]}),
            uri=str(tmp_path / "clips.lance"),
            storage_profile="default",
        )
