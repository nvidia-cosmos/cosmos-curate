# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distributed clip fragment writes and ordered snapshot commits."""

from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_split import lance_sink
from cosmos_curator.next.recipes.video_split.config import resolve_config_data
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"
_UNKNOWN_FRAME_COUNT = -1


def _record(
    clip_id: str,
    *,
    status: str = "success",
    record_type: str = "clip_outcome",
    frame_count: int = 300,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "status": status,
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
    """Run the worker-side fragment write over one batch."""
    return lance_sink.write_clip_fragments(pa.Table.from_pylist(records), uri=uri, storage_profile="default")


def _source_row(*, planned: int, published: int) -> dict[str, Any]:
    return {
        "record_schema_version": 1,
        "source_id": make_source_id(_SOURCE_URI),
        "source_uri": _SOURCE_URI,
        "status": "success",
        "source_size_bytes": 1024,
        "source_duration_ns": 30_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": 900,
        "source_video_codec": "h264",
        "planned_clip_count": planned,
        "published_clip_count": published,
        "failed_clip_count": planned - published,
        "error_stage": None,
        "error_message": None,
    }


def test_fragments_from_independent_workers_commit_as_one_snapshot(tmp_path: Path) -> None:
    """Workers assign colliding fragment ids, so the commit must reassign them."""
    uri = str(tmp_path / "clips.lance")
    first = _write([_record("a"), _record("b")], uri=uri)
    second = _write([_record("c")], uri=uri)

    version = lance_sink.commit_clip_snapshot(first + second, uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert len(first + second) == len(dataset.get_fragments()) == 2
    assert sorted(row["clip_id"] for row in dataset.to_table().to_pylist()) == ["a", "b", "c"]


def test_one_batch_writes_exactly_one_fragment(tmp_path: Path) -> None:
    """Fragment size is the publish batch size, so a batch is never subdivided."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write([_record(f"c{i}") for i in range(10)], uri=uri)

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert len(fragments) == len(dataset.get_fragments()) == 1
    assert dataset.count_rows() == 10


def test_only_successful_clips_reach_the_snapshot(tmp_path: Path) -> None:
    """Non-success clips and source outcomes are filtered out on the worker."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write(
        [
            _record("published"),
            _record("failed", status="failed"),
            _record("source", record_type="source_outcome"),
        ],
        uri=uri,
    )

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    assert [row["clip_id"] for row in lance.dataset(uri).to_table().to_pylist()] == ["published"]


def test_batch_without_published_clips_writes_no_fragments(tmp_path: Path) -> None:
    """A batch of failures contributes nothing rather than an empty fragment."""
    uri = str(tmp_path / "clips.lance")

    assert _write([_record("failed", status="failed")], uri=uri) == []


def test_unknown_frame_counts_are_published_as_null(tmp_path: Path) -> None:
    """The work-record sentinel becomes a real null in the published schema."""
    uri = str(tmp_path / "clips.lance")
    fragments = _write(
        [_record("a", frame_count=_UNKNOWN_FRAME_COUNT)],
        uri=uri,
    )

    lance_sink.commit_clip_snapshot(fragments, uri=uri, storage_profile="default")

    published = lance.dataset(uri).to_table().to_pylist()[0]
    assert published["clip_frame_count"] is None
    assert published["source_frame_count"] is None


def test_empty_run_commits_an_empty_clip_snapshot(tmp_path: Path) -> None:
    """Zero published clips still publish a complete snapshot with the canonical schema."""
    uri = str(tmp_path / "clips.lance")

    version = lance_sink.commit_clip_snapshot([], uri=uri, storage_profile="default")

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert dataset.count_rows() == 0
    assert dataset.schema == CLIP_SCHEMA


def test_source_snapshot_binds_the_committed_clip_version(tmp_path: Path) -> None:
    """Source rows and the source transaction both record the exact clip version."""
    output = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": [_SOURCE_URI]},
            "output": {
                "media_root": "s3://example-bucket/output",
                "clips_lance_uri": str(tmp_path / "clips.lance"),
                "sources_lance_uri": str(tmp_path / "sources.lance"),
            },
        }
    ).output
    fragments = _write([_record("a")], uri=output.clips_lance_uri)

    snapshots = lance_sink.publish_snapshots(
        fragments,
        [_source_row(planned=1, published=1)],
        output=output,
        storage_profile="default",
    )

    sources = lance.dataset(output.sources_lance_uri)
    assert sources.version == snapshots.sources_version
    published = sources.to_table().to_pylist()[0]
    assert published["clips_lance_uri"] == output.clips_lance_uri
    assert published["clips_lance_version"] == snapshots.clips_version
    transaction = sources.read_transaction(snapshots.sources_version)
    assert transaction is not None
    assert transaction.transaction_properties["clips_lance_version"] == str(snapshots.clips_version)


def test_rerun_overwrites_both_snapshots(tmp_path: Path) -> None:
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


def test_source_publication_failure_leaves_completed_clip_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The required non-atomic failure mode propagates without rollback."""
    output = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": [_SOURCE_URI]},
            "output": {
                "media_root": "s3://example-bucket/output",
                "clips_lance_uri": str(tmp_path / "clips.lance"),
                "sources_lance_uri": str(tmp_path / "sources.lance"),
            },
        }
    ).output
    fragments = _write([_record("a")], uri=output.clips_lance_uri)

    def fail_sources(*_args: object, **_kwargs: object) -> int:
        msg = "source commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance_sink, "write_source_snapshot", fail_sources)

    with pytest.raises(RuntimeError, match="source commit failed"):
        lance_sink.publish_snapshots(fragments, [], output=output, storage_profile="default")

    assert [row["clip_id"] for row in lance.dataset(output.clips_lance_uri).to_table().to_pylist()] == ["a"]
