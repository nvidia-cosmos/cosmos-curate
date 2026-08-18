# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ray Data composition and run-level publication.

These run a real local Ray cluster and publish real Lance datasets. Only the
media boundary is replaced: source processing stands in for S3 and FFmpeg so
the composition, fan-out and publication paths are the code under test.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import pytest
import ray

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split import pipeline
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config_data
from cosmos_curator.next.recipes.video_split.contracts import UNKNOWN_FRAME_COUNT as _UNKNOWN_FRAME_COUNT
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.processing import _clip_work_record, _source_failure, _source_outcome
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA

# Clip counts per source name: a long video, one that loses a clip, one that
# fails to probe, and a valid video too short to retain any span.
_PLANNED_CLIPS = {"long": 10, "partial": 3, "unreadable": -1, "short": 0}
_URIS = tuple(f"s3://example-bucket/raw/{name}.mp4" for name in _PLANNED_CLIPS)
_FAILED_CLIPS = 1
_TERMINAL_ROWS = len(_PLANNED_CLIPS) + sum(max(count, 0) for count in _PLANNED_CLIPS.values()) - _FAILED_CLIPS


def _name(source_uri: str) -> str:
    return source_uri.rsplit("/", 1)[-1].removesuffix(".mp4")


def _source_fields(source_uri: str) -> dict[str, Any]:
    return {
        "source_id": make_source_id(source_uri),
        "source_uri": source_uri,
        "source_media_known": True,
        "source_size_bytes": 1024,
        "source_duration_ns": 30_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        # The long source stands in for a container with no frame count.
        "source_frame_count": _UNKNOWN_FRAME_COUNT if _name(source_uri) == "long" else 900,
        "source_video_codec": "h264",
    }


def _fake_clip_work(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Stand in for probing plus fixed-stride span generation."""
    source_uri = str(row["source_uri"])
    fields = _source_fields(source_uri)
    planned = _PLANNED_CLIPS[_name(source_uri)]
    records: list[dict[str, Any]] = []
    for index in range(planned):
        clip_id = f"{_name(source_uri)}-{index}"
        records.append(
            _clip_work_record(
                fields,
                span=Span(start_ns=index * 10_000_000_000, end_ns=(index + 1) * 10_000_000_000),
                clip_id=clip_id,
                clip_uri=f"{config.output.media_root}/clips/{clip_id}.mp4",
            )
        )
    return records


def _fake_process_work(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> dict[str, Any]:
    """Stand in for transcoding and the clip media write."""
    del config
    outcome = dict(row)
    # The middle clip of the partial source fails after its siblings succeed.
    if outcome["clip_id"] == "partial-1":
        return outcome | {
            "record_type": "clip_outcome",
            "status": "failed",
            "error_stage": "transcode",
            "error_message": "broken GOP",
        }
    return outcome | {
        "record_type": "clip_outcome",
        "status": "success",
        "clip_size_bytes": 512,
        "clip_duration_ns": 10_000_000_000,
        "clip_width": 1920,
        "clip_height": 1080,
        "clip_frame_rate": 30.0,
        "clip_frame_count": _UNKNOWN_FRAME_COUNT if _name(str(row["source_uri"])) == "long" else 300,
        "clip_video_codec": "h264",
        "error_stage": "",
        "error_message": "",
    }


def _fake_process_source(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Stand in for the complete source-level worker function."""
    source_uri = str(row["source_uri"])
    source_id = make_source_id(source_uri)
    if _PLANNED_CLIPS[_name(source_uri)] < 0:
        return [_source_failure(source_uri, source_id, stage="source-probe", error=RuntimeError("unreadable header"))]
    outcomes = [_fake_process_work(record, config=config) for record in _fake_clip_work(row, config=config)]
    return [
        _source_outcome(_source_fields(source_uri), outcomes),
        *(outcome for outcome in outcomes if outcome["status"] == "success"),
    ]


def _record_process_task(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Encode the source-processing Ray task ID in successful rows."""
    result = _fake_process_source(row, config=config)
    task_id = ray.get_runtime_context().get_task_id().encode()
    task_marker = int.from_bytes(hashlib.blake2b(task_id, digest_size=7).digest(), byteorder="big")
    for record in result:
        if record["record_type"] == "clip_outcome" and record["status"] == "success":
            record["clip_size_bytes"] = task_marker
    return result


def _config(tmp_path: Path, *, clips_per_publish_batch: int = 4) -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": list(_URIS)},
            "output": {
                "media_root": "s3://example-bucket/output/",
                "clips_lance_uri": str(tmp_path / "clips.lance"),
                "sources_lance_uri": str(tmp_path / "sources.lance"),
            },
            "execution": {"clips_per_publish_batch": clips_per_publish_batch},
        }
    )


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster() -> Iterator[None]:
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def _fake_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "resolve_input_selection", lambda *_args, **_kwargs: _URIS)
    monkeypatch.setattr(pipeline, "assert_video_encoder_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "process_source", _fake_process_source)


def test_source_processing_fans_out_to_terminal_rows(tmp_path: Path) -> None:
    """Each realized source contributes one outcome plus its successful clips."""
    rows = pipeline.source_result_dataset(_URIS, _config(tmp_path)).take_all()

    assert len(rows) == _TERMINAL_ROWS
    outcomes = [row for row in rows if row["record_type"] == "source_outcome"]
    assert sorted(row["source_uri"] for row in outcomes) == sorted(_URIS)


def test_source_outcomes_are_validated_and_restored_to_selection_order() -> None:
    """Publication receives exactly one source-owned outcome per selected URI."""
    first, second = _URIS[:2]
    outcomes = [_source_outcome(_source_fields(second), []), _source_outcome(_source_fields(first), [])]

    ordered = pipeline._order_source_outcomes(outcomes, (first, second))

    assert [row["source_uri"] for row in ordered] == [first, second]


def test_duplicate_source_outcome_fails_the_run() -> None:
    """Worker replay may not silently put duplicate source rows in a snapshot."""
    source_uri = _URIS[0]
    outcome = _source_outcome(_source_fields(source_uri), [])

    with pytest.raises(RuntimeError, match="multiple source outcomes"):
        pipeline._order_source_outcomes([outcome, outcome], (source_uri,))


def test_missing_or_unselected_source_outcome_fails_the_run() -> None:
    """The direct outcomes must cover exactly the realized source selection."""
    first, second = _URIS[:2]
    first_outcome = _source_outcome(_source_fields(first), [])
    second_outcome = _source_outcome(_source_fields(second), [])

    with pytest.raises(RuntimeError, match="did not produce source outcomes"):
        pipeline._order_source_outcomes([first_outcome], (first, second))
    with pytest.raises(RuntimeError, match="unselected sources"):
        pipeline._order_source_outcomes([first_outcome, second_outcome], (second,))


def test_inconsistent_source_identity_fails_the_run() -> None:
    """A source outcome cannot claim an identity derived from another URI."""
    source_uri = _URIS[0]
    outcome = _source_outcome(_source_fields(source_uri), []) | {"source_id": "wrong"}

    with pytest.raises(RuntimeError, match="inconsistent source_id"):
        pipeline._order_source_outcomes([outcome], (source_uri,))


def test_publication_batch_does_not_collapse_transcodes_into_one_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication bundling happens after, rather than around, transcode tasks."""
    monkeypatch.setattr(pipeline, "process_source", _record_process_task)
    config = _config(tmp_path, clips_per_publish_batch=1024)

    pipeline.run_config(config)

    clips = lance.dataset(config.output.clips_lance_uri).to_table()
    process_task_markers = set(clips["clip_size_bytes"].to_pylist())
    assert len(process_task_markers) > 1


def test_run_publishes_clips_and_sources_from_worker_written_fragments(tmp_path: Path) -> None:
    """A full run publishes both snapshots without clip rows passing through the driver."""
    config = _config(tmp_path)

    summary = pipeline.run_config(config)

    counts = {
        "sources": 4,
        "sources_succeeded": 2,
        "sources_failed": 2,
        "clips_planned": 13,
        "clips_published": 12,
        "clips_failed": 1,
    }
    assert {key: summary[key] for key in counts} == counts

    clips = lance.dataset(config.output.clips_lance_uri)
    assert clips.schema == CLIP_SCHEMA
    assert clips.version == summary["clips_lance_version"]
    # Batches wrote fragments independently and committed together exactly once.
    assert len(clips.get_fragments()) > 1
    published = clips.to_table().to_pylist()
    assert sorted(row["clip_id"] for row in published) == [
        *(f"long-{index}" for index in range(10)),
        "partial-0",
        "partial-2",
    ]
    assert clips.read_transaction(clips.version).transaction_properties["snapshot"] == "clips"


def test_run_records_one_bound_outcome_for_every_realized_source(tmp_path: Path) -> None:
    """Source rows distinguish full success, partial output, probe failure and zero clips."""
    config = _config(tmp_path)

    summary = pipeline.run_config(config)

    published = lance.dataset(config.output.sources_lance_uri).to_table().to_pylist()
    assert [row["source_uri"] for row in published] == list(_URIS)
    assert all(row["clips_lance_version"] == summary["clips_lance_version"] for row in published)

    outcomes = {
        _name(row["source_uri"]): (
            row["status"],
            row["planned_clip_count"],
            row["published_clip_count"],
            row["failed_clip_count"],
            row["error_stage"],
        )
        for row in published
    }
    assert outcomes == {
        "long": ("success", 10, 10, 0, None),
        "partial": ("failed", 3, 2, 1, "transcode"),
        "unreadable": ("failed", 0, 0, 0, "source-probe"),
        "short": ("success", 0, 0, 0, None),
    }
    assert next(row["error_message"] for row in published if _name(row["source_uri"]) == "partial") == "broken GOP"

    # A source that produced no clip rows anywhere still records what it was,
    # and one that never probed records that it does not know.
    media = {_name(row["source_uri"]): (row["source_duration_ns"], row["source_width"]) for row in published}
    assert media["short"] == (30_000_000_000, 1920)
    assert media["unreadable"] == (None, None)


def test_unknown_frame_counts_survive_the_run_as_nulls(tmp_path: Path) -> None:
    """The work-record sentinel is published as null rather than a negative count."""
    config = _config(tmp_path)

    pipeline.run_config(config)

    published = {row["clip_id"]: row for row in lance.dataset(config.output.clips_lance_uri).to_table().to_pylist()}
    assert published["long-0"]["clip_frame_count"] is None
    assert published["long-0"]["source_frame_count"] is None
    assert published["partial-0"]["clip_frame_count"] == 300


def test_run_without_sources_fails_instead_of_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty selection is a mistyped root far more often than a real intent."""
    config = _config(tmp_path)
    pipeline.run_config(config)
    published_clips = lance.dataset(config.output.clips_lance_uri).count_rows()
    monkeypatch.setattr(pipeline, "resolve_input_selection", lambda *_args, **_kwargs: ())

    with pytest.raises(ValueError, match="realized 0 source videos"):
        pipeline.run_config(config)

    # The previous run's snapshots survive rather than being overwritten empty.
    assert lance.dataset(config.output.clips_lance_uri).count_rows() == published_clips
