# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Ray Data streaming path and clip-only publication."""

import hashlib
import json
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
from cosmos_curator.next.recipes.video_split.discovery import ResolvedInputSelection
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.processing import _clip_failure, _clip_work_record, _source_failure
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA

_PLANNED_CLIPS = {"long": 10, "partial": 3, "unreadable": -1, "short": 0}
_URIS = tuple(f"s3://example-bucket/raw/{name}.mp4" for name in _PLANNED_CLIPS)
_TERMINAL_ROWS = 14

# These tests drive real Ray Data plans, so they take the session cluster rather
# than start one of their own: a module that re-initialises Ray strands the
# cached session fixture for every module collected after it.
pytestmark = pytest.mark.usefixtures("ray_local")


class _NullLogger:
    def info(self, _message: str, *_args: object) -> None:
        pass


class _RecordingLogger(_NullLogger):
    def __init__(self) -> None:
        self.info_messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.info_messages.append(message.format(*args))


def _name(source_uri: str) -> str:
    return source_uri.rsplit("/", 1)[-1].removesuffix(".mp4")


def _source_fields(source_uri: str) -> dict[str, Any]:
    return {
        "source_id": make_source_id(source_uri),
        "source_uri": source_uri,
        "source_size_bytes": 1024,
        "source_duration_ns": 30_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": _UNKNOWN_FRAME_COUNT if _name(source_uri) == "long" else 900,
        "source_video_codec": "h264",
    }


def _fake_download_and_plan_source(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> dict[str, Any]:
    del config
    return {"source_uri": str(row["source_uri"])}


def _fake_transcode_source(
    row: dict[str, Any],
    *,
    config: ResolvedVideoSplitConfig,
) -> Iterator[dict[str, Any]]:
    source_uri = str(row["source_uri"])
    source_id = make_source_id(source_uri)
    planned = _PLANNED_CLIPS[_name(source_uri)]
    if planned < 0:
        yield _source_failure(source_uri, source_id, stage="source-probe", message="unreadable header")
        return

    fields = _source_fields(source_uri)
    for index in range(planned):
        clip_id = f"{_name(source_uri)}-{index}"
        work = _clip_work_record(
            fields,
            span=Span(start_ns=index * 10_000_000_000, end_ns=(index + 1) * 10_000_000_000),
            clip_id=clip_id,
            clip_uri=f"{config.output.media_root}/clips/{clip_id}.mp4",
        )
        if clip_id == "partial-1":
            yield _clip_failure(work, stage="transcode", error=RuntimeError("broken GOP"))
            continue
        yield work | {
            "record_type": "clip",
            "clip_bytes": b"clip bytes",
            "clip_size_bytes": 512,
            "clip_duration_ns": 10_000_000_000,
            "clip_width": 1920,
            "clip_height": 1080,
            "clip_frame_rate": 30.0,
            "clip_frame_count": _UNKNOWN_FRAME_COUNT if _name(source_uri) == "long" else 300,
            "clip_video_codec": "h264",
        }


def _fake_upload_clip(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> dict[str, Any]:
    del config
    if row["record_type"] == "error":
        return row
    assert row["clip_bytes"] == b"clip bytes"
    return dict(row) | {"clip_bytes": b""}


def _record_transcode_task(
    row: dict[str, Any],
    *,
    config: ResolvedVideoSplitConfig,
) -> Iterator[dict[str, Any]]:
    task_id = ray.get_runtime_context().get_task_id().encode()
    marker = int.from_bytes(hashlib.blake2b(task_id, digest_size=7).digest(), byteorder="big")
    for record in _fake_transcode_source(row, config=config):
        if record["record_type"] == "clip":
            record["clip_size_bytes"] = marker
        yield record


def _config(tmp_path: Path, *, clips_per_publish_batch: int = 4) -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": list(_URIS)},
            "output": {
                "media_root": "s3://example-bucket/output/",
                "clips_lance_uri": str(tmp_path / "lance"),
            },
            "execution": {"transcode_cpus": 1.0, "clips_per_publish_batch": clips_per_publish_batch},
        }
    )


@pytest.fixture(autouse=True)
def _fake_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "logger", _NullLogger())
    monkeypatch.setattr(
        pipeline,
        "resolve_input_selection",
        lambda *_args, **_kwargs: ResolvedInputSelection(
            canonical_uris=_URIS,
            scheduled_uris=tuple(reversed(_URIS)),
        ),
    )
    monkeypatch.setattr(pipeline, "assert_video_encoder_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "download_and_plan_source", _fake_download_and_plan_source)
    monkeypatch.setattr(pipeline, "transcode_source", _fake_transcode_source)
    monkeypatch.setattr(pipeline, "upload_clip", _fake_upload_clip)


@pytest.fixture(autouse=True)
def uploaded_reports(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Capture the driver-written S3 error report for each test."""
    reports: dict[str, str] = {}

    def capture_report(source_path: str, destination_uri: str, **_kwargs: object) -> None:
        reports[destination_uri] = Path(source_path).read_text(encoding="utf-8")

    monkeypatch.setattr(pipeline, "upload_file", capture_report)
    return reports


def test_source_processing_streams_clips_and_errors(tmp_path: Path) -> None:
    """The terminal Ray stream has successful clips and sparse failures only."""
    rows = pipeline.clip_result_dataset(_URIS, _config(tmp_path)).take_all()

    assert len(rows) == _TERMINAL_ROWS
    assert sum(row["record_type"] == "clip" for row in rows) == 12
    assert sorted(row["error_stage"] for row in rows if row["record_type"] == "error") == [
        "source-probe",
        "transcode",
    ]
    assert all(row["clip_bytes"] == b"" for row in rows)


def test_download_and_upload_each_request_an_io_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both IO stages are independent from the CPU-heavy flat-map stage."""
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeDataset:
        def map(self, _function: object, **kwargs: object) -> "_FakeDataset":
            calls.append(("map", kwargs))
            return self

        def flat_map(self, _function: object, **kwargs: object) -> "_FakeDataset":
            calls.append(("flat_map", kwargs))
            return self

    dataset = _FakeDataset()
    monkeypatch.setattr(pipeline.ray.data, "from_items", lambda *_args, **_kwargs: dataset)

    assert pipeline.clip_result_dataset(_URIS, _config(tmp_path)) is dataset
    assert [call for call, _ in calls] == ["map", "flat_map", "map"]
    assert calls[0][1]["resources"] == {"curator_io": 1.0}
    assert "resources" not in calls[1][1]
    assert calls[2][1]["resources"] == {"curator_io": 1.0}


def test_run_publishes_clips_and_a_complete_error_report(
    tmp_path: Path,
    uploaded_reports: dict[str, str],
) -> None:
    """One run commits clip metadata and replaces the sparse JSON diagnostics."""
    config = _config(tmp_path)

    summary = pipeline.run_config(config)

    assert summary == {
        "sources": 4,
        "clips_published": 12,
        "errors": 2,
        "clips_lance_uri": config.output.clips_lance_uri,
        "clips_lance_version": summary["clips_lance_version"],
        "errors_uri": config.output.errors_uri,
    }
    clips = lance.dataset(config.output.clips_lance_uri)
    assert clips.schema == CLIP_SCHEMA
    assert clips.version == summary["clips_lance_version"]
    assert sorted(row["clip_id"] for row in clips.to_table().to_pylist()) == [
        *(f"long-{index}" for index in range(10)),
        "partial-0",
        "partial-2",
    ]
    errors = json.loads(uploaded_reports[config.output.errors_uri])
    assert {(row["scope"], row["error_stage"], row["clip_id"]) for row in errors} == {
        ("source", "source-probe", None),
        ("clip", "transcode", "partial-1"),
    }


def test_error_report_is_overwritten_with_an_empty_array(
    tmp_path: Path,
    uploaded_reports: dict[str, str],
) -> None:
    """A successful run cannot leave a previous run's diagnostics looking current."""
    config = _config(tmp_path)
    record = next(_fake_transcode_source({"source_uri": _URIS[0]}, config=config)) | {"clip_bytes": b""}
    terminal = ray.data.from_items([record])

    _, _, errors = pipeline._publish_results(terminal, config)

    assert errors == 0
    assert json.loads(uploaded_reports[config.output.errors_uri]) == []


def test_all_zero_clip_sources_publish_empty_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uploaded_reports: dict[str, str],
) -> None:
    """A nonempty selection may validly produce no terminal Ray rows."""
    short_uri = next(uri for uri in _URIS if _name(uri) == "short")
    monkeypatch.setattr(
        pipeline,
        "resolve_input_selection",
        lambda *_args, **_kwargs: ResolvedInputSelection(
            canonical_uris=(short_uri,),
            scheduled_uris=(short_uri,),
        ),
    )
    config = _config(tmp_path)

    summary = pipeline.run_config(config)

    assert summary["sources"] == 1
    assert summary["clips_published"] == 0
    assert summary["errors"] == 0
    assert lance.dataset(config.output.clips_lance_uri).count_rows() == 0
    assert json.loads(uploaded_reports[config.output.errors_uri]) == []


def test_publication_batch_does_not_collapse_transcodes_into_one_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata coalescing happens downstream of source-level transcode tasks."""
    monkeypatch.setattr(pipeline, "transcode_source", _record_transcode_task)
    config = _config(tmp_path, clips_per_publish_batch=1024)

    pipeline.run_config(config)

    clips = lance.dataset(config.output.clips_lance_uri).to_table()
    assert len(set(clips["clip_size_bytes"].to_pylist())) > 1


def test_unknown_frame_counts_survive_the_run_as_nulls(
    tmp_path: Path,
) -> None:
    """Unknown source and clip frame counts remain nullable in Lance."""
    config = _config(tmp_path)

    pipeline.run_config(config)

    published = {row["clip_id"]: row for row in lance.dataset(config.output.clips_lance_uri).to_table().to_pylist()}
    assert published["long-0"]["clip_frame_count"] is None
    assert published["long-0"]["source_frame_count"] is None
    assert published["partial-0"]["clip_frame_count"] == 300


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected"),
    [
        (12.5, "12.5s"),
        (754.0, "12m 34s"),
        (11_528.0, "3h 12m 8s"),
    ],
)
def test_elapsed_time_is_human_readable(elapsed_seconds: float, expected: str) -> None:
    """Driver timing logs retain useful precision with compact units."""
    assert pipeline._format_elapsed(elapsed_seconds) == expected


def test_run_without_sources_fails_instead_of_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uploaded_reports: dict[str, str],
) -> None:
    """A mistyped empty root cannot replace the previous clip snapshot or report."""
    config = _config(tmp_path)
    pipeline.run_config(config)
    published_clips = lance.dataset(config.output.clips_lance_uri).count_rows()
    published_errors = uploaded_reports[config.output.errors_uri]
    monkeypatch.setattr(
        pipeline,
        "resolve_input_selection",
        lambda *_args, **_kwargs: ResolvedInputSelection(canonical_uris=(), scheduled_uris=()),
    )

    with pytest.raises(ValueError, match="realized 0 source videos"):
        pipeline.run_config(config)

    assert lance.dataset(config.output.clips_lance_uri).count_rows() == published_clips
    assert uploaded_reports[config.output.errors_uri] == published_errors
