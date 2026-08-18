# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for source-granular processing and worker outcomes."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.next.media.ffmpeg import ProbeError, TranscodeError, VideoMetadata
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split import processing
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config_data

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"


@pytest.fixture(autouse=True)
def _worker_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep remote recovery and the real FFmpeg binary outside unit tests."""
    monkeypatch.setattr(processing, "restore_source_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processing, "write_source_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processing, "assert_video_encoder_available", lambda *_args, **_kwargs: None)


def _config() -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": [_SOURCE_URI]},
            "split": {"duration_s": 10.0, "stride_s": 10.0, "min_duration_s": 2.0},
            "output": {"media_root": "s3://example-bucket/output"},
            "execution": {
                "storage_attempts": 1,
                "probe_attempts": 1,
                "transcode_attempts": 1,
            },
        }
    )


def _metadata(duration_ns: int = 25_000_000_000) -> VideoMetadata:
    return VideoMetadata(
        duration_ns=duration_ns,
        width=1920,
        height=1080,
        frame_rate=30.0,
        frame_count=750,
        video_codec="h264",
    )


def _happy_media(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_ns: int = 25_000_000_000,
    observed: dict[str, Any] | None = None,
) -> None:
    state = observed if observed is not None else {}

    def fake_download(source_uri: str, destination_path: str, **_kwargs: object) -> None:
        state.setdefault("downloads", []).append(source_uri)
        Path(destination_path).write_bytes(b"source bytes")

    def fake_probe(path: Path, **_kwargs: object) -> VideoMetadata:
        return _metadata(duration_ns) if path.name == "source.mp4" else _metadata(10_000_000_000)

    def fake_transcode(source: Path, destination: Path, span: Span, *_args: object, **_kwargs: object) -> None:
        assert source.read_bytes() == b"source bytes"
        state.setdefault("transcode_sources", []).append(source)
        state.setdefault("spans", []).append(span)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"clip bytes")

    def fake_upload(source_path: str, destination_uri: str, **_kwargs: object) -> None:
        state.setdefault("uploads", []).append((destination_uri, Path(source_path).read_bytes()))

    monkeypatch.setattr(processing, "download_file", fake_download)
    monkeypatch.setattr(processing, "probe_video_path", fake_probe)
    monkeypatch.setattr(processing, "transcode_span_to_path", fake_transcode)
    monkeypatch.setattr(processing, "upload_file", fake_upload)


def test_source_is_downloaded_once_and_reused_for_all_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning and all three transcodes share one worker-local source file."""
    observed: dict[str, Any] = {}
    saved: list[list[dict[str, Any]]] = []
    _happy_media(monkeypatch, observed=observed)
    monkeypatch.setattr(processing, "write_source_result", lambda records, **_kwargs: saved.append(records))

    result = processing.process_source({"source_uri": _SOURCE_URI}, config=_config())

    assert observed["downloads"] == [_SOURCE_URI]
    assert len(set(observed["transcode_sources"])) == 1
    assert [(span.start_ns, span.end_ns) for span in observed["spans"]] == [
        (0, 10_000_000_000),
        (10_000_000_000, 20_000_000_000),
        (20_000_000_000, 25_000_000_000),
    ]
    assert len(observed["uploads"]) == 3
    assert all(payload == b"clip bytes" for _, payload in observed["uploads"])
    assert [row["record_type"] for row in result] == ["source_outcome", *("clip_outcome" for _ in range(3))]
    assert result[0]["status"] == "success"
    assert (result[0]["planned_clip_count"], result[0]["published_clip_count"]) == (3, 3)
    assert all(row["status"] == "success" for row in result[1:])
    assert saved == [result]


def test_saved_result_skips_source_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replacement Ray run can restore a source without downloading it."""
    saved = [{"record_type": "source_outcome"}, {"record_type": "clip_outcome"}]
    monkeypatch.setattr(processing, "restore_source_result", lambda *_args, **_kwargs: saved)
    monkeypatch.setattr(processing, "download_file", _raise(AssertionError("source should not be read")))

    assert processing.process_source({"source_uri": _SOURCE_URI}, config=_config()) is saved


def test_valid_short_source_is_checkpointed_without_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successfully probed source below the minimum tail length is complete."""
    saved: list[list[dict[str, Any]]] = []
    _happy_media(monkeypatch, duration_ns=1_000_000_000)
    monkeypatch.setattr(processing, "write_source_result", lambda records, **_kwargs: saved.append(records))

    result = processing.process_source({"source_uri": _SOURCE_URI}, config=_config())

    assert [record["record_type"] for record in result] == ["source_outcome"]
    assert result[0]["status"] == "success"
    assert result[0]["planned_clip_count"] == 0
    assert saved == [result]


def _raise(error: Exception) -> Callable[..., object]:
    def _raiser(*_args: object, **_kwargs: object) -> object:
        raise error

    return _raiser


@pytest.mark.parametrize(
    ("failing", "error", "stage"),
    [
        ("download_file", OSError("connection reset"), "source-read"),
        ("probe_video_path", ProbeError("unreadable header"), "source-probe"),
    ],
)
def test_expected_source_failures_become_one_outcome_naming_their_stage(
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    error: Exception,
    stage: str,
) -> None:
    """A source that cannot be read is a retryable data outcome, not a checkpoint."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, failing, _raise(error))
    saved: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(processing, "write_source_result", lambda records, **_kwargs: saved.append(records))

    records = processing.process_source({"source_uri": _SOURCE_URI}, config=_config())

    assert [record["record_type"] for record in records] == ["source_outcome"]
    assert records[0]["status"] == "failed"
    assert records[0]["error_stage"] == stage
    assert not saved


def test_unexpected_source_error_fails_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bug must not turn every source into a failure row and publish emptiness."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, "probe_video_path", _raise(KeyError("width")))

    with pytest.raises(KeyError):
        processing.process_source({"source_uri": _SOURCE_URI}, config=_config())


def test_missing_probe_tool_fails_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken worker environment must not be published as a bad source."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, "probe_video_path", _raise(FileNotFoundError("ffprobe")))

    with pytest.raises(FileNotFoundError, match="ffprobe"):
        processing.process_source({"source_uri": _SOURCE_URI}, config=_config())


def test_one_clip_failure_keeps_siblings_but_not_a_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The current run keeps good clips, while a rerun retries the whole source."""
    observed: dict[str, Any] = {}
    _happy_media(monkeypatch, observed=observed)

    def fail_middle_clip(
        source: Path,
        destination: Path,
        span: Span,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        if span.start_ns == 10_000_000_000:
            message = "broken GOP"
            raise TranscodeError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(processing, "transcode_span_to_path", fail_middle_clip)
    saved: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(processing, "write_source_result", lambda records, **_kwargs: saved.append(records))

    records = processing.process_source({"source_uri": _SOURCE_URI}, config=_config())

    assert [row["status"] for row in records] == ["failed", "success", "success"]
    assert records[0]["error_stage"] == "transcode"
    assert (records[0]["planned_clip_count"], records[0]["published_clip_count"], records[0]["failed_clip_count"]) == (
        3,
        2,
        1,
    )
    assert len(observed["uploads"]) == 2
    assert not saved


def test_media_write_failure_fails_the_task_without_a_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared unwritable destination is an operational failure, not a bad clip."""
    _happy_media(monkeypatch)
    error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "write permission denied"}},
        "PutObject",
    )
    monkeypatch.setattr(processing, "upload_file", _raise(error))
    saved: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(processing, "write_source_result", lambda records, **_kwargs: saved.append(records))

    with pytest.raises(ClientError):
        processing.process_source({"source_uri": _SOURCE_URI}, config=_config())

    assert not saved


def test_unexpected_clip_error_fails_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A programming error in transcoding is not converted into a media outcome."""
    _happy_media(monkeypatch)
    monkeypatch.setattr(processing, "transcode_span_to_path", _raise(ValueError("bad argument")))

    with pytest.raises(ValueError, match="bad argument"):
        processing.process_source({"source_uri": _SOURCE_URI}, config=_config())
