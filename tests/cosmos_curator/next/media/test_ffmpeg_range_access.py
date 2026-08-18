# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that cloud transcoding uses FFmpeg's range-capable input path."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_curator.next.media import ffmpeg
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split.config import TranscodeConfig


def test_http_transcode_uses_input_seek_before_signed_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Input-side seeking lets FFmpeg issue HTTP ranges instead of staging a whole object."""
    seen: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen.extend(command)
        Path(command[-1]).write_bytes(b"mp4")
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    destination = tmp_path / "clip.mp4"
    source = "https://s3.example/object?X-Amz-Signature=secret"

    ffmpeg.transcode_span_to_path(
        source,
        destination,
        Span(start_ns=10_000_000_000, end_ns=20_000_000_000),
        TranscodeConfig(),
    )

    assert seen.index("-ss") < seen.index("-i")
    assert seen[seen.index("-i") + 1] == source
    assert seen[seen.index("-seekable") + 1] == "1"
    assert destination.read_bytes() == b"mp4"


def test_signed_url_is_redacted_from_ffmpeg_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Durable source diagnostics never persist presigned credentials."""
    source = "https://s3.example/object?X-Amz-Signature=secret"

    def fail(command: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command, stderr=f"failed to read {source}".encode())

    monkeypatch.setattr(ffmpeg.subprocess, "run", fail)

    with pytest.raises(ffmpeg.TranscodeError) as exc_info:
        ffmpeg.transcode_span_to_path(
            source,
            tmp_path / "clip.mp4",
            Span(start_ns=0, end_ns=1_000_000_000),
            TranscodeConfig(),
        )

    assert "secret" not in str(exc_info.value)
    assert "<source>" in str(exc_info.value)
