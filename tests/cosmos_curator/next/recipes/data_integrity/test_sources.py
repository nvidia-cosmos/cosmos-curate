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

"""Unit tests for turning a URI into an open sensor and running the metrics over it."""

import contextlib
import io
import pathlib
import threading
from collections.abc import Callable
from typing import BinaryIO

import av
import pytest

from cosmos_curator.next.recipes.data_integrity.cli_support import cancellable_reader
from cosmos_curator.next.recipes.data_integrity.sources import open_source, run_checks


def _open_local(path: pathlib.Path, wrapper: object = None) -> None:
    """Read one byte from *path* through open_source, exercising the local branch."""
    with open_source(
        str(path),
        s3_profile_name=None,
        azure_profile_name="default",
        stream_wrapper=wrapper,  # type: ignore[arg-type]
    ) as stream:
        stream.read(1)


def test_a_local_path_is_opened_as_a_stream_and_wrapped(tmp_path: pathlib.Path) -> None:
    """The wrapper reaches local paths too, not just cloud URIs.

    A local path is not always a fast one: on a shared filesystem it reads like a remote
    object, so progress counting and cancellation have to apply there as well. Handing
    the sensor a Path instead would skip the wrapper entirely.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")
    seen: list[bytes] = []

    def _wrap(stream: BinaryIO) -> BinaryIO:
        seen.append(stream.read(3))
        stream.seek(0)
        return stream

    _open_local(clip, _wrap)

    assert seen == [b"not"], "a local path bypassed the stream wrapper, or arrived pre-read"


def test_a_local_stream_is_closed_on_exit(tmp_path: pathlib.Path) -> None:
    """open_source owns the handle it opened, so the caller cannot leak descriptors."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")
    captured: list[BinaryIO] = []

    with open_source(str(clip), s3_profile_name=None, azure_profile_name="default") as stream:
        captured.append(stream)
        assert not stream.closed

    assert captured[0].closed


def test_cancelling_stops_a_local_read_too(tmp_path: pathlib.Path) -> None:
    """Ctrl-C aborts a local read at the next read boundary, as it does for a cloud one."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 4096)
    cancel = threading.Event()

    with open_source(
        str(clip),
        s3_profile_name=None,
        azure_profile_name="default",
        stream_wrapper=lambda stream: cancellable_reader(stream, cancel),
    ) as stream:
        assert stream.read(16) == b"x" * 16
        cancel.set()
        assert stream.read(16) == b"", "kept reading a local file after the abort"


def test_an_abort_stops_a_real_local_check(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> None:
    """End to end through libav: an abort leaves a local source with nothing to decode.

    The pairing is the point. The same clip evaluates normally when the event is clear,
    so the failure below is the abort taking effect rather than a bad file, which is only
    possible because local paths now go through the wrapper instead of being handed to
    the sensor as a Path and read straight off disk.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(h264_video(bframes=0))
    cancel = threading.Event()

    def _check() -> object:
        return run_checks(str(clip), expected_hz=None, stream_wrapper=lambda stream: cancellable_reader(stream, cancel))

    results, _info, _cfg = _check()  # type: ignore[misc]
    assert results, "the clip should evaluate normally while the event is clear"

    cancel.set()
    with pytest.raises(av.error.InvalidDataError):
        _check()


def test_a_missing_local_path_raises_at_open(tmp_path: pathlib.Path) -> None:
    """Opening eagerly moves the failure to open_source, where both CLIs report exit 2."""
    with pytest.raises(FileNotFoundError):
        _open_local(tmp_path / "nope.mp4")


def test_a_cloud_source_still_takes_the_cloud_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening local paths here must not divert cloud URIs into the filesystem."""
    opened: list[str] = []

    @contextlib.contextmanager
    def _fake_open_storage_source(source: str, **_kwargs: object) -> "object":
        opened.append(source)
        yield io.BytesIO(b"cloud bytes")

    monkeypatch.setattr(
        "cosmos_curator.next.recipes.data_integrity.sources.open_storage_source", _fake_open_storage_source
    )

    with open_source("s3://bucket/key.mp4", s3_profile_name=None, azure_profile_name="default") as stream:
        assert stream.read() == b"cloud bytes"

    assert opened == ["s3://bucket/key.mp4"]
