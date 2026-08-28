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

"""Tests for the check_video_index diagnostic CLI."""

import pathlib
from collections.abc import Callable

import pytest

from cosmos_curator.next.recipes.data_integrity.check_video_index import (
    PASS_EXIT_CODE,
    IndexVerdict,
    _check_video_index,
    main,
)


def _write(tmp_path: pathlib.Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _verdict(source: str) -> IndexVerdict:
    verdict, _ = _check_video_index(
        source,
        stream_idx=0,
        video_format=None,
        s3_profile_name=None,
        azure_profile_name="default",
    )
    return verdict


def test_bframe_file_is_header_bypassed_not_corrupt(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> None:
    """A valid B-frame file is reported as header-bypassed, not a mismatch."""
    # B-frame input is a checked-in pre-encoded clip (this env's openh264 can't
    # encode B-frames; libx264/GPL is not bundled). See conftest.h264_video.
    source = _write(tmp_path, "bframe.mp4", h264_video(bframes=2))

    assert _verdict(source) is IndexVerdict.HEADER_BYPASSED


def test_no_bframe_file_is_consistent(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> None:
    """A B-frame-free file's header index matches the full scan."""
    source = _write(tmp_path, "plain.mp4", h264_video(bframes=0))

    assert _verdict(source) is IndexVerdict.CONSISTENT


def test_main_passes_bframe_file_without_corruption_language(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], h264_video: Callable[..., bytes]
) -> None:
    """main() exits 0 for a B-frame file and does not call it corrupt/converted incorrectly."""
    source = _write(tmp_path, "bframe.mp4", h264_video(bframes=2))

    code = main(["--source", source])

    out = capsys.readouterr().out
    assert code == PASS_EXIT_CODE
    assert "intentionally bypassed" in out
    assert "FAIL" not in out
    assert "converted incorrectly" not in out


def test_main_passes_consistent_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], h264_video: Callable[..., bytes]
) -> None:
    """main() exits 0 and reports consistency for a B-frame-free file."""
    source = _write(tmp_path, "plain.mp4", h264_video(bframes=0))

    code = main(["--source", source])

    out = capsys.readouterr().out
    assert code == PASS_EXIT_CODE
    assert "consistent" in out.lower()
