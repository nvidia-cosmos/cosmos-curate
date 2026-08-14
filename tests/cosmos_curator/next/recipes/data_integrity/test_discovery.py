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

"""Unit tests for session stream discovery."""

import pathlib

import pytest

from cosmos_curator.next.recipes.data_integrity import discovery
from cosmos_curator.next.recipes.data_integrity.discovery import discover_streams


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_local_dir_finds_videos_recursively_and_ignores_side_files(tmp_path: pathlib.Path) -> None:
    """Discovery recurses recorder-XX subdirs for videos and skips rig JSON / markers."""
    _touch(tmp_path / "recorder-00" / "a.mp4")
    _touch(tmp_path / "recorder-01" / "b.mkv")
    _touch(tmp_path / "rig_meta.json")
    _touch(tmp_path / "_UPLOADED_TS")
    _touch(tmp_path / "notes.txt")

    found = discover_streams(str(tmp_path))

    assert found == sorted([str(tmp_path / "recorder-00" / "a.mp4"), str(tmp_path / "recorder-01" / "b.mkv")])


def test_local_dir_limit_caps_and_takes_sorted_prefix(tmp_path: pathlib.Path) -> None:
    """limit>0 returns the first N in sorted order."""
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        _touch(tmp_path / name)

    found = discover_streams(str(tmp_path), limit=2)

    assert found == [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]


def test_single_file_returns_one_element(tmp_path: pathlib.Path) -> None:
    """A lone file path is returned as a single-element list."""
    video = tmp_path / "solo.mp4"
    _touch(video)
    assert discover_streams(str(video)) == [str(video)]


def test_single_non_video_file_is_filtered_out(tmp_path: pathlib.Path) -> None:
    """An explicit non-video file (e.g. rig_meta.json) is filtered, not returned as a bogus stream."""
    side = tmp_path / "rig_meta.json"
    _touch(side)
    assert discover_streams(str(side)) == []


def test_negative_limit_raises(tmp_path: pathlib.Path) -> None:
    """A negative limit is rejected at the discovery boundary, consistently across backends."""
    with pytest.raises(ValueError, match="limit"):
        discover_streams(str(tmp_path), limit=-1)


def test_missing_local_path_raises(tmp_path: pathlib.Path) -> None:
    """A nonexistent local path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        discover_streams(str(tmp_path / "nope"))


@pytest.mark.parametrize("scheme", ["s3", "az"])
def test_cloud_prefix_delegates_to_listing_with_video_suffixes(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    """Both cloud schemes reach list_cloud_objects with the video suffix filter and limit, sorted."""
    captured: dict[str, object] = {}
    prefix = f"{scheme}://b/clips/u/"

    def _fake_list(listed_prefix: str, **kwargs: object) -> list[str]:
        captured["prefix"] = listed_prefix
        captured.update(kwargs)
        return [f"{prefix}recorder-01/b.mp4", f"{prefix}recorder-00/a.mp4"]

    monkeypatch.setattr(discovery, "list_cloud_objects", _fake_list)

    found = discover_streams(prefix, limit=5, s3_profile_name="maglev", endpoint_url="https://endpoint.io")

    assert found == [f"{prefix}recorder-00/a.mp4", f"{prefix}recorder-01/b.mp4"]
    assert captured["prefix"] == prefix
    assert captured["limit"] == 5
    assert captured["suffixes"] == discovery.VIDEO_SUFFIXES
    assert captured["endpoint_url"] == "https://endpoint.io"
    assert captured["s3_profile_name"] == "maglev"
