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
"""Tests for capture-start parsing out of source video paths."""

import zoneinfo

import pytest

from cosmos_curator.pipelines.video.read_write import mcap_time

PACIFIC = zoneinfo.ZoneInfo("America/Los_Angeles")
UTC = zoneinfo.ZoneInfo("UTC")


@pytest.mark.parametrize(
    ("path", "expected_epoch_s", "expected_match"),
    [
        # Capture folder holding the minute, as the 375edge exports are laid out.
        pytest.param(
            "s3://zenith-edge-assets-playground/375edge/01/2026-08-18-09-00/3.mp4",
            1787068800,
            "2026-08-18-09-00",
            id="capture_folder",
        ),
        # Minute in the file stem instead.
        pytest.param(
            "/config/raw_videos_mp4/2026-08-10-09-00-bullet-far.mp4",
            1786377600,
            "2026-08-10-09-00",
            id="file_stem",
        ),
        # Dotted convention carrying both a start and an end time: the start wins.
        pytest.param(
            "s3://bucket/Various Samples/2018-03-05.13-10-00.13-15-00.admin.G326.r13.mp4",
            1520284200,
            "2018-03-05.13-10-00",
            id="dotted_start_and_end",
        ),
    ],
)
def test_parses_real_naming_conventions(path: str, expected_epoch_s: int, expected_match: str) -> None:
    """The three capture-naming conventions in use all resolve to the right instant."""
    parsed = mcap_time.parse_capture_start_ns(path, PACIFIC)
    assert parsed == (expected_epoch_s * 1_000_000_000, expected_match)


def test_zone_changes_the_instant() -> None:
    """The same path names a different instant in a different zone."""
    path = "s3://bucket/site/2026-08-18-09-00/3.mp4"
    pacific = mcap_time.parse_capture_start_ns(path, PACIFIC)
    utc = mcap_time.parse_capture_start_ns(path, UTC)
    assert pacific is not None
    assert utc is not None
    assert pacific[0] - utc[0] == 7 * 3600 * 1_000_000_000  # PDT is UTC-7 in August


def test_capture_folder_preferred_over_file_stem() -> None:
    """A capture folder wins over a stem, whose digits may be an unrelated run label."""
    path = "s3://bucket/2026-08-18-09-00/2020-01-01-00-00-run.mp4"
    parsed = mcap_time.parse_capture_start_ns(path, UTC)
    assert parsed is not None
    assert parsed[1] == "2026-08-18-09-00"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/config/raw_videos_mp4_mike/3PANEL.mp4", id="no_digits"),
        pytest.param("/x/MVI_0001.mp4", id="unrelated_digits"),
        pytest.param("/x/site-24-cam-evening-3-bullet-near.mp4", id="site_label"),
        pytest.param("/x/2026-13-45-09-00-bad.mp4", id="impossible_date"),
        pytest.param("/x/20260818-0900.mp4", id="wrong_separators"),
    ],
)
def test_unparseable_paths_yield_none(path: str) -> None:
    """Paths naming no usable time leave the writer on its 0-based fallback."""
    assert mcap_time.parse_capture_start_ns(path, UTC) is None
