# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for dataset dimension helpers."""

import pytest

from cosmos_curate.core.utils.dataset.dimensions import (
    Dimensions,
    ResolutionAspectRatio,
    ResolutionAspectRatioFrames,
    ResolutionAspectRatioFramesBinsSpec,
    _AspectRatio,
    _AspectRatioBinSpec,
    _AspectRatioBinsSpec,
    _round_to_nearest_even,
)


def test_round_to_nearest_even_handles_even_and_odd_values() -> None:
    """Ensure rounding hits the closest even integer with ties rounding up."""
    assert _round_to_nearest_even(2) == 2
    assert _round_to_nearest_even(3) == 4
    assert _round_to_nearest_even(5) == 6  # tie goes to upper even
    assert _round_to_nearest_even(8) == 8
    assert _round_to_nearest_even(9) == 10


def test_dimensions_ratio_and_resize_preserve_aspect() -> None:
    """Resizing uses the shorter edge and keeps ratios via even rounding."""
    dims = Dimensions(width=400, height=200)
    assert dims.w_by_h == pytest.approx(2.0)
    resized = dims.resize_by_shortest_dimension(new_short_size=100)
    assert resized == Dimensions(width=200, height=100)

    portrait = Dimensions(width=300, height=500)
    resized_portrait = portrait.resize_by_shortest_dimension(new_short_size=200)
    assert resized_portrait == Dimensions(width=200, height=334)


def test_aspect_ratio_to_and_from_path_strings() -> None:
    """String conversions round-trip even when embedded in longer paths."""
    ratio = _AspectRatio(width=16, height=9)
    assert ratio.to_path_string() == "aspect_ratio_16_9"
    parsed = _AspectRatio.from_path_string("foo/bar/aspect_ratio_16_9/baz")
    assert parsed == ratio


def test_resolution_aspect_ratio_path_roundtrip() -> None:
    """Resolution/aspect-ratio composite strings parse correctly."""
    ar = _AspectRatio(width=4, height=3)
    resolution_ar = ResolutionAspectRatio(aspect_ratio=ar, resolution="1080")
    path = resolution_ar.to_path_string()
    assert path == "resolution_1080/aspect_ratio_4_3"

    parsed = ResolutionAspectRatio.from_path_string(f"/some/{path}")
    assert parsed == resolution_ar


def test_resolution_aspect_ratio_frames_path_roundtrip() -> None:
    """Resolution/aspect-ratio/frames strings parse correctly."""
    ar = _AspectRatio(width=1, height=1)
    rar_frames = ResolutionAspectRatioFrames(aspect_ratio=ar, resolution="720", frames="256_1023")
    path = rar_frames.to_path_string()
    assert path == "resolution_720/aspect_ratio_1_1/frames_256_1023"

    parsed = ResolutionAspectRatioFrames.from_path_string(path)
    assert parsed == rar_frames


def test_aspect_ratio_bins_spec_requires_contiguous_ranges() -> None:
    """Bins with gaps trigger validation errors."""
    with pytest.raises(ValueError, match="Expected bins to be contiguous"):
        _AspectRatioBinsSpec(
            [
                _AspectRatioBinSpec(0.0, 0.5, _AspectRatio(width=1, height=2)),
                _AspectRatioBinSpec(0.6, 1.0, _AspectRatio(width=1, height=1)),
            ],
        )


def test_standard_aspect_ratio_bins_find_expected_bin() -> None:
    """Standard bins bucket images based on width/height ratio."""
    bins = _AspectRatioBinsSpec.for_standard_image_datasets()
    dims = Dimensions(width=400, height=300)  # ratio 4:3
    ar = bins.find_appropriate_bin(dims)
    assert ar == _AspectRatio(width=4, height=3)

    wide_dims = Dimensions(width=10000, height=100)  # ratio > 10
    assert bins.find_appropriate_bin(wide_dims) is None


def test_resolution_frames_bins_select_resolution_and_frame_bucket() -> None:
    """Video bins consider resolution, aspect ratio, and frame count."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    dims = Dimensions(width=1920, height=1080)
    bucket = bins.find_appropriate_bin(dims, num_frames=80)
    assert bucket == ResolutionAspectRatioFrames(_AspectRatio(16, 9), "1080", "0_120")

    high_frame_bucket = bins.find_appropriate_bin(dims, num_frames=600)
    assert high_frame_bucket == ResolutionAspectRatioFrames(_AspectRatio(16, 9), "1080", "256_1023")


def test_resolution_frames_bins_reject_low_resolution_inputs() -> None:
    """Frames that do not meet the min resolution are discarded."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    low_res_dims = Dimensions(width=640, height=480)
    assert bins.find_appropriate_bin(low_res_dims, num_frames=30) is None


def test_find_appropriate_image_bin_wraps_video_bin_result() -> None:
    """Image bin lookup reuses the video bin logic with num_frames=1."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    dims = Dimensions(width=1600, height=900)  # min edge >= 720 so use 720 bin
    result = bins.find_appropriate_image_bin(dims)
    assert result == ResolutionAspectRatio(_AspectRatio(16, 9), "720")
