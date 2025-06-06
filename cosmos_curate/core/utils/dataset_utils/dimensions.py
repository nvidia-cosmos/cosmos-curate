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
"""Utilities for managing dimensions and aspect ratios of images."""

import re

import attrs

from cosmos_curate.core.utils import grouping

# Constants for aspect ratio string representation
_AR_PREFIX = "aspect_ratio_"
_AR_COMPILED_RE = re.compile(r"aspect_ratio_(\d+)_(\d+)")
_RAR_COMPILED_RE = re.compile(r"resolution_(\d+)/aspect_ratio_(\d+)_(\d+)")
_RARF_COMPILED_RE = re.compile(r"resolution_(\d+)/aspect_ratio_(\d+)_(\d+)/frames_(.+)")


def _round_to_nearest_even(n: int) -> int:
    """Round the provided number to the nearest even integer.

    Args:
        n (int): Number to round.

    Returns:
        int: Nearest even integer.

    """
    # If the number is already even, return it
    if n % 2 == 0:
        return int(n)

    # Find the nearest even numbers
    floor_even = int(n) // 2 * 2  # Largest even number <= n
    ceil_even = floor_even + 2  # Smallest even number > n

    # Return the even number that n is closest to
    return floor_even if n - floor_even < ceil_even - n else ceil_even


@attrs.define(frozen=True)
class Dimensions:
    """Represents the dimensions (height and width) of an image."""

    width: int
    height: int

    @property
    def w_by_h(self) -> float:
        """Calculate the width-to-height ratio.

        Returns:
            float: Width-to-height ratio.

        """
        return float(self.width) / self.height

    def resize_by_shortest_dimension(self, new_short_size: int) -> "Dimensions":
        """Resize the dimensions while maintaining the aspect ratio, based on a new shortest dimension value.

        Args:
            new_short_size (int): New size for the shorter dimension.

        Returns:
            Dimensions: New resized dimensions.

        """
        assert new_short_size % 2 == 0
        h, w = self.height, self.width
        if h < w:
            return Dimensions(_round_to_nearest_even(round((new_short_size / h) * w)), new_short_size)
        return Dimensions(new_short_size, _round_to_nearest_even(round((new_short_size / w) * h)))


@attrs.define(frozen=True)
class _AspectRatio:
    """Represents the aspect ratio of an image or a group of images."""

    width: int
    height: int

    def to_path_string(self) -> str:
        """Convert aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the aspect ratio.

        """
        return _AR_PREFIX + f"{self.width}_{self.height}"

    @classmethod
    def from_path_string(cls, path_str: str) -> "_AspectRatio":
        """Create an _AspectRatio object from its string representation.

        Args:
            path_str (str): String representation of the aspect ratio.

        Returns:
            _AspectRatio: Corresponding _AspectRatio object.

        """
        match = _AR_COMPILED_RE.search(path_str)
        assert match
        w, h = int(match.group(1)), int(match.group(2))
        return _AspectRatio(height=h, width=w)


@attrs.define(frozen=True)
class ResolutionAspectRatio:
    """Represents the resolution/aspect_ratio of an image or a group of images."""

    aspect_ratio: _AspectRatio
    resolution: str

    def to_path_string(self) -> str:
        """Convert resolution and aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the resolution and aspect ratio.

        """
        return f"resolution_{self.resolution}/aspect_ratio_{self.aspect_ratio.width}_{self.aspect_ratio.height}"

    @classmethod
    def from_path_string(cls, path_str: str) -> "ResolutionAspectRatio":
        """Create an ResolutionAspectRatio object from its string representation.

        Args:
            path_str (str): String representation of the resolution/aspect_ratio.

        Returns:
            ResolutionAspectRatio: Corresponding ResolutionAspectRatio object.

        """
        match = _RAR_COMPILED_RE.search(path_str)
        assert match
        resolution = match.group(1)
        w, h = int(match.group(2)), int(match.group(3))
        aspect_ratio = _AspectRatio(height=h, width=w)
        return ResolutionAspectRatio(aspect_ratio, resolution)


@attrs.define(frozen=True)
class ResolutionAspectRatioFrames:
    """Represents the resolution/aspect_ratio/number_of_frames of an video or a group of videos."""

    aspect_ratio: _AspectRatio
    resolution: str
    frames: str

    def to_path_string(self) -> str:
        """Convert aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the aspect ratio.

        """
        return f"resolution_{self.resolution}/aspect_ratio_{self.aspect_ratio.width}_{self.aspect_ratio.height}/frames_{self.frames}"  # noqa: E501

    @classmethod
    def from_path_string(cls, path_str: str) -> "ResolutionAspectRatioFrames":
        """Create an ResolutionAspectRatioFrames object from its string representation.

        Args:
            path_str (str): String representation of the resolution/aspect_ratio/number_of_frames.

        Returns:
            ResolutionAspectRatioFrames: Corresponding ResolutionAspectRatioFrames object.

        """
        match = _RARF_COMPILED_RE.search(path_str)
        assert match
        resolution = match.group(1)
        w, h = int(match.group(2)), int(match.group(3))
        aspect_ratio = _AspectRatio(height=h, width=w)
        frames = match.group(4)
        return ResolutionAspectRatioFrames(aspect_ratio, resolution, frames)


@attrs.define
class _AspectRatioBinSpec:
    """Defines a bin specification for categorizing images based on their aspect ratios."""

    min_w_by_h: float  # Minimum width-to-height ratio for the bin
    max_w_by_h: float  # Maximum width-to-height ratio for the bin
    aspect_ratio: _AspectRatio  # Associated aspect ratio for the bin


@attrs.define
class _AspectRatioBinsSpec:
    """Collection of _AspectRatioBinSpec objects, providing utilities to categorize and manage aspect ratios."""

    bins: list[_AspectRatioBinSpec]

    def __attrs_post_init__(self) -> None:
        self._validate_contiguous_bins()

    @classmethod
    def for_standard_image_datasets(cls) -> "_AspectRatioBinsSpec":
        """Create a standard set of aspect ratio bins suitable for VFM datasets.

        Returns:
            _AspectRatioBinsSpec: Collection of bins.

        """
        out = []

        def append_bin(min_ratio: float, max_ratio: float, ar_width: int, ar_height: int) -> None:
            out.append(_AspectRatioBinSpec(min_ratio, max_ratio, _AspectRatio(ar_width, ar_height)))

        append_bin(0, 0.65, 9, 16)
        append_bin(0.65, 0.88, 3, 4)
        append_bin(0.88, 1.16, 1, 1)
        append_bin(1.16, 1.55, 4, 3)
        append_bin(1.55, 10, 16, 9)

        return _AspectRatioBinsSpec(out)

    def _validate_contiguous_bins(self) -> None:
        """Ensure that the aspect ratio bins are contiguous, i.e., there are no gaps between them.

        Raises:
            ValueError: If any non-contiguous bins are found.

        """
        for first, second in grouping.pairwise(self.bins):
            if first.max_w_by_h != second.min_w_by_h:
                error_msg = f"Expected bins to be contiguous, but got {first} and {second}."
                raise ValueError(error_msg)

    def find_appropriate_bin(self, dimensions: Dimensions) -> _AspectRatio | None:
        """Identify the appropriate aspect ratio bin for given image dimensions.

        Args:
            dimensions (Dimensions): Image dimensions to categorize.

        Returns:
            _AspectRatio: Appropriate aspect ratio bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        for sbin in self.bins:
            if sbin.min_w_by_h < dimensions.w_by_h <= sbin.max_w_by_h:
                return sbin.aspect_ratio
        return None


@attrs.define
class _ResolutionAspectRatioFramesBinSpec:
    """Defines a bin specification for categorizing images based on their resolution/aspect_ratio/number_of_frames."""

    min_w_by_h: float  # Minimum width by height aspect ratio
    max_w_by_h: float  # Maximum width by height aspect ratio
    aspect_ratio: _AspectRatio  # Aspect ratio
    min_frames: int  # Minimum number of frames
    max_frames: int  # Maximum number of frames
    frames: str  # Frames 'lt/gt_number_of_frames"
    resolution: str  # Resolution, assuming string format but adjust as needed


@attrs.define
class ResolutionAspectRatioFramesBinsSpec:
    """Collection of _AspectRatioBinSpec objects.

    providing utilities to categorize and manage resolution/aspect_ratio/number_of_frames.
    """

    bins: list[_ResolutionAspectRatioFramesBinSpec]

    def __attrs_post_init__(self) -> None:
        """Post-init."""
        self._validate_contiguous_bins()

    @classmethod
    def for_standard_video_datasets(cls) -> "ResolutionAspectRatioFramesBinsSpec":
        """Create a standard set of bins suitable for VFM datasets.

        Returns:
            ResolutionAspectRatioFramesBinsSpec: Collection of bins.

        """
        out = []

        def append_bin(  # noqa: PLR0913
            min_ratio: float,
            max_ratio: float,
            ar_width: int,
            ar_height: int,
            min_frames: int,
            max_frames: int,
            frames: str,
            resolution: str,
        ) -> None:
            out.append(
                _ResolutionAspectRatioFramesBinSpec(
                    min_ratio,
                    max_ratio,
                    _AspectRatio(ar_width, ar_height),
                    min_frames,
                    max_frames,
                    frames,
                    resolution,
                ),
            )

        append_bin(0, 0.65, 9, 16, 0, 120, "0_120", "1080")
        append_bin(0.65, 0.88, 3, 4, 0, 120, "0_120", "1080")
        append_bin(0.88, 1.16, 1, 1, 0, 120, "0_120", "1080")
        append_bin(1.16, 1.55, 4, 3, 0, 120, "0_120", "1080")
        append_bin(1.55, 10, 16, 9, 0, 120, "0_120", "1080")
        append_bin(0, 0.65, 9, 16, 0, 120, "0_120", "720")
        append_bin(0.65, 0.88, 3, 4, 0, 120, "0_120", "720")
        append_bin(0.88, 1.16, 1, 1, 0, 120, "0_120", "720")
        append_bin(1.16, 1.55, 4, 3, 0, 120, "0_120", "720")
        append_bin(1.55, 10, 16, 9, 0, 120, "0_120", "720")
        append_bin(0, 0.65, 9, 16, 121, 255, "121_255", "1080")
        append_bin(0.65, 0.88, 3, 4, 121, 255, "121_255", "1080")
        append_bin(0.88, 1.16, 1, 1, 121, 255, "121_255", "1080")
        append_bin(1.16, 1.55, 4, 3, 121, 255, "121_255", "1080")
        append_bin(1.55, 10, 16, 9, 121, 255, "121_255", "1080")
        append_bin(0, 0.65, 9, 16, 121, 255, "121_255", "720")
        append_bin(0.65, 0.88, 3, 4, 121, 255, "121_255", "720")
        append_bin(0.88, 1.16, 1, 1, 121, 255, "121_255", "720")
        append_bin(1.16, 1.55, 4, 3, 121, 255, "121_255", "720")
        append_bin(1.55, 10, 16, 9, 121, 255, "121_255", "720")
        append_bin(0, 0.65, 9, 16, 256, 1023, "256_1023", "1080")
        append_bin(0.65, 0.88, 3, 4, 256, 1023, "256_1023", "1080")
        append_bin(0.88, 1.16, 1, 1, 256, 1023, "256_1023", "1080")
        append_bin(1.16, 1.55, 4, 3, 256, 1023, "256_1023", "1080")
        append_bin(1.55, 10, 16, 9, 256, 1023, "256_1023", "1080")
        append_bin(0, 0.65, 9, 16, 256, 1023, "256_1023", "720")
        append_bin(0.65, 0.88, 3, 4, 256, 1023, "256_1023", "720")
        append_bin(0.88, 1.16, 1, 1, 256, 1023, "256_1023", "720")
        append_bin(1.16, 1.55, 4, 3, 256, 1023, "256_1023", "720")
        append_bin(1.55, 10, 16, 9, 256, 1023, "256_1023", "720")
        append_bin(0, 0.65, 9, 16, 1024, 2**63 - 1, "1024_inf", "1080")
        append_bin(0.65, 0.88, 3, 4, 1024, 2**63 - 1, "1024_inf", "1080")
        append_bin(0.88, 1.16, 1, 1, 1024, 2**63 - 1, "1024_inf", "1080")
        append_bin(1.16, 1.55, 4, 3, 1024, 2**63 - 1, "1024_inf", "1080")
        append_bin(1.55, 10, 16, 9, 1024, 2**63 - 1, "1024_inf", "1080")
        append_bin(0, 0.65, 9, 16, 1024, 2**63 - 1, "1024_inf", "720")
        append_bin(0.65, 0.88, 3, 4, 1024, 2**63 - 1, "1024_inf", "720")
        append_bin(0.88, 1.16, 1, 1, 1024, 2**63 - 1, "1024_inf", "720")
        append_bin(1.16, 1.55, 4, 3, 1024, 2**63 - 1, "1024_inf", "720")
        append_bin(1.55, 10, 16, 9, 1024, 2**63 - 1, "1024_inf", "720")

        return ResolutionAspectRatioFramesBinsSpec(out)

    def _validate_contiguous_bins(self) -> None:
        """Ensure that the aspect ratio bins are contiguous, i.e., there are no gaps between them.

        Raises:
            ValueError: If any non-contiguous bins are found.

        """

    def find_appropriate_bin(self, dimensions: Dimensions, num_frames: int) -> ResolutionAspectRatioFrames | None:
        """Identify the appropriate aspect ratio bin for given video metadata.

        Currently the resolution bin only has two categories:
            greater than 1080 and great than 720,
            low resolution videos will be discarded.


        Args:
            dimensions: image dimensions
            num_frames: number of frames

        Returns:
            ResolutionAspectRatioFrames: Appropriate resolution/aspect_ratio/number_of_frames bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        min_vres: int = 1080
        min_hres: int = 720
        width = dimensions.width
        height = dimensions.height
        w_by_h = dimensions.w_by_h
        if min(height, width) >= min_vres:
            resolution = "1080"
        elif min(height, width) >= min_hres:
            resolution = "720"
        else:
            resolution = "0"
        for sbin in self.bins:
            if (
                sbin.resolution == resolution
                and sbin.min_w_by_h < w_by_h <= sbin.max_w_by_h
                and sbin.min_frames <= num_frames <= sbin.max_frames
            ):
                return ResolutionAspectRatioFrames(sbin.aspect_ratio, resolution, sbin.frames)
        return None

    def find_appropriate_image_bin(self, dimensions: Dimensions) -> ResolutionAspectRatio | None:
        """Identify the appropriate aspect ratio bin for given image metadata.

        Currently the resolution bin only has two categories:
            greater than 1080 and great than 720,
            low resolution images will be discarded.


        Args:
            dimensions: image dimensions to categorize.

        Returns:
            ResolutionAspectRatio: Appropriate resolution/aspect_ratio bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        image_bin = self.find_appropriate_bin(dimensions, 1)
        if image_bin:
            return ResolutionAspectRatio(image_bin.aspect_ratio, image_bin.resolution)
        return None
