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
"""Tests for clip_extraction_stages (chunking and slice_video_clips)."""

import pathlib
import uuid
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import slice_video_clips
from cosmos_curate.pipelines.video.utils.data_model import Clip, Video, VideoMetadata


def _make_video_for_slice(num_clips: int, clip_chunk_index: int = 0) -> Video:
    """Build a Video with num_clips dummy clips for slice_video_clips tests."""
    clips = [
        Clip(
            uuid=uuid.uuid4(),
            source_video="test.mp4",
            span=(float(i * 10), float((i + 1) * 10)),
        )
        for i in range(num_clips)
    ]
    return Video(
        input_video=pathlib.Path("test.mp4"),
        relative_path="test.mp4",
        metadata=VideoMetadata(duration=float(num_clips * 10), size=1000),
        clips=clips,
        num_total_clips=num_clips,
        num_clip_chunks=2,
        clip_chunk_index=clip_chunk_index,
        errors={"stage": "err"},
    )


@pytest.mark.parametrize(
    (
        "num_clips",
        "start",
        "end",
        "chunk_index",
        "num_chunks",
        "expected_num_clips",
        "expected_chunk_index",
        "source_clip_chunk_index",
        "raises",
    ),
    [
        # Success: sub set
        (5, 1, 4, 0, 2, 3, 0, 0, nullcontext()),
        # Success: explicit chunk_index
        (5, 0, 2, 1, 2, 2, 1, 0, nullcontext()),
        # Success: chunk_index from video
        (5, 0, 2, 2, 2, 2, 2, 2, nullcontext()),
        # Success: full range
        (4, 0, 4, 0, 2, 4, 0, 0, nullcontext()),
        # Success: single clip
        (5, 2, 3, 0, 2, 1, 0, 0, nullcontext()),
        # Failure: end < start  # noqa: ERA001
        (5, 3, 2, 0, 2, None, None, 0, pytest.raises(ValueError, match="End index 2 is less than start index 3")),
        # Failure: start < 0  # noqa: ERA001
        (5, -1, 2, 0, 2, None, None, 0, pytest.raises(ValueError, match="out of range")),
        # Failure: end > len(clips)  # noqa: ERA001
        (5, 0, 6, 0, 2, None, None, 0, pytest.raises(ValueError, match="out of range")),
    ],
)
def test_slice_video_clips(  # noqa: PLR0913
    num_clips: int,
    start: int,
    end: int,
    chunk_index: int,
    num_chunks: int,
    expected_num_clips: int | None,
    expected_chunk_index: int | None,
    source_clip_chunk_index: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test slice_video_clips: valid slices return new Video with correct clips/chunk_index; invalid args raise."""
    video = _make_video_for_slice(num_clips, clip_chunk_index=source_clip_chunk_index)
    with raises:
        result = slice_video_clips(video, start, end, chunk_index, num_chunks)
        assert result is not video
        assert expected_num_clips is not None
        assert expected_chunk_index is not None
        assert len(result.clips) == expected_num_clips
        assert result.clip_chunk_index == expected_chunk_index
        assert result.num_clip_chunks == num_chunks
        assert result.input_video == video.input_video
        assert result.relative_path == video.relative_path
        assert result.num_total_clips == num_clips
        if expected_num_clips > 0:
            assert result.clips[0] is video.clips[start]
        result.errors["other"] = "new"
        assert "other" not in video.errors
        assert result.errors is not video.errors
