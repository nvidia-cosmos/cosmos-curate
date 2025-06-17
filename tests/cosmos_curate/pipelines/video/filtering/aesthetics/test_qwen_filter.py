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

"""Test the QWEN result."""

import pathlib
import uuid

import pytest

from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (  # type: ignore[import-untyped]
    ClipTranscodingStage,
)
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_stages import (  # type: ignore[import-untyped]
    QwenFilteringStage,
    QwenInputPreparationStageFiltering,
)
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video  # type: ignore[import-untyped]
from tests.utils.sequential_runner import run_pipeline


@pytest.fixture
def sample_filtering_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clips = []
    for start, end in [(0, 3), (11, 14)]:
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"sample_video.mp4#{start}-{end}"),
            source_video="sample_video.mp4",
            span=(start, end),
        )
        clips.append(clip)

    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        source_bytes=sample_video_data,
        clips=clips,
    )
    return SplitPipeTask(
        video=video,
    )


@pytest.mark.env("vllm")
def test_generate_embedding(sample_filtering_task: SplitPipeTask) -> None:
    """Test the QwenCaptioning result."""
    filtering_prompt = "blue car"
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        QwenInputPreparationStageFiltering(sampling_fps=2.0, filter_categories=filtering_prompt),
        QwenFilteringStage(verbose=True, user_prompt=filtering_prompt),
    ]
    tasks = run_pipeline([sample_filtering_task], stages)

    assert tasks is not None
    assert len(tasks) > 0

    passing_clips = tasks[0].video.clips
    assert len(passing_clips) == 1
    assert passing_clips[0].uuid == uuid.uuid5(uuid.NAMESPACE_URL, "sample_video.mp4#0-3")

    failing_clips = tasks[0].video.filtered_clips
    assert len(failing_clips) == 1
    assert failing_clips[0].uuid == uuid.uuid5(uuid.NAMESPACE_URL, "sample_video.mp4#11-14")
