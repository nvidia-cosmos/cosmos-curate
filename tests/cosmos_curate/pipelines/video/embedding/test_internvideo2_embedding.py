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

"""Test the InternVideo2Embedding result."""

import pathlib
import uuid

import pytest
from loguru import logger

from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import ClipTranscodingStage
from cosmos_curate.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curate.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2EmbeddingStage,
    InternVideo2FrameCreationStage,
)
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curate.pipelines.video.utils.decoder_utils import FrameExtractionPolicy
from tests.utils.sequential_runner import run_pipeline

_MATCH_CONFIDENCE_SCORE = 0.9


def _get_texts() -> list[str]:
    return [
        "A man is sitting on a red truck and then the same red truck is seen driving on the highway",
        "A man is sitting on a blue truck and then the same blue truck is seen driving on the highway",
        "A man is playing soccer in the field",
        "A man is working on repair a truck",
    ]


@pytest.fixture
def sample_embedding_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clip = Clip(
        uuid=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        source_video="sample_video.mp4",
        span=(2.5, 6.5),
    )
    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        source_bytes=sample_video_data,
        clips=[clip],
    )
    return SplitPipeTask(
        video=video,
    )


@pytest.mark.env("unified")
def test_generate_embedding(sample_embedding_task: SplitPipeTask) -> None:
    """Test the InternVideo2Embedding result."""
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        ClipFrameExtractionStage(extraction_policies=(FrameExtractionPolicy.sequence,), target_fps=[2.0]),
        InternVideo2FrameCreationStage(target_fps=2.0),
        InternVideo2EmbeddingStage(num_gpus_per_worker=1.0, texts_to_verify=_get_texts()),
    ]
    tasks = run_pipeline([sample_embedding_task], stages)
    result_task = tasks[0]
    clip = result_task.video.clips[0]

    assert clip.buffer is not None, "Expected clip buffer to be not None, but it is None."
    assert clip.intern_video_2_embedding is not None, "Expected InternVideo2 embedding to be not None, but it is None."

    text_match = clip.intern_video_2_text_match

    assert text_match is not None, "Expected InternVideo2 text match to be not None, but it is None."

    logger.info(f"Best text match (score={text_match[1]}): {text_match[0]}")

    assert text_match[0] == _get_texts()[0], f"Expected text match [{_get_texts()[0]}], got [{text_match[0]}]"
    assert text_match[1] > _MATCH_CONFIDENCE_SCORE, (
        f"Expected text match score > {_MATCH_CONFIDENCE_SCORE}, got [{text_match[0]}]"
    )

    logger.info("InternVideo2 embedding test passed.")
