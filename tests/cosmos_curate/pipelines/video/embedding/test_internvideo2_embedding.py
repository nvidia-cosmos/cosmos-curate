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
        "A black car is driving on the highway and passing a red car",
    ]


@pytest.fixture
def sample_embedding_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clips = [
        Clip(
            uuid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            source_video="sample_video.mp4",
            span=(2.5, 6.5),
        ),
        Clip(
            uuid=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            source_video="sample_video.mp4",
            span=(23, 26),
        ),
    ]
    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        encoded_data=sample_video_data,
        clips=clips,
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

    assert len(result_task.video.clips) == len(sample_embedding_task.video.clips), (
        f"Expected {len(sample_embedding_task.video.clips)} clips, got {len(result_task.video.clips)}"
    )

    clip = result_task.video.clips[0]

    for clip, matching_text_idx in zip(result_task.video.clips, [0, 4], strict=True):
        assert clip.encoded_data is not None, "Expected clip.encoded_data to be not None, but it is None."
        assert clip.intern_video_2_embedding is not None, (
            "Expected InternVideo2 embedding to be not None, but it is None."
        )

        text_match = clip.intern_video_2_text_match

        assert text_match is not None, "Expected InternVideo2 text match to be not None, but it is None."

        logger.info(f"Best text match (score={text_match[1]}): {text_match[0]}")

        matching_text = _get_texts()[matching_text_idx]
        assert text_match[0] == matching_text, f"Expected text match [{matching_text}], got [{text_match[0]}]"
        assert text_match[1] > _MATCH_CONFIDENCE_SCORE, (
            f"Expected text match score > {_MATCH_CONFIDENCE_SCORE}, got [{text_match[0]}]"
        )

    logger.info("InternVideo2 embedding test passed.")
