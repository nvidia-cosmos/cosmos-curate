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

"""Enhance caption tests for ChatLM variants."""

import pathlib
import uuid

import pytest

from cosmos_curate.pipelines.video.captioning.captioning_stages import (  # type: ignore[import-untyped]
    EnhanceCaptionStage,
)
from cosmos_curate.pipelines.video.utils.data_model import (  # type: ignore[import-untyped]
    Clip,
    SplitPipeTask,
    Video,
    Window,
)
from tests.utils.sequential_runner import run_pipeline


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", ["qwen_lm", "gpt_oss_20b"])
def test_enhance_caption_lm_variants(model_variant: str) -> None:
    """EnhanceCaptionStage with real LM and pre-filled captions (no prior stages)."""
    base_captions = [
        "A red pickup truck is parked on a cobblestone street.",
        "Interior car shot with a driver speaking into a microphone.",
    ]
    clips: list[Clip] = []
    base_caption_key = "caption_model_variant"
    for i, text in enumerate(base_captions):
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"sample_video.mp4#fake-{i}"),
            source_video="sample_video.mp4",
            span=(0.0, 5.0),
        )
        clip.windows.append(
            Window(
                start_frame=0,
                end_frame=256,
            )
        )
        clip.windows[0].caption[base_caption_key] = text
        clips.append(clip)

    video = Video(input_video=pathlib.Path("sample_video.mp4"), clips=clips)
    task = SplitPipeTask(video=video)

    stages = [EnhanceCaptionStage(model_variant=model_variant)]
    result_tasks = run_pipeline([task], stages)

    assert result_tasks is not None
    assert len(result_tasks) == 1

    # Verify enhanced captions were written back and are longer than base
    for i, c in enumerate(result_tasks[0].video.clips):
        assert model_variant in c.windows[0].enhanced_caption
        enhanced = c.windows[0].enhanced_caption[model_variant]
        assert isinstance(enhanced, str)
        assert len(enhanced) > len(base_captions[i])
