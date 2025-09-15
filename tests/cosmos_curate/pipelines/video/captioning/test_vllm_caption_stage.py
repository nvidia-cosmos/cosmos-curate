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
"""Test vllm_caption_stage.py."""

from pathlib import Path

import pytest

from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import _get_video_from_task
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video


def test_get_video_from_task_success() -> None:
    """Test get_video_from_task."""
    task = SplitPipeTask(video=Video(input_video=Path("test.mp4")))
    video = _get_video_from_task(task)
    assert video.input_video == Path("test.mp4")


@pytest.mark.env("unified")
def test_get_video_from_task_fail() -> None:
    """Test get_video_from_task."""
    task = 10
    with pytest.raises(TypeError, match=".*"):
        _get_video_from_task(task)
