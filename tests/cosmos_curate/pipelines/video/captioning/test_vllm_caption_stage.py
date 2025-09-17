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

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.vllm_model_ids import _VLLM_MODELS
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video, VllmConfig

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import VllmModelInterface, _get_video_from_task


@pytest.mark.env("unified")
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


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("config_variant", "raises"),
    [(k, nullcontext()) for k in _VLLM_MODELS] + [("_fail_model", pytest.raises(ValueError, match=".*"))],
)
def test_vllm_model_interface_model_id_names(config_variant: str, raises: AbstractContextManager[Any]) -> None:
    """Validate model_id_names are strings for each configured plugin variant."""
    vllm_config = VllmConfig(variant=config_variant)
    vllm_model_interface = VllmModelInterface(vllm_config)

    with raises:
        model_id_names = vllm_model_interface.model_id_names
        for model_id_name in model_id_names:
            assert isinstance(model_id_name, str)
