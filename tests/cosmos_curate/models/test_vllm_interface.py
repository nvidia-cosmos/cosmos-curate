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
"""Test vllm_interface.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    Video,
    VllmCaptionRequest,
    VllmConfig,
    Window,
)

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.models.vllm_interface import (
        _VLLM_PLUGINS,
        gather_vllm_requests,
        scatter_vllm_captions,
    )

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())


@pytest.mark.env("unified")
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_gather_vllm_requests(mock_get_plugin: MagicMock) -> None:
    """Test gather_vllm_requests function."""
    # Mock the plugin with get_llm_input_from_window method
    mock_plugin = MagicMock()
    mock_plugin.get_llm_input_from_window.return_value = {"test": "data"}
    mock_get_plugin.return_value = mock_plugin

    # Create test windows
    window1 = Window(start_frame=0, end_frame=10)  # Will be extracted (even start_frame)
    window2 = Window(start_frame=1, end_frame=11)  # Will not be extracted (odd start_frame)
    window3 = Window(start_frame=2, end_frame=12)  # Will be extracted (even start_frame)
    window4 = Window(start_frame=3, end_frame=13)  # Will not be extracted (odd start_frame)

    # Create test clips
    clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 5.0), windows=[window1, window2])
    clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(5.0, 10.0), windows=[window3, window4])

    # Create test videos
    video1 = Video(input_video=Path("test1.mp4"), clips=[clip1])
    video2 = Video(input_video=Path("test2.mp4"), clips=[clip2])
    videos = [video1, video2]

    cfg = VllmConfig(variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests(videos, cfg))

    assert len(vllm_requests) == 4  # noqa: PLR2004

    # Verify results
    expected_mappings = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    for req, expected_mapping in zip(vllm_requests, expected_mappings, strict=True):
        assert req.video_idx == expected_mapping[0]
        assert req.clip_idx == expected_mapping[1]
        assert req.window_idx == expected_mapping[2]


@pytest.mark.env("unified")
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_gather_vllm_requests_empty(mock_get_plugin: MagicMock) -> None:
    """Test gather_vllm_requests with empty videos list."""
    # Mock the plugin with get_llm_input_from_window method
    mock_plugin = MagicMock()
    mock_plugin.get_llm_input_from_window.return_value = {"test": "data"}
    mock_get_plugin.return_value = mock_plugin

    cfg = VllmConfig(variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests([], cfg))

    assert len(vllm_requests) == 0


@pytest.mark.env("unified")
def test_gather_vllm_requests_empty_clip() -> None:
    """Test gather_vllm_requests with empty clip."""
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), windows=[])
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    videos = [video]

    cfg = VllmConfig(variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests(videos, cfg))
    assert len(vllm_requests) == 0
    assert clip.errors["clip_windowing"]


@pytest.mark.env("unified")
def test_gather_vllm_requests_empty_llm_inputs() -> None:
    """Test gather_vllm_requests with empty llm_inputs in window."""
    window = Window(start_frame=0, end_frame=10)
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), windows=[window])
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    videos = [video]

    assert not clip.errors

    cfg = VllmConfig(variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests(videos, cfg))

    assert vllm_requests == []
    assert clip.errors


@pytest.mark.env("unified")
def test_scatter_vllm_captions() -> None:
    """Test scatter_vllm_captions function."""
    # Create test windows
    window1 = Window(start_frame=0, end_frame=10)
    window2 = Window(start_frame=10, end_frame=20)
    window3 = Window(start_frame=0, end_frame=15)

    # Create test clips
    clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 5.0), windows=[window1, window2])
    clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(5.0, 10.0), windows=[window3])

    # Create test videos
    video1 = Video(input_video=Path("test1.mp4"), clips=[clip1])
    video2 = Video(input_video=Path("test2.mp4"), clips=[clip2])
    videos = [video1, video2]

    # Build requests corresponding to mapping
    model_variant = "test_model"
    requests = [
        VllmCaptionRequest(
            request_id="r1",
            inputs={},
            video_idx=0,
            clip_idx=0,
            window_idx=0,
            caption="First caption",
            finished=True,
        ),
        VllmCaptionRequest(
            request_id="r2",
            inputs={},
            video_idx=0,
            clip_idx=0,
            window_idx=1,
            caption="Second caption",
            finished=True,
        ),
        VllmCaptionRequest(
            request_id="r3",
            inputs={},
            video_idx=1,
            clip_idx=0,
            window_idx=0,
            caption="Third caption",
            finished=True,
        ),
    ]

    # Call the function
    scatter_vllm_captions(model_variant, videos, requests)

    # Verify captions were assigned correctly
    assert videos[0].clips[0].windows[0].caption[model_variant] == "First caption"
    assert videos[0].clips[0].windows[1].caption[model_variant] == "Second caption"
    assert videos[1].clips[0].windows[0].caption[model_variant] == "Third caption"


@pytest.mark.env("unified")
def test_scatter_vllm_captions_empty() -> None:
    """Test assign_captions with empty mapping and captions."""
    # Create a simple video structure
    window = Window(start_frame=0, end_frame=10)
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), windows=[window])
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    videos = [video]

    # Empty requests
    model_variant = "test_model"
    requests: list[VllmCaptionRequest] = []

    # Call the function - should not raise any errors
    scatter_vllm_captions(model_variant, videos, requests)

    # Verify no captions were added
    assert model_variant not in videos[0].clips[0].windows[0].caption


@pytest.mark.env("unified")
def test_scatter_vllm_captions_multiple_models() -> None:
    """Test assign_captions with multiple model variants."""
    # Create test structure
    window = Window(start_frame=0, end_frame=10)
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), windows=[window])
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    videos = [video]

    # Assign captions from different models
    scatter_vllm_captions(
        "model1",
        videos,
        [
            VllmCaptionRequest(
                request_id="r1",
                inputs={},
                video_idx=0,
                clip_idx=0,
                window_idx=0,
                caption="Caption from model 1",
                finished=True,
            )
        ],
    )
    scatter_vllm_captions(
        "model2",
        videos,
        [
            VllmCaptionRequest(
                request_id="r2",
                inputs={},
                video_idx=0,
                clip_idx=0,
                window_idx=0,
                caption="Caption from model 2",
                finished=True,
            )
        ],
    )

    # Verify both captions exist
    assert videos[0].clips[0].windows[0].caption["model1"] == "Caption from model 1"
    assert videos[0].clips[0].windows[0].caption["model2"] == "Caption from model 2"


@pytest.mark.env("unified")
def test_scatter_vllm_captions_overwrite() -> None:
    """Test assign_captions overwrites existing captions for same model."""
    # Create test structure
    window = Window(start_frame=0, end_frame=10)
    clip = Clip(uuid=uuid4(), source_video="test.mp4", span=(0.0, 5.0), windows=[window])
    video = Video(input_video=Path("test.mp4"), clips=[clip])
    videos = [video]

    model_variant = "test_model"

    # Assign initial caption
    scatter_vllm_captions(
        model_variant,
        videos,
        [
            VllmCaptionRequest(
                request_id="r1",
                inputs={},
                video_idx=0,
                clip_idx=0,
                window_idx=0,
                caption="Initial caption",
                finished=True,
            )
        ],
    )
    assert videos[0].clips[0].windows[0].caption[model_variant] == "Initial caption"

    # Assign new caption - should overwrite
    scatter_vllm_captions(
        model_variant,
        videos,
        [
            VllmCaptionRequest(
                request_id="r2",
                inputs={},
                video_idx=0,
                clip_idx=0,
                window_idx=0,
                caption="Updated caption",
                finished=True,
            )
        ],
    )
    assert videos[0].clips[0].windows[0].caption[model_variant] == "Updated caption"
