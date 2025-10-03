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

from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    VllmCaptionRequest,
    VllmConfig,
    Window,
)

if conda_utils.is_running_in_env("unified"):
    import torch

    from cosmos_curate.models.vllm_interface import (
        _VLLM_PLUGINS,
        free_vllm_inputs,
        gather_vllm_requests,
        prep_windows_for_vllm,
        scatter_vllm_captions,
    )

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:
    VALID_VARIANTS = []


# Test UUIDs for deterministic testing
UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
UUID_2 = UUID("00000000-0000-0000-0000-000000000002")


@pytest.mark.env("unified")
def test_gather_vllm_requests() -> None:
    """Test gather_vllm_requests function."""
    model_variant = VALID_VARIANTS[0]

    # Create test windows
    window1 = Window(start_frame=0, end_frame=10, model_input={model_variant: {"test": "data"}})
    window2 = Window(start_frame=1, end_frame=11, model_input={model_variant: {"test": "data"}})
    window3 = Window(start_frame=2, end_frame=12, model_input={model_variant: {"test": "data"}})
    window4 = Window(start_frame=3, end_frame=13, model_input={model_variant: {"test": "data"}})

    # Create test clips
    clip1 = Clip(uuid=UUID_1, source_video="test1.mp4", span=(0.0, 5.0), windows=[window1, window2])
    clip2 = Clip(uuid=UUID_2, source_video="test2.mp4", span=(5.0, 10.0), windows=[window3, window4])

    clip_uuids = []
    for clip in [clip1, clip2]:
        clip_uuids += [str(clip.uuid)] * len(clip.windows)

    windows = [window for clip in [clip1, clip2] for window in clip.windows]

    cfg = VllmConfig(model_variant=model_variant)
    vllm_requests = list(gather_vllm_requests(windows, clip_uuids, cfg))

    assert len(vllm_requests) == len(windows)

    # Verify results
    expected_mappings = list(range(len(windows)))
    for req, expected_mapping in zip(vllm_requests, expected_mappings, strict=True):
        assert isinstance(req, VllmCaptionRequest)
        assert req.window_idx == expected_mapping
        assert req.inputs == windows[expected_mapping].model_input[model_variant]


@pytest.mark.env("unified")
def test_gather_vllm_requests_empty() -> None:
    """Test gather_vllm_requests with empty videos list."""
    cfg = VllmConfig(model_variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests([], [], cfg))

    assert len(vllm_requests) == 0


@pytest.mark.env("unified")
def test_gather_vllm_requests_empty_model_input() -> None:
    """Test gather_vllm_requests with empty model_input."""
    window = Window(start_frame=0, end_frame=10, model_input={})

    cfg = VllmConfig(model_variant=VALID_VARIANTS[0])
    vllm_requests = list(gather_vllm_requests([window], [str(UUID_1)], cfg))
    assert len(vllm_requests) == 0
    assert window.errors["model_input"]


@pytest.mark.env("unified")
def test_scatter_vllm_captions() -> None:
    """Test scatter_vllm_captions function."""
    # Create test windows
    window1 = Window(start_frame=0, end_frame=10)
    window2 = Window(start_frame=10, end_frame=20)
    window3 = Window(start_frame=0, end_frame=15)

    # Create test clips
    clip1 = Clip(uuid=UUID_1, source_video="test1.mp4", span=(0.0, 5.0), windows=[window1, window2])
    clip2 = Clip(uuid=UUID_2, source_video="test2.mp4", span=(5.0, 10.0), windows=[window3])

    clip_uuids = []
    for clip in [clip1, clip2]:
        clip_uuids += [str(clip.uuid)] * len(clip.windows)

    windows = [window for clip in [clip1, clip2] for window in clip.windows]

    # Build requests corresponding to mapping
    model_variant = "test_model"
    requests = [
        VllmCaptionRequest(
            request_id="r1",
            inputs={},
            window_idx=0,
            caption="First caption",
            finished=True,
        ),
        VllmCaptionRequest(
            request_id="r2",
            inputs={},
            window_idx=1,
            caption="Second caption",
            finished=True,
        ),
        VllmCaptionRequest(
            request_id="r3",
            inputs={},
            window_idx=2,
            caption="Third caption",
            finished=True,
        ),
    ]

    # Call the function
    scatter_vllm_captions(model_variant, windows, clip_uuids, requests)

    # Verify captions were assigned correctly
    assert windows[0].caption[model_variant] == "First caption"
    assert windows[1].caption[model_variant] == "Second caption"
    assert windows[2].caption[model_variant] == "Third caption"


@pytest.mark.env("unified")
def test_scatter_vllm_captions_empty() -> None:
    """Test assign_captions with empty mapping and captions."""
    # Create a simple video structure
    window = Window(start_frame=0, end_frame=10)

    # Empty requests
    model_variant = "test_model"
    requests: list[VllmCaptionRequest] = []

    # Call the function - should not raise any errors
    scatter_vllm_captions(model_variant, [window], [str(UUID_1)], requests)

    # Verify no captions were added
    assert model_variant not in window.caption


@pytest.mark.env("unified")
def test_scatter_vllm_captions_multiple_models() -> None:
    """Test assign_captions with multiple model variants."""
    # Create test structure
    window = Window(start_frame=0, end_frame=10)
    windows = [window]
    clip_uuids = [str(UUID_1)]

    # Assign captions from different models
    scatter_vllm_captions(
        "model1",
        windows,
        clip_uuids,
        [
            VllmCaptionRequest(
                request_id="r1",
                inputs={},
                window_idx=0,
                caption="Caption from model 1",
                finished=True,
            )
        ],
    )
    scatter_vllm_captions(
        "model2",
        windows,
        clip_uuids,
        [
            VllmCaptionRequest(
                request_id="r2",
                inputs={},
                window_idx=0,
                caption="Caption from model 2",
                finished=True,
            )
        ],
    )

    # Verify both captions exist
    assert window.caption["model1"] == "Caption from model 1"
    assert window.caption["model2"] == "Caption from model 2"


@pytest.mark.env("unified")
def test_scatter_vllm_captions_overwrite() -> None:
    """Test assign_captions overwrites existing captions for same model."""
    # Create test structure
    window = Window(start_frame=0, end_frame=10)
    windows = [window]
    clip_uuids = [str(UUID_1)]

    model_variant = "test_model"

    # Assign initial caption
    scatter_vllm_captions(
        model_variant,
        windows,
        clip_uuids,
        [
            VllmCaptionRequest(
                request_id="r1",
                inputs={},
                window_idx=0,
                caption="Initial caption",
                finished=True,
            )
        ],
    )
    assert window.caption[model_variant] == "Initial caption"

    # Assign new caption - should overwrite
    scatter_vllm_captions(
        model_variant,
        windows,
        clip_uuids,
        [
            VllmCaptionRequest(
                request_id="r2",
                inputs={},
                window_idx=0,
                caption="Updated caption",
                finished=True,
            )
        ],
    )
    assert window.caption[model_variant] == "Updated caption"


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("windows_count", "frames_count", "raises"),
    [
        (2, 2, nullcontext()),
        (1, 2, pytest.raises(ValueError, match=r".*")),
        (3, 1, pytest.raises(ValueError, match=r".*")),
    ],
)
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_prep_windows_for_vllm_and_free_vllm_inputs(
    mock_get_plugin: MagicMock,
    model_variant: str,
    windows_count: int,
    frames_count: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test prep_windows_for_vllm function and free_vllm_inputs functions."""
    processor = MagicMock()
    mock_plugin = MagicMock()
    mock_plugin.processor.return_value = MagicMock()
    mock_plugin.model_variant.return_value = model_variant
    mock_plugin.make_llm_input.return_value = {"test": "data"}

    mock_get_plugin.return_value = mock_plugin

    config = VllmConfig(model_variant=model_variant)
    prompt = "test prompt"

    # Create test data
    windows = [Window(start_frame=0, end_frame=frames_count) for _ in range(windows_count)]
    frames = [torch.randn(frames_count, 3, 224, 224) for _ in range(frames_count)]

    with raises:
        prep_windows_for_vllm(windows, frames, config, processor, prompt)

        for window in windows:
            llm_input = window.model_input.get(model_variant)
            assert isinstance(llm_input, dict)

            free_vllm_inputs(window, config.model_variant)
            assert model_variant not in window.model_input
