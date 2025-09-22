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
"""Test vllm_cosmos_reason1_vl.py."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import torch

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    Video,
    VllmCaptionRequest,
    Window,
)

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.models.vllm_cosmos_reason1_vl import (
        VllmCosmosReason1VL,
        _extract_from_reasoning_format,
        make_message,
        make_prompt,
    )


@pytest.mark.env("unified")
def test_make_llm_input_cosmos_r1() -> None:
    """Test make_llm_input for Cosmos-Reason1 plugin."""
    # Mock processor with apply_chat_template
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "mocked_reasoning_prompt"

    frames = torch.rand(2, 3, 32, 32)
    prompt = "Describe the video"

    result = VllmCosmosReason1VL.make_llm_input(prompt, frames, mock_processor)

    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert result["prompt"] == "mocked_reasoning_prompt"
    assert len(result["multi_modal_data"]["video"]) == 1
    assert result["multi_modal_data"]["video"][0].shape == (2, 3, 32, 32)


@pytest.mark.env("unified")
def test_add_and_get_llm_input_window_cosmos_r1() -> None:
    """Test adding and retrieving llm input on Window for cosmos_reason1."""
    window = Window(start_frame=0, end_frame=10)
    llm_input = {"prompt": "p", "multi_modal_data": {"video": [torch.rand(1, 3, 8, 8)]}}

    VllmCosmosReason1VL.add_llm_input_to_window(window, llm_input)
    assert window.cosmos_reason1_llm_input == llm_input

    got = VllmCosmosReason1VL.get_llm_input_from_window(window)
    assert got == llm_input


@pytest.mark.env("unified")
def test_free_vllm_inputs_cosmos_r1() -> None:
    """Test freeing vllm inputs for cosmos_reason1 windows."""
    window1 = Window(start_frame=0, end_frame=10, cosmos_reason1_llm_input={"a": 1})
    window2 = Window(start_frame=10, end_frame=20, cosmos_reason1_llm_input={"b": 2})

    clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 5.0), windows=[window1])
    clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(5.0, 10.0), windows=[window2])
    video = Video(input_video=Path("test.mp4"), clips=[clip1, clip2])

    VllmCosmosReason1VL.free_vllm_inputs(video)
    assert window1.cosmos_reason1_llm_input is None
    assert window2.cosmos_reason1_llm_input is None


@pytest.mark.env("unified")
def test_extract_from_reasoning_format() -> None:
    """Test that decode extracts <answer>...</answer> content."""
    text = "<think>some thoughts</think>\n<answer>final caption</answer>"
    assert _extract_from_reasoning_format(text) == "final caption"

    # Fallback if missing tags
    plain = "no tags here"
    assert _extract_from_reasoning_format(plain) == plain


@pytest.mark.env("unified")
def test_make_prompt_uses_chat_template() -> None:
    """Ensure make_prompt uses processor.apply_chat_template and wires video correctly."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "chat-prompt"

    frames = torch.rand(1, 3, 16, 16)
    result = make_prompt(make_message("hello"), frames, mock_processor)
    assert result["prompt"] == "chat-prompt"
    assert result["multi_modal_data"]["video"][0].shape == (1, 3, 16, 16)


@pytest.mark.env("unified")
def test_make_refined_llm_request() -> None:
    """Test refine flow creates a new request preserving video and updating prompt."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "refined-prompt"

    frames = torch.rand(1, 3, 8, 8)
    base_inputs = {"prompt": "base", "multi_modal_data": {"video": [frames]}}

    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs=base_inputs,
        video_idx=0,
        clip_idx=0,
        window_idx=0,
        caption="stage1 caption",
        iterations=0,
    )

    refined = VllmCosmosReason1VL.make_refined_llm_request(base_req, mock_processor, refine_prompt=None)
    assert refined.inputs["prompt"] == "refined-prompt"
    assert refined.inputs["multi_modal_data"]["video"][0].shape == (1, 3, 8, 8)
    assert refined.video_idx == base_req.video_idx
    assert refined.clip_idx == base_req.clip_idx
    assert refined.window_idx == base_req.window_idx


@pytest.mark.env("unified")
def test_stage2_refine_prompt_equivalence_with_real_processor() -> None:
    """Integration test: verify refine prompt equivalence using the real processor.

    Skips if model weights are unavailable or processor lacks apply_chat_template.

    This test is used as part of the migration from cosmos_reason1_vl to vllm_cosmos_reason1_vl.

    It is expected that the prompt generated by the chat template provided by the model's processor
    will be the same as the prompt generated by the regex substitution used previously.

    Over the long term, we expect to migrate away from the regex substitution and use the chat template
    provided by the model's processor, making this test obsolete.
    """
    model_path = Path(str(VllmCosmosReason1VL.model_path()))
    if not model_path.exists():
        pytest.skip("Cosmos-Reason1 weights not available locally; skipping integration test.")

    processor = VllmCosmosReason1VL.processor()
    if not hasattr(processor, "apply_chat_template"):
        pytest.skip("Processor lacks apply_chat_template; skipping integration test.")

    frames = torch.rand(1, 3, 8, 8)

    # Generate initial prompt via real processor
    initial_inputs = VllmCosmosReason1VL.make_llm_input("initial user text", frames, processor)
    initial_prompt = initial_inputs["prompt"]

    caption = "stage1 caption"
    refine_prompt = "REFINE:\n"

    pattern = (
        r"(<\|im_start\|>system\s*.*?<\|im_end\|>\s*"
        r"<\|im_start\|>user\s*<\|vision_start\|><\|video_pad\|><\|vision_end\|>\s*)(.*?)(\s*<\|im_end\|>)"
    )
    expected = re.sub(pattern, rf"\1{refine_prompt + caption}\3", initial_prompt, flags=re.DOTALL)

    base_req = VllmCaptionRequest(
        request_id="r1",
        inputs=initial_inputs,
        video_idx=0,
        clip_idx=0,
        window_idx=0,
        caption=caption,
        iterations=0,
    )
    refined = VllmCosmosReason1VL.make_refined_llm_request(base_req, processor, refine_prompt=refine_prompt)

    assert refined.inputs["prompt"] == expected
