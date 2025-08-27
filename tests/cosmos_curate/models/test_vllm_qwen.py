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
"""Test vllm_qwen.py."""

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import torch

from cosmos_curate.models.vllm_qwen import VllmQwen, make_message, make_prompt
from cosmos_curate.pipelines.video.utils.data_model import Clip, Video, Window


@pytest.mark.env("unified")
def test_make_llm_input_qwen() -> None:
    """Test make_llm_input_qwen function."""
    # Create mock processor with tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked_prompt"

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    # Create test frames tensor
    frames = torch.rand(2, 3, 32, 32)  # 2 frames, 3 channels, 32x32
    prompt = "Describe the video"

    # Call the function
    result = VllmQwen.make_llm_input(prompt, frames, mock_processor)

    # Verify structure
    assert "multi_modal_data" in result
    assert "video" in result["multi_modal_data"]
    assert result["prompt"] == "mocked_prompt"
    assert len(result["multi_modal_data"]["video"]) == 1
    assert result["multi_modal_data"]["video"][0].shape == (2, 3, 32, 32)


# Hmmm, might not be needed for qwen
@pytest.mark.env("unified")
def test_make_llm_input_qwen_no_tokenizer() -> None:
    """Test make_llm_input_qwen with processor without tokenizer."""
    mock_processor = MagicMock()
    mock_processor.tokenizer = None

    frames = torch.rand(1, 3, 32, 32)
    prompt = "Test prompt"

    with pytest.raises(ValueError, match=".*"):
        VllmQwen.make_llm_input(prompt, frames, mock_processor)


@pytest.mark.env("unified")
def test_add_llm_input_to_window_qwen() -> None:
    """Test add_llm_input_to_window function."""
    start = 0
    end = 10
    mp4_bytes = b"mock_mp4_data"
    llm_input = {"prompt": "test", "data": "mock"}

    window = Window(start_frame=start, end_frame=end, mp4_bytes=mp4_bytes)

    VllmQwen.add_llm_input_to_window(window, llm_input)

    # Verify window properties
    assert isinstance(window, Window)
    assert window.start_frame == start
    assert window.end_frame == end
    assert window.mp4_bytes == mp4_bytes
    assert window.qwen_llm_input == llm_input


@pytest.mark.env("unified")
def test_get_qwen_llm_input_from_window() -> None:
    """Test get_qwen_llm_input_from_window function."""
    llm_input = {"prompt": "test", "data": "mock"}
    window = Window(start_frame=0, end_frame=10, qwen_llm_input=llm_input)

    # Call the function
    result = VllmQwen.get_llm_input_from_window(window)

    # Verify result
    assert result == llm_input


@pytest.mark.env("unified")
def test_free_vllm_inputs_qwen() -> None:
    """Test free_vllm_inputs function."""
    # Create windows with data
    window1 = Window(start_frame=0, end_frame=10, mp4_bytes=b"data1", qwen_llm_input={"test": "data1"})
    window2 = Window(start_frame=10, end_frame=20, mp4_bytes=b"data2", qwen_llm_input={"test": "data2"})

    # Create clips with windows
    clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 5.0), windows=[window1])
    clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(5.0, 10.0), windows=[window2])

    # Create video with clips
    video = Video(input_video=Path("test.mp4"), clips=[clip1, clip2])

    # Verify initial state
    assert window1.mp4_bytes is not None
    assert window1.qwen_llm_input is not None
    assert window2.mp4_bytes is not None
    assert window2.qwen_llm_input is not None

    # Call the function
    VllmQwen.free_vllm_inputs(video)

    # Verify memory was freed, but mp4 bytes are not freed, that is handled elsewhere
    assert window1.mp4_bytes is not None
    assert window1.qwen_llm_input is None
    assert window2.mp4_bytes is not None
    assert window2.qwen_llm_input is None


@pytest.mark.env("unified")
def test_free_vllm_inputs_empty_video() -> None:
    """Test free_unused_phi with empty video."""
    video = Video(input_video=Path("test.mp4"), clips=[])

    # Should not raise any errors
    VllmQwen.free_vllm_inputs(video)


@pytest.mark.env("unified")
def test_make_message() -> None:
    """Test make_message function."""
    prompt = "Test prompt"
    message = make_message(prompt)
    assert "role" in message
    assert message["role"] == "user"
    assert "content" in message
    assert isinstance(message["content"], list)
    content = message["content"]
    assert len(content) == 2  # noqa: PLR2004


@pytest.mark.env("unified")
def test_make_prompt() -> None:
    """Test make_prompt function."""
    # Create mock processor with tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked_prompt"

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    prompt = "Test prompt"
    frames = torch.rand(2, 3, 32, 32)
    message = make_message(prompt)
    result = make_prompt(message, frames, mock_processor)
    assert result["prompt"] == "mocked_prompt"
    assert result["multi_modal_data"]["video"] == [frames]
