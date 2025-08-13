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
"""Test vllm_phi.py."""

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import torch
from PIL import Image

from cosmos_curate.models.vllm_phi import VLLMPhi4, get_image_placeholder, make_message, make_prompt, tensor_to_pil
from cosmos_curate.pipelines.video.utils.data_model import Clip, Video, Window


@pytest.mark.env("unified")
def test_tensor_to_pil() -> None:
    """Test tensor_to_pil function."""
    # Create a test tensor (3, 32, 32) with values in [0, 1]
    test_tensor = torch.rand(3, 32, 32)

    # Call the function
    pil_image = tensor_to_pil(test_tensor)

    # Verify output is PIL Image
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (32, 32)  # PIL uses (width, height)


@pytest.mark.env("unified")
def test_tensor_to_pil_cuda() -> None:
    """Test tensor_to_pil with CUDA tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create CUDA tensor
    test_tensor = torch.rand(3, 32, 32).cuda()

    # Call the function
    pil_image = tensor_to_pil(test_tensor)

    # Verify output
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (32, 32)


@pytest.mark.env("unified")
def test_tensor_to_pil_invalid_shape() -> None:
    """Test tensor_to_pil with invalid tensor shape."""
    # Create tensor with wrong dimensions
    invalid_tensor = torch.rand(32, 32)  # Missing channel dimension

    with pytest.raises(ValueError, match="Tensor has incorrect shape"):
        tensor_to_pil(invalid_tensor)


@pytest.mark.env("unified")
def test_make_llm_input_phi() -> None:
    """Test make_llm_input_phi function."""
    # Create mock processor with tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked_prompt"

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    # Create test frames tensor
    frames = torch.rand(2, 3, 32, 32)  # 2 frames, 3 channels, 32x32
    prompt = "Describe the video"

    # Call the function
    result = VLLMPhi4.make_llm_input(prompt, frames, mock_processor)

    # Verify structure
    expected_frame_count = 2
    assert "prompt" in result
    assert "multi_modal_data" in result
    assert "image" in result["multi_modal_data"]
    assert result["prompt"] == "mocked_prompt"
    assert len(result["multi_modal_data"]["image"]) == expected_frame_count  # 2 frames


@pytest.mark.env("unified")
def test_make_llm_input_phi_no_tokenizer() -> None:
    """Test make_llm_input_phi with processor without tokenizer."""
    mock_processor = MagicMock()
    mock_processor.tokenizer = None

    frames = torch.rand(1, 3, 32, 32)
    prompt = "Test prompt"

    with pytest.raises(ValueError, match=".*"):
        VLLMPhi4.make_llm_input(prompt, frames, mock_processor)


@pytest.mark.env("unified")
def test_add_llm_input_to_window_phi() -> None:
    """Test add_llm_input_to_window function."""
    start = 0
    end = 10
    mp4_bytes = b"mock_mp4_data"
    llm_input = {"prompt": "test", "data": "mock"}

    window = Window(start_frame=start, end_frame=end, mp4_bytes=mp4_bytes)

    VLLMPhi4.add_llm_input_to_window(window, llm_input)

    # Verify window properties
    assert isinstance(window, Window)
    assert window.start_frame == start
    assert window.end_frame == end
    assert window.mp4_bytes == mp4_bytes
    assert window.phi_llm_input == llm_input


@pytest.mark.env("unified")
def test_get_phi_llm_input_from_window() -> None:
    """Test get_phi_llm_input_from_window function."""
    llm_input = {"prompt": "test", "data": "mock"}
    window = Window(start_frame=0, end_frame=10, phi_llm_input=llm_input)

    # Call the function
    result = VLLMPhi4.get_llm_input_from_window(window)

    # Verify result
    assert result == llm_input


@pytest.mark.env("unified")
def test_free_vllm_inputs_phi() -> None:
    """Test free_vllm_inputs function."""
    # Create windows with data
    window1 = Window(start_frame=0, end_frame=10, mp4_bytes=b"data1", phi_llm_input={"test": "data1"})
    window2 = Window(start_frame=10, end_frame=20, mp4_bytes=b"data2", phi_llm_input={"test": "data2"})

    # Create clips with windows
    clip1 = Clip(uuid=uuid4(), source_video="test1.mp4", span=(0.0, 5.0), windows=[window1])
    clip2 = Clip(uuid=uuid4(), source_video="test2.mp4", span=(5.0, 10.0), windows=[window2])

    # Create video with clips
    video = Video(input_video=Path("test.mp4"), clips=[clip1, clip2])

    # Verify initial state
    assert window1.mp4_bytes is not None
    assert window1.phi_llm_input is not None
    assert window2.mp4_bytes is not None
    assert window2.phi_llm_input is not None

    # Call the function
    VLLMPhi4.free_vllm_inputs(video)

    # Verify memory was freed, but mp4 bytes are not freed, that is handled elsewhere
    assert window1.mp4_bytes is not None
    assert window1.phi_llm_input is None
    assert window2.mp4_bytes is not None
    assert window2.phi_llm_input is None


@pytest.mark.env("unified")
def test_free_vllm_inputs_empty_video() -> None:
    """Test free_unused_phi with empty video."""
    video = Video(input_video=Path("test.mp4"), clips=[])

    # Should not raise any errors
    VLLMPhi4.free_vllm_inputs(video)


@pytest.mark.env("unified")
def test_get_image_placeholder() -> None:
    """Test get_image_placeholder function."""
    assert get_image_placeholder(1) == "<|image_1|>"
    assert get_image_placeholder(2) == "<|image_1|><|image_2|>"
    assert get_image_placeholder(3) == "<|image_1|><|image_2|><|image_3|>"


@pytest.mark.env("unified")
def test_make_message() -> None:
    """Test make_message function."""
    prompt = "Test prompt"
    images = [Image.new("RGB", (32, 32)) for _ in range(2)]
    message = make_message(prompt, images)
    assert "role" in message
    assert message["role"] == "user"
    assert "content" in message
    assert "images" in message
    assert message["images"] == images


@pytest.mark.env("unified")
def test_make_prompt() -> None:
    """Test make_prompt function."""
    # Create mock processor with tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked_prompt"

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    prompt = "Test prompt"
    images = [Image.new("RGB", (32, 32)) for _ in range(2)]
    message = make_message(prompt, images)
    result = make_prompt(message, mock_processor)
    assert result["prompt"] == "mocked_prompt"
    assert result["multi_modal_data"]["image"] == images
