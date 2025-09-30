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

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from cosmos_curate.core.utils.model import conda_utils

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.models.vllm_phi import VllmPhi4, get_image_placeholder, make_message, make_prompt, tensor_to_pil

    _MODEL_VARIANT = VllmPhi4.model_variant()


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
    result = VllmPhi4.make_llm_input(prompt, frames, mock_processor)

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

    with pytest.raises(ValueError, match=r".*"):
        VllmPhi4.make_llm_input(prompt, frames, mock_processor)


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
