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

from unittest.mock import MagicMock

import pytest
import torch

from cosmos_curate.core.utils.model import conda_utils

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.models.vllm_qwen import VllmQwen, VllmQwen7B, make_message, make_prompt

    _MODEL_VARIANT = VllmQwen7B.model_variant()


@pytest.mark.env("unified")
def test_make_llm_input_qwen() -> None:
    """Test make_llm_input_qwen function."""
    # Create mock processor with tokenizer
    mock_tokenizer = MagicMock()
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (1, 5)
    mock_tokenizer.apply_chat_template.return_value = mock_tensor

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
    assert result["prompt_token_ids"] == [1, 2, 3, 4, 5]  # Should be the token IDs as list
    assert result["multi_modal_data"]["video"].shape == (2, 3, 32, 32)


# Hmmm, might not be needed for qwen
@pytest.mark.env("unified")
def test_make_llm_input_qwen_no_tokenizer() -> None:
    """Test make_llm_input_qwen with processor without tokenizer."""
    mock_processor = MagicMock()
    mock_processor.tokenizer = None

    frames = torch.rand(1, 3, 32, 32)
    prompt = "Test prompt"

    with pytest.raises(ValueError, match=r".*"):
        VllmQwen.make_llm_input(prompt, frames, mock_processor)


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
    # Mock the tokenizer to return a tensor that can be indexed and converted to list
    mock_tensor = torch.tensor([[10, 20, 30, 40]])  # Shape: (1, 4)
    mock_tokenizer.apply_chat_template.return_value = mock_tensor

    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tokenizer

    prompt = "Test prompt"
    frames = torch.rand(2, 3, 32, 32)
    message = make_message(prompt)
    result = make_prompt(message, frames, mock_processor)
    assert result["prompt_token_ids"] == [10, 20, 30, 40]  # Should be the token IDs as list
    assert result["multi_modal_data"]["video"].shape == (2, 3, 32, 32)
