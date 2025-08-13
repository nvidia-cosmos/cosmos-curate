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
"""vLLM plugin definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curate.core.utils.model import model_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Video,
    VLLMConfig,
    Window,
)


class VLLMPlugin(ABC):
    """vLLM plugin interface."""

    @staticmethod
    @abstractmethod
    def model_variant() -> str:
        """Return the model variant name."""

    @staticmethod
    @abstractmethod
    def model_id() -> str:
        """Return the model ID."""

    @classmethod
    def model_path(cls) -> Path:
        """Return the path to the model."""
        return model_utils.get_local_dir_for_weights_name(cls.model_id())

    @classmethod
    @abstractmethod
    def processor(cls) -> AutoProcessor:
        """Return the AutoProcessor for the model."""

    @classmethod
    @abstractmethod
    def model(cls, config: VLLMConfig) -> LLM:
        """Instantiate the vLLM model.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """

    @staticmethod
    @abstractmethod
    def make_llm_input(prompt: str, frames: torch.Tensor, processor: AutoProcessor) -> dict[str, Any]:
        """Make a single LLM input for a vLLM model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM.
            processor: The AutoProcessor to use for the LLM.

        Returns:
            A dictionary containing the LLM inputs.

        """

    @staticmethod
    @abstractmethod
    def make_refined_llm_input(
        caption: str, prev_input: dict[str, Any], processor: AutoProcessor, refine_prompt: str | None = None
    ) -> dict[str, Any]:
        """Make refined LLM input.

        Take a generated caption and the prompt (prev_input) used to
        generate that caption and create an refinement prompt.

        Args:
            caption: The caption to refine
            prev_input: The prompt that was used to generate the caption
            processor: The processor to use for the stage 2 prompt
            refine_prompt: An optional prompt to use to refine the caption. If
                None, the default refineprompt will be used.

        Returns:
            A prompt used to refine an existing caption.

        """

    @staticmethod
    @abstractmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode one vllm output into a caption string.

        Args:
            vllm_output: The output from vllm_generate

        Returns:
            A caption string.

        """

    @staticmethod
    @abstractmethod
    def add_llm_input_to_window(window: Window, llm_input: dict[str, Any]) -> None:
        """Add LLM input to a Window.

        Args:
            window: The window.
            llm_input: The LLM input for the window.

        """

    @staticmethod
    @abstractmethod
    def get_llm_input_from_window(window: Window) -> dict[str, Any]:
        """Get the LLM input for a window.

        Args:
            window: The window.

        Returns:
            The LLM input for the window.

        Raises:
            ValueError: If the LLM input is None.

        """

    @staticmethod
    @abstractmethod
    def free_vllm_inputs(video: Video) -> None:
        """Free unused memory for this model variant.

        Args:
            video: The video to free unused memory for.

        """
