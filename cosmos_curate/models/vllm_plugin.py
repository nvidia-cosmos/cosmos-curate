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
"""vLLM plugin definition.

This interface defines the contract for adding new vLLM models to cosmos-curate.

To add a new model:
1. Create a class inheriting from VllmPlugin
2. Implement all abstract methods below
3. Register in cosmos_curate/models/vllm_interface.py:_VLLM_PLUGINS
4. Add model ID mapping in cosmos_curate/models/vllm_model_ids.py

References:
[VLLM_INTERFACE_PLUGIN.md](../../docs/curator/VLLM_INTERFACE_PLUGIN.md)
[Complete Example](vllm_qwen.py)

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curate.core.utils.model import model_utils
from cosmos_curate.models.vllm_model_ids import get_vllm_model_id
from cosmos_curate.pipelines.video.utils.data_model import (
    VllmCaptionRequest,
    VllmConfig,
)


class VllmPlugin(ABC):
    """vLLM plugin interface."""

    @staticmethod
    @abstractmethod
    def model_variant() -> str:
        """Return the model variant name."""

    @classmethod
    def model_id(cls) -> str:
        """Return the model ID."""
        return get_vllm_model_id(cls.model_variant())

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
    def model(cls, config: VllmConfig) -> LLM:
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
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        """Make a refined LLM request.

        Args:
            request: The request to refine.
            processor: The processor to use for the stage 2 prompt
            refine_prompt: An optional prompt to use to refine the caption. If
                None, the default refineprompt will be used.

        Returns:
            A refined LLM request.

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
