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
"""Phi-4 vLLM plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torchvision import transforms  # type: ignore[import-untyped]
from transformers import AutoProcessor
from vllm import LLM

from cosmos_curate.models.vllm_plugin import VLLMPlugin

if TYPE_CHECKING:
    from PIL import Image
    from vllm import RequestOutput
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from cosmos_curate.pipelines.video.utils.data_model import (
        Video,
        VLLMConfig,
        Window,
    )


GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
TRUST_REMOTE_CODE = True
LIMIT_MM_PER_PROMPT = {"image": 100}


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW torch tensor to PIL image."""
    tensor_to_pil_transform = transforms.ToPILImage()

    # Ensure tensor is on CPU and has correct shape
    if tensor.is_cuda:
        tensor = tensor.cpu()

    EXPECTED_TENSOR_DIM = 3  # Channel x Height x Width, e.g. torch.Size([3, 560, 1008])
    if tensor.dim() != EXPECTED_TENSOR_DIM:
        msg = f"Tensor has incorrect shape: {tensor.shape} (expected dim of {EXPECTED_TENSOR_DIM})"
        raise ValueError(msg)

    tensor = torch.clamp(tensor, 0, 255) if tensor.dtype == torch.uint8 else torch.clamp(tensor, 0, 1)
    pil_image = tensor_to_pil_transform(tensor)
    return cast("Image.Image", pil_image)


class VLLMPhi4(VLLMPlugin):
    """Phi-4 VLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "phi4"

    @staticmethod
    def model_id() -> str:
        """Return the model ID."""
        return "microsoft/Phi-4-multimodal-instruct"

    @classmethod
    def model(cls, config: VLLMConfig) -> LLM:
        """Instantiate the vLLM model.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        quantization: QuantizationMethods | None = None
        if config.fp8:
            quantization = "fp8"

        return LLM(
            model=str(cls.model_path()),
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            quantization=quantization,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            disable_mm_preprocessor_cache=config.disable_mmcache,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    @classmethod
    def processor(cls) -> AutoProcessor:
        """Return the AutoProcessor for the model."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cls.model_path(),
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        return cast("AutoProcessor", processor)

    @staticmethod
    def make_llm_input(prompt: str, frames: torch.Tensor, processor: AutoProcessor) -> dict[str, Any]:
        """Make LLM inputs for the Phi model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM.
            processor: The AutoProcessor to use for the LLM.

        Returns:
            A dictionary containing the LLM inputs.

        """
        placeholder = ""
        for i in range(frames.shape[0]):
            placeholder += f"<|image_{i + 1}|>"

        pil_images = [tensor_to_pil(frames[i]) for i in range(frames.shape[0])]

        message = {
            "role": "user",
            "content": placeholder + prompt,
            "images": pil_images,
        }

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            msg = "Tokenizer is not set"
            raise ValueError(msg)

        token_prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        return {
            "prompt": token_prompt,
            "multi_modal_data": {
                "image": pil_images,
            },
        }

    @staticmethod
    def add_llm_input_to_window(window: Window, llm_input: dict[str, Any]) -> None:
        """Add LLM input to a Phi window.

        Args:
            window: The window.
            llm_input: The LLM input for the window.

        """
        window.phi_llm_input = llm_input

    @staticmethod
    def get_llm_input_from_window(window: Window) -> dict[str, Any]:
        """Get the LLM input for a Phi window.

        Args:
            window: The window.

        Returns:
            The LLM input for the window.

        Raises:
            ValueError: If the Phi LLM input is None.

        """
        if window.phi_llm_input is None:
            msg = "Phi LLM input is None"
            raise ValueError(msg)
        return window.phi_llm_input

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vllm output into a caption."""
        return vllm_output.outputs[0].text

    @staticmethod
    def free_vllm_inputs(video: Video) -> None:
        """Free vllm inputs from the video for this model.

        Args:
            video: The video.

        """
        for clip in video.clips:
            for window in clip.windows:
                window.phi_llm_input = None
