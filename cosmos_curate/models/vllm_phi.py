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

import secrets
from typing import TYPE_CHECKING, Any, cast

import torch
from torchvision import transforms  # type: ignore[import-untyped]
from transformers import AutoProcessor
from vllm import LLM

from cosmos_curate.models.vllm_plugin import VllmPlugin
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest

if TYPE_CHECKING:
    from PIL import Image
    from vllm import RequestOutput
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig


GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
TRUST_REMOTE_CODE = True
LIMIT_MM_PER_PROMPT = {"image": 100}
_DEFAULT_REFINE_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.\n
"""


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


def get_image_placeholder(num_frames: int) -> str:
    """Get the image placeholder for the Phi model.

    Args:
        num_frames: The number of frames to get the placeholder for.

    Returns:
        A string containing the image placeholder.

    """
    return "".join(f"<|image_{i + 1}|>" for i in range(num_frames))


def make_message(prompt: str, images: list[Image.Image]) -> dict[str, Any]:
    """Make a message for the Phi model.

    Args:
        prompt: The prompt to use for the message.
        images: The images to use for the message.

    Returns:
        A message for the Phi model.

    """
    placeholder = get_image_placeholder(len(images))
    return {
        "role": "user",
        "content": placeholder + prompt,
        "images": images,
    }


def make_prompt(message: dict[str, Any], processor: AutoProcessor) -> dict[str, Any]:
    """Make a prompt for the Phi model.

    Args:
        message: The message to use to create the prompt
        processor: The processor to use for the prompt.

    Returns:
        A prompt for the Phi model.

    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        msg = "Tokenizer is not set"
        raise ValueError(msg)

    if "images" not in message:
        msg = "Message does not contain images"
        raise ValueError(msg)

    return {
        "prompt": tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True),
        "multi_modal_data": {"image": message["images"]},
    }


class VllmPhi4(VllmPlugin):
    """Phi-4 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "phi4"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
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
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],  # noqa: ARG004
        processor: AutoProcessor,
    ) -> dict[str, Any]:
        """Make LLM inputs for the model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM.
            metadata: The metadata to use for the LLM.
            processor: The AutoProcessor to use for the LLM.

        Returns:
            A dictionary containing the LLM inputs.

        """
        pil_images = [tensor_to_pil(frames[i]) for i in range(frames.shape[0])]
        message = make_message(prompt, pil_images)
        return make_prompt(message, processor)

    @staticmethod
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        """Get a prompt to refine an existing caption.

        Args:
            request: The request to refine.
            processor: The processor to use for the stage 2 prompt
            refine_prompt: An optional prompt to use to refine the caption. If
                None, the default refine prompt will be used.

        Returns:
            A refined prompt

        """
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt

        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)

        final_prompt = _refine_prompt + request.caption

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        if "image" not in request.inputs["multi_modal_data"]:
            msg = "Message does not contain image"
            raise ValueError(msg)

        message = make_message(final_prompt, request.inputs["multi_modal_data"]["image"])
        inputs = make_prompt(message, processor)

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vllm output into a caption."""
        return vllm_output.outputs[0].text
