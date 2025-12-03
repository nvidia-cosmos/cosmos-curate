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
"""vLLM plugin for Nemotron Nano 12B v2 model."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curate.models.vllm_plugin import VllmPlugin
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest

if TYPE_CHECKING:
    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig


# Constants tuned similarly to existing plugins
GPU_MEMORY_UTILIZATION = 0.9
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 32768
TRUST_REMOTE_CODE = True
LIMIT_MM_PER_PROMPT = {"video": 1}

_DEFAULT_REFINE_PROMPT = (
    """
    Improve and refine following video description.
    Focus on highlighting the key visual and sensory elements.
    Ensure the description is clear, precise, and paints a compelling
    picture of the scene.
    """.strip()
    + "\n"
)

# Constants for tensor dimensions and channel counts
EXPECTED_TENSOR_DIMENSIONS = 4
EXPECTED_NUMPY_DIMENSIONS = 4
EXPECTED_CHANNELS = 3


def _validate_numpy_array(array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Validate and normalize numpy array format."""
    if array.ndim != EXPECTED_NUMPY_DIMENSIONS:
        msg = f"Expected 4D numpy array (T, H, W, C), got shape {array.shape}"
        raise ValueError(msg)

    if array.shape[-1] != EXPECTED_CHANNELS:
        msg = f"Expected channels-last format (T, H, W, 3), got shape {array.shape}."
        raise ValueError(msg)

    return _normalize_dtype(array)


def _normalize_dtype(array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Normalize array dtype to uint8."""
    if array.dtype != np.uint8:
        if array.dtype in (np.float32, np.float16) and array.max() <= 1.0:
            return (array * 255).astype(np.uint8)
        return array.astype(np.uint8)
    return array


def _convert_tensor_to_numpy(tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
    """Convert torch.Tensor (T, C, H, W) to numpy (T, H, W, C)."""
    if tensor.ndim != EXPECTED_TENSOR_DIMENSIONS:
        msg = f"Expected 4D torch.Tensor (T, C, H, W), got shape {tensor.shape}"
        raise ValueError(msg)

    video_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return _normalize_dtype(video_np)


def _convert_video_format(
    video_inputs: torch.Tensor | npt.NDArray[np.uint8] | None,
) -> npt.NDArray[np.uint8] | None:
    """Convert torch.Tensor (T, C, H, W) or np.ndarray to vLLM format (T, H, W, C)."""
    retval: npt.NDArray[np.uint8] | None = None
    if video_inputs is None:
        retval = None
    elif isinstance(video_inputs, torch.Tensor):
        retval = _convert_tensor_to_numpy(video_inputs)
    else:  # isinstance(video_inputs, np.ndarray):
        retval = _validate_numpy_array(video_inputs)

    return retval


def make_prompt(
    message: dict[str, Any], frames: torch.Tensor, metadata: dict[str, Any], processor: AutoProcessor
) -> dict[str, Any]:
    """Make a prompt for the Nemotron Nano 12B v2 model.

    Args:
        message: The message to use for the prompt.
        frames: The frames to use for the prompt.
        metadata: The metadata of the video clip.
        processor: The processor to use for the prompt.

    Returns:
        A prompt for the Nemotron Nano 12B v2 model.

    """
    video_np = _convert_video_format(frames)
    prompt_ids = processor.apply_chat_template(  # type: ignore[attr-defined]
        [message], add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )[0].tolist()

    nemotron_metadata = {
        "total_num_frames": frames.shape[0],
        "fps": metadata["fps"],
        "duration": metadata["duration"],
        "frames_indices": metadata["frames_indices"],
        "video_backend": metadata["video_backend"],
    }

    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"video": (video_np, nemotron_metadata)},
    }


def make_message(text_input: str) -> dict[str, Any]:
    """Create a chat message structure for Nemotron Nano 12B v2.

    Args:
        text_input: The text input to create a message for.

    Returns:
        A chat message structure for Nemotron Nano 12B v2.

    """
    return {
        "role": "user",
        "content": [{"type": "video"}, {"type": "text", "text": text_input}],
    }


class VllmNemotronNano12Bv2VL(VllmPlugin):
    """Nemotron Nano 12B v2 vLLM model variant plugin."""

    @classmethod
    def model_variant(cls) -> str:
        """Return the model variant name."""
        return "nemotron"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model for Nemotron Nano 12B v2.

        Args:
            config: Configuration for the model.

        Returns:
            The vLLM model.

        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return LLM(
            model=str(cls.model_path(config)),
            trust_remote_code=TRUST_REMOTE_CODE,
            tensor_parallel_size=config.num_gpus,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
        )

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return the AutoProcessor for the model."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(cls.model_path(config)),
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=False,  # No fast processor available for nemotron, be explicit to silence warnings.
        )
        return cast("AutoProcessor", processor)

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],
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
        message = make_message(prompt)
        return make_prompt(message, frames, metadata, processor)

    @staticmethod
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
                None, the default refine prompt will be used.

        Returns:
            A refined LLM request.

        """
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt

        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        if "video" not in request.inputs["multi_modal_data"]:
            msg = "Message does not contain video"
            raise ValueError(msg)

        video_frames = request.inputs["multi_modal_data"]["video"][0]
        final_prompt = _refine_prompt + request.caption

        nemotron_metadata = request.inputs["multi_modal_data"]["video"][1]
        # nemotron_metadata is now a dict (converted in make_prompt to avoid pickle issues)
        metadata = nemotron_metadata if isinstance(nemotron_metadata, dict) else nemotron_metadata.__dict__

        inputs = make_prompt(make_message(final_prompt), video_frames, metadata, processor)

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vLLM output into a caption (extract <answer> section)."""
        return vllm_output.outputs[0].text
