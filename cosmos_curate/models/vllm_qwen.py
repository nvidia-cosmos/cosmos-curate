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
"""Qwen vLLM plugin."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, TypedDict, cast

from transformers import AutoProcessor
from vllm import LLM

from cosmos_curate.models.vllm_plugin import VllmPlugin
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest

if TYPE_CHECKING:
    import torch
    from vllm import RequestOutput
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig


MAX_SEQ_LEN_TO_CAPTURE = 32768
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
DEFAULT_BATCH_SIZE = 16
TRUST_REMOTE_CODE = False
LIMIT_MM_PER_PROMPT = {"images": 0, "video": 1}

_DEFAULT_REFINE_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""


class QwenContentType(TypedDict):  # noqa: D101
    type: str


class QwenContentTypeText(TypedDict):  # noqa: D101
    type: str
    text: str


class QwenMessage(TypedDict):  # noqa: D101
    role: str
    content: list[QwenContentType | QwenContentTypeText]


def make_message(
    text_input: str,
) -> QwenMessage:
    """Create a message for the Qwen model.

    Args:
        text_input: The text input to create a message for.

    Returns:
        A message for the Qwen model.

    """
    return QwenMessage(
        role="user",
        content=[
            QwenContentType(type="video"),
            QwenContentTypeText(type="text", text=text_input),
        ],
    )


def make_prompt(message: QwenMessage, frames: torch.Tensor, processor: AutoProcessor) -> dict[str, Any]:
    """Make a prompt for the Qwen model.

    Args:
        message: The message to use to create the prompt
        frames: The frames to use for the prompt.
        processor: The processor to use for the prompt.

    Returns:
        A prompt for the Qwen model.

    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        msg = "Tokenizer is not set"
        raise ValueError(msg)

    prompt_ids = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": ({"video": frames}),
    }


class VllmQwen(VllmPlugin):
    """Qwen vLLM model variant plugin."""

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

        mm_processor_kwargs = {
            "do_resize": config.preprocess,
            "do_rescale": config.preprocess,
            "do_normalize": config.preprocess,
        }

        return LLM(
            model=str(cls.model_path()),
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            quantization=quantization,
            max_seq_len_to_capture=MAX_SEQ_LEN_TO_CAPTURE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            mm_processor_kwargs=mm_processor_kwargs,
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
        """Make LLM inputs for the model.

        Args:
            prompt: The prompt to use for the LLM.
            frames: The frames to use for the LLM.
            processor: The AutoProcessor to use for the LLM.

        Returns:
            A dictionary containing the LLM inputs.

        """
        message = make_message(prompt)
        return make_prompt(message, frames, processor)

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

        final_prompt = _refine_prompt + request.caption

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        if "video" not in request.inputs["multi_modal_data"]:
            msg = "Message does not contain video"
            raise ValueError(msg)

        video_frames = request.inputs["multi_modal_data"]["video"]

        message = make_message(final_prompt)
        inputs = make_prompt(message, video_frames, processor)

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
            video_idx=request.video_idx,
            clip_idx=request.clip_idx,
            window_idx=request.window_idx,
            iterations=request.iterations,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vllm output into a caption."""
        return vllm_output.outputs[0].text


class VllmQwen7B(VllmQwen):
    """Qwen-7B vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen"
