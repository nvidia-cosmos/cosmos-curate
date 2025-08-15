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
"""Qwen-4 VLLM plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

from transformers import AutoProcessor
from vllm import LLM

from cosmos_curate.models.vllm_plugin import VLLMPlugin

if TYPE_CHECKING:
    import torch
    from vllm import RequestOutput
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from cosmos_curate.pipelines.video.utils.data_model import (
        Video,
        VLLMConfig,
        Window,
    )

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

    return {
        "prompt": tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True),
        "multi_modal_data": {"video": [frames]},
    }


class VLLMQwen(VLLMPlugin):
    """Qwen VLLM model variant plugin."""

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
    def make_refined_llm_input(
        caption: str, prev_input: dict[str, Any], processor: AutoProcessor, refine_prompt: str | None = None
    ) -> dict[str, Any]:
        """Get a prompt to refine an existing caption.

        Args:
            caption: The caption to refine
            prev_input: The prompt that was used to generate the caption
            processor: The processor to use for the stage 2 prompt
            refine_prompt: An optional prompt to use to refine the caption. If
                None, the default refine prompt will be used.

        Returns:
            A refined prompt

        """
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt
        final_prompt = _refine_prompt + caption

        if "multi_modal_data" not in prev_input:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        if "video" not in prev_input["multi_modal_data"]:
            msg = "Message does not contain video"
            raise ValueError(msg)

        videos = prev_input["multi_modal_data"]["video"]

        if len(videos) == 0:
            msg = "No videos provided"
            raise ValueError(msg)

        if len(videos) > 1:
            msg = "Multiple videos provided, only one is supported"
            raise ValueError(msg)

        message = make_message(final_prompt)
        video = videos[0]
        return make_prompt(message, video, processor)

    @staticmethod
    def add_llm_input_to_window(window: Window, llm_input: dict[str, Any]) -> None:
        """Add LLM input to a Qwen window.

        Args:
            window: The window.
            llm_input: The LLM input for the window.

        """
        window.qwen_llm_input = llm_input

    @staticmethod
    def get_llm_input_from_window(window: Window) -> dict[str, Any] | None:
        """Get the LLM input for a Qwen window.

        Args:
            window: The window.

        Returns:
            The LLM input for the window.

        """
        return window.qwen_llm_input

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
                window.qwen_llm_input = None


class VLLMQwen7B(VLLMQwen):
    """Qwen VLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "qwen"

    @staticmethod
    def model_id() -> str:
        """Return the model ID."""
        return "Qwen/Qwen2.5-VL-7B-Instruct"
