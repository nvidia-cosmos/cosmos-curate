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
"""vLLM plugin for Cosmos-Reason1 vision-language model."""

from __future__ import annotations

import re
import secrets
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoProcessor
from vllm import LLM, RequestOutput

from cosmos_curate.models.vllm_plugin import VllmPlugin
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig


# Constants tuned similarly to existing plugins
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 32768
TRUST_REMOTE_CODE = False
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


def _extract_from_reasoning_format(text: str) -> str:
    """Extract the <answer>...</answer> content if present.

    Falls back to the original text if the reasoning format is missing.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def make_message(text_input: str) -> list[dict[str, Any]]:
    """Create a chat message structure for Cosmos-Reason1.

    The system prompt instructs the reasoning format. The user content
    includes a video placeholder and the user's text prompt.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question in the following format: "
                "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": text_input},
            ],
        },
    ]


def make_prompt(message: list[dict[str, Any]], frames: torch.Tensor, processor: AutoProcessor) -> dict[str, Any]:
    """Create a prompt payload for vLLM using the processor chat template.

    Returns a dict containing the prompt string and the multi_modal_data payload
    that includes the video frames.
    """
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is None:
        msg = "Processor does not support apply_chat_template"
        raise ValueError(msg)

    prompt_str = apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": cast("str", prompt_str),
        # vLLM expects a list of tensors for video under multi_modal_data
        "multi_modal_data": {"video": [frames]},
    }


class VllmCosmosReason1VL(VllmPlugin):
    """Cosmos-Reason1 vLLM model variant plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "cosmos_r1"

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model for Cosmos-Reason1.

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
            model=str(cls.model_path(config)),
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            quantization=quantization,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            mm_processor_kwargs=mm_processor_kwargs,
            disable_mm_preprocessor_cache=config.disable_mmcache,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
            compilation_config={"cudagraph_mode": "piecewise"},
        )

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return the AutoProcessor for the model."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cls.model_path(config),
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

        if "multi_modal_data" not in request.inputs:
            msg = "Message does not contain multi_modal_data"
            raise ValueError(msg)

        if "video" not in request.inputs["multi_modal_data"]:
            msg = "Message does not contain video"
            raise ValueError(msg)

        video_frames = request.inputs["multi_modal_data"]["video"][0]
        final_prompt = _refine_prompt + request.caption

        inputs = make_prompt(make_message(final_prompt), video_frames, processor)

        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vLLM output into a caption (extract <answer> section)."""
        return _extract_from_reasoning_format(vllm_output.outputs[0].text)
