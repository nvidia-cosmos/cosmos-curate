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

"""Qwen Video Model."""

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.core.utils.model import conda_utils, model_utils

_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_QWEN_VARIANTS_INFO = {
    "qwen": _QWEN2_5_VL_MODEL_ID,
}

_DEFAULT_STAGE2_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.\n
"""

# pyright: reportMissingImports=false
if conda_utils.is_running_in_env("unified"):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if TYPE_CHECKING:
        from vllm.model_executor.layers.quantization import QuantizationMethods

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


class QwenUtils:
    """Utility class for handling Qwen model inputs and message formatting."""

    def __init__(
        self,
        model_variant: str = "qwen",
    ) -> None:
        """Initialize the QwenUtils class.

        Args:
            model_variant: The variant of the Qwen model to use.

        """
        self.weight_file = model_utils.get_local_dir_for_weights_name(_QWEN_VARIANTS_INFO[model_variant])
        self.text_prompt = None
        self.processor: AutoProcessor | None = None

    def setup(self) -> None:
        """Set up the Qwen model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        self.processor = AutoProcessor.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

    @staticmethod
    def create_message(
        text_input: str,
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        """Create a message for the Qwen model.

        Args:
            text_input: The text input to create a message for.

        Returns:
            List of messages for the Qwen model.

        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": text_input,
                    },
                ],
            },
        ]

    @nvtx.annotate("Generate LLM inputs")  # type: ignore[misc]
    def generate_llm_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | None = None,
        *,
        override_text_prompt: bool = False,
    ) -> dict[str, Any]:
        """Generate inputs for the Qwen language model from video and text data.

        Processes video and text inputs to create the input for the Qwen model. It handles both video and
        image inputs, decoding video and applying preprocessing if needed, and creates a structured
        input dictionary containing the processed prompt and multimodal data.

        Args:
            prompt: Text prompt to be included with the input.
            video_inputs: Pre-processed video inputs. If None, and video data is to be passed to
                          the model, then video cannot be None.
            override_text_prompt: whether the text prompt should be overridden

        Returns:
            dict containing:
                - "prompt": The processed text prompt with chat template applied
                - "multi_modal_data": Dictionary containing processed "image" and/or "video" inputs

        """
        if video_inputs is None:
            error_msg = "No input frames provided, cannot call process_vision_info"
            raise ValueError(error_msg)

        messages = self.create_message(prompt)
        if override_text_prompt or self.text_prompt is None:
            assert self.processor is not None
            self.text_prompt = self.processor.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        mm_data = {}
        mm_data["video"] = [video_inputs]
        return {
            "prompt": self.text_prompt,
            "multi_modal_data": mm_data,
        }


class QwenVL(ModelInterface):
    """Interface for Qwen vision-language model for video understanding and captioning."""

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        *,
        fp8: bool = True,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,
        stage2_prompt_text: str | None = None,
        disable_mmcache: bool = False,
        num_gpus: int = 1,
    ) -> None:
        """Initialize the QwenVL model.

        Args:
            model_variant: The variant of the Qwen model to use.
            fp8: Whether to use FP8 quantization.
            max_output_tokens: The maximum number of tokens to generate.
            model_does_preprocess: Whether to preprocess the model.
            stage2_prompt_text: The prompt for the stage 2 caption.
            disable_mmcache: Whether to disable the MM cache.
            num_gpus: Number of GPUs to use for processing.

        """
        super().__init__()
        self._weights_name = _QWEN_VARIANTS_INFO[model_variant]
        self.weight_file = str(model_utils.get_local_dir_for_weights_name(self._weights_name))
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.model_does_preprocess = model_does_preprocess
        self.disable_mmcache = disable_mmcache
        self.llm: LLM | None = None
        self.model_variant = model_variant
        self.sampling_params: SamplingParams | None = None
        self.num_gpus = num_gpus
        self.pattern = (
            r"(<\|im_start\|>user\s*<\|vision_start\|><\|video_pad\|><\|vision_end\|>\s*)(.*?)(\s*<\|im_end\|>)"
        )
        self.stage2_prompt: str = _DEFAULT_STAGE2_PROMPT
        if stage2_prompt_text is not None:
            self.stage2_prompt = stage2_prompt_text

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [self._weights_name]

    @nvtx.annotate("Setup Qwen model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the Qwen model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        logger.info("Setting up Qwen model")
        mm_processor_kwargs = {
            "do_resize": self.model_does_preprocess,
            "do_rescale": self.model_does_preprocess,
            "do_normalize": self.model_does_preprocess,
        }

        quantization: QuantizationMethods | None = None
        if self.fp8:
            quantization = "fp8"

        self.llm = LLM(
            model=self.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization=quantization,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mmcache,
            max_num_batched_tokens=32768,
            tensor_parallel_size=self.num_gpus,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )

        logger.info(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences",
        )

    @nvtx.annotate("Qwen Generate tokens")  # type: ignore[misc]
    def generate(
        self,
        videos: list[dict[str, Any]],
        *,
        generate_stage2_caption: bool = False,
        batch_size: int = 16,
    ) -> list[str]:
        """Generate text for a list of videos.

        Args:
            videos: List of input dictionaries for the LLM.
            generate_stage2_caption: Whether to generate a stage 2 caption.
            batch_size: Batch size for processing.

        Returns:
            List of generated captions.

        """
        generated_text = []
        for batch_videos in grouping.split_by_chunk_size(videos, batch_size):
            llm_inputs = list(batch_videos)

            try:
                assert self.llm is not None
                assert self.sampling_params is not None
                outputs = self.llm.generate(
                    llm_inputs,  # type: ignore[arg-type]
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                if generate_stage2_caption:
                    for i, out in enumerate(outputs):
                        updated_prompt = self.stage2_prompt + out.outputs[0].text
                        llm_inputs[i]["prompt"] = re.sub(
                            self.pattern,
                            rf"\1{updated_prompt}\3",
                            llm_inputs[i]["prompt"],
                            flags=re.DOTALL,
                        )

                    outputs = self.llm.generate(
                        llm_inputs,  # type: ignore[arg-type]
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )

                generated_text.extend([out.outputs[0].text for out in outputs])

            except Exception as e:
                logger.exception(f"Error generating text for batch of {len(batch_videos)}: {e}")
                raise

        return generated_text
