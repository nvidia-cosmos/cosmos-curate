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
"""Phi Video Model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import nvtx  # type: ignore[import-untyped]
from loguru import logger

if TYPE_CHECKING:
    from PIL import Image
    from transformers import AutoTokenizer

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.model import conda_utils, model_utils

_PHI4_VL_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

_PHI_VARIANTS_INFO = {
    "phi4": _PHI4_VL_MODEL_ID,
}

# pyright: reportMissingImports=false
if conda_utils.is_running_in_env("phi"):
    import torch
    from torchvision import transforms  # type: ignore[import-untyped]
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.processing_auto import AutoProcessor

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


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

    # Clamp values to valid range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)  # TODO: We use UINT8 not FP16 so we may need to change this clamp to 255

    pil_image = tensor_to_pil_transform(tensor)
    return cast("Image.Image", pil_image)


def tensor_stack_to_pil_list(tensor_stack: torch.Tensor) -> list[Image.Image]:
    """Convert a NCHW tensor batch to a list of PIL images."""
    pil_images = []

    for i in range(tensor_stack.shape[0]):
        single_tensor = tensor_stack[i]
        pil_img = tensor_to_pil(single_tensor)
        pil_images.append(pil_img)
    return pil_images


class PhiUtils:
    """Utility class for handling Phi model inputs and message formatting."""

    def __init__(
        self,
        model_variant: str = "phi4",
    ) -> None:
        """Initialize the PhiUtils class.

        Args:
            model_variant: The variant of the Phi model to use.

        """
        self.weight_file = model_utils.get_local_dir_for_weights_name(_PHI_VARIANTS_INFO[model_variant])
        self.text_prompt = None
        self.processor: AutoProcessor | None = None

    def setup(self) -> None:
        """Set up the Phi model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        self.processor = AutoProcessor.from_pretrained(self.weight_file, trust_remote_code=True)  # type: ignore[no-untyped-call]

    @nvtx.annotate("Generate LLM inputs")  # type: ignore[misc]
    def generate_llm_inputs(
        self,
        prompt: str,
        video_inputs: torch.Tensor | None = None,
        *,
        override_text_prompt: bool = False,  # noqa: ARG002, we are likely to use this in the future
    ) -> dict[str, Any]:
        """Generate inputs for the Phi language model from video and text data.

        Processes video and text inputs to create the input for the Phi model. It handles both video and
        image inputs, decoding video and applying preprocessing if needed, and creates a structured
        input dictionary containing the processed prompt and multimodal data.

        Args:
            prompt: Text prompt to be included with the input.
            fps: Frames per second of the input video.
            preprocess_dtype: Data type to use for preprocessing the video/image inputs.
            num_frames_to_use: Number of frames to extract from the video. If 0, uses all frames.
            flip_input: Whether to flip the input video/image horizontally.
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

        pil_images = tensor_stack_to_pil_list(video_inputs)

        placeholder = ""

        for i in range(len(pil_images)):
            placeholder += f"<|image_{i + 1}|>"

        messages = {
            "role": "user",
            "content": (placeholder + prompt),
            "images": pil_images,
        }

        if self.processor is None:
            msg = "Processor is not set"
            raise ValueError(msg)

        processor = cast("AutoTokenizer", self.processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            msg = "Tokenizer is not set"
            raise ValueError(msg)

        token_prompt = tokenizer.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)

        messages["token_prompt"] = token_prompt

        return messages


class PhiVL(ModelInterface):
    """Interface for Phi vision-language model for video understanding and captioning."""

    def __init__(
        self,
        model_variant: str = "phi4",
        *,
        fp8: bool = True,
        max_output_tokens: int = 512,
        disable_mmcache: bool = False,
    ) -> None:
        """Initialize the PhiVL model.

        Args:
            model_variant: The variant of the Phi model to use.
            max_output_tokens: The maximum number of tokens to generate.
            model_does_preprocess: Whether to preprocess the model.
            stage2_prompt_text: The prompt for the stage 2 caption.
            disable_mmcache: Whether to disable the MM cache.
            fp8: Whether to use FP8 quantization.

        """
        super().__init__()
        self._weights_name = _PHI_VARIANTS_INFO[model_variant]
        self.weight_file = str(model_utils.get_local_dir_for_weights_name(self._weights_name))
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens
        self.disable_mmcache = disable_mmcache
        self.model_variant = model_variant
        self.processor: AutoProcessor | None = None
        self.model: AutoModelForCausalLM | None = None
        self.generation_config: GenerationConfig | None = None

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

    @nvtx.annotate("Setup Phi model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the Phi model.

        This method initializes the model and its configuration for processing video and text data.
        It also sets up the image processor for preprocessing video frames if needed.

        """
        logger.info("Setting up Phi model")
        self.processor = AutoProcessor.from_pretrained(self.weight_file, trust_remote_code=True)  # type: ignore[no-untyped-call]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weight_file,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )

        self.generation_config = GenerationConfig.from_pretrained(self.weight_file)

    @nvtx.annotate("Phi Generate tokens")  # type: ignore[misc]
    def generate(
        self,
        videos: list[dict[str, Any]],
        *,
        generate_stage2_caption: bool = False,  # noqa: ARG002
        batch_size: int = 16,  # noqa: ARG002
    ) -> list[str]:
        """Generate text for a list of videos.

        Args:
            videos: List of video dictionaries.
            generate_stage2_caption: Whether to generate a stage 2 caption.
            batch_size: Batch size for processing.

        Returns:
            List of generated captions.

        """
        generated_text = []
        for video in videos:
            token_prompt = video["token_prompt"]
            pil_images = video["images"]

            if self.processor is None:
                msg = "Processor is not set"
                raise ValueError(msg)

            if self.model is None:
                msg = "Model is not set"
                raise ValueError(msg)

            if self.generation_config is None:
                msg = "Generation config is not set"
                raise ValueError(msg)

            processor = cast("AutoProcessor", self.processor)  # type: ignore[redundant-cast]
            if not callable(processor):
                msg = "Processor is not callable"
                raise TypeError(msg)
            inputs = processor(token_prompt, images=pil_images, return_tensors="pt").to("cuda")

            generation_args = {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "do_sample": True,
                "repetition_penalty": 1.0,
            }

            model = cast("AutoModelForCausalLM", self.model)  # type: ignore[redundant-cast]
            generate = getattr(model, "generate", None)
            if not callable(generate):
                msg = "Model generate method is not callable"
                raise TypeError(msg)
            generate_ids = cast(
                "torch.Tensor",
                generate(
                    **inputs,
                    **generation_args,
                    generation_config=self.generation_config,
                    num_logits_to_keep=0,
                ),
            )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

            processor = cast("AutoProcessor", self.processor)  # type: ignore[redundant-cast]
            batch_decode = getattr(processor, "batch_decode", None)
            if not callable(batch_decode):
                msg = "Processor batch_decode method is not callable"
                raise TypeError(msg)
            raw_response = batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            response = cast("list[str]", raw_response)[0]

            generated_text.append(response)

        return generated_text
