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

"""Qwen Lang Model."""

from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils import conda_utils, model_utils

_QWEN_LM_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

if conda_utils.is_running_in_env("unified"):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams


class QwenLM(ModelInterface):
    """Interface for Qwen language model text generation."""

    def __init__(self, *, fp8: bool = True, max_output_tokens: int = 2048) -> None:
        """Initialize the QwenLM model.

        Args:
            fp8: Whether to use FP8 quantization.
            max_output_tokens: The maximum number of tokens to generate.

        """
        super().__init__()
        self.fp8 = fp8
        self.max_output_tokens = max_output_tokens

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
            List of model IDs.

        """
        return [_QWEN_LM_MODEL_ID]

    @nvtx.annotate("Setup Qwen-LM model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the Qwen-LM model.

        This method initializes the model and its configuration for text generation.
        It also sets up the sampling parameters for the model.

        """
        self.weight_file = str(model_utils.get_local_dir_for_weights_name(_QWEN_LM_MODEL_ID))
        if self.fp8:
            self.llm = LLM(
                model=self.weight_file,
                quantization="fp8",
                enforce_eager=False,
            )
        else:
            self.llm = LLM(model=self.weight_file, enforce_eager=False)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

    @nvtx.annotate("Qwen-LM Generate tokens")  # type: ignore[misc]
    def generate(
        self,
        prompts: list[list[dict[str, str]]],
    ) -> list[str]:
        """Generate text using the Qwen-LM model.

        Args:
            prompts: List of prompts to generate text from.

        Returns:
            List of generated text.

        """
        formatted_prompt = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate(formatted_prompt, sampling_params=self.sampling_params)
        return [out.outputs[0].text for out in outputs]


def make_qwen_lm_input(
    user_content: list[str],
    prompt_variant_key: str | None = None,
    prompt_variants: dict[str, str] | None = None,
    prompt_text: str | None = None,
) -> list[list[dict[str, str]]]:
    """Generate model inputs for qwen-lm based on the user content and a prompt.

    The prompt can be provided as either:
    - A prompt variant and a dictionary of prompts keyed on prompt variant
    - a prompt text, the string of the prompt

    But not both. If both are provided, ValueError is raised.

    Args:
        user_content: List of user content to send to the model
        prompt_variant_key: Type of prompt, for example: (visibility,
            road_conditions,illumination, default)
        prompt_variants: Dictionary of prompts to send to the model.
            prompt_variant_key is used to choose the prompt
        prompt_text: Text of the prompt to send to the model
    Returns:
        List of formatted inputs for the model, each containing system and
        user messages

    Raises:
        ValueError if:
        - prompt_variant is provided but no prompts
        - all of prompt_variant, prompts, and prompt_text are provided
        - none of prompt_variant, prompts, and prompt_text are provided

        KeyError if:
        - prompt_variant_key is provided but not in prompt_variants

    """
    if prompt_variant_key is not None and prompt_variants is None:
        error_msg = "prompt_variant_key provided but no prompt_variants"
        raise ValueError(error_msg)
    if prompt_variant_key is not None and prompt_text is not None:
        error_msg = "Cannot provide both prompt_variant_key and prompt_text"
        raise ValueError(error_msg)
    if prompt_variant_key is None and prompt_variants is None and prompt_text is None:
        error_msg = "Must provide either prompt_variant_key+prompt_variants or prompt_text"
        raise ValueError(error_msg)

    if prompt_text is not None:
        prompt = prompt_text
    else:
        assert prompt_variants is not None
        assert prompt_variant_key is not None
        prompt = prompt_variants[prompt_variant_key]

    return [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        for content in user_content
    ]
