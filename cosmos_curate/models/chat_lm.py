# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-focused LM wrapper around vLLM and Transformers for local-weight inference.

This module provides a minimal, reusable implementation for chat-style text
generation backed by vLLM, while preferring local model/tokenizer directories.
It exposes a generic `ChatLM` with sensible defaults and a small variant
factory, plus a helper to build chat-formatted inputs.

Key capabilities:
- Resolves local weights/tokenizer via `model_utils` and constructs a vLLM `LLM`.
- Honors model-config quantization by default; allows explicit overrides.
- Optional FP8 request for Qwen via `ChatLM.from_variant`.
- Formats prompts with the tokenizer's chat template and supports batching.
- Provides `make_chat_lm_input` to assemble system/user message lists.
- Designed for the "unified" conda environment and emits NVTX ranges.
"""

from typing import TYPE_CHECKING, cast

from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.core.utils.model import conda_utils, model_utils

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

if conda_utils.is_running_in_env("unified"):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams


_VARIANT_TO_MODEL_ID: dict[str, str] = {
    "qwen_lm": "Qwen/Qwen2.5-14B-Instruct",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}


class ChatLM(ModelInterface):
    """Generic chat LM with configurable model id and quantization."""

    def __init__(
        self,
        model_variant: str,
        *,
        max_output_tokens: int = 2048,
        quantization: str | None = None,
    ) -> None:
        """Initialize the ChatLM.

        Args:
            model_variant: Short variant key (e.g., "qwen_lm").
            max_output_tokens: Maximum tokens to generate per prompt.
            quantization: Optional quantization override for vLLM (e.g., "fp8").
                If None, vLLM uses the model's config-defined quantization.

        """
        super().__init__()
        if model_variant not in _VARIANT_TO_MODEL_ID:
            error = f"Unsupported chat LM variant: {model_variant}"
            raise ValueError(error)
        self._model_id = _VARIANT_TO_MODEL_ID[model_variant]
        self.max_output_tokens = max_output_tokens
        self._quantization = cast("QuantizationMethods | None", quantization)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name used for this model."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Return the underlying model identifiers."""
        return [self._model_id]

    @nvtx.annotate("Setup Chat LM model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the model and tokenizer, and sampling parameters."""
        self.weight_file = str(model_utils.get_local_dir_for_weights_name(self._model_id))

        # Construct vLLM LLM. Avoid forcing quantization unless explicitly requested,
        # so that model-config (e.g., mxfp4) is honored by default.
        if self._quantization is not None:
            self.llm = LLM(
                model=self.weight_file,
                quantization=self._quantization,
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

        # Prefer local tokenizer to avoid Hub lookups when weights are local.
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

    @nvtx.annotate("Chat LM Generate tokens")  # type: ignore[misc]
    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        batch_size: int = 32,
    ) -> list[str]:
        """Generate text given chat prompts.

        Args:
            prompts: Batched chat prompts, each as a list of role/content dicts.
            batch_size: Batch size for generation.

        Returns:
            List of generated strings, one per prompt.

        """
        generated_text: list[str] = []
        for batch_prompts in grouping.split_by_chunk_size(prompts, batch_size):
            formatted_prompts = self.tokenizer.apply_chat_template(
                list(batch_prompts), tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=False)
            generated_text.extend([out.outputs[0].text for out in outputs])

        return generated_text


def make_chat_lm_input(
    user_content: list[str],
    *,
    prompt_variant_key: str | None = None,
    prompt_variants: dict[str, str] | None = None,
    prompt_text: str | None = None,
) -> list[list[dict[str, str]]]:
    """Generate chat-style inputs given user content and a prompt source.

    Exactly one of (prompt_text) or (prompt_variant_key+prompt_variants) must be provided.

    Args:
        user_content: List of user messages to send to the model
        prompt_variant_key: Key to select prompt from prompt_variants
        prompt_variants: Mapping of prompt variants to prompt text
        prompt_text: Direct prompt text

    Returns:
        A list of chat messages (system+user) per input content.

    """
    if prompt_variant_key is not None and prompt_variants is None:
        error = "prompt_variant_key provided but no prompt_variants"
        raise ValueError(error)
    if prompt_variant_key is not None and prompt_text is not None:
        error = "Cannot provide both prompt_variant_key and prompt_text"
        raise ValueError(error)
    if prompt_variant_key is None and prompt_variants is None and prompt_text is None:
        error = "Must provide either prompt_variant_key+prompt_variants or prompt_text"
        raise ValueError(error)

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
