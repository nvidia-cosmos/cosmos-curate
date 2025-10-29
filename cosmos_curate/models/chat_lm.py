# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-focused LM wrapper supporting both local (vLLM) and remote (Azure OpenAI) inference.

This module provides a unified implementation for chat-style text generation that can use:
- Local models via vLLM for on-premises inference
- Azure OpenAI API for cloud-based inference

Key capabilities:
- Unified interface for both local and remote models via variant selection
- For local: Resolves weights/tokenizer and constructs a vLLM `LLM` with optional quantization
- For remote: Uses Azure OpenAI API with deployment-based model selection
- Formats prompts with chat templates for consistent behavior across backends
- Supports batching and provides `make_chat_lm_input` helper
- Designed for the "unified" conda environment and emits NVTX ranges
"""

from typing import TYPE_CHECKING, cast

from loguru import logger
from nvtx import nvtx  # type: ignore[import-untyped]

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.config.config import load_config
from cosmos_curate.core.utils.environment import CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE
from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.core.utils.model import conda_utils, model_utils

if TYPE_CHECKING:
    from openai import AzureOpenAI
    from vllm.model_executor.layers.quantization import QuantizationMethods

if conda_utils.is_running_in_env("unified"):
    from openai import AzureOpenAI
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams


_VARIANT_TO_MODEL_ID: dict[str, str] = {
    "qwen_lm": "Qwen/Qwen2.5-14B-Instruct",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}

_LOCAL_VARIANTS = {"qwen_lm", "gpt_oss_20b"}
_REMOTE_VARIANTS = {"azure_openai"}


class ChatLM(ModelInterface):
    """Unified chat LM supporting both local (vLLM) and remote (Azure OpenAI) backends."""

    def __init__(
        self,
        model_variant: str,
        *,
        max_output_tokens: int = 2048,
        quantization: str | None = None,
        azure_deployment: str = "gpt-5-chat-20250807",
        verbose: bool = False,
    ) -> None:
        """Initialize the ChatLM.

        Args:
            model_variant: Short variant key (e.g., "qwen_lm", "azure_openai").
            max_output_tokens: Maximum tokens to generate per prompt.
            quantization: Optional quantization override for vLLM (e.g., "fp8").
                Only applies to local variants. If None, vLLM uses the model's config-defined quantization.
            azure_deployment: Azure OpenAI deployment name (only used when model_variant is "azure_openai").
                Defaults to "gpt-5-chat-20250807".
            verbose: Whether to emit verbose debug logs (e.g., Azure request metadata).

        """
        super().__init__()
        self._model_variant = model_variant
        self._is_local = model_variant in _LOCAL_VARIANTS
        self._is_remote = model_variant in _REMOTE_VARIANTS

        if not self._is_local and not self._is_remote:
            error = f"Unsupported chat LM variant: {model_variant}"
            raise ValueError(error)

        self._model_id = _VARIANT_TO_MODEL_ID.get(model_variant) if self._is_local else None
        self.max_output_tokens = max_output_tokens
        self._quantization = cast("QuantizationMethods | None", quantization)
        self._azure_deployment = azure_deployment
        self._temperature = 0.1
        self._top_p = 0.001
        self._verbose = verbose

        # Warn about ignored parameters for remote variants
        if self._is_remote and quantization is not None:
            logger.warning(
                f"quantization parameter ('{quantization}') is ignored for remote model variant '{model_variant}'"
            )

        # Early validation for Azure OpenAI to fail fast
        if self._is_remote:
            self._resolve_azure_openai_settings()

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name used for this model."""
        return "unified"

    @property
    def requires_gpu(self) -> bool:
        """Check if this model requires GPU resources.

        Returns:
            True for local models (vLLM), False for remote API models.

        """
        return self._is_local

    @property
    def model_id_names(self) -> list[str]:
        """Return the underlying model identifiers."""
        if self._is_local:
            assert self._model_id is not None
            return [self._model_id]
        # Remote API models don't require local weight downloads
        return []

    @staticmethod
    def _resolve_azure_openai_settings() -> tuple[str, str, str]:
        """Load Azure OpenAI settings from the cosmos-curate config file.

        Returns:
            Tuple of (api_version, azure_endpoint, api_key).

        """
        config = load_config()
        if config.azure_openai is None:
            error_msg = (
                f"Azure OpenAI configuration not found. Ensure {CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE} contains "
                "azure_openai.api_version, azure_openai.azure_endpoint, and azure_openai.api_key."
            )
            raise RuntimeError(error_msg)

        api_version = config.azure_openai.api_version
        if not api_version:
            error_msg = "Azure OpenAI API version missing from config file."
            raise RuntimeError(error_msg)
        azure_endpoint = config.azure_openai.azure_endpoint
        if not azure_endpoint:
            error_msg = "Azure OpenAI endpoint missing from config file."
            raise RuntimeError(error_msg)
        api_key = config.azure_openai.api_key
        if not api_key:
            error_msg = "Azure OpenAI API key missing from config file."
            raise RuntimeError(error_msg)

        return api_version, azure_endpoint, api_key

    @nvtx.annotate("Setup Chat LM model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the model and tokenizer, and sampling parameters."""
        if self._is_local:
            assert self._model_id is not None
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
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=1.05,
                max_tokens=self.max_output_tokens,
                stop_token_ids=[],
            )

            # Prefer local tokenizer to avoid Hub lookups when weights are local.
            self.tokenizer = AutoTokenizer.from_pretrained(self.weight_file)  # type: ignore[no-untyped-call]

        elif self._is_remote:
            # Set up Azure OpenAI client
            version, endpoint, key = self._resolve_azure_openai_settings()
            self.azure_client: AzureOpenAI = AzureOpenAI(
                api_version=version,
                azure_endpoint=endpoint,
                azure_deployment=self._azure_deployment,
                api_key=key,
            )
            # For Azure OpenAI, we'll format messages manually in generate()

    @nvtx.annotate("Chat LM Generate tokens")  # type: ignore[misc]
    def generate(
        self,
        prompts: list[list[dict[str, str]]],
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate text given chat prompts.

        Args:
            prompts: Batched chat prompts, each as a list of role/content dicts.
            batch_size: Batch size for generation. Defaults to 32 for local models,
                full batch for remote models.

        Returns:
            List of generated strings, one per prompt.

        """
        if self._is_local:
            return self._generate_local(prompts, batch_size or 32)
        if self._is_remote:
            return self._generate_remote(prompts, batch_size)
        error = f"Unknown model variant: {self._model_variant}"
        raise RuntimeError(error)

    def _generate_local(self, prompts: list[list[dict[str, str]]], batch_size: int) -> list[str]:
        """Generate text using local vLLM backend."""
        generated_text: list[str] = []
        for batch_prompts in grouping.split_by_chunk_size(prompts, batch_size):
            formatted_prompts = self.tokenizer.apply_chat_template(
                list(batch_prompts), tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=False)
            generated_text.extend([out.outputs[0].text for out in outputs])

        return generated_text

    def _generate_remote(self, prompts: list[list[dict[str, str]]], batch_size: int | None) -> list[str]:
        """Generate text using Azure OpenAI API."""
        if not prompts:
            return []

        chunk_size = batch_size or len(prompts)
        outputs: list[str] = []

        for chunk_start in range(0, len(prompts), chunk_size):
            chunk = prompts[chunk_start : chunk_start + chunk_size]
            # The OpenAI Chat Completions API processes one conversation at a time
            for message_bundle in chunk:
                messages: list[dict[str, str]] = [
                    {"role": message["role"], "content": message["content"]} for message in message_bundle
                ]
                try:
                    if self._verbose:
                        logger.info(
                            (
                                "Azure OpenAI request (deployment='{}', messages={}, roles={}, "
                                "max_tokens={}, temperature={}, top_p={})"
                            ),
                            self._azure_deployment,
                            len(messages),
                            [msg["role"] for msg in messages],
                            self.max_output_tokens,
                            self._temperature,
                            self._top_p,
                        )
                    response = self.azure_client.chat.completions.create(
                        model=self._azure_deployment,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=self.max_output_tokens,
                        temperature=self._temperature,
                        top_p=self._top_p,
                    )
                    usage_info = getattr(response, "usage", None)
                    if self._verbose:
                        if usage_info:
                            logger.info(
                                (
                                    "Azure OpenAI response usage (deployment='{}', prompt_tokens={}, "
                                    "completion_tokens={}, total_tokens={})"
                                ),
                                self._azure_deployment,
                                getattr(usage_info, "prompt_tokens", None),
                                getattr(usage_info, "completion_tokens", None),
                                getattr(usage_info, "total_tokens", None),
                            )
                        else:
                            logger.info(
                                "Azure OpenAI response (deployment='{}') returned without usage metadata",
                                self._azure_deployment,
                            )
                except Exception as exc:  # noqa: BLE001  # pragma: no cover
                    logger.error(
                        "Azure OpenAI API call failed for deployment {}: {}",
                        self._azure_deployment,
                        exc,
                    )
                    outputs.append("")
                    continue

                if not response.choices:
                    logger.warning(
                        "Azure OpenAI API returned no choices for deployment {}",
                        self._azure_deployment,
                    )
                    outputs.append("")
                    continue

                choice = response.choices[0]
                content = ""
                if choice.message and choice.message.content:
                    content = choice.message.content
                outputs.append(content)

        return outputs


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
