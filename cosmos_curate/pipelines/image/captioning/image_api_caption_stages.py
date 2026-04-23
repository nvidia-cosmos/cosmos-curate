# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Remote API-backed image captioning stages for Gemini and OpenAI-compatible endpoints."""

import base64
import mimetypes
from typing import TYPE_CHECKING, Any

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.config import load_config, maybe_load_config
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.prompts import get_prompt
from cosmos_curate.pipelines.common.api_caption_utils import (
    create_openai_client_and_resolve_model,
    gemini_error_result_from_exception,
    handle_gemini_client_exception,
    normalize_gemini_response_with_detail,
    normalize_openai_response_with_detail,
    openai_error_result_from_exception,
    should_retry_gemini_exception,
)
from cosmos_curate.pipelines.image.captioning.image_prep_utils import (
    DEFAULT_PREP_MAX_PIXELS,
    DEFAULT_PREP_MIN_PIXELS,
    prepare_image_endpoint_input,
)
from cosmos_curate.pipelines.image.utils.data_model import Image, ImagePipeTask
from cosmos_curate.pipelines.video.utils.data_model import (
    CaptionOutcome,
    CaptionResult,
)

if TYPE_CHECKING:
    import openai
    from google import genai
    from google.genai import types as genai_types

if conda_utils.is_running_in_env("unified"):
    import openai
    from google import genai
    from google.api_core.exceptions import DeadlineExceeded
    from google.genai import types as genai_types
else:

    class DeadlineExceeded(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Fallback placeholder when Gemini deps are not installed."""


def _guess_media_type(image: Image, payload_bytes: bytes) -> str:
    """Infer a MIME type for uploaded image bytes."""
    if image.image_data is not None and image.image_data.metadata.image_format is not None:
        fmt = image.image_data.metadata.image_format.lower()
        if fmt == "jpg":
            fmt = "jpeg"
        return f"image/{fmt}"
    if image.relative_path:
        guessed, _ = mimetypes.guess_type(image.relative_path)
        if guessed is not None and guessed.startswith("image/"):
            return guessed
    if payload_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "image/jpeg"


def _write_caption_result(image: Image, model_variant: str, result: CaptionResult) -> None:
    """Write a normalized caption result back to the image task."""
    if result.text is not None:
        image.caption = result.text
        image.captions[model_variant] = result.text
    image.caption_status = result.outcome.value
    image.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None


class ImageOpenAIPrepStage(CuratorStage):
    """Prepare resized endpoint payloads for OpenAI-compatible image captioning."""

    def __init__(
        self,
        *,
        caption_prep_min_pixels: int | None = None,
        caption_prep_max_pixels: int | None = None,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize prep stage with pixel bounds and preprocessing options."""
        self._timer = StageTimer(self)
        self._min_pixels = caption_prep_min_pixels if caption_prep_min_pixels is not None else DEFAULT_PREP_MIN_PIXELS
        self._max_pixels = caption_prep_max_pixels if caption_prep_max_pixels is not None else DEFAULT_PREP_MAX_PIXELS
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Return the CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment name required by this stage."""
        return "unified"

    @nvtx.annotate("ImageOpenAIPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Resize images and cache an endpoint-ready PNG payload."""
        for task in tasks:
            image = task.image
            self._timer.reinit(self, task.get_major_size())
            if image.image_data is None:
                image.errors["caption_prep"] = "no image_data"
                continue
            if len(image.image_data.frames) == 0:
                image.errors["caption_prep"] = "image_data has no frames"
                continue
            with self._timer.time_process():
                try:
                    prepared = prepare_image_endpoint_input(
                        image.image_data.frames[0],
                        min_pixels=self._min_pixels,
                        max_pixels=self._max_pixels,
                    )
                except Exception as exc:  # noqa: BLE001
                    image.errors["caption_prep"] = str(exc)
                    logger.warning(f"Caption prep failed for {task.session_id}: {exc}")
                    continue
            image.model_input["openai"] = {
                "payload_bytes": prepared["payload_bytes"],
                "media_type": "image/png",
            }
            image.height = prepared["height"]
            image.width = prepared["width"]
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks


class ImageGeminiCaptionStage(CuratorStage):
    """Caption images using the Gemini API."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_variant: str = "gemini",
        model_name: str = "models/gemini-2.5-pro",
        prompt_variant: str = "image",
        prompt_text: str | None = None,
        max_output_tokens: int = 8192,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize Gemini caption stage with model, prompt, and retry configuration."""
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._model_name = model_name
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._verbose = verbose
        self._log_stats = log_stats
        config = load_config()
        if config.gemini is None or not config.gemini.api_key:
            msg = "Gemini API key missing from config file."
            raise RuntimeError(msg)
        self._api_key: str = config.gemini.api_key
        self._client: genai.Client | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Return the CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment name required by this stage."""
        return "unified"

    def stage_setup(self) -> None:
        """Initialize the Gemini client using the API key loaded at construction."""
        self._client = genai.Client(api_key=self._api_key)

    def _generate_caption_with_error_detail(self, image: Image) -> tuple[CaptionResult, str | None]:
        client = self._client
        if client is None:
            msg = "Gemini client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)
        raw = image.encoded_data.resolve()
        if raw is None:
            return CaptionResult(
                outcome=CaptionOutcome.ERROR, failure_reason="exception"
            ), "Image missing encoded_data."
        raw_bytes = bytes(raw)
        inline_data = genai_types.Blob(data=raw_bytes, mime_type=_guess_media_type(image, raw_bytes))
        content = genai_types.Content(
            parts=[
                genai_types.Part(inline_data=inline_data),
                genai_types.Part(text=self._prompt.strip()),
            ]
        )
        generate_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "contents": content,
            "config": genai_types.GenerateContentConfig(max_output_tokens=self._max_output_tokens),
        }

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_caption_retries),
            wait=tenacity.wait_fixed(self._retry_delay_seconds),
            retry=tenacity.retry_if_exception(should_retry_gemini_exception),
            reraise=True,
        )
        def _call() -> object:
            try:
                return client.models.generate_content(**generate_kwargs)
            except Exception as exc:
                new_exc = handle_gemini_client_exception(exc)
                if new_exc is exc:
                    raise
                raise new_exc from exc

        try:
            response = _call()
        except Exception as exc:  # noqa: BLE001
            return gemini_error_result_from_exception(exc, timeout_error_type=DeadlineExceeded)
        return normalize_gemini_response_with_detail(response)

    @nvtx.annotate("ImageGeminiCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Caption images in the batch using the Gemini API."""
        for task in tasks:
            image = task.image
            self._timer.reinit(self, task.get_major_size())
            if image.has_caption():
                continue
            with self._timer.time_process():
                try:
                    result, detail = self._generate_caption_with_error_detail(image)
                except Exception as exc:  # noqa: BLE001
                    result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
                    detail = str(exc)
                    image.errors[f"{self._model_variant}_caption"] = detail
                    if self._verbose:
                        logger.exception(f"Gemini captioning failed for image {task.session_id}")
                    else:
                        logger.warning(f"Gemini captioning failed for image {task.session_id}: {exc}")
                else:
                    if result.outcome == CaptionOutcome.ERROR:
                        error_msg = detail or result.failure_reason or "unknown"
                        image.errors[f"{self._model_variant}_caption"] = error_msg
                        logger.warning(f"Gemini captioning failed for image {task.session_id}: {error_msg}")
                    elif result.outcome == CaptionOutcome.BLOCKED:
                        logger.warning(f"Gemini captioning blocked for image {task.session_id}")
                    elif self._verbose and result.text is not None:
                        logger.info(f"Gemini caption for image {task.session_id}: {result.text}")
            _write_caption_result(image, self._model_variant, result)
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks


class ImageOpenAICaptionStage(CuratorStage):
    """Caption images using an OpenAI-compatible vision endpoint."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        model_variant: str = "openai",
        prompt_variant: str = "image",
        prompt_text: str | None = None,
        max_output_tokens: int = 8192,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize OpenAI caption stage with model, prompt, and retry configuration."""
        self._timer = StageTimer(self)
        self._model_name = model_name
        self._model_variant = model_variant
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: openai.OpenAI | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Return the CPU resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment name required by this stage."""
        return "unified"

    def stage_setup(self) -> None:
        """Initialize the OpenAI client using endpoint config."""
        config = maybe_load_config()
        endpoint = config.openai.caption if config is not None and config.openai is not None else None
        if endpoint is None or not endpoint.api_key:
            msg = (
                "OpenAI caption configuration not found. "
                "Provide openai.caption.api_key in ~/.config/cosmos_curate/config.yaml"
            )
            raise RuntimeError(msg)
        self._client, self._model_name = create_openai_client_and_resolve_model(
            openai,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model_name=self._model_name,
            endpoint_label="OpenAI caption",
        )

    def _resolve_payload(self, image: Image) -> tuple[bytes, str]:
        cached = image.model_input.get("openai")
        if isinstance(cached, dict):
            payload_bytes = cached.get("payload_bytes")
            media_type = cached.get("media_type")
            if isinstance(payload_bytes, bytes) and isinstance(media_type, str):
                return payload_bytes, media_type
        raw = image.encoded_data.resolve()
        if raw is None:
            msg = "Image missing encoded_data."
            raise RuntimeError(msg)
        raw_bytes = bytes(raw)
        return raw_bytes, _guess_media_type(image, raw_bytes)

    def _generate_caption_with_error_detail(self, image: Image) -> tuple[CaptionResult, str | None]:
        client = self._client
        if client is None:
            msg = "OpenAI client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)
        payload_bytes, media_type = self._resolve_payload(image)
        image_b64 = base64.b64encode(payload_bytes).decode("utf-8")
        content_parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self._prompt.strip()},
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": self._max_output_tokens,
        }

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_caption_retries),
            wait=tenacity.wait_fixed(self._retry_delay_seconds),
            retry=tenacity.retry_if_not_exception_type(
                (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError),
            ),
            reraise=True,
        )
        def _call() -> object:
            return client.chat.completions.create(**request_kwargs)

        try:
            response = _call()
        except Exception as exc:  # noqa: BLE001
            timeout_error = getattr(openai, "APITimeoutError", None)
            return openai_error_result_from_exception(exc, timeout_error_type=timeout_error)
        return normalize_openai_response_with_detail(response)

    @nvtx.annotate("ImageOpenAICaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Caption images in the batch using the OpenAI-compatible API."""
        for task in tasks:
            image = task.image
            self._timer.reinit(self, task.get_major_size())
            if image.has_caption():
                continue
            with self._timer.time_process():
                try:
                    result, detail = self._generate_caption_with_error_detail(image)
                except Exception as exc:  # noqa: BLE001
                    result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
                    detail = str(exc)
                    image.errors[f"{self._model_variant}_caption"] = detail
                    if self._verbose:
                        logger.exception(f"OpenAI captioning failed for image {task.session_id}")
                    else:
                        logger.warning(f"OpenAI captioning failed for image {task.session_id}: {exc}")
                else:
                    if result.outcome == CaptionOutcome.ERROR:
                        error_msg = detail or result.failure_reason or "unknown"
                        image.errors[f"{self._model_variant}_caption"] = error_msg
                        logger.warning(f"OpenAI captioning failed for image {task.session_id}: {error_msg}")
                    elif result.outcome == CaptionOutcome.BLOCKED:
                        logger.warning(f"OpenAI captioning blocked for image {task.session_id}")
                    elif self._verbose and result.text is not None:
                        logger.info(f"OpenAI caption for image {task.session_id}: {result.text}")
            _write_caption_result(image, self._model_variant, result)
            image.model_input.pop("openai", None)
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
