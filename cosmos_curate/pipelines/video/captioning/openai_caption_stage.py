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
"""OpenAI-compatible API captioning stage for remote VLM inference (e.g. vLLM serving)."""

import base64
from typing import TYPE_CHECKING, Any, Literal

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.config import maybe_load_config, resolve_model_name_auto
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.prompts import get_prompt
from cosmos_curate.pipelines.video.utils.data_model import (
    CaptionFailureReason,
    CaptionOutcome,
    CaptionResult,
    SplitPipeTask,
    Window,
)

if TYPE_CHECKING:
    import openai


if conda_utils.is_running_in_env("unified"):
    import openai


class OpenAICaptionStage(CuratorStage):
    """Caption video windows using an OpenAI-compatible vision API.

    Sends each window's MP4 bytes as a base64-encoded video to a remote
    OpenAI-compatible endpoint (e.g. vLLM serving a VLM) and stores the
    returned caption.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        model_variant: str = "openai",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        max_output_tokens: int = 8192,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        use_filter_windows: bool = False,
        endpoint_key: Literal["caption", "enhance", "embedding", "filter", "classifier"] = "caption",
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the OpenAI-compatible API caption stage.

        Args:
            model_name: Model name to pass in the API request.
            model_variant: Identifier stored alongside generated captions.
            prompt_variant: Prompt variant used to build caption instructions.
            prompt_text: Optional custom prompt text.
            max_output_tokens: Maximum output tokens requested from the API.
            max_caption_retries: Number of retries per window before giving up.
            retry_delay_seconds: Delay between retries.
            use_filter_windows: If True, iterate clip.filter_windows instead of clip.windows.
            endpoint_key: Key under config.openai to read credentials from; must be one of the
                fields defined on OpenAIConfig ("caption", "enhance", "embedding", "filter", "classifier").
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_name = model_name
        self._model_variant = model_variant
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._use_filter_windows = use_filter_windows
        self._endpoint_key = endpoint_key
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: openai.OpenAI | None = None

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment (openai package lives there)."""
        return "unified"

    def stage_setup(self) -> None:
        """Create the OpenAI API client using credentials from the config file."""
        config = maybe_load_config()
        endpoint = (
            getattr(config.openai, self._endpoint_key, None)
            if config is not None and config.openai is not None
            else None
        )
        if endpoint is None or not endpoint.api_key:
            error_msg = (
                f"OpenAI {self._endpoint_key} configuration not found. "
                f"Provide openai.{self._endpoint_key}.api_key in ~/.config/cosmos_curate/config.yaml"
            )
            raise RuntimeError(error_msg)

        client_kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
        if endpoint.base_url:
            client_kwargs["base_url"] = endpoint.base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._model_name = resolve_model_name_auto(
            self._client, self._model_name, endpoint_label=f"OpenAI {self._endpoint_key}"
        )

    @staticmethod
    def _error_result_from_exception_with_detail(exc: BaseException) -> tuple[CaptionResult, str]:
        """Map an OpenAI exception to an error result and log detail."""
        timeout_error = getattr(openai, "APITimeoutError", None)
        failure_reason: CaptionFailureReason = (
            "timeout" if timeout_error is not None and isinstance(exc, timeout_error) else "exception"
        )
        return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason=failure_reason), str(exc)

    @staticmethod
    def _write_caption_result(window: Window, model_variant: str, result: CaptionResult) -> None:
        """Write an OpenAI caption result onto a window."""
        if result.text is not None:
            window.caption[model_variant] = result.text
        window.caption_status = result.outcome.value
        window.caption_failure_reason = result.failure_reason if result.outcome == CaptionOutcome.ERROR else None

    @staticmethod
    def _normalize_response_with_detail(response: object) -> tuple[CaptionResult, str | None]:
        """Map an OpenAI chat completion response to a caption result."""
        choices = getattr(response, "choices", None)
        if not choices:
            return (
                CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
                "OpenAI-compatible API returned no choices.",
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = getattr(choice.message, "content", None)
        text = content.strip() if isinstance(content, str) and content.strip() else None

        if finish_reason == "content_filter":
            return CaptionResult(outcome=CaptionOutcome.BLOCKED), None
        if content is None:
            return (
                CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
                f"OpenAI-compatible API returned null content. finish_reason={finish_reason!r}",
            )
        if finish_reason == "length":
            outcome = CaptionOutcome.TRUNCATED if text is not None else CaptionOutcome.ERROR
            failure_reason: CaptionFailureReason | None = None if text is not None else "exception"
            detail = None
            if text is None:
                detail = (
                    "OpenAI-compatible API returned no caption text after truncation. "
                    f"finish_reason={finish_reason!r}, content={content!r}"
                )
            return CaptionResult(outcome=outcome, text=text, failure_reason=failure_reason), detail
        if text is not None:
            return CaptionResult(outcome=CaptionOutcome.SUCCESS, text=text), None
        return (
            CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
            f"OpenAI-compatible API returned empty caption text. finish_reason={finish_reason!r}, content={content!r}",
        )

    @staticmethod
    def _normalize_response(response: object) -> CaptionResult:
        """Map an OpenAI chat completion response to a caption result."""
        result, _detail = OpenAICaptionStage._normalize_response_with_detail(response)
        return result

    def _generate_caption_with_error_detail(self, window: Window) -> tuple[CaptionResult, str | None]:
        """Generate a caption result for a single window with error detail when available."""
        client = self._client
        if client is None:
            msg = "OpenAI client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            return (
                CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
                "Window missing mp4 bytes; enable keep_mp4 in the prep stage.",
            )

        video_b64 = base64.b64encode(bytes(mp4_data)).decode("utf-8")
        instruction = self._prompt.strip()

        content_parts: list[dict[str, Any]] = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": instruction},
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
            return self._error_result_from_exception_with_detail(exc)
        return self._normalize_response_with_detail(response)

    def _generate_caption(self, window: Window) -> CaptionResult:
        """Generate a caption result for a single window."""
        result, _detail = self._generate_caption_with_error_detail(window)
        return result

    def _process_task(self, task: SplitPipeTask) -> None:
        """Process a single SplitPipeTask and populate captions."""
        for clip in task.video.clips:
            window_source = clip.filter_windows if self._use_filter_windows else clip.windows
            for window_index, window in enumerate(window_source):
                try:
                    result, error_detail = self._generate_caption_with_error_detail(window)
                except Exception as exc:  # noqa: BLE001
                    result = CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception")
                    clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
                    if self._verbose:
                        logger.exception(f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}")
                    else:
                        logger.warning(
                            f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}: {exc}"
                        )
                else:
                    if result.outcome == CaptionOutcome.ERROR:
                        clip.errors[f"{self._model_variant}_caption_{window_index}"] = error_detail or (
                            f"OpenAI API captioning failed: {result.failure_reason}"
                        )
                        logger.warning(
                            f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}: "
                            f"{clip.errors[f'{self._model_variant}_caption_{window_index}']}"
                        )
                    elif result.outcome == CaptionOutcome.BLOCKED:
                        logger.warning(f"OpenAI API captioning blocked for clip {clip.uuid} window {window_index}")
                    elif self._verbose and result.text is not None:
                        logger.info(f"OpenAI API caption clip {clip.uuid} window {window_index}: {result.text}")
                self._write_caption_result(window, self._model_variant, result)

    @nvtx.annotate("OpenAICaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Caption each window in the provided tasks using the OpenAI-compatible API."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            with self._timer.time_process():
                self._process_task(task)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
