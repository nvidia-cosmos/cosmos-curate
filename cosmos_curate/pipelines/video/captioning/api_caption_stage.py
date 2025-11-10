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
"""Remote API-backed caption preparation and captioning stages (Gemini)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curate.core.utils.config.config import load_config
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models.prompts import get_prompt
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import get_video_from_task

if TYPE_CHECKING:
    from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video, Window, WindowConfig

from google import genai
from google.genai import types as genai_types

TTask = TypeVar("TTask", bound=PipelineTask)


class ApiPrepStage(CuratorStage):
    """Stage that prepares windows for remote API captioning."""

    def __init__(
        self,
        window_config: WindowConfig,
        *,
        model_variant: str = "gemini",
        num_cpus_for_prepare: float = 1.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the API prep stage."""
        super().__init__()
        self._timer = StageTimer(self)
        self._window_config = window_config
        self._model_variant = model_variant
        self._num_cpus_for_prepare = num_cpus_for_prepare
        self._verbose = verbose
        self._log_stats = log_stats

    def secondary_name(self) -> str:
        """Return the model variant for logging."""
        return self._model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=self._num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment for window preparation."""
        return "unified"

    def _prep_windows(self, video: Video) -> None:
        """Create windows for the provided video."""
        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)
        windows, _ = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=True,
            return_frames=False,
        )
        if self._verbose:
            logger.debug(f"Prepared {len(windows)} windows for {video.input_video}")

    @nvtx.annotate("ApiPrepStage")  # type: ignore[misc]
    def process_data(self, tasks: list[TTask]) -> list[TTask]:
        """Prepare data for API captioning."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            video = get_video_from_task(task)
            with self._timer.time_process():
                self._prep_windows(video)

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


class NonRetryableGeminiError(RuntimeError):
    """Error raised when retrying a Gemini request will not succeed."""


class ApiCaptionStage(CuratorStage):
    """Caption video windows using the Google Gemini API.

    The Gemini API key must be provided in the cosmos-curate config file under the `gemini` section.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_variant: str = "gemini",
        model_name: str = "models/gemini-2.5-pro",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        max_output_tokens: int = 4096,
        max_caption_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        max_video_size_bytes: int = 20 * 1024 * 1024,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the API caption stage.

        Args:
            model_variant: Identifier stored alongside generated captions.
            model_name: Gemini model name to invoke.
            prompt_variant: Prompt variant used to build caption instructions.
            prompt_text: Optional custom prompt text.
            max_output_tokens: Maximum output tokens requested from the API.
            max_caption_retries: Number of retries per window before giving up.
            retry_delay_seconds: Delay between retries.
            max_video_size_bytes: Maximum inline video size supported by the API.
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._model_name = model_name
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._prompt = get_prompt(prompt_variant, prompt_text, verbose=verbose)
        self._max_output_tokens = max_output_tokens
        self._max_caption_retries = max_caption_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_video_size_bytes = max_video_size_bytes
        self._verbose = verbose
        self._log_stats = log_stats
        config = load_config()
        if config.gemini is None or not config.gemini.api_key:
            msg = "Gemini API key missing from config file."
            raise RuntimeError(msg)
        self._api_key = config.gemini.api_key
        self._client: genai.Client | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    def stage_setup(self) -> None:
        """Create the Gemini API client."""
        self._client = genai.Client(api_key=self._api_key)

    @staticmethod
    def _extract_text(response: object) -> str:
        """Extract plain text from a Gemini response object."""
        # Check if the prompt was blocked
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            reason_str = str(block_reason) if not isinstance(block_reason, str) else block_reason
            msg = f"Gemini request blocked: {reason_str}"
            raise NonRetryableGeminiError(msg)

        # Try direct text access first (most common case)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Fall back to candidates structure
        candidates = getattr(response, "candidates", None)
        if not candidates:
            msg = "Gemini response does not contain text"
            raise RuntimeError(msg)

        # Collect text from all candidates
        collected: list[str] = []
        finish_reasons: list[str] = []
        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                finish_reasons.append(str(finish_reason))

            content = getattr(candidate, "content", None) or candidate
            parts = getattr(content, "parts", None)
            if parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        collected.append(part_text.strip())

        result = "\n".join(collected).strip()
        if result:
            return result

        # No text found - include finish reasons in error if available
        detail = f" (finish_reasons={sorted(set(finish_reasons))})" if finish_reasons else ""
        msg = f"Gemini response parts are empty{detail}"
        raise RuntimeError(msg)

    @staticmethod
    def _handle_client_exception(exc: BaseException) -> BaseException:
        """Wrap Gemini client exceptions with clearer messaging when appropriate."""
        message = str(exc).lower()
        if "api key not found" in message or "api_key_invalid" in message:
            msg = (
                "Gemini rejected the API key (API_KEY_INVALID). "
                "Update the cosmos-curate config with a valid Gemini API key."
            )
            return NonRetryableGeminiError(msg)
        return exc

    @staticmethod
    def _should_retry_exception(exc: BaseException) -> bool:
        """Decide whether the Gemini request should be retried."""
        # Don't retry if we've wrapped it as NonRetryableGeminiError
        return not isinstance(exc, NonRetryableGeminiError)

    def _generate_caption(self, window: Window, _clip_index: int, _window_index: int) -> str:
        """Generate a caption for a single window with retry logic."""
        client = self._client
        if client is None:
            msg = "Gemini client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        instruction = self._prompt.strip()
        inline_data = genai_types.Blob(data=window.mp4_bytes, mime_type="video/mp4")
        content = genai_types.Content(
            parts=[
                genai_types.Part(inline_data=inline_data),
                genai_types.Part(text=instruction),
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
            retry=tenacity.retry_if_exception(ApiCaptionStage._should_retry_exception),
            reraise=True,
        )
        def _call() -> str:
            try:
                response = client.models.generate_content(**generate_kwargs)
            except Exception as exc:
                new_exc = ApiCaptionStage._handle_client_exception(exc)
                if new_exc is exc:
                    raise
                raise new_exc from exc
            return self._extract_text(response)

        return _call()

    def _validate_window(self, window: Window) -> None:
        """Validate that the window contains data suitable for Gemini."""
        if window.mp4_bytes is None:
            msg = "Window missing mp4 bytes; enable keep_mp4 in the prep stage."
            raise RuntimeError(msg)
        if len(window.mp4_bytes) > self._max_video_size_bytes:
            size_mb = len(window.mp4_bytes) / (1024 * 1024)
            max_mb = self._max_video_size_bytes / (1024 * 1024)
            msg = f"Window MP4 ({size_mb:.2f} MB) exceeds Gemini inline limit ({max_mb:.2f} MB)."
            raise RuntimeError(msg)

    def _process_task(self, task: SplitPipeTask) -> None:
        """Process a single SplitPipeTask and populate captions."""
        for clip_index, clip in enumerate(task.video.clips):
            for window_index, window in enumerate(clip.windows):
                try:
                    self._validate_window(window)
                    caption = self._generate_caption(window, clip_index, window_index)
                except Exception as exc:  # noqa: BLE001
                    clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
                    if self._verbose:
                        logger.exception(f"Gemini captioning failed for clip {clip.uuid} window {window_index}")
                    else:
                        logger.warning(f"Gemini captioning failed for clip {clip.uuid} window {window_index}: {exc}")
                    continue
                window.caption[self._model_variant] = caption
                if self._verbose:
                    logger.info(f"Gemini caption clip {clip.uuid} window {window_index}: {caption}")

    @nvtx.annotate("ApiCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Caption each window in the provided tasks using Gemini."""
        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            with self._timer.time_process():
                self._process_task(task)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
