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
from typing import TYPE_CHECKING, Any

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.config import maybe_load_config
from cosmos_curate.core.utils.environment import CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.prompts import get_prompt
from cosmos_curate.pipelines.video.utils.data_model import (
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
        if config is None or config.openai is None or not config.openai.api_key:
            error_msg = (
                "OpenAI configuration not found. Provide openai.api_key in "
                f"{CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE}"
            )
            raise RuntimeError(error_msg)

        client_kwargs: dict[str, Any] = {"api_key": config.openai.api_key}
        if config.openai.base_url:
            client_kwargs["base_url"] = config.openai.base_url
        self._client = openai.OpenAI(**client_kwargs)

    def _generate_caption(self, window: Window) -> str:
        """Generate a caption for a single window with retry logic."""
        client = self._client
        if client is None:
            msg = "OpenAI client not initialized; call stage_setup before generating captions."
            raise RuntimeError(msg)

        mp4_data = window.mp4_bytes.resolve()
        if mp4_data is None:
            msg = "Window missing mp4 bytes; enable keep_mp4 in the prep stage."
            raise RuntimeError(msg)

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
        def _call() -> str:
            response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                msg = f"OpenAI-compatible API returned no choices (possible content filter). model={self._model_name!r}"
                raise RuntimeError(msg)
            choice = response.choices[0]
            text: str | None = choice.message.content
            if not text or not text.strip():
                msg = (
                    f"OpenAI-compatible API returned empty caption."
                    f" finish_reason={choice.finish_reason!r},"
                    f" content={choice.message.content!r}"
                )
                raise RuntimeError(msg)
            result: str = text.strip()
            return result

        return _call()

    def _process_task(self, task: SplitPipeTask) -> None:
        """Process a single SplitPipeTask and populate captions."""
        for clip in task.video.clips:
            for window_index, window in enumerate(clip.windows):
                try:
                    caption = self._generate_caption(window)
                except Exception as exc:  # noqa: BLE001
                    clip.errors[f"{self._model_variant}_caption_{window_index}"] = str(exc)
                    if self._verbose:
                        logger.exception(f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}")
                    else:
                        logger.warning(
                            f"OpenAI API captioning failed for clip {clip.uuid} window {window_index}: {exc}"
                        )
                    continue
                window.caption[self._model_variant] = caption
                if self._verbose:
                    logger.info(f"OpenAI API caption clip {clip.uuid} window {window_index}: {caption}")

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
