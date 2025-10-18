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
"""vLLM caption stages.

The VllmPrepStage and VllmCaptionStage classes are designed to be used
in any pipeline. Because they are designed to be used in any pipeline, they
are generic and not specific to any particular pipeline or task type.

For the VllmPrepStage and VllmCaptionStage to function properly, the
the tasks must have these attributes/methods:

- video: The video to process.
- stage_perf: A dictionary to store performance statistics.
- get_major_size: A method to get the major size of the task.

"""

import logging
from typing import TYPE_CHECKING, TypeVar, cast

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.all_models import get_all_models_by_id
from cosmos_curate.models.prompts import get_prompt
from cosmos_curate.models.vllm_model_ids import get_vllm_model_id
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Video,
    VllmConfig,
    Window,
    WindowConfig,
)

if TYPE_CHECKING:
    from vllm import SamplingParams

if conda_utils.is_running_in_env("unified"):
    if TYPE_CHECKING:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

    from cosmos_curate.models.vllm_interface import (
        auto_processor,
        make_model_inputs,
        sampling_params,
        vllm_caption,
        vllm_model,
    )

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


T = TypeVar("T", bound=PipelineTask)


def _get_major_size_task(task: T) -> int:
    get_major_size = getattr(task, "get_major_size", None)
    if callable(get_major_size):
        return cast("int", get_major_size())
    msg = f"{type(task)} does not have a callable `get_major_size()` method"
    raise RuntimeError(msg)


def _get_major_size_tasks(tasks: list[T]) -> int:
    """Get the major size of the tasks.

    Args:
        tasks: The tasks to get the major size of.

    Returns:
        The major size of the tasks.

    Raises:
        RuntimeError: If the task does not have a callable `get_major_size()` method.

    """
    return sum(_get_major_size_task(t) for t in tasks)


def _get_video_from_task(task: T) -> Video:
    """Get the video from a task.

    Args:
        task: The task.

    Returns:
        The video from the task.

    Raises:
        TypeError: If the task does not have a video attribute.

    """
    video = getattr(task, "video", None)
    if not isinstance(video, Video):
        msg = f"task.video type={type(video)}, expected `Video`"
        raise TypeError(msg)
    return video


def _get_windows_from_tasks(tasks: list[T]) -> tuple[list[Window], list[str]]:
    """Get the windows from a list of tasks.

    Args:
        tasks: The tasks with video -> clips -> windows.

    Returns:
        The windows and clip uuids from the task.

    Raises:
        TypeError: If the task does not have a video attribute.

    """
    windows: list[Window] = []
    clip_uuids: list[str] = []
    for task in tasks:
        video = _get_video_from_task(task)
        for clip in video.clips:
            if not clip.windows:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["clip_windowing"] = "empty"
                continue
            windows += clip.windows
            clip_uuids += [str(clip.uuid)] * len(clip.windows)

    return windows, clip_uuids


def _free_vllm_inputs(windows: list[Window], model_variant: str, *, keep_mp4: bool = False) -> None:
    """Free unused memory for the model variant.

    Args:
        windows: The windows to free unused memory for.
        model_variant: The variant of the model.
        keep_mp4: Whether to keep the mp4 bytes.

    """
    for window in windows:
        window.model_input.pop(model_variant, None)
        if not keep_mp4:
            window.mp4_bytes = None


class VllmModelInterface(ModelInterface):
    """Information about a vLLM model."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        """Initialize the vLLM model interface."""
        self._vllm_config = vllm_config

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names."""
        model_variant = get_vllm_model_id(self._vllm_config.model_variant)
        models = get_all_models_by_id()
        model = models.get(model_variant)

        if model is None:
            msg = f"Model not found for {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        model_id = model.get("model_id")
        if model_id is None:
            msg = f"Model ID not found for variant {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        return [cast("str", model_id)]

    def setup(self) -> None:
        """Set up the vLLM model interface."""


class VllmPrepStage(CuratorStage):
    """Stage that prepares cosmos-curate video data for vLLM multimodal model processing."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        window_config: WindowConfig,
        *,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the vLLM Preparation Stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            window_config: Configuration for the windowing.
            keep_mp4: Keep mp4 bytes for the clips in memory.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._window_config = window_config
        self._verbose = verbose
        self._log_stats = log_stats
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        return self._vllm_config.model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=self._vllm_config.num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        self._processor = auto_processor(self._vllm_config)

    def _prep_windows(self, video: Video, prompt: str) -> None:
        """Prep the windows for the vLLM model.

        The videos are modified in-place by creating the windows
        for each clip in the videos and adding windows to each clip.

        Model inputs are added to each window.

        Args:
            video: The video to prep the windows for.
            prompt: The prompt to use for the vLLM model.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)

        windows, frames = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=self._keep_mp4,
        )

        llm_inputs = make_model_inputs(frames, self._vllm_config, self._processor, prompt)

        for window, llm_input in zip(windows, llm_inputs, strict=True):
            window.model_input[self._vllm_config.model_variant] = llm_input

    @nvtx.annotate("VllmPrepStage")  # type: ignore[misc]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Prepare the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        prompt = get_prompt(
            self._vllm_config.prompt_variant,
            self._vllm_config.prompt_text,
            verbose=self._verbose,
        )

        for task in tasks:
            major_size = _get_major_size_task(task)
            self._timer.reinit(self, major_size)

            video = _get_video_from_task(task)

            with self._timer.time_process():
                self._prep_windows(video, prompt)

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


class VllmCaptionStage(CuratorStage):
    """Stage that prepares video windows for vLLM multimodal model processing.

    This stage handles the preparation of video windows and prompts for vLLM-based models.
    """

    def __init__(  # noqa: PLR0913
        self,
        vllm_config: VllmConfig,
        max_inflight_requests: int = 0,
        *,
        inflight_batching: bool = False,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the vLLM caption stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            max_inflight_requests: Maximum number of inflight requests to vLLM
               engine. Set to 0 for unlimited inflight requests. Ignored if
               inflight_batching is False.
            inflight_batching: set to True to enable inflight batching.
            keep_mp4: Whether to keep the mp4 bytes.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4
        self._verbose = verbose
        self._log_stats = log_stats
        self._vllm_use_tqdm = False
        self._model = VllmModelInterface(self._vllm_config)
        self._max_inflight_requests = max_inflight_requests
        self._inflight_batching = inflight_batching

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._llm = vllm_model(self._vllm_config)
        self._sampling_params = sampling_params(self._vllm_config)
        self._processor = auto_processor(self._vllm_config)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up GPU resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        return self._vllm_config.model_variant

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=self._vllm_config.num_gpus)

    @property
    def model(self) -> VllmModelInterface:
        """Get the model for this stage.

        Returns:
            The model for this stage.

        """
        return self._model

    @nvtx.annotate("VllmCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Process the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._llm is None:
            msg = "vLLM model not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._sampling_params is None:
            msg = "Sampling parameters not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._processor is None:
            msg = "Processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        major_size = _get_major_size_tasks(tasks)
        self._timer.reinit(self, major_size)

        with self._timer.time_process():
            # Gather model inputs and clip uuids
            windows, clip_uuids = _get_windows_from_tasks(tasks)
            model_inputs = [window.model_input[self._vllm_config.model_variant] for window in windows]

            # Generate captions
            captions = vllm_caption(
                model_inputs,
                self._llm,
                self._processor,
                self._sampling_params,
                self._vllm_config,
                inflight_batching=self._inflight_batching,
                max_inflight_requests=self._max_inflight_requests,
            )

            # Scatter captions back to windows
            for window, caption, clip_uuid in zip(windows, captions, clip_uuids, strict=True):
                window.caption[self._vllm_config.model_variant] = caption
                if self._verbose:
                    logger.info(f"Caption for clip {clip_uuid}: {caption}")

            logger.info(f"Generated {len(captions)} captions for {len(tasks)} tasks")

        if self._log_stats:
            # Because there's a single call to caption all tasks, just log the first task's stage_perf.
            stage_name, stage_perf_stats = self._timer.log_stats()
            stage_perf = getattr(tasks[0], "stage_perf", None)
            if stage_perf is not None:
                stage_perf[stage_name] = stage_perf_stats

        _free_vllm_inputs(windows, self._vllm_config.model_variant, keep_mp4=self._keep_mp4)
        return tasks
