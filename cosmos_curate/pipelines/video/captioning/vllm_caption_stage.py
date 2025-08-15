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
"""VLLM caption stages.

The VLLMEncodeStage and VLLMCaptionStage classes are designed to be used
in any pipeline. Because they are designed to be used in any pipeline, they
are generic and not specific to any particular pipeline or task type.

For the VLLMEncodeStage and VLLMCaptionStage to function properly, the
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
from cosmos_curate.pipelines.video.captioning.captioning_stages import _get_prompt
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Video,
    VLLMConfig,
    WindowConfig,
)

if conda_utils.is_running_in_env("unified"):
    if TYPE_CHECKING:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

    from cosmos_curate.models.vllm_interface import (
        auto_processor,
        encode_windows_for_vllm,
        free_vllm_inputs,
        sampling_params,
        vllm_caption,
        vllm_model,
    )

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


T = TypeVar("T", bound=PipelineTask)


# Map of model variants to model IDs.
_VLLM_VARIANTS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "phi4": "microsoft/Phi-4-multimodal-instruct",
}


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


class VLLMModelInterface(ModelInterface):
    """Information about a VLLM model."""

    def __init__(self, vllm_config: VLLMConfig) -> None:
        """Initialize the VLLM model interface."""
        self._vllm_config = vllm_config

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names."""
        variant = _VLLM_VARIANTS.get(self._vllm_config.variant)
        if variant is None:
            msg = f"Variant {self._vllm_config.variant} not supported"
            raise ValueError(msg)

        models = get_all_models_by_id()
        model = models.get(variant)

        if model is None:
            msg = f"Model not found for{self._vllm_config.variant} -> {variant}"
            raise ValueError(msg)

        model_id = model.get("model_id")
        if model_id is None:
            msg = f"Model ID not found for variant {self._vllm_config.variant} -> {variant}"
            raise ValueError(msg)

        return [cast("str", model_id)]

    def setup(self) -> None:
        """Set up the VLLM model interface."""


class VLLMEncodeStage(CuratorStage):
    """Stage that encodes cosmos-curate video data for VLLM multimodal model processing."""

    def __init__(
        self,
        vllm_config: VLLMConfig,
        window_config: WindowConfig,
        *,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the VLLM Preparation Stage.

        Args:
            vllm_config: Configuration for the VLLM model.
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

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=4.0)

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

    @nvtx.annotate("VLLMEncodeStage")  # type: ignore[misc]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Process the data for the VLLM caption preparation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        prompt = _get_prompt(
            self._vllm_config.prompt_variant,
            self._vllm_config.prompt_text,
            verbose=self._verbose,
        )

        for task in tasks:
            major_size = _get_major_size_task(task)
            self._timer.reinit(self, major_size)

            video = _get_video_from_task(task)
            num_video_decode_threads = max(int(self.resources.cpus), 1)

            with self._timer.time_process():
                windows, frames = windowing_utils.make_windows_for_video(
                    video,
                    self._window_config,
                    num_video_decode_threads,
                    keep_mp4=self._keep_mp4,
                )

                encode_windows_for_vllm(
                    windows,
                    frames,
                    self._vllm_config,
                    self._processor,
                    prompt,
                )

            stage_perf = getattr(task, "stage_perf", None)
            if self._log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks


class VLLMCaptionStage(CuratorStage):
    """Stage that prepares video windows for VLLM multimodal model processing.

    This stage handles the preparation of video windows and prompts for VLLM-based models.
    """

    def __init__(
        self,
        vllm_config: VLLMConfig,
        *,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the VLLM caption stage.

        Args:
            vllm_config: Configuration for the VLLM model.
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
        self._model = VLLMModelInterface(self._vllm_config)

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._llm = vllm_model(self._vllm_config)
        self._sampling_params = sampling_params(self._vllm_config)
        self._processor = auto_processor(self._vllm_config)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)

    def destroy(self) -> None:
        """Clean up GPU resources."""
        gpu_stage_cleanup(self.__class__.__name__)

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
    def model(self) -> VLLMModelInterface:
        """Get the model for this stage.

        Returns:
            The model for this stage.

        """
        return self._model

    def free_unused(self, tasks: list[T]) -> None:
        """Free unused memory, if enabled.

        Args:
            tasks: The tasks to process.

        """
        for task in tasks:
            video = _get_video_from_task(task)
            free_vllm_inputs(video, self._vllm_config.variant)

            if not self._keep_mp4:
                for clip in video.clips:
                    for window in clip.windows:
                        window.mp4_bytes = None

    @nvtx.annotate("VLLMCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Process the data for the VLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._llm is None:
            msg = "VLLM model not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._sampling_params is None:
            msg = "Sampling parameters not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._processor is None:
            msg = "Processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        major_size = _get_major_size_tasks(tasks)
        self._timer.reinit(self, major_size)
        videos = [_get_video_from_task(task) for task in tasks]

        with self._timer.time_process():
            num_captions = vllm_caption(
                videos,
                self._llm,
                self._processor,
                self._vllm_config,
                self._sampling_params,
                use_tqdm=self._vllm_use_tqdm,
            )

        if self._verbose:
            logger.info(f"Generated {num_captions} captions for {len(tasks)} tasks")

        if self._log_stats:
            # Because there's a single call to caption all tasks, just log the first task's stage_perf.
            stage_name, stage_perf_stats = self._timer.log_stats()
            stage_perf = getattr(tasks[0], "stage_perf", None)
            if stage_perf is not None:
                stage_perf[stage_name] = stage_perf_stats

        self.free_unused(tasks)
        return tasks
