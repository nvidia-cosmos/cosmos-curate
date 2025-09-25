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
"""Captioning stage."""

import asyncio
import math
from collections.abc import Iterable
from itertools import zip_longest

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config import operation_context
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models import (
    cosmos_reason1_vl,
    phi_vl,
    qwen_vl,
    t5_encoder,
)
from cosmos_curate.models.chat_lm import ChatLM
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    ShardPipeTask,
    SplitPipeTask,
    Video,
    Window,
)
from cosmos_curate.pipelines.video.utils.decoder_utils import DEFAULT_TRANSCODE_BITRATE_M

_PROMPTS = {
    "default": """
        Elaborate on the visual and narrative elements of the video in detail.
    """,
    "av": """
        The video depicts the view from a camera mounted on a car as it is driving.
        Pay special attention to the motion of the cars, including the primary car
        whose point-of-view we observe in the video. Also note important factors
        that would relate to driving safety like the relative positions of pedestrians,
        lane markers, road signs, traffic signals, and any aggressive driving behavior
        of other vehicles. Also pay attention to interesting landmarks and describe
        them in detail.
    """,
    "av-surveillance": """
        The video depicts the view from a surveillance camera. Pay special attention
        to the motion of the cars and other important factors that would relate to
        driving safety like the relative positions of pedestrians, lane markers,
        road signs, traffic signals, and any aggressive driving behavior of vehicles.
        Also pay attention to interesting landmarks and describe them in detail.
    """,
}

_ENHANCE_PROMPTS = {
    "default": """
        You are a chatbot that enhances video caption inputs, adding more color and details to the text.
        The output should be longer than the provided input caption.
    """,
    "av-surveillance": """
        You are a chatbot that enhances video captions from vehicle dashboard cameras or surveillance cameras.
        Add more details and generate a summary from the original text.
        The output should be longer than the provided input caption.
    """,
}


# Use with Captioning Stage
def _get_prompt(
    prompt_variant: str,
    prompt_text: str | None,
    *,
    verbose: bool = False,
) -> str:
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Captioning prompt: {prompt}")
    return prompt


def _get_enhance_prompt(prompt_variant: str, prompt_text: str | None, *, verbose: bool = False) -> str:
    if prompt_text is not None:
        prompt = prompt_text
    else:
        if prompt_variant not in _ENHANCE_PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _ENHANCE_PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Enhance Captioning prompt: {prompt}")
    return prompt


def _assign_captions(  # noqa: PLR0913
    video: Video,
    mapping: dict[int, tuple[int, int]],
    captions: Iterable[tuple[int, str]],
    model_variant: str,
    *,
    keep_mp4_bytes: bool,
    verbose: bool,
) -> None:
    _captions = list(captions)
    for req_id, caption in _captions:
        clip_idx, window_idx = mapping[req_id]
        video.clips[clip_idx].windows[window_idx].caption[model_variant] = caption
        if verbose:
            logger.info(f"Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {caption}")

    logger.info(
        f"Generated {len(_captions)} captions for video {video.input_path} "
        f"chunk-{video.clip_chunk_index} with {len(video.clips)} clips",
    )

    for clip in video.clips:
        for window in clip.windows:
            window.qwen_llm_input = None
            window.cosmos_reason1_llm_input = None
            window.phi_llm_input = None
            if not keep_mp4_bytes:
                window.mp4_bytes = None


# utilities common for different captioning stages
def _handle_empty_clip_buffer(clip: Clip) -> None:
    logger.warning(f"Clip {clip.uuid} has no buffer.")
    clip.errors["buffer"] = "empty"


def _handle_empty_clip_windows(clip: Clip) -> None:
    logger.warning(f"Clip {clip.uuid} has no windows.")
    clip.errors["windows"] = "empty"


def _handle_empty_llm_inputs(clip: Clip, window_idx: int) -> None:
    logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
    clip.errors[f"window-{window_idx}"] = "empty"


def _get_target_bit_rate(video: Video, *, use_input_bit_rate: bool) -> str:
    return f"{video.metadata.bit_rate_k}K" if use_input_bit_rate else f"{DEFAULT_TRANSCODE_BITRATE_M}M"


class WindowingStage(CuratorStage):
    """Stage that splits video clips into fixed-size windows for processing.

    This stage handles the windowing of video clips into smaller segments based on
    specified window size and remainder threshold parameters.
    """

    def __init__(
        self,
        window_size: int = 256,
        remainder_threshold: int = 128,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the windowing stage.

        Args:
            window_size: Size of each window in frames.
            remainder_threshold: Minimum frames required for a remainder window.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._window_size = window_size
        self._remainder_threshold = remainder_threshold
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=4.0)

    @nvtx.annotate("WindowingStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the windowing stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.encoded_data is None:
                    logger.warning(f"Clip {clip.uuid} has no buffer.")
                    clip.errors["buffer"] = "empty"
                    continue
                with self._timer.time_process():
                    for window_mp4_bytes, _, window_frame_info in zip(
                        *windowing_utils.split_video_into_windows(
                            clip.encoded_data,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            num_threads=max(int(self.resources.cpus), 1),
                            return_bytes=True,
                            return_video_frames=False,
                        ),
                        strict=True,
                    ):
                        clip.windows.append(
                            Window(
                                window_frame_info.start,
                                window_frame_info.end,
                                mp4_bytes=window_mp4_bytes,
                            ),
                        )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class QwenInputPreparationStage(CuratorStage):
    """Stage that prepares video windows for Qwen model processing.

    This stage handles the preparation of video windows and prompts for the Qwen vision-language
    model, including frame sampling, preprocessing, and input formatting.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        sampling_fps: float = 2.0,
        window_size: int = 256,
        remainder_threshold: int = 128,
        preprocess_dtype: str = "float32",
        *,
        model_does_preprocess: bool = False,
        generate_previews: bool = True,
        prepare_cosmos_predict_dataset: bool = False,
        use_input_bit_rate: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Qwen input preparation stage.

        Args:
            model_variant: Name of the model variant to use.
            prompt_variant: Type of prompt to use.
            prompt_text: Custom prompt text if provided.
            sampling_fps: Frames per second for sampling.
            window_size: Size of each window in frames.
            remainder_threshold: Minimum frames required for a remainder window.
            preprocess_dtype: Data type for preprocessing.
            model_does_preprocess: Whether model handles preprocessing.
            generate_previews: Whether to generate previews.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            use_input_bit_rate: Whether to use the input video's bit rate for processing.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats
        self._qwen_utils = qwen_vl.QwenUtils(model_variant)
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._sampling_fps = sampling_fps
        self._window_size = window_size
        self._remainder_threshold = remainder_threshold
        self._preprocess_dtype = preprocess_dtype
        self._model_does_preprocess = model_does_preprocess
        self._generate_previews = generate_previews
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._use_input_bit_rate = use_input_bit_rate

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
        """Initialize stage resources and configuration."""
        self._qwen_utils.setup()

    @nvtx.annotate("QwenInputPreparationStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Qwen input preparation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.encoded_data is None:
                    _handle_empty_clip_buffer(clip)
                    continue
                with self._timer.time_process():
                    for window_bytes, window_frames, window_frame_info in zip_longest(
                        *windowing_utils.split_video_into_windows(
                            clip.encoded_data,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            sampling_fps=self._sampling_fps,
                            model_does_preprocess=self._model_does_preprocess,
                            preprocess_dtype=self._preprocess_dtype,
                            return_bytes=(self._generate_previews or self._prepare_cosmos_predict_dataset),
                            target_bit_rate=_get_target_bit_rate(video, use_input_bit_rate=self._use_input_bit_rate),
                            num_threads=max(int(self.resources.cpus), 1),
                        ),
                    ):
                        prompt = _get_prompt(
                            self._prompt_variant,
                            self._prompt_text,
                            verbose=self._verbose,
                        )
                        try:
                            llm_input = self._qwen_utils.generate_llm_inputs(
                                prompt=prompt,
                                video_inputs=window_frames,
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Error in Qwen input preparation: {e}")
                            clip.errors["qwen_input"] = str(e)
                        else:
                            clip.windows.append(
                                Window(
                                    window_frame_info.start,
                                    window_frame_info.end,
                                    mp4_bytes=window_bytes,
                                    qwen_llm_input=llm_input,
                                ),
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class QwenCaptionStage(CuratorStage):
    """Stage that generates captions for video windows using the Qwen model.

    This stage processes prepared video windows through the Qwen vision-language model to
    generate descriptive captions, with support for both synchronous and asynchronous processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        batch_size: int = 16,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,
        generate_stage2_caption: bool = False,
        stage2_prompt_text: str | None = None,
        disable_mmcache: bool = False,
        use_async_engine: bool = False,
        num_gpus_per_worker: float = 1.0,
        prepare_cosmos_predict_dataset: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Qwen caption generation stage.

        Args:
            model_variant: Name of the model variant to use.
            batch_size: Number of samples to process in parallel.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            model_does_preprocess: Whether model handles preprocessing.
            generate_stage2_caption: Whether to generate second stage captions.
            stage2_prompt_text: Custom prompt for second stage.
            disable_mmcache: Whether to disable model cache.
            use_async_engine: Whether to use the asynchronous engine for processing.
            num_gpus_per_worker: Number of GPUs to allocate per worker.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._batch_size = batch_size
        self._generate_stage2_caption = generate_stage2_caption
        self._disable_mmcache = disable_mmcache
        self._use_async_engine = use_async_engine
        self._num_gpus_per_worker = num_gpus_per_worker
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._verbose = verbose
        self._log_stats = log_stats
        self._raw_model = qwen_vl.QwenVL(
            model_variant,
            fp8=fp8_enable,
            max_output_tokens=max_output_tokens,
            model_does_preprocess=model_does_preprocess,
            stage2_prompt_text=stage2_prompt_text,
            disable_mmcache=self._disable_mmcache,
            use_async_engine=self._use_async_engine,
            num_gpus=math.ceil(self._num_gpus_per_worker),
        )

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._raw_model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._raw_model

    def process_data_sync(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process the data for the Qwen caption generation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            inputs = []
            mapping: dict[int, tuple[int, int]] = {}
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        _handle_empty_clip_windows(clip)
                    for window_idx, window in enumerate(clip.windows):
                        if window.qwen_llm_input is None:
                            _handle_empty_llm_inputs(clip, window_idx)
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        inputs.append(window.qwen_llm_input)
                        idx += 1

                captions = self._raw_model.generate(
                    inputs,
                    generate_stage2_caption=self._generate_stage2_caption,
                    batch_size=self._batch_size,
                )

                _assign_captions(
                    video,
                    mapping,
                    enumerate(captions),
                    self._model_variant,
                    keep_mp4_bytes=self._prepare_cosmos_predict_dataset,
                    verbose=self._verbose,
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    async def _process_data_async(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            input_req_id = 0
            vllm_tasks = []
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        _handle_empty_clip_windows(clip)
                    for window_idx, window in enumerate(clip.windows):
                        if window.qwen_llm_input is None:
                            _handle_empty_llm_inputs(clip, window_idx)
                            continue
                        mapping[input_req_id] = (clip_idx, window_idx)
                        vllm_task = asyncio.create_task(
                            self._raw_model.generate_async(
                                window.qwen_llm_input,
                                input_req_id,
                                generate_stage2_caption=self._generate_stage2_caption,
                            ),
                        )
                        vllm_tasks.append(vllm_task)
                        input_req_id += 1

                # Wait for all VLLM tasks to complete
                vllm_captions = await asyncio.gather(*vllm_tasks)
                # Assign captions to the corresponding clip/window
                _assign_captions(
                    video,
                    mapping,
                    vllm_captions,
                    self._model_variant,
                    keep_mp4_bytes=self._prepare_cosmos_predict_dataset,
                    verbose=self._verbose,
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    def get_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        """Get the asyncio event loop.

        Returns:
            The asyncio event loop.

        """
        try:
            asyncio_loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio_loop)
        return asyncio_loop

    @nvtx.annotate("QwenCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Qwen caption generation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._use_async_engine:
            asyncio_loop = self.get_asyncio_loop()
            return asyncio_loop.run_until_complete(self._process_data_async(tasks))
        return self.process_data_sync(tasks)


class CosmosReason1InputPreparationStage(CuratorStage):
    """Stage that prepares video inputs for the Cosmos-Reason1 vision-language model.

    This stage processes video clips and prepares them for Cosmos-Reason1 caption generation by
    extracting frames, applying preprocessing, and creating structured inputs for the model.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "cosmos_r1",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        sampling_fps: float = 4.0,
        window_size: int = 256,
        remainder_threshold: int = 128,
        preprocess_dtype: str = "float32",
        *,
        model_does_preprocess: bool = False,
        generate_previews: bool = True,
        prepare_cosmos_predict_dataset: bool = False,
        use_input_bit_rate: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Cosmos-Reason1 input preparation stage.

        Args:
            model_variant: Name of the model variant to use.
            prompt_variant: Type of prompt to use.
            prompt_text: Custom prompt text if provided.
            sampling_fps: Frames per second to sample from input clips.
            window_size: Size of each window in frames.
            remainder_threshold: Minimum frames required for the last window.
            preprocess_dtype: Data type for preprocessing operations.
            model_does_preprocess: Whether model handles preprocessing.
            generate_previews: Whether to generate preview images.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            use_input_bit_rate: Whether to use the input video's bit rate for processing.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats
        self._cosmos_r1_utils = cosmos_reason1_vl.CosmosReason1Utils(model_variant)
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._sampling_fps = sampling_fps
        self._window_size = window_size
        self._remainder_threshold = remainder_threshold
        self._preprocess_dtype = preprocess_dtype
        self._model_does_preprocess = model_does_preprocess
        self._generate_previews = generate_previews
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._use_input_bit_rate = use_input_bit_rate

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
        """Set up the Cosmos-Reason1 input preparation stage."""
        self._cosmos_r1_utils.setup()

    @nvtx.annotate("CosmosReason1InputPreparationStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Cosmos-Reason1 input preparation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.encoded_data is None:
                    _handle_empty_clip_buffer(clip)
                    continue
                with self._timer.time_process():
                    for window_bytes, window_frames, window_frame_info in zip_longest(
                        *windowing_utils.split_video_into_windows(
                            clip.encoded_data,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            sampling_fps=self._sampling_fps,
                            model_does_preprocess=self._model_does_preprocess,
                            preprocess_dtype=self._preprocess_dtype,
                            return_bytes=(self._generate_previews or self._prepare_cosmos_predict_dataset),
                            target_bit_rate=_get_target_bit_rate(video, use_input_bit_rate=self._use_input_bit_rate),
                            num_threads=max(int(self.resources.cpus), 1),
                        ),
                    ):
                        prompt = _get_prompt(
                            self._prompt_variant,
                            self._prompt_text,
                            verbose=self._verbose,
                        )
                        try:
                            # For some reason the `strip` is critical specifically for cosmos-reason1
                            # when enabling batch inference
                            llm_input = self._cosmos_r1_utils.generate_llm_inputs(
                                prompt=prompt.strip(),
                                video_inputs=window_frames,
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Error in Cosmos-Reason1 input preparation: {e}")
                            clip.errors["cosmos_reason1_input"] = str(e)
                        else:
                            clip.windows.append(
                                Window(
                                    window_frame_info.start,
                                    window_frame_info.end,
                                    mp4_bytes=window_bytes,
                                    cosmos_reason1_llm_input=llm_input,
                                ),
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class CosmosReason1CaptionStage(CuratorStage):
    """Stage that generates captions for video windows using the Cosmos-Reason1 model.

    This stage processes prepared video windows through the Cosmos-Reason1 vision-language model to
    generate descriptive captions with physical reasoning capabilities.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "cosmos_r1",
        batch_size: int = 8,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,
        generate_stage2_caption: bool = False,
        stage2_prompt_text: str | None = None,
        disable_mmcache: bool = False,
        use_async_engine: bool = False,
        prepare_cosmos_predict_dataset: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Cosmos-Reason1 caption generation stage.

        Args:
            model_variant: Name of the model variant to use.
            batch_size: Number of samples to process in parallel.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            model_does_preprocess: Whether model handles preprocessing.
            generate_stage2_caption: Whether to generate second stage captions.
            stage2_prompt_text: Custom prompt for second stage.
            disable_mmcache: Whether to disable model cache.
            use_async_engine: Whether to use the asynchronous engine for processing.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._batch_size = batch_size
        self._generate_stage2_caption = generate_stage2_caption
        self._disable_mmcache = disable_mmcache
        self._use_async_engine = use_async_engine
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._verbose = verbose
        self._log_stats = log_stats
        self._raw_model = cosmos_reason1_vl.CosmosReason1VL(
            model_variant,
            fp8=fp8_enable,
            max_output_tokens=max_output_tokens,
            model_does_preprocess=model_does_preprocess,
            stage2_prompt_text=stage2_prompt_text,
            disable_mmcache=self._disable_mmcache,
            use_async_engine=self._use_async_engine,
        )

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._raw_model

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._raw_model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up GPU resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    def process_data_sync(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process the data for the Cosmos-Reason1 caption generation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            inputs = []
            mapping: dict[int, tuple[int, int]] = {}
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        _handle_empty_clip_windows(clip)
                    for window_idx, window in enumerate(clip.windows):
                        if window.cosmos_reason1_llm_input is None:
                            _handle_empty_llm_inputs(clip, window_idx)
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        inputs.append(window.cosmos_reason1_llm_input)
                        idx += 1

                captions = self._raw_model.generate(
                    inputs,
                    generate_stage2_caption=self._generate_stage2_caption,
                    batch_size=self._batch_size,
                )

                _assign_captions(
                    video,
                    mapping,
                    enumerate(captions),
                    self._model_variant,
                    keep_mp4_bytes=self._prepare_cosmos_predict_dataset,
                    verbose=self._verbose,
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    async def _process_data_async(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            input_req_id = 0
            vllm_tasks = []
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        _handle_empty_clip_windows(clip)
                    for window_idx, window in enumerate(clip.windows):
                        if window.cosmos_reason1_llm_input is None:
                            _handle_empty_llm_inputs(clip, window_idx)
                            continue
                        mapping[input_req_id] = (clip_idx, window_idx)
                        vllm_task = asyncio.create_task(
                            self._raw_model.generate_async(
                                window.cosmos_reason1_llm_input,
                                input_req_id,
                                generate_stage2_caption=self._generate_stage2_caption,
                            ),
                        )
                        vllm_tasks.append(vllm_task)
                        input_req_id += 1

                # Wait for all VLLM tasks to complete
                vllm_captions = await asyncio.gather(*vllm_tasks)
                # Assign captions to the corresponding clip/window
                _assign_captions(
                    video,
                    mapping,
                    vllm_captions,
                    self._model_variant,
                    keep_mp4_bytes=self._prepare_cosmos_predict_dataset,
                    verbose=self._verbose,
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    def get_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        """Get the asyncio event loop.

        Returns:
            The asyncio event loop.

        """
        try:
            asyncio_loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio_loop)
        return asyncio_loop

    @nvtx.annotate("CosmosReason1CaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Cosmos-Reason1 caption generation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._use_async_engine:
            asyncio_loop = self.get_asyncio_loop()
            return asyncio_loop.run_until_complete(self._process_data_async(tasks))
        return self.process_data_sync(tasks)


class PhiInputPreparationStage(CuratorStage):
    """Stage that prepares video windows for Phi-4 multimodal model processing.

    This stage handles the preparation of video windows and prompts for the Phi-4 vision-language
    model, including frame sampling, preprocessing, and input formatting.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "phi4",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        sampling_fps: float = 1.0,
        window_size: int = 256,
        remainder_threshold: int = 128,
        preprocess_dtype: str = "float32",
        *,
        model_does_preprocess: bool = False,
        generate_previews: bool = True,
        prepare_cosmos_predict_dataset: bool = False,
        use_input_bit_rate: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Phi input preparation stage.

        Args:
            model_variant: Name of the model variant to use.
            prompt_variant: Type of prompt to use.
            prompt_text: Custom prompt text if provided.
            sampling_fps: Frames per second for sampling.
            window_size: Size of each window in frames.
            remainder_threshold: Minimum frames required for a remainder window.
            preprocess_dtype: Data type for preprocessing.
            model_does_preprocess: Whether model handles preprocessing.
            generate_previews: Whether to generate previews.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            use_input_bit_rate: Whether to use the input video's bit rate for processing.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats
        self._phi_utils = phi_vl.PhiUtils(model_variant)
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._sampling_fps = sampling_fps
        self._window_size = window_size
        self._remainder_threshold = remainder_threshold
        self._preprocess_dtype = preprocess_dtype
        self._model_does_preprocess = model_does_preprocess
        self._generate_previews = generate_previews
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._use_input_bit_rate = use_input_bit_rate

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
        return "phi"

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._phi_utils.setup()

    @nvtx.annotate("PhiInputPreparationStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Phi-4 input preparation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.encoded_data is None:
                    _handle_empty_clip_buffer(clip)
                    continue
                with self._timer.time_process():
                    for window_bytes, window_frames, window_frame_info in zip_longest(
                        *windowing_utils.split_video_into_windows(
                            clip.encoded_data,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            sampling_fps=self._sampling_fps,
                            model_does_preprocess=self._model_does_preprocess,
                            preprocess_dtype=self._preprocess_dtype,
                            return_bytes=(self._generate_previews or self._prepare_cosmos_predict_dataset),
                            target_bit_rate=_get_target_bit_rate(video, use_input_bit_rate=self._use_input_bit_rate),
                            num_threads=max(int(self.resources.cpus), 1),
                        ),
                    ):
                        prompt = _get_prompt(
                            self._prompt_variant,
                            self._prompt_text,
                            verbose=self._verbose,
                        )
                        try:
                            llm_input = self._phi_utils.generate_llm_inputs(
                                prompt=prompt,
                                video_inputs=window_frames,
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.exception(f"Error in Phi input preparation: {e}")
                            clip.errors["phi_input"] = str(e)
                        else:
                            clip.windows.append(
                                Window(
                                    window_frame_info.start,
                                    window_frame_info.end,
                                    mp4_bytes=window_bytes,
                                    phi_llm_input=llm_input,
                                ),
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class PhiCaptionStage(CuratorStage):
    """Stage that generates captions for video windows using the Phi-4 multimodal model.

    This stage processes prepared video windows through the Phi-4 multimodal vision-language model to
    generate descriptive captions, with support for both synchronous processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "phi4",
        batch_size: int = 16,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 512,
        model_does_preprocess: bool = False,  # noqa: ARG002, we are likely to use this in the future
        disable_mmcache: bool = False,
        use_async_engine: bool = False,
        prepare_cosmos_predict_dataset: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Phi caption generation stage.

        Args:
            model_variant: Name of the model variant to use.
            batch_size: Number of samples to process in parallel.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            model_does_preprocess: Whether model handles preprocessing.
            disable_mmcache: Whether to disable model cache.
            use_async_engine: Whether to use the asynchronous engine for processing.
            prepare_cosmos_predict_dataset: Whether to prepare dataset for Cosmos-Predict.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._batch_size = batch_size
        self._disable_mmcache = disable_mmcache
        self._use_async_engine = use_async_engine
        self._prepare_cosmos_predict_dataset = prepare_cosmos_predict_dataset
        self._verbose = verbose
        self._log_stats = log_stats
        self._raw_model = phi_vl.PhiVL(
            model_variant,
            fp8=fp8_enable,
            max_output_tokens=max_output_tokens,
            disable_mmcache=self._disable_mmcache,
        )

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._raw_model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._raw_model

    @nvtx.annotate("PhiCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Phi caption generation stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            inputs = []
            mapping: dict[int, tuple[int, int]] = {}
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        _handle_empty_clip_windows(clip)
                    for window_idx, window in enumerate(clip.windows):
                        if window.phi_llm_input is None:
                            _handle_empty_llm_inputs(clip, window_idx)
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        assert window.phi_llm_input is not None
                        inputs.append(window.phi_llm_input)
                        idx += 1

                captions = self._raw_model.generate(
                    inputs,
                    batch_size=self._batch_size,
                )

                _assign_captions(
                    video,
                    mapping,
                    enumerate(captions),
                    self._model_variant,
                    keep_mp4_bytes=self._prepare_cosmos_predict_dataset,
                    verbose=self._verbose,
                )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "phi"


class _T5Stage(CuratorStage):
    """Stage that encodes captions using the T5 model.

    This stage processes video captions through the T5 encoder model to generate
    text embeddings for downstream tasks.
    """

    def __init__(
        self,
        caption_fields: list[str] | None = None,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the T5 caption encoding stage.

        Args:
            caption_fields: List of caption fields to encode.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        if caption_fields is None:
            caption_fields = ["qwen_caption"]
        self._timer = StageTimer(self)
        self._caption_fields = caption_fields
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = t5_encoder.T5Encoder(t5_encoder.ModelVariant.T5_XXL)
        self._batch_size = 16 if operation_context.is_running_on_the_cloud() else 4

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    def _add_prompt(self, all_prompts: list[str], caption_window: dict[str, str]) -> str | None:
        found_caption_field = None
        for field in self._caption_fields:
            if field in caption_window:
                all_prompts.append(caption_window[field])
                found_caption_field = field
                break
        return found_caption_field

    def _encode_prompts(self, all_prompts: list[str]) -> list[t5_encoder.EncodedSample]:
        return self._model.encode(all_prompts, batch_size=self._batch_size)


class T5StageForSplit(_T5Stage):
    """Stage that encodes captions using the T5 model for shard processing."""

    @nvtx.annotate("T5StageForSplit")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the T5 caption encoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            all_prompts: list[str] = []
            mapping: list[tuple[int, int, str]] = []

            for clip_idx, clip in enumerate(task.video.clips):
                for window_idx, window in enumerate(clip.windows):
                    found_caption_field = self._add_prompt(all_prompts, window.caption)
                    if found_caption_field is None:
                        logger.error(f"Clip {clip.uuid} window {window_idx} has no caption, drop this clip!")
                        continue
                    mapping.append((clip_idx, window_idx, found_caption_field))

            encoded_results = self._encode_prompts(all_prompts)

            for result_idx, result in enumerate(encoded_results):
                clip_idx, window_idx, caption_field = mapping[result_idx]
                task.video.clips[clip_idx].windows[window_idx].t5_xxl_embedding[caption_field] = result.encoded_text

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class T5StageForShard(_T5Stage):
    """Stage that encodes captions using the T5 model for shard processing."""

    @nvtx.annotate("T5StageForShard")  # type: ignore[misc]
    def process_data(self, tasks: list[ShardPipeTask]) -> list[ShardPipeTask] | None:
        """Process the data for the T5 caption encoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            all_prompts: list[str] = []
            mapping: list[tuple[int, int]] = []

            for clip_idx, clip in enumerate(task.samples):
                for window_idx, window in enumerate(clip.clip_metadata["windows"]):
                    found_caption_field = self._add_prompt(all_prompts, window)
                    if found_caption_field is None:
                        logger.error(f"Clip {clip.uuid} window {window_idx} has no caption, drop this clip!")
                        clip.clip_metadata["valid"] = False
                        continue
                    mapping.append((clip_idx, window_idx))

            encoded_results = self._encode_prompts(all_prompts)

            for result_idx, result in enumerate(encoded_results):
                clip_idx, window_idx = mapping[result_idx]
                task.samples[clip_idx].t5_xxl_embeddings.append(result.encoded_text)
                assert (window_idx + 1) == len(task.samples[clip_idx].t5_xxl_embeddings)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


# Post-Caption-Stage to further enhance the generated Caption, either using
# additional Metadata or just 'Tuned Instructions in the Prompt'
class EnhanceCaptionStage(CuratorStage):
    """Stage that enhances video captions using a chat language model.

    This stage takes existing captions and uses a chat language model to generate
    more detailed and refined descriptions of the video content.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen_lm",
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        batch_size: int = 32,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 2048,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the caption enhancement stage.

        Args:
            model_variant: Language model to use for enhancement. One of
                "qwen_lm" or "gpt_oss_20b".
            prompt_variant: Type of prompt to use.
            prompt_text: Custom prompt text if provided.
            batch_size: Number of samples to process in parallel.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._enhanced_key = model_variant
        self._raw_model = ChatLM(
            model_variant,
            max_output_tokens=max_output_tokens,
            quantization=("fp8" if fp8_enable and model_variant == "qwen_lm" else None),
        )
        self._prompt_variant = prompt_variant
        self._prompt_text = prompt_text
        self._prompt = _get_enhance_prompt(
            prompt_variant,
            prompt_text,
            verbose=verbose,
        )

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._raw_model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    @nvtx.annotate("EnhanceCaptionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process the data for the caption enhancement stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            inputs = []
            mapping: dict[int, tuple[int, int]] = {}
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if len(clip.windows) == 0:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                    for window_idx, window in enumerate(clip.windows):
                        if window.caption is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no captions generated.")  # type: ignore[unreachable]
                            clip.errors[f"window-{window_idx}"] = "empty"
                            continue
                        mapping[idx] = (clip_idx, window_idx)

                        caption = None
                        for caption_model_variant in window.caption:
                            caption = window.caption[caption_model_variant]
                            break

                        if caption is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no captions")
                            continue

                        captionInput = [
                            {"role": "system", "content": self._prompt},
                            {"role": "user", "content": caption},
                        ]
                        inputs.append(captionInput)
                        idx += 1

                if inputs:
                    captions = self._raw_model.generate(
                        inputs,
                        batch_size=self._batch_size,
                    )

                    for idx, result in enumerate(captions):
                        clip_idx, window_idx = mapping[idx]
                        video.clips[clip_idx].windows[window_idx].enhanced_caption[self._enhanced_key] = result
                        if self._verbose:
                            logger.info(
                                f"Enhanced {self._enhanced_key} Caption for clip {video.clips[clip_idx].uuid} "
                                f"window {window_idx}: {result}",
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
