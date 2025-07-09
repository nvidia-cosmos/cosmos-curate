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
from collections.abc import Iterable
from itertools import zip_longest

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.runtime import operation_utils
from cosmos_curate.core.utils.runtime.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.runtime.performance_utils import StageTimer
from cosmos_curate.models import (
    qwen_lm,
    qwen_vl,
    t5_encoder,
)
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    ShardPipeTask,
    SplitPipeTask,
    Video,
    _Window,
)

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
                if clip.buffer is None:
                    logger.warning(f"Clip {clip.uuid} has no buffer.")
                    clip.errors["buffer"] = "empty"
                    continue
                with self._timer.time_process():
                    for window_mp4_bytes, _, window_frame_info in zip(
                        *windowing_utils.split_video_into_windows(
                            clip.buffer,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            num_threads=max(int(self.resources.cpus), 1),
                            return_bytes=True,
                            return_video_frames=False,
                        ),
                        strict=True,
                    ):
                        clip.windows.append(
                            _Window(
                                window_frame_info.start,
                                window_frame_info.end,
                                window_mp4_bytes,
                                None,
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
                if clip.buffer is None:
                    logger.warning(f"Clip {clip.uuid} has no buffer.")
                    clip.errors["buffer"] = "empty"
                    continue
                with self._timer.time_process():
                    for window_bytes, window_frames, window_frame_info in zip_longest(
                        *windowing_utils.split_video_into_windows(
                            clip.buffer,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            sampling_fps=self._sampling_fps,
                            model_does_preprocess=self._model_does_preprocess,
                            preprocess_dtype=self._preprocess_dtype,
                            return_bytes=self._generate_previews,
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
                                _Window(
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
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._batch_size = batch_size
        self._generate_stage2_caption = generate_stage2_caption
        self._disable_mmcache = disable_mmcache
        self._use_async_engine = use_async_engine
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
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                    for window_idx, window in enumerate(clip.windows):
                        if window.qwen_llm_input is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
                            clip.errors[f"window-{window_idx}"] = "empty"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        assert window.qwen_llm_input is not None
                        inputs.append(window.qwen_llm_input)
                        idx += 1

                captions = self._raw_model.generate(
                    inputs,
                    generate_stage2_caption=self._generate_stage2_caption,
                    batch_size=self._batch_size,
                )

                self._assgin_captions(video, mapping, enumerate(captions))

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
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                    for window_idx, window in enumerate(clip.windows):
                        if window.qwen_llm_input is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
                            clip.errors[f"window-{window_idx}"] = "empty"
                            continue
                        mapping[input_req_id] = (clip_idx, window_idx)
                        assert window.qwen_llm_input is not None
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
                self._assgin_captions(video, mapping, vllm_captions)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    def _assgin_captions(
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        _captions = list(captions)
        for req_id, caption in _captions:
            clip_idx, window_idx = mapping[req_id]
            video.clips[clip_idx].windows[window_idx].caption["qwen"] = caption
            if self._verbose:
                logger.info(f"Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {caption}")

        logger.info(
            f"Generated {len(_captions)} captions for video {video.input_path} "
            f"chunk-{video.clip_chunk_index} with {len(video.clips)} clips",
        )

        for clip in video.clips:
            for window in clip.windows:
                window.qwen_llm_input = None
                window.mp4_bytes = None

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


class T5Stage(CuratorStage):
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

    @nvtx.annotate("T5Stage")  # type: ignore[misc]
    def process_data(self, tasks: list[ShardPipeTask]) -> list[ShardPipeTask] | None:
        """Process the data for the T5 caption encoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            all_prompts = []
            sample_key_mapping: list[tuple[int, int]] = []

            for clip_idx, clip in enumerate(task.samples):
                for window_idx, window in enumerate(clip.clip_metadata["windows"]):
                    has_caption = False
                    for field in self._caption_fields:
                        if field in window:
                            all_prompts.append(window[field])
                            has_caption = True
                            break
                    if not has_caption:
                        logger.error(f"Clip {clip.uuid} window {window_idx} has no caption, drop this clip!")
                        clip.clip_metadata["valid"] = False
                        continue
                    sample_key_mapping.append((clip_idx, window_idx))

            batch_size = 16 if operation_utils.is_running_on_the_cloud() else 4

            with self._timer.time_process():
                # Encode all prompts at once
                encoded_results = self._model.encode(all_prompts, batch_size=batch_size)

            for result_idx, result in enumerate(encoded_results):
                clip_idx, window_idx = sample_key_mapping[result_idx]
                task.samples[clip_idx].t5_xxl_embeddings.append(result.encoded_text)
                assert (window_idx + 1) == len(task.samples[clip_idx].t5_xxl_embeddings)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


# Post-Caption-Stage to further enhance the generated Caption, either using
# additional Metadata or just 'Tuned Instructions in the Prompt'
class EnhanceCaptionStage(CuratorStage):
    """Stage that enhances video captions using the Qwen language model.

    This stage takes existing captions and uses the Qwen language model to generate
    more detailed and refined descriptions of the video content.
    """

    def __init__(  # noqa: PLR0913
        self,
        prompt_variant: str = "default",
        prompt_text: str | None = None,
        batch_size: int = 128,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 2048,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the caption enhancement stage.

        Args:
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
        self._raw_model = qwen_lm.QwenLM(fp8=fp8_enable, max_output_tokens=max_output_tokens)
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
                        assert window.caption is not None

                        caption = None
                        if "qwen" in window.caption:
                            caption = window.caption["qwen"]

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
                        qwen_vl_caption = video.clips[clip_idx].windows[window_idx].caption["qwen"]
                        video.clips[clip_idx].windows[window_idx].enhanced_caption["qwen_lm"] = result
                        if self._verbose:
                            logger.info(
                                f"Caption for clip {video.clips[clip_idx].uuid} window {window_idx}: {qwen_vl_caption}",
                            )
                            logger.info(
                                f"Enhanced QwenLM Caption for clip {video.clips[clip_idx].uuid} "
                                f"window {window_idx}: {result}",
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
