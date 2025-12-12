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
"""Aesthetic score filtering stages."""

import json
import re
from collections.abc import Iterable
from itertools import zip_longest

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models import qwen_vl
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video, Window

"""
Custom prompts are supported by passing a comma separated list of categories to the --qwen-filter-categories flag.
If the prompt is not a comma separated list, the default prompt will be used.
As an example, if the user passes --qwen-filter-categories flag "blue car, red car, green car", the prompt will be:

Can you answer the following questions about this video:
Is there blue car in the video?
Is there red car in the video?
Is there green car in the video?
Answer format:
{
"blue car": "yes" or "no",
"red car": "yes" or "no",
"green car": "yes" or "no"
}
"""

_PROMPTS = {
    "custom": """
    Can you answer the following questions about this video:
    """,
    "default": """Can you answer the following questions about this video:
        1. Is this a slideshow (e.g., only showing static images, animated text / image, etc.)
        or a video with slide transitions?
        2. Is this a synthetic video (e.g., screen recording, motion graphics, AI-generated video,
        stop motion, slideshow, etc.)
        as opposed to a video captured by an optical camera sensor?
        3. Does this video have visual filters (e.g., editing properties
        like grain, noise, saturation, color, aliasing, simulated weather effect, etc.)?
        4. Does this video have text overlaid in post-production (e.g., waterwark, subtitles, logo, graphics, etc.)?
        Text that is part of the original video content is not considered as post-production text.
        5. Is this video a video in video (e.g., video overlay/collage, etc.)?
        6. Does this video have bad photographic artifacts(e.g., over/under exposure, lens flare, poor focus, etc.)?
        7. Does this video have distorted view (e.g., fisheye effect form wide field of view)?
        8. Is the video rotated to an uncommon view?
        9. Does this video have low resolution or is it blurry for all or some of the frames?
        10. Does this video contain any blurred or pixelated region
        (e.g., on specific human faces or objects, background, logo region, etc.)?
        11. Does this video mainly involve camera movement and little scene dynamics?
        12. Does this video contain abrupt / very large camera motion or camera shake (e.g., in some
        frames the camera is moving or rotated so fast that you cannot see clearly what's happening).
        13. Does this video involve fast zoom in or zoom out of the camera?
        14. Does this video have unnatural speed (e.g., slow motion, time lapse, frame skipping, etc.)?
        Answer format:
        {
        "slideshow": "yes" or "no",
        "synthetic video": "yes" or "no",
        "visual filter": "yes" or "no",
        "post-production text": "yes" or "no",
        "video in video": "yes" or "no",
        "photographic artifacts": "yes" or "no",
        "distorted view": "yes" or "no",
        "rotated view": "yes" or "no",
        "low resolution": "yes" or "no",
        "blurred region": "yes" or "no",
        "little scene dynamics": "yes" or "no",
        "abrupt camera motion": "yes" or "no",
        "fast zoom": "yes" or "no",
        "unnatural speed": "yes" or "no"
}""",
}

"""
The default filter criteria are listed below.
When using a custom prompt with "--qwen-filter-categories" the criteria will be generated automatically.
If you wish to create and save a longer custom prompt above,
add the criteria list to the _FILTER_CRITERIA dictionary below.
Ensure the name of the custom prompt matches the name of the created criteria list.
"""

_FILTER_CRITERIA = {
    "default": [
        "slideshow",
        "synthetic video",
        "visual filter",
        "post-production text",
        "video in video",
        "photographic artifacts",
        "distorted view",
        "rotated view",
        "low resolution",
        "blurred region",
        "abrupt camera motion",
        "fast zoom",
        "unnatural speed",
    ],
}


def _get_prompt(
    prompt_variant: str,
    filter_categories: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Get the filtering prompt for Qwen model."""
    if filter_categories is not None:
        try:
            categories = filter_categories.split(",")
            prompt = _PROMPTS["custom"]
            for category in categories:
                prompt += f"Is there {category} in the video?\n"
            prompt += "\nAnswer format: {\n"
            for i, category in enumerate(categories):
                comma = "," if i < len(categories) - 1 else ""  # No comma for last item
                prompt += f""""{category}": "yes" or "no"{comma}\n"""
            prompt += "}"
        except AttributeError:
            logger.warning(f"Prompt text is not a comma separated list: {filter_categories}")
            prompt = _PROMPTS["default"]
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Filtering prompt: {prompt}")
    return prompt


class QwenInputPreparationStageFiltering(CuratorStage):
    """Stage that prepares video windows for Qwen model processing.

    This stage handles the preparation of video windows and prompts for the Qwen vision-language
    model, including frame sampling, preprocessing, and input formatting.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        prompt_variant: str = "default",
        filter_categories: str | None = None,
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
            filter_categories: Custom prompt categories as a list if provided.
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
        self._filter_categories = filter_categories
        self._sampling_fps = sampling_fps
        self._window_size = window_size
        self._remainder_threshold = remainder_threshold
        self._preprocess_dtype = preprocess_dtype
        self._model_does_preprocess = model_does_preprocess
        self._generate_previews = generate_previews
        self._model_variant = model_variant

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
        return "vllm"

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._qwen_utils.setup()

    @nvtx.annotate("QwenInputPreparationStage")  # type: ignore[untyped-decorator]
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
                    logger.warning(f"Clip {clip.uuid} has no encoded_data.")
                    clip.errors["encoded_data"] = "empty"
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
                            return_bytes=self._generate_previews,
                            num_threads=max(int(self.resources.cpus), 1),
                        ),
                    ):
                        prompt = _get_prompt(
                            self._prompt_variant,
                            self._filter_categories,
                            verbose=self._verbose,
                        )
                        try:
                            llm_input = self._qwen_utils.generate_llm_inputs(
                                prompt=prompt, video_inputs=window_frames, override_text_prompt=False
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Error in Qwen input preparation: {e}")
                            clip.errors["qwen_input"] = str(e)
                        else:
                            clip.filter_windows.append(
                                Window(
                                    window_frame_info.start,
                                    window_frame_info.end,
                                    mp4_bytes=window_bytes,
                                    model_input={self._model_variant: llm_input},
                                ),
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class QwenFilteringStage(CuratorStage):
    """Stage that generates filtering results for video windows using the Qwen model.

    This stage processes prepared video windows through the Qwen vision-language model to
    generate filtering results.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        batch_size: int = 16,
        user_prompt: str | None = None,
        filter_variant: str = "default",
        rejection_threshold: float = 0.5,
        *,
        fp8_enable: bool = False,
        max_output_tokens: int = 512,
        verbose: bool = False,
        log_stats: bool = False,
        model_does_preprocess: bool = False,
        disable_mmcache: bool = False,
        score_only: bool = False,
    ) -> None:
        """Initialize the Qwen filtering stage.

        Args:
            model_variant: Name of the model variant to use.
            batch_size: Number of samples to process in parallel.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            model_does_preprocess: Whether model handles preprocessing.
            disable_mmcache: Whether to disable model cache.
            user_prompt: Custom prompt categories as a list if provided.
            filter_variant: Variant of filter criteria to use.
            rejection_threshold: Threshold for clip rejection.
            score_only: Whether to only calculate Qwen-based content filtering scores without filtering clips.

        """
        self._timer = StageTimer(self)
        self._filter_variant = filter_variant
        self._rejection_threshold = rejection_threshold
        self._batch_size = batch_size
        self._user_prompt = user_prompt
        self._verbose = verbose
        self._log_stats = log_stats
        self._score_only = score_only
        self._model_does_preprocess = model_does_preprocess
        self._disable_mmcache = disable_mmcache
        self._model = qwen_vl.QwenVL(
            model_variant,
            fp8=fp8_enable,
            max_output_tokens=max_output_tokens,
            model_does_preprocess=self._model_does_preprocess,
            disable_mmcache=self._disable_mmcache,
        )
        self._model_variant = model_variant

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "vllm"

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Get the model.

        Returns:
            The model.

        """
        return self._model

    def _process_data_sync(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process the data for the Qwen filtering stage.

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
                    if len(clip.filter_windows) == 0:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                    for window_idx, window in enumerate(clip.filter_windows):
                        llm_input = window.model_input.get(self._model_variant)
                        if llm_input is None:
                            logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
                            clip.errors[f"window-{window_idx}"] = "empty"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        inputs.append(llm_input)
                        idx += 1

                results = self._model.generate(
                    inputs,
                    batch_size=self._batch_size,
                )

                self._filter_clips(video, mapping, enumerate(results))

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips "
                    "passed/filtered"
                )

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    @nvtx.annotate("QwenFilteringStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Qwen filtering stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        return self._process_data_sync(tasks)

    def _filter_clips(  # noqa: C901, PLR0912
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        """Filter clips based on the captions."""
        clip_results: dict[int, list[tuple[int, str]]] = {}
        for idx, result in captions:
            clip_idx, window_idx = mapping[idx]
            if clip_idx not in clip_results:
                clip_results[clip_idx] = []
            clip_results[clip_idx].append((window_idx, result))

        # A list of clips that pass the filter
        passing_clips = []

        # For each clip, check if it should pass the filter
        for clip_idx, window_results in clip_results.items():
            clip_should_pass = True
            all_issues = set()
            rejected_windows = set()

            # Inner loop: look at all windows for this clip
            for window_idx, result in window_results:
                filtering_dict = _parse_results(result)
                if filtering_dict is None:
                    # If the filtering dict is None, it means the model failed to generate a valid JSON string.
                    # We should not reject the clip in this case.
                    continue

                if self._user_prompt is None:
                    filter_criteria = _FILTER_CRITERIA[self._filter_variant]
                else:
                    filter_criteria = self._user_prompt.split(",")

                # Check if the window passes the filter
                window_specific_issues = {}
                for criterion in filter_criteria:
                    if filtering_dict.get(criterion, "no") == "yes":
                        all_issues.add(criterion)
                        rejected_windows.add(window_idx)
                        window_specific_issues[criterion] = filtering_dict.get(criterion, "no")

                video.clips[clip_idx].filter_windows[window_idx].caption["qwen_rejection_reasons"] = str(
                    window_specific_issues
                )

            if not self._score_only:  # noqa: SIM102
                # Reject the clip if more than half of the windows are rejected
                if (
                    len(window_results) > 0
                    and (len(rejected_windows) / len(window_results)) > self._rejection_threshold
                ):
                    clip_should_pass = False

            if not clip_should_pass:
                if self._verbose:
                    logger.info(f"Clip {video.clips[clip_idx].uuid} filtered out due to: {set(all_issues)}")
                video.filtered_clips.append(video.clips[clip_idx])
            else:
                passing_clips.append(video.clips[clip_idx])

            # Clean up filter windows after processing

        video.clips = passing_clips
        for clip in video.clips:
            for window in clip.filter_windows:
                window.model_input.clear()
                window.mp4_bytes = None


def _clean_json_string(output_text: str) -> dict[str, str] | None:
    """Clean and fix common JSON formatting issues. Qwen2.5 sometimes fails to output valid JSON."""
    # Remove markdown code block markers if present
    output_text = output_text.removeprefix("```json")
    output_text = output_text.removesuffix("```")

    # Replace escaped quotes with single quotes
    output_text = output_text.replace('\\"', "'")

    # Remove any BOM or special characters at the start
    output_text = output_text.lstrip("\ufeff\n\r\t ")

    # Normalize line endings
    output_text = output_text.replace("\r\n", "\n")

    # Add missing commas between properties
    output_text = re.sub(r'"\s*\n\s*"', '",\n"', output_text)

    # Fix any trailing or leading whitespace
    output_text = output_text.strip()

    try:
        return json.loads(output_text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}")
        logger.error(f"Output text: {output_text}")
        return None


def _parse_results(caption: str) -> dict[str, str] | None:
    """Parse the caption into a dictionary."""
    # this assumes we are doing one round of captioning
    output_text = caption

    return _clean_json_string(output_text)
