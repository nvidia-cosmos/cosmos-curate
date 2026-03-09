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

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models import qwen_vl
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_prompts import (
    FILTER_CRITERIA,
    VIDEO_TYPE_LABELS,
    get_qwen_filter_prompt,
)
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video, Window


def parse_comma_separated_types(cs: str | None) -> list[str]:
    """Return list of non-empty stripped tokens from a comma-separated string."""
    if not cs:
        return []
    return [s.strip() for s in cs.split(",") if s.strip()]


def custom_categories_union(type_allow: str | None, type_block: str | None) -> str | None:
    """Return comma-separated sorted union of allow and block types, or None if empty."""
    categories = set(parse_comma_separated_types(type_allow)) | set(parse_comma_separated_types(type_block))
    return ",".join(sorted(categories)) if categories else None


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
        extra_outputs: (list[tuple[str, str] | tuple[str, str, str | None]] | None) = None,
    ) -> None:
        """Initialize the Qwen input preparation stage.

        Args:
            model_variant: Name of the model variant (for QwenUtils and key for main output in model_input).
            prompt_variant: Type of prompt to use for the main output.
            filter_categories: Custom prompt categories as a list if provided.
            sampling_fps: Frames per second for sampling.
            window_size: Size of each window in frames.
            remainder_threshold: Minimum frames required for a remainder window.
            preprocess_dtype: Data type for preprocessing.
            model_does_preprocess: Whether model handles preprocessing.
            generate_previews: Whether to generate previews.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            extra_outputs: Optional list of (prompt_variant, output_key) or
                (prompt_variant, output_key, filter_categories_override). All keys are
                stored in the same Window.model_input so multiple stages (e.g. filter +
                classifier) can share one preparation. When the third element is present
                it is used as filter_categories for that output's prompt.

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
        self._extra_outputs = list(extra_outputs) if extra_outputs else []

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
            prefetch([clip.encoded_data for clip in video.clips])
            for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
                if data is None:
                    logger.warning(f"Clip {clip.uuid} has no encoded_data.")
                    clip.errors["encoded_data"] = "empty"
                    continue
                clip.filter_windows.clear()
                with self._timer.time_process():
                    for window_bytes, window_frames, window_frame_info in zip(
                        *windowing_utils.split_video_into_windows(
                            data,
                            window_size=self._window_size,
                            remainder_threshold=self._remainder_threshold,
                            sampling_fps=self._sampling_fps,
                            model_does_preprocess=self._model_does_preprocess,
                            preprocess_dtype=self._preprocess_dtype,
                            return_bytes=self._generate_previews,
                            num_threads=max(int(self.resources.cpus), 1),
                        ),
                        strict=True,
                    ):
                        model_input: dict[str, dict[str, object]] = {}
                        main = (self._prompt_variant, self._model_variant)
                        for item in [main, *self._extra_outputs]:
                            pv, out_key = item[0], item[1]
                            fc_override: str | None = None
                            if len(item) >= 3:  # noqa: PLR2004
                                fc_override = item[2]
                            filter_cats = (
                                fc_override
                                if fc_override is not None
                                else (self._filter_categories if pv == self._prompt_variant else None)
                            )
                            prompt = get_qwen_filter_prompt(
                                pv,
                                filter_cats,
                                verbose=self._verbose,
                            )
                            try:
                                llm_input = self._qwen_utils.generate_llm_inputs(
                                    prompt=prompt,
                                    video_inputs=window_frames,
                                )
                            except Exception as e:  # noqa: BLE001
                                logger.error(f"Error in Qwen input preparation ({out_key}): {e}")
                                clip.errors["qwen_input"] = str(e)
                                break
                            model_input[out_key] = llm_input
                        else:
                            clip.filter_windows.append(
                                Window(
                                    window_frame_info.start,
                                    window_frame_info.end,
                                    mp4_bytes=window_bytes,
                                    model_input=model_input,
                                ),
                            )

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class QwenFilteringStage(CuratorStage):
    """Stage that applies semantic (criteria-based) filtering using the Qwen model.

    Processes prepared video windows through the Qwen vision-language model and
    filters clips based on configurable criteria (e.g. slideshow, synthetic video,
    visual filter) and a rejection threshold.
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
        clear_model_input_after: bool = True,
        model_input_key: str | None = None,
    ) -> None:
        """Initialize the Qwen semantic filtering stage.

        Args:
            model_variant: Name of the model variant to use for the QwenVL model.
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
            score_only: Whether to only calculate scores without filtering clips.
            clear_model_input_after: If True, clear window.model_input after filtering so
                downstream stages do not see it; set False when another stage (e.g. classifier)
                will read from the same windows.
            model_input_key: Key in Window.model_input to read prepared inputs from; default
                is model_variant. Use a different key (e.g. "qwen_filter") when sharing prep
                with the classifier stage.

        """
        self._timer = StageTimer(self)
        self._clear_model_input_after = clear_model_input_after
        self._model_input_key = model_input_key if model_input_key is not None else model_variant
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
        return "unified"

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
                        llm_input = window.model_input.get(self._model_input_key)
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
        """Filter clips based on semantic criteria and rejection threshold."""
        clip_results: dict[int, list[tuple[int, str]]] = {}
        for idx, result in captions:
            clip_idx, window_idx = mapping[idx]
            if clip_idx not in clip_results:
                clip_results[clip_idx] = []
            clip_results[clip_idx].append((window_idx, result))

        passing_clips = []
        for clip_idx, window_results in clip_results.items():
            clip_should_pass = True
            all_issues: set[str] = set()
            rejected_windows: set[int] = set()

            if self._user_prompt is None:
                filter_criteria = FILTER_CRITERIA[self._filter_variant]
            else:
                filter_criteria = [s.strip() for s in self._user_prompt.split(",") if s.strip()]

            for window_idx, result in window_results:
                filtering_dict = _parse_results(result)
                if filtering_dict is None:
                    continue
                window_specific_issues = {}
                for criterion in filter_criteria:
                    if filtering_dict.get(criterion, "no") == "yes":
                        all_issues.add(criterion)
                        rejected_windows.add(window_idx)
                        window_specific_issues[criterion] = filtering_dict.get(criterion, "no")
                video.clips[clip_idx].filter_windows[window_idx].caption["qwen_rejection_reasons"] = str(
                    window_specific_issues
                )

            if (
                not self._score_only
                and len(window_results) > 0
                and (len(rejected_windows) / len(window_results)) > self._rejection_threshold
            ):
                clip_should_pass = False

            if not clip_should_pass:
                if self._verbose:
                    logger.info(f"Clip {video.clips[clip_idx].uuid} filtered out due to: {set(all_issues)}")
                clip = video.clips[clip_idx]
                clip.qwen_rejection_stage = "semantic"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_semantic += 1
            else:
                passing_clips.append(video.clips[clip_idx])

        for clip_idx in range(len(video.clips)):
            if clip_idx not in clip_results:
                clip = video.clips[clip_idx]
                clip.errors["qwen"] = "all_windows_failed_preparation"
                clip.qwen_rejection_stage = "semantic"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_semantic += 1
                logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")

        video.clips = passing_clips
        if self._clear_model_input_after:
            for clip in video.clips + video.filtered_clips:
                for window in clip.filter_windows:
                    window.model_input.clear()
                    window.mp4_bytes.drop()


class QwenVideoClassifierStage(CuratorStage):
    """Stage that applies video-type (allow/block list) filtering using the Qwen model.

    By default classifies each window into the 27 VIDEO_TYPE_LABELS and filters clips
    based on type_allow (keep if any window matches) and/or type_block (reject if too
    many windows match). When custom_categories is True, type_allow and type_block
    define the full set of categories (union); the model is prompted only for those.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str = "qwen",
        batch_size: int = 16,
        rejection_threshold: float = 0.5,
        *,
        type_allow: str | None = None,
        type_block: str | None = None,
        custom_categories: bool = False,
        fp8_enable: bool = False,
        max_output_tokens: int = 512,
        verbose: bool = False,
        log_stats: bool = False,
        model_does_preprocess: bool = False,
        disable_mmcache: bool = False,
        clear_model_input_after: bool = True,
    ) -> None:
        """Initialize the Qwen video classifier stage.

        Args:
            model_variant: Name of the model variant to use.
            batch_size: Number of samples to process in parallel.
            rejection_threshold: Threshold for clip rejection (block ratio).
            type_allow: Comma-separated video types to keep; only clips with at least
                one window matching any of these types pass.
            type_block: Comma-separated video types to reject; clips with too many
                windows matching any of these types are filtered out.
            custom_categories: If True, type_allow and type_block define the full set
                of categories (union); model is prompted only for these. No validation
                against VIDEO_TYPE_LABELS. Requires at least one of type_allow or
                type_block to be set.
            fp8_enable: Whether to enable FP8 precision.
            max_output_tokens: Maximum number of tokens to generate.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            model_does_preprocess: Whether model handles preprocessing.
            disable_mmcache: Whether to disable model cache.
            clear_model_input_after: If True, clear window.model_input after classification.
                Set False when the semantic filter stage will run next and read from the same
                windows (e.g. when both classifier and filter are enabled with shared prep).

        """
        self._timer = StageTimer(self)
        self._clear_model_input_after = clear_model_input_after
        self._rejection_threshold = rejection_threshold
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._type_allow: list[str] = parse_comma_separated_types(type_allow)
        self._type_block: list[str] = parse_comma_separated_types(type_block)
        self._custom_categories = custom_categories
        if custom_categories:
            combined = set(self._type_allow) | set(self._type_block)
            if not combined:
                msg = "custom_categories=True requires at least one of type_allow or type_block to be set"
                raise ValueError(msg)
            self._valid_type_labels: tuple[str, ...] = tuple(sorted(combined))
        else:
            valid = set(VIDEO_TYPE_LABELS)
            for t in self._type_allow + self._type_block:
                if t not in valid:
                    msg = f"Unknown video type {t!r}; must be one of VIDEO_TYPE_LABELS: {sorted(valid)}"
                    raise ValueError(msg)
            self._valid_type_labels = VIDEO_TYPE_LABELS
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
        """Get the resource requirements for this stage."""
        return CuratorStageResource(gpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

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
        """Get the model."""
        return self._model

    def _process_data_sync(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Process the data for the Qwen video classifier stage."""
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

    @nvtx.annotate("QwenVideoClassifierStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the Qwen video classifier stage."""
        return self._process_data_sync(tasks)

    def _type_mode_filter_clip(  # noqa: C901, PLR0912
        self,
        video: Video,
        clip_idx: int,
        window_results: list[tuple[int, str]],
        all_issues: set[str],
        rejected_windows: set[int],
    ) -> bool:
        """Apply type filtering: allow list (keep if matches) and/or block list (reject if matches).

        When both type_allow and type_block are empty, acts as a classifier only: no filtering
        (all clips pass) and full classification is written to metadata.

        - Block list: include a category with "yes" only when the model said yes (matched block list).
        - Allow list: include a category with "no" only when the model said no (did not match allow list).

        When type_allow is set and every window fails to parse, the clip is passed and a warning
        is logged so allow-list behavior matches block-list (no parse -> no rejection).
        """
        has_allowed = False
        any_parsed = False
        all_types_yes: set[str] = set()
        for window_idx, result in window_results:
            filtering_dict = _parse_results(result)
            if filtering_dict is None:
                continue
            any_parsed = True
            all_types_yes.update(k for k, v in filtering_dict.items() if v == "yes" and k in self._valid_type_labels)
            rejection_reasons: dict[str, str] = {}
            if self._type_block:
                for t in self._type_block:
                    if filtering_dict.get(t, "no") == "yes":
                        rejection_reasons[t] = "yes"
            if self._type_allow:
                for t in self._type_allow:
                    if filtering_dict.get(t, "no") == "no":
                        rejection_reasons[t] = "no"
            video.clips[clip_idx].filter_windows[window_idx].caption["qwen_rejection_reasons"] = str(rejection_reasons)
            if self._type_allow:
                if any(filtering_dict.get(t, "no") == "yes" for t in self._type_allow):
                    has_allowed = True
            else:
                has_allowed = True
            if self._type_block:
                for t in self._type_block:
                    if filtering_dict.get(t, "no") == "yes":
                        all_issues.add(t)
                        rejected_windows.add(window_idx)
                        break
        clip = video.clips[clip_idx]
        clip.qwen_type_classification = (
            sorted(all_types_yes) if all_types_yes else (["unclassified"] if self._custom_categories else [])
        )
        if self._verbose and clip.qwen_type_classification:
            logger.info(f"Clip {clip.uuid} type classification: {clip.qwen_type_classification}")
        if self._type_allow and not any_parsed:
            logger.warning(
                f"Clip {clip.uuid}: all windows failed to parse; passing for allow-list (no rejection reason)"
            )
            has_allowed = True
        allow_ok = has_allowed
        block_ok = True
        if self._type_block and window_results:
            ratio = len(rejected_windows) / len(window_results)
            block_ok = ratio <= self._rejection_threshold
        return bool(allow_ok and block_ok)

    def _filter_clips(  # noqa: C901
        self,
        video: Video,
        mapping: dict[int, tuple[int, int]],
        captions: Iterable[tuple[int, str]],
    ) -> None:
        """Filter clips based on type allow/block lists and rejection threshold."""
        clip_results: dict[int, list[tuple[int, str]]] = {}
        for idx, result in captions:
            clip_idx, window_idx = mapping[idx]
            if clip_idx not in clip_results:
                clip_results[clip_idx] = []
            clip_results[clip_idx].append((window_idx, result))

        passing_clips = []
        for clip_idx, window_results in clip_results.items():
            all_issues: set[str] = set()
            rejected_windows: set[int] = set()
            clip_should_pass = self._type_mode_filter_clip(
                video, clip_idx, window_results, all_issues, rejected_windows
            )

            if not clip_should_pass:
                clip = video.clips[clip_idx]
                clip.qwen_rejection_stage = "classifier"
                if self._verbose:
                    logger.info(
                        f"Clip {clip.uuid} filtered out due to: {set(all_issues)} "
                        f"(classified as: {clip.qwen_type_classification or []})"
                    )
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_classifier += 1
            else:
                passing_clips.append(video.clips[clip_idx])

        for clip_idx in range(len(video.clips)):
            if clip_idx not in clip_results:
                clip = video.clips[clip_idx]
                clip.errors["qwen"] = "all_windows_failed_preparation"
                clip.qwen_rejection_stage = "classifier"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_classifier += 1
                logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")

        video.clips = passing_clips
        if self._clear_model_input_after:
            for clip in video.clips + video.filtered_clips:
                for window in clip.filter_windows:
                    window.model_input.clear()
                    window.mp4_bytes.drop()


def _clean_json_string(output_text: str) -> dict[str, str] | None:
    """Clean and fix common JSON formatting issues."""
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
