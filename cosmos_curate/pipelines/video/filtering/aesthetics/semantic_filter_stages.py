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
"""Aesthetic score filtering stages."""

import json
import pathlib
import re
from collections.abc import Iterable

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.filtering.aesthetics.semantic_filter_prompts import (  # type: ignore[import-untyped]
    FILTER_CRITERIA,
    VIDEO_TYPE_LABELS,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video


def parse_comma_separated_types(cs: str | None) -> list[str]:
    """Return list of non-empty stripped tokens from a comma-separated string."""
    if not cs:
        return []
    return [s.strip() for s in cs.split(",") if s.strip()]


def read_categories_file(path: str | None) -> str | None:
    """Read a newline-separated categories file and return a comma-separated string, or None if path is None."""
    if path is None:
        return None
    categories = [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    if not categories:
        msg = f"Categories file {path!r} is empty."
        raise ValueError(msg)
    return ",".join(categories)


def custom_categories_union(type_allow: str | None, type_block: str | None) -> str | None:
    """Return comma-separated sorted union of allow and block types, or None if empty."""
    categories = set(parse_comma_separated_types(type_allow)) | set(parse_comma_separated_types(type_block))
    return ",".join(sorted(categories)) if categories else None


class VllmFilteringStage(CuratorStage):
    """CPU post-processing stage for semantic (criteria-based) filtering.

    Reads captions written by VllmCaptionStage(use_filter_windows=True) and filters
    clips based on configurable criteria and a rejection threshold. Supports any
    model variant supported by VllmCaptionStage.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        user_prompt: str | None = None,
        filter_variant: str = "default",
        rejection_threshold: float = 0.5,
        *,
        score_only: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the vLLM semantic filtering stage.

        Args:
            model_variant: Model variant key used by the upstream VllmCaptionStage;
                used to read window.caption[model_variant].
            user_prompt: Comma-separated custom filter criteria. If None, uses the
                built-in FILTER_CRITERIA for filter_variant.
            filter_variant: Variant of filter criteria to use when user_prompt is None.
            rejection_threshold: Fraction of windows that must match a criterion before
                the clip is rejected.
            score_only: If True, annotate windows but do not filter clips.
            verbose: Whether to log per-clip decisions.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._filter_variant = filter_variant
        self._rejection_threshold = rejection_threshold
        self._user_prompt = user_prompt
        self._score_only = score_only
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str | None:
        """Get the conda environment name."""
        return None

    @nvtx.annotate("VllmFilteringStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Filter clips using captions already written by VllmCaptionStage."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            results: list[str] = []
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if not clip.filter_windows:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                        continue
                    for window_idx, window in enumerate(clip.filter_windows):
                        caption = window.caption.get(self._model_variant)
                        if caption is None:
                            logger.error(
                                f"Clip {clip.uuid} window {window_idx} has no caption for {self._model_variant!r}."
                            )
                            clip.errors[f"window-{window_idx}"] = "no_caption"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        results.append(caption)
                        idx += 1

                self._filter_clips(video, mapping, enumerate(results))

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered"
                )

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return tasks

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
                    # _parse_results normalizes keys to underscores; match that here.
                    criterion_key = criterion.replace(" ", "_")
                    if filtering_dict.get(criterion_key, "no") == "yes":
                        all_issues.add(criterion)
                        rejected_windows.add(window_idx)
                        window_specific_issues[criterion] = "yes"
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

        original_clips = list(video.clips)
        video.clips = passing_clips

        for clip_idx in range(len(original_clips)):
            if clip_idx not in clip_results:
                clip = original_clips[clip_idx]
                clip.errors["qwen"] = "all_windows_failed_preparation"
                clip.qwen_rejection_stage = "semantic"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_semantic += 1
                logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")
        for clip in video.clips + video.filtered_clips:
            for window in clip.filter_windows:
                window.model_input.clear()
                window.mp4_bytes.drop()


class VllmVideoClassifierStage(CuratorStage):
    """CPU post-processing stage for video-type (allow/block list) classification.

    Reads captions written by VllmCaptionStage(use_filter_windows=True) and classifies
    or filters clips based on allow/block lists. Supports any model variant supported
    by VllmCaptionStage.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_variant: str,
        rejection_threshold: float = 0.5,
        *,
        type_allow: str | None = None,
        type_block: str | None = None,
        custom_categories: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        clear_model_input_after: bool = True,
    ) -> None:
        """Initialize the vLLM video classifier stage.

        Args:
            model_variant: Model variant key used by the upstream VllmCaptionStage;
                used to read window.caption[model_variant].
            rejection_threshold: Threshold for clip rejection (block ratio).
            type_allow: Comma-separated video types to keep.
            type_block: Comma-separated video types to reject.
            custom_categories: If True, type_allow and type_block define the full
                category set; no validation against VIDEO_TYPE_LABELS.
            verbose: Whether to log per-clip decisions.
            log_stats: Whether to log performance statistics.
            clear_model_input_after: If True, clear window.model_input after classification.

        """
        self._timer = StageTimer(self)
        self._model_variant = model_variant
        self._clear_model_input_after = clear_model_input_after
        self._rejection_threshold = rejection_threshold
        self._verbose = verbose
        self._log_stats = log_stats
        # Normalize to underscores at init so lookup sites don't need to.
        self._type_allow: list[str] = [t.replace(" ", "_") for t in parse_comma_separated_types(type_allow)]
        self._type_block: list[str] = [t.replace(" ", "_") for t in parse_comma_separated_types(type_block)]
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

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str | None:
        """Get the conda environment name."""
        return None

    @nvtx.annotate("VllmVideoClassifierStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Classify and filter clips using captions already written by VllmCaptionStage."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            mapping: dict[int, tuple[int, int]] = {}
            results: list[str] = []
            idx = 0
            with self._timer.time_process(len(video.clips)):
                for clip_idx, clip in enumerate(video.clips):
                    if not clip.filter_windows:
                        logger.warning(f"Clip {clip.uuid} has no windows.")
                        clip.errors["windows"] = "empty"
                        continue
                    for window_idx, window in enumerate(clip.filter_windows):
                        caption = window.caption.get(self._model_variant)
                        if caption is None:
                            logger.error(
                                f"Clip {clip.uuid} window {window_idx} has no caption for {self._model_variant!r}."
                            )
                            clip.errors[f"window-{window_idx}"] = "no_caption"
                            continue
                        mapping[idx] = (clip_idx, window_idx)
                        results.append(caption)
                        idx += 1

                self._filter_clips(video, mapping, enumerate(results))

            if self._verbose:
                logger.info(
                    f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                    f"{len(video.clips)}/{len(video.filtered_clips)} clips passed/filtered"
                )

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    def _type_mode_filter_clip(  # noqa: C901, PLR0912
        self,
        video: Video,
        clip_idx: int,
        window_results: list[tuple[int, str]],
        all_issues: set[str],
        rejected_windows: set[int],
    ) -> bool:
        """Apply type filtering: allow list (keep if matches) and/or block list (reject if matches)."""
        # When no allow list is configured, clips pass by default unless blocked.
        has_allowed = not bool(self._type_allow)
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
            if self._type_allow and any(filtering_dict.get(t, "no") == "yes" for t in self._type_allow):
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

        original_clips = list(video.clips)
        video.clips = passing_clips

        for clip_idx in range(len(original_clips)):
            if clip_idx not in clip_results:
                clip = original_clips[clip_idx]
                clip.errors["qwen"] = "all_windows_failed_preparation"
                clip.qwen_rejection_stage = "classifier"
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_qwen_classifier += 1
                logger.warning(f"Clip {clip.uuid} had no successfully mapped windows; added to filtered_clips")
        if self._clear_model_input_after:
            for clip in video.clips + video.filtered_clips:
                for window in clip.filter_windows:
                    window.model_input.clear()
                    window.mp4_bytes.drop()


def _clean_json_string(output_text: str) -> dict[str, str] | None:
    """Clean and fix common JSON formatting issues."""
    # Strip thinking tokens — take only what comes after the last </think> tag.
    # If the model truncated mid-think (no closing tag), the output has no JSON.
    last_think_end = output_text.rfind("</think>")
    if last_think_end != -1:
        output_text = output_text[last_think_end + len("</think>") :]

    # Strip <answer> wrapper tags used by some models (e.g. cosmos_r1).
    output_text = re.sub(r"</?answer>", "", output_text)

    # Replace escaped quotes with single quotes
    output_text = output_text.replace('\\"', "'")

    # Remove any BOM or special characters at the start
    output_text = output_text.lstrip("\ufeff\n\r\t ")

    # Normalize line endings
    output_text = output_text.replace("\r\n", "\n")

    # Add missing commas between properties
    output_text = re.sub(r'"\s*\n\s*"', '",\n"', output_text)

    # Extract the first complete JSON object by tracking brace depth.
    # This handles models that output JSON multiple times (e.g. once as plain text
    # and once inside a markdown code fence) or append trailing text after the object.
    start = output_text.find("{")
    if start == -1:
        logger.error(f"No JSON object found in output: {output_text!r}")
        return None
    depth = 0
    end = -1
    for i in range(start, len(output_text)):
        if output_text[i] == "{":
            depth += 1
        elif output_text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        logger.error(f"Unmatched braces in output: {output_text!r}")
        return None
    output_text = output_text[start : end + 1].strip()

    try:
        return json.loads(output_text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}")
        logger.error(f"Output text: {output_text}")
        return None


def _parse_results(caption: str) -> dict[str, str] | None:
    """Parse the caption into a dictionary."""
    result = _clean_json_string(caption)
    if result is None:
        return None
    # Normalize keys: some models use spaces instead of underscores (e.g. "exterior building/cityscape")
    return {k.replace(" ", "_"): v for k, v in result.items()}
