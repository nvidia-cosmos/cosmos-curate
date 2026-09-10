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
"""Process summary.json written by the pipeline."""

import math
from datetime import UTC, datetime
from typing import Any, TypeGuard

_CAPTION_QUALITY_SCHEMA_VERSION = 1
_CAPTION_STATUS_KEYS = ("success", "truncated", "blocked", "error", "skipped")
_CAPTION_QUALITY_FLAG_KEYS = ("flag_length_outlier", "flag_repetition", "flag_near_duplicate")


def video_hours_per_day_per_gpu(
    video_seconds: float, runtime_minutes: float, num_nodes: int, gpus_per_node: int
) -> float:
    """Calculate video hours per day per GPU.

    Args:
        video_seconds: Total seconds of video processed.
        runtime_minutes: Total pipeline runtime in minutes.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.

    Returns:
        Video hours per day per GPU.

    """
    return (video_seconds * 24) / (60 * runtime_minutes * num_nodes * gpus_per_node)


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Return whether a value can be used as a finite numeric input."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_non_negative_int(value: object) -> TypeGuard[int]:
    """Return whether a value is a non-bool, non-negative integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _caption_quality_absent_metrics() -> dict[str, Any]:
    """Return metrics for missing or unusable caption quality stats."""
    return {"caption_quality_stats_present": 0}


def make_caption_quality_metrics(stats: object | None) -> dict[str, Any]:  # noqa: PLR0911
    """Flatten caption_quality_stats.json counters into benchmark metrics."""
    if not isinstance(stats, dict):
        return _caption_quality_absent_metrics()

    schema_version = stats.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _CAPTION_QUALITY_SCHEMA_VERSION
    ):
        return _caption_quality_absent_metrics()

    caption_windows_checked = stats.get("caption_windows_checked")
    empty_caption_count = stats.get("empty_caption_count")
    sentinel_caption_count = stats.get("sentinel_caption_count")
    caption_status_counts = stats.get("caption_status_counts")
    caption_quality_flags_evaluated_count = stats.get("caption_quality_flags_evaluated_count")
    caption_quality_flag_counts = stats.get("caption_quality_flag_counts")
    if (
        not _is_non_negative_int(caption_windows_checked)
        or not _is_non_negative_int(empty_caption_count)
        or not _is_non_negative_int(sentinel_caption_count)
        or not isinstance(caption_status_counts, dict)
        or set(caption_status_counts) != set(_CAPTION_STATUS_KEYS)
        or not _is_non_negative_int(caption_quality_flags_evaluated_count)
        or not isinstance(caption_quality_flag_counts, dict)
        or set(caption_quality_flag_counts) != set(_CAPTION_QUALITY_FLAG_KEYS)
    ):
        return _caption_quality_absent_metrics()

    caption_status_metrics: dict[str, Any] = {}
    caption_status_values: dict[str, int] = {}
    caption_status_total = 0
    for status_key in _CAPTION_STATUS_KEYS:
        status_count = caption_status_counts[status_key]
        if not _is_non_negative_int(status_count):
            return _caption_quality_absent_metrics()
        caption_status_metrics[f"caption_status_{status_key}"] = status_count
        caption_status_values[status_key] = status_count
        caption_status_total += status_count

    caption_quality_flag_metrics: dict[str, Any] = {}
    for flag_key in _CAPTION_QUALITY_FLAG_KEYS:
        flag_count = caption_quality_flag_counts[flag_key]
        if not _is_non_negative_int(flag_count) or flag_count > caption_quality_flags_evaluated_count:
            return _caption_quality_absent_metrics()
        caption_quality_flag_metrics[flag_key] = flag_count

    if (
        caption_status_total != caption_windows_checked
        or empty_caption_count + sentinel_caption_count
        > caption_status_values["success"] + caption_status_values["truncated"]
    ):
        return _caption_quality_absent_metrics()

    return {
        "caption_quality_stats_present": 1,
        "caption_windows_checked": caption_windows_checked,
        **caption_status_metrics,
        "caption_quality_flags_evaluated_count": caption_quality_flags_evaluated_count,
        **caption_quality_flag_metrics,
        "empty_caption_count": empty_caption_count,
        "sentinel_caption_count": sentinel_caption_count,
    }


def _make_token_metrics(
    summary: dict[str, Any], runtime_minutes: float, num_nodes: int, gpus_per_node: int
) -> dict[str, Any]:
    """Build optional token metrics from summary.json.

    The per-GPU token metrics are benchmark-owned run-level rates, derived from
    total output tokens over total GPU time for the run.

    Args:
        summary: summary.json written by the pipeline.
        runtime_minutes: Total pipeline runtime in minutes.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.

    Returns:
        Raw token passthrough fields and derived token metrics when available.

    """
    token_metrics: dict[str, Any] = {}
    raw_token_keys = [
        "total_prompt_tokens",
        "total_output_tokens",
        "total_num_caption_windows",
        "output_tokens_per_s",
    ]
    for key in raw_token_keys:
        if key in summary:
            token_metrics[key] = summary[key]

    num_caption_windows = summary.get("total_num_caption_windows")
    total_prompt_tokens = summary.get("total_prompt_tokens")
    total_output_tokens = summary.get("total_output_tokens")
    if _is_finite_number(num_caption_windows) and num_caption_windows > 0:
        if _is_finite_number(total_prompt_tokens):
            token_metrics["avg_prompt_tokens_per_window"] = total_prompt_tokens / num_caption_windows
        if _is_finite_number(total_output_tokens):
            token_metrics["avg_output_tokens_per_window"] = total_output_tokens / num_caption_windows

    if (
        _is_finite_number(total_output_tokens)
        and total_output_tokens > 0
        and _is_finite_number(runtime_minutes)
        and runtime_minutes > 0
        and _is_finite_number(num_nodes)
        and num_nodes > 0
        and _is_finite_number(gpus_per_node)
        and gpus_per_node > 0
    ):
        output_tokens_per_s_per_gpu = total_output_tokens / (runtime_minutes * 60 * num_nodes * gpus_per_node)
        token_metrics["output_tokens_per_s_per_gpu"] = output_tokens_per_s_per_gpu

    return token_metrics


def make_summary_metrics(  # noqa: PLR0913
    summary: dict[str, Any], num_nodes: int, gpus_per_node: int, *, caption: bool, env: str, splitting_algorithm: str
) -> dict[str, Any]:
    """Get metrics from summary.json.

    Args:
        summary: summary.json written by the pipeline.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.
        caption: Whether captions are enabled.
        env: Environment, nvcf or slurm.
        splitting_algorithm: Splitting algorithm used.

    Returns:
        Summary metrics from the pipeline

    """
    # Make sure all the keys are present in the summary.json
    keys_from_json = [
        "num_input_videos",
        "num_processed_videos",
        "total_video_duration",
        "total_clip_duration",
        "max_clip_duration",
        "pipeline_run_time",
        "total_num_clips_filtered_by_motion",
        "total_num_clips_filtered_by_aesthetic",
        "total_num_clips_filtered_by_qwen_classifier",
        "total_num_clips_filtered_by_qwen_semantic",
        "total_num_clips_passed",
        "total_num_clips_transcoded",
        "total_num_clips_with_embeddings",
        "total_num_clips_with_caption",
        "total_num_clips_with_webp",
    ]

    missing_keys = [key for key in keys_from_json if key not in summary]
    if missing_keys:
        msg = f"Missing keys in summary.json: {missing_keys}"
        raise ValueError(msg)

    data = {key: summary[key] for key in keys_from_json}
    # num_input_videos_selected is handled as an optional compatibility field.
    # It should stay out of keys_from_json so older summary.json files remain valid,
    # and be conditionally added to data only when present.
    if "num_input_videos_selected" in summary:
        data["num_input_videos_selected"] = summary["num_input_videos_selected"]

    # TODO: should summary data metrics use the same units for all measurements?
    video_seconds = data["total_video_duration"]
    runtime_minutes = data["pipeline_run_time"]

    data.update(
        {
            "env": env,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "video_hours_per_day_per_gpu": video_hours_per_day_per_gpu(
                video_seconds, runtime_minutes, num_nodes, gpus_per_node
            ),
            "caption": int(caption),
            "splitting_algorithm": splitting_algorithm,
        }
    )
    data.update(_make_token_metrics(summary, runtime_minutes, num_nodes, gpus_per_node))

    return data
