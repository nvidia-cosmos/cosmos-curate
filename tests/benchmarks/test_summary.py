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
"""Test benchmarks/summary.py."""

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.summary import make_caption_quality_metrics, make_summary_metrics, video_hours_per_day_per_gpu


def _make_required_summary() -> dict[str, Any]:
    return {
        "num_input_videos": 100,
        "num_input_videos_selected": 95,
        "num_processed_videos": 95,
        "total_video_duration": 3600,
        "total_clip_duration": 1800,
        "max_clip_duration": 300,
        "pipeline_run_time": 60,
        "total_num_clips_filtered_by_motion": 10,
        "total_num_clips_filtered_by_aesthetic": 5,
        "total_num_clips_filtered_by_qwen_classifier": 0,
        "total_num_clips_filtered_by_qwen_semantic": 0,
        "total_num_clips_passed": 80,
        "total_num_clips_transcoded": 80,
        "total_num_clips_with_embeddings": 80,
        "total_num_clips_with_caption": 80,
        "total_num_clips_with_webp": 80,
    }


def _make_caption_quality_stats() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "split_video_pipeline",
        "caption_windows_checked": 5,
        "caption_status_counts": {
            "success": 2,
            "truncated": 1,
            "blocked": 1,
            "error": 1,
            "skipped": 0,
        },
        "caption_quality_flags_evaluated_count": 3,
        "caption_quality_flag_counts": {
            "flag_length_outlier": 1,
            "flag_repetition": 2,
            "flag_near_duplicate": 0,
        },
        "empty_caption_count": 1,
        "sentinel_caption_count": 2,
    }


def _make_summary_metrics_with_token_fields(
    token_fields: dict[str, Any],
    *,
    runtime_minutes: float = 10,
    num_nodes: int = 2,
    gpus_per_node: int = 4,
) -> dict[str, Any]:
    summary = {
        **_make_required_summary(),
        "pipeline_run_time": runtime_minutes,
        **token_fields,
    }
    return make_summary_metrics(
        summary,
        num_nodes,
        gpus_per_node,
        caption=True,
        env="nvcf",
        splitting_algorithm="transnetv2",
    )


@pytest.mark.parametrize(
    ("video_seconds", "runtime_minutes", "num_nodes", "gpus_per_node", "expected"),
    [
        (3600, 60, 1, 1, 24.0),  # 1 hour video, 1 hour runtime, 1 node, 1 GPU = 24 video hours/day/GPU
        (7200, 120, 2, 4, 3.0),  # 2 hour video, 2 hour runtime, 2 nodes, 4 GPUs = 3 video hours/day/GPU
        (1800, 30, 1, 2, 12.0),  # 0.5 hour video, 0.5 hour runtime, 1 node, 2 GPUs = 12 video hours/day/GPU
        (3600, 30, 1, 1, 48.0),  # 1 hour video, 0.5 hour runtime, 1 node, 1 GPU = 48 video hours/day/GPU
    ],
)
def test_video_hours_per_day_per_gpu(
    video_seconds: float, runtime_minutes: float, num_nodes: int, gpus_per_node: int, expected: float
) -> None:
    """Test video_hours_per_day_per_gpu calculation."""
    result = video_hours_per_day_per_gpu(video_seconds, runtime_minutes, num_nodes, gpus_per_node)
    assert result == expected


def test_make_caption_quality_metrics_flattens_valid_stats() -> None:
    """Flatten schema-v1 caption quality stats into benchmark metric fields."""
    result = make_caption_quality_metrics(_make_caption_quality_stats())

    assert result == {
        "caption_quality_stats_present": 1,
        "caption_windows_checked": 5,
        "caption_status_success": 2,
        "caption_status_truncated": 1,
        "caption_status_blocked": 1,
        "caption_status_error": 1,
        "caption_status_skipped": 0,
        "caption_quality_flags_evaluated_count": 3,
        "flag_length_outlier": 1,
        "flag_repetition": 2,
        "flag_near_duplicate": 0,
        "empty_caption_count": 1,
        "sentinel_caption_count": 2,
    }


def test_make_caption_quality_metrics_marks_missing_stats_absent() -> None:
    """Return the absent marker for missing caption quality stats."""
    assert make_caption_quality_metrics(None) == {"caption_quality_stats_present": 0}


def test_make_caption_quality_metrics_marks_non_dict_stats_absent() -> None:
    """Return the absent marker for parsed JSON that is not an object."""
    assert make_caption_quality_metrics([]) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize("schema_version", [2, True, "1", None])
def test_make_caption_quality_metrics_rejects_unrecognized_schema(schema_version: object) -> None:
    """Reject caption quality stats that do not match the supported schema version."""
    stats = {**_make_caption_quality_stats(), "schema_version": schema_version}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize(
    "caption_status_counts",
    [
        None,
        [],
        {"success": 5},
        {
            "success": 2,
            "truncated": 1,
            "blocked": 1,
            "error": 1,
            "skipped": 0,
            "unknown": 0,
        },
    ],
)
def test_make_caption_quality_metrics_rejects_bad_status_counts(caption_status_counts: object) -> None:
    """Reject missing or malformed caption status counts."""
    stats = {**_make_caption_quality_stats(), "caption_status_counts": caption_status_counts}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


def test_make_caption_quality_metrics_rejects_status_sum_mismatch() -> None:
    """Reject stats whose status counts do not reconcile with checked windows."""
    stats = {**_make_caption_quality_stats(), "caption_windows_checked": 6}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


def test_make_caption_quality_metrics_rejects_empty_sentinel_over_ok_status_count() -> None:
    """Reject empty and sentinel counters that exceed success plus truncated windows."""
    stats = {**_make_caption_quality_stats(), "sentinel_caption_count": 3}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize(
    ("counter_key", "counter_value"),
    [
        ("caption_windows_checked", True),
        ("caption_windows_checked", "5"),
        ("caption_windows_checked", -1),
        ("empty_caption_count", True),
        ("empty_caption_count", 1.5),
        ("sentinel_caption_count", -1),
        ("caption_quality_flags_evaluated_count", True),
        ("caption_quality_flags_evaluated_count", "3"),
        ("caption_quality_flags_evaluated_count", -1),
    ],
)
def test_make_caption_quality_metrics_rejects_bad_top_level_counters(counter_key: str, counter_value: object) -> None:
    """Reject bool, non-int, and negative top-level caption quality counters."""
    stats = {**_make_caption_quality_stats(), counter_key: counter_value}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize("status_count", [True, "2", -1])
def test_make_caption_quality_metrics_rejects_bad_status_counter(status_count: object) -> None:
    """Reject bool, non-int, and negative caption status counters."""
    stats = _make_caption_quality_stats()
    stats["caption_status_counts"]["success"] = status_count

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize(
    "flag_counts",
    [
        None,
        [],
        {"flag_length_outlier": 1, "flag_repetition": 2},
        {
            "flag_length_outlier": 1,
            "flag_repetition": 2,
            "flag_near_duplicate": 0,
            "unknown": 0,
        },
    ],
)
def test_make_caption_quality_metrics_rejects_bad_flag_map(flag_counts: object) -> None:
    """Reject missing or malformed caption quality flag counter maps."""
    stats = {**_make_caption_quality_stats(), "caption_quality_flag_counts": flag_counts}

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


@pytest.mark.parametrize("flag_count", [True, "1", -1, 4])
def test_make_caption_quality_metrics_rejects_bad_flag_counter(flag_count: object) -> None:
    """Reject invalid flag numerators and values above the evaluated denominator."""
    stats = _make_caption_quality_stats()
    stats["caption_quality_flag_counts"]["flag_length_outlier"] = flag_count

    assert make_caption_quality_metrics(stats) == {"caption_quality_stats_present": 0}


def test_make_caption_quality_metrics_accepts_disabled_zero_flag_counts() -> None:
    """A disabled producer should remain valid with a zero denominator and numerators."""
    stats = _make_caption_quality_stats()
    stats["caption_quality_flags_evaluated_count"] = 0
    stats["caption_quality_flag_counts"] = {
        "flag_length_outlier": 0,
        "flag_repetition": 0,
        "flag_near_duplicate": 0,
    }

    metrics = make_caption_quality_metrics(stats)

    assert metrics["caption_quality_stats_present"] == 1
    assert metrics["caption_quality_flags_evaluated_count"] == 0
    assert metrics["flag_length_outlier"] == 0
    assert metrics["flag_repetition"] == 0
    assert metrics["flag_near_duplicate"] == 0


@pytest.mark.parametrize(
    ("caption"),
    [
        (True),
        (False),
    ],
)
@patch("benchmarks.summary.datetime")
@patch("benchmarks.summary.video_hours_per_day_per_gpu")
def test_make_summary_metrics(mock_video_calc: MagicMock, mock_datetime: MagicMock, *, caption: bool) -> None:
    """Test make_summary_metrics function."""
    # Arrange
    test_summary = _make_required_summary()
    test_num_nodes = 2
    test_gpus_per_node = 4
    test_env = "nvcf"
    test_timestamp = "2023-01-01T12:00:00.000000Z"
    test_video_hours_per_day_per_gpu = 12.0

    # Mock datetime
    mock_now = MagicMock()
    mock_now.strftime.return_value = test_timestamp
    mock_datetime.now.return_value = mock_now

    # Mock video calculation
    mock_video_calc.return_value = test_video_hours_per_day_per_gpu

    # Act
    result = make_summary_metrics(
        test_summary,
        test_num_nodes,
        test_gpus_per_node,
        caption=caption,
        env=test_env,
        splitting_algorithm="transnetv2",
    )

    # Assert
    expected_result = {
        **test_summary,
        "env": test_env,
        "num_nodes": test_num_nodes,
        "gpus_per_node": test_gpus_per_node,
        "time": test_timestamp,
        "video_hours_per_day_per_gpu": test_video_hours_per_day_per_gpu,
        "caption": int(caption),
        "splitting_algorithm": "transnetv2",
    }

    assert result == expected_result


def test_make_summary_metrics_with_nonzero_output_tokens() -> None:
    """Preserve raw token fields and compute derived token metrics."""
    result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": 1200,
            "total_output_tokens": 600,
            "total_num_caption_windows": 12,
            "output_tokens_per_s": 999.0,
        }
    )

    assert result["gpus_per_node"] == 4
    assert result["total_prompt_tokens"] == 1200
    assert result["total_output_tokens"] == 600
    assert result["total_num_caption_windows"] == 12
    assert result["output_tokens_per_s"] == 999.0
    assert result["avg_prompt_tokens_per_window"] == 100.0
    assert result["avg_output_tokens_per_window"] == 50.0
    assert result["output_tokens_per_s_per_gpu"] == pytest.approx(600 / (10 * 60 * 2 * 4))


def test_make_summary_metrics_with_zero_output_tokens() -> None:
    """Preserve zero token totals while omitting generated-output throughput metrics."""
    result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_num_caption_windows": 3,
        }
    )

    assert result["total_prompt_tokens"] == 0
    assert result["total_output_tokens"] == 0
    assert result["total_num_caption_windows"] == 3
    assert result["avg_prompt_tokens_per_window"] == 0.0
    assert result["avg_output_tokens_per_window"] == 0.0
    assert "output_tokens_per_s" not in result
    assert "output_tokens_per_s_per_gpu" not in result

    prompt_only_result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": 120,
            "total_output_tokens": 0,
            "total_num_caption_windows": 3,
        }
    )

    assert prompt_only_result["total_prompt_tokens"] == 120
    assert prompt_only_result["total_output_tokens"] == 0
    assert prompt_only_result["total_num_caption_windows"] == 3
    assert prompt_only_result["avg_prompt_tokens_per_window"] == 40.0
    assert prompt_only_result["avg_output_tokens_per_window"] == 0.0
    assert "output_tokens_per_s_per_gpu" not in prompt_only_result

    zero_window_result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_num_caption_windows": 0,
        }
    )

    assert zero_window_result["total_prompt_tokens"] == 0
    assert zero_window_result["total_output_tokens"] == 0
    assert zero_window_result["total_num_caption_windows"] == 0
    assert "avg_prompt_tokens_per_window" not in zero_window_result
    assert "avg_output_tokens_per_window" not in zero_window_result
    assert "output_tokens_per_s_per_gpu" not in zero_window_result


def test_make_summary_metrics_with_absent_token_fields() -> None:
    """Omit token metrics when older summary.json files do not include token fields."""
    result = _make_summary_metrics_with_token_fields({})

    assert "total_prompt_tokens" not in result
    assert "total_output_tokens" not in result
    assert "total_num_caption_windows" not in result
    assert "output_tokens_per_s" not in result
    assert "avg_prompt_tokens_per_window" not in result
    assert "avg_output_tokens_per_window" not in result
    assert "output_tokens_per_s_per_gpu" not in result


def test_make_summary_metrics_with_token_totals_but_no_caption_windows() -> None:
    """Preserve token totals and omit per-window averages when caption windows are absent."""
    result = _make_summary_metrics_with_token_fields({"total_prompt_tokens": 1200, "total_output_tokens": 600})

    assert result["total_prompt_tokens"] == 1200
    assert result["total_output_tokens"] == 600
    assert "avg_prompt_tokens_per_window" not in result
    assert "avg_output_tokens_per_window" not in result
    assert result["output_tokens_per_s_per_gpu"] == pytest.approx(600 / (10 * 60 * 2 * 4))


def test_make_summary_metrics_omits_derived_fields_for_invalid_optional_values() -> None:
    """Preserve raw passthrough values while omitting derived fields for invalid optional values."""
    result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": None,
            "total_output_tokens": None,
            "total_num_caption_windows": None,
        }
    )

    assert result["total_prompt_tokens"] is None
    assert result["total_output_tokens"] is None
    assert result["total_num_caption_windows"] is None
    assert "avg_prompt_tokens_per_window" not in result
    assert "avg_output_tokens_per_window" not in result
    assert "output_tokens_per_s_per_gpu" not in result


def test_make_summary_metrics_omits_derived_fields_for_non_finite_optional_values() -> None:
    """Preserve raw passthrough values while omitting derived fields for nan and inf inputs."""
    result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": float("nan"),
            "total_output_tokens": float("inf"),
            "total_num_caption_windows": 12,
        }
    )

    assert math.isnan(result["total_prompt_tokens"])
    assert math.isinf(result["total_output_tokens"])
    assert result["total_num_caption_windows"] == 12
    assert "avg_prompt_tokens_per_window" not in result
    assert "avg_output_tokens_per_window" not in result
    assert "output_tokens_per_s_per_gpu" not in result


def test_make_summary_metrics_omits_derived_fields_for_bool_optional_values() -> None:
    """Preserve raw passthrough values while excluding bool values from derived metrics."""
    result = _make_summary_metrics_with_token_fields(
        {
            "total_prompt_tokens": True,
            "total_output_tokens": True,
            "total_num_caption_windows": 12,
        }
    )

    assert result["total_prompt_tokens"] is True
    assert result["total_output_tokens"] is True
    assert result["total_num_caption_windows"] == 12
    assert "avg_prompt_tokens_per_window" not in result
    assert "avg_output_tokens_per_window" not in result
    assert "output_tokens_per_s_per_gpu" not in result


def test_make_summary_metrics_missing_keys() -> None:
    """Test make_summary_metrics function with missing keys."""
    # Arrange
    incomplete_summary = {
        "num_input_videos": 100,
        "num_processed_videos": 95,
        # Missing other required keys
    }

    # Act & Assert
    with pytest.raises(ValueError, match=r"Missing keys in summary\.json"):
        make_summary_metrics(incomplete_summary, 1, 1, caption=True, env="test", splitting_algorithm="transnetv2")


@pytest.mark.parametrize(
    ("missing_key"),
    [
        "num_input_videos",
        "total_video_duration",
        "pipeline_run_time",
        "total_num_clips_with_webp",
    ],
)
def test_make_summary_metrics_specific_missing_key(missing_key: str) -> None:
    """Test make_summary_metrics with specific missing keys."""
    # Arrange
    complete_summary = _make_required_summary()

    # Remove the specific key
    incomplete_summary = {k: v for k, v in complete_summary.items() if k != missing_key}

    # Act & Assert
    with pytest.raises(ValueError, match=f"Missing keys in summary.json: \\['{missing_key}'\\]"):
        make_summary_metrics(incomplete_summary, 1, 1, caption=True, env="test", splitting_algorithm="transnetv2")
