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

from unittest.mock import MagicMock, patch

import pytest

from benchmarks.summary import make_summary_metrics, video_hours_per_day_per_gpu


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
    test_summary = {
        "num_input_videos": 100,
        "num_processed_videos": 95,
        "total_video_duration": 3600,
        "total_clip_duration": 1800,
        "max_clip_duration": 300,
        "pipeline_run_time": 60,
        "total_num_clips_filtered_by_motion": 10,
        "total_num_clips_filtered_by_aesthetic": 5,
        "total_num_clips_passed": 80,
        "total_num_clips_transcoded": 80,
        "total_num_clips_with_embeddings": 80,
        "total_num_clips_with_caption": 80,
        "total_num_clips_with_webp": 80,
    }
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
    result = make_summary_metrics(test_summary, test_num_nodes, test_gpus_per_node, caption=caption, env=test_env)

    # Assert
    expected_result = {
        **test_summary,
        "env": test_env,
        "num_nodes": test_num_nodes,
        "time": test_timestamp,
        "video_hours_per_day_per_gpu": test_video_hours_per_day_per_gpu,
        "caption": caption,
    }

    assert result == expected_result


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
        make_summary_metrics(incomplete_summary, 1, 1, caption=True, env="test")


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
    complete_summary = {
        "num_input_videos": 100,
        "num_processed_videos": 95,
        "total_video_duration": 3600,
        "total_clip_duration": 1800,
        "max_clip_duration": 300,
        "pipeline_run_time": 60,
        "total_num_clips_filtered_by_motion": 10,
        "total_num_clips_filtered_by_aesthetic": 5,
        "total_num_clips_passed": 80,
        "total_num_clips_transcoded": 80,
        "total_num_clips_with_embeddings": 80,
        "total_num_clips_with_caption": 80,
        "total_num_clips_with_webp": 80,
    }

    # Remove the specific key
    incomplete_summary = {k: v for k, v in complete_summary.items() if k != missing_key}

    # Act & Assert
    with pytest.raises(ValueError, match=f"Missing keys in summary.json: \\['{missing_key}'\\]"):
        make_summary_metrics(incomplete_summary, 1, 1, caption=True, env="test")
