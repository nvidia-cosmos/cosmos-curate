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

from datetime import UTC, datetime
from typing import Any


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


def make_summary_metrics(
    summary: dict[str, Any], num_nodes: int, gpus_per_node: int, *, caption: bool, env: str
) -> dict[str, Any]:
    """Get metrics from summary.json.

    Args:
        summary: summary.json written by the pipeline.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.
        caption: Whether captions are enabled.
        env: Environment, nvcf or slurm.

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

    # TODO: should summary data metrics use the same units for all measurements?
    video_seconds = data["total_video_duration"]
    runtime_minutes = data["pipeline_run_time"]

    data.update(
        {
            "env": env,
            "num_nodes": num_nodes,
            "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "video_hours_per_day_per_gpu": video_hours_per_day_per_gpu(
                video_seconds, runtime_minutes, num_nodes, gpus_per_node
            ),
            "caption": int(caption),
        }
    )

    return data
