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

"""Performance Utitilies."""

from __future__ import annotations

import contextlib
import statistics
import time
from typing import TYPE_CHECKING

import attrs
import pandas as pd
from loguru import logger

from cosmos_curate.core.utils.misc import summarize
from cosmos_curate.core.utils.storage import storage_utils, writer_utils

if TYPE_CHECKING:
    from collections.abc import Generator

    from cosmos_curate.core.interfaces.stage_interface import CuratorStage


@attrs.define
class StagePerfStats:
    """Statistics for tracking stage performance metrics.

    Attributes:
        process_time: Total processing time in seconds.
        actor_idle_time: Time the actor spent idle in seconds.
        input_data_size_mb: Size of input data in megabytes.

    """

    process_time: float = 0.0
    actor_idle_time: float = 0.0
    input_data_size_mb: float = 0.0

    def __add__(self, other: StagePerfStats) -> StagePerfStats:
        """Add two StagePerfStats."""
        return StagePerfStats(
            process_time=self.process_time + other.process_time,
            actor_idle_time=self.actor_idle_time + other.actor_idle_time,
            input_data_size_mb=self.input_data_size_mb + other.input_data_size_mb,
        )

    def __radd__(self, other: int | StagePerfStats) -> StagePerfStats:
        """Add two StagePerfStats together, if right is 0, returns itself."""
        if other == 0:
            return self
        assert isinstance(other, StagePerfStats)
        return self.__add__(other)

    def reset(self) -> None:
        """Reset the stats."""
        self.process_time = 0.0
        self.actor_idle_time = 0.0
        self.input_data_size_mb = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert the stats to a dictionary."""
        return attrs.asdict(self)


def _summarize_perf_stats(
    task_stats: list[dict[str, StagePerfStats]],
) -> dict[str, dict[str, float]]:
    data = {}
    if len(task_stats) > 0:
        for stage in task_stats[0]:
            data[stage] = sum((x[stage] for x in task_stats), StagePerfStats()).to_dict()
    return data


def dump_and_write_perf_stats(
    task_stats: list[dict[str, StagePerfStats]],
    output_path: str | None,
    output_s3_profile_name: str,
) -> None:
    """Dump and write performance stats.

    Args:
        task_stats: List of task stats.
        output_path: Output path.
        output_s3_profile_name: Output S3 profile name.

    """
    if len(task_stats) == 0:
        return
    data = _summarize_perf_stats(task_stats)
    with summarize.turn_off_pandas_display_limits():
        pdf = pd.DataFrame.from_dict(data, orient="index")
        logger.info(f"Per-stage performance stats:\n{pdf!s}")
    if output_path is not None:
        client = storage_utils.get_storage_client(target_path=output_path, profile_name=output_s3_profile_name)
        dest = storage_utils.get_next_file("performance_stats", "json", output_path, client)
        writer_utils.write_json(
            data,
            dest,
            "performance stats",
            "pipeline",
            verbose=True,
            client=client,
        )


class StageTimer:
    """Tracker for stage stats.

    Tracks things at a per "process_data" call level.
    """

    def __init__(self, stage: CuratorStage) -> None:
        """Initialize the stage timer.

        Args:
            stage: The stage to track.

        """
        self._stage_name = str(stage.__class__.__name__)
        self._reset()
        self._last_active_time = time.time()
        self._initialized = False

    def _reset(self) -> None:
        self._num_gpus = 0.0
        self._num_cpus = 0.0
        self._num_samples = 0
        self._durations_s: list[float] = []
        self._source_video_duration_s = 0.0
        self._input_data_size_b = 0
        self._start = 0.0
        self._idle_time_s = 0.0
        self._startup_time_s = 0.0

    def reinit(self, stage: CuratorStage, stage_input_size: int = 1) -> None:
        """Reinitialize the stage timer.

        Args:
            stage: The stage to reinitialize the timer for.
            stage_input_size: The size of the stage input.

        """
        self._reset()
        self._num_gpus = stage.resources.gpus
        self._num_cpus = stage.resources.cpus
        self._input_data_size_b = stage_input_size
        self._start = time.time()
        if self._initialized:
            self._idle_time_s = self._start - self._last_active_time
        else:
            self._startup_time_s = self._start - self._last_active_time
        self._initialized = True

    @contextlib.contextmanager
    def time_process(self, num_samples: int = 1, source_video_duration_s: float = 0) -> Generator[None, None, None]:
        """Time the process of the stage.

        Args:
            num_samples: The number of samples to process.
            source_video_duration_s: The duration of the source video.

        """
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        self._num_samples += num_samples
        self._source_video_duration_s += source_video_duration_s
        for _ in range(num_samples):
            self._durations_s.append(duration / num_samples)

    def log_stats(self, *, verbose: bool = False) -> tuple[str, StagePerfStats]:
        """Log the stats of the stage.

        Args:
            verbose: Whether to log the stats.

        Returns:
            A tuple of the stage name and the stage performance stats.

        """
        num_gpus = self._num_gpus
        num_cpus = self._num_cpus
        end = time.time()
        process_data_dur_s = end - self._start
        num_samples = self._num_samples
        avg_dur_s = statistics.mean(self._durations_s) if self._durations_s else 0
        source_video_len_s = self._source_video_duration_s
        input_data_size_mb = self._input_data_size_b / 1024 / 1024
        start_time_s = self._startup_time_s
        idle_time_s = self._idle_time_s

        if verbose:
            logger.info(
                f"Stats: {process_data_dur_s=:.3f} - {num_samples=} - {avg_dur_s=:.3f} - "
                f"{num_gpus=} - {num_cpus=} - {start_time_s=:.3f} - {idle_time_s=:.3f} - "
                f"{source_video_len_s=:.1f} - {input_data_size_mb=:.3f}",
            )
        self._last_active_time = time.time()

        stage_perf_stats = StagePerfStats(
            process_time=process_data_dur_s,
            actor_idle_time=idle_time_s,
            input_data_size_mb=input_data_size_mb,
        )
        return self._stage_name, stage_perf_stats
