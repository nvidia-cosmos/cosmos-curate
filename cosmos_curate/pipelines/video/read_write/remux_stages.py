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
"""Remuxing stages for video pipelines."""

import subprocess
import tempfile
from math import ceil
from pathlib import Path

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.utils.data_model import (
    SplitPipeTask,
    Video,
)

REMUX_FORMATS = {"mpegts"}


def remux_to_mp4(source_bytes: bytes, threads: int = 1) -> bytes:
    """Remux a video to a MP4 container using ffmpeg.

    Notes:
    A moov atom is placed at the beginning of the file. This atom contains
    information about the video, such as the duration, number of frames,
    time base, etc.

    By default, ffmpeg will place the moov atom at the end of the file, and
    the entire stream must be read to the end to start decoding.

    By placing the moov atom at the beginning, decode can start immediately.

    To achieve this, FFmpeg transcodes the entire stream, and then writes the
    moov atom at the end.

    And then the moov atom is moved to the beginning of the file, which
    requires that the output file is seekable.

    Thus, we incur overhead of file io when writing to the temp file instead
    of being able to operate purely in memory.

    Args:
        source_bytes: The bytes of the input video (e.g., MPEG-TS).
        threads: The number of threads to use for ffmpeg.

    Returns:
        The bytes of the remuxed MP4 video.

    """
    # ffmpeg needs the output to be seekable, so write ffmpeg output to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as output_file:
        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            f"{threads}",
            "-fflags",
            "+genpts",
            "-i",
            "-",  # Read from stdin
            "-c",
            "copy",  # Copy streams without re-encoding
            "-movflags",
            "+faststart",  # Place the moov atom at the beginning of the file
            "-f",
            "mp4",  # Force MP4 format (since stdout is ambiguous)
            f"{output_file.name}",  # Write to disk - ffmpeg needs output to be seekable
        ]

        logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
        proc = subprocess.run(  # noqa: S603
            cmd, input=source_bytes, capture_output=True, check=False
        )

        if proc.returncode != 0:
            msg = f"ffmpeg failed with return code {proc.returncode}:\n{proc.stderr.decode('utf-8')}"
            raise RuntimeError(msg)

        stderr_output = proc.stderr.decode("utf-8", errors="replace")
        logger.debug(f"ffmpeg stderr:\n{stderr_output}")
        return Path(output_file.name).read_bytes()


def remux_if_needed(video: Video, threads: int) -> None:
    """Remux the video if it is not in the correct format.

    Args:
        video: The video to remux, modified in place.
        threads: The number of threads to use for ffmpeg.

    """
    if video.source_bytes is None:
        msg = "Video source bytes are not set"
        raise ValueError(msg)

    if not video.metadata:
        logger.warning(f"Video {video.input_video} has no metadata, skipping remux")
        return

    format_name = video.metadata.format_name.lower() if video.metadata.format_name else "unknown"

    if any(remux_format in format_name for remux_format in REMUX_FORMATS):
        logger.info(f"Video {video.input_video} is in `{format_name}` format, remuxing to mp4")
        video.source_bytes = remux_to_mp4(video.source_bytes, threads=threads)
        video.populate_metadata()


class RemuxStage(CuratorStage):
    """Remuxing stage for video pipelines, no-op if the video is already in the correct format."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the video downloader stage.

        Args:
            input_path: Path to input videos.
            input_s3_profile_name: S3 profile name for input.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=1)

    @nvtx.annotate("RemuxStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Remux the video if it is not in the correct format."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())

            with self._timer.time_process():
                try:
                    remux_if_needed(task.video, threads=ceil(self.resources.cpus))
                except Exception as e:  # noqa: BLE001
                    task.video.errors["remux"] = str(e)
                    logger.exception(f"Failed to remux video {task.video.input_video}")

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
