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
"""Clip extraction stages."""

import copy
import pathlib
import subprocess
import uuid

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils import grouping
from cosmos_curate.core.utils.runtime.operation_utils import make_pipeline_temporary_dir
from cosmos_curate.core.utils.runtime.performance_utils import StageTimer
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
)


class ClipTranscodingStage(CuratorStage):
    """Stage that transcodes video clips into a standardized format.

    This stage handles the conversion of video clips using FFmpeg, supporting both
    software (libopenh264) and hardware (NVENC) encoding with configurable parameters.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_cpus_per_worker: float = 6.0,
        encoder: str = "libopenh264",
        encoder_threads: int = 1,
        encode_batch_size: int = 16,
        nb_streams_per_gpu: int = 3,
        *,
        use_hwaccel: bool = False,
        use_input_bit_rate: bool = False,
        num_clips_per_chunk: int = 32,
        verbose: bool = False,
        ffmpeg_verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip transcoding stage.

        Args:
            num_cpus_per_worker: Number of CPUs per worker.
            encoder: Video encoder to use.
            encoder_threads: Number of threads per encoder.
            encode_batch_size: Number of clips to encode in parallel.
            nb_streams_per_gpu: Number of streams per GPU.
            use_hwaccel: Whether to use hardware acceleration.
            use_input_bit_rate: Whether to use input video bit rate.
            num_clips_per_chunk: Number of clips per chunk.
            verbose: Whether to print verbose logs.
            ffmpeg_verbose: Whether to print FFmpeg verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._num_cpus_per_worker = num_cpus_per_worker
        self._encoder = encoder
        self._encoder_threads = encoder_threads
        self._encode_batch_size = encode_batch_size
        self._nb_streams_per_gpu = nb_streams_per_gpu
        self._use_hwaccel = use_hwaccel
        self._use_input_bit_rate = use_input_bit_rate
        self._num_clips_per_chunk = num_clips_per_chunk
        self._verbose = verbose
        self._ffmpeg_verbose = ffmpeg_verbose
        self._log_stats = log_stats
        if encoder not in {"libopenh264", "h264_nvenc"}:
            error_msg = f"Expected encoder of `libopenh264` or `h264_nvenc`. Got {encoder}"
            raise ValueError(error_msg)

    @nvtx.annotate("ClipTranscodingStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process the data for the clip transcoding stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed task.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            if video.source_bytes is None:
                error_msg = "Please load video!"
                raise ValueError(error_msg)
            with self._timer.time_process(
                len(video.clips),
                video.metadata.duration if video.metadata.duration else 0,
            ):
                if not video.clips:
                    logger.warning(f"No clips to transcode for {video.input_video}. Skipping...")
                    video.source_bytes = None
                    continue
                with make_pipeline_temporary_dir(sub_dir="transcode") as tmp_dir:
                    # write video to file
                    video_file = tmp_dir / "input.mp4"
                    video_file.write_bytes(video.source_bytes)
                    force_pix_fmt = video.is_10_bit_color() or False

                    # use input video bit-rate
                    use_bit_rate = None
                    if self._use_input_bit_rate:
                        use_bit_rate = str(video.metadata.bit_rate_k) + "K"

                    # extract clips in batches
                    for i in range(0, len(video.clips), self._encode_batch_size):
                        batch = video.clips[i : i + self._encode_batch_size]
                        self._extract_clips(
                            tmp_dir,
                            video_file.name,
                            force_pix_fmt=force_pix_fmt,
                            use_bit_rate=use_bit_rate,
                            clips=batch,
                            input_video=str(video.input_video),
                        )
            # we are done with source_bytes
            video.source_bytes = None

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        output_tasks = []
        for task in tasks:
            # consider cracking into smaller chunks of clips
            clip_durations = [x.duration for x in task.video.clips]
            if len(clip_durations) > 0:
                logger.info(
                    f"video {task.video.input_video} has {len(task.video.clips)} "
                    f"clips and weight={task.weight:.2f}; "
                    f"min-clip={min(clip_durations):.2f}s, "
                    f"max-clip={max(clip_durations):.1f}s.",
                )
            clip_chunks = list(
                grouping.split_by_chunk_size(
                    task.video.clips,
                    self._num_clips_per_chunk * 8,
                    lambda x: int(x.span[1] - x.span[0]),
                ),
            )
            for idx in range(len(clip_chunks)):
                subtask = SplitPipeTask(
                    video=Video(
                        input_video=task.video.input_video,
                        metadata=task.video.metadata,
                        clips=clip_chunks[idx],
                        num_total_clips=len(task.video.clips),
                        num_clip_chunks=len(clip_chunks),
                        clip_chunk_index=idx,
                        errors=copy.deepcopy(task.video.errors),
                    ),
                    stage_perf=copy.deepcopy(task.stage_perf),
                )
                if idx > 0:
                    for stats in subtask.stage_perf.values():
                        stats.reset()
                if self._verbose:
                    logger.info(
                        f"Spawning subtask {idx} with {len(subtask.video.clips)} clips and weight={subtask.weight:.2f}",
                    )
                output_tasks.append(subtask)
            logger.info(f"Creating {len(clip_chunks)} tasks for downstream from {task.video.input_video}.")

        return output_tasks

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        if self._encoder == "h264_nvenc" or self._use_hwaccel:
            if self._nb_streams_per_gpu > 0:
                return CuratorStageResource(gpus=1.0 / self._nb_streams_per_gpu)
            return CuratorStageResource(gpus=1.0)
        return CuratorStageResource(cpus=self._num_cpus_per_worker)

    @nvtx.annotate("ClipLoadingStage:_extract_clips")  # type: ignore[misc]
    def _extract_clips(  # noqa: C901, PLR0912, PLR0913
        self,
        working_dir: pathlib.Path,
        video_filename: str,
        *,
        force_pix_fmt: bool,
        use_bit_rate: str | None,
        clips: list[Clip],
        input_video: str,
    ) -> None:
        # construct ffmpeg command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning" if self._ffmpeg_verbose else "error",
        ]

        for i, clip in enumerate(clips):
            # set decoder threads
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self._encoder_threads)])
            # hwaccel needs to specified before each input
            if self._use_hwaccel:
                if self._encoder == "h264_nvenc":
                    command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
                else:
                    command.extend(["-hwaccel", "auto"])
            start_s, end_s = clip.span
            command.extend(
                [
                    "-ss",
                    str(start_s),
                    "-to",
                    str(end_s),
                    "-i",
                    video_filename,
                    "-map",
                    f"{i}:v:0",
                    "-c:v",
                    self._encoder,
                ],
            )
            if use_bit_rate is not None:
                command.extend(["-b:v", use_bit_rate])
            else:
                command.extend(["-b:v", "4M"])
            if self._encoder == "h264_nvenc":
                # IMPORTANT! these settings are necessary for high quality!
                command.extend(
                    [
                        "-rc:v",
                        "vbr",
                        "-cq:v",
                        "21",
                        "-tune",
                        "hq",
                        "-b_ref_mode",
                        "middle",
                        "-temporal-aq",
                        "1",
                        "-rc-lookahead",
                        "20",
                        "-spatial-aq",
                        "1",
                    ],
                )
                # To fix `10 bit encode not supported` error
                if force_pix_fmt:
                    command.extend(["-pix_fmt", "yuv420p"])
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self._encoder_threads)])
            command.extend(
                [
                    "-map",
                    f"{i}:a:0?",
                    "-c:a",
                    "copy",
                    f"{clip.uuid}.mp4",
                ],
            )

        # run ffmpeg command
        try:
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self._ffmpeg_verbose:
                logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed with return code {e.returncode} on {input_video}")
            logger.warning(f"Command: {' '.join(command)}")
            if e.output:
                logger.warning(f"Error output: {e.output.decode('utf-8')}")
            for clip in clips:
                clip.errors["transcode"] = e.output.decode("utf-8") if e.output else str(e)
            return

        # read clips back into memory
        for clip in clips:
            clip.buffer = (working_dir / f"{clip.uuid}.mp4").read_bytes()


class FixedStrideExtractorStage(CuratorStage):
    """Stage that extracts video clips using fixed-length intervals.

    This stage splits videos into clips of specified length and stride, ensuring
    each clip meets minimum length requirements and optionally limiting total clips.
    """

    def __init__(  # noqa: PLR0913
        self,
        clip_len_s: float = 10,
        clip_stride_s: float = 10,
        min_clip_length_s: float = 10,
        limit_clips: int = 0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the fixed stride extractor stage.

        Args:
            clip_len_s: clip length.
            clip_stride_s: Stride length.
            min_clip_length_s: Minimum clip length. If raw video is smaller, will yield no spans.
            log_stats: Whether to log statistics. Default False.
            limit_clips: limit clips
            verbose: verbose

        """
        self._timer = StageTimer(self)
        self.clip_stride_s = clip_stride_s
        assert clip_stride_s
        self.clip_len_s = clip_len_s
        self.min_clip_length_s = min_clip_length_s
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Process the data for the fixed stride extractor stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            if video.source_bytes is None:
                error_msg = "Please load video bytes!"
                raise ValueError(error_msg)

            if not video.has_metadata():
                logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
                video.errors["metadata"] = "incomplete"
                continue

            with self._timer.time_process():
                s3_file = video.input_video
                assert video.metadata.num_frames  # silence mypy
                assert video.metadata.framerate  # silence mypy
                duration = video.metadata.num_frames / video.metadata.framerate if video.metadata.framerate > 0 else -1

                clip_start = 0.0
                clip_bounds: list[tuple[float, float]] = []
                while clip_start < duration:
                    clip_end = min(clip_start + self.clip_len_s, duration)
                    if (clip_end - clip_start) >= self.min_clip_length_s:
                        clip_bounds.append((clip_start, clip_end))
                    clip_start += self.clip_stride_s

                # assign information to task data struct
                for span in clip_bounds:
                    start_event = int(span[0] * video.metadata.framerate)
                    end_event = int(span[1] * video.metadata.framerate)
                    clip = Clip(
                        uuid=uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{s3_file}_{start_event}_{end_event}",
                        ),
                        source_video=str(s3_file),
                        span=span,
                    )
                    video.clips.append(clip)
                    if self._limit_clips > 0 and len(video.clips) >= self._limit_clips:
                        break

            if not video.clips:
                logger.warning(f"No clips extracted for {s3_file} with duration {duration}")

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
