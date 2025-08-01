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
"""Ray stage for loading resized frames from videos as 4-D numpy array."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.operation_context import make_pipeline_named_temporary_file
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask

if conda_utils.is_running_in_env("video-splitting") and torch.cuda.is_available():
    from cosmos_curate.pipelines.video.utils.nvcodec_utils import PyNvcFrameExtractor
else:
    PyNvcFrameExtractor = None


def get_frames_from_ffmpeg(
    video_file: Path,
    width: int,
    height: int,
    *,
    use_gpu: bool = False,
) -> npt.NDArray[np.uint8] | None:
    """Fetch resized frames for video."""
    if use_gpu:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "1",
            "-hwaccel",
            "auto",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            video_file.as_posix(),
            "-vf",
            f"scale_npp={width}:{height},hwdownload,format=nv12",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "4",
            "-i",
            video_file.as_posix(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-",
        ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    video_stream, err = process.communicate()
    if process.returncode != 0:
        if use_gpu:
            logger.warning("Caught ffmpeg runtime error with `use_gpu=True` option, falling back to CPU.")
            return get_frames_from_ffmpeg(video_file, width, height, use_gpu=False)
        logger.exception(f"FFmpeg error: {err.decode('utf-8')}")
        return None
    return np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])


class VideoFrameExtractionStage(CuratorStage):
    """Stage that extracts frames from videos into numpy arrays.

    This stage handles video frame extraction using either FFmpeg (CPU/GPU) or PyNvCodec,
    converting video content into standardized frame arrays for downstream processing.
    """

    def __init__(
        self,
        output_hw: tuple[int, int] = (27, 48),
        decoder_mode: str = "ffmpeg_cpu",
        *,
        raise_on_pynvc_error_without_cpu_fallback: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the frame extraction stage.

        Args:
            output_hw: (tuple) output height and width of frame array.
                Default is (27, 48), which is the default for TransNetV2 and AutoShot models.
            decoder_mode: (str) decoder mode
            raise_on_pynvc_error_without_cpu_fallback: (bool) raise an exception if PyNvCodec fails without CPU fallback
            log_stats: (bool) whether to log stats
            verbose: (bool) verbose

        """
        super().__init__()
        self.output_hw = output_hw
        self.decoder_mode = decoder_mode
        self._raise_on_pynvc_error_without_cpu_fallback = raise_on_pynvc_error_without_cpu_fallback
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        if self.decoder_mode == "pynvc":
            if PyNvcFrameExtractor is None:
                msg = "decoder_mode='pynvc' requires running inside the 'video-splitting' environment with GPU support."
                raise RuntimeError(msg)
            self.pynvc_frame_extractor = PyNvcFrameExtractor(self.output_hw[1], self.output_hw[0], batch_size=64)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "video-splitting"

    @nvtx.annotate("VideoFrameExtractionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the frame extraction stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        self._timer.reinit(self, sum(x.get_major_size() for x in tasks))
        height, width = self.output_hw
        for task in tasks:
            video = task.video
            if video.source_bytes is None:
                error_msg = "Please load video bytes!"
                raise ValueError(error_msg)
            if not video.has_metadata():
                logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
                continue

            with (
                self._timer.time_process(),
                make_pipeline_named_temporary_file(sub_dir="video_frame_extraction") as video_path,
            ):
                with video_path.open("wb") as fp:
                    fp.write(video.source_bytes)
                if self.decoder_mode == "pynvc":
                    try:
                        video.frame_array = self.pynvc_frame_extractor(video_path).cpu().numpy().astype(np.uint8)
                    except Exception as e:
                        if not self._raise_on_pynvc_error_without_cpu_fallback:
                            logger.warning(f"Got exception {e} with PyNvVideoCodec decode, trying ffmpeg CPU fallback")
                            video.frame_array = get_frames_from_ffmpeg(
                                video_path,
                                width=width,
                                height=height,
                                use_gpu=False,
                            )
                        else:
                            # for CI to test PyNvCodec path without CPU fallback
                            msg = f"PyNvCodec decode failed for {video.input_path}. "
                            raise RuntimeError(msg) from e
                else:
                    video.frame_array = get_frames_from_ffmpeg(
                        video_path,
                        width=width,
                        height=height,
                        use_gpu=self.decoder_mode == "ffmpeg_gpu",
                    )
                if video.frame_array is None:
                    logger.error(f"Video frame extraction failed on {video.input_video}, skipping ...")
                    video.errors["frame_extraction"] = "null"
                    continue
                if self._verbose:
                    logger.info(f"Loaded video as numpy uint8 array with shape {video.frame_array.shape}")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return tasks

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        if self.decoder_mode != "ffmpeg_cpu":
            return CuratorStageResource(gpus=0.1)
        return CuratorStageResource(cpus=4.0)
