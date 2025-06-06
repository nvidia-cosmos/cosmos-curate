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

"""Clip Frame Extraction Stage."""

import io
import math
from functools import reduce

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.runtime.performance_utils import StageTimer
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask
from cosmos_curate.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    extract_frames,
)


class ClipFrameExtractionStage(CuratorStage):
    """Stage for extracting frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    target frame rate selection, and frame extraction signature creation.
    """

    def __init__(
        self,
        extraction_policies: tuple[FrameExtractionPolicy, ...] = (FrameExtractionPolicy.sequence,),
        target_fps: list[float | int] | None = None,
        target_res: tuple[int, int] | None = None,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip frame extraction stage.

        Args:
            extraction_policies: Frame extraction policies to use.
            target_fps: Target frames per second for extraction.
            target_res: Target resolution for extracted frames.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        if target_fps is None:
            target_fps = [2]
        if target_res is None:
            target_res = (-1, -1)
        self._timer = StageTimer(self)
        self._extraction_policies = extraction_policies
        self._target_fps = target_fps
        self._target_res = target_res
        self._num_cpus = 4
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(cpus=self._num_cpus)

    def lcm_multiple(self, fps: list[float | int]) -> float | int:
        """Compute LCM of a list of fps targets."""

        def lcm(a: float, b: float) -> float | int:
            return abs(a * b) // math.gcd(int(a), int(b))

        return reduce(lcm, fps)

    @nvtx.annotate("ClipFrameExtractionStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process the data for the clip frame extraction stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.buffer is None:
                    logger.warning(f"Clip {clip.uuid} has no buffer.")
                    clip.errors["buffer"] = "empty"
                    continue
                with self._timer.time_process():
                    try:
                        for policy in self._extraction_policies:
                            """
                            To save on decode costs, calculate the least-common-multiple(LCM) of fps
                            targets and apply decord.get_batch on this LCM fps
                            """
                            use_lcm_fps = len(self._target_fps) > 1 and all(
                                (fps.is_integer() if isinstance(fps, float) else isinstance(fps, int))
                                for fps in self._target_fps
                            )
                            if use_lcm_fps:
                                lcm = self.lcm_multiple(self._target_fps)
                                with io.BytesIO(clip.buffer) as fp:
                                    frames = extract_frames(
                                        fp,
                                        extraction_policy=policy,
                                        sample_rate_fps=lcm,
                                        target_res=self._target_res,
                                        num_threads=self._num_cpus,
                                    )
                                    for fps in self._target_fps:
                                        signature = FrameExtractionSignature(
                                            extraction_policy=policy,
                                            target_fps=fps,
                                        ).to_str()
                                        clip.extracted_frames[signature] = frames[:: int(lcm / fps)]
                            else:
                                for fps in self._target_fps:
                                    with io.BytesIO(clip.buffer) as fp:
                                        frames = extract_frames(
                                            fp,
                                            extraction_policy=policy,
                                            sample_rate_fps=fps,
                                            target_res=self._target_res,
                                            num_threads=self._num_cpus,
                                        )
                                        signature = FrameExtractionSignature(
                                            extraction_policy=policy,
                                            target_fps=fps,
                                        ).to_str()
                                        clip.extracted_frames[signature] = frames
                                        if self._verbose:
                                            logger.info(
                                                f"Extracted {len(frames)} frames from clip {clip.uuid} at {fps=}"
                                            )
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error extracting frames from clip {clip.uuid}: {e}")
                        clip.errors["frame_extraction"] = "video_decode_failed"
                        # reset the buffer to disable further operations on this clip
                        clip.buffer = None
                        continue

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
