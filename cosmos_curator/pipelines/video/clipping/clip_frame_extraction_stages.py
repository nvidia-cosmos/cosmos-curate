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

import math
from functools import reduce

import attrs
import cv2
import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.sensors.sampling.compat import make_decoder_utils_compat_grid
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.utils.video import CpuVideoDecodeConfig
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.data.ref_resolver import prefetch, resolve_as_ready
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.filtering.motion import motion_vector_backend as motion
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
)

_NS_PER_SECOND = 1_000_000_000

# Legacy ``decode_for_motion`` used ``max(10, ...)`` as a minimum-frame floor, so short clips were
# still sampled across roughly their first ``min_motion_frames / target_fps`` seconds regardless of
# ``target_duration_ratio``. Preserve that floor here so short clips are not under-sampled (which
# would shift motion scores and make ``no_motion_frames`` drops more likely).
_DEFAULT_MIN_MOTION_FRAMES = 10


@attrs.define(frozen=True)
class CameraSensorMotionVectorConfig:
    """Configuration for exporting motion vectors from the camera-sensor clip backend."""

    target_fps: float = 2.0
    target_duration_ratio: float = 0.5
    min_motion_frames: int = _DEFAULT_MIN_MOTION_FRAMES


def motion_sampling_stop_ns(
    start_ns: int,
    end_ns: int,
    *,
    target_fps: float,
    target_duration_ratio: float,
    min_motion_frames: int,
) -> int:
    """Return the exclusive stop timestamp for CameraSensor motion-vector sampling.

    Mirrors legacy ``decode_for_motion`` frame budgeting: sample the first
    ``max(min_motion_frames, round(target_fps * duration * ratio))`` frames at ``target_fps``,
    i.e. a window of ``target_frames / target_fps`` seconds, clamped to the clip. The
    ``min_motion_frames`` floor keeps short clips (whose ``duration * ratio`` window would otherwise
    yield only a couple of frames) sampled across enough of the clip to match the legacy path.
    """
    source_duration_ns = max(0, end_ns - start_ns)
    if source_duration_ns == 0 or target_fps <= 0:
        return end_ns
    duration_s = source_duration_ns / _NS_PER_SECOND
    # The leading 0 clamps against negative min_motion_frames / target_duration_ratio so the window
    # is never negative and the returned stop stays within [start_ns, end_ns].
    target_frames = max(0, min_motion_frames, round(target_fps * duration_s * target_duration_ratio))
    window_ns = round(target_frames / target_fps * _NS_PER_SECOND)
    return min(end_ns, start_ns + window_ns)


class ClipFrameExtractionStage(CuratorStage):
    """Stage for extracting frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    target frame rate selection, and frame extraction signature creation.

    Clip frames are always sampled through ``CameraSensor``, which only supports the
    ``FrameExtractionPolicy.sequence`` policy.
    """

    def __init__(  # noqa: PLR0913
        self,
        target_fps: list[float | int] | None = None,
        target_res: tuple[int, int] | None = None,
        *,
        motion_vectors: CameraSensorMotionVectorConfig | None = None,
        num_cpus_per_worker: float = 3.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip frame extraction stage.

        Args:
            target_fps: Target frames per second for extraction.
            target_res: Target resolution for extracted frames.
            motion_vectors: Optional camera-sensor motion-vector export config.
            num_cpus_per_worker: Number of CPU cores to allocate per worker.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        if target_fps is None:
            target_fps = [2]
        if target_res is None:
            target_res = (-1, -1)
        self._timer = StageTimer(self)
        self._target_fps = target_fps
        self._target_res = target_res
        self._motion_vector_config = motion_vectors
        self._num_cpus = num_cpus_per_worker
        self._num_threads = max(1, int(num_cpus_per_worker) + 1)
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

    def _make_signature(self, fps: float) -> str:
        signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=fps,
        )
        return signature.to_str()  # type: ignore[no-any-return]

    def _use_lcm_fps(self) -> bool:
        return len(self._target_fps) > 1 and all(
            (fps.is_integer() if isinstance(fps, float) else isinstance(fps, int)) for fps in self._target_fps
        )

    def _resize_frames(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self._target_res[0] > 0 and self._target_res[1] > 0:
            interpolation = cv2.INTER_CUBIC
            return np.array(
                [
                    cv2.resize(frame, (self._target_res[1], self._target_res[0]), interpolation=interpolation)
                    for frame in frames
                ]
            )
        return frames

    def _sample_with_camera_sensor(
        self,
        data: bytes | npt.NDArray[np.uint8],
        sample_rate_fps: float,
    ) -> npt.NDArray[np.uint8]:
        sensor = CameraSensor(
            bytes(data),
            decode_config=CpuVideoDecodeConfig(thread_count=self._num_threads),
        )
        spec = self._make_sensor_sampling_spec(sensor, sample_rate_fps, stop_ns=sensor.end_ns)
        sampled_batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))
        if len(sampled_batches) != 1:
            msg = f"Expected exactly one sampled batch, got {len(sampled_batches)}"
            raise RuntimeError(msg)
        return self._resize_frames(sampled_batches[0].frames)

    def _make_sensor_sampling_spec(
        self,
        sensor: CameraSensor,
        sample_rate_fps: float,
        *,
        stop_ns: int,
    ) -> SamplingSpec:
        start_ns, exclusive_end_ns, timestamps_ns = make_decoder_utils_compat_grid(
            start_ns=sensor.start_ns,
            stop_ns=stop_ns,
            sample_rate_hz=float(sample_rate_fps),
        )
        grid = SamplingGrid(
            start_ns=start_ns,
            exclusive_end_ns=exclusive_end_ns,
            timestamps_ns=timestamps_ns,
            stride_ns=max(1, exclusive_end_ns - start_ns),
            duration_ns=max(1, exclusive_end_ns - start_ns),
        )
        return SamplingSpec(grid=grid)

    def _extract_motion_vectors_for_clip(
        self,
        data: bytes | npt.NDArray[np.uint8],
    ) -> motion.DecodedData | None:
        motion_vector_config = self._motion_vector_config
        if motion_vector_config is None:
            msg = "motion vector config is not enabled"
            raise RuntimeError(msg)

        sensor = CameraSensor(
            bytes(data),
            decode_config=CpuVideoDecodeConfig(export_mvs=True, thread_count=self._num_threads),
        )
        stop_ns = motion_sampling_stop_ns(
            sensor.start_ns,
            sensor.end_ns,
            target_fps=motion_vector_config.target_fps,
            target_duration_ratio=motion_vector_config.target_duration_ratio,
            min_motion_frames=motion_vector_config.min_motion_frames,
        )
        spec = self._make_sensor_sampling_spec(sensor, motion_vector_config.target_fps, stop_ns=stop_ns)
        sampled_batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))
        if len(sampled_batches) != 1:
            msg = f"Expected exactly one sampled batch, got {len(sampled_batches)}"
            raise RuntimeError(msg)

        camera_data = sampled_batches[0]
        if camera_data.motion_vectors is None:
            return None

        frame_size = (camera_data.metadata.height, camera_data.metadata.width, 3)
        min_side_resolution = motion._MIN_SIDE_RESOLUTION  # noqa: SLF001
        if frame_size[0] < min_side_resolution or frame_size[1] < min_side_resolution:
            error_msg = f"Expected min resolution of {min_side_resolution}, but got a resolution of {frame_size}"
            raise motion.VideoResolutionTooSmallError(error_msg)

        frames = [
            legacy_matrix
            for legacy_matrix in motion.sensor_motion_vector_data_to_legacy_matrices(camera_data.motion_vectors)
            if legacy_matrix.shape[0] > 0
        ]
        return motion.DecodedData(frames=frames, frame_size=frame_size)  # type: ignore[arg-type]

    def _populate_motion_vectors_for_clip(self, clip: Clip, data: bytes | npt.NDArray[np.uint8]) -> None:
        if self._motion_vector_config is None:
            return

        try:
            clip.decoded_motion_data = self._extract_motion_vectors_for_clip(data)
        except motion.VideoResolutionTooSmallError:
            if self._verbose:
                logger.warning(f"Clip {clip.uuid} has too small resolution.")
            clip.decoded_motion_data = None
            clip.errors["motion_decode"] = "resolution_too_small"
        except Exception as e:  # noqa: BLE001
            if self._verbose:
                logger.exception(f"Clip {clip.uuid} failed to decode motion data: {e}")
            clip.decoded_motion_data = None
            clip.errors["motion_decode"] = "decode_failed"
        else:
            if clip.decoded_motion_data is None or len(clip.decoded_motion_data.frames) == 0:
                logger.warning(f"Clip {clip.uuid} has no motion frames.")
                clip.decoded_motion_data = None
                clip.errors["motion_decode"] = "no_motion_frames"

    def _extract_frames(
        self,
        data: bytes | npt.NDArray[np.uint8],
    ) -> dict[str, npt.NDArray[np.uint8]]:
        local_frames: dict[str, npt.NDArray[np.uint8]] = {}

        # Decode-reuse optimization: for multiple integer FPS targets, sample once at the
        # least-common-multiple rate and stride-subsample the lower rates instead of decoding
        # the clip once per target FPS.
        if self._use_lcm_fps():
            lcm = self.lcm_multiple(self._target_fps)
            frames = self._sample_with_camera_sensor(data, lcm)
            for fps in self._target_fps:
                stride = int(lcm / fps)
                local_frames[self._make_signature(fps)] = frames[::stride]
            return local_frames

        for fps in self._target_fps:
            local_frames[self._make_signature(fps)] = self._sample_with_camera_sensor(data, fps)
        return local_frames

    def _process_video(self, video: Video) -> None:
        if self._verbose:
            logger.info(f"Processing video {video.input_video} with {len(video.clips)} clips")

        prefetch([clip.encoded_data for clip in video.clips])
        for clip, data in resolve_as_ready([(clip, clip.encoded_data) for clip in video.clips]):
            if data is None:
                logger.warning(f"Clip {clip.uuid} has no encoded_data.")
                clip.errors["encoded_data"] = "empty"
                continue

            try:
                local_frames = self._extract_frames(data)
                if self._verbose:
                    for signature, frame_array in local_frames.items():
                        logger.info(f"Extracted {len(frame_array)} frames from clip {clip.uuid} for {signature=}")

                total_nbytes = sum(arr.nbytes for arr in local_frames.values())
                clip.extracted_frames = LazyData(value=local_frames, nbytes=total_nbytes)
                # Phase 2: call extracted_frames.store() here to push to Plasma
                # before the stage boundary for split-field transport
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error extracting frames from clip {clip.uuid}: {e}")
                clip.errors["frame_extraction"] = "video_decode_failed"
                # drop the transported buffer, but still attempt motion-vector extraction below from
                # the in-hand bytes: frame decode and motion export are independent (as they were
                # when ClipFrameExtractionStage and MotionVectorDecodeStage were separate), so a frame
                # failure should still record its own motion_decode outcome rather than be silently
                # skipped.
                clip.encoded_data.drop()

            self._populate_motion_vectors_for_clip(clip, data)

    @nvtx.annotate("ClipFrameExtractionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process the data for the clip frame extraction stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process():
                    try:
                        self._process_video(video)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Error processing video {video.input_video}")
                        video.errors[self.__class__.__name__] = str(e)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
