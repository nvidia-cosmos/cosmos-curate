# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Sensor-library decode helper for the tracking stages.

Decodes clip mp4 bytes and samples frames at a target fps using the shared
sensor library ([cosmos_curator.core.sensors][]). Unlike the legacy OpenCV
stride decode, every returned frame carries its *real* presentation timestamp
(PTS), read from the video file's index, so downstream stages never recompute
time as ``frame_idx / fps``.
"""

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, make_ts_grid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor

_NS_PER_SECOND = 1_000_000_000


@attrs.define(frozen=True)
class DecodedClip:
    """Frames sampled from a clip, each tagged with its real timestamp.

    Attributes:
        frames_rgb: Decoded RGB frames, each ``(H, W, 3)`` ``uint8``, in
            presentation order.
        timestamps_s: Per-frame real presentation timestamp in seconds since
            the clip's first frame (PTS-derived, not ``frame_idx / fps``). This
            is the robust per-frame identity downstream consumers key on.
        width: Frame width in pixels.
        height: Frame height in pixels.

    """

    frames_rgb: list[npt.NDArray[np.uint8]]
    timestamps_s: list[float]
    width: int
    height: int


def decode_clip_at_fps(
    mp4_bytes: bytes,
    target_fps: float,
    *,
    max_delta_ns: int | None = None,
) -> DecodedClip:
    """Decode ``mp4_bytes`` and sample frames at ``target_fps`` with real PTS.

    Builds a target sampling grid at ``target_fps`` over the clip's full span
    and asks the sensor library for the real frame nearest each grid time. The
    whole clip is decoded as a single ordered batch (matching the clip frame
    extraction pattern).

    Args:
        mp4_bytes: Encoded clip (mp4 container) bytes.
        target_fps: Desired sampling rate in frames per second; maps directly
            onto the sensor sampling-grid rate.
        max_delta_ns: Optional maximum delta (ns) between a grid time and the
            chosen real frame. ``None`` (default) applies no tolerance policy,
            i.e. nearest-frame snapping with no drops.

    Returns:
        A :class:`DecodedClip` with frames, real per-frame ``timestamps_s``,
        and frame geometry. Empty (no displayable frames matched) yields empty
        frame/timestamp lists.

    """
    sensor = CameraSensor(bytes(mp4_bytes))
    width = sensor.video_metadata.width
    height = sensor.video_metadata.height

    start_ns = sensor.start_ns
    end_ns = sensor.end_ns
    # Single full-span window: one batch covering every sampled frame.
    grid_start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(
        start_ns=start_ns,
        end_ns=end_ns,
        sample_rate_hz=float(target_fps),
    )
    span_ns = max(1, exclusive_end_ns - grid_start_ns)
    grid = SamplingGrid(
        start_ns=grid_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=timestamps_ns,
        stride_ns=span_ns,
        duration_ns=span_ns,
    )
    policy = NearestTimestampPolicy(max_delta_ns=max_delta_ns)
    spec = SamplingSpec(grid=grid)

    frames_rgb: list[npt.NDArray[np.uint8]] = []
    sensor_ts_ns: list[int] = []
    for batch in sensor.sample(spec, policy=policy):
        if len(batch.frames) == 0:
            continue
        frames_rgb.extend(np.asarray(frame, dtype=np.uint8) for frame in batch.frames)
        sensor_ts_ns.extend(int(ts) for ts in batch.sensor_timestamps_ns)

    if not frames_rgb:
        return DecodedClip(
            frames_rgb=[],
            timestamps_s=[],
            width=width,
            height=height,
        )

    # Real time since the clip's first displayable frame. This (the actual PTS)
    # is the per-frame anchor; we deliberately do not try to recover a native
    # source frame index, since that mapping is ambiguous under duplicated /
    # dropped PTS (VFR, supersampling). Downstream keys on timestamp_s and the
    # contiguous sampled position instead.
    timestamps_s = [round((ts - start_ns) / _NS_PER_SECOND, 3) for ts in sensor_ts_ns]

    return DecodedClip(
        frames_rgb=frames_rgb,
        timestamps_s=timestamps_s,
        width=width,
        height=height,
    )
