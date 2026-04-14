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

"""Camera data structures for cosmos_curate.core.sensors package."""

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curate.core.sensors.data.video import VideoMetadata
from cosmos_curate.core.sensors.utils.helpers import make_numpy_fields_readonly
from cosmos_curate.core.sensors.utils.validation import require_nondecreasing, require_strictly_increasing

_MOTION_VECTOR_FRAME_NDIM = 2
_MOTION_VECTOR_COLUMNS = 10
_CAMERA_FRAME_NDIM = 4
_RGB_CHANNELS = 3


@attrs.define(hash=False, frozen=True)
class MotionVectorData:
    """Per-frame motion vectors from H.264/HEVC (FFmpeg AVMotionVector). Variable num_blocks per frame."""

    __hash__ = None  # type: ignore[assignment]

    # List of length N; frame i has shape (num_blocks_i, 10).
    # Columns:
    # [0:2] w, h (block size)
    # [2:4] src_x, src_y
    # [4:6] dst_x, dst_y
    # [6:7] flags
    # [7:9] motion_x, motion_y (sub-pixel; divide by motion_scale for pixel delta)
    # [9]   motion_scale
    frames: tuple[npt.NDArray[np.float64], ...]  # length N; each (num_blocks_i, 10)

    def __attrs_post_init__(self) -> None:
        """Post-initialization checks."""
        object.__setattr__(self, "frames", tuple(self.frames))
        for i, frame in enumerate(self.frames):
            if frame.ndim != _MOTION_VECTOR_FRAME_NDIM:
                msg = f"motion_vectors.frames[{i}] must be 2-D, got ndim={frame.ndim}"
                raise ValueError(msg)
            if frame.shape[1] != _MOTION_VECTOR_COLUMNS:
                msg = (
                    f"motion_vectors.frames[{i}] must have shape (N, {_MOTION_VECTOR_COLUMNS}), got shape={frame.shape}"
                )
                raise ValueError(msg)
            frame.flags.writeable = False


@attrs.define(hash=False, frozen=True)
class CameraData:
    """Decoded RGB video: ``N`` frames, with row ``i`` indexing the same moment across arrays.

    Satisfies ``SensorData`` (``cosmos_curate.core.sensors.data.sensor_data``).

    Attributes:
        timestamps_ns: 1-D reference timeline (ns) each sample row is aligned to; length ``N``,
            row ``i`` with ``frames[i]``
        canonical_timestamps_ns: 1-D sensor-reported times (ns); may differ from ``timestamps_ns``
            (resampling/grid); length ``N``
        pts_stream: 1-D presentation timestamps in producer-specific int units, length ``N``;
            ``CameraSensor`` uses stream-native ``time_base`` for exact seeks; ``McapCameraSensor``
            uses nanoseconds matching ``canonical_timestamps_ns`` (example sensor only; production
            ``pts_stream`` is expected to follow the ``CameraSensor`` contract)
        frames: decoded RGB, shape ``(N, H, W, 3)``, ``uint8``; row ``i`` is the image at index ``i``
        metadata: stream geometry and related fields (``VideoMetadata``)
        motion_vectors: optional per-frame motion vectors; when set, length ``N`` matches ``frames``

    """

    __hash__ = None  # type: ignore[assignment]
    timestamps_ns: npt.NDArray[np.int64]  # (N,) reference sample times, nanoseconds
    canonical_timestamps_ns: npt.NDArray[np.int64]  # (N,) actual frame times, nanoseconds
    pts_stream: npt.NDArray[np.int64]  # (N,) sensor-specific frame PTS units; see class docstring

    frames: npt.NDArray[np.uint8]  # (N, H, W, 3) RGB
    metadata: VideoMetadata
    motion_vectors: MotionVectorData | None = None  # optional; requires decoder with export_mvs

    def __attrs_post_init__(self) -> None:
        """Post-initialization checks."""
        for name, arr in (
            ("timestamps_ns", self.timestamps_ns),
            ("canonical_timestamps_ns", self.canonical_timestamps_ns),
            ("pts_stream", self.pts_stream),
        ):
            if arr.ndim != 1:
                msg = f"{name} must be 1-D, got ndim={arr.ndim}"
                raise ValueError(msg)
            if arr.dtype != np.int64:
                msg = f"{name} must have dtype int64, got {arr.dtype}"
                raise ValueError(msg)

        require_strictly_increasing("timestamps_ns", self.timestamps_ns)
        require_nondecreasing("canonical_timestamps_ns", self.canonical_timestamps_ns)
        require_nondecreasing("pts_stream", self.pts_stream)

        if self.frames.ndim != _CAMERA_FRAME_NDIM:
            msg = f"frames must be 4-D with shape (N, H, W, 3), got ndim={self.frames.ndim}"
            raise ValueError(msg)
        if self.frames.shape[1:] != (self.metadata.height, self.metadata.width, _RGB_CHANNELS):
            msg = (
                "frames must have shape "
                f"(N, {self.metadata.height}, {self.metadata.width}, {_RGB_CHANNELS}), got {self.frames.shape}"
            )
            raise ValueError(msg)
        if self.frames.dtype != np.uint8:
            msg = f"frames must have dtype uint8, got {self.frames.dtype}"
            raise ValueError(msg)

        if not (
            len(self.timestamps_ns) == len(self.canonical_timestamps_ns) == len(self.pts_stream) == len(self.frames)
        ):
            error_msg = (
                "All arrays must be the same length: "
                f"timestamps_ns={len(self.timestamps_ns)} "
                f"canonical_timestamps_ns={len(self.canonical_timestamps_ns)} "
                f"pts_stream={len(self.pts_stream)} "
                f"frames={len(self.frames)}"
            )
            raise ValueError(error_msg)

        if self.motion_vectors is not None and len(self.motion_vectors.frames) != len(self.frames):
            error_msg = (
                f"motion_vectors.frames length {len(self.motion_vectors.frames)} != frames length {len(self.frames)}"
            )
            raise ValueError(error_msg)

        make_numpy_fields_readonly(self)
