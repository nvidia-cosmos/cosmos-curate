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
"""Video index and metadata classes for the sensor library."""

from fractions import Fraction
from typing import Self

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curate.core.sensors.utils.helpers import make_numpy_fields_readonly
from cosmos_curate.core.sensors.utils.validation import require_strictly_increasing

VIDEO_METADATA_VERSION = "1"


@attrs.define(hash=False, frozen=True)
class VideoIndex:
    """Structure-of-Arrays container for video packet timestamps.

    Rows form an indexed timeline in presentation timestamp (PTS) order. Decode scheduling
    and ``container.seek`` use keyframe times (``kf_pts_stream`` / ``kf_pts_ns``) with
    the full per-packet ``pts_stream`` axis (see ``make_decode_plan`` in
    ``cosmos_curate.core.sensors.utils.video``).

    Stores per-packet arrays alongside the stream ``time_base`` needed to
    convert between stream-native pts units and nanoseconds.  Scalar stream
    properties (codec name, resolution, etc.) live in :class:`VideoMetadata`.

    All timestamps are in nanoseconds (``int64``), matching MCAP
    ``log_time`` / ``publish_time``.

    All arrays are in ascending PTS order.  For B-frame video, this differs
    from file/decode order (DTS order), so ``offset`` is not monotonically
    increasing — it is a per-packet lookup (``offset[i]`` is the byte
    position of the packet with ``pts_ns[i]``), not a sequential scan order.

    Attributes:
        offset: byte-offset in the file for this packet (complete packet-level index;
            not used for seeks in this library).  Not monotonically increasing for
            B-frame video (arrays are in PTS order, not file order).
        size: packet size in bytes; paired with ``offset`` for the same complete index
        pts_ns: presentation timestamps in nanoseconds, in ascending order.
        pts_stream: presentation timestamps in stream-native time_base units,
            in ascending order.  Use these for seeks and decode plans to avoid
            lossy ns↔stream_pts round-trips.
        is_keyframe: boolean array indicating keyframe packets (I-frames).
        is_discard: boolean array indicating discarded packets.  Discarded
            packets are part of the decode stream, but are not displayed.
        kf_pts_ns: keyframe presentation timestamps in nanoseconds, in
            ascending order.
        kf_pts_stream: keyframe presentation timestamps in stream-native
            time_base units, in ascending order.  Use these for
            ``make_decode_plan`` and ``container.seek``.
        time_base: stream time_base as a ``Fraction``
            (e.g. ``Fraction(1, 15360)``).  Satisfies
            ``pts_to_ns(pts_stream, time_base) == pts_ns`` exactly.
            **Do not** compute ``pts_stream`` from ``pts_ns`` via floor
            division — the round-trip is lossy for fps-rate time_bases
            (e.g. ``Fraction(1, 30)``).  Always preserve ``pts_stream``
            directly from the container index.

    """

    __hash__ = None  # type: ignore[assignment]

    offset: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    pts_ns: npt.NDArray[np.int64]
    pts_stream: npt.NDArray[np.int64]
    is_keyframe: npt.NDArray[np.bool_]
    is_discard: npt.NDArray[np.bool_]
    kf_pts_ns: npt.NDArray[np.int64]
    kf_pts_stream: npt.NDArray[np.int64]
    time_base: Fraction
    _display_mask: npt.NDArray[np.bool_] = attrs.field(init=False, repr=False, eq=False)
    _display_pts_ns: npt.NDArray[np.int64] = attrs.field(init=False, repr=False, eq=False)
    _display_pts_stream: npt.NDArray[np.int64] = attrs.field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self) -> None:
        """Post-initialization checks."""
        for name, arr, dtype in (
            ("offset", self.offset, np.int64),
            ("size", self.size, np.int64),
            ("pts_ns", self.pts_ns, np.int64),
            ("pts_stream", self.pts_stream, np.int64),
            ("is_keyframe", self.is_keyframe, np.bool_),
            ("is_discard", self.is_discard, np.bool_),
            ("kf_pts_ns", self.kf_pts_ns, np.int64),
            ("kf_pts_stream", self.kf_pts_stream, np.int64),
        ):
            if arr.ndim != 1:
                msg = f"{name} must be 1-D, got ndim={arr.ndim}"
                raise ValueError(msg)
            if arr.dtype != dtype:
                msg = f"{name} must have dtype {np.dtype(dtype).name}, got {arr.dtype}"
                raise ValueError(msg)

        lens = (
            len(self.offset),
            len(self.size),
            len(self.pts_ns),
            len(self.pts_stream),
            len(self.is_keyframe),
            len(self.is_discard),
        )
        if len(set(lens)) != 1:
            error_msg = (
                "All arrays must be the same length: "
                f"offset={lens[0]} size={lens[1]} pts_ns={lens[2]} pts_stream={lens[3]} "
                f"is_keyframe={lens[4]} is_discard={lens[5]}"
            )
            raise ValueError(error_msg)

        if len(self.kf_pts_ns) != len(self.kf_pts_stream):
            error_msg = (
                "kf_pts_ns and kf_pts_stream must have equal length: "
                f"{len(self.kf_pts_ns)} != {len(self.kf_pts_stream)}"
            )
            raise ValueError(error_msg)

        if len(self.kf_pts_ns) != int(self.is_keyframe.sum()):
            error_msg = "kf_pts_ns length must equal number of keyframes in is_keyframe"
            raise ValueError(error_msg)

        require_strictly_increasing("pts_ns", self.pts_ns)
        require_strictly_increasing("pts_stream", self.pts_stream)
        require_strictly_increasing("kf_pts_ns", self.kf_pts_ns)
        require_strictly_increasing("kf_pts_stream", self.kf_pts_stream)

        expected_kf_pts_ns = self.pts_ns[self.is_keyframe]
        if not np.array_equal(self.kf_pts_ns, expected_kf_pts_ns):
            error_msg = "kf_pts_ns must equal pts_ns[is_keyframe]"
            raise ValueError(error_msg)

        expected_kf_pts_stream = self.pts_stream[self.is_keyframe]
        if not np.array_equal(self.kf_pts_stream, expected_kf_pts_stream):
            error_msg = "kf_pts_stream must equal pts_stream[is_keyframe]"
            raise ValueError(error_msg)

        display_mask = ~self.is_discard
        display_mask.flags.writeable = False
        object.__setattr__(self, "_display_mask", display_mask)

        display_pts_ns = self.pts_ns[display_mask]
        display_pts_ns.flags.writeable = False
        object.__setattr__(self, "_display_pts_ns", display_pts_ns)

        display_pts_stream = self.pts_stream[display_mask]
        display_pts_stream.flags.writeable = False
        object.__setattr__(self, "_display_pts_stream", display_pts_stream)

        make_numpy_fields_readonly(self)

    def __len__(self) -> int:
        """Return number of packets."""
        return len(self.pts_ns)

    @property
    def display_mask(self) -> npt.NDArray[np.bool_]:
        """Return a read-only mask selecting displayable packets."""
        return self._display_mask

    @property
    def display_pts_ns(self) -> npt.NDArray[np.int64]:
        """Return displayable presentation timestamps in nanoseconds."""
        return self._display_pts_ns

    @property
    def display_pts_stream(self) -> npt.NDArray[np.int64]:
        """Return displayable presentation timestamps in stream-native units."""
        return self._display_pts_stream

    def __eq__(self, other: object) -> bool:
        """Check if two VideoIndex objects are equal."""
        if not isinstance(other, VideoIndex):
            return False
        if len(self) != len(other):
            return False
        if len(self.kf_pts_ns) != len(other.kf_pts_ns):
            return False
        return (
            bool(np.all(self.offset == other.offset))
            and bool(np.all(self.size == other.size))
            and bool(np.all(self.pts_ns == other.pts_ns))
            and bool(np.all(self.pts_stream == other.pts_stream))
            and bool(np.all(self.is_keyframe == other.is_keyframe))
            and bool(np.all(self.is_discard == other.is_discard))
            and bool(np.all(self.kf_pts_ns == other.kf_pts_ns))
            and bool(np.all(self.kf_pts_stream == other.kf_pts_stream))
            and self.time_base == other.time_base
        )


@attrs.define(frozen=True)
class VideoMetadata:
    """Scalar properties of a video stream.

    Pairs with :class:`VideoIndex` (which holds per-packet arrays).  Together
    they represent everything extracted from the container index without
    demuxing.  Obtain both via :func:`~cosmos_curate.core.sensors.utils.video.make_index_and_metadata`.

    Attributes:
        codec_name: video codec name as reported by libavcodec
            (e.g., ``'h264'``, ``'hevc'``, ``'vp9'``, ``'av1'``).
        codec_max_bframes: maximum number of consecutive B-frames between I and
            P frames that the encoder was configured to generate.  This is a
            codec setting, not a count of B-frames actually present in the
            stream.  ``0`` means the encoder was configured for no B-frames;
            the stream may still contain B-frames if the header is inaccurate.
        codec_profile: human-readable codec profile string as reported by
            libavcodec (e.g., ``'High'``, ``'Main'``, ``'Baseline'`` for
            H.264; ``'Main'``, ``'High'`` for HEVC).  Together with
            ``codec_max_bframes``, indicates whether B-frames are structurally
            possible for this stream.  Empty string if not set.
        container_format: container format name as reported by libavformat
            (e.g., ``'mp4'``, ``'matroska,webm'``).
        height: frame height in pixels.
        width: frame width in pixels.
        avg_frame_rate: average frame rate as a ``Fraction``
            (e.g., ``Fraction(30, 1)`` for 30 fps,
            ``Fraction(30000, 1001)`` for 29.97 fps).  Derived from the
            container header; use this to understand the source capture rate
            relative to your sampling rate.
        pix_fmt: pixel format string as reported by libavcodec
            (e.g., ``'yuv420p'``, ``'yuv420p10le'``, ``'yuvj420p'``).
            Indicates chroma subsampling and bit depth; relevant when callers
            need to reason about color precision or GPU texture formats.
        bit_rate_bps: actual video bitrate in bits per second, computed from
            total packet byte sizes and stream duration extracted from the
            container index.  More reliable than the header-declared bitrate,
            which is often absent or inaccurate in camera-recorder output.
            ``0`` if the stream contains fewer than two frames (duration
            indeterminate).

    """

    codec_name: str
    codec_max_bframes: int
    codec_profile: str
    container_format: str
    height: int = attrs.field(validator=attrs.validators.gt(0))
    width: int = attrs.field(validator=attrs.validators.gt(0))
    avg_frame_rate: Fraction
    pix_fmt: str
    bit_rate_bps: int

    @classmethod
    def from_string_dict(cls, data: dict[str, str]) -> Self:
        """Deserialize :class:`VideoMetadata` from a string-only mapping.

        Args:
            data: Serialized metadata values.

        Raises:
            ValueError: if required keys are missing, version mismatches, or values are invalid.

        """
        required = {
            "version",
            "codec_name",
            "codec_max_bframes",
            "codec_profile",
            "container_format",
            "height",
            "width",
            "avg_frame_rate_numerator",
            "avg_frame_rate_denominator",
            "pix_fmt",
            "bit_rate_bps",
        }
        missing = sorted(required.difference(data))
        if missing:
            msg = f"VideoMetadata payload missing required keys: {', '.join(missing)}"
            raise ValueError(msg)

        actual_version = data["version"]
        if actual_version != VIDEO_METADATA_VERSION:
            msg = f"unsupported VideoMetadata payload version {actual_version!r}"
            raise ValueError(msg)

        try:
            avg_frame_rate = Fraction(
                int(data["avg_frame_rate_numerator"]),
                int(data["avg_frame_rate_denominator"]),
            )
            return cls(
                codec_name=data["codec_name"],
                codec_max_bframes=int(data["codec_max_bframes"]),
                codec_profile=data["codec_profile"],
                container_format=data["container_format"],
                height=int(data["height"]),
                width=int(data["width"]),
                avg_frame_rate=avg_frame_rate,
                pix_fmt=data["pix_fmt"],
                bit_rate_bps=int(data["bit_rate_bps"]),
            )
        except (ValueError, ZeroDivisionError) as e:
            msg = f"invalid VideoMetadata payload: {data!r}"
            raise ValueError(msg) from e

    def to_string_dict(self) -> dict[str, str]:
        """Serialize :class:`VideoMetadata` to a string-only mapping."""
        return {
            "version": VIDEO_METADATA_VERSION,
            "codec_name": self.codec_name,
            "codec_max_bframes": str(self.codec_max_bframes),
            "codec_profile": self.codec_profile,
            "container_format": self.container_format,
            "height": str(self.height),
            "width": str(self.width),
            "avg_frame_rate_numerator": str(self.avg_frame_rate.numerator),
            "avg_frame_rate_denominator": str(self.avg_frame_rate.denominator),
            "pix_fmt": self.pix_fmt,
            "bit_rate_bps": str(self.bit_rate_bps),
        }
