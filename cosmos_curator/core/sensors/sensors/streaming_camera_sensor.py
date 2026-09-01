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
"""Forward-only streaming camera sensor."""

import itertools
from collections.abc import Generator, Iterator

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.streaming_camera_data import StreamingCameraData, StreamingFrame
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.sampler import find_closest_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.validation import require_1d, require_strictly_increasing
from cosmos_curator.core.sensors.utils.video import (
    DEFAULT_VIDEO_DECODE_CONFIG,
    CpuVideoDecodeConfig,
    iter_video_frames,
)

_MIN_FOR_AN_INTERVAL = 2
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validate_timeline(timestamps_ns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Reject a timeline that cannot describe a recording.

    These are the sensor's bounds, asked for before anything is decoded, so an
    array that is empty or out of order gives an answer no recording could have
    produced. Whether it matches *this* video is the caller's guarantee: checking
    means walking the file, which is the cost this sensor exists to avoid, and
    the walk catches a mismatch at its own boundary for free.
    """
    require_1d("timestamps_ns", timestamps_ns, np.int64)
    if timestamps_ns.size == 0:
        msg = "timestamps_ns must describe at least one observation"
        raise ValueError(msg)
    require_strictly_increasing("timestamps_ns", timestamps_ns)
    return timestamps_ns


def _serving_timestamps(
    timeline_ns: npt.NDArray[np.int64], references_ns: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Return the capture timestamp that will serve each reference.

    Which observation serves a reference is decided by capture times alone, so
    the answer exists before anything is decoded. The walk reaches it by
    stepping and this by lookup, and they agree because ``find_closest_indices``
    is the same rule the indexed sensor samples by.
    """
    if references_ns.size == 0:
        return references_ns
    return timeline_ns[find_closest_indices(timeline_ns, references_ns)]


@attrs.define(slots=True)
class _WindowValidationState:
    """Carries the previous window's bounds so overlap is caught as it happens."""

    previous_start: int | None = None
    previous_end: int | None = None


def _validate_next_window(window: SamplingWindow, state: _WindowValidationState) -> None:
    """Reject overlapping or non-monotonic sampling windows incrementally.

    A forward-only walk cannot serve a window it has already passed, and
    buffering the overlap would reintroduce the memory this sensor exists to
    avoid. ``McapCameraSensor`` refuses overlap for the same reason. Checked as
    windows arrive rather than up front, because the grid is iterated lazily.
    """
    if state.previous_start is not None and window.start_ns < state.previous_start:
        msg = "SamplingSpec.grid windows must be monotonically increasing"
        raise ValueError(msg)
    if state.previous_end is not None and window.start_ns < state.previous_end:
        msg = "StreamingCameraSensor requires non-overlapping sampling windows"
        raise ValueError(msg)
    state.previous_start = int(window.start_ns)
    state.previous_end = int(window.exclusive_end_ns)


class StreamingCameraSensor:
    """A forward-only camera sensor that yields frames as the caller pulls them.

    ``CameraSensor`` indexes the video up front and seeks to a keyframe per
    target, so sampling a window materializes that window's frames all at once.
    This walks the container once from the first frame and hands over one frame
    at a time, holding peak at a frame rather than a window.

    Selection is unchanged: the reference timestamps come from ``spec.grid``, and
    each is served by its nearest source frame, which is what the indexed sensor
    picks too. The walk needs one frame of lookahead to know whether the current
    frame or the next is nearer.

    Windows must not overlap, which a forward-only walk cannot serve.
    """

    def __init__(
        self,
        source: DataSource,
        stream_idx: int = 0,
        decode_config: CpuVideoDecodeConfig = DEFAULT_VIDEO_DECODE_CONFIG,
        *,
        timestamps_ns: npt.NDArray[np.int64],
    ) -> None:
        """Initialize the streaming camera sensor.

        Args:
            source: Video data source. See
                :data:`cosmos_curator.core.sensors.types.types.DataSource`.
                The library accepts no URIs; callers open their own stream.
            stream_idx: PyAV index of the video stream to decode, usually 0.
            decode_config: Backend configuration for frame decoding, matching
                ``CameraSensor``. Its ``thread_count`` is worth choosing against
                whatever CPU reservation the caller holds.
            timestamps_ns: The recording's own capture times, one per frame, in
                ascending int64 nanoseconds. Supplied rather than read, because
                the only way to learn them from the container is to index it,
                which is the eager pass this sensor exists to avoid.

        Raises:
            ValueError: If *timestamps_ns* is not a non-empty, strictly
                ascending int64 array.

        """
        self._source = source
        self._stream_idx = stream_idx
        self._decode_config = decode_config
        self._timestamps_ns = _validate_timeline(timestamps_ns)

    @property
    def start_ns(self) -> int:
        """Return the first capture timestamp, in nanoseconds."""
        return int(self._timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        """Return the last capture timestamp, in nanoseconds."""
        return int(self._timestamps_ns[-1])

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NearestTimestampPolicy)

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Yield the recording's capture times in order, without resampling.

        Args:
            batch_size: ``0`` yields every timestamp in one batch; ``> 0`` yields
                consecutive batches of that size.

        Yields:
            Consecutive, non-overlapping int64 nanosecond arrays whose union is
            the whole timeline.

        Raises:
            ValueError: If *batch_size* is negative.

        """
        if batch_size < 0:
            msg = f"batch_size must be non-negative, got {batch_size}"
            raise ValueError(msg)
        if batch_size == 0:
            yield self._timestamps_ns
            return
        for start in range(0, len(self._timestamps_ns), batch_size):
            yield self._timestamps_ns[start : start + batch_size]

    def sample(self, spec: SamplingSpec, *, policy: NearestTimestampPolicy) -> Generator[StreamingCameraData]:
        """Walk the video once, yielding one payload per window of ``spec.grid``.

        One payload per window including empty ones, because ``SensorGroup``
        advances every sensor in lockstep and a skipped window would hand the
        next window's frames to the wrong one.

        Every payload draws on one shared decode, so they must be consumed in the
        order they were yielded. Skipping a payload is fine; going back to one
        raises, because the walk cannot rewind to serve it.

        Args:
            spec: The sampling grid to serve.
            policy: Its ``max_delta_ns`` bounds how far a reference may sit from
                the frame chosen for it.

        Yields:
            One :class:`StreamingCameraData` per window, in grid order.

        Raises:
            RuntimeError: If a payload is consumed after a later one has been.
            ValueError: If the stream has no displayable frames, if the grid's
                windows overlap or run backwards, if the recording's origin
                shifts its timestamps outside signed int64, or if a served frame
                disagrees with the timeline the payload published.

        """
        frames = iter_video_frames(self._source, self._stream_idx, self._decode_config)
        cursor = _Cursor(frames, timeline_ns=self._timestamps_ns)
        state = _WindowValidationState()
        sequence = itertools.count()
        for window in spec.grid:
            _validate_next_window(window, state)
            expected_ns = _serving_timestamps(self._timestamps_ns, window.timestamps_ns)
            yield StreamingCameraData(
                align_timestamps_ns=window.timestamps_ns,
                sensor_timestamps_ns=expected_ns,
                walk=cursor.serve(window, policy.max_delta_ns, next(sequence)),
            )


@attrs.define(slots=True)
class _Cursor:
    """The shared decode position, advanced by whichever window is being served."""

    _frames: Iterator[tuple[int, npt.NDArray[np.uint8]]] = attrs.field(alias="frames")
    _timeline_ns: npt.NDArray[np.int64] = attrs.field(alias="timeline_ns")
    _current: tuple[int, npt.NDArray[np.uint8]] | None = attrs.field(init=False, default=None)
    _upcoming: tuple[int, npt.NDArray[np.uint8]] | None = attrs.field(init=False, default=None)
    _started: bool = attrs.field(init=False, default=False)
    _index: int = attrs.field(init=False, default=0)
    _drawn_from: int = attrs.field(init=False, default=-1)

    def serve(
        self,
        window: SamplingWindow,
        max_delta_ns: int | None,
        sequence: int,
    ) -> Generator[StreamingFrame]:
        """Yield one row per reference timestamp in *window*."""
        for reference_ns in window.timestamps_ns:
            self._claim(sequence)
            self._start()
            if self._current is None:
                msg = "video stream contains no displayable frames"
                raise ValueError(msg)
            reference = int(reference_ns)
            while self._upcoming is not None and abs(self._upcoming[0] - reference) < abs(self._current[0] - reference):
                self._current = self._upcoming
                self._upcoming = self._advance()

            pts_ns, frame = self._current
            delta_ns = abs(pts_ns - reference)
            if max_delta_ns is not None and delta_ns > max_delta_ns:
                msg = (
                    f"max_delta_ns={max_delta_ns} exceeded: "
                    f"delta was {delta_ns} ns for reference={reference}, pts={pts_ns}"
                )
                raise ValueError(msg)
            yield reference, pts_ns, frame

    def _claim(self, sequence: int) -> None:
        """Refuse a payload that would draw from behind the walk's position.

        Every payload draws from one decode position that cannot rewind, so a
        window can only be served while nothing later has been. Skipping ahead is
        fine -- the walk reaches later references by advancing -- but going back
        would pick real frames for the wrong timestamps, which a tolerance check
        catches only by luck. Checked per row, so interleaving is caught too.
        """
        if sequence < self._drawn_from:
            msg = (
                "StreamingCameraData payloads must be consumed in the order they were yielded; "
                f"window {sequence} was asked for after window {self._drawn_from}, and one forward "
                "walk cannot go back"
            )
            raise RuntimeError(msg)
        self._drawn_from = sequence

    def _start(self) -> None:
        """Prime the first two frames on first use, so decode is lazy."""
        if self._started:
            return
        self._started = True
        first = next(self._frames, None)
        if first is None:
            return
        self._current = self._observed(first)
        self._upcoming = self._advance()

    def _advance(self) -> tuple[int, npt.NDArray[np.uint8]] | None:
        """Take the next frame, timestamped by the timeline rather than the container.

        A container ending before the timeline does is indistinguishable from a
        clean end unless checked here: left alone, later references would reuse
        the final decoded frame while still publishing the timeline's timestamps
        for observations no frame ever backed.
        """
        row = next(self._frames, None)
        if row is None:
            if self._index != len(self._timeline_ns):
                msg = (
                    f"container ended after {self._index} frames but timestamps_ns describes "
                    f"{len(self._timeline_ns)} observations"
                )
                raise ValueError(msg)
            return None
        return self._observed(row)

    def _observed(self, row: tuple[int, npt.NDArray[np.uint8]]) -> tuple[int, npt.NDArray[np.uint8]]:
        """Timestamp one frame from the timeline.

        The nth frame decoded is the nth observation recorded. Nothing compares
        their times: a container written at a declared rate spaces its frames
        evenly and cannot express a camera pausing, so a timeline that records
        one disagrees with it by a frame period from then on, while still
        describing the same frames in the same order.

        How many observations there are is a different matter, and is checked --
        here for a recording that outruns its timeline, and at the end of the
        walk for a timeline that outruns its recording.
        """
        _pts_ns, frame = row
        index = self._index
        self._index += 1
        if index >= len(self._timeline_ns):
            msg = (
                f"timestamps_ns describes {len(self._timeline_ns)} observations "
                "but the recording holds more frames than that"
            )
            raise ValueError(msg)
        return int(self._timeline_ns[index]), frame
