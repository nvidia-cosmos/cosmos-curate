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
"""Timestamp sampling grid."""

import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def make_ts_grid(
    start_ns: int,
    end_ns: int,
    sample_rate_hz: float,
) -> npt.NDArray[np.int64]:
    """Make a grid of timestamps in nanoseconds.

    Samples are ``start_ns + k * (1/sample_rate_hz)`` in float space, rounded
    to int64 nanoseconds.

    The returned timestamps stay on that regular sample interval, the grid
    always includes ``start_ns`` and continues until the final timestamp is
    strictly greater than ``end_ns``.

    This ensures that the last window produced by :class:`SamplingGrid` can
    always reach ``end_ns`` via nearest-neighbour lookup, even when
    ``end_ns - start_ns`` is not evenly divisible by the sample interval.

    Args:
        start_ns: the start timestamp in nanoseconds
        end_ns: the end timestamp in nanoseconds. The returned array satisfies
            ``arr[-2] <= end_ns < arr[-1]``.
        sample_rate_hz: the sample rate in Hz

    Returns:
        A numpy array of int64 timestamps in nanoseconds, strictly ascending
        and read-only.

    """
    if sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be greater than 0, got {sample_rate_hz=}"
        raise ValueError(msg)
    if end_ns < start_ns:
        msg = f"end_ns must be greater than or equal to start_ns, got {start_ns=} {end_ns=}"
        raise ValueError(msg)

    sample_interval = 1.0 / sample_rate_hz
    start = start_ns / 1_000_000_000
    end = end_ns / 1_000_000_000

    # Calculate the number of samples needed to cover the range, guarding against
    # floating-point roundoff at exact boundaries
    intervals_to_end = np.nextafter((end - start) / sample_interval, np.inf)
    sample_intervals_to_end = max(2, math.floor(intervals_to_end) + 2)

    # Build the grid of timestamps in float space, then round to int64 nanoseconds
    timestamps_s = start + np.arange(sample_intervals_to_end, dtype=np.float64) * sample_interval
    retval = np.round(timestamps_s * 1_000_000_000).astype(np.int64)
    if np.any(np.diff(retval) <= 0):
        msg = (
            "sample_rate_hz does not produce a strictly increasing nanosecond grid after rounding, "
            f"got {sample_rate_hz=}"
        )
        raise ValueError(msg)
    retval.flags.writeable = False
    return retval


class SamplingGrid:
    """Timestamp sampling grid.

    The expected use-case for SamplingGrid is:

    1. Given a set of timestamps spread evenly across a regular grid
    2. That extends from ``start_ns`` through the first on-grid timestamp
       strictly after the desired end time
    3. Data is sampled from that grid at a regularly spaced, fixed time interval
    4. The SamplingGrid iterator yields timestamp slices (views) per window

    However, the iterator is designed to be flexible enough to handle other
    use-cases, such as timestamps that are not evenly distributed over a regular
    grid.

    Surprising but correct behavior:

    Duplicate windows:

    If the supplied grid is sufficiently sparse, and the duration and stride
    are small enough, the iterator may return duplicated windows.

    This is correct behavior for uneven logging / burst sampling; window
    bounds still advance by stride_ns.

    If you are experiencing this, consider using data integrity checks to
    exclude data sources with insufficient density.

    Zero-length / empty windows:

    When SamplingGrid keeps empty windows, there's a clean invariant:

    yielded window at index i  ←→  time window starting at  start_ns + i * stride_ns

    If empty windows are filtered, that mapping breaks:

    # Works today (no filtering):
    for i, window in enumerate(grid):
        window_start = grid.start_ns + i * grid.stride_ns  # always correct

    # Breaks with filtering:
    # i=0 → window 0  (correct)
    # i=1 → window 2  (window 1 was empty, silently dropped — i is now wrong)

    Any code that uses window index as a proxy for time position silently
    produces wrong results. That's harder to debug than a ValueError.

    This is why SamplingGrid keeps empty windows.

    """

    def __init__(
        self,
        timestamps_ns: npt.NDArray[np.int64],
        stride_ns: int,
        duration_ns: int,
    ) -> None:
        """Initialize ``SamplingGrid``.

        Args:
            timestamps_ns: the timestamps to use when sampling data, in nanoseconds.
                Must be sorted in ascending order. If callers want the final
                real timestamp to be sampled by consumers using the half-open
                window convention, ``timestamps_ns`` must also include a
                boundary marker strictly after that final timestamp.
                :func:`make_ts_grid` does this automatically.

                The grid stores an internal defensive copy of this array so
                later caller-side mutation cannot change future windows.
            stride_ns: the stride to use when sampling data
            duration_ns: the duration to use when sampling data

        Raises:
            ValueError: if the timestamps are not sorted in ascending order
            ValueError: if the duration is not positive
            ValueError: if the stride is not positive
            ValueError: if the timestamps are empty

        """
        if duration_ns <= 0:
            msg = f"Duration must be positive, got {duration_ns=}"
            raise ValueError(msg)

        if stride_ns <= 0:
            msg = f"Stride must be positive, got {stride_ns=}"
            raise ValueError(msg)

        if len(timestamps_ns) == 0:
            msg = "Timestamps must be non-empty"
            raise ValueError(msg)

        if timestamps_ns.ndim != 1:
            msg = f"timestamps_ns must be 1-D, got ndim={timestamps_ns.ndim}"
            raise ValueError(msg)

        if timestamps_ns.dtype != np.int64:
            msg = f"timestamps_ns must have dtype int64, got {timestamps_ns.dtype}"
            raise ValueError(msg)

        if not np.all(np.diff(timestamps_ns) > 0):
            msg = "Timestamps must be strictly sorted in ascending order, no duplicates allowed"
            raise ValueError(msg)

        # Store an internal read-only copy so caller-side mutation of the
        # source array cannot silently change future windows.
        self._timestamps_ns = np.array(timestamps_ns, copy=True)
        self._timestamps_ns.flags.writeable = False
        self._stride_ns = stride_ns
        self._duration_ns = duration_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Return the timestamps to use when sampling data (nanoseconds)."""
        return self._timestamps_ns

    @property
    def start_ns(self) -> int:
        """Return the first timestamp (nanoseconds) in the series."""
        return int(self._timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        """Return the last timestamp (nanoseconds) in the series."""
        return int(self._timestamps_ns[-1])

    @property
    def stride_ns(self) -> int:
        """Window stride in nanoseconds."""
        return self._stride_ns

    @property
    def duration_ns(self) -> int:
        """Window duration in nanoseconds."""
        return self._duration_ns

    def __iter__(self) -> Iterator[npt.NDArray[np.int64]]:
        """Iterate over timestamp windows on the timeline.

        Window start times are ``start_ns + k * stride_ns`` for ``k = 0, 1, ...``,
        window end times are ``start_ns + k * stride_ns + duration_ns``.

        Non-empty yielded windows are closed timestamp slices used to define a
        half-open sampling interval ``[window[0], window[-1])``.

        Interpretation:
        - ``window[-1]`` is the exclusive right boundary marker.
        - If a timestamp lands exactly on a window boundary, it is sampled in the next
          window, not the current one.

        This prevents double-counting across adjacent windows.

        Empty windows:
        - A yielded window may be length 0 if no timestamps fall in that time range.
        - Empty windows are kept so window index ``i`` still maps to the time window
          starting at ``start_ns + i * stride_ns``.

        Degenerate case:
        - If a yielded window has length 1, that single timestamp acts as a boundary
          marker only; consumers using the half-open convention will produce zero
          samples for that window.

        Yields:
            View into ``timestamps_ns`` for samples in that window (dtype ``int64``).
            May be length 0 if no samples fall in the interval.

        """
        if self.start_ns == self.end_ns:
            yield self.timestamps_ns
            return

        start_ns = self.start_ns
        ts = self.timestamps_ns
        while start_ns < self.end_ns:
            window_end_ns = start_ns + self._duration_ns
            i = np.searchsorted(ts, start_ns, side="left")
            j = np.searchsorted(ts, window_end_ns, side="right")
            yield ts[i:j]
            start_ns += self._stride_ns
