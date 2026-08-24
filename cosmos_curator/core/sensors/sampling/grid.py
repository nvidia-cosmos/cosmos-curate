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
from fractions import Fraction
from typing import Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt
from attrs import validators

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    INT64_MAX,
    INT64_MIN,
    positive_value,
    strictly_increasing_int64_array,
)


class _HasHalfOpenWindowBounds(Protocol):
    start_ns: int
    exclusive_end_ns: int


def _end_ns_ge_start_ns(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: int,
) -> None:
    if value < instance.start_ns:
        msg = f"end_ns must be greater than or equal to start_ns, got {instance.start_ns=} {value=}"
        raise ValueError(msg)


def make_ts_grid(
    start_ns: int,
    end_ns: int | None = None,
    sample_rate_hz: float | None = None,
    *,
    exclusive_end_ns: int | None = None,
) -> tuple[int, int, npt.NDArray[np.int64]]:
    """Make a grid of timestamps in nanoseconds.

    Samples are ``start_ns + round(k * step_ns)``, where ``step_ns`` is
    ``1e9 / sample_rate_hz`` evaluated once in ``float64``. The association
    matters: ``(k * 1e9) / sample_rate_hz`` rounds once instead of twice and
    disagrees on a handful of near-ties at rates like 29.97. The origin is an
    integer and only the *relative* offsets go through a float, so no absolute
    nanosecond timestamp does, and the grid does not lose precision at
    epoch-scale origins. The grid always includes ``start_ns`` and
    the returned ``timestamps_ns`` is strictly ascending and read-only.

    Accuracy: every timestamp is within 1 ns of the exact rational grid, and the
    deviation does not accumulate along the grid. It is exactly zero when
    ``1e9 / sample_rate_hz`` is representable in ``float64`` (10, 25, 1000 Hz) and
    for shorter grids at rates like 30 Hz; isolated 1 ns deviations appear on rates
    whose exact interval has a large denominator, such as 29.97, after roughly an
    hour of grid. That is seven orders of magnitude below the sample interval and
    does not change which source observation a nearest-neighbour lookup selects.
    The bound holds while offsets stay under 2**53 ns, about 104 days of grid.
    Past that ``float64`` can no longer represent every integer nanosecond, so
    deviations grow in proportion to the offset -- roughly 5 ns at 8 years and
    50 ns at 80 -- and the half-open bracket is no longer guaranteed. Nothing
    guards against it: such grids are hundreds of times longer than any this is
    built for, and low sample rates reach them with an unremarkable sample count.

    Exactly one of ``end_ns`` and ``exclusive_end_ns`` must be supplied.

    This function constructs a numeric timestamp grid only. It does not inspect
    sensor data or determine whether the requested bounds are covered by any
    source observations. Callers are responsible for choosing bounds that are
    valid for their use case; sensor sampling and alignment code is responsible
    for determining whether observations can satisfy the returned timestamps.

    Inclusive end (``end_ns``):
        ``end_ns`` is the last timestamp to *include*. The grid continues
        until the final sample is strictly greater than ``end_ns``; that
        sample is retained as an exclusive boundary marker so the requested
        ``end_ns`` remains reachable under half-open sampling-window
        semantics, even when ``end_ns - start_ns`` is not evenly divisible by
        the sample interval. The returned tuple satisfies
        ``timestamps_ns[-1] <= end_ns < exclusive_end_ns``.

    Exclusive end (``exclusive_end_ns``):
        ``exclusive_end_ns`` is the half-open right boundary; the grid stops
        strictly before it. The supplied value is returned unchanged as the
        second element of the tuple, which makes this convenient for clip
        spans that already use ``[start_s, end_s)`` semantics. The returned
        tuple satisfies ``timestamps_ns[-1] < exclusive_end_ns`` and
        ``exclusive_end_ns`` equals the supplied value exactly.

    Args:
        start_ns: the start timestamp in nanoseconds.
        end_ns: optional inclusive end timestamp in nanoseconds. Mutually
            exclusive with ``exclusive_end_ns``.
        sample_rate_hz: the sample rate in Hz. Required.
        exclusive_end_ns: optional exclusive end timestamp in nanoseconds.
            Mutually exclusive with ``end_ns``.

    Returns:
        A ``(start_ns, exclusive_end_ns, timestamps_ns)`` tuple where
        ``timestamps_ns`` is a strictly ascending, read-only ``int64`` array.

    Raises:
        ValueError: if neither or both of ``end_ns`` and ``exclusive_end_ns``
            are supplied, if ``sample_rate_hz`` is missing or non-positive,
            if the supplied bound precedes ``start_ns``, if the grid would
            extend outside signed int64 nanoseconds, or if rounding to
            nanoseconds does not produce a strictly increasing grid.

    """
    if sample_rate_hz is None or sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be greater than 0, got {sample_rate_hz=}"
        raise ValueError(msg)
    if end_ns is not None and exclusive_end_ns is not None:
        msg = f"exactly one of end_ns or exclusive_end_ns must be supplied, got both {end_ns=} and {exclusive_end_ns=}"
        raise ValueError(msg)

    # Reduce the half-open form to the existing inclusive-end algorithm by
    # shifting one nanosecond inward, then override the returned boundary so
    # the supplied exclusive_end_ns is preserved exactly.
    if exclusive_end_ns is not None:
        if exclusive_end_ns <= start_ns:
            msg = f"exclusive_end_ns must be greater than start_ns, got {start_ns=} {exclusive_end_ns=}"
            raise ValueError(msg)
        inclusive_end_ns = exclusive_end_ns - 1
    elif end_ns is not None:
        if end_ns < start_ns:
            msg = f"end_ns must be greater than or equal to start_ns, got {start_ns=} {end_ns=}"
            raise ValueError(msg)
        inclusive_end_ns = end_ns
    else:
        msg = "exactly one of end_ns or exclusive_end_ns must be supplied, got neither"
        raise ValueError(msg)

    sample_interval = 1.0 / sample_rate_hz
    span_s = (inclusive_end_ns - start_ns) / 1_000_000_000

    # Calculate the number of samples needed to cover the range, guarding against
    # floating-point roundoff at exact boundaries. The span is a relative
    # nanosecond count rather than an absolute timestamp, so this stays clear of
    # the hundreds-of-nanoseconds float64 spacing at epoch scale.
    intervals_to_end = np.nextafter(span_s / sample_interval, np.inf)
    sample_intervals_to_end = max(2, math.floor(intervals_to_end) + 2)

    # Range-check in exact Python arithmetic *before* building anything. Reading the
    # bound back off the constructed array cannot work: the offsets are int64 and wrap
    # silently, and np.diff of a wrapped pair wraps back positive, so the
    # strictly-increasing check below does not catch it either. The two wraps together
    # would return a grid whose exclusive_end_ns sits below its last timestamp.
    # Offsets ascend from zero, so the last one bounds the grid.
    interval_ns = Fraction(1_000_000_000) / Fraction(sample_rate_hz)
    last_offset_ns = round(interval_ns * (sample_intervals_to_end - 1))
    last_ns = start_ns + last_offset_ns
    if not (INT64_MIN <= start_ns <= INT64_MAX and last_offset_ns <= INT64_MAX and INT64_MIN <= last_ns <= INT64_MAX):
        msg = (
            "grid extends outside signed int64 nanoseconds, got "
            f"{start_ns=} last_ns={last_ns} last_offset_ns={last_offset_ns}"
        )
        raise ValueError(msg)

    # start_ns is deliberately absent from the arithmetic below: at epoch magnitudes
    # adjacent float64 values are 256 ns apart, while across a clip-length offset they
    # are sub-nanosecond. Two properties of this expression also matter, and are pinned
    # by tests:
    #
    #   * Each offset is step_ns * k, computed independently. Accumulating step_ns
    #     instead (a running sum, np.cumsum) lets rounding error compound: 1132 ns of
    #     drift over 18 h at 29.97 Hz, against <= 1 ns here.
    #   * step_ns keeps the interval's fractional part. Pre-rounding it to an integer
    #     nanosecond step makes every sample inherit all prior truncation -- 36 us per
    #     hour at 30 Hz -- which pulls the boundary marker below end_ns and breaks the
    #     half-open contract on spans as ordinary as one second.
    step_ns = 1_000_000_000.0 / sample_rate_hz
    offsets_ns = np.round(step_ns * np.arange(sample_intervals_to_end, dtype=np.float64)).astype(np.int64)
    offsets_ns += np.int64(start_ns)
    retval = offsets_ns
    if np.any(np.diff(retval) <= 0):
        msg = (
            "sample_rate_hz does not produce a strictly increasing nanosecond grid after rounding, "
            f"got {sample_rate_hz=}"
        )
        raise ValueError(msg)

    # The pre-check bounds round(interval_ns * k) in exact arithmetic; the offsets above
    # come from round(step_ns * k) in float64, rounded into int64. The two agree to 1 ns,
    # and 1 ns is the entire margin at the int64 boundary. Offsets ascend from zero, so a
    # last timestamp at or below the origin can only mean the shift wrapped. O(1), and
    # independent of how the offsets were built.
    if int(retval[-1]) <= start_ns:
        msg = f"grid wrapped past signed int64 nanoseconds, got {start_ns=} last_ns={int(retval[-1])}"
        raise ValueError(msg)

    retval.flags.writeable = False

    timestamps_ns = retval[:-1]
    out_exclusive_end_ns = exclusive_end_ns if exclusive_end_ns is not None else int(retval[-1])
    return start_ns, out_exclusive_end_ns, timestamps_ns


def _start_ns_le_first_timestamp(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    """Validate start_ns is less than or equal to the first timestamp."""
    if len(value) == 0:
        return
    first_ts = int(value[0])
    if instance.start_ns > first_ts:
        msg = f"start_ns must be <= timestamps_ns[0], got {instance.start_ns} > {first_ts}"
        raise ValueError(msg)


def _start_ns_eq_first_timestamp(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    """Validate start_ns is equal to the first timestamp."""
    if len(value) == 0:
        return
    first_ts = int(value[0])
    if instance.start_ns != first_ts:
        msg = f"start_ns must == timestamps_ns[0], got {instance.start_ns} != {first_ts}"
        raise ValueError(msg)


def _end_ns_lt_exclusive_end_ns(
    instance: _HasHalfOpenWindowBounds,
    _attribute: object,
    value: npt.NDArray[np.int64],
) -> None:
    if len(value) == 0:
        return
    end_ns = int(value[-1])
    if end_ns >= instance.exclusive_end_ns:
        msg = f"end_ns must be < exclusive_end_ns, got {end_ns} >= {instance.exclusive_end_ns}"
        raise ValueError(msg)


def _copy_numpy_array(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Copy a numpy array."""
    return np.array(array, copy=True)


def _copy_as_readonly_view(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Copy a numpy array and expose the copy as a read-only view."""
    return as_readonly_view(_copy_numpy_array(array))


@attrs.define(frozen=True)
class SamplingWindow:
    """One half-open sampling window `[start_ns, exclusive_end_ns)`.

    For non-empty windows, the timestamps are strictly increasing and satisfy
    ``start_ns <= timestamps_ns[0]`` and
    ``timestamps_ns[-1] < exclusive_end_ns``.

    Attributes:
        start_ns:
            Left boundary of the half-open interval, must be less than or
            equal to the first timestamp in timestamps_ns.
        exclusive_end_ns:
            Exclusive right boundary of the window, must be greater than the
            last timestamp in timestamps_ns.
        timestamps_ns:
            Strictly increasing ``int64`` timestamps. For non-empty windows,
            the first timestamp must be greater than or equal to ``start_ns``
            and the last timestamp must be strictly less than
            ``exclusive_end_ns``.

    """

    start_ns: int
    exclusive_end_ns: int = attrs.field(validator=_end_ns_ge_start_ns)
    timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=validators.and_(
            strictly_increasing_int64_array,
            _start_ns_le_first_timestamp,
            _end_ns_lt_exclusive_end_ns,
        ),
    )

    def __len__(self) -> int:
        """Return the number of timestamps in this window."""
        return len(self.timestamps_ns)


@attrs.define(frozen=True, hash=False)
class SamplingGrid:
    """Iterable view over timestamped sampling windows.

    ``SamplingGrid`` turns a strictly increasing timestamp series into a
    sequence of half-open windows whose nominal starts are
    ``start_ns + k * stride_ns``. Iteration yields :class:`SamplingWindow`
    objects that preserve the exclusive right boundary needed by downstream
    samplers.

    The common use case is a regular timestamp grid that includes one extra
    boundary marker strictly after the final sample that should remain
    reachable under half-open window semantics. :func:`make_ts_grid` produces
    timestamps in that format automatically. The iterator is also intentionally
    permissive enough to support irregular or bursty sensor timelines.

    Empty windows are yielded instead of being filtered out. That preserves the
    invariant that window index ``i`` always corresponds to the nominal time
    range starting at ``start_ns + i * stride_ns``. On sparse or irregular
    grids, consecutive yielded windows may therefore look identical even though
    their nominal time bounds differ; this is expected.

    Attributes:
        start_ns:
            Left boundary of the half-open interval, must be equal to the first
            timestamp in timestamps_ns.
        exclusive_end_ns:
            Exclusive right boundary of the window, must be greater than the
            last timestamp in timestamps_ns.
        timestamps_ns:
            One-dimensional, strictly increasing ``int64`` timestamp array in
            nanoseconds.
        stride_ns:
            Distance in nanoseconds between consecutive nominal window starts.
            Must be positive.
        duration_ns:
            Width in nanoseconds of each sampling window. Must be positive.

    """

    __hash__ = None  # type: ignore[assignment]
    start_ns: int
    exclusive_end_ns: int = attrs.field(validator=_end_ns_ge_start_ns)
    timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=_copy_as_readonly_view,
        validator=validators.and_(
            strictly_increasing_int64_array,
            _start_ns_eq_first_timestamp,
            _end_ns_lt_exclusive_end_ns,
        ),
    )
    stride_ns: int = attrs.field(validator=positive_value)
    duration_ns: int = attrs.field(validator=positive_value)

    def __iter__(self) -> Iterator[SamplingWindow]:
        """Iterate over timestamp windows on the timeline.

        The nominal sampling-window start times are
        ``start_ns + k * stride_ns`` for ``k = 0, 1, ...``, and the nominal
        window end times are ``start_ns + k * stride_ns + duration_ns``.

        Each yielded :class:`SamplingWindow` stores active timestamps in
        ``window.timestamps_ns`` and the nominal half-open bounds in
        ``window.start_ns`` and ``window.exclusive_end_ns``. The bounds advance
        by ``stride_ns`` even when a window is empty or its timestamps begin
        later within that interval.

        Interpretation:
        - ``window.exclusive_end_ns`` is the exclusive right boundary marker.
        - If a timestamp lands exactly on a window boundary, it is sampled in
          the next window, not the current one.

        This prevents double-counting across adjacent windows.

        Empty windows:
        - A yielded window may have ``len(window) == 0`` if no active
          timestamps fall in that time range.
        - Empty windows are kept so window index ``i`` still maps to the time window
          starting at ``start_ns + i * stride_ns``.

        Boundary-only case:
        - A yielded window may have zero active timestamps while still carrying
          a real boundary marker in ``exclusive_end_ns``. This represents a
          valid half-open sampling window with no active timestamps assigned to
          it.

        Yields:
            ``SamplingWindow`` describing one half-open sampling window.

        """
        if self.start_ns == self.exclusive_end_ns:
            yield SamplingWindow(
                start_ns=self.start_ns,
                exclusive_end_ns=self.exclusive_end_ns,
                timestamps_ns=self.timestamps_ns[:-1],
            )
            return

        start_ns = self.start_ns
        ts = np.concatenate([self.timestamps_ns, [self.exclusive_end_ns]])
        while start_ns < self.exclusive_end_ns:
            window_end_ns = min(start_ns + self.duration_ns, self.exclusive_end_ns)
            i, j = np.searchsorted(ts, [start_ns, window_end_ns], side="left")
            yield SamplingWindow(start_ns=start_ns, exclusive_end_ns=window_end_ns, timestamps_ns=ts[i:j])
            start_ns += self.stride_ns
