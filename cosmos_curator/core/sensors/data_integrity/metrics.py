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

"""Data-integrity metrics.

A metric is a stateful measuring *instrument*. Feed it successive windows of a
stream with ``update(**inputs)`` and finalize an immutable ``Measurement`` with
``measurement()``. The instrument is mutable (that is what carries the running
state); the measurement it produces is frozen and carries the measured facts only
-- it holds no pass/fail policy (that is the evaluators' job). See
``docs/curator/design/data-integrity-design.md``.

Instruments do not buffer the samples: each keeps only the seam state needed to
stitch consecutive windows (for example, the previous window's last timestamp) plus
O(1) running summaries (counts, extremes, the first offending index), so state is
bounded regardless of stream length -- no metric grows its output with the number of
events found. A one-shot measurement over a whole array is a single ``update()``
followed by ``measurement()``.

What a metric folds is its own business. Most fold windows of one stream's timeline;
the multi-sensor metrics at the end of this module fold one *sensor's* recording
bounds per ``update()``, which is what lets a caller measuring a whole session feed
each stream as it finishes.

Time-domain values are nanoseconds in the ``int64`` range, matching ``core.sensors``:
array fields are ``np.int64``, while scalar discrete fields (``expected_period_ns``,
``max_gap_ns``) are Python ``int`` in that range. Aggregate statistics over them (for
example ``actual_mean_period_ns``) are float nanoseconds, since a mean or spread is not
generally an integer.
"""

from typing import ClassVar, Protocol

import attrs
import numpy as np
from numpy.typing import NDArray

# Nanoseconds per second — time-domain fields are int64 ns, matching core.sensors.
NSEC_PER_SEC = 1_000_000_000
_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


def _checked_diffs(timestamps_ns: NDArray[np.int64]) -> NDArray[np.int64]:
    """Consecutive int64 differences, rejecting intervals too large to represent.

    An int64 subtraction wraps silently when the true interval exceeds int64 range
    (for example a corrupt sentinel next to an epoch timestamp), which could turn a
    backward step or a huge gap into a clean forward interval and mask a defect.
    Detect the wrap by comparing the computed sign against the true order (a plain
    comparison never overflows) and raise rather than report a corrupted value.
    """
    diffs = np.diff(timestamps_ns)
    later, earlier = timestamps_ns[1:], timestamps_ns[:-1]
    wrapped = ((later > earlier) & (diffs < 0)) | ((later < earlier) & (diffs > 0))
    if bool(np.any(wrapped)):
        msg = "timestamp interval exceeds int64 range (corrupt or unrepresentable timestamps)"
        raise ValueError(msg)
    return diffs


def _validate_timestamps(timestamps_ns: object) -> None:
    """Reject input that violates the 1-D int64 contract, before any state changes.

    Metrics mutate running state as they fold a window, so a bad array must be caught up
    front rather than after a partial update. int64 is required because both the overflow
    check in :func:`_checked_diffs` and the direct-comparison ordering assume it. The
    parameter is typed ``object`` so the runtime guards hold even for a caller that
    bypasses the type checker (the declared inputs are ``NDArray[np.int64]``).
    """
    if not isinstance(timestamps_ns, np.ndarray):
        msg = f"timestamps_ns must be a numpy ndarray, got {type(timestamps_ns).__name__}"
        raise TypeError(msg)
    if timestamps_ns.ndim != 1:
        msg = f"timestamps_ns must be 1-D, got {timestamps_ns.ndim}-D"
        raise ValueError(msg)
    if timestamps_ns.dtype != np.int64:
        msg = f"timestamps_ns must be int64, got {timestamps_ns.dtype}"
        raise ValueError(msg)


def _expected_period_ns(expected_hz: float) -> int:
    """Validate a sample rate and return its expected period as int64 nanoseconds.

    The rate-based metrics share this so their bounds agree. It rejects any rate whose
    discrete period is not a representable int64 in ``[1, _INT64_MAX]``:

    - non-positive or non-finite (``NaN`` / ``inf``)
    - so high the period rounds below 1 ns
    - so low the period overflows int64 nanoseconds (which would otherwise raise an
      incidental ``OverflowError`` or leave an infinite period in place)
    """
    if not (expected_hz > 0.0 and np.isfinite(expected_hz)):
        msg = f"expected_hz must be positive and finite, got {expected_hz}"
        raise ValueError(msg)
    period_float = NSEC_PER_SEC / expected_hz
    if not np.isfinite(period_float) or period_float > _INT64_MAX:
        msg = f"expected_hz too low: the expected period exceeds int64 nanoseconds, got {expected_hz}"
        raise ValueError(msg)
    period = int((NSEC_PER_SEC + expected_hz / 2.0) / expected_hz)  # round half up
    if period < 1:
        msg = f"expected_hz too high: the discrete period rounds below 1 ns, got {expected_hz}"
        raise ValueError(msg)
    return period


class Measurement(Protocol):
    """Common protocol for the immutable facts a metric produces.

    Each concrete measurement is a frozen type with typed fields. Evaluators read a
    field via an explicit accessor.

    Every measurement reports ``is_defined``: whether the input met the metric's
    computational minimum so the measured values are well-defined (the discrete
    analogue of ``NaN`` for, say, the mean of an empty array). Callers must not
    evaluate an undefined measurement; check ``is_defined`` first.
    """

    @property
    def is_defined(self) -> bool:
        """Whether the measurement is well-defined for the input it was taken over."""
        ...  # pragma: no cover


@attrs.define(frozen=True)
class TimestampOrderingMeasurement:
    """How a timestamp stream steps: increasing, equal, or decreasing.

    Each consecutive step is one of three cases, by the sign of the difference:
    ``> 0`` increasing, ``== 0`` duplicate, ``< 0`` decreasing (backward). The two
    failure counts are reported separately so evaluation policy can choose which
    matter:

    - non-decreasing (duplicates allowed): require ``decreasing_count == 0``
    - strictly increasing (no duplicates): require ``strict_violation_count == 0``

    ``duplicate_count`` counts *adjacent* duplicates only. Combined with
    ``decreasing_count == 0`` that is exact for strict increase: a non-decreasing,
    adjacent-duplicate-free stream is strictly increasing, hence globally unique.

    Attributes:
        decreasing_count: number of backward steps (``t[i] < t[i-1]``)
        duplicate_count: number of adjacent duplicate steps (``t[i] == t[i-1]``)
        first_decreasing_index: index of the first backward sample, or ``None``
        first_duplicate_index: index of the first adjacent duplicate, or ``None``
        num_samples: number of timestamps analyzed

    """

    decreasing_count: int
    duplicate_count: int
    first_decreasing_index: int | None
    first_duplicate_index: int | None
    num_samples: int

    #: Fewer than this many samples has no step to classify (measurement undefined).
    MIN_SAMPLES: ClassVar[int] = 2

    @property
    def is_defined(self) -> bool:
        """True once there are at least two samples (one step) to classify."""
        return self.num_samples >= self.MIN_SAMPLES

    @property
    def strict_violation_count(self) -> int:
        """Backward steps plus adjacent duplicates; for a defined measurement, zero iff strictly increasing."""
        return self.decreasing_count + self.duplicate_count


class TimestampOrderingMetric:
    """Instrument for :class:`TimestampOrderingMeasurement`.

    Feed successive windows with ``update(timestamps_ns=...)``; the instrument
    stitches the seam between windows using the previous window's last timestamp,
    and tracks global indices via a running sample offset. State is O(1) regardless
    of stream length.
    """

    def __init__(self) -> None:
        """Initialize empty running state."""
        self._decreasing = 0
        self._duplicate = 0
        self._first_decreasing: int | None = None
        self._first_duplicate: int | None = None
        self._num_samples = 0
        self._last_ts: int | None = None

    def update(self, *, timestamps_ns: NDArray[np.int64]) -> None:
        """Fold one window of nanosecond timestamps into the running state."""
        _validate_timestamps(timestamps_ns)
        window_len = len(timestamps_ns)
        if window_len == 0:
            return
        base = self._num_samples

        # Seam: the step from the previous window's last sample to this window's first.
        if self._last_ts is not None:
            seam = int(timestamps_ns[0]) - self._last_ts
            if seam < 0:
                self._decreasing += 1
                if self._first_decreasing is None:
                    self._first_decreasing = base
            elif seam == 0:
                self._duplicate += 1
                if self._first_duplicate is None:
                    self._first_duplicate = base

        # Internal steps within this window. Compare timestamps directly rather than
        # via np.diff: subtraction can overflow int64 on corrupt input and flip a
        # backward step into a clean forward one; a comparison never overflows.
        decreasing = timestamps_ns[1:] < timestamps_ns[:-1]
        duplicate = timestamps_ns[1:] == timestamps_ns[:-1]
        dec_count = int(np.count_nonzero(decreasing))
        dup_count = int(np.count_nonzero(duplicate))
        if dec_count and self._first_decreasing is None:
            self._first_decreasing = base + int(np.argmax(decreasing)) + 1
        if dup_count and self._first_duplicate is None:
            self._first_duplicate = base + int(np.argmax(duplicate)) + 1
        self._decreasing += dec_count
        self._duplicate += dup_count

        self._num_samples += window_len
        self._last_ts = int(timestamps_ns[-1])

    def measurement(self) -> TimestampOrderingMeasurement:
        """Finalize the immutable ordering measurement."""
        return TimestampOrderingMeasurement(
            decreasing_count=self._decreasing,
            duplicate_count=self._duplicate,
            first_decreasing_index=self._first_decreasing,
            first_duplicate_index=self._first_duplicate,
            num_samples=self._num_samples,
        )


@attrs.define(frozen=True)
class RateMeasurement:
    """Deviation of the observed mean sample period from the expected period.

    Compares consecutive inter-sample intervals to the expected period and reports a
    percentage deviation in *period* space. Every forward interval is included; only
    non-monotonic steps are dropped. Detecting dropouts is ``TimestampGapMetric``'s
    job, not this metric's.

    Attributes:
        period_deviation_percent: relative deviation of the mean interval from the
            expected period, as a percentage. Period-space: a 2x rate error (actual
            200 Hz vs expected 100 Hz) reads as 50%, not 100%.
        expected_hz: expected sampling rate in Hz
        expected_period_ns: expected inter-sample period in nanoseconds
        actual_mean_hz: reciprocal of the mean interval (``1e9 / actual_mean_period_ns``),
            not the arithmetic mean of per-interval rates
        actual_mean_period_ns: mean inter-sample interval in nanoseconds
        num_samples: number of timestamps
        num_intervals: number of intervals analyzed after filtering
        num_filtered: number of non-monotonic (<= 0) intervals dropped

    """

    period_deviation_percent: float
    expected_hz: float
    expected_period_ns: int
    actual_mean_hz: float
    actual_mean_period_ns: float
    num_samples: int
    num_intervals: int
    num_filtered: int

    @property
    def is_defined(self) -> bool:
        """True when at least one valid interval remained after filtering."""
        return self.num_intervals > 0


class RateMetric:
    """Instrument for :class:`RateMeasurement`.

    Folds windows of nanosecond timestamps, accumulating the sum and count of the
    valid intervals plus one seam timestamp, so state is O(1). ``deviation`` is
    computed from the running sum at ``measurement()`` time.
    """

    def __init__(self, *, expected_hz: float) -> None:
        """Initialize with the expected sampling rate in Hz."""
        self._expected_hz = expected_hz
        self._expected_period_ns = _expected_period_ns(expected_hz)
        self._sum_valid = 0.0
        self._num_valid = 0
        self._num_filtered = 0
        self._num_samples = 0
        self._last_ts: int | None = None

    def update(self, *, timestamps_ns: NDArray[np.int64]) -> None:
        """Fold one window of nanosecond timestamps into the running state."""
        _validate_timestamps(timestamps_ns)
        window_len = len(timestamps_ns)
        if window_len == 0:
            return
        # Prepend the previous window's last sample so the seam interval is overflow-checked
        # like any other (matching TimestampGapMetric / JitterMetric); computing it separately
        # would bypass _checked_diffs and let an unrepresentable seam through.
        if self._last_ts is not None:
            ts = np.concatenate((np.array([self._last_ts], dtype=np.int64), timestamps_ns))
        else:
            ts = timestamps_ns

        diffs = _checked_diffs(ts)
        forward = diffs[diffs > 0]
        # Sum in float, not int64: individually representable intervals can overflow int64
        # cumulatively, which would corrupt the mean period into a negative value.
        self._sum_valid += float(forward.astype(np.float64).sum())
        self._num_valid += len(forward)
        self._num_filtered += int(len(diffs) - len(forward))

        self._num_samples += window_len
        self._last_ts = int(timestamps_ns[-1])

    def measurement(self) -> RateMeasurement:
        """Finalize the immutable rate measurement."""
        if self._num_valid == 0:
            # No usable intervals (empty or all filtered): the rate is undefined.
            return RateMeasurement(
                period_deviation_percent=float("nan"),
                expected_hz=self._expected_hz,
                expected_period_ns=self._expected_period_ns,
                actual_mean_hz=float("nan"),
                actual_mean_period_ns=float("nan"),
                num_samples=self._num_samples,
                num_intervals=0,
                num_filtered=self._num_filtered,
            )

        # period_deviation_percent is a dimensionless ratio (interval error over expected
        # period); the API is nanoseconds throughout (expected_period_ns from the fixed 1e9 ns/s).
        total_error = self._sum_valid - self._expected_period_ns * self._num_valid
        period_deviation_percent = abs(total_error) / (self._expected_period_ns * self._num_valid) * 100.0
        actual_mean_period_ns = self._sum_valid / self._num_valid
        actual_mean_hz = NSEC_PER_SEC / actual_mean_period_ns

        return RateMeasurement(
            period_deviation_percent=period_deviation_percent,
            expected_hz=self._expected_hz,
            expected_period_ns=self._expected_period_ns,
            actual_mean_hz=actual_mean_hz,
            actual_mean_period_ns=actual_mean_period_ns,
            num_samples=self._num_samples,
            num_intervals=self._num_valid,
            num_filtered=self._num_filtered,
        )


@attrs.define(frozen=True, eq=False)
class TimestampGapMeasurement:
    """Gap summary for a timestamp stream (missing-sample detection).

    Flags inter-sample intervals that reach 1.5 or more expected periods (the exact
    cutoff ``interval >= (3 * period + 1) // 2``), implying one or more missing
    samples across that interval. The measurement is an O(1) summary — a count, the
    largest gap, and the location of the first gap — not a per-event log; see the
    "first-event, not a log" note on :class:`TimestampGapMetric`.

    Attributes:
        max_gap_ns: the largest gap interval (later minus earlier timestamp), in nanoseconds
        expected_period_ns: expected inter-sample period in nanoseconds
        expected_hz: expected sampling rate in Hz
        num_samples: number of timestamps
        num_gaps: number of gap events detected
        first_gap_index: index of the first sample immediately following a gap (the
            later endpoint of the first gap interval), or ``None`` if there are no
            gaps -- mirroring ``TimestampOrderingMeasurement.first_decreasing_index``

    """

    max_gap_ns: int
    expected_period_ns: int
    expected_hz: float
    num_samples: int
    num_gaps: int
    first_gap_index: int | None

    #: Need at least this many timestamps to form one interval (undefined below it).
    MIN_TIMESTAMPS: ClassVar[int] = 2

    @property
    def is_defined(self) -> bool:
        """True once there is at least one interval to check for gaps."""
        return self.num_samples >= self.MIN_TIMESTAMPS


class TimestampGapMetric:
    """Instrument for :class:`TimestampGapMeasurement`.

    An inter-sample interval is a gap when ``interval >= (3 * period + 1) // 2`` —
    the exact integer form of ``interval / period >= 1.5``, with an exact 1.5-period
    tie counting as a gap. It implies one or more missing samples. Folds windows with
    O(1) seam state (the previous window's last timestamp), so the interval across a
    window boundary is checked like any other.

    First-event, not a log: the measurement summarizes gaps in O(1) space — how many
    (``num_gaps``), the largest (``max_gap_ns``), and where the first one is
    (``first_gap_index``) — rather than recording every gap event. This mirrors
    ``TimestampOrderingMetric``, keeping the metric's space bounded regardless of how
    many gaps a pathological stream contains; a full per-event log can be added later
    if a consumer needs one.
    """

    def __init__(self, *, expected_hz: float) -> None:
        """Initialize with the expected sampling rate in Hz."""
        self._expected_hz = expected_hz
        self._expected_period_ns = _expected_period_ns(expected_hz)
        self._num_samples = 0
        self._last_ts: int | None = None
        self._num_gaps = 0
        self._max_gap_ns = 0
        self._first_gap_index: int | None = None

    def update(self, *, timestamps_ns: NDArray[np.int64]) -> None:
        """Fold one window of nanosecond timestamps into the running state."""
        _validate_timestamps(timestamps_ns)
        window_len = len(timestamps_ns)
        if window_len == 0:
            return
        period = self._expected_period_ns
        base = self._num_samples

        # Prepend the previous window's last sample so the seam interval is checked too;
        # ``prepended`` records the offset it adds so gap indices map back to global samples.
        if self._last_ts is not None:
            ts = np.concatenate((np.array([self._last_ts], dtype=np.int64), timestamps_ns))
            prepended = 1
        else:
            ts = timestamps_ns
            prepended = 0
        gaps = _checked_diffs(ts)

        # A gap is an interval that rounds to >= 2 expected periods, i.e. interval >= 1.5 * period.
        # Use exact integer arithmetic — round(x / period) >= 2  <=>  x >= (3*period + 1)//2 — rather
        # than a float division and int cast, which lose precision past 2^53 and can warn on the cast.
        gap_threshold = (3 * period + 1) // 2
        if gap_threshold > _INT64_MAX:
            idx = np.array([], dtype=np.intp)  # no representable interval can reach 1.5 periods
        else:
            idx = np.flatnonzero(gaps >= np.int64(gap_threshold))
        if idx.size:
            self._num_gaps += int(idx.size)
            self._max_gap_ns = max(self._max_gap_ns, int(gaps[idx].max()))
            if self._first_gap_index is None:
                # Later endpoint of the first gap interval, in global sample indices: interval k
                # spans ts[k]..ts[k+1]; the prepended seam sample shifts local indices by one.
                self._first_gap_index = base + int(idx[0]) + 1 - prepended

        self._num_samples += window_len
        self._last_ts = int(timestamps_ns[-1])

    def measurement(self) -> TimestampGapMeasurement:
        """Finalize the immutable gap measurement."""
        return TimestampGapMeasurement(
            max_gap_ns=self._max_gap_ns,
            expected_period_ns=self._expected_period_ns,
            expected_hz=self._expected_hz,
            num_samples=self._num_samples,
            num_gaps=self._num_gaps,
            first_gap_index=self._first_gap_index,
        )


@attrs.define(frozen=True)
class JitterMeasurement:
    """Short-term variation in inter-sample timing, independent of average rate.

    Summarizes how much inter-sample intervals vary around their mean period.
    Distinct from ``RateMeasurement``, which measures average-rate deviation, not
    variation.

    Attributes:
        jitter_percent: jitter as a percentage
        expected_hz: expected sample rate in Hz
        num_samples: number of timestamps
        num_intervals: number of intervals analyzed
        num_filtered: number of non-monotonic (<= 0) intervals dropped

    """

    jitter_percent: float
    expected_hz: float
    num_samples: int
    num_intervals: int
    num_filtered: int

    #: Need at least this many intervals for a standard deviation.
    MIN_INTERVALS: ClassVar[int] = 2

    @property
    def is_defined(self) -> bool:
        """True once there are at least two intervals to measure spread."""
        return self.num_intervals >= self.MIN_INTERVALS


class JitterMetric:
    """Instrument for :class:`JitterMeasurement`.

    Folds windows, combining each window's count/mean/sum-of-squared-deviations into
    the running aggregate (Chan's parallel variance), so state is O(1) and the
    result matches a one-shot sample standard deviation (``ddof=1``). Non-monotonic
    steps are dropped.
    """

    def __init__(self, *, expected_hz: float) -> None:
        """Initialize with the expected sampling rate in Hz."""
        if not (expected_hz > 0.0 and np.isfinite(expected_hz)):
            msg = f"expected_hz must be positive and finite, got {expected_hz}"
            raise ValueError(msg)
        self._expected_hz = expected_hz
        self._expected_period_ns = _expected_period_ns(expected_hz)
        self._n = 0  # count of positive (monotonic) intervals
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared deviations from the mean
        self._num_samples = 0
        self._num_filtered = 0
        self._last_ts: int | None = None

    def update(self, *, timestamps_ns: NDArray[np.int64]) -> None:
        """Fold one window of nanosecond timestamps into the running state."""
        _validate_timestamps(timestamps_ns)
        window_len = len(timestamps_ns)
        if window_len == 0:
            return
        # Prepend the previous window's last sample so the seam interval is included.
        if self._last_ts is not None:
            ts = np.concatenate((np.array([self._last_ts], dtype=np.int64), timestamps_ns))
        else:
            ts = timestamps_ns
        diffs = _checked_diffs(ts)
        intervals = diffs[diffs > 0].astype(np.float64)
        self._num_filtered += len(diffs) - len(intervals)

        self._num_samples += window_len
        self._last_ts = int(timestamps_ns[-1])

        n_b = len(intervals)
        if n_b == 0:
            return
        # Chan's parallel algorithm: combine this window's (count, mean, M2) with the aggregate.
        mean_b = float(intervals.mean())
        m2_b = float(((intervals - mean_b) ** 2).sum())
        n_a = self._n
        if n_a == 0:
            self._n, self._mean, self._m2 = n_b, mean_b, m2_b
            return
        n_ab = n_a + n_b
        delta = mean_b - self._mean
        self._m2 += m2_b + delta * delta * n_a * n_b / n_ab
        self._mean += delta * n_b / n_ab
        self._n = n_ab

    def measurement(self) -> JitterMeasurement:
        """Finalize the immutable jitter measurement."""
        if self._n < JitterMeasurement.MIN_INTERVALS:
            jitter_percent = float("nan")
        else:
            variance = self._m2 / (self._n - 1)
            std = max(variance, 0.0) ** 0.5
            jitter_percent = std / self._expected_period_ns * 100.0
        return JitterMeasurement(
            jitter_percent=jitter_percent,
            expected_hz=self._expected_hz,
            num_samples=self._num_samples,
            num_intervals=self._n,
            num_filtered=self._num_filtered,
        )


@attrs.define(frozen=True)
class FrameReorderingPresentMeasurement:
    """Whether a supplied flag indicates frame reordering (B-frames, for H.264).

    Holds a caller-supplied boolean; it does not open a video or read a header. The
    flag is normally ``VideoMetadata.has_bframes`` -- libavcodec's ``has_b_frames``,
    the reorder-buffer size parsed from the header (not a scan of the stream). For
    H.264/AVC, the codec this targets, a non-empty reorder buffer reliably indicates
    B-frames, which complicate frame-accurate seeking and GPU decode scheduling. This
    reports presence, not a count: an authoritative maximum-consecutive count would
    require a whole-stream pass over every frame's type (far more than the single
    header field behind this flag), and the encoder's configured maximum (libavcodec
    ``max_b_frames``) is unusable here because it is "decoding: unused" -- ``0`` on a
    decoded stream.

    Attributes:
        has_reordering: the supplied frame-reordering flag (B-frames for H.264), or
            ``None`` if no input was recorded (measurement before update)

    """

    has_reordering: bool | None

    @property
    def is_defined(self) -> bool:
        """True once a flag has been recorded (``has_reordering`` is not ``None``)."""
        return self.has_reordering is not None


class FrameReorderingPresentMetric:
    """Instrument for :class:`FrameReorderingPresentMeasurement`.

    Takes the decoder frame-reordering flag as a plain boolean (for example
    ``VideoMetadata.has_bframes``) rather than a metadata object, keeping the metric
    decoupled from any sensor type. There is nothing to stream, so ``update`` is
    called once and a second ``update`` raises.
    """

    def __init__(self) -> None:
        """Initialize with no input consumed yet."""
        self._has_reordering: bool | None = None

    def update(self, *, has_reordering: bool) -> None:
        """Record the header frame-reordering flag (B-frame presence for H.264)."""
        # Guard the runtime input even though the type says bool: a caller bypassing the
        # type checker could pass the None sentinel, which would silently look like "no
        # update". Launder through ``object`` so the check is not statically dead code.
        flag: object = has_reordering
        if not isinstance(flag, (bool, np.bool_)):
            msg = f"has_reordering must be a bool, got {type(flag).__name__}"
            raise TypeError(msg)
        if self._has_reordering is not None:
            msg = "FrameReorderingPresentMetric consumes a single video's flag; call update() once"
            raise RuntimeError(msg)
        # Coerce numpy bools to the Python singleton so the stored fact is a real bool.
        self._has_reordering = bool(flag)

    def measurement(self) -> FrameReorderingPresentMeasurement:
        """Finalize the immutable B-frame measurement.

        If no input was consumed (``update`` never called), the measurement is
        undefined (``has_reordering is None``, ``is_defined is False``) rather than an
        error — matching the other metrics; callers must check ``is_defined`` before
        evaluating.
        """
        return FrameReorderingPresentMeasurement(has_reordering=self._has_reordering)


def _validate_bounds(start_ns: object, end_ns: object) -> None:
    """Reject recording bounds that are not int64 nanoseconds, before any state changes.

    The scalar counterpart of :func:`_validate_timestamps`, for the multi-sensor
    metrics: parameters are typed ``object`` so the runtime guards hold for a caller
    that bypasses the type checker, and ``bool`` is rejected explicitly because it is
    an ``int`` subclass, so a flag folded in as a timestamp would otherwise look
    perfectly plausible.

    Ordering is deliberately not required. A sensor whose last sample precedes its
    first is corrupt rather than unrepresentable, and the metrics say so through their
    own arithmetic -- no common window, an undefined ratio -- which is more useful to a
    session than an exception that takes the other sensors' measurement down with it.
    """
    for name, value in (("start_ns", start_ns), ("end_ns", end_ns)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            msg = f"{name} must be an integer number of nanoseconds, got {type(value).__name__}"
            raise TypeError(msg)
        if not _INT64_MIN <= int(value) <= _INT64_MAX:
            msg = f"{name} must be within int64 nanoseconds, got {value}"
            raise ValueError(msg)


class _SensorBounds:
    """Running extremes of a set of sensors' recording bounds.

    Shared by the two multi-sensor metrics, which differ only in what they derive from
    the same four numbers. Folds one sensor at a time, which is what lets a caller
    measuring a session feed each stream as it finishes rather than holding them all.
    """

    def __init__(self) -> None:
        """Initialize with no sensor folded in."""
        self._count = 0
        # Meaningless until the first fold, which is what ``extremes`` guards.
        self._min_start = 0
        self._max_start = 0
        self._min_stop = 0
        self._max_stop = 0

    @property
    def count(self) -> int:
        """How many sensors have been folded in."""
        return self._count

    def add(self, start_ns: int, end_ns: int) -> None:
        """Fold one sensor's bounds into the running extremes."""
        _validate_bounds(start_ns, end_ns)
        start, stop = int(start_ns), int(end_ns)
        if self._count == 0:
            self._min_start = self._max_start = start
            self._min_stop = self._max_stop = stop
        else:
            self._min_start = min(self._min_start, start)
            self._max_start = max(self._max_start, start)
            self._min_stop = min(self._min_stop, stop)
            self._max_stop = max(self._max_stop, stop)
        self._count += 1

    def extremes(self) -> tuple[int, int, int, int] | None:
        """``(min_start, max_start, min_stop, max_stop)``, or ``None`` if nothing was folded."""
        if self._count == 0:
            return None
        return self._min_start, self._max_start, self._min_stop, self._max_stop


@attrs.define(frozen=True)
class MultiSensorSpreadMeasurement:
    """How far apart a set of sensors began and stopped recording.

    A session-level alignment signal: from each sensor's recording bounds it takes the
    spread of the starts and the spread of the stops. Large spreads mean sensors that
    came up, or shut down, at very different times.

    Bounds only. Two sensors that started together can still disagree everywhere in
    between, which is what the per-stream gap and jitter metrics are for.

    Attributes:
        start_spread_ns: ``max(starts) - min(starts)``, in nanoseconds
        stop_spread_ns: ``max(stops) - min(stops)``, in nanoseconds
        num_sensors: how many sensors contributed bounds

    """

    start_spread_ns: int
    stop_spread_ns: int
    num_sensors: int

    #: One sensor is never out of step with itself, so there is nothing to spread.
    MIN_SENSORS: ClassVar[int] = 2

    @property
    def is_defined(self) -> bool:
        """True once at least two sensors contributed bounds to compare."""
        return self.num_sensors >= self.MIN_SENSORS

    @property
    def max_spread_ns(self) -> int:
        """The worse of the two spreads -- the one end of the rig furthest out of step."""
        return max(self.start_spread_ns, self.stop_spread_ns)


class MultiSensorSpreadMetric:
    """Instrument for :class:`MultiSensorSpreadMeasurement`.

    Takes bounds rather than timelines: ``update(start_ns=..., end_ns=...)`` once per
    sensor. A sensor with no timestamps has no bounds to compare and is simply not
    folded in, which is the caller's call to make -- an absent sensor is a different
    finding from a late one, and belongs to whoever knows which sensors were expected.
    """

    def __init__(self) -> None:
        """Initialize with no sensor folded in."""
        self._bounds = _SensorBounds()

    def update(self, *, start_ns: int, end_ns: int) -> None:
        """Fold one sensor's recording bounds into the running state."""
        self._bounds.add(start_ns, end_ns)

    def measurement(self) -> MultiSensorSpreadMeasurement:
        """Finalize the immutable spread measurement."""
        extremes = self._bounds.extremes()
        if extremes is None:
            return MultiSensorSpreadMeasurement(start_spread_ns=0, stop_spread_ns=0, num_sensors=0)
        min_start, max_start, min_stop, max_stop = extremes
        return MultiSensorSpreadMeasurement(
            start_spread_ns=max_start - min_start,
            stop_spread_ns=max_stop - min_stop,
            num_sensors=self._bounds.count,
        )


@attrs.define(frozen=True)
class MultiSensorOverlapMeasurement:
    """How much of a session every sensor was recording at once.

    The effective duration is ``min(stops) - max(starts)`` clamped at zero -- the
    window every sensor was up for -- and the total is ``max(stops) - min(starts)``,
    the span from the first sensor starting to the last one stopping. The fraction is
    their ratio.

    Bounding intervals only, never inside them: an overlap of 1.0 says every sensor was
    nominally recording throughout, not that any of them produced usable samples.

    Attributes:
        overlap_fraction: effective over total, or ``nan`` when undefined
        non_overlap_fraction: ``1 - overlap_fraction``, or ``nan`` when undefined.
            Carried rather than derived because it is the form policy judges: an
            overlap has to clear a floor, and every threshold in this tool is a ceiling.
        effective_duration_ns: the window every sensor was up for, clamped at zero
        total_duration_ns: first start to last stop
        num_sensors: how many sensors contributed bounds

    """

    overlap_fraction: float
    non_overlap_fraction: float
    effective_duration_ns: int
    total_duration_ns: int
    num_sensors: int

    #: A single sensor trivially overlaps itself; there is no agreement to measure.
    MIN_SENSORS: ClassVar[int] = 2

    @property
    def non_overlap_percent(self) -> float:
        """:attr:`non_overlap_fraction` as a percentage, the form policy is stated in."""
        return self.non_overlap_fraction * 100.0

    @property
    def is_defined(self) -> bool:
        """True once at least two sensors span a positive total duration.

        A non-positive total means there is nothing to divide by: either every sensor
        shares the same instant, or their bounds are inconsistent enough that the last
        stop precedes the first start. Both are undefined rather than zero -- a zero
        overlap would read as a finding about the rig, when the truth is that the
        question does not have an answer.
        """
        return self.num_sensors >= self.MIN_SENSORS and self.total_duration_ns > 0


class MultiSensorOverlapMetric:
    """Instrument for :class:`MultiSensorOverlapMeasurement`.

    Takes bounds rather than timelines: ``update(start_ns=..., end_ns=...)`` once per
    sensor, on the same terms as :class:`MultiSensorSpreadMetric`.
    """

    def __init__(self) -> None:
        """Initialize with no sensor folded in."""
        self._bounds = _SensorBounds()

    def update(self, *, start_ns: int, end_ns: int) -> None:
        """Fold one sensor's recording bounds into the running state."""
        self._bounds.add(start_ns, end_ns)

    def measurement(self) -> MultiSensorOverlapMeasurement:
        """Finalize the immutable overlap measurement."""
        extremes = self._bounds.extremes()
        count = self._bounds.count
        if extremes is None or count < MultiSensorOverlapMeasurement.MIN_SENSORS:
            # Durations of zero rather than of one sensor's own span: with nothing to
            # compare against, reporting its length would look like a measured overlap.
            return MultiSensorOverlapMeasurement(
                overlap_fraction=float("nan"),
                non_overlap_fraction=float("nan"),
                effective_duration_ns=0,
                total_duration_ns=0,
                num_sensors=count,
            )
        min_start, max_start, min_stop, max_stop = extremes
        effective = max(min_stop - max_start, 0)
        total = max_stop - min_start
        if total <= 0:
            # The durations stay on the row: they are what shows a reader whether this
            # is one shared instant or a set of bounds that contradict each other.
            return MultiSensorOverlapMeasurement(
                overlap_fraction=float("nan"),
                non_overlap_fraction=float("nan"),
                effective_duration_ns=effective,
                total_duration_ns=total,
                num_sensors=count,
            )
        overlap = effective / total
        return MultiSensorOverlapMeasurement(
            overlap_fraction=overlap,
            non_overlap_fraction=1.0 - overlap,
            effective_duration_ns=effective,
            total_duration_ns=total,
            num_sensors=count,
        )
