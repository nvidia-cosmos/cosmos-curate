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

"""Unit tests for the v1 data-integrity metrics."""

import math
import warnings

import numpy as np
import pytest

from cosmos_curator.core.sensors.data_integrity.metrics import (
    FrameReorderingPresentMetric,
    JitterMetric,
    MultiSensorOverlapMeasurement,
    MultiSensorOverlapMetric,
    MultiSensorSpreadMeasurement,
    MultiSensorSpreadMetric,
    RateMetric,
    TimestampGapMetric,
    TimestampOrderingMetric,
)

HZ_100_PERIOD_NS = 10_000_000  # one sample every 10 ms at 100 Hz

# The two session-grain metrics share a shape but no base class, so the cases that
# run against both are parametrized over this pair.
type SessionMetric = type[MultiSensorSpreadMetric | MultiSensorOverlapMetric]
type SessionMeasurement = MultiSensorSpreadMeasurement | MultiSensorOverlapMeasurement
type Bounds = list[tuple[int, int]]


def _ts(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)


# One-shot helpers: a single update() + measurement() over a whole array.


def _ordering(timestamps_ns: np.ndarray):  # noqa: ANN202
    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=timestamps_ns)
    return metric.measurement()


def _rate(timestamps_ns: np.ndarray, *, expected_hz: float):  # noqa: ANN202
    metric = RateMetric(expected_hz=expected_hz)
    metric.update(timestamps_ns=timestamps_ns)
    return metric.measurement()


def _gap(timestamps_ns: np.ndarray, *, expected_hz: float):  # noqa: ANN202
    metric = TimestampGapMetric(expected_hz=expected_hz)
    metric.update(timestamps_ns=timestamps_ns)
    return metric.measurement()


def _jitter(timestamps_ns: np.ndarray, *, expected_hz: float):  # noqa: ANN202
    metric = JitterMetric(expected_hz=expected_hz)
    metric.update(timestamps_ns=timestamps_ns)
    return metric.measurement()


# --- TimestampOrderingMetric --------------------------------------------------


def test_ordering_strictly_increasing() -> None:
    """Strictly increasing timestamps have no violations of either kind."""
    m = _ordering(_ts([0, 1, 2, 3]))
    assert m.decreasing_count == 0
    assert m.duplicate_count == 0
    assert m.first_decreasing_index is None
    assert m.first_duplicate_index is None
    assert m.strict_violation_count == 0
    assert m.num_samples == 4
    assert m.is_defined is True


def test_ordering_duplicates_allowed_but_counted() -> None:
    """Duplicates are counted separately and are not decreasing steps."""
    m = _ordering(_ts([0, 1, 1, 2, 2, 2]))
    assert m.decreasing_count == 0
    assert m.duplicate_count == 3
    assert m.first_duplicate_index == 2
    assert m.strict_violation_count == 3


def test_ordering_backward_step() -> None:
    """A backward step is counted with its first sample index."""
    m = _ordering(_ts([0, 5, 3, 8]))
    assert m.decreasing_count == 1
    assert m.duplicate_count == 0
    assert m.first_decreasing_index == 2
    assert m.strict_violation_count == 1


def test_ordering_backward_and_duplicate() -> None:
    """Backward and duplicate steps are tracked independently."""
    m = _ordering(_ts([0, 0, 5, 3]))
    assert m.duplicate_count == 1
    assert m.first_duplicate_index == 1
    assert m.decreasing_count == 1
    assert m.first_decreasing_index == 3
    assert m.strict_violation_count == 2


def test_ordering_too_few_samples() -> None:
    """A single sample has no steps to classify (undefined)."""
    m = _ordering(_ts([5]))
    assert m.strict_violation_count == 0
    assert m.num_samples == 1
    assert m.is_defined is False


def test_ordering_streaming_matches_one_shot() -> None:
    """Folding windows gives the same result as one measurement over the whole array."""
    full = _ts([0, 1, 1, 5, 3, 4])
    one_shot = _ordering(full)

    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=full[:2])  # [0, 1]
    metric.update(timestamps_ns=full[2:4])  # [1, 5] -> seam 1 -> 1 is a duplicate
    metric.update(timestamps_ns=full[4:])  # [3, 4] -> seam 5 -> 3 is decreasing
    streamed = metric.measurement()

    assert streamed == one_shot
    assert streamed.duplicate_count == 1
    assert streamed.decreasing_count == 1


def test_ordering_streaming_seam_index() -> None:
    """A violation at a window boundary is recorded at the correct global index."""
    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=_ts([0, 10]))  # samples 0, 1
    metric.update(timestamps_ns=_ts([5, 20]))  # sample 2 (=5) < 10 -> decreasing at index 2
    m = metric.measurement()
    assert m.decreasing_count == 1
    assert m.first_decreasing_index == 2
    assert m.num_samples == 4


# --- RateMetric ---------------------------------------------------------------


def test_rate_perfect_cadence() -> None:
    """Exact cadence yields zero deviation and the expected rate."""
    ts = _ts([0, HZ_100_PERIOD_NS, 2 * HZ_100_PERIOD_NS, 3 * HZ_100_PERIOD_NS])
    m = _rate(ts, expected_hz=100.0)
    assert m.period_deviation_percent == pytest.approx(0.0)
    assert m.actual_mean_hz == pytest.approx(100.0)
    assert m.num_intervals == 3
    assert m.num_filtered == 0
    assert m.is_defined is True


def test_rate_includes_large_gaps() -> None:
    """A dropout is not special-cased; its interval counts toward the deviation.

    Detecting the dropout itself is TimestampGapMetric's job, not RateMetric's.
    """
    # two 10 ms intervals, then one ~3 s interval, at 100 Hz (period 10 ms)
    ts = _ts([0, HZ_100_PERIOD_NS, 2 * HZ_100_PERIOD_NS, 2 * HZ_100_PERIOD_NS + 3_000_000_000])
    m = _rate(ts, expected_hz=100.0)
    assert m.num_intervals == 3
    assert m.num_filtered == 0
    # net error = sum(intervals) - period*n = 3.02e9 - 3e7 = 2.99e9; over period*n = 3e7
    assert m.period_deviation_percent == pytest.approx(2_990_000_000 / 30_000_000 * 100)


def test_rate_invalid_hz() -> None:
    """A non-positive expected rate is rejected at construction."""
    with pytest.raises(ValueError, match="expected_hz"):
        RateMetric(expected_hz=0.0)


def test_expected_period_rounds_half_up() -> None:
    """The discrete expected period rounds half up: 400 MHz -> 2.5 ns -> 3 ns (not 2, as ties-to-even would give)."""
    m = _rate(_ts([0, 3, 6]), expected_hz=400_000_000.0)
    assert m.expected_period_ns == 3


def test_rate_undefined_when_no_valid_intervals() -> None:
    """All-filtered intervals leave the rate undefined (not an error)."""
    m = _rate(_ts([100, 50]), expected_hz=100.0)
    assert m.is_defined is False
    assert m.num_intervals == 0


def test_rate_undefined_when_empty() -> None:
    """Empty input leaves the rate undefined, without raising."""
    m = _rate(_ts([]), expected_hz=100.0)
    assert m.is_defined is False
    assert m.num_samples == 0


def test_rate_streaming_matches_one_shot() -> None:
    """Folding windows (including the seam interval) matches one measurement."""
    full = _ts([0, 9_000_000, 21_000_000, 29_000_000, 40_000_000, 51_000_000])
    one_shot = _rate(full, expected_hz=100.0)

    metric = RateMetric(expected_hz=100.0)
    metric.update(timestamps_ns=full[:3])
    metric.update(timestamps_ns=full[3:])  # seam interval 29e6 - 21e6 must be counted
    streamed = metric.measurement()

    assert streamed.num_intervals == one_shot.num_intervals
    assert streamed.period_deviation_percent == pytest.approx(one_shot.period_deviation_percent)
    assert streamed.actual_mean_period_ns == pytest.approx(one_shot.actual_mean_period_ns)


def test_rate_streaming_filters_seam_backward_step() -> None:
    """A non-monotonic step landing on a window seam is filtered, not counted as an interval."""
    metric = RateMetric(expected_hz=100.0)
    metric.update(timestamps_ns=_ts([0, 10_000_000]))  # one forward interval
    metric.update(timestamps_ns=_ts([5_000_000, 15_000_000]))  # seam 5e6 - 10e6 < 0 -> filtered
    m = metric.measurement()
    assert m.num_filtered == 1  # the backward seam interval
    assert m.num_intervals == 2  # the two forward intervals


# --- TimestampGapMetric -------------------------------------------------------


def test_gap_none() -> None:
    """Regular cadence produces no gap events."""
    ts = _ts([i * HZ_100_PERIOD_NS for i in range(5)])
    m = _gap(ts, expected_hz=100.0)
    assert m.num_gaps == 0
    assert m.max_gap_ns == 0
    assert m.first_gap_index is None
    assert m.is_defined is True


def test_gap_detected() -> None:
    """A multi-period jump is counted, sized by max_gap_ns, and located by first_gap_index."""
    # jump from 2e7 to 5e7 (a 3e7 gap ~= two missing samples at 100 Hz)
    ts = _ts([0, 10_000_000, 20_000_000, 50_000_000, 60_000_000])
    m = _gap(ts, expected_hz=100.0)
    assert m.num_gaps == 1
    assert m.max_gap_ns == 30_000_000  # the raw gap interval (50ms - 20ms), not the excess
    # the gap is the 20ms -> 50ms interval; first_gap_index points at its later endpoint (sample 3)
    assert m.first_gap_index == 3


def test_gap_first_index_when_first_interval_is_a_gap() -> None:
    """A gap on the very first interval is located at sample 1 (the later endpoint), with no seam offset."""
    m = _gap(_ts([0, 50_000_000, 60_000_000]), expected_hz=100.0)
    assert m.num_gaps == 1
    assert m.first_gap_index == 1


def test_gap_first_index_stable_across_windows() -> None:
    """A gap in a later window is still counted but does not overwrite first_gap_index from an earlier one."""
    metric = TimestampGapMetric(expected_hz=100.0)
    metric.update(timestamps_ns=_ts([0, 50_000_000]))  # gap on interval 0 -> first_gap_index 1
    metric.update(timestamps_ns=_ts([100_000_000]))  # 50 -> 100 is a second (seam) gap, first already set
    m = metric.measurement()
    assert m.num_gaps == 2
    assert m.first_gap_index == 1


def test_gap_single_sample_is_undefined() -> None:
    """A single sample has no interval to check (undefined)."""
    m = _gap(_ts([5]), expected_hz=100.0)
    assert m.num_gaps == 0
    assert m.first_gap_index is None
    assert m.is_defined is False


def test_gap_two_samples_is_defined() -> None:
    """Two samples form one interval, enough for a defined result."""
    m = _gap(_ts([0, HZ_100_PERIOD_NS]), expected_hz=100.0)
    assert m.num_gaps == 0
    assert m.is_defined is True


def test_gap_empty_is_undefined() -> None:
    """Empty input leaves the gap metric undefined, without raising."""
    m = _gap(_ts([]), expected_hz=100.0)
    assert m.is_defined is False
    assert m.num_samples == 0


def test_gap_invalid_hz() -> None:
    """A non-positive expected rate is rejected at construction (parity with rate/jitter)."""
    with pytest.raises(ValueError, match="expected_hz"):
        TimestampGapMetric(expected_hz=0.0)


def test_gap_streaming_matches_one_shot() -> None:
    """Folding windows, with the gap landing on the seam, matches one measurement."""
    full = _ts([0, 10_000_000, 20_000_000, 50_000_000, 60_000_000])
    one_shot = _gap(full, expected_hz=100.0)

    metric = TimestampGapMetric(expected_hz=100.0)
    metric.update(timestamps_ns=full[:3])  # [0, 10, 20]
    metric.update(timestamps_ns=full[3:])  # [50, 60] -> the 20->50 gap is on the seam
    streamed = metric.measurement()

    assert streamed.num_gaps == one_shot.num_gaps == 1
    assert streamed.max_gap_ns == one_shot.max_gap_ns == 30_000_000
    # the gap lands on the window seam; its later endpoint is global sample 3 either way
    assert streamed.first_gap_index == one_shot.first_gap_index == 3


# --- JitterMetric -------------------------------------------------------------


def test_jitter_perfect_cadence() -> None:
    """Uniform intervals have zero jitter."""
    ts = _ts([0, HZ_100_PERIOD_NS, 2 * HZ_100_PERIOD_NS, 3 * HZ_100_PERIOD_NS])
    m = _jitter(ts, expected_hz=100.0)
    assert m.jitter_percent == pytest.approx(0.0)
    assert m.num_intervals == 3
    assert m.is_defined is True


def test_jitter_variation() -> None:
    """Interval spread is reported as a percent of the expected period."""
    # intervals 10 ms, 12 ms, 8 ms -> sample std 2 ms = 20% of the 10 ms period
    ts = _ts([0, 10_000_000, 22_000_000, 30_000_000])
    m = _jitter(ts, expected_hz=100.0)
    assert m.jitter_percent == pytest.approx(20.0)


def test_jitter_too_few_intervals() -> None:
    """A single interval cannot express variation (undefined -> nan, not 0.0)."""
    m = _jitter(_ts([0, HZ_100_PERIOD_NS]), expected_hz=100.0)
    assert np.isnan(m.jitter_percent)
    assert m.is_defined is False


def test_jitter_invalid_hz() -> None:
    """A non-positive expected rate is rejected at construction."""
    with pytest.raises(ValueError, match="expected_hz"):
        JitterMetric(expected_hz=-1.0)


def test_jitter_streaming_matches_one_shot() -> None:
    """Folding windows (including the seam interval) matches one measurement."""
    full = _ts([0, 10_000_000, 22_000_000, 30_000_000, 41_000_000, 49_000_000])
    one_shot = _jitter(full, expected_hz=100.0)

    metric = JitterMetric(expected_hz=100.0)
    metric.update(timestamps_ns=full[:3])
    metric.update(timestamps_ns=full[3:])  # seam interval 30e6 - 22e6 must be included
    streamed = metric.measurement()

    assert streamed.num_intervals == one_shot.num_intervals
    assert streamed.jitter_percent == pytest.approx(one_shot.jitter_percent)


def test_jitter_reports_filtered_count() -> None:
    """Jitter drops non-monotonic steps and reports how many (parity with RateMetric)."""
    # diffs 10ms, -5ms, 10ms -> the backward step is filtered
    m = _jitter(_ts([0, 10_000_000, 5_000_000, 15_000_000]), expected_hz=100.0)
    assert m.num_filtered == 1
    assert m.num_intervals == 2


# --- FrameReorderingPresentMetric -----------------------------------------------------


def test_bframes_present() -> None:
    """The metric reports that the header signals B-frames (reordering)."""
    metric = FrameReorderingPresentMetric()
    metric.update(has_reordering=True)
    m = metric.measurement()
    assert m.has_reordering is True
    assert m.is_defined is True


def test_bframes_absent() -> None:
    """A stream with no reordering signal reports has_reordering False."""
    metric = FrameReorderingPresentMetric()
    metric.update(has_reordering=False)
    assert metric.measurement().has_reordering is False


def test_bframes_undefined_without_update() -> None:
    """measurement() before update() is undefined (not an error), like the other metrics."""
    m = FrameReorderingPresentMetric().measurement()
    assert m.has_reordering is None
    assert m.is_defined is False


def test_bframes_second_update_rejected() -> None:
    """A second update() is rejected — the metric consumes a single video's flag."""
    metric = FrameReorderingPresentMetric()
    metric.update(has_reordering=True)
    with pytest.raises(RuntimeError, match="single video"):
        metric.update(has_reordering=False)


# --- overflow / bad-input hardening -------------------------------------------

INT64_MIN = int(np.iinfo(np.int64).min)
INT64_MAX = int(np.iinfo(np.int64).max)


def test_ordering_handles_extreme_backward_timestamps() -> None:
    """A backward step spanning the full int64 range is classified, not rejected."""
    m = _ordering(_ts([INT64_MAX, INT64_MIN]))  # a huge backward step
    assert m.decreasing_count == 1
    assert m.first_decreasing_index == 1


def test_ordering_accepts_full_int64_forward_span() -> None:
    """The opposite extreme is a valid forward step, not an overflow error."""
    m = _ordering(_ts([INT64_MIN, INT64_MAX]))
    assert m.decreasing_count == 0
    assert m.duplicate_count == 0


@pytest.mark.parametrize("make", [_rate, _gap, _jitter])
@pytest.mark.parametrize("pair", [[INT64_MIN, INT64_MAX], [INT64_MAX, INT64_MIN]])
def test_interval_overflow_is_rejected(make, pair) -> None:  # noqa: ANN001
    """An interval too large to represent in int64 is rejected in either direction, not wrapped."""
    with pytest.raises(ValueError, match="int64 range"):
        make(_ts(pair), expected_hz=100.0)


@pytest.mark.parametrize("metric_cls", [RateMetric, TimestampGapMetric, JitterMetric])
def test_rate_too_high_is_rejected(metric_cls) -> None:  # noqa: ANN001
    """A rate whose discrete period rounds below 1 ns is rejected at construction."""
    with pytest.raises(ValueError, match="expected_hz too high"):
        metric_cls(expected_hz=5e9)


def test_gap_boundary_around_1_5_periods() -> None:
    """The gap cutoff is round(interval / period) >= 2: a 1.4x interval is not a gap, 1.5x is."""
    # period is 10 ms at 100 Hz
    not_a_gap = _gap(_ts([0, 10_000_000, 24_000_000]), expected_hz=100.0)  # 14 ms -> round(1.4) = 1
    assert not_a_gap.num_gaps == 0
    a_gap = _gap(_ts([0, 10_000_000, 25_000_000]), expected_hz=100.0)  # 15 ms -> round(1.5) = 2
    assert a_gap.num_gaps == 1


def test_rate_seam_interval_overflow_is_rejected() -> None:
    """An overflowing interval split across a window seam is rejected, like the one-shot case.

    The one-shot [INT64_MIN, INT64_MAX] already raises; the same pair delivered as two
    windows must not slip through by computing the seam outside the overflow check.
    """
    metric = RateMetric(expected_hz=100.0)
    metric.update(timestamps_ns=_ts([INT64_MIN]))
    with pytest.raises(ValueError, match="int64 range"):
        metric.update(timestamps_ns=_ts([INT64_MAX]))


def test_rate_handles_total_duration_exceeding_int64() -> None:
    """A total interval duration larger than int64 still yields a correct, positive mean period.

    Each interval is representable, but two together exceed int64; the reported mean period and
    rate stay positive and correct rather than wrapping negative.
    """
    big = 9_000_000_000_000_000_000  # 9e18 < INT64_MAX; two together exceed int64
    ts = _ts([0, big, 0, big])  # two forward intervals of `big` (the drop back to 0 is filtered)
    m = _rate(ts, expected_hz=100.0)
    assert m.num_intervals == 2
    assert m.actual_mean_period_ns == pytest.approx(float(big))
    assert m.actual_mean_hz > 0.0


def _construct(metric_cls):  # noqa: ANN001, ANN202
    """Construct a timestamp metric, supplying a default rate where the ctor requires one."""
    if metric_cls is TimestampOrderingMetric:
        return metric_cls()
    return metric_cls(expected_hz=100.0)


TIMESTAMP_METRICS = [TimestampOrderingMetric, RateMetric, TimestampGapMetric, JitterMetric]


@pytest.mark.parametrize("metric_cls", TIMESTAMP_METRICS)
def test_timestamp_update_rejects_non_1d(metric_cls) -> None:  # noqa: ANN001
    """A 2-D timestamp array is rejected up front, before any running state is mutated."""
    metric = _construct(metric_cls)
    with pytest.raises(ValueError, match="1-D"):
        metric.update(timestamps_ns=np.array([[0, 1], [2, 3]], dtype=np.int64))
    # state untouched: a subsequent valid one-shot behaves as if the bad call never happened
    metric.update(timestamps_ns=_ts([0, 10_000_000, 20_000_000]))
    assert metric.measurement().num_samples == 3


@pytest.mark.parametrize("metric_cls", TIMESTAMP_METRICS)
def test_timestamp_update_rejects_wrong_dtype(metric_cls) -> None:  # noqa: ANN001
    """A non-int64 timestamp array is rejected."""
    metric = _construct(metric_cls)
    with pytest.raises(ValueError, match="int64"):
        metric.update(timestamps_ns=np.array([0.0, 1.0, 2.0], dtype=np.float64))


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_bframes_rejects_non_bool(bad) -> None:  # noqa: ANN001
    """A non-boolean flag (notably the None sentinel) is rejected, not silently stored."""
    metric = FrameReorderingPresentMetric()
    with pytest.raises(TypeError, match="bool"):
        metric.update(has_reordering=bad)
    # state untouched: still undefined, and still accepts a real update afterwards
    assert metric.measurement().is_defined is False
    metric.update(has_reordering=True)
    assert metric.measurement().has_reordering is True


def test_bframes_accepts_numpy_bool() -> None:
    """A numpy bool is coerced to a Python bool, so has_reordering is the real singleton."""
    metric = FrameReorderingPresentMetric()
    metric.update(has_reordering=np.True_)
    assert metric.measurement().has_reordering is True


def test_gap_applies_exact_boundary_at_large_periods() -> None:
    """The 1.5-period gap boundary stays exact even when period and interval exceed float64's 2^53.

    expected_hz 1e-7 gives a period of 1e16 ns (> 2^53), so 1.5 periods is 1.5e16. An interval one
    ns below the boundary is not a gap; exactly 1.5 periods is.
    """
    below_1_5_periods = 15_000_000_000_000_000 - 1  # 1.5 periods minus 1 ns
    m = _gap(_ts([0, below_1_5_periods]), expected_hz=1e-7)
    assert m.num_gaps == 0
    at_1_5_periods = 15_000_000_000_000_000  # exactly 1.5 periods -> a gap
    assert _gap(_ts([0, at_1_5_periods]), expected_hz=1e-7).num_gaps == 1


def test_gap_handles_maximum_representable_interval_without_warnings() -> None:
    """The maximum representable interval classifies cleanly, with no numpy warnings raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = _gap(_ts([0, INT64_MAX]), expected_hz=1e9)  # period 1 ns
    assert m.num_gaps == 1


@pytest.mark.parametrize("metric_cls", [RateMetric, TimestampGapMetric, JitterMetric])
@pytest.mark.parametrize("bad_hz", [1e-320, 1e-11])
def test_rate_too_low_is_rejected(metric_cls, bad_hz) -> None:  # noqa: ANN001
    """A rate so low its period is not a representable int64 (finite, <= INT64_MAX ns) is rejected.

    Before: Rate/Gap raised an incidental OverflowError and Jitter accepted an infinite period
    (reporting a misleading zero jitter). Now all three reject it at construction.
    """
    with pytest.raises(ValueError, match="expected_hz too low"):
        metric_cls(expected_hz=bad_hz)


# --- additional behavioral coverage -------------------------------------------


def test_jitter_all_intervals_filtered_is_undefined() -> None:
    """When every interval is non-positive (here, duplicates), jitter is undefined, not zero."""
    m = _jitter(_ts([5, 5, 5]), expected_hz=100.0)  # two zero-diff (duplicate) intervals
    assert m.num_intervals == 0
    assert m.num_filtered == 2
    assert m.is_defined is False
    assert np.isnan(m.jitter_percent)


def test_gap_threshold_above_max_interval_yields_no_gap() -> None:
    """At a rate so low that 1.5 x period exceeds int64, no representable interval is a gap."""
    # period ~ 8e18 ns (valid, <= INT64_MAX); 1.5 * period > INT64_MAX, so even INT64_MAX
    # cannot reach the gap boundary.
    m = _gap(_ts([0, INT64_MAX]), expected_hz=1.25e-10)
    assert m.num_gaps == 0


def test_ordering_streaming_keeps_earliest_violation_index() -> None:
    """A later seam violation does not overwrite the earliest recorded violation index."""
    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=_ts([0, 5, 3]))  # decreasing at index 2
    metric.update(timestamps_ns=_ts([1, 2]))  # seam 3 -> 1 decreasing at index 3
    m = metric.measurement()
    assert m.decreasing_count == 2
    assert m.first_decreasing_index == 2  # earliest, not the later seam index 3


def test_ordering_streaming_keeps_earliest_duplicate_index() -> None:
    """A later duplicate on a window seam does not overwrite the earliest duplicate index."""
    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=_ts([0, 5, 5]))  # duplicate at index 2
    metric.update(timestamps_ns=_ts([5, 6]))  # seam 5 -> 5 duplicate at index 3
    m = metric.measurement()
    assert m.duplicate_count == 2
    assert m.first_duplicate_index == 2  # earliest, not the later seam index 3


def test_ordering_streaming_clean_seam_has_no_violation() -> None:
    """An increasing seam between windows produces no violation."""
    metric = TimestampOrderingMetric()
    metric.update(timestamps_ns=_ts([0, 1]))
    metric.update(timestamps_ns=_ts([2, 3]))  # seam 1 -> 2 is increasing
    m = metric.measurement()
    assert m.decreasing_count == 0
    assert m.duplicate_count == 0
    assert m.num_samples == 4


def test_ordering_empty_is_undefined() -> None:
    """Empty input is undefined (parity with rate/gap)."""
    m = _ordering(_ts([]))
    assert m.num_samples == 0
    assert m.is_defined is False


def test_jitter_empty_is_undefined() -> None:
    """Empty input leaves jitter undefined, without raising."""
    m = _jitter(_ts([]), expected_hz=100.0)
    assert m.is_defined is False
    assert m.num_intervals == 0


@pytest.mark.parametrize("metric_cls", [TimestampOrderingMetric, RateMetric, JitterMetric])
def test_empty_window_mid_stream_is_noop(metric_cls) -> None:  # noqa: ANN001
    """An empty window between two non-empty windows does not change the measurement."""
    w1 = _ts([0, 10_000_000, 20_000_000])
    w2 = _ts([30_000_000, 41_000_000, 49_000_000])

    with_empty = _construct(metric_cls)
    with_empty.update(timestamps_ns=w1)
    with_empty.update(timestamps_ns=_ts([]))
    with_empty.update(timestamps_ns=w2)

    without = _construct(metric_cls)
    without.update(timestamps_ns=w1)
    without.update(timestamps_ns=w2)

    assert with_empty.measurement() == without.measurement()


def test_gap_empty_window_mid_stream_is_noop() -> None:
    """An empty window between two non-empty windows does not change the gap measurement."""
    w1 = _ts([0, 10_000_000, 20_000_000])
    w2 = _ts([50_000_000, 60_000_000])  # the 20 -> 50 gap lands on the seam

    with_empty = TimestampGapMetric(expected_hz=100.0)
    with_empty.update(timestamps_ns=w1)
    with_empty.update(timestamps_ns=_ts([]))
    with_empty.update(timestamps_ns=w2)
    a = with_empty.measurement()

    without = TimestampGapMetric(expected_hz=100.0)
    without.update(timestamps_ns=w1)
    without.update(timestamps_ns=w2)
    b = without.measurement()

    assert a.num_gaps == b.num_gaps == 1
    assert a.max_gap_ns == b.max_gap_ns
    assert a.first_gap_index == b.first_gap_index == 3


def test_gap_multiple_events_summarized_by_count_max_and_first() -> None:
    """Multiple gaps: num_gaps counts them, max_gap_ns is the largest, first_gap_index is the first."""
    # 100 Hz (period 10 ms): a 30 ms gap (20 -> 50) then a larger 40 ms gap (60 -> 100)
    ts = _ts([0, 10_000_000, 20_000_000, 50_000_000, 60_000_000, 100_000_000])
    m = _gap(ts, expected_hz=100.0)
    assert m.num_gaps == 2
    assert m.max_gap_ns == 40_000_000  # the largest gap, not the first
    assert m.first_gap_index == 3  # the first gap (20 -> 50), even though a later gap is larger


def test_timestamp_update_rejects_non_ndarray() -> None:
    """A non-array input (e.g. a Python list) is rejected before any state changes."""
    metric = TimestampOrderingMetric()
    with pytest.raises(TypeError, match="ndarray"):
        metric.update(timestamps_ns=[0, 1, 2])  # type: ignore[arg-type]


@pytest.mark.parametrize("metric_cls", [RateMetric, TimestampGapMetric, JitterMetric])
@pytest.mark.parametrize("bad_hz", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_hz_rejected(metric_cls, bad_hz) -> None:  # noqa: ANN001
    """NaN and +/- infinity rates are rejected at construction."""
    with pytest.raises(ValueError, match="positive and finite"):
        metric_cls(expected_hz=bad_hz)


def test_rate_filters_nonpositive_intervals() -> None:
    """Rate drops a duplicate (zero) interval as non-positive, keeping only forward intervals."""
    m = _rate(_ts([0, 0, 10_000_000]), expected_hz=100.0)  # first interval 0 (duplicate) -> filtered
    assert m.num_filtered == 1
    assert m.num_intervals == 1


def test_jitter_filters_nonpositive_intervals() -> None:
    """Jitter drops a duplicate (zero) interval as non-positive."""
    m = _jitter(_ts([0, 0, 10_000_000, 20_000_000]), expected_hz=100.0)
    assert m.num_filtered == 1
    assert m.num_intervals == 2


def test_rate_faster_than_expected() -> None:
    """200 Hz observed against 100 Hz expected reads as 50% period deviation and 200 Hz actual."""
    ts = _ts([0, 5_000_000, 10_000_000, 15_000_000])  # 5 ms intervals -> 200 Hz
    m = _rate(ts, expected_hz=100.0)
    assert m.period_deviation_percent == pytest.approx(50.0)
    assert m.actual_mean_hz == pytest.approx(200.0)


def test_rate_signed_errors_cancel() -> None:
    """Fast and slow intervals cancel: 5 ms and 15 ms against a 10 ms period net zero deviation."""
    # a mean-absolute-error metric would report ~50% here, so this pins the signed-sum semantics
    ts = _ts([0, 5_000_000, 20_000_000, 25_000_000, 40_000_000])  # intervals 5, 15, 5, 15 ms
    m = _rate(ts, expected_hz=100.0)
    assert m.num_intervals == 4
    assert m.period_deviation_percent == pytest.approx(0.0)


def test_gap_ignores_backward_and_duplicate_steps() -> None:
    """A backward step and a duplicate are never gaps, even when the backward jump exceeds the threshold."""
    # 100 Hz (period 10 ms, gap threshold 15 ms): a duplicate (20 -> 20) then a 20 ms backward step
    # (20 -> 0). An implementation keying on the interval magnitude would wrongly flag the backward jump.
    m = _gap(_ts([20_000_000, 20_000_000, 0, 5_000_000]), expected_hz=100.0)
    assert m.num_gaps == 0


def _ordering_measurement():  # noqa: ANN202
    return _ordering(_ts([0, 1, 2]))


def _rate_measurement():  # noqa: ANN202
    return _rate(_ts([0, 10_000_000, 20_000_000]), expected_hz=100.0)


def _gap_measurement():  # noqa: ANN202
    return _gap(_ts([0, 10_000_000, 20_000_000]), expected_hz=100.0)


def _jitter_measurement():  # noqa: ANN202
    return _jitter(_ts([0, 10_000_000, 20_000_000]), expected_hz=100.0)


# --- MultiSensorSpreadMetric / MultiSensorOverlapMetric -----------------------


def _spread(bounds: Bounds) -> MultiSensorSpreadMeasurement:
    metric = MultiSensorSpreadMetric()
    for start_ns, end_ns in bounds:
        metric.update(start_ns=start_ns, end_ns=end_ns)
    return metric.measurement()


def _overlap(bounds: Bounds) -> MultiSensorOverlapMeasurement:
    metric = MultiSensorOverlapMetric()
    for start_ns, end_ns in bounds:
        metric.update(start_ns=start_ns, end_ns=end_ns)
    return metric.measurement()


def test_spread_measures_each_end_of_the_rig_separately() -> None:
    """Coming up together and shutting down together are different facts about a rig."""
    m = _spread([(0, 1_000), (100, 1_000), (250, 1_400)])
    assert m.start_spread_ns == 250
    assert m.stop_spread_ns == 400
    assert m.num_sensors == 3
    assert m.max_spread_ns == 400
    assert m.is_defined is True


def test_a_rig_recording_in_lockstep_spreads_by_nothing() -> None:
    """Identical bounds are the zero this metric is measured against."""
    m = _spread([(500, 1_500), (500, 1_500)])
    assert (m.start_spread_ns, m.stop_spread_ns, m.max_spread_ns) == (0, 0, 0)
    assert m.is_defined is True


@pytest.mark.parametrize("bounds", [[], [(0, 1_000)]])
def test_one_sensor_is_never_out_of_step_with_itself(bounds: Bounds) -> None:
    """Fewer than two sensors leaves nothing to compare, so the measurement is undefined."""
    m = _spread(bounds)
    assert m.is_defined is False
    assert m.num_sensors == len(bounds)


def test_overlap_is_the_window_every_sensor_was_up_for() -> None:
    """Effective over total: one sensor late and another early cost the session both ends."""
    m = _overlap([(0, 1_000), (250, 1_250)])
    assert m.effective_duration_ns == 750  # 1000 - 250
    assert m.total_duration_ns == 1_250  # 1250 - 0
    assert m.overlap_fraction == pytest.approx(0.6)
    assert m.non_overlap_fraction == pytest.approx(0.4)
    assert m.non_overlap_percent == pytest.approx(40.0)
    assert m.num_sensors == 2
    assert m.is_defined is True


def test_sensors_that_never_ran_together_overlap_by_zero() -> None:
    """A latest start after the earliest stop is a real finding, not an undefined one."""
    m = _overlap([(0, 1_000), (2_000, 3_000)])
    assert m.effective_duration_ns == 0
    assert m.total_duration_ns == 3_000
    assert m.overlap_fraction == 0.0
    assert m.non_overlap_fraction == 1.0
    assert m.is_defined is True


def test_a_session_of_one_shared_instant_has_no_ratio_to_report() -> None:
    """A zero total span is undefined rather than a division by zero."""
    m = _overlap([(500, 500), (500, 500)])
    assert m.total_duration_ns == 0
    assert math.isnan(m.overlap_fraction)
    assert math.isnan(m.non_overlap_fraction)
    assert m.is_defined is False


def test_bounds_that_contradict_themselves_are_undefined_rather_than_negative() -> None:
    """A stream whose last sample precedes its first would otherwise divide by a negative span.

    Corrupt rather than unrepresentable, so the metric reports it instead of raising:
    the per-stream ordering metric is what names the stream responsible.
    """
    m = _overlap([(1_000, 900), (1_100, 950)])
    assert m.total_duration_ns == -50
    assert math.isnan(m.overlap_fraction)
    assert m.is_defined is False


@pytest.mark.parametrize("bounds", [[], [(0, 1_000)]])
def test_a_lone_sensor_overlaps_nothing_and_reports_no_duration(bounds: Bounds) -> None:
    """Its own span must not read back as a measured overlap."""
    m = _overlap(bounds)
    assert m.is_defined is False
    assert (m.effective_duration_ns, m.total_duration_ns) == (0, 0)
    assert m.num_sensors == len(bounds)


@pytest.mark.parametrize("metric_cls", [MultiSensorSpreadMetric, MultiSensorOverlapMetric])
def test_the_order_sensors_are_folded_in_does_not_matter(metric_cls: SessionMetric) -> None:
    """These are symmetric in their inputs, which is what lets a session feed them as streams finish."""
    bounds: Bounds = [(300, 1_400), (0, 1_000), (150, 1_250)]

    def measure(items: Bounds) -> SessionMeasurement:
        metric = metric_cls()
        for start_ns, end_ns in items:
            metric.update(start_ns=start_ns, end_ns=end_ns)
        return metric.measurement()

    assert measure(bounds) == measure(list(reversed(bounds)))


@pytest.mark.parametrize("metric_cls", [MultiSensorSpreadMetric, MultiSensorOverlapMetric])
@pytest.mark.parametrize("bad", [1.5, "0", None, True, np.True_])
def test_bounds_must_be_integer_nanoseconds(metric_cls: SessionMetric, bad: object) -> None:
    """A bool is rejected too: an int subclass folded in as a timestamp would look plausible."""
    with pytest.raises(TypeError):
        metric_cls().update(start_ns=bad, end_ns=1_000)


@pytest.mark.parametrize("metric_cls", [MultiSensorSpreadMetric, MultiSensorOverlapMetric])
def test_bounds_outside_int64_are_rejected(metric_cls: SessionMetric) -> None:
    """The store's column is int64; a bound that cannot land in one is caught at the boundary."""
    with pytest.raises(ValueError, match="int64"):
        metric_cls().update(start_ns=0, end_ns=2**63)


@pytest.mark.parametrize("metric_cls", [MultiSensorSpreadMetric, MultiSensorOverlapMetric])
def test_numpy_integer_bounds_are_accepted(metric_cls: SessionMetric) -> None:
    """Bounds arrive from numpy timelines, so the int64 scalars they yield must fold in as they are."""
    metric = metric_cls()
    metric.update(start_ns=np.int64(0), end_ns=np.int64(1_000))
    metric.update(start_ns=np.int64(100), end_ns=np.int64(1_100))
    assert metric.measurement().num_sensors == 2


def _bframes_measurement():  # noqa: ANN202
    metric = FrameReorderingPresentMetric()
    metric.update(has_reordering=True)
    return metric.measurement()


def _spread_measurement() -> MultiSensorSpreadMeasurement:
    return _spread([(0, 1_000), (100, 1_100)])


def _overlap_measurement() -> MultiSensorOverlapMeasurement:
    return _overlap([(0, 1_000), (100, 1_100)])


@pytest.mark.parametrize(
    ("make_measurement", "field"),
    [
        (_ordering_measurement, "num_samples"),
        (_rate_measurement, "period_deviation_percent"),
        (_gap_measurement, "max_gap_ns"),
        (_jitter_measurement, "jitter_percent"),
        (_bframes_measurement, "has_reordering"),
        (_spread_measurement, "start_spread_ns"),
        (_overlap_measurement, "overlap_fraction"),
    ],
)
def test_measurement_is_frozen(make_measurement, field) -> None:  # noqa: ANN001
    """Every concrete measurement is frozen: a field cannot be reassigned after construction."""
    m = make_measurement()
    with pytest.raises(AttributeError):
        setattr(m, field, 0)
