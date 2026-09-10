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
"""Unit tests for make_ts_grid and SamplingGrid."""

from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.sampling.grid import (
    SamplingGrid,
    SamplingWindow,
    make_ts_grid,
)
from cosmos_curator.core.sensors.utils.validation import INT64_MAX
from tests.cosmos_curator.core.sensors.test_utils import (
    EPOCH_ODD_NS,
    EPOCH_ROUND_NS,
    ZERO_ORIGIN_NS,
    make_sampling_grid,
)


def exact_ts_grid(start_ns: int, inclusive_end_ns: int, sample_rate_hz: float) -> tuple[int, int, np.ndarray]:
    """The grid ``make_ts_grid`` should return, in exact arithmetic: the interval is held as a
    rational and each offset rounded to an integer before being added to ``start_ns``, so no
    absolute timestamp passes through ``float64``.
    """  # noqa: D205, D401
    # This sample count deliberately mirrors grid.py:140-141, but floors the span exactly where
    # the implementation floors nextafter(span / interval, inf) in float64. The nudge absorbs
    # float undershoot at an exact boundary; for a span landing within ~1e-13 *below* an integer
    # it overshoots instead and the two counts differ by one -- 29.97 Hz over exactly 100 s is
    # such a case, and there make_ts_grid is correct and this helper is not. Any new zero-base
    # rate or span must be added to _ZERO_BASE_GRID_CASES below, which checks for that.
    n = max(2, int(Fraction(inclusive_end_ns - start_ns) * Fraction(sample_rate_hz) // 1_000_000_000) + 2)
    interval_ns = Fraction(1_000_000_000) / Fraction(sample_rate_hz)
    full = np.array([start_ns + round(interval_ns * k) for k in range(n)], dtype=np.int64)
    return int(full[0]), int(full[-1]), full[:-1]


def _iter_window_arrays(
    ts: npt.NDArray[np.int64],
    *,
    stride_ns: int,
    duration_ns: int,
) -> list[SamplingWindow]:
    p = make_sampling_grid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    return list(p)


def test_sampling_grid_iter() -> None:
    """SamplingGrid should yield raw window slices including the right boundary marker."""
    ts = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    grid = make_sampling_grid(ts, 20, 20)
    got = list(grid)
    want = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=20,
            timestamps_ns=np.array([0, 10], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=20,
            exclusive_end_ns=40,
            timestamps_ns=np.array([20, 30], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=40,
            exclusive_end_ns=50,
            timestamps_ns=np.array([40], dtype=np.int64),
        ),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_sampling_grid_half_open_window_contract() -> None:
    """A yielded window should include the exclusive right boundary marker as its final element."""
    ts = np.array([100, 200, 300, 400], dtype=np.int64)
    grid = make_sampling_grid(ts, 200, 200)
    windows = list(grid)

    np.testing.assert_array_equal(windows[0].timestamps_ns, np.array([100, 200], dtype=np.int64))
    assert windows[0].exclusive_end_ns == 300
    np.testing.assert_array_equal(windows[1].timestamps_ns, np.array([300], dtype=np.int64))
    assert windows[1].exclusive_end_ns == 400


def test_sampling_grid_adjacent_windows_share_boundary_marker() -> None:
    """Adjacent windows should share the boundary timestamp that separates their half-open intervals."""
    ts = np.array([0, 100, 200, 300], dtype=np.int64)
    grid = make_sampling_grid(ts, 100, 100)
    windows = list(grid)

    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=100,
            timestamps_ns=np.array([0], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=100,
            exclusive_end_ns=200,
            timestamps_ns=np.array([100], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=200,
            exclusive_end_ns=300,
            timestamps_ns=np.array([200], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for window, expected_window in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(window.timestamps_ns, expected_window.timestamps_ns)
        assert window.start_ns == expected_window.start_ns
        assert window.exclusive_end_ns == expected_window.exclusive_end_ns


# Every zero-origin (start_ns, inclusive_end_ns, sample_rate_hz) the make_ts_grid tests below
# exercise. Zero-origin cells are required to pass, so exact_ts_grid must agree with the
# implementation on all of them. Add new zero-base cases here when adding them above.
_ZERO_BASE_GRID_CASES = [
    (ZERO_ORIGIN_NS, ZERO_ORIGIN_NS + 5_000_000_000, 30.0),
    (0, 1_000_000_000, 30.0),
    (123, 987_654_321, 29.97),
    (0, 5_000_000_000, 59.94),
    (42, 42 + 1_000_000, 1_000.0),
    (42, 42, 30.0),
    (42, 42 + 1_000_000_000, 30.0),
]


@pytest.mark.parametrize(("start_ns", "inclusive_end_ns", "sample_rate_hz"), _ZERO_BASE_GRID_CASES)
def test_exact_ts_grid_agrees_with_implementation_at_zero_base(
    start_ns: int,
    inclusive_end_ns: int,
    sample_rate_hz: float,
) -> None:
    """exact_ts_grid must produce the same grid as make_ts_grid at a zero-scale origin.

    The two derive their sample count differently (exact floor vs. floored ``nextafter``), and
    for a span landing just below an integer they disagree by one -- with make_ts_grid on the
    correct side. Catch that here, where the cause is named, rather than as an opaque
    grid-comparison failure in a test that is nominally about epoch-scale precision.
    """
    expected = exact_ts_grid(start_ns, inclusive_end_ns, sample_rate_hz)
    got = make_ts_grid(start_ns, inclusive_end_ns, sample_rate_hz)

    assert got[0] == expected[0]
    assert got[1] == expected[1]
    np.testing.assert_array_equal(got[2], expected[2])


@pytest.mark.parametrize(
    ("start_ns", "end_ns"),
    [
        pytest.param(ZERO_ORIGIN_NS, ZERO_ORIGIN_NS + 5_000_000_000, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 5_000_000_000, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 5_000_000_000, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid(start_ns: int, end_ns: int) -> None:
    """make_ts_grid should reproduce the exact-arithmetic grid at any time origin."""
    sample_rate_hz = 30.0
    expected_start_ns, expected_exclusive_end_ns, expected_timestamps_ns = exact_ts_grid(
        start_ns,
        end_ns,
        sample_rate_hz,
    )

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    assert got_start_ns == expected_start_ns
    assert got_exclusive_end_ns == expected_exclusive_end_ns
    np.testing.assert_array_equal(got_timestamps_ns, expected_timestamps_ns)


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        pytest.param(0, 1_000_000_000, 30.0, id="zero_origin-30hz"),
        pytest.param(123, 987_654_321, 29.97, id="zero_origin-29.97hz"),
        pytest.param(0, 5_000_000_000, 59.94, id="zero_origin-59.94hz"),
        pytest.param(42, 42 + 1_000_000, 1_000.0, id="zero_origin-1000hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 1_000_000_000, 30.0, id="epoch_round_origin-30hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 987_654_198, 29.97, id="epoch_round_origin-29.97hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 5_000_000_000, 59.94, id="epoch_round_origin-59.94hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 1_000_000, 1_000.0, id="epoch_round_origin-1000hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 1_000_000_000, 30.0, id="epoch_odd_origin-30hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 987_654_198, 29.97, id="epoch_odd_origin-29.97hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 5_000_000_000, 59.94, id="epoch_odd_origin-59.94hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 1_000_000, 1_000.0, id="epoch_odd_origin-1000hz"),
    ],
)
def test_make_ts_grid_brackets_end_ns(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should always produce a final pair that strictly brackets exclusive_end_ns."""
    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    assert got_start_ns == start_ns
    assert len(got_timestamps_ns) >= 2
    assert int(got_timestamps_ns[-1]) <= end_ns < got_exclusive_end_ns


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        pytest.param(0, 1_000_000_000, 30.0, id="zero_origin-30hz"),
        pytest.param(123, 987_654_321, 29.97, id="zero_origin-29.97hz"),
        pytest.param(0, 5_000_000_000, 59.94, id="zero_origin-59.94hz"),
        pytest.param(42, 42 + 1_000_000, 1_000.0, id="zero_origin-1000hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 1_000_000_000, 30.0, id="epoch_round_origin-30hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 987_654_198, 29.97, id="epoch_round_origin-29.97hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 5_000_000_000, 59.94, id="epoch_round_origin-59.94hz"),
        pytest.param(EPOCH_ROUND_NS, EPOCH_ROUND_NS + 1_000_000, 1_000.0, id="epoch_round_origin-1000hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 1_000_000_000, 30.0, id="epoch_odd_origin-30hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 987_654_198, 29.97, id="epoch_odd_origin-29.97hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 5_000_000_000, 59.94, id="epoch_odd_origin-59.94hz"),
        pytest.param(EPOCH_ODD_NS, EPOCH_ODD_NS + 1_000_000, 1_000.0, id="epoch_odd_origin-1000hz"),
    ],
)
def test_make_ts_grid_is_strictly_increasing_and_on_grid(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should stay strictly increasing on the rounded sample interval."""
    _, _, grid = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    deltas = np.diff(grid)
    expected_step_ns = int(np.round(1_000_000_000 / sample_rate_hz))

    assert np.all(deltas > 0)
    assert np.all(np.abs(deltas - expected_step_ns) <= 1)


@pytest.mark.parametrize(
    "origin_ns",
    [
        pytest.param(ZERO_ORIGIN_NS, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid_single_timestamp(origin_ns: int) -> None:
    """When start_ns == end_ns, make_ts_grid should add the next on-grid sample."""
    base_ns = origin_ns + 42
    expected_start_ns, expected_exclusive_end_ns, expected_timestamps_ns = exact_ts_grid(base_ns, base_ns, 30.0)

    start_ns, exclusive_end_ns, timestamps_ns = make_ts_grid(base_ns, base_ns, 30.0)

    assert start_ns == expected_start_ns
    assert exclusive_end_ns == expected_exclusive_end_ns
    np.testing.assert_array_equal(timestamps_ns, expected_timestamps_ns)

    assert len(timestamps_ns) == 1
    assert int(timestamps_ns[0]) == base_ns
    assert start_ns <= int(timestamps_ns[0]) < exclusive_end_ns
    expected_delta_ns = int(np.round(1_000_000_000 / 30.0))
    assert (exclusive_end_ns - start_ns) == expected_delta_ns


@pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0])
def test_make_ts_grid_raises_on_non_positive_sample_rate(sample_rate_hz: float) -> None:
    """make_ts_grid should reject zero or negative sampling rates with ValueError."""
    with pytest.raises(ValueError, match="sample_rate_hz must be greater than 0"):
        make_ts_grid(0, 1_000_000_000, sample_rate_hz)


def test_make_ts_grid_raises_when_end_precedes_start() -> None:
    """make_ts_grid should reject intervals whose end precedes the start."""
    with pytest.raises(ValueError, match="end_ns must be greater than or equal to start_ns"):
        make_ts_grid(10, 0, 1.0)


# Rates spanning every shape the exact interval can take: exactly representable in
# float64 (denominator 1), a repeating third (30/60 Hz), and the NTSC family whose
# exact interval has a ~2e14 denominator and therefore sits arbitrarily close to a
# rounding tie. Only the last group deviates from exact arithmetic at all.
_ACCURACY_RATE_CASES = [
    pytest.param(1_000.0, id="exact_in_float64-1000hz"),
    pytest.param(10.0, id="exact_in_float64-10hz"),
    pytest.param(25.0, id="exact_in_float64-25hz"),
    pytest.param(30.0, id="repeating_third-30hz"),
    pytest.param(60.0, id="repeating_third-60hz"),
    pytest.param(29.97, id="ntsc-29.97hz"),
    pytest.param(59.94, id="ntsc-59.94hz"),
    pytest.param(23.976, id="ntsc-23.976hz"),
]

_ONE_NANOSECOND = 1


def _exact_timestamps_at(start_ns: int, sample_rate_hz: float, indices: np.ndarray) -> np.ndarray:
    """Exact timestamps at the given sample indices.

    Indexed rather than counted on purpose. ``exact_ts_grid`` derives its own sample
    count by an exact floor where ``make_ts_grid`` floors a ``nextafter`` nudge, and
    the two differ by one on spans landing just below a whole multiple of the interval
    -- 10 min at 29.97 Hz is such a span. That is a counting disagreement, not a
    precision one, so accuracy assertions index into the grid actually returned.
    """
    interval_ns = Fraction(1_000_000_000) / Fraction(sample_rate_hz)
    return np.array([start_ns + round(interval_ns * int(k)) for k in indices], dtype=np.int64)


def _spread_indices(count: int, limit: int = 2_000) -> np.ndarray:
    """Up to ``limit`` indices spanning the grid, always including its final sample."""
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, limit).astype(np.int64))


@pytest.mark.parametrize("sample_rate_hz", _ACCURACY_RATE_CASES)
@pytest.mark.parametrize(
    "duration_ns",
    [
        pytest.param(10_000_000_000, id="10s"),
        pytest.param(600_000_000_000, id="10min"),
        pytest.param(7_200_000_000_000, id="2h"),
    ],
)
def test_make_ts_grid_stays_within_one_nanosecond_of_exact(sample_rate_hz: float, duration_ns: int) -> None:
    """Timestamps must track the exact rational grid to within 1 ns.

    Offsets are computed in float64, which is sound because they are relative to an
    integer origin and therefore small -- but only sound to within a nanosecond. A
    rate whose exact interval has a large denominator puts values arbitrarily close
    to a rounding tie, and float64 tips some of them. Pin the bound so it cannot
    quietly grow.
    """
    end_ns = EPOCH_ODD_NS + duration_ns

    got_start_ns, _, got_timestamps_ns = make_ts_grid(EPOCH_ODD_NS, end_ns, sample_rate_hz)

    indices = _spread_indices(len(got_timestamps_ns))
    expected = _exact_timestamps_at(EPOCH_ODD_NS, sample_rate_hz, indices)
    assert got_start_ns == EPOCH_ODD_NS
    deviation = np.abs(got_timestamps_ns.astype(np.int64)[indices] - expected)
    assert deviation.max() <= _ONE_NANOSECOND


@pytest.mark.parametrize("sample_rate_hz", _ACCURACY_RATE_CASES)
def test_make_ts_grid_deviation_does_not_accumulate(sample_rate_hz: float) -> None:
    """The deviation must stay bounded along the grid rather than growing with it.

    This is what separates ``step_ns * k`` from the two tempting rewrites, both of
    which stay under 1 ns on a short grid and so cannot be caught by a small case:
    accumulating the step (a running sum, ``np.cumsum``) drifts ~1132 ns over 18 h at
    29.97 Hz, and pre-rounding the step to whole nanoseconds drifts ~36 us per hour
    at 30 Hz -- enough to pull the boundary marker below ``end_ns``.
    """
    end_ns = 10_800_000_000_000  # 3 h

    _, _, got_timestamps_ns = make_ts_grid(0, end_ns, sample_rate_hz)
    got = got_timestamps_ns.astype(np.int64)
    count = len(got)

    # Contiguous runs at each end: any formulation that accumulates error puts the
    # tail run far off the head run, whatever it does in between.
    window = min(1_000, count)
    head = np.arange(window, dtype=np.int64)
    tail = np.arange(count - window, count, dtype=np.int64)
    head_deviation = np.abs(got[head] - _exact_timestamps_at(0, sample_rate_hz, head))
    tail_deviation = np.abs(got[tail] - _exact_timestamps_at(0, sample_rate_hz, tail))

    assert head_deviation.max() <= _ONE_NANOSECOND
    assert tail_deviation.max() <= _ONE_NANOSECOND


@pytest.mark.parametrize("sample_rate_hz", _ACCURACY_RATE_CASES)
def test_make_ts_grid_spacing_is_no_more_jittery_than_exact(sample_rate_hz: float) -> None:
    """float64 offsets must not add jitter beyond what exact rounding already has.

    The exact grid itself jitters by 1 ns whenever the interval is not a whole number
    of nanoseconds, so that is the ceiling, not zero.
    """
    end_ns = 600_000_000_000  # 10 min
    interval_ns = Fraction(1_000_000_000) / Fraction(sample_rate_hz)
    # An exact grid steps by floor or ceil of the interval, so it is perfectly regular
    # only when the interval is a whole number of nanoseconds.
    allowed_jitter_ns = 0 if interval_ns.denominator == 1 else 1

    _, _, got_timestamps_ns = make_ts_grid(0, end_ns, sample_rate_hz)
    spacing = np.diff(got_timestamps_ns.astype(np.int64))

    assert spacing.min() > 0
    assert spacing.max() - spacing.min() <= allowed_jitter_ns
    assert int(interval_ns) <= spacing.min() <= spacing.max() <= int(interval_ns) + allowed_jitter_ns


@pytest.mark.parametrize("origin_ns", [ZERO_ORIGIN_NS, EPOCH_ROUND_NS, EPOCH_ODD_NS])
def test_make_ts_grid_matches_exact_on_an_hour_long_grid(origin_ns: int) -> None:
    """At 30 Hz an hour-long grid is exact, not merely within a nanosecond.

    30 Hz reduces to denominator 3, so no value comes nearer than 1/6 ns to a
    rounding tie while float64 error at these offsets is ~1e-3 ns. Exactness holds
    out to roughly 8 days of grid; assert it here so a regression that merely stays
    inside the 1 ns bound still shows up.
    """
    sample_rate_hz = 30.0
    end_ns = origin_ns + 3_600_000_000_000

    expected = exact_ts_grid(origin_ns, end_ns, sample_rate_hz)
    got = make_ts_grid(origin_ns, end_ns, sample_rate_hz)

    assert got[0] == expected[0]
    assert got[1] == expected[1]
    np.testing.assert_array_equal(got[2], expected[2])


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        # A 292-year span at a very low rate: the offsets run past int64 while the grid
        # is only ~11 samples, so nothing preempts it. Both the offsets array and the
        # np.diff guard are int64 and wrap, and the wraps cancel well enough that the
        # grid used to be returned with a negative exclusive_end_ns below its own last
        # timestamp.
        pytest.param(0, 2**63 - 1, 1e-9, id="huge_span_low_rate"),
        # Same, but split across the origin so the offsets alone exceed int64 even
        # though start_ns + last_offset does not.
        pytest.param(-(2**62), 2**62, 1e-9, id="span_straddling_zero"),
    ],
)
def test_make_ts_grid_rejects_offsets_that_leave_int64(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """A span whose offsets overflow int64 must be rejected, not silently wrapped."""
    with pytest.raises(ValueError, match="outside signed int64 nanoseconds"):
        make_ts_grid(start_ns, end_ns, sample_rate_hz)


def test_make_ts_grid_raises_when_grid_leaves_int64() -> None:
    """A grid that would run past int64 must raise rather than wrap silently.

    ValueError, not OverflowError: every other make_ts_grid rejection is a ValueError,
    and VideoIndex.with_timestamp_offset raises ValueError for the same condition.
    """
    start_ns = 2**63 - 1 - 10

    with pytest.raises(ValueError, match="outside signed int64 nanoseconds"):
        make_ts_grid(start_ns, start_ns + 5, 30.0)


def test_make_ts_grid_rejects_a_grid_whose_float_offsets_wrap_int64() -> None:
    """A grid the exact-arithmetic pre-check clears, but whose float64 offsets wrap.

    The pre-check bounds ``round(Fraction(1e9) / Fraction(rate) * k)``; the array is built
    from ``round(step_ns * k)`` in float64. The two agree to 1 ns, and 1 ns is the whole
    margin at the int64 boundary. At 29.97 Hz the first index whose float offset lands one
    nanosecond *above* the exact one is k = 104880, about an hour of grid.

    Place ``start_ns`` so the exact last timestamp is ``INT64_MAX`` exactly -- the largest
    value the pre-check accepts -- and the float offset carries it one past, wrapping to
    ``INT64_MIN``. ``np.diff`` cannot catch that: the wrapped pair's difference wraps back
    positive. Without the post-check this returns a grid whose ``exclusive_end_ns`` sits
    below its own last timestamp.

    Reachable only for footage timestamped April 2262, so this pins the guard, not a
    workload.
    """
    sample_rate_hz = 29.97
    last_index = 104_880
    interval_ns = Fraction(1_000_000_000) / Fraction(sample_rate_hz)

    exact_last_offset_ns = round(interval_ns * last_index)
    start_ns = INT64_MAX - exact_last_offset_ns
    # Half an interval past index last_index - 1, so last_index is the final grid index.
    end_ns = start_ns + round(interval_ns * Fraction(2 * last_index - 1, 2))

    # The setup is only meaningful if the pre-check passes: it sees the exact offset, which
    # lands the last timestamp exactly on INT64_MAX. Assert that rather than trusting it,
    # so a drift in either bound fails here instead of silently testing nothing.
    assert start_ns + exact_last_offset_ns == INT64_MAX
    assert round(np.float64(1_000_000_000.0 / sample_rate_hz) * last_index) == exact_last_offset_ns + 1

    with pytest.raises(ValueError, match="wrapped past signed int64 nanoseconds"):
        make_ts_grid(start_ns, end_ns, sample_rate_hz)


def test_make_ts_grid_raises_when_rounding_makes_grid_non_increasing() -> None:
    """make_ts_grid should reject sample rates that cannot produce a strictly increasing ns grid."""
    with pytest.raises(ValueError, match="does not produce a strictly increasing nanosecond grid"):
        make_ts_grid(0, 10, 1.5e9)


@pytest.mark.parametrize(
    "origin_ns",
    [
        pytest.param(ZERO_ORIGIN_NS, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid_exclusive_end_aligned_boundary(origin_ns: int) -> None:
    """Aligned exclusive_end_ns should be returned unchanged and stop strictly before the boundary."""
    start_ns = origin_ns
    sample_rate_hz = 10.0
    exclusive_end_ns = origin_ns + 1_000_000_000  # 10 samples at 10 Hz lands exactly on the boundary

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == exclusive_end_ns
    assert int(got_timestamps_ns[-1]) < exclusive_end_ns
    expected_step_ns = int(np.round(1_000_000_000 / sample_rate_hz))
    deltas = np.diff(got_timestamps_ns)
    assert np.all(deltas > 0)
    assert np.all(np.abs(deltas - expected_step_ns) <= 1)


@pytest.mark.parametrize(
    "origin_ns",
    [
        pytest.param(ZERO_ORIGIN_NS, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid_exclusive_end_non_aligned_boundary(origin_ns: int) -> None:
    """Non-aligned exclusive_end_ns should be returned unchanged with timestamps strictly inside it."""
    start_ns = origin_ns
    sample_rate_hz = 30.0
    exclusive_end_ns = origin_ns + 5_000_000_000

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == exclusive_end_ns
    assert int(got_timestamps_ns[-1]) < exclusive_end_ns
    assert exclusive_end_ns not in got_timestamps_ns


@pytest.mark.parametrize(
    "origin_ns",
    [
        pytest.param(ZERO_ORIGIN_NS, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid_exclusive_end_with_sampling_grid(origin_ns: int) -> None:
    """make_ts_grid with exclusive_end_ns should compose cleanly with SamplingGrid."""
    start_ns = origin_ns
    sample_rate_hz = 10.0
    exclusive_end_ns = origin_ns + 500_000_000  # 0.5 s
    stride_ns = 200_000_000
    duration_ns = 200_000_000

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_ts_grid(
        start_ns,
        sample_rate_hz=sample_rate_hz,
        exclusive_end_ns=exclusive_end_ns,
    )
    grid = SamplingGrid(
        start_ns=got_start_ns,
        exclusive_end_ns=got_exclusive_end_ns,
        timestamps_ns=got_timestamps_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows = list(grid)

    assert grid.exclusive_end_ns == exclusive_end_ns
    assert all(w.exclusive_end_ns <= exclusive_end_ns for w in windows)
    assert all(int(w.timestamps_ns[-1]) < exclusive_end_ns for w in windows if len(w) > 0)


def test_make_ts_grid_raises_when_both_ends_supplied() -> None:
    """make_ts_grid should reject having both end_ns and exclusive_end_ns supplied."""
    with pytest.raises(ValueError, match="exactly one of end_ns or exclusive_end_ns"):
        make_ts_grid(0, 1_000_000_000, 30.0, exclusive_end_ns=1_000_000_000)


def test_make_ts_grid_raises_when_neither_end_supplied() -> None:
    """make_ts_grid should reject having neither end_ns nor exclusive_end_ns supplied."""
    with pytest.raises(ValueError, match="exactly one of end_ns or exclusive_end_ns"):
        make_ts_grid(0, sample_rate_hz=30.0)


def test_make_ts_grid_raises_when_exclusive_end_le_start() -> None:
    """make_ts_grid should reject exclusive_end_ns that is not strictly greater than start_ns."""
    with pytest.raises(ValueError, match="exclusive_end_ns must be greater than start_ns"):
        make_ts_grid(100, sample_rate_hz=30.0, exclusive_end_ns=100)


def _expected_window_count(first: int, last: int, stride_ns: int) -> int:
    """Windows emitted while start <= last with start = first + k * stride_ns."""
    if stride_ns <= 0 or first > last:
        return 0
    if first == last:
        return 1
    return (last - first) // stride_ns


@pytest.mark.parametrize(
    ("timestamps_ns", "stride_ns", "duration_ns", "raises"),
    [
        # valid
        (np.array([0, 1], dtype=np.int64), 1, 1, nullcontext()),
        # timestamps must be int64
        (np.array([0, 1], dtype=np.int32), 1, 1, pytest.raises(ValueError, match=r".*")),
        # zero duration
        (np.array([0], dtype=np.int64), 1, 0, pytest.raises(ValueError, match=r".*")),
        # negative duration
        (np.array([0], dtype=np.int64), 1, -1, pytest.raises(ValueError, match=r".*")),
        # zero stride
        (np.array([0], dtype=np.int64), 0, 1, pytest.raises(ValueError, match=r".*")),
        # negative stride
        (np.array([0], dtype=np.int64), -1, 1, pytest.raises(ValueError, match=r".*")),
        # not sorted
        (np.array([1, 0], dtype=np.int64), 1, 1, pytest.raises(ValueError, match=r".*")),
    ],
)
def test_initializer_asserts(
    timestamps_ns: npt.NDArray[np.int64],
    stride_ns: int,
    duration_ns: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test initializer asserts."""
    with raises:
        make_sampling_grid(timestamps_ns=timestamps_ns, stride_ns=stride_ns, duration_ns=duration_ns)


def test_initializer_state() -> None:
    """Test state after initialization."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    p = make_sampling_grid(timestamps_ns=ts, stride_ns=1, duration_ns=10)
    for t, p_t in zip(ts[:-1], p.timestamps_ns, strict=True):
        assert t == p_t
    assert p.start_ns == ts[0]
    assert p.exclusive_end_ns == ts[-1]
    assert p.stride_ns == 1
    assert p.duration_ns == 10
    assert not np.shares_memory(p.timestamps_ns, ts)


def test_irregular_grid() -> None:
    """Test an irregular grid.

    * Irregular grid should produce the same number of windows as the regular grid.
    * Windows should have the same start and exclusive end times.
    """
    regular_ts_ns = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    irregular_ts_ns = np.array([0, 9, 21, 29, 40, 51], dtype=np.int64)
    expected_irregular_windows = [
        np.array([0, 9], dtype=np.int64),
        np.array([21, 29], dtype=np.int64),
        np.array([40, 51], dtype=np.int64),
    ]
    start_ns = regular_ts_ns[0]
    exclusive_end_ns = 60
    duration_ns = 20
    stride_ns = 20

    grid_regular = SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=regular_ts_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    grid_irregular = SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
        timestamps_ns=irregular_ts_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows_regular = list(grid_regular)
    windows_irregular = list(grid_irregular)

    assert len(windows_regular) == len(windows_irregular)

    for window_regular, window_irregular, expected_irregular in zip(
        windows_regular, windows_irregular, expected_irregular_windows, strict=True
    ):
        assert len(window_regular.timestamps_ns) == len(window_irregular.timestamps_ns)
        np.testing.assert_array_equal(window_irregular.timestamps_ns, expected_irregular)
        assert window_regular.start_ns == window_irregular.start_ns
        assert window_regular.exclusive_end_ns == window_irregular.exclusive_end_ns


def test_stride_equals_duration_tiling() -> None:
    """With stride == duration, windows tile the timeline and reuse boundary markers between windows."""
    ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    duration_ns = 200
    stride_ns = 200
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    want = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=200,
            timestamps_ns=np.array([0, 100], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=200,
            exclusive_end_ns=400,
            timestamps_ns=np.array([200, 300], dtype=np.int64),
        ),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_stride_equals_duration_last_window_full_query_one_sample() -> None:
    """The final yielded window may have no timestamps when the last sample only serves as the terminal boundary."""
    ts = np.array([0, 100, 200, 500], dtype=np.int64)
    duration_ns = 400
    stride_ns = 400
    windows = list(make_sampling_grid(ts, stride_ns, duration_ns))
    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=400,
            timestamps_ns=np.array([0, 100, 200], dtype=np.int64),
        ),
        SamplingWindow(
            start_ns=400,
            exclusive_end_ns=500,
            timestamps_ns=np.array([], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for window, expected_window in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(window.timestamps_ns, expected_window.timestamps_ns)
        assert window.start_ns == expected_window.start_ns
        assert window.exclusive_end_ns == expected_window.exclusive_end_ns


def test_iter_stride_less_than_duration_overlap() -> None:
    """Stride smaller than duration yields overlapping windows and predictable count."""
    ts = np.arange(0, 501, 50, dtype=np.int64)
    duration_ns = 200
    stride_ns = 100
    p = make_sampling_grid(ts, stride_ns, duration_ns)
    windows = list(p)
    n = _expected_window_count(p.start_ns, p.exclusive_end_ns, stride_ns)
    assert len(windows) == n

    # Check that windows actually overlap when stride_ns < duration_ns
    # Restrict this to the two windows that share the 150us timestamp
    t_overlap = 150
    windows_with_t = [w for w in windows if np.any(w.timestamps_ns == t_overlap)]
    assert len(windows_with_t) == 2

    # Check that timestamps in each window are within the expected range
    starts = [p.start_ns + k * stride_ns for k in range(len(windows))]
    for w, start in zip(windows, starts, strict=True):
        end = start + duration_ns
        for t in w.timestamps_ns:
            assert start <= int(t) <= end

    # Check that the indices of the windows are monotonic and non-overlapping
    # This is a thin belt & suspenders check, indices should never decrease
    # This grid has a sample on every window start, so indices strictly increase
    first_indices = [int(np.searchsorted(ts, w.timestamps_ns[0], side="left")) for w in windows]
    for prev, cur in pairwise(first_indices):
        assert cur > prev


def test_iter_irregular_sparse_grid_repeated_first_sample_across_windows() -> None:
    """Sparse, irregular timestamps can cause consecutive windows to start with the same sample.

    Here the windows starting at ``3000`` and ``5000`` both begin with ``5000`` and include the
    same sparse pair ``[5000, 5100]``. This is expected on irregular timelines even though the
    window bounds still advance by ``stride_ns``.
    """
    # Gaps: 4ms idle, ~0.1ms pair, ~3.9ms idle — not on a fixed grid.
    ts = np.array([1_000, 5_000, 5_100, 9_000], dtype=np.int64)
    duration_ns = 3_000
    stride_ns = 2_000
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(start_ns=1000, exclusive_end_ns=4000, timestamps_ns=np.array([1000], dtype=np.int64)),
        SamplingWindow(start_ns=3000, exclusive_end_ns=6000, timestamps_ns=np.array([5000, 5100], dtype=np.int64)),
        SamplingWindow(start_ns=5000, exclusive_end_ns=8000, timestamps_ns=np.array([5000, 5100], dtype=np.int64)),
        SamplingWindow(start_ns=7000, exclusive_end_ns=9000, timestamps_ns=np.array([], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_stride_greater_than_duration_gaps() -> None:
    """Stride larger than duration leaves timestamps that fall in no yielded window.

    grid:    0    50   100    200    300    400    500
    windows: |----------|          |------|         |----|
             0         100        250    350        500  500
                                                    No later boundary marker, so no final window
    """
    ts = np.array([0, 50, 100, 200, 300, 400, 500], dtype=np.int64)
    duration_ns = 100
    stride_ns = 250
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(timestamps_ns=np.array([0, 50], dtype=np.int64), start_ns=0, exclusive_end_ns=100),
        SamplingWindow(timestamps_ns=np.array([300], dtype=np.int64), start_ns=250, exclusive_end_ns=350),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns

    expected_orphans = [200, 400, 500]
    covered = np.array(
        list({int(x) for x in np.concatenate([w.timestamps_ns for w in got])} | {w.exclusive_end_ns for w in want}),
        dtype=np.int64,
    )
    orphans = np.setdiff1d(ts, covered)
    np.testing.assert_array_equal(orphans, expected_orphans)


def test_boundaries_on_every_timestamp() -> None:
    """Every timestamp after the first can serve as a shared boundary marker between adjacent windows."""
    ts = np.array([0, 100, 200], dtype=np.int64)
    duration_ns = 100
    stride_ns = 100
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    want = [
        SamplingWindow(start_ns=0, exclusive_end_ns=100, timestamps_ns=np.array([0], dtype=np.int64)),
        SamplingWindow(start_ns=100, exclusive_end_ns=200, timestamps_ns=np.array([100], dtype=np.int64)),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_initializer_defensively_copies_timestamps() -> None:
    """Mutating the caller-owned array after construction must not affect the grid."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    got = _iter_window_arrays(ts, stride_ns=100, duration_ns=100)
    ts[1] = 999
    want = [
        SamplingWindow(start_ns=100, exclusive_end_ns=200, timestamps_ns=np.array([100], dtype=np.int64)),
        SamplingWindow(start_ns=200, exclusive_end_ns=300, timestamps_ns=np.array([200], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_iter_sample_span_inside_window_may_be_shorter_than_duration() -> None:
    """Timestamps in a yielded window can span less than ``duration_ns`` when the timeline has gaps.

    grid:     0    100   200   400   700   900 <-- exclusive_end_ns, not included in the last window
    windows:  |--------------|-----|--------|
              0             300   600      900
    """
    duration_ns = 300
    stride_ns = 300
    ts = np.array([0, 100, 200, 400, 700, 900], dtype=np.int64)
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    want = [
        SamplingWindow(start_ns=0, exclusive_end_ns=300, timestamps_ns=np.array([0, 100, 200], dtype=np.int64)),
        SamplingWindow(start_ns=300, exclusive_end_ns=600, timestamps_ns=np.array([400], dtype=np.int64)),
        SamplingWindow(start_ns=600, exclusive_end_ns=900, timestamps_ns=np.array([700], dtype=np.int64)),
    ]

    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a.timestamps_ns, b.timestamps_ns)
        assert a.start_ns == b.start_ns
        assert a.exclusive_end_ns == b.exclusive_end_ns


def test_duration_shorter_than_stride() -> None:
    """Duration shorter than stride yields disjoint windows whose slices may end before the nominal interval end.

    grid:     0    60    120   180     240   300
    windows:  |-------|   |--------|    |--------|
              0      100 120       220  240     340

    The final yielded slice ends at 300 because no later timestamp is available.
    """
    ts = np.array([0, 60, 120, 180, 240, 300], dtype=np.int64)
    duration_ns = 100
    stride_ns = 120
    p = make_sampling_grid(ts, stride_ns, duration_ns)
    got_windows = list(p)
    expected_windows = [
        SamplingWindow(start_ns=0, exclusive_end_ns=100, timestamps_ns=np.array([0, 60], dtype=np.int64)),
        SamplingWindow(start_ns=120, exclusive_end_ns=220, timestamps_ns=np.array([120, 180], dtype=np.int64)),
        SamplingWindow(start_ns=240, exclusive_end_ns=300, timestamps_ns=np.array([240], dtype=np.int64)),
    ]

    assert len(got_windows) == len(expected_windows)
    for got_window, expected_window in zip(got_windows, expected_windows, strict=True):
        np.testing.assert_array_equal(got_window.timestamps_ns, expected_window.timestamps_ns)
        assert got_window.start_ns == expected_window.start_ns
        assert got_window.exclusive_end_ns == expected_window.exclusive_end_ns


def test_iter_stride_equals_duration_many_steps() -> None:
    """With stride == duration, a coarse rounded grid still tiles into predictable windows even when counts differ."""
    # 31 timestamps 0, 1, 2, ..., 30 each scaled by 1_000_000_000 // 3 ns (~333.3 ms steps).
    # >>> ts = np.arange(0, 31, dtype=np.int64) * (1_000_000_000 // 3)
    # This is a slightly irregular grid, the first window has more samples than the others.
    ts = np.array(
        [
            # window 0: start_ns: 0, exclusive_end_ns: 2_000_000_000
            0,
            333333333,
            666666666,
            999999999,
            1333333332,
            1666666665,
            1999999998,
            # window 1: start_ns: 2_000_000_000, exclusive_end_ns: 4_000_000_000
            2333333331,
            2666666664,
            2999999997,
            3333333330,
            3666666663,
            3999999996,
            # window 2: start_ns: 4_000_000_000, exclusive_end_ns: 6_000_000_000
            4333333329,
            4666666662,
            4999999995,
            5333333328,
            5666666661,
            5999999994,
            # window 3: start_ns: 6_000_000_000, exclusive_end_ns: 8_000_000_000
            6333333327,
            6666666660,
            6999999993,
            7333333326,
            7666666659,
            7999999992,
            # window 4: start_ns: 8_000_000_000, exclusive_end_ns: 10_000_000_000
            8333333325,
            8666666658,
            8999999991,
            9333333324,
            9666666657,
            9999999990,
        ],
        dtype=np.int64,
    )

    duration_ns = 2 * 1_000_000_000
    stride_ns = duration_ns
    windows = list(make_sampling_grid(ts, stride_ns, duration_ns))
    expected_windows = [
        SamplingWindow(
            start_ns=0,
            exclusive_end_ns=2_000_000_000,
            timestamps_ns=np.array(
                [0, 333333333, 666666666, 999999999, 1333333332, 1666666665, 1999999998], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=2_000_000_000,
            exclusive_end_ns=4_000_000_000,
            timestamps_ns=np.array(
                [2333333331, 2666666664, 2999999997, 3333333330, 3666666663, 3999999996], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=4_000_000_000,
            exclusive_end_ns=6_000_000_000,
            timestamps_ns=np.array(
                [4333333329, 4666666662, 4999999995, 5333333328, 5666666661, 5999999994], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=6_000_000_000,
            exclusive_end_ns=8_000_000_000,
            timestamps_ns=np.array(
                [6333333327, 6666666660, 6999999993, 7333333326, 7666666659, 7999999992], dtype=np.int64
            ),
        ),
        SamplingWindow(
            start_ns=8_000_000_000,
            exclusive_end_ns=ts[-1],
            timestamps_ns=np.array([8333333325, 8666666658, 8999999991, 9333333324, 9666666657], dtype=np.int64),
        ),
    ]

    assert len(windows) == len(expected_windows)
    for w, expected_w in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(w.timestamps_ns, expected_w.timestamps_ns)
        assert w.start_ns == expected_w.start_ns
        assert w.exclusive_end_ns == expected_w.exclusive_end_ns


@pytest.mark.parametrize(
    ("timestamps_ns", "start_ns", "exclusive_end_ns"),
    [
        (np.array([10, 20], dtype=np.int64), 10, 30),
        (np.array([10, 20], dtype=np.int64), 5, 30),
        (np.array([], dtype=np.int64), 10, 20),
    ],
)
def test_sampling_window_accepts_valid_bounds(
    timestamps_ns: npt.NDArray[np.int64],
    start_ns: int,
    exclusive_end_ns: int,
) -> None:
    """SamplingWindow should allow start_ns to precede the first timestamp."""
    window = SamplingWindow(
        timestamps_ns=timestamps_ns,
        start_ns=start_ns,
        exclusive_end_ns=exclusive_end_ns,
    )
    np.testing.assert_array_equal(window.timestamps_ns, timestamps_ns)


@pytest.mark.parametrize(
    ("timestamps_ns", "start_ns", "exclusive_end_ns", "raises"),
    [
        (np.array([10], dtype=np.int64), 10, 9, pytest.raises(ValueError, match="end_ns must be greater than")),
        (
            np.array([9, 10], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="start_ns must be <="),
        ),
        (
            np.array([10, 20], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="end_ns must be < exclusive_end_ns"),
        ),
        (
            np.array([10, 10], dtype=np.int64),
            10,
            20,
            pytest.raises(ValueError, match="strictly sorted"),
        ),
        (
            np.array([10, 20], dtype=np.int32),
            10,
            30,
            pytest.raises(ValueError, match="must have dtype int64"),
        ),
        (
            np.array([[10, 20]], dtype=np.int64),
            10,
            30,
            pytest.raises(ValueError, match="must be 1-D"),
        ),
    ],
)
def test_sampling_window_rejects_invalid_inputs(
    timestamps_ns: npt.NDArray[np.int64],
    start_ns: int,
    exclusive_end_ns: int,
    raises: AbstractContextManager[Any],
) -> None:
    """SamplingWindow should reject inputs that violate its bounds or array invariants."""
    with raises:
        SamplingWindow(
            timestamps_ns=timestamps_ns,
            start_ns=start_ns,
            exclusive_end_ns=exclusive_end_ns,
        )


def test_sampling_window_timestamps_are_read_only() -> None:
    """SamplingWindow should expose a read-only timestamp array."""
    window = SamplingWindow(
        timestamps_ns=np.array([0, 100, 200], dtype=np.int64),
        start_ns=0,
        exclusive_end_ns=300,
    )

    assert not window.timestamps_ns.flags.writeable
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        window.timestamps_ns[0] = 1


def test_sampling_window_does_not_mutate_caller_owned_timestamps() -> None:
    """SamplingWindow should keep the caller's timestamp array writeable."""
    timestamps_ns = np.array([0, 100, 200], dtype=np.int64)

    window = SamplingWindow(
        timestamps_ns=timestamps_ns,
        start_ns=0,
        exclusive_end_ns=300,
    )

    assert timestamps_ns.flags.writeable is True
    assert window.timestamps_ns.flags.writeable is False
    assert window.timestamps_ns is not timestamps_ns
    assert np.shares_memory(window.timestamps_ns, timestamps_ns)


def test_sampling_window_len() -> None:
    """SamplingWindow should return the number of active timestamps in the window."""
    ts = np.array([0, 100, 200], dtype=np.int64)
    window = SamplingWindow(
        timestamps_ns=ts,
        start_ns=0,
        exclusive_end_ns=300,
    )
    assert len(window) == len(ts)
    assert len(window.timestamps_ns) == len(ts)

    ts = np.array([], dtype=np.int64)
    window = SamplingWindow(
        timestamps_ns=ts,
        start_ns=0,
        exclusive_end_ns=300,
    )
    assert len(window) == 0
    assert len(window.timestamps_ns) == 0


def test_sampling_grid_iter_is_origin_invariant() -> None:
    """Shifting a SamplingGrid to an epoch origin should shift every window by the same offset."""
    timestamps_ns = np.array([0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000], dtype=np.int64)
    stride_ns = 20_000_000
    duration_ns = 20_000_000

    base_windows = list(
        SamplingGrid(
            start_ns=int(timestamps_ns[0]),
            exclusive_end_ns=int(timestamps_ns[-1]),
            timestamps_ns=timestamps_ns[:-1],
            stride_ns=stride_ns,
            duration_ns=duration_ns,
        )
    )
    shifted_timestamps_ns = timestamps_ns + EPOCH_ODD_NS
    shifted_windows = list(
        SamplingGrid(
            start_ns=int(shifted_timestamps_ns[0]),
            exclusive_end_ns=int(shifted_timestamps_ns[-1]),
            timestamps_ns=shifted_timestamps_ns[:-1],
            stride_ns=stride_ns,
            duration_ns=duration_ns,
        )
    )

    assert len(shifted_windows) == len(base_windows)
    for shifted, base in zip(shifted_windows, base_windows, strict=True):
        assert shifted.start_ns == base.start_ns + EPOCH_ODD_NS
        assert shifted.exclusive_end_ns == base.exclusive_end_ns + EPOCH_ODD_NS
        np.testing.assert_array_equal(shifted.timestamps_ns, base.timestamps_ns + EPOCH_ODD_NS)


@pytest.mark.parametrize(
    "origin_ns",
    [
        pytest.param(ZERO_ORIGIN_NS, id="zero_origin"),
        pytest.param(EPOCH_ROUND_NS, id="epoch_round_origin"),
        pytest.param(EPOCH_ODD_NS, id="epoch_odd_origin"),
    ],
)
def test_make_ts_grid_preserves_start_ns(origin_ns: int) -> None:
    """make_ts_grid should return the caller's start_ns unchanged, not a float64 round trip of it."""
    start_ns = origin_ns + 42

    got_start_ns, _, got_timestamps_ns = make_ts_grid(start_ns, start_ns + 1_000_000_000, 30.0)

    assert got_start_ns == start_ns
    assert int(got_timestamps_ns[0]) == start_ns
