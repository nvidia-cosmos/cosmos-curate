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
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curate.core.sensors.sampling.grid import SamplingGrid, make_ts_grid


def _iter_window_arrays(
    ts: npt.NDArray[np.int64],
    *,
    stride_ns: int,
    duration_ns: int,
) -> list[npt.NDArray[np.int64]]:
    p = SamplingGrid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    return [w.copy() for w in p]


def test_sampling_grid_iter() -> None:
    """SamplingGrid should yield raw window slices including the right boundary marker."""
    ts = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    sampling_grid = SamplingGrid(ts, 20, 20)

    expected_windows = [
        np.array([0, 10, 20], dtype=np.int64),
        np.array([20, 30, 40], dtype=np.int64),
        np.array([40, 50], dtype=np.int64),
    ]

    for window, expected_window in zip(sampling_grid, expected_windows, strict=True):
        np.testing.assert_array_equal(window, expected_window)


def test_sampling_grid_half_open_window_contract() -> None:
    """A yielded window should include the exclusive right boundary marker as its final element."""
    ts = np.array([100, 200, 300, 400], dtype=np.int64)
    windows = list(SamplingGrid(ts, 200, 200))

    np.testing.assert_array_equal(windows[0], np.array([100, 200, 300], dtype=np.int64))
    np.testing.assert_array_equal(windows[1], np.array([300, 400], dtype=np.int64))

    # The final element is the boundary marker; downstream active references are window[:-1].
    np.testing.assert_array_equal(windows[0][:-1], np.array([100, 200], dtype=np.int64))
    assert int(windows[0][-1]) == 300


def test_sampling_grid_adjacent_windows_share_boundary_marker() -> None:
    """Adjacent windows should share the boundary timestamp that separates their half-open intervals."""
    ts = np.array([0, 100, 200, 300], dtype=np.int64)
    windows = list(SamplingGrid(ts, 100, 100))

    assert len(windows) == 3
    assert int(windows[0][-1]) == 100
    assert int(windows[1][0]) == 100
    assert int(windows[1][-1]) == 200
    assert int(windows[2][0]) == 200


def test_sampling_grid_singleton_window_is_boundary_only() -> None:
    """A singleton yielded window contains only a boundary marker and therefore zero active reference timestamps."""
    ts = np.array([42], dtype=np.int64)
    windows = list(SamplingGrid(ts, 1, 10))

    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0], np.array([42], dtype=np.int64))
    np.testing.assert_array_equal(windows[0][:-1], np.array([], dtype=np.int64))


def test_make_ts_grid() -> None:
    """Test the make_ts_grid function."""
    start_s = 0.0
    end_s = 5.0
    sample_rate_hz = 30.0
    sample_interval_s = 1 / sample_rate_hz
    n_samples = int(np.floor(np.nextafter((end_s - start_s) / sample_interval_s, np.inf))) + 2
    expected_ts = start_s + np.arange(n_samples, dtype=np.float64) * sample_interval_s
    expected_grid = np.round(expected_ts * 1_000_000_000).astype(np.int64)

    start_ns = int(start_s * 1_000_000_000)
    end_ns = int(end_s * 1_000_000_000)
    grid = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    np.testing.assert_array_equal(grid, expected_grid)


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        (0, 1_000_000_000, 30.0),
        (123, 987_654_321, 29.97),
        (0, 5_000_000_000, 59.94),
        (42, 42 + 1_000_000, 1_000.0),
    ],
)
def test_make_ts_grid_brackets_end_ns(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should always produce a final pair that strictly brackets end_ns."""
    grid = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    assert len(grid) >= 2
    assert int(grid[-2]) <= end_ns < int(grid[-1])


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "sample_rate_hz"),
    [
        (0, 1_000_000_000, 30.0),
        (123, 987_654_321, 29.97),
        (0, 5_000_000_000, 59.94),
        (42, 42 + 1_000_000, 1_000.0),
    ],
)
def test_make_ts_grid_is_strictly_increasing_and_on_grid(start_ns: int, end_ns: int, sample_rate_hz: float) -> None:
    """make_ts_grid should stay strictly increasing on the rounded sample interval."""
    grid = make_ts_grid(start_ns, end_ns, sample_rate_hz)

    deltas = np.diff(grid)
    expected_step_ns = int(np.round(1_000_000_000 / sample_rate_hz))

    assert np.all(deltas > 0)
    assert np.all(np.abs(deltas - expected_step_ns) <= 1)


def test_make_ts_grid_single_timestamp() -> None:
    """When start_ns == end_ns, make_ts_grid should add the next on-grid sample."""
    grid = make_ts_grid(42, 42, 30.0)

    assert len(grid) == 2
    assert int(grid[0]) == 42
    assert int(grid[0]) <= 42 < int(grid[1])
    expected_delta_ns = int(np.round(1_000_000_000 / 30.0))
    assert int(grid[1] - grid[0]) == expected_delta_ns


@pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0])
def test_make_ts_grid_raises_on_non_positive_sample_rate(sample_rate_hz: float) -> None:
    """make_ts_grid should reject zero or negative sampling rates with ValueError."""
    with pytest.raises(ValueError, match="sample_rate_hz must be greater than 0"):
        make_ts_grid(0, 1_000_000_000, sample_rate_hz)


def test_make_ts_grid_raises_when_end_precedes_start() -> None:
    """make_ts_grid should reject intervals whose end precedes the start."""
    with pytest.raises(ValueError, match="end_ns must be greater than or equal to start_ns"):
        make_ts_grid(10, 0, 1.0)


def test_make_ts_grid_raises_when_rounding_makes_grid_non_increasing() -> None:
    """make_ts_grid should reject sample rates that cannot produce a strictly increasing ns grid."""
    with pytest.raises(ValueError, match="does not produce a strictly increasing nanosecond grid"):
        make_ts_grid(0, 10, 1.5e9)


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
        # empty timestamps
        (np.array([], dtype=np.int64), 1, 1, pytest.raises(ValueError, match=r".*")),
        # timestamps must be 1-D
        (np.array([[0, 1]], dtype=np.int64), 1, 1, pytest.raises(ValueError, match=r".*")),
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
        SamplingGrid(timestamps_ns=timestamps_ns, stride_ns=stride_ns, duration_ns=duration_ns)


def test_initializer_state() -> None:
    """Test state after initialization."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    p = SamplingGrid(timestamps_ns=ts, stride_ns=1, duration_ns=10)
    for t, p_t in zip(ts, p.timestamps_ns, strict=True):
        assert t == p_t
    assert p.start_ns == ts[0]
    assert p.end_ns == ts[-1]
    assert p.stride_ns == 1
    assert p.duration_ns == 10
    assert not np.shares_memory(p.timestamps_ns, ts)


def test_stride_equals_duration_tiling() -> None:
    """With stride == duration, windows tile the timeline and reuse boundary markers between windows."""
    ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    duration_ns = 200
    stride_ns = 200
    got = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    want = [
        np.array([0, 100, 200], dtype=np.int64),
        np.array([200, 300, 400], dtype=np.int64),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a, b)


def test_iter_stride_equals_duration_last_window_full_query_one_sample() -> None:
    """The final yielded window may contain only a boundary marker when no later samples exist."""
    ts = np.array([0, 100, 200, 500], dtype=np.int64)
    duration_ns = 400
    stride_ns = 400
    windows = list(
        SamplingGrid(
            timestamps_ns=ts,
            stride_ns=stride_ns,
            duration_ns=duration_ns,
        )
    )
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], [0, 100, 200])
    np.testing.assert_array_equal(windows[1], [500])


def test_iter_stride_less_than_duration_overlap() -> None:
    """Stride smaller than duration yields overlapping windows and predictable count."""
    ts = np.arange(0, 501, 50, dtype=np.int64)
    duration_ns = 200
    stride_ns = 100
    p = SamplingGrid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows = list(p)
    n = _expected_window_count(p.start_ns, p.end_ns, stride_ns)
    assert len(windows) == n

    # Check that windows actually overlap when stride_ns < duration_ns
    # Restrict this to the two windows that share the 150us timestamp
    t_overlap = 150
    windows_with_t = [w for w in windows if np.any(w == t_overlap)]
    assert len(windows_with_t) == 2

    # Check that timestamps in each window are within the expected range
    starts = [p.start_ns + k * stride_ns for k in range(len(windows))]
    for w, start in zip(windows, starts, strict=True):
        end = start + duration_ns
        for t in w:
            assert start <= int(t) <= end

    # Check that the indices of the windows are monotonic and non-overlapping
    # This is a thin belt & suspenders check, indices should never decrease
    # This grid has a sample on every window start, so indices strictly increase
    first_indices = [int(np.searchsorted(ts, w[0], side="left")) for w in windows]
    for prev, cur in pairwise(first_indices):
        assert cur > prev


def test_iter_irregular_sparse_grid_repeated_first_sample_across_windows() -> None:
    """Sparse, irregular µs times: later window can start at the same sample as a prior minimum.

    Here two consecutive windows both lead with ``5000`` — the first sample in each slice is
    the same event, but the slices still differ. This is expected for irregular sparse timelines;
    window bounds still advance by ``stride_ns``.
    """
    # Gaps: 4ms idle, ~0.1ms pair, ~3.9ms idle — not on a fixed grid.
    ts = np.array([1_000, 5_000, 5_100, 9_000], dtype=np.int64)
    duration_ns = 3_000
    stride_ns = 2_000
    p = SamplingGrid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    windows = list(p)

    assert len(windows) == _expected_window_count(p.start_ns, p.end_ns, stride_ns)
    np.testing.assert_array_equal(windows[0], [1_000])
    # Repeated leading samples are expected for sparse irregular timelines.
    np.testing.assert_array_equal(windows[1], [5_000, 5_100])
    np.testing.assert_array_equal(windows[2], [5_000, 5_100])
    np.testing.assert_array_equal(windows[3], [9_000])

    # Check that timestamps in each window are within the expected range
    first_usec = [int(w[0]) for w in windows]
    for prev, cur in pairwise(first_usec):
        assert cur >= prev
    assert first_usec[1] == first_usec[2] == 5_000

    starts = [p.start_ns + k * stride_ns for k in range(len(windows))]
    for w, start in zip(windows, starts, strict=True):
        end = start + duration_ns
        for t in w:
            assert start <= int(t) <= end

    # Window starts should remain monotonic even when the first sample repeats.
    first_indices = [int(np.searchsorted(ts, w[0], side="left")) for w in windows]
    for prev, cur in pairwise(first_indices):
        assert cur >= prev


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
    windows = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)
    expected_windows = [
        np.array([0, 50, 100], dtype=np.int64),
        np.array([300], dtype=np.int64),
    ]
    expected_orphans = [200, 400, 500]

    assert len(windows) == len(expected_windows)
    covered = np.unique(np.concatenate(windows))
    orphans = np.setdiff1d(ts, covered)
    np.testing.assert_array_equal(orphans, expected_orphans)


def test_boundaries_on_every_timestamp() -> None:
    """Every timestamp after the first can serve as a shared boundary marker between adjacent windows."""
    ts = np.array([0, 100, 200], dtype=np.int64)
    duration_ns = 100
    stride_ns = 100
    got = _iter_window_arrays(
        ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    want = [
        np.array([0, 100], dtype=np.int64),
        np.array([100, 200], dtype=np.int64),
    ]
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a, b)


def test_iter_child_slices_share_parent_memory() -> None:
    """Each yielded window's timestamps view into the grid-owned timestamp array."""
    ts = np.array([0, 50, 100, 200, 250], dtype=np.int64)
    p = SamplingGrid(
        timestamps_ns=ts,
        stride_ns=100,
        duration_ns=100,
    )
    for w in p:
        assert np.shares_memory(w, p.timestamps_ns)
        assert not np.shares_memory(w, ts)
        assert not w.flags.owndata
        assert not w.flags.writeable


def test_initializer_defensively_copies_timestamps() -> None:
    """Mutating the caller-owned array after construction must not affect the grid."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    grid = SamplingGrid(timestamps_ns=ts, stride_ns=100, duration_ns=100)

    ts[1] = 999

    np.testing.assert_array_equal(grid.timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    windows = list(grid)
    np.testing.assert_array_equal(windows[0], np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(windows[1], np.array([200, 300], dtype=np.int64))


def test_iter_single_timestamp_yields_one_window() -> None:
    """A single timestamp still yields one degenerate boundary-only window."""
    ts = np.array([42], dtype=np.int64)
    p = SamplingGrid(timestamps_ns=ts, stride_ns=1, duration_ns=10)
    windows = list(p)
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0], [42])


def test_iter_sample_span_inside_window_may_be_shorter_than_duration() -> None:
    """Samples in a yielded window can span less than ``duration_ns`` when the timeline has gaps.

    grid:     0    100   200   400   700   900
    windows:  |--------------|-----|--------|
              0             300   600      900
    """
    duration_ns = 300
    stride_ns = 300
    ts = np.array([0, 100, 200, 400, 700, 900], dtype=np.int64)
    windows = _iter_window_arrays(ts, stride_ns=stride_ns, duration_ns=duration_ns)

    expected_windows = [
        np.array([0, 100, 200], dtype=np.int64),
        np.array([400], dtype=np.int64),
        np.array([700, 900], dtype=np.int64),
    ]

    assert len(windows) == len(expected_windows)

    def sample_span_ns(w: npt.NDArray[np.int64]) -> int:
        return int(w[-1] - w[0]) if len(w) >= 2 else 0

    assert sample_span_ns(windows[0]) < duration_ns
    assert sample_span_ns(windows[1]) < duration_ns
    assert sample_span_ns(windows[-1]) < duration_ns

    for w, expected_w in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(w, expected_w)


def test_duration_short_than_stride() -> None:
    """Duration shorter than stride yields disjoint windows whose slices may end before the nominal interval end.

    grid:     0    60    120   180     240   300
    windows:  |-------|   |--------|    |--------|
              0      100 120       220  240     340

    The final yielded slice ends at 300 because no later timestamp is available.
    """
    ts = np.array([0, 60, 120, 180, 240, 300], dtype=np.int64)
    duration_ns = 100
    stride_ns = 120
    p = SamplingGrid(
        timestamps_ns=ts,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )
    got_windows = list(p)
    expected_windows = [
        np.array([0, 60], dtype=np.int64),
        np.array([120, 180], dtype=np.int64),
        np.array([240, 300], dtype=np.int64),
    ]

    for got_window, expected_window in zip(got_windows, expected_windows, strict=True):
        np.testing.assert_array_equal(got_window, expected_window)


def test_iter_stride_equals_duration_many_steps() -> None:
    """Coarser grid with stride == duration produces full expected windows."""
    # 31 timestamps 0, 1, 2, ..., 30 each scaled by 1_000_000_000 // 3 ns (~333.3 ms steps).
    # >>> ts = np.arange(0, 31, dtype=np.int64) * (1_000_000_000 // 3)
    ts = np.array(
        [
            0,
            333333333,
            666666666,
            999999999,
            1333333332,
            1666666665,
            1999999998,
            2333333331,
            2666666664,
            2999999997,
            3333333330,
            3666666663,
            3999999996,
            4333333329,
            4666666662,
            4999999995,
            5333333328,
            5666666661,
            5999999994,
            6333333327,
            6666666660,
            6999999993,
            7333333326,
            7666666659,
            7999999992,
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
    windows = list(
        SamplingGrid(
            timestamps_ns=ts,
            stride_ns=stride_ns,
            duration_ns=duration_ns,
        )
    )

    expected_windows = [
        np.array([0, 333333333, 666666666, 999999999, 1333333332, 1666666665, 1999999998], dtype=np.int64),
        np.array([2333333331, 2666666664, 2999999997, 3333333330, 3666666663, 3999999996], dtype=np.int64),
        np.array([4333333329, 4666666662, 4999999995, 5333333328, 5666666661, 5999999994], dtype=np.int64),
        np.array([6333333327, 6666666660, 6999999993, 7333333326, 7666666659, 7999999992], dtype=np.int64),
        np.array([8333333325, 8666666658, 8999999991, 9333333324, 9666666657, 9999999990], dtype=np.int64),
    ]

    assert len(windows) == len(expected_windows)
    for w, expected_w in zip(windows, expected_windows, strict=True):
        np.testing.assert_array_equal(w, expected_w)
