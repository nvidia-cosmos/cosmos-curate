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
"""Test sampling utilities for the sensor library."""

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.sampler import (
    find_closest_indices,
    sample_window_indices,
)
from tests.cosmos_curator.core.sensors.test_utils import EPOCH_ODD_NS


def _window_from_grid(grid: npt.NDArray[np.int64]) -> SamplingWindow:
    if len(grid) == 0:
        return SamplingWindow(start_ns=0, exclusive_end_ns=0, timestamps_ns=np.array([], dtype=np.int64))
    return SamplingWindow(start_ns=grid[0], exclusive_end_ns=grid[-1], timestamps_ns=grid[:-1])


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_indices"),
    [
        # Exact matches.
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ),
        # Standard nearest-neighbour matching.
        (
            np.array([0, 10, 20, 30, 40], dtype=np.int64),
            np.array([6, 26], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        ),
        # Returned indices refer to positions in the original canonical array.
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([150, 350], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
        ),
    ],
)
def test_find_closest_indices_core_contract(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
) -> None:
    """find_closest_indices should return nearest-neighbour indices into the original canonical array."""
    result = find_closest_indices(canonical, grid)

    np.testing.assert_array_equal(result, expected_indices)
    assert np.all(result >= 0)
    assert np.all(result < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_indices"),
    [
        # Midpoint ties go left.
        (
            np.array([0, 10], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
        # Values before the first canonical snap to the first index.
        (
            np.array([10, 20], dtype=np.int64),
            np.array([0, 5], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        ),
        # Values after the last canonical snap to the last index.
        (
            np.array([10, 20], dtype=np.int64),
            np.array([25, 30], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ),
        # A single-element canonical maps every grid point to index 0.
        (
            np.array([10], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 0, 0], dtype=np.int64),
        ),
    ],
)
def test_find_closest_indices_boundary_and_tie_behavior(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
) -> None:
    """find_closest_indices should handle ties and out-of-range values consistently."""
    result = find_closest_indices(canonical, grid)

    np.testing.assert_array_equal(result, expected_indices)
    assert np.all(result >= 0)
    assert np.all(result < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "match"),
    [
        (
            np.array([], dtype=np.int64),
            np.array([0, 10, 20], dtype=np.int64),
            "canonical must be non-empty",
        ),
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([], dtype=np.int64),
            "grid must be non-empty",
        ),
        (
            np.array([0, 10, 10, 20], dtype=np.int64),
            np.array([0, 5, 15], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([0, 10, 20], dtype=np.int64),
            np.array([0, 10, 10], dtype=np.int64),
            "grid must be strictly sorted in ascending order with no duplicates",
        ),
    ],
)
def test_find_closest_indices_input_validation(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    match: str,
) -> None:
    """find_closest_indices should reject invalid canonical and grid inputs."""
    with pytest.raises(ValueError, match=match):
        find_closest_indices(canonical, grid)


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Basic matching.
        # Why:
        # - active reference timestamps are 150, 260
        # - 150 is equidistant from 100 and 200, so the left one wins
        # - 260 is nearest to 300
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([100, 300], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Boundary marker is not sampled: 300 ends the window, so only 150 and 250
        # get rows.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 300], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # A canonical timestamp equal to the exclusive end is still selectable.
        (
            np.array([150, 250, 350], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Repeated picks collapse under dedup: 110 and 140 both take 100.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
            True,
        ),
        # Repeated picks are preserved when dedup is disabled.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([100, 100, 200], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Returned indices refer to the original canonical array.
        (
            np.array([50, 150, 250, 350, 450], dtype=np.int64),
            np.array([140, 260, 340], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_core_contract(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """Test the core contract of sample_window_indices."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=dedup,
    )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Left boundary is inclusive: canonical == grid[0] is eligible.
        (
            np.array([100, 200], dtype=np.int64),
            np.array([100, 150, 250], dtype=np.int64),
            np.array([100, 100], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Right boundary is exclusive: canonical == grid[-1] is not eligible.
        (
            np.array([150, 250, 350], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # grid[-1] is a boundary marker, not an active reference timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 300], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # A singleton window has no active reference timestamps.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            True,
        ),
        # A two-element window has exactly one active reference timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([100], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_half_open_window_semantics(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should obey the half-open window contract."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=dedup,
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts"),
    [
        # A canonical timestamp before the window is selected when it is nearest.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 260, 350], dtype=np.int64),
            np.array([100, 300], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ),
        # A window past the end of the timeline still emits a row per reference
        # timestamp, taking the nearest canonical timestamp that exists.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([400, 500], dtype=np.int64),
            np.array([300], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ),
        # An in-window canonical timestamp still wins when it is the nearest one.
        (
            np.array([100, 200, 300, 400], dtype=np.int64),
            np.array([180, 220, 280], dtype=np.int64),
            np.array([200], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ),
    ],
)
def test_sample_window_indices_ignores_window_bounds_for_eligibility(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
) -> None:
    """Window bounds select reference timestamps, not eligible canonical timestamps."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(canonical=canonical, window=window, policy=NearestTimestampPolicy())

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Exact matches are preserved.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([100, 200, 300, 350], dtype=np.int64),
            np.array([100, 200, 300], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Left/right nearest choice.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([160, 260, 350], dtype=np.int64),
            np.array([200, 300], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Midpoint ties resolve to the left canonical timestamp.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 250, 350], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
        # Values beyond the last canonical timestamp snap to it.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([260, 340], dtype=np.int64),
            np.array([300], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            False,
        ),
        # A tie and a clear win against the same neighbouring pair.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([150, 170, 350], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            False,
        ),
    ],
)
def test_sample_window_indices_nearest_neighbour_selection(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should perform nearest-neighbour matching within the eligible window-local subset."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=dedup,
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


def test_sample_window_indices_returns_original_indices_for_sidecar_arrays() -> None:
    """Returned indices should address the original canonical-aligned sidecar arrays, not the filtered subset."""
    canonical = np.array([50, 150, 250, 350, 450], dtype=np.int64)
    pts_stream = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
    grid = np.array([140, 260, 340], dtype=np.int64)
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=False,
    )

    # The eligible canonical subset is [150, 250], but the returned indices
    # must still refer to positions in the original canonical / pts_stream arrays.
    np.testing.assert_array_equal(indices, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(canonical[indices], np.array([150, 250], dtype=np.int64))
    np.testing.assert_array_equal(pts_stream[indices], np.array([1500, 2500], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_max_delta_passes() -> None:
    """Matches within max_delta_ns should pass."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 205, 350], dtype=np.int64)
    window = _window_from_grid(grid)
    policy = NearestTimestampPolicy(max_delta_ns=50)

    indices, counts = sample_window_indices(canonical=canonical, window=window, policy=policy, dedup=False)

    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(canonical[indices], np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_max_delta_none_disables_delta_check() -> None:
    """max_delta_ns=None requests nearest selection without a maximum-delta constraint."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 260, 350], dtype=np.int64)
    window = _window_from_grid(grid)

    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(max_delta_ns=None),
        dedup=False,
    )

    np.testing.assert_array_equal(indices, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(canonical[indices], np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_zero_max_delta_requires_exact_match() -> None:
    """max_delta_ns=0 should require exact timestamp matches."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([100, 201, 300], dtype=np.int64)
    window = _window_from_grid(grid)

    with pytest.raises(
        AlignmentError, match=r"max_delta_ns=0 exceeded: max delta was 1 ns for grid=201, canonical=200"
    ):
        sample_window_indices(canonical=canonical, window=window, policy=NearestTimestampPolicy(max_delta_ns=0))


def test_sample_window_indices_rejects_no_sampling_policy() -> None:
    """Nearest selection requires a nearest-timestamp policy."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    window = _window_from_grid(np.array([100, 200], dtype=np.int64))

    with pytest.raises(TypeError, match="policy must be NearestTimestampPolicy, got NoSamplingPolicy"):
        sample_window_indices(canonical=canonical, window=window, policy=NoSamplingPolicy())  # type: ignore[arg-type]


def test_sample_window_indices_max_delta_raises_with_offending_pair() -> None:
    """A max-delta failure should report the offending grid and canonical timestamps."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 260, 350], dtype=np.int64)
    window = _window_from_grid(grid)
    policy = NearestTimestampPolicy(max_delta_ns=30)

    with pytest.raises(
        AlignmentError, match=r"max_delta_ns=30 exceeded: max delta was 50 ns for grid=150, canonical=100"
    ):
        sample_window_indices(canonical=canonical, window=window, policy=policy)


def test_sample_window_indices_max_delta_failure_carries_structured_fields() -> None:
    """A tolerance failure is diagnosable from the exception without re-deriving the selection."""
    canonical = np.array([100, 200, 300], dtype=np.int64)
    grid = np.array([150, 260, 350], dtype=np.int64)
    window = _window_from_grid(grid)

    with pytest.raises(AlignmentError) as caught:
        sample_window_indices(
            canonical=canonical,
            window=window,
            policy=NearestTimestampPolicy(max_delta_ns=30),
        )

    error = caught.value
    assert error.reason is AlignmentFailureReason.TOLERANCE_EXCEEDED
    assert error.max_delta_ns == 30
    # The sampler has no sensor id to give; SensorGroup supplies its configured one.
    assert error.sensor_id is None
    np.testing.assert_array_equal(error.align_timestamps_ns, window.timestamps_ns)

    # One selected timestamp per reference timestamp, so a caller can see which
    # pairing broke the tolerance and re-derive the delta from the fields alone.
    # Asserted as a relationship rather than fixed values: which observation is
    # nearest is the selection rule's business, not this contract's.
    assert error.sensor_timestamps_ns is not None
    assert len(error.sensor_timestamps_ns) == len(window.timestamps_ns)
    worst = int(np.abs(error.sensor_timestamps_ns - error.align_timestamps_ns).max())
    assert error.delta_ns == worst
    assert error.delta_ns > error.max_delta_ns


@pytest.mark.parametrize(
    ("canonical", "grid", "match"),
    [
        (
            np.array([], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be non-empty",
        ),
        (
            np.array([100, 200, 150], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
        (
            np.array([100, 200, 200, 300], dtype=np.int64),
            np.array([100, 200], dtype=np.int64),
            "canonical must be strictly sorted in ascending order with no duplicates",
        ),
    ],
)
def test_sample_window_indices_input_validation(
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    match: str,
) -> None:
    """sample_window_indices should reject invalid canonical and grid inputs."""
    window = _window_from_grid(grid)
    with pytest.raises(ValueError, match=match):
        sample_window_indices(canonical=canonical, window=window, policy=NearestTimestampPolicy())


@pytest.mark.parametrize(
    ("canonical", "grid", "expected_canonical", "expected_indices", "expected_counts", "dedup"),
    [
        # Supersampling: more reference timestamps than canonical timestamps.
        (
            np.array([100, 200, 300], dtype=np.int64),
            np.array([110, 140, 240, 350], dtype=np.int64),
            np.array([100, 100, 200], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            False,
        ),
        # Subsampling: fewer reference timestamps than canonical timestamps.
        (
            np.array([100, 150, 200, 250, 300], dtype=np.int64),
            np.array([140, 260, 350], dtype=np.int64),
            np.array([150, 250], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
        # Jittery canonical timestamps sampled against a regular grid.
        (
            np.array([101, 199, 301, 399], dtype=np.int64),
            np.array([110, 210, 310, 410], dtype=np.int64),
            np.array([101, 199, 301], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 1, 1], dtype=np.int64),
            True,
        ),
        # Sparse canonical timestamps inside a wider window.
        (
            np.array([100, 300], dtype=np.int64),
            np.array([110, 210, 310, 410], dtype=np.int64),
            np.array([100, 300], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            True,
        ),
        # Dense canonical timestamps with only a few active reference timestamps.
        (
            np.array([100, 120, 140, 160, 180, 200, 220, 240, 260], dtype=np.int64),
            np.array([115, 235, 300], dtype=np.int64),
            np.array([120, 240], dtype=np.int64),
            np.array([1, 7], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            True,
        ),
    ],
)
def test_sample_window_indices_shape_and_density_cases(  # noqa: PLR0913
    canonical: npt.NDArray[np.int64],
    grid: npt.NDArray[np.int64],
    expected_canonical: npt.NDArray[np.int64],
    expected_indices: npt.NDArray[np.int64],
    expected_counts: npt.NDArray[np.int64],
    *,
    dedup: bool,
) -> None:
    """sample_window_indices should behave predictably across realistic grid/canonical density patterns."""
    window = _window_from_grid(grid)
    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=dedup,
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(canonical[indices], expected_canonical)
    np.testing.assert_array_equal(counts, expected_counts)

    assert np.all(indices >= 0)
    assert np.all(indices < len(canonical))


def test_sampler_selection_is_origin_invariant() -> None:
    """Shifting canonical and grid timestamps to an epoch origin should not change what gets selected."""
    canonical = np.array([0, 90_000_000, 210_000_000, 290_000_000, 400_000_000], dtype=np.int64)
    grid = np.array([0, 100_000_000, 200_000_000, 300_000_000, 400_000_000], dtype=np.int64)

    base_closest = find_closest_indices(canonical, grid)
    shifted_closest = find_closest_indices(canonical + EPOCH_ODD_NS, grid + EPOCH_ODD_NS)

    np.testing.assert_array_equal(shifted_closest, base_closest)

    base_indices, base_counts = sample_window_indices(
        canonical=canonical,
        window=_window_from_grid(grid),
        policy=NearestTimestampPolicy(),
    )
    shifted_indices, shifted_counts = sample_window_indices(
        canonical=canonical + EPOCH_ODD_NS,
        window=_window_from_grid(grid + EPOCH_ODD_NS),
        policy=NearestTimestampPolicy(),
    )

    # Indices and counts address the same canonical slots, so they are identical rather than shifted.
    np.testing.assert_array_equal(shifted_indices, base_indices)
    np.testing.assert_array_equal(shifted_counts, base_counts)


def _select_over_grid(
    canonical: npt.NDArray[np.int64],
    grid: SamplingGrid,
) -> tuple[list[int], list[int]]:
    """Concatenate per-window sampler output, as a pipeline consuming batches would.

    Returns ``(reference_timestamps, selected_canonical_timestamps)``.
    """
    reference: list[int] = []
    selected: list[int] = []
    for window in grid:
        if len(window) == 0:
            continue
        indices, _counts = sample_window_indices(
            canonical=canonical,
            window=window,
            policy=NearestTimestampPolicy(),
            dedup=False,
        )
        reference.extend(int(value) for value in window.timestamps_ns)
        selected.extend(int(value) for value in canonical[indices])
    return reference, selected


_INVARIANCE_CANONICAL = np.array(
    [100, 900, 1900, 2800, 4100, 5100, 5900, 6800, 8100, 9100],
    dtype=np.int64,
)
_INVARIANCE_GRID_TS = np.arange(0, 10_000, 1000, dtype=np.int64)


@pytest.mark.parametrize("duration_ns", [1000, 2000, 3000, 5000, 10_000])
def test_sample_window_indices_selection_is_invariant_to_window_size(duration_ns: int) -> None:
    """Splitting one span into more windows must not change which timestamps are selected."""
    whole_span = SamplingGrid(
        start_ns=0,
        exclusive_end_ns=10_000,
        timestamps_ns=_INVARIANCE_GRID_TS,
        stride_ns=10_000,
        duration_ns=10_000,
    )
    subdivided = SamplingGrid(
        start_ns=0,
        exclusive_end_ns=10_000,
        timestamps_ns=_INVARIANCE_GRID_TS,
        stride_ns=duration_ns,
        duration_ns=duration_ns,
    )

    expected_reference, expected_selected = _select_over_grid(_INVARIANCE_CANONICAL, whole_span)
    reference, selected = _select_over_grid(_INVARIANCE_CANONICAL, subdivided)

    assert reference == expected_reference
    assert selected == expected_selected


def test_sample_window_indices_reaches_across_the_window_edge() -> None:
    """A reference timestamp must select its nearest canonical timestamp even across a window edge."""
    canonical = np.array([1900, 2800, 4100], dtype=np.int64)
    window = SamplingWindow(
        start_ns=2000,
        exclusive_end_ns=4000,
        timestamps_ns=np.array([2000, 3000], dtype=np.int64),
    )

    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=False,
    )

    # 1900 is 100 ns from the reference timestamp 2000 and 2800 is 800 ns from it.
    np.testing.assert_array_equal(canonical[indices], np.array([1900, 2800], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_selection_depends_only_on_the_reference_timestamp() -> None:
    """The same reference timestamp must select the same canonical timestamp in any window."""
    canonical = np.array([1900, 2800, 4100], dtype=np.int64)
    reference_ts = np.array([2000], dtype=np.int64)

    picks = set()
    for start_ns, exclusive_end_ns in ((2000, 4000), (1500, 2500), (0, 10_000), (2000, 2001)):
        window = SamplingWindow(
            start_ns=start_ns,
            exclusive_end_ns=exclusive_end_ns,
            timestamps_ns=reference_ts,
        )
        indices, _counts = sample_window_indices(
            canonical=canonical,
            window=window,
            policy=NearestTimestampPolicy(),
            dedup=False,
        )
        picks.add(int(canonical[indices][0]))

    assert picks == {1900}


def test_sample_window_indices_emits_one_row_per_reference_timestamp() -> None:
    """A window whose span holds no canonical timestamp must still emit a row per reference timestamp."""
    canonical = np.array([100, 9500], dtype=np.int64)
    window = SamplingWindow(
        start_ns=4000,
        exclusive_end_ns=6000,
        timestamps_ns=np.array([4000, 5000], dtype=np.int64),
    )

    indices, counts = sample_window_indices(
        canonical=canonical,
        window=window,
        policy=NearestTimestampPolicy(),
        dedup=False,
    )

    assert len(indices) == len(window)
    np.testing.assert_array_equal(canonical[indices], np.array([100, 9500], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))


def test_sample_window_indices_max_delta_raises_when_nothing_is_within_tolerance() -> None:
    """A reference timestamp stranded in a dropout must raise rather than silently emit nothing."""
    canonical = np.array([100, 9500], dtype=np.int64)
    window = SamplingWindow(
        start_ns=4000,
        exclusive_end_ns=6000,
        timestamps_ns=np.array([4000, 5000], dtype=np.int64),
    )

    with pytest.raises(AlignmentError, match="max_delta_ns=500 exceeded"):
        sample_window_indices(
            canonical=canonical,
            window=window,
            policy=NearestTimestampPolicy(max_delta_ns=500),
        )


def test_sample_window_indices_max_delta_ignores_window_size() -> None:
    """Subdividing a span must not make a clip that satisfies max_delta_ns start raising."""
    policy = NearestTimestampPolicy(max_delta_ns=500)
    for duration_ns in (10_000, 2000, 1000):
        grid = SamplingGrid(
            start_ns=0,
            exclusive_end_ns=10_000,
            timestamps_ns=_INVARIANCE_GRID_TS,
            stride_ns=duration_ns,
            duration_ns=duration_ns,
        )
        for window in grid:
            if len(window) == 0:
                continue
            sample_window_indices(canonical=_INVARIANCE_CANONICAL, window=window, policy=policy)


def test_closest_index_is_not_fooled_by_a_wrapping_distance() -> None:
    """Distances are compared as integers, not as int64 arithmetic that wraps.

    A subtraction that overflows makes the furthest candidate look adjacent, so
    the sampler selects the wrong source observation and reports no error.
    """
    limits = np.iinfo(np.int64)
    canonical = np.array([limits.min, 0], dtype=np.int64)

    closest = find_closest_indices(canonical, np.array([limits.max - 1], dtype=np.int64))

    assert int(closest[0]) == 1
