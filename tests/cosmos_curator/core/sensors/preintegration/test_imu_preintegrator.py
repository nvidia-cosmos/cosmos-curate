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
"""Tests for eager IMU preintegration."""

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.data.preintegrated_imu_data import ImuIntegrationInvalidReason
from cosmos_curator.core.sensors.preintegration.imu_preintegrator import preintegrate_imu

_HALF_SECOND_NS = 500_000_000
_ONE_SECOND_NS = 1_000_000_000


def _imu_data(  # noqa: PLR0913
    *,
    align_timestamps_ns: npt.NDArray[np.int64] | None = None,
    sensor_timestamps_ns: npt.NDArray[np.int64] | None = None,
    angular_velocity_rad_s: npt.NDArray[np.float64] | None = None,
    linear_acceleration_m_s2: npt.NDArray[np.float64] | None = None,
    angular_velocity_valid: npt.NDArray[np.bool_] | None = None,
    linear_acceleration_valid: npt.NDArray[np.bool_] | None = None,
    angular_velocity_bias_rad_s: npt.NDArray[np.float64] | None = None,
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64] | None = None,
    angular_velocity_bias_valid: npt.NDArray[np.bool_] | None = None,
    linear_acceleration_bias_valid: npt.NDArray[np.bool_] | None = None,
) -> ImuData:
    timestamps = (
        np.array([0, _HALF_SECOND_NS, _ONE_SECOND_NS], dtype=np.int64)
        if align_timestamps_ns is None
        else align_timestamps_ns
    )
    sensor_timestamps = timestamps + 100 if sensor_timestamps_ns is None else sensor_timestamps_ns
    zeros = np.zeros((len(timestamps), 3), dtype=np.float64)
    return ImuData(
        align_timestamps_ns=timestamps,
        sensor_timestamps_ns=sensor_timestamps,
        angular_velocity_rad_s=zeros if angular_velocity_rad_s is None else angular_velocity_rad_s,
        linear_acceleration_m_s2=zeros if linear_acceleration_m_s2 is None else linear_acceleration_m_s2,
        angular_velocity_valid=angular_velocity_valid,
        linear_acceleration_valid=linear_acceleration_valid,
        angular_velocity_bias_rad_s=angular_velocity_bias_rad_s,
        linear_acceleration_bias_m_s2=linear_acceleration_bias_m_s2,
        angular_velocity_bias_valid=angular_velocity_bias_valid,
        linear_acceleration_bias_valid=linear_acceleration_bias_valid,
    )


def test_preintegrate_imu_integrates_constant_acceleration() -> None:
    """Constant specific force produces analytic velocity and position deltas."""
    acceleration = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(linear_acceleration_m_s2=acceleration),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.delta_rotation_quat_xyzw[1], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_velocity_m_s[1], [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1], [1.0, 0.0, 0.0], atol=1e-12)
    assert result.integration_valid.tolist() == [False, True]
    assert result.sample_count_total.tolist() == [0, 3]
    assert result.max_inter_sample_gap_ns.tolist() == [0, _HALF_SECOND_NS]


def test_preintegrate_imu_uses_sensor_clock_for_physical_time() -> None:
    """Sensor-clock drift controls motion, duration, and maximum sample gap."""
    align_timestamps = np.array([0, _HALF_SECOND_NS, _ONE_SECOND_NS], dtype=np.int64)
    sensor_timestamps = np.array([100, 600_000_100, 1_200_000_100], dtype=np.int64)
    acceleration = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
            linear_acceleration_m_s2=acceleration,
        ),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    assert result.align_timestamps_ns.tolist() == [0, _ONE_SECOND_NS]
    assert result.sensor_timestamps_ns.tolist() == [100, 1_200_000_100]
    assert result.integration_duration_ns.tolist() == [0, 1_200_000_000]
    assert result.max_inter_sample_gap_ns.tolist() == [0, 600_000_000]
    np.testing.assert_allclose(result.delta_velocity_m_s[1], [2.4, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1], [1.44, 0.0, 0.0], atol=1e-12)


def test_preintegrate_imu_uses_rounded_sensor_fraction_for_boundary_payloads() -> None:
    """Boundary values match the rounded sensor endpoint."""
    align_timestamps = np.array([0, 3], dtype=np.int64)
    sensor_timestamps = np.array([0, 5], dtype=np.int64)
    acceleration = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    bias = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
            linear_acceleration_m_s2=acceleration,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=np.ones((2, 3), dtype=np.bool_),
        ),
        np.array([0, 1], dtype=np.int64),
    )

    assert result.sensor_timestamps_ns.tolist() == [0, 2]
    assert result.integration_duration_ns.tolist() == [0, 2]
    np.testing.assert_allclose(result.delta_velocity_m_s[1, 0], 1.2e-9, rtol=0.0, atol=1e-20)
    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1, 0], 0.6, atol=1e-12)


def test_preintegrate_imu_ignores_zero_weight_boundary_sources() -> None:
    """Rounded zero-weight rows do not invalidate measurements or biases."""
    align_timestamps = np.array([0, 3, 6], dtype=np.int64)
    sensor_timestamps = np.array([0, 1, 2], dtype=np.int64)
    validity = np.array([[True, True, True], [True, True, True], [False, False, False]], dtype=np.bool_)
    acceleration = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]], dtype=np.float64)
    bias = np.zeros((3, 3), dtype=np.float64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
            linear_acceleration_m_s2=acceleration,
            linear_acceleration_valid=validity,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=validity,
        ),
        np.array([1, 4], dtype=np.int64),
    )

    assert result.integration_valid.tolist() == [False, True]
    assert result.linear_acceleration_bias_available[1].tolist() == [True, True, True]
    np.testing.assert_allclose(result.delta_velocity_m_s[1], [5e-10, 0.0, 0.0], atol=1e-20)


def test_preintegrate_imu_deduplicates_rounded_boundary_and_interior_sample() -> None:
    """A rounded boundary collision does not create a zero-duration segment."""
    align_timestamps = np.array([0, 100, 200], dtype=np.int64)
    sensor_timestamps = np.array([0, 1, 101], dtype=np.int64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
        ),
        np.array([60, 150], dtype=np.int64),
    )

    assert result.sensor_timestamps_ns.tolist() == [1, 51]
    assert result.integration_duration_ns.tolist() == [0, 50]
    assert result.integration_valid.tolist() == [False, True]


def test_preintegrate_imu_uses_sensor_clock_for_bias_average() -> None:
    """Time-weighted bias reporting uses physical sensor-clock segment lengths."""
    align_timestamps = np.array([0, _HALF_SECOND_NS, _ONE_SECOND_NS], dtype=np.int64)
    sensor_timestamps = np.array([0, 250_000_000, _ONE_SECOND_NS], dtype=np.int64)
    bias = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=np.ones((3, 3), dtype=np.bool_),
        ),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1], [0.75, 0.0, 0.0])
    assert result.max_inter_sample_gap_ns[1] == 750_000_000


def test_preintegrate_imu_localizes_repeated_sensor_timestamp_failure() -> None:
    """A duplicate sensor timestamp invalidates only intervals that contain it."""
    align_timestamps = np.arange(4, dtype=np.int64) * _ONE_SECOND_NS
    sensor_timestamps = np.array([0, _ONE_SECOND_NS, _ONE_SECOND_NS, 2 * _ONE_SECOND_NS], dtype=np.int64)
    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
        ),
        align_timestamps,
    )

    assert result.integration_valid.tolist() == [False, True, False, True]
    assert result.integration_invalid_reason.tolist() == [
        ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value,
        ImuIntegrationInvalidReason.NONE.value,
        ImuIntegrationInvalidReason.NON_INCREASING_SENSOR_TIME.value,
        ImuIntegrationInvalidReason.NONE.value,
    ]
    assert result.integration_duration_ns.tolist() == [0, _ONE_SECOND_NS, 0, _ONE_SECOND_NS]


def test_preintegrate_imu_integrates_constant_angular_velocity() -> None:
    """Constant angular velocity produces the expected SO(3) delta."""
    angular_velocity = np.repeat(np.array([[0.0, 0.0, np.pi / 2.0]], dtype=np.float64), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(angular_velocity_rad_s=angular_velocity),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    expected = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    np.testing.assert_allclose(result.delta_rotation_quat_xyzw[1], expected, atol=1e-12)


def test_preintegrate_imu_rotates_endpoint_accelerations_before_averaging() -> None:
    """Coupled rotation and acceleration use the conventional endpoint-frame midpoint rule."""
    timestamps = np.array([0, _ONE_SECOND_NS], dtype=np.int64)
    angular_velocity = np.repeat(np.array([[0.0, 0.0, np.pi / 2.0]], dtype=np.float64), 2, axis=0)
    acceleration = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), 2, axis=0)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=timestamps,
            sensor_timestamps_ns=timestamps,
            angular_velocity_rad_s=angular_velocity,
            linear_acceleration_m_s2=acceleration,
        ),
        timestamps,
    )

    np.testing.assert_allclose(result.delta_velocity_m_s[1], [0.5, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1], [0.25, 0.25, 0.0], atol=1e-12)


def test_preintegrate_imu_subtracts_available_biases() -> None:
    """Per-axis sensor biases are subtracted and recorded."""
    gyro = np.repeat(np.array([[0.0, 0.0, 2.0]], dtype=np.float64), 3, axis=0)
    gyro_bias = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), 3, axis=0)
    accel = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    accel_bias = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    valid = np.ones((3, 3), dtype=np.bool_)

    result = preintegrate_imu(
        _imu_data(
            angular_velocity_rad_s=gyro,
            linear_acceleration_m_s2=accel,
            angular_velocity_bias_rad_s=gyro_bias,
            linear_acceleration_bias_m_s2=accel_bias,
            angular_velocity_bias_valid=valid,
            linear_acceleration_bias_valid=valid,
        ),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.angular_velocity_bias_used_rad_s[1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1], [1.0, 0.0, 0.0])
    assert result.angular_velocity_bias_available[1].tolist() == [True, True, True]
    assert result.linear_acceleration_bias_available[1].tolist() == [True, True, True]
    np.testing.assert_allclose(result.delta_rotation_quat_xyzw[1, 2], np.sin(0.5), atol=1e-12)


def test_preintegrate_imu_uses_zero_for_unavailable_bias() -> None:
    """An absent bias does not invalidate integration or claim availability."""
    result = preintegrate_imu(
        _imu_data(),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_array_equal(result.angular_velocity_bias_used_rad_s[1], np.zeros(3))
    np.testing.assert_array_equal(result.linear_acceleration_bias_used_m_s2[1], np.zeros(3))
    assert not np.any(result.angular_velocity_bias_available[1])
    assert not np.any(result.linear_acceleration_bias_available[1])
    assert result.integration_valid[1]


def test_preintegrate_imu_rejects_interval_with_invalid_measurement() -> None:
    """Invalid source axes invalidate the interval and preserve sample accounting."""
    valid = np.ones((3, 3), dtype=np.bool_)
    valid[1, 0] = False

    result = preintegrate_imu(
        _imu_data(angular_velocity_valid=valid),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    assert not result.integration_valid[1]
    reason = ImuIntegrationInvalidReason(int(result.integration_invalid_reason[1]))
    assert ImuIntegrationInvalidReason.INVALID_MEASUREMENT in reason
    assert result.sample_count_total[1] == 3
    assert result.sample_count_used[1] == 2
    assert result.sample_count_rejected[1] == 1


def test_preintegrate_imu_requires_boundary_support() -> None:
    """An interval outside the source timeline is marked invalid."""
    result = preintegrate_imu(
        _imu_data(),
        np.array([-1, _HALF_SECOND_NS], dtype=np.int64),
    )

    assert result.sensor_timestamps_ns.tolist() == [100, _HALF_SECOND_NS + 100]
    assert result.integration_invalid_reason.tolist() == [
        ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value,
        ImuIntegrationInvalidReason.MISSING_BOUNDARY_SUPPORT.value,
    ]


def test_preintegrate_imu_interpolates_alignment_boundaries() -> None:
    """Grid boundaries between raw samples preserve constant-motion deltas."""
    acceleration = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(linear_acceleration_m_s2=acceleration),
        np.array([_HALF_SECOND_NS // 2, _HALF_SECOND_NS + _HALF_SECOND_NS // 2], dtype=np.int64),
    )

    np.testing.assert_allclose(result.delta_velocity_m_s[1], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1], [0.25, 0.0, 0.0], atol=1e-12)
    assert result.sample_count_total[1] == 3


def test_preintegrate_imu_reports_full_raw_support_gap() -> None:
    """Clipped boundaries do not hide the raw sensor gap supporting interpolation."""
    timestamps = np.array([0, 10 * _ONE_SECOND_NS], dtype=np.int64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=timestamps,
            sensor_timestamps_ns=timestamps,
        ),
        np.array([4 * _ONE_SECOND_NS, 6 * _ONE_SECOND_NS], dtype=np.int64),
    )

    assert result.integration_duration_ns[1] == 2 * _ONE_SECOND_NS
    assert result.max_inter_sample_gap_ns[1] == 10 * _ONE_SECOND_NS


def test_preintegrate_imu_applies_bias_per_available_axis() -> None:
    """Unavailable bias axes use zero while valid axes remain corrected."""
    acceleration = np.repeat(np.array([[2.0, 2.0, 2.0]], dtype=np.float64), 3, axis=0)
    bias = np.ones((3, 3), dtype=np.float64)
    validity = np.repeat(np.array([[True, False, True]], dtype=np.bool_), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(
            linear_acceleration_m_s2=acceleration,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=validity,
        ),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(result.delta_velocity_m_s[1], [1.0, 2.0, 1.0])
    assert result.linear_acceleration_bias_available[1].tolist() == [True, False, True]


def test_preintegrate_imu_applies_bias_only_where_each_sample_is_valid() -> None:
    """A validity transition retains the valid endpoint correction without claiming full availability."""
    acceleration = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    bias = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)
    validity = np.array(
        [[True, False, False], [False, False, False], [False, False, False]],
        dtype=np.bool_,
    )

    result = preintegrate_imu(
        _imu_data(
            linear_acceleration_m_s2=acceleration,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=validity,
        ),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.delta_velocity_m_s[1], [1.75, 0.0, 0.0])
    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1], [0.25, 0.0, 0.0])
    assert not result.linear_acceleration_bias_available[1, 0]


def test_preintegrate_imu_preserves_large_sensor_timestamp_precision() -> None:
    """Interpolated epoch-scale sensor timestamps retain integer nanosecond precision."""
    align_timestamps = np.array([0, 100], dtype=np.int64)
    sensor_start = 1_700_000_000_000_000_001
    sensor_timestamps = np.array([sensor_start, sensor_start + 100], dtype=np.int64)
    zeros = np.zeros((2, 3), dtype=np.float64)
    imu_data = ImuData(
        align_timestamps_ns=align_timestamps,
        sensor_timestamps_ns=sensor_timestamps,
        angular_velocity_rad_s=zeros,
        linear_acceleration_m_s2=zeros,
    )

    result = preintegrate_imu(imu_data, np.array([0, 50], dtype=np.int64))

    assert result.sensor_timestamps_ns[1] == sensor_start + 50


def test_preintegrate_imu_adjacent_intervals_do_not_duplicate_elapsed_time() -> None:
    """Adjacent half-open intervals each integrate only their own duration."""
    acceleration = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), 3, axis=0)

    result = preintegrate_imu(
        _imu_data(linear_acceleration_m_s2=acceleration),
        np.array([0, _HALF_SECOND_NS, _ONE_SECOND_NS], dtype=np.int64),
    )

    np.testing.assert_allclose(result.delta_velocity_m_s[1:, 0], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1:, 0], [0.25, 0.25], atol=1e-12)
    assert result.integration_duration_ns.tolist() == [0, _HALF_SECOND_NS, _HALF_SECOND_NS]
    assert result.sample_count_total.tolist() == [0, 2, 2]


def test_preintegrate_imu_accepts_empty_alignment_grid() -> None:
    """An empty requested grid produces a structurally valid empty result."""
    result = preintegrate_imu(_imu_data(), np.empty(0, dtype=np.int64))

    assert len(result.align_timestamps_ns) == 0
    assert result.delta_rotation_quat_xyzw.shape == (0, 4)
    assert result.delta_velocity_m_s.shape == (0, 3)
