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


def _quaternion_to_rotation_xyzw(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a unit XYZW quaternion into a rotation matrix."""
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _so3_log(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the rotation vector of a near-identity rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * vee
    return angle * vee / (2.0 * np.sin(angle))


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
    angular_velocity_covariance: npt.NDArray[np.float64] | None = None,
    linear_acceleration_covariance: npt.NDArray[np.float64] | None = None,
    with_covariance: bool = False,
) -> ImuData:
    timestamps = (
        np.array([0, _HALF_SECOND_NS, _ONE_SECOND_NS], dtype=np.int64)
        if align_timestamps_ns is None
        else align_timestamps_ns
    )
    sensor_timestamps = timestamps + 100 if sensor_timestamps_ns is None else sensor_timestamps_ns
    zeros = np.zeros((len(timestamps), 3), dtype=np.float64)
    covariance = np.repeat(np.eye(3, dtype=np.float64)[None, :, :] * 0.04, len(timestamps), axis=0)
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
        angular_velocity_covariance=(
            covariance if with_covariance and angular_velocity_covariance is None else angular_velocity_covariance
        ),
        linear_acceleration_covariance=(
            covariance if with_covariance and linear_acceleration_covariance is None else linear_acceleration_covariance
        ),
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
    """Boundary values and covariance weights match the rounded sensor endpoint."""
    align_timestamps = np.array([0, 3], dtype=np.int64)
    sensor_timestamps = np.array([0, 5], dtype=np.int64)
    acceleration = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    bias = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    gyro_covariance = np.array([np.eye(3) * 0.04, np.eye(3) * 0.16], dtype=np.float64)
    accel_covariance = np.zeros((2, 3, 3), dtype=np.float64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=align_timestamps,
            sensor_timestamps_ns=sensor_timestamps,
            linear_acceleration_m_s2=acceleration,
            linear_acceleration_bias_m_s2=bias,
            linear_acceleration_bias_valid=np.ones((2, 3), dtype=np.bool_),
            angular_velocity_covariance=gyro_covariance,
            linear_acceleration_covariance=accel_covariance,
        ),
        np.array([0, 1], dtype=np.int64),
    )

    assert result.sensor_timestamps_ns.tolist() == [0, 2]
    assert result.integration_duration_ns.tolist() == [0, 2]
    np.testing.assert_allclose(result.delta_velocity_m_s[1, 0], 1.2e-9, rtol=0.0, atol=1e-20)
    np.testing.assert_allclose(result.linear_acceleration_bias_used_m_s2[1, 0], 0.6, atol=1e-12)
    assert result.integration_covariance is not None
    np.testing.assert_allclose(result.integration_covariance[1, 0, 0], 1.28e-19, rtol=0.0, atol=1e-30)


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


def test_preintegrate_imu_propagates_measurement_covariance() -> None:
    """Shared midpoint samples contribute once with their combined weight."""
    result = preintegrate_imu(
        _imu_data(with_covariance=True),
        np.array([0, _ONE_SECOND_NS], dtype=np.int64),
    )

    assert result.integration_covariance is not None
    covariance = result.integration_covariance[1]
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)
    assert np.trace(covariance) > 0.0
    np.testing.assert_allclose(covariance[0, 0], 0.015, atol=1e-12)


def test_preintegrate_imu_covariance_preserves_interpolated_source_weights() -> None:
    """Boundary interpolation combines raw-source Jacobians before covariance."""
    timestamps = np.array([0, _ONE_SECOND_NS], dtype=np.int64)
    gyro_covariance = np.zeros((2, 3, 3), dtype=np.float64)
    gyro_covariance[0] = np.eye(3) * 0.04
    gyro_covariance[1] = np.eye(3) * 0.16
    accel_covariance = np.zeros((2, 3, 3), dtype=np.float64)

    result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=timestamps,
            sensor_timestamps_ns=timestamps,
            angular_velocity_covariance=gyro_covariance,
            linear_acceleration_covariance=accel_covariance,
        ),
        np.array([100_000_000, 600_000_000], dtype=np.int64),
    )

    assert result.integration_covariance is not None
    expected = 0.325**2 * 0.04 + 0.175**2 * 0.16
    np.testing.assert_allclose(result.integration_covariance[1, 0, 0], expected, atol=1e-12)


def test_preintegrate_imu_covariance_matches_full_raw_finite_difference() -> None:
    """Analytic covariance matches raw-sample Jacobians for a 3-D interpolated interval."""
    timestamps = np.array([0, 400_000_000, _ONE_SECOND_NS], dtype=np.int64)
    grid = np.array([100_000_000, 900_000_000], dtype=np.int64)
    gyro = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1], [-0.1, 0.3, 0.2]], dtype=np.float64)
    acceleration = np.array([[1.0, 0.2, -0.1], [0.4, -0.3, 0.8], [0.2, 0.5, 0.6]], dtype=np.float64)
    gyro_covariance = np.repeat(np.diag([1e-4, 2e-4, 3e-4])[None], 3, axis=0)
    accel_covariance = np.repeat(np.diag([4e-4, 5e-4, 6e-4])[None], 3, axis=0)
    covariance_result = preintegrate_imu(
        _imu_data(
            align_timestamps_ns=timestamps,
            sensor_timestamps_ns=timestamps,
            angular_velocity_rad_s=gyro,
            linear_acceleration_m_s2=acceleration,
            angular_velocity_covariance=gyro_covariance,
            linear_acceleration_covariance=accel_covariance,
        ),
        grid,
    )
    assert covariance_result.integration_covariance is not None
    nominal_rotation = _quaternion_to_rotation_xyzw(covariance_result.delta_rotation_quat_xyzw[1])
    epsilon = 1e-6
    jacobian = np.zeros((9, 18), dtype=np.float64)

    for raw_index in range(3):
        for measurement_axis in range(6):
            gyro_plus = np.array(gyro, copy=True)
            gyro_minus = np.array(gyro, copy=True)
            accel_plus = np.array(acceleration, copy=True)
            accel_minus = np.array(acceleration, copy=True)
            if measurement_axis < 3:
                gyro_plus[raw_index, measurement_axis] += epsilon
                gyro_minus[raw_index, measurement_axis] -= epsilon
            else:
                axis = measurement_axis - 3
                accel_plus[raw_index, axis] += epsilon
                accel_minus[raw_index, axis] -= epsilon
            plus = preintegrate_imu(
                _imu_data(
                    align_timestamps_ns=timestamps,
                    sensor_timestamps_ns=timestamps,
                    angular_velocity_rad_s=gyro_plus,
                    linear_acceleration_m_s2=accel_plus,
                ),
                grid,
            )
            minus = preintegrate_imu(
                _imu_data(
                    align_timestamps_ns=timestamps,
                    sensor_timestamps_ns=timestamps,
                    angular_velocity_rad_s=gyro_minus,
                    linear_acceleration_m_s2=accel_minus,
                ),
                grid,
            )
            plus_rotation = _quaternion_to_rotation_xyzw(plus.delta_rotation_quat_xyzw[1])
            minus_rotation = _quaternion_to_rotation_xyzw(minus.delta_rotation_quat_xyzw[1])
            column = 6 * raw_index + measurement_axis
            jacobian[:3, column] = (
                _so3_log(nominal_rotation.T @ plus_rotation) - _so3_log(nominal_rotation.T @ minus_rotation)
            ) / (2.0 * epsilon)
            jacobian[3:6, column] = (plus.delta_velocity_m_s[1] - minus.delta_velocity_m_s[1]) / (2.0 * epsilon)
            jacobian[6:9, column] = (plus.delta_position_m[1] - minus.delta_position_m[1]) / (2.0 * epsilon)

    raw_covariance = np.zeros((18, 18), dtype=np.float64)
    for raw_index in range(3):
        start = 6 * raw_index
        raw_covariance[start : start + 3, start : start + 3] = gyro_covariance[raw_index]
        raw_covariance[start + 3 : start + 6, start + 3 : start + 6] = accel_covariance[raw_index]
    expected = jacobian @ raw_covariance @ jacobian.T

    np.testing.assert_allclose(covariance_result.integration_covariance[1], expected, rtol=2e-5, atol=1e-9)


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
