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
"""Tests for eager IMU preintegration data contracts."""

from typing import Any, cast

import attrs
import numpy as np
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.preintegrated_imu_data import (
    ImuIntegrationInvalidReason,
    PreintegratedImuData,
)
from cosmos_curator.core.sensors.data.sensor_data import SensorData


def _make_preintegrated_data(**overrides: object) -> PreintegratedImuData:
    """Build a valid two-row preintegration batch."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "align_interval_start_timestamps_ns": np.array([100, 100], dtype=np.int64),
        "align_interval_end_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "integration_duration_ns": np.array([0, 100], dtype=np.int64),
        "delta_rotation_quat_xyzw": np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "delta_velocity_m_s": np.zeros((2, 3), dtype=np.float64),
        "delta_position_m": np.zeros((2, 3), dtype=np.float64),
        "angular_velocity_bias_used_rad_s": np.zeros((2, 3), dtype=np.float64),
        "linear_acceleration_bias_used_m_s2": np.zeros((2, 3), dtype=np.float64),
        "angular_velocity_bias_available": np.ones((2, 3), dtype=np.bool_),
        "linear_acceleration_bias_available": np.array(
            [[True, True, False], [True, True, False]],
            dtype=np.bool_,
        ),
        "sample_count_total": np.array([0, 2], dtype=np.uint32),
        "sample_count_used": np.array([0, 2], dtype=np.uint32),
        "sample_count_rejected": np.array([0, 0], dtype=np.uint32),
        "max_inter_sample_gap_ns": np.array([0, 100], dtype=np.int64),
        "integration_valid": np.array([False, True], dtype=np.bool_),
        "integration_invalid_reason": np.array(
            [ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value, ImuIntegrationInvalidReason.NONE.value],
            dtype=np.uint32,
        ),
    }
    values.update(overrides)
    return PreintegratedImuData(**values)


def test_preintegrated_imu_data_satisfies_sensor_data_and_aligned_frame() -> None:
    """Preintegrated data should satisfy the existing aligned sensor contract."""
    data = _make_preintegrated_data()
    sensor_data: SensorData = data

    frame = AlignedFrame(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_data={"imu0": cast("SensorData", data)},
    )

    assert frame["imu0"] is data
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([100, 200], dtype=np.int64))


def test_preintegrated_imu_data_accepts_valid_sliced_first_row() -> None:
    """A window slice may begin with an already-integrated interval."""
    full_data = _make_preintegrated_data()
    sliced_values: dict[str, object] = {}
    for field in attrs.fields(type(full_data)):
        value = getattr(full_data, field.name)
        sliced_values[field.name] = None if value is None else value[1:]

    sliced = PreintegratedImuData(**sliced_values)

    assert sliced.integration_valid.tolist() == [True]
    assert sliced.align_interval_start_timestamps_ns[0] < sliced.align_interval_end_timestamps_ns[0]
    assert sliced.integration_duration_ns[0] > 0


def test_preintegrated_imu_data_duration_uses_sensor_clock() -> None:
    """Physical duration may differ from the external alignment-clock interval."""
    data = _make_preintegrated_data(
        sensor_timestamps_ns=np.array([1_000, 1_250], dtype=np.int64),
        integration_duration_ns=np.array([0, 250], dtype=np.int64),
    )

    assert data.align_interval_start_timestamps_ns.tolist() == [100, 100]
    assert data.align_interval_end_timestamps_ns.tolist() == [100, 200]
    assert data.integration_duration_ns.tolist() == [0, 250]


def test_preintegrated_imu_data_rejects_valid_zero_duration() -> None:
    """A physically valid interval must advance the sensor clock."""
    with pytest.raises(ValueError, match="Valid integration intervals require positive integration_duration_ns"):
        _make_preintegrated_data(
            sensor_timestamps_ns=np.array([100, 100], dtype=np.int64),
            integration_duration_ns=np.array([0, 0], dtype=np.int64),
        )


def test_preintegrated_imu_data_accepts_invalid_zero_duration() -> None:
    """A repeated sensor-time interval remains representable as an invalid row."""
    data = _make_preintegrated_data(
        sensor_timestamps_ns=np.array([100, 100], dtype=np.int64),
        integration_duration_ns=np.array([0, 0], dtype=np.int64),
        integration_valid=np.array([False, False], dtype=np.bool_),
        integration_invalid_reason=np.array(
            [
                ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value,
                ImuIntegrationInvalidReason.NON_INCREASING_SENSOR_TIME.value,
            ],
            dtype=np.uint32,
        ),
    )

    assert not data.integration_valid[1]
    assert data.integration_duration_ns[1] == 0


def test_preintegrated_imu_data_rejects_first_alignment_reason_on_sliced_row() -> None:
    """A sliced interval must not be mislabeled as the full-grid identity row."""
    full_data = _make_preintegrated_data()
    sliced_values: dict[str, object] = {}
    for field in attrs.fields(type(full_data)):
        value = getattr(full_data, field.name)
        sliced_values[field.name] = None if value is None else value[1:]
    sliced_values["integration_valid"] = np.array([False], dtype=np.bool_)
    sliced_values["integration_invalid_reason"] = np.array(
        [ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value],
        dtype=np.uint32,
    )

    with pytest.raises(ValueError, match="sliced interval row must not include FIRST_ALIGNMENT"):
        PreintegratedImuData(**sliced_values)


def test_preintegrated_imu_data_accepts_optional_motion_covariance() -> None:
    """Motion covariance should be optional when ImuData lacks measurement covariance."""
    without_covariance = _make_preintegrated_data()
    covariance = np.zeros((2, 9, 9), dtype=np.float64)
    with_covariance = _make_preintegrated_data(integration_covariance=covariance)

    assert without_covariance.integration_covariance is None
    np.testing.assert_array_equal(with_covariance.integration_covariance, covariance)


def test_preintegrated_imu_data_arrays_are_readonly_without_mutating_callers() -> None:
    """Every output array should be a read-only view over caller-owned storage."""
    delta_velocity = np.zeros((2, 3), dtype=np.float64)
    data = _make_preintegrated_data(delta_velocity_m_s=delta_velocity)

    assert delta_velocity.flags.writeable is True
    assert data.delta_velocity_m_s.flags.writeable is False
    assert np.shares_memory(data.delta_velocity_m_s, delta_velocity)
    for field in attrs.fields(type(data)):
        value = getattr(data, field.name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = value.flat[0]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("delta_velocity_m_s", np.zeros((2, 2), dtype=np.float64), r"shape \(N, 3\)"),
        ("delta_position_m", np.zeros((2, 3), dtype=np.float32), "dtype float64"),
        ("delta_rotation_quat_xyzw", np.zeros((2, 4), dtype=np.float64), "unit norm"),
        ("angular_velocity_bias_available", np.ones((2, 3), dtype=np.uint8), "dtype bool"),
        ("integration_covariance", np.zeros((2, 15, 15), dtype=np.float64), r"shape \(N, 9, 9\)"),
        (
            "integration_covariance",
            np.tile(np.diag([1.0] * 8 + [-1.0]), (2, 1, 1)),
            "positive semidefinite",
        ),
    ],
)
def test_preintegrated_imu_data_rejects_invalid_array_contracts(
    field_name: str,
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    """Data arrays should enforce their documented shapes, dtypes, and numeric invariants."""
    with pytest.raises(ValueError, match=match):
        _make_preintegrated_data(**{field_name: value})


def test_preintegrated_imu_data_rejects_length_mismatch() -> None:
    """Every present array should share the alignment batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_preintegrated_data(sample_count_total=np.array([0], dtype=np.uint32))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"align_interval_start_timestamps_ns": np.array([100, 101], dtype=np.int64)},
            "previous alignment timestamp",
        ),
        (
            {"align_interval_end_timestamps_ns": np.array([100, 201], dtype=np.int64)},
            "must equal align_timestamps_ns",
        ),
        (
            {"integration_duration_ns": np.array([0, 99], dtype=np.int64)},
            "must equal adjacent sensor timestamp differences",
        ),
        (
            {
                "sample_count_used": np.array([0, 1], dtype=np.uint32),
                "sample_count_rejected": np.array([0, 0], dtype=np.uint32),
            },
            "must equal sample_count_total",
        ),
        (
            {"max_inter_sample_gap_ns": np.array([0, -1], dtype=np.int64)},
            "must be nonnegative",
        ),
    ],
)
def test_preintegrated_imu_data_rejects_invalid_interval_accounting(
    overrides: dict[str, object],
    match: str,
) -> None:
    """Interval boundaries, durations, sample counts, and gaps should be self-consistent."""
    with pytest.raises(ValueError, match=match):
        _make_preintegrated_data(**overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"integration_valid": np.array([False, False], dtype=np.bool_)},
            "exactly when integration_invalid_reason is NONE",
        ),
        (
            {
                "sample_count_total": np.array([0, 1], dtype=np.uint32),
                "sample_count_used": np.array([0, 1], dtype=np.uint32),
            },
            "at least two used samples",
        ),
        (
            {
                "delta_rotation_quat_xyzw": np.array(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            },
            "identity quaternion",
        ),
        (
            {
                "sample_count_total": np.array([1, 2], dtype=np.uint32),
                "sample_count_rejected": np.array([1, 0], dtype=np.uint32),
            },
            "first row must not contain integrated samples",
        ),
    ],
)
def test_preintegrated_imu_data_rejects_invalid_validity_or_identity_row(
    overrides: dict[str, object],
    match: str,
) -> None:
    """Validity reasons and the first identity row should remain unambiguous."""
    with pytest.raises(ValueError, match=match):
        _make_preintegrated_data(**overrides)
