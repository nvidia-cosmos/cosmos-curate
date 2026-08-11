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
"""Tests for sensor-library ``ImuData``."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.data.sensor_data import SensorData


def _make_imu_data(**overrides: object) -> ImuData:
    """Build a minimal valid ImuData batch."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([90, 210], dtype=np.int64),
        "angular_velocity_rad_s": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        "linear_acceleration_m_s2": np.array([[1.0, 2.0, 9.8], [1.1, 2.1, 9.7]], dtype=np.float64),
    }
    values.update(overrides)
    return ImuData(**values)


def test_imu_data_accepts_required_fields() -> None:
    """ImuData should accept a minimal required-field batch."""
    imu_data = _make_imu_data()

    assert len(imu_data.align_timestamps_ns) == 2
    np.testing.assert_array_equal(imu_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


def test_imu_data_accepts_bias_vectors_and_validity() -> None:
    """ImuData should preserve optional three-axis bias estimates and validity."""
    angular_bias = np.array([[0.01, 0.02, 0.03], [np.nan, 0.05, 0.06]], dtype=np.float64)
    acceleration_bias = np.array([[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]], dtype=np.float64)
    angular_valid = np.array([[True, True, True], [False, True, True]], dtype=np.bool_)
    acceleration_valid = np.array([[True, True, True], [True, False, True]], dtype=np.bool_)

    imu_data = _make_imu_data(
        angular_velocity_bias_rad_s=angular_bias,
        linear_acceleration_bias_m_s2=acceleration_bias,
        angular_velocity_bias_valid=angular_valid,
        linear_acceleration_bias_valid=acceleration_valid,
    )

    np.testing.assert_array_equal(imu_data.angular_velocity_bias_rad_s, angular_bias)
    np.testing.assert_array_equal(imu_data.linear_acceleration_bias_m_s2, acceleration_bias)
    np.testing.assert_array_equal(imu_data.angular_velocity_bias_valid, angular_valid)
    np.testing.assert_array_equal(imu_data.linear_acceleration_bias_valid, acceleration_valid)


def test_imu_data_accepts_temperature_validity() -> None:
    """ImuData should preserve optional temperature validity."""
    imu_data = _make_imu_data(
        temperature_c=np.array([33.0, 33.5], dtype=np.float64),
        temperature_valid=np.array([True, False], dtype=np.bool_),
    )

    np.testing.assert_array_equal(imu_data.temperature_c, np.array([33.0, 33.5], dtype=np.float64))
    np.testing.assert_array_equal(imu_data.temperature_valid, np.array([True, False], dtype=np.bool_))


def test_imu_data_accepts_nonfinite_raw_values_when_validity_false() -> None:
    """Raw invalid IMU values may be non-finite when matching validity masks are false."""
    imu_data = _make_imu_data(
        angular_velocity_rad_s=np.array([[np.nan, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        linear_acceleration_m_s2=np.array([[np.nan, 2.0, 9.8], [1.1, 2.1, 9.7]], dtype=np.float64),
        angular_velocity_valid=np.array([[False, True, True], [True, True, True]], dtype=np.bool_),
        linear_acceleration_valid=np.array([[False, True, True], [True, True, True]], dtype=np.bool_),
        temperature_c=np.array([np.nan, 33.5], dtype=np.float64),
        temperature_valid=np.array([False, True], dtype=np.bool_),
    )

    assert np.isnan(imu_data.angular_velocity_rad_s[0, 0])
    assert np.isnan(imu_data.linear_acceleration_m_s2[0, 0])
    assert imu_data.temperature_c is not None
    assert np.isnan(imu_data.temperature_c[0])


@pytest.mark.parametrize(
    ("field_name", "value", "validity_name", "validity"),
    [
        (
            "angular_velocity_rad_s",
            np.array([[np.nan, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
            "angular_velocity_valid",
            np.array([[True, True, True], [True, True, True]], dtype=np.bool_),
        ),
        (
            "linear_acceleration_m_s2",
            np.array([[np.nan, 2.0, 9.8], [1.1, 2.1, 9.7]], dtype=np.float64),
            "linear_acceleration_valid",
            np.array([[True, True, True], [True, True, True]], dtype=np.bool_),
        ),
        (
            "temperature_c",
            np.array([np.nan, 33.5], dtype=np.float64),
            "temperature_valid",
            np.array([True, True], dtype=np.bool_),
        ),
    ],
)
def test_imu_data_rejects_nonfinite_raw_values_when_validity_true(
    field_name: str,
    value: npt.NDArray[Any],
    validity_name: str,
    validity: npt.NDArray[np.bool_],
) -> None:
    """Non-finite raw values still require a matching false validity-mask entry."""
    with pytest.raises(ValueError, match="validity mask entries to be false"):
        _make_imu_data(**{field_name: value, validity_name: validity})


def test_imu_data_satisfies_sensor_data_protocol() -> None:
    """ImuData should be structurally usable anywhere SensorData is expected."""
    sensor_data: SensorData = _make_imu_data()

    np.testing.assert_array_equal(sensor_data.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


def test_aligned_frame_accepts_matching_imu_data() -> None:
    """AlignedFrame should accept ImuData sampled on the same reference timeline."""
    imu_data = _make_imu_data()

    frame = AlignedFrame(
        align_timestamps_ns=np.array([100, 200], dtype=np.int64),
        sensor_data={"imu0": cast("SensorData", imu_data)},
    )

    assert frame["imu0"] is imu_data


def test_aligned_frame_rejects_mismatched_imu_data_reference_timeline() -> None:
    """AlignedFrame should reject ImuData sampled on a different reference timeline."""
    imu_data = _make_imu_data(align_timestamps_ns=np.array([100, 300], dtype=np.int64))

    with pytest.raises(ValueError, match="align_timestamps_ns must exactly match"):
        AlignedFrame(
            align_timestamps_ns=np.array([100, 200], dtype=np.int64),
            sensor_data={"imu0": cast("SensorData", imu_data)},
        )


def test_imu_data_arrays_are_readonly() -> None:
    """ImuData should expose read-only NumPy arrays."""
    imu_data = _make_imu_data(
        orientation_quat_xyzw=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64),
        angular_velocity_covariance=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        linear_acceleration_covariance=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        orientation_covariance=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        angular_velocity_valid=np.ones((2, 3), dtype=np.bool_),
        linear_acceleration_valid=np.ones((2, 3), dtype=np.bool_),
        orientation_valid=np.array([True, False], dtype=np.bool_),
        angular_velocity_bias_rad_s=np.zeros((2, 3), dtype=np.float64),
        linear_acceleration_bias_m_s2=np.zeros((2, 3), dtype=np.float64),
        angular_velocity_bias_valid=np.ones((2, 3), dtype=np.bool_),
        linear_acceleration_bias_valid=np.ones((2, 3), dtype=np.bool_),
        host_timestamps_ns=np.array([95, 215], dtype=np.int64),
        sequence_counter=np.array([7, 8], dtype=np.uint64),
        temperature_c=np.array([33.0, 33.5], dtype=np.float64),
        temperature_valid=np.array([True, False], dtype=np.bool_),
    )

    arrays = [
        imu_data.align_timestamps_ns,
        imu_data.sensor_timestamps_ns,
        imu_data.angular_velocity_rad_s,
        imu_data.linear_acceleration_m_s2,
        imu_data.orientation_quat_xyzw,
        imu_data.angular_velocity_covariance,
        imu_data.linear_acceleration_covariance,
        imu_data.orientation_covariance,
        imu_data.angular_velocity_valid,
        imu_data.linear_acceleration_valid,
        imu_data.orientation_valid,
        imu_data.angular_velocity_bias_rad_s,
        imu_data.linear_acceleration_bias_m_s2,
        imu_data.angular_velocity_bias_valid,
        imu_data.linear_acceleration_bias_valid,
        imu_data.host_timestamps_ns,
        imu_data.sequence_counter,
        imu_data.temperature_c,
        imu_data.temperature_valid,
    ]

    for array in arrays:
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]


def test_imu_data_does_not_mutate_caller_owned_arrays() -> None:
    """ImuData should expose read-only views without changing caller-owned arrays."""
    angular_velocity_rad_s = np.zeros((2, 3), dtype=np.float64)
    linear_acceleration_m_s2 = np.ones((2, 3), dtype=np.float64)
    angular_velocity_valid = np.ones((2, 3), dtype=np.bool_)

    imu_data = _make_imu_data(
        angular_velocity_rad_s=angular_velocity_rad_s,
        linear_acceleration_m_s2=linear_acceleration_m_s2,
        angular_velocity_valid=angular_velocity_valid,
    )

    assert angular_velocity_rad_s.flags.writeable is True
    assert linear_acceleration_m_s2.flags.writeable is True
    assert angular_velocity_valid.flags.writeable is True
    assert imu_data.angular_velocity_rad_s.flags.writeable is False
    assert imu_data.linear_acceleration_m_s2.flags.writeable is False
    assert imu_data.angular_velocity_valid is not None
    assert imu_data.angular_velocity_valid.flags.writeable is False
    assert imu_data.angular_velocity_rad_s is not angular_velocity_rad_s
    assert imu_data.linear_acceleration_m_s2 is not linear_acceleration_m_s2
    assert imu_data.angular_velocity_valid is not angular_velocity_valid
    assert np.shares_memory(imu_data.angular_velocity_rad_s, angular_velocity_rad_s)
    assert np.shares_memory(imu_data.linear_acceleration_m_s2, linear_acceleration_m_s2)
    assert np.shares_memory(imu_data.angular_velocity_valid, angular_velocity_valid)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("align_timestamps_ns", np.array([2, 1], dtype=np.int64), "strictly sorted"),
        ("sensor_timestamps_ns", np.array([2, 1], dtype=np.int64), "sorted in ascending order"),
        ("angular_velocity_rad_s", np.zeros((2,), dtype=np.float64), r"shape \(N, 3\)"),
        ("linear_acceleration_m_s2", np.zeros((2, 2), dtype=np.float64), r"shape \(N, 3\)"),
        ("angular_velocity_rad_s", np.zeros((2, 3), dtype=np.float32), "dtype float64"),
        ("linear_acceleration_m_s2", np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]), "finite"),
    ],
)
def test_imu_data_rejects_invalid_required_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """ImuData should validate required timestamp and vector fields."""
    with pytest.raises(ValueError, match=match):
        _make_imu_data(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("orientation_quat_xyzw", np.zeros((2, 3), dtype=np.float64), r"shape \(N, 4\)"),
        ("orientation_quat_xyzw", np.zeros((2, 4), dtype=np.float64), "unit norm"),
        ("orientation_quat_xyzw", np.array([[0.0, 0.0, 0.0, 1.0000015]], dtype=np.float64), "unit norm"),
        ("orientation_quat_xyzw", np.array([[0.0, 0.0, 0.0, 1.0], [np.nan, 0.0, 0.0, 1.0]]), "finite"),
        ("angular_velocity_covariance", np.zeros((2, 3), dtype=np.float64), r"shape \(N, 3, 3\)"),
        ("linear_acceleration_covariance", np.full((2, 3, 3), np.nan, dtype=np.float64), "finite"),
        ("orientation_covariance", np.tile(np.diag([1.0, -1.0, 1.0]), (2, 1, 1)), "positive semidefinite"),
        (
            "orientation_covariance",
            np.tile(np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), (2, 1, 1)),
            "symmetric",
        ),
        ("angular_velocity_valid", np.ones((2,), dtype=np.bool_), r"shape \(N, 3\)"),
        ("linear_acceleration_valid", np.ones((2, 3), dtype=np.int8), "dtype bool"),
        ("angular_velocity_bias_rad_s", np.ones((2, 2), dtype=np.float64), r"shape \(N, 3\)"),
        ("linear_acceleration_bias_m_s2", np.ones((2, 3), dtype=np.float32), "dtype float64"),
        ("angular_velocity_bias_valid", np.ones((2,), dtype=np.bool_), r"shape \(N, 3\)"),
        ("linear_acceleration_bias_valid", np.ones((2, 3), dtype=np.int8), "dtype bool"),
        ("temperature_c", np.array([32.0, np.inf], dtype=np.float64), "finite"),
    ],
)
def test_imu_data_rejects_invalid_optional_fields(
    field_name: str,
    value: npt.NDArray[Any],
    match: str,
) -> None:
    """ImuData should validate optional field dtype, shape, and finite-value constraints."""
    with pytest.raises(ValueError, match=match):
        _make_imu_data(**{field_name: value})


def test_imu_data_preserves_relative_covariance_symmetry_tolerance() -> None:
    """Large covariance values should retain the historical relative symmetry tolerance."""
    covariance = np.tile(
        np.array(
            [
                [1_000.0, 100.0, 0.0],
                [100.0 + 5e-8, 1_000.0, 0.0],
                [0.0, 0.0, 1_000.0],
            ],
            dtype=np.float64,
        ),
        (2, 1, 1),
    )

    imu_data = _make_imu_data(angular_velocity_covariance=covariance)

    assert imu_data.angular_velocity_covariance is not None
    np.testing.assert_array_equal(imu_data.angular_velocity_covariance, covariance)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("orientation_quat_xyzw", np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)),
        ("angular_velocity_covariance", np.eye(3, dtype=np.float64).reshape(1, 3, 3)),
        ("linear_acceleration_covariance", np.eye(3, dtype=np.float64).reshape(1, 3, 3)),
        ("orientation_covariance", np.eye(3, dtype=np.float64).reshape(1, 3, 3)),
        ("angular_velocity_valid", np.ones((1, 3), dtype=np.bool_)),
        ("linear_acceleration_valid", np.ones((1, 3), dtype=np.bool_)),
        ("orientation_valid", np.ones(1, dtype=np.bool_)),
        ("angular_velocity_bias_rad_s", np.ones((1, 3), dtype=np.float64)),
        ("linear_acceleration_bias_m_s2", np.ones((1, 3), dtype=np.float64)),
        ("host_timestamps_ns", np.ones(1, dtype=np.int64)),
        ("sequence_counter", np.ones(1, dtype=np.uint64)),
        ("temperature_c", np.ones(1, dtype=np.float64)),
    ],
)
def test_imu_data_rejects_optional_batch_length_mismatches(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """ImuData should require optional arrays to share the required batch length."""
    overrides: dict[str, npt.NDArray[Any]] = {field_name: value}
    if field_name == "orientation_valid":
        overrides["orientation_quat_xyzw"] = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="same length"):
        _make_imu_data(**overrides)


def test_imu_data_rejects_temperature_valid_without_temperature_c() -> None:
    """temperature_valid should not be present without matching temperature values."""
    with pytest.raises(ValueError, match="temperature_valid requires temperature_c"):
        _make_imu_data(temperature_valid=np.array([True, False], dtype=np.bool_))


def test_imu_data_rejects_temperature_valid_batch_length_mismatch() -> None:
    """temperature_valid should share the required IMU batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_imu_data(
            temperature_c=np.array([33.0, 33.5], dtype=np.float64),
            temperature_valid=np.array([True], dtype=np.bool_),
        )


@pytest.mark.parametrize(
    ("value_name", "validity_name"),
    [
        ("angular_velocity_bias_rad_s", "angular_velocity_bias_valid"),
        ("linear_acceleration_bias_m_s2", "linear_acceleration_bias_valid"),
    ],
)
def test_imu_data_rejects_bias_valid_batch_length_mismatch(value_name: str, validity_name: str) -> None:
    """Bias validity masks should share the required IMU batch length."""
    with pytest.raises(ValueError, match="same length"):
        _make_imu_data(
            **{
                value_name: np.zeros((2, 3), dtype=np.float64),
                validity_name: np.ones((1, 3), dtype=np.bool_),
            }
        )


@pytest.mark.parametrize(
    ("value_name", "validity_name"),
    [
        ("angular_velocity_bias_rad_s", "angular_velocity_bias_valid"),
        ("linear_acceleration_bias_m_s2", "linear_acceleration_bias_valid"),
    ],
)
def test_imu_data_rejects_nonfinite_valid_bias(value_name: str, validity_name: str) -> None:
    """Non-finite bias values require their corresponding axes to be invalid."""
    values = np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    validity = np.ones((2, 3), dtype=np.bool_)

    with pytest.raises(ValueError, match="validity mask entries to be false"):
        _make_imu_data(**{value_name: values, validity_name: validity})


@pytest.mark.parametrize(
    ("value_name", "validity_name"),
    [
        ("angular_velocity_bias_rad_s", "angular_velocity_bias_valid"),
        ("linear_acceleration_bias_m_s2", "linear_acceleration_bias_valid"),
    ],
)
def test_imu_data_rejects_bias_valid_without_value(value_name: str, validity_name: str) -> None:
    """A bias validity mask cannot be present without its paired value array."""
    with pytest.raises(ValueError, match=rf"{validity_name} requires {value_name}"):
        _make_imu_data(**{validity_name: np.ones((2, 3), dtype=np.bool_)})


@pytest.mark.parametrize(
    "value_name",
    ["angular_velocity_bias_rad_s", "linear_acceleration_bias_m_s2"],
)
def test_imu_data_accepts_bias_value_without_validity(value_name: str) -> None:
    """A finite bias value is accepted without an explicit validity mask."""
    imu_data = _make_imu_data(**{value_name: np.zeros((2, 3), dtype=np.float64)})

    np.testing.assert_array_equal(getattr(imu_data, value_name), np.zeros((2, 3), dtype=np.float64))
