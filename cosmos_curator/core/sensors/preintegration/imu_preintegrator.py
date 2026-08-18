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
"""Eager midpoint preintegration of aligned IMU point samples."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.data.preintegrated_imu_data import (
    MIN_INTEGRATION_SAMPLES,
    ImuIntegrationInvalidReason,
    PreintegratedImuData,
)
from cosmos_curator.core.sensors.utils.validation import require_strictly_increasing

_NS_PER_SECOND = 1_000_000_000
_VECTOR_SIZE = 3
_SMALL_ANGLE = 1e-10


@dataclass(frozen=True)
class _InterpolationPoint:
    """Linear interpolation coordinates on the raw IMU timeline."""

    timestamp_ns: int
    left_index: int
    right_index: int
    alpha: float


@dataclass(frozen=True)
class PreparedImuSamples:
    """Recording-wide validity and bias arrays prepared once for all intervals."""

    corrected_angular_velocity_rad_s: npt.NDArray[np.float64]
    corrected_linear_acceleration_m_s2: npt.NDArray[np.float64]
    angular_velocity_bias_rad_s: npt.NDArray[np.float64]
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64]
    angular_velocity_bias_available: npt.NDArray[np.bool_]
    linear_acceleration_bias_available: npt.NDArray[np.bool_]
    measurement_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _IntervalSamples:
    """Interpolated IMU samples spanning one requested interval."""

    timestamps_ns: npt.NDArray[np.int64]
    corrected_angular_velocity_rad_s: npt.NDArray[np.float64]
    corrected_linear_acceleration_m_s2: npt.NDArray[np.float64]
    angular_velocity_bias_rad_s: npt.NDArray[np.float64]
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64]
    angular_velocity_bias_available: npt.NDArray[np.bool_]
    linear_acceleration_bias_available: npt.NDArray[np.bool_]
    measurement_valid: npt.NDArray[np.bool_]
    support_slice: slice


@dataclass(frozen=True)
class _IntervalBounds:
    """Boundary interpolation coordinates and contiguous raw-array slices."""

    start: _InterpolationPoint
    end: _InterpolationPoint
    interior_slice: slice
    support_slice: slice


@dataclass(frozen=True)
class _IntegratedMotion:
    """One interval's integrated motion."""

    rotation: npt.NDArray[np.float64]
    velocity_m_s: npt.NDArray[np.float64]
    position_m: npt.NDArray[np.float64]


def _skew(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the skew-symmetric matrix for a three-vector."""
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _so3_exp(rotation_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map a rotation vector to an SO(3) rotation matrix."""
    angle = float(np.linalg.norm(rotation_vector))
    skew = _skew(rotation_vector)
    identity = np.eye(_VECTOR_SIZE, dtype=np.float64)
    if angle < _SMALL_ANGLE:
        return identity + skew + 0.5 * (skew @ skew)
    angle_squared = angle * angle
    return np.asarray(
        identity + (np.sin(angle) / angle) * skew + ((1.0 - np.cos(angle)) / angle_squared) * (skew @ skew),
        dtype=np.float64,
    )


def _rotation_to_quaternion_xyzw(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a rotation matrix to a normalized XYZW quaternion."""
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion[3] = 0.25 * scale
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(rotation)))
        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion[3] = (rotation[2, 1] - rotation[1, 2]) / scale
            quaternion[0] = 0.25 * scale
            quaternion[1] = (rotation[0, 1] + rotation[1, 0]) / scale
            quaternion[2] = (rotation[0, 2] + rotation[2, 0]) / scale
        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion[3] = (rotation[0, 2] - rotation[2, 0]) / scale
            quaternion[0] = (rotation[0, 1] + rotation[1, 0]) / scale
            quaternion[1] = 0.25 * scale
            quaternion[2] = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion[3] = (rotation[1, 0] - rotation[0, 1]) / scale
            quaternion[0] = (rotation[0, 2] + rotation[2, 0]) / scale
            quaternion[1] = (rotation[1, 2] + rotation[2, 1]) / scale
            quaternion[2] = 0.25 * scale
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion / np.linalg.norm(quaternion)


def _interpolation_point(
    timestamps_ns: npt.NDArray[np.int64],
    timestamp_ns: int,
) -> _InterpolationPoint | None:
    """Return interpolation coordinates for one timestamp, or ``None`` outside the source range."""
    right = int(np.searchsorted(timestamps_ns, timestamp_ns, side="left"))
    if right < len(timestamps_ns) and int(timestamps_ns[right]) == timestamp_ns:
        return _InterpolationPoint(timestamp_ns, right, right, 0.0)
    if right == 0 or right == len(timestamps_ns):
        return None
    left = right - 1
    duration_ns = int(timestamps_ns[right]) - int(timestamps_ns[left])
    alpha = (timestamp_ns - int(timestamps_ns[left])) / duration_ns
    return _InterpolationPoint(timestamp_ns, left, right, alpha)


def _interpolate_array(
    values: npt.NDArray[np.float64],
    point: _InterpolationPoint,
) -> npt.NDArray[np.float64]:
    """Linearly interpolate an array row at one point."""
    if point.left_index == point.right_index or point.alpha <= 0.0:
        return np.array(values[point.left_index], dtype=np.float64, copy=True)
    if point.alpha >= 1.0:
        return np.array(values[point.right_index], dtype=np.float64, copy=True)
    left = values[point.left_index]
    right = values[point.right_index]
    return np.asarray((1.0 - point.alpha) * left + point.alpha * right, dtype=np.float64)


def _interpolate_timestamp(values: npt.NDArray[np.int64], point: _InterpolationPoint) -> int:
    """Interpolate an integer timestamp without losing epoch-scale precision."""
    if point.left_index == point.right_index:
        return int(values[point.left_index])
    left = int(values[point.left_index])
    right = int(values[point.right_index])
    return left + round(point.alpha * (right - left))


def _sensor_domain_point(
    sensor_timestamps_ns: npt.NDArray[np.int64],
    align_point: _InterpolationPoint,
) -> _InterpolationPoint:
    """Express one align-domain boundary at its rounded sensor-clock coordinate."""
    sensor_timestamp_ns = _interpolate_timestamp(sensor_timestamps_ns, align_point)
    if align_point.left_index == align_point.right_index:
        return _InterpolationPoint(
            sensor_timestamp_ns,
            align_point.left_index,
            align_point.right_index,
            0.0,
        )
    left_sensor_ns = int(sensor_timestamps_ns[align_point.left_index])
    right_sensor_ns = int(sensor_timestamps_ns[align_point.right_index])
    if left_sensor_ns == right_sensor_ns:
        # The containing interval is invalidated before integration. Retaining
        # the align-domain fraction keeps its diagnostic payload deterministic.
        sensor_alpha = align_point.alpha
    else:
        sensor_alpha = (sensor_timestamp_ns - left_sensor_ns) / (right_sensor_ns - left_sensor_ns)
    return _InterpolationPoint(
        sensor_timestamp_ns,
        align_point.left_index,
        align_point.right_index,
        sensor_alpha,
    )


def _interpolate_validity(
    validity: npt.NDArray[np.bool_],
    point: _InterpolationPoint,
) -> npt.NDArray[np.bool_]:
    """Require both interpolation endpoints to be valid."""
    if point.left_index == point.right_index or point.alpha <= 0.0:
        return np.array(validity[point.left_index], dtype=np.bool_, copy=True)
    if point.alpha >= 1.0:
        return np.array(validity[point.right_index], dtype=np.bool_, copy=True)
    return np.asarray(validity[point.left_index] & validity[point.right_index], dtype=np.bool_)


def _measurement_validity(
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Return explicit validity or infer it from finiteness."""
    finite = np.isfinite(values)
    if validity is None:
        return finite
    return np.asarray(validity & finite, dtype=np.bool_)


def _bias_values_and_availability(
    values: npt.NDArray[np.float64] | None,
    validity: npt.NDArray[np.bool_] | None,
    row_count: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return finite bias values with unavailable axes replaced by zero."""
    if values is None:
        return (
            np.zeros((row_count, _VECTOR_SIZE), dtype=np.float64),
            np.zeros((row_count, _VECTOR_SIZE), dtype=np.bool_),
        )
    available = np.isfinite(values)
    if validity is not None:
        available &= validity
    return np.where(available, values, 0.0), available


def prepare_imu_samples(imu_data: ImuData) -> PreparedImuSamples:
    """Prepare recording-wide validity and bias arrays once."""
    row_count = len(imu_data.align_timestamps_ns)
    gyro_valid = _measurement_validity(imu_data.angular_velocity_rad_s, imu_data.angular_velocity_valid)
    accel_valid = _measurement_validity(imu_data.linear_acceleration_m_s2, imu_data.linear_acceleration_valid)
    gyro_bias, gyro_bias_available = _bias_values_and_availability(
        imu_data.angular_velocity_bias_rad_s,
        imu_data.angular_velocity_bias_valid,
        row_count,
    )
    accel_bias, accel_bias_available = _bias_values_and_availability(
        imu_data.linear_acceleration_bias_m_s2,
        imu_data.linear_acceleration_bias_valid,
        row_count,
    )
    return PreparedImuSamples(
        corrected_angular_velocity_rad_s=imu_data.angular_velocity_rad_s - gyro_bias,
        corrected_linear_acceleration_m_s2=imu_data.linear_acceleration_m_s2 - accel_bias,
        angular_velocity_bias_rad_s=gyro_bias,
        linear_acceleration_bias_m_s2=accel_bias,
        angular_velocity_bias_available=gyro_bias_available,
        linear_acceleration_bias_available=accel_bias_available,
        measurement_valid=np.all(gyro_valid & accel_valid, axis=1),
    )


def _interval_bounds(
    timestamps_ns: npt.NDArray[np.int64],
    start_ns: int,
    end_ns: int,
) -> _IntervalBounds | None:
    """Locate interval boundaries and contiguous interior/support slices."""
    start = _interpolation_point(timestamps_ns, start_ns)
    end = _interpolation_point(timestamps_ns, end_ns)
    if start is None or end is None:
        return None
    first_interior = int(np.searchsorted(timestamps_ns, start_ns, side="right"))
    past_interior = int(np.searchsorted(timestamps_ns, end_ns, side="left"))
    return _IntervalBounds(
        start=start,
        end=end,
        interior_slice=slice(first_interior, past_interior),
        support_slice=slice(start.left_index, end.right_index + 1),
    )


def _assemble_float64_interval(
    values: npt.NDArray[np.float64],
    bounds: _IntervalBounds,
) -> npt.NDArray[np.float64]:
    """Assemble interpolated boundaries around a contiguous float64 interior."""
    return np.concatenate(
        (
            np.expand_dims(_interpolate_array(values, bounds.start), axis=0),
            values[bounds.interior_slice],
            np.expand_dims(_interpolate_array(values, bounds.end), axis=0),
        ),
        axis=0,
    )


def _assemble_bool_interval(
    values: npt.NDArray[np.bool_],
    bounds: _IntervalBounds,
) -> npt.NDArray[np.bool_]:
    """Assemble conservative boundary validity around a contiguous interior."""
    return np.concatenate(
        (
            np.expand_dims(_interpolate_validity(values, bounds.start), axis=0),
            values[bounds.interior_slice],
            np.expand_dims(_interpolate_validity(values, bounds.end), axis=0),
        ),
        axis=0,
    )


def _build_interval_samples(
    imu_data: ImuData,
    prepared: PreparedImuSamples,
    start_ns: int,
    end_ns: int,
) -> _IntervalSamples | None:
    """Map align-clock boundaries and assemble samples on the sensor clock."""
    bounds = _interval_bounds(imu_data.align_timestamps_ns, start_ns, end_ns)
    if bounds is None:
        return None
    sensor_bounds = _IntervalBounds(
        start=_sensor_domain_point(imu_data.sensor_timestamps_ns, bounds.start),
        end=_sensor_domain_point(imu_data.sensor_timestamps_ns, bounds.end),
        interior_slice=bounds.interior_slice,
        support_slice=bounds.support_slice,
    )
    first_interior = sensor_bounds.interior_slice.start or 0
    past_interior = sensor_bounds.interior_slice.stop or 0
    if (
        first_interior < past_interior
        and sensor_bounds.start.alpha >= 1.0
        and sensor_bounds.start.timestamp_ns == int(imu_data.sensor_timestamps_ns[first_interior])
    ):
        first_interior += 1
    if (
        first_interior < past_interior
        and sensor_bounds.end.alpha <= 0.0
        and sensor_bounds.end.timestamp_ns == int(imu_data.sensor_timestamps_ns[past_interior - 1])
    ):
        past_interior -= 1
    sensor_bounds = _IntervalBounds(
        start=sensor_bounds.start,
        end=sensor_bounds.end,
        interior_slice=slice(first_interior, past_interior),
        support_slice=sensor_bounds.support_slice,
    )
    return _IntervalSamples(
        timestamps_ns=np.concatenate(
            (
                np.asarray([sensor_bounds.start.timestamp_ns], dtype=np.int64),
                imu_data.sensor_timestamps_ns[sensor_bounds.interior_slice],
                np.asarray([sensor_bounds.end.timestamp_ns], dtype=np.int64),
            )
        ),
        corrected_angular_velocity_rad_s=_assemble_float64_interval(
            prepared.corrected_angular_velocity_rad_s,
            sensor_bounds,
        ),
        corrected_linear_acceleration_m_s2=_assemble_float64_interval(
            prepared.corrected_linear_acceleration_m_s2,
            sensor_bounds,
        ),
        angular_velocity_bias_rad_s=_assemble_float64_interval(
            prepared.angular_velocity_bias_rad_s,
            sensor_bounds,
        ),
        linear_acceleration_bias_m_s2=_assemble_float64_interval(
            prepared.linear_acceleration_bias_m_s2,
            sensor_bounds,
        ),
        angular_velocity_bias_available=_assemble_bool_interval(
            prepared.angular_velocity_bias_available,
            sensor_bounds,
        ),
        linear_acceleration_bias_available=_assemble_bool_interval(
            prepared.linear_acceleration_bias_available,
            sensor_bounds,
        ),
        measurement_valid=_assemble_bool_interval(prepared.measurement_valid, sensor_bounds),
        support_slice=bounds.support_slice,
    )


def _integrate_interval(samples: _IntervalSamples) -> _IntegratedMotion:
    """Integrate one interval with midpoint updates."""
    rotation = np.eye(_VECTOR_SIZE, dtype=np.float64)
    velocity = np.zeros(_VECTOR_SIZE, dtype=np.float64)
    position = np.zeros(_VECTOR_SIZE, dtype=np.float64)

    for index in range(len(samples.timestamps_ns) - 1):
        dt = (int(samples.timestamps_ns[index + 1]) - int(samples.timestamps_ns[index])) / _NS_PER_SECOND
        rotation_increment = _so3_exp(
            0.5
            * (samples.corrected_angular_velocity_rad_s[index] + samples.corrected_angular_velocity_rad_s[index + 1])
            * dt
        )
        acceleration_start_frame = 0.5 * (
            rotation @ samples.corrected_linear_acceleration_m_s2[index]
            + rotation @ rotation_increment @ samples.corrected_linear_acceleration_m_s2[index + 1]
        )

        position = position + velocity * dt + 0.5 * acceleration_start_frame * dt * dt
        velocity = velocity + acceleration_start_frame * dt
        rotation = rotation @ rotation_increment

    return _IntegratedMotion(
        rotation=rotation,
        velocity_m_s=velocity,
        position_m=position,
    )


def _time_average(
    timestamps_ns: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return a trapezoidal time average over an interval."""
    durations = np.diff(timestamps_ns).astype(np.float64) / _NS_PER_SECOND
    total_duration = float(np.sum(durations))
    if total_duration == 0.0:
        return np.zeros(values.shape[1], dtype=np.float64)
    midpoint_values = 0.5 * (values[:-1] + values[1:])
    return np.asarray(np.sum(midpoint_values * durations[:, None], axis=0) / total_duration, dtype=np.float64)


def _sensor_timestamp_at(
    imu_data: ImuData,
    align_timestamp_ns: int,
) -> int:
    """Interpolate a representative sensor timestamp at one alignment timestamp."""
    if not len(imu_data.align_timestamps_ns):
        return align_timestamp_ns
    point = _interpolation_point(imu_data.align_timestamps_ns, align_timestamp_ns)
    if point is None:
        if align_timestamp_ns < int(imu_data.align_timestamps_ns[0]):
            return int(imu_data.sensor_timestamps_ns[0])
        return int(imu_data.sensor_timestamps_ns[-1])
    return _interpolate_timestamp(imu_data.sensor_timestamps_ns, point)


def preintegrate_imu(  # noqa: C901, PLR0915
    imu_data: ImuData,
    align_timestamps_ns: npt.NDArray[np.int64],
    *,
    prepared_samples: PreparedImuSamples | None = None,
) -> PreintegratedImuData:
    """Preintegrate a complete ``ImuData`` recording onto an alignment grid.

    Each row after the first represents ``[align[i - 1], align[i])`` in the
    external reference clock. Paired raw alignment timestamps convert those
    boundaries to the IMU sensor clock, which defines interpolation and physical
    integration time. Missing bias axes use zero correction and remain marked
    unavailable in the output. Callers that repeatedly integrate one recording
    may provide ``prepared_samples`` from :func:`prepare_imu_samples` for that
    same ``imu_data`` to reuse its recording-wide preparation.
    """
    if align_timestamps_ns.dtype != np.int64 or align_timestamps_ns.ndim != 1:
        msg = "align_timestamps_ns must be a 1-D int64 array"
        raise ValueError(msg)
    require_strictly_increasing("align_timestamps_ns", align_timestamps_ns)
    row_count = len(align_timestamps_ns)
    interval_starts = np.array(align_timestamps_ns, copy=True)
    if row_count > 1:
        interval_starts[1:] = align_timestamps_ns[:-1]
    sensor_timestamps = np.asarray(
        [_sensor_timestamp_at(imu_data, int(timestamp)) for timestamp in align_timestamps_ns],
        dtype=np.int64,
    )
    durations = np.zeros(row_count, dtype=np.int64)
    if row_count > 1:
        durations[1:] = np.diff(sensor_timestamps)
    delta_rotation = np.zeros((row_count, 4), dtype=np.float64)
    delta_rotation[:, 3] = 1.0
    delta_velocity = np.zeros((row_count, _VECTOR_SIZE), dtype=np.float64)
    delta_position = np.zeros((row_count, _VECTOR_SIZE), dtype=np.float64)
    gyro_bias_used = np.zeros((row_count, _VECTOR_SIZE), dtype=np.float64)
    accel_bias_used = np.zeros((row_count, _VECTOR_SIZE), dtype=np.float64)
    gyro_bias_available = np.zeros((row_count, _VECTOR_SIZE), dtype=np.bool_)
    accel_bias_available = np.zeros((row_count, _VECTOR_SIZE), dtype=np.bool_)
    sample_count_total = np.zeros(row_count, dtype=np.uint32)
    sample_count_used = np.zeros(row_count, dtype=np.uint32)
    sample_count_rejected = np.zeros(row_count, dtype=np.uint32)
    max_gap_ns = np.zeros(row_count, dtype=np.int64)
    integration_valid = np.zeros(row_count, dtype=np.bool_)
    invalid_reason = np.zeros(row_count, dtype=np.uint32)
    if row_count:
        invalid_reason[0] = ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value

    prepared = prepared_samples if prepared_samples is not None else prepare_imu_samples(imu_data)

    for row in range(1, row_count):
        samples = _build_interval_samples(
            imu_data,
            prepared,
            int(interval_starts[row]),
            int(align_timestamps_ns[row]),
        )
        if samples is None:
            invalid_reason[row] = ImuIntegrationInvalidReason.MISSING_BOUNDARY_SUPPORT.value
            continue

        support_valid = prepared.measurement_valid[samples.support_slice]
        sample_count_total[row] = len(support_valid)
        rejected = int(np.count_nonzero(~support_valid))
        sample_count_rejected[row] = rejected
        sample_count_used[row] = len(support_valid) - rejected
        sensor_time_steps_ns = np.diff(samples.timestamps_ns)
        support_sensor_time_steps_ns = np.diff(imu_data.sensor_timestamps_ns[samples.support_slice])
        if len(support_sensor_time_steps_ns):
            max_gap_ns[row] = int(np.max(support_sensor_time_steps_ns))
        gyro_bias_used[row] = _time_average(samples.timestamps_ns, samples.angular_velocity_bias_rad_s)
        accel_bias_used[row] = _time_average(samples.timestamps_ns, samples.linear_acceleration_bias_m_s2)
        gyro_bias_available[row] = np.all(samples.angular_velocity_bias_available, axis=0)
        accel_bias_available[row] = np.all(samples.linear_acceleration_bias_available, axis=0)

        reason = ImuIntegrationInvalidReason.NONE
        if np.any(sensor_time_steps_ns <= 0) or np.any(support_sensor_time_steps_ns <= 0):
            reason |= ImuIntegrationInvalidReason.NON_INCREASING_SENSOR_TIME
        if not np.all(samples.measurement_valid):
            reason |= ImuIntegrationInvalidReason.INVALID_MEASUREMENT
        if sample_count_used[row] < MIN_INTEGRATION_SAMPLES:
            reason |= ImuIntegrationInvalidReason.INSUFFICIENT_SAMPLES
        if reason != ImuIntegrationInvalidReason.NONE:
            invalid_reason[row] = reason.value
            continue

        integrated = _integrate_interval(samples)
        delta_rotation[row] = _rotation_to_quaternion_xyzw(integrated.rotation)
        delta_velocity[row] = integrated.velocity_m_s
        delta_position[row] = integrated.position_m
        integration_valid[row] = True

    return PreintegratedImuData(
        align_timestamps_ns=np.array(align_timestamps_ns, copy=True),
        sensor_timestamps_ns=sensor_timestamps,
        align_interval_start_timestamps_ns=interval_starts,
        align_interval_end_timestamps_ns=np.array(align_timestamps_ns, copy=True),
        integration_duration_ns=durations,
        delta_rotation_quat_xyzw=delta_rotation,
        delta_velocity_m_s=delta_velocity,
        delta_position_m=delta_position,
        angular_velocity_bias_used_rad_s=gyro_bias_used,
        linear_acceleration_bias_used_m_s2=accel_bias_used,
        angular_velocity_bias_available=gyro_bias_available,
        linear_acceleration_bias_available=accel_bias_available,
        sample_count_total=sample_count_total,
        sample_count_used=sample_count_used,
        sample_count_rejected=sample_count_rejected,
        max_inter_sample_gap_ns=max_gap_ns,
        integration_valid=integration_valid,
        integration_invalid_reason=invalid_reason,
    )
