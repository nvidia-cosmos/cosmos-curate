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
"""IMU data structures for cosmos_curator.core.sensors package."""

from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_optional_readonly_view, as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    bool_batch,
    float64_batch,
    nondecreasing_int64_array,
    optional_bool_array,
    optional_float64_array,
    optional_int64_array,
    optional_uint64_array,
    require_finite_or_marked_invalid,
    strictly_increasing_int64_array,
    symmetric_psd_covariance_batch,
    unit_quaternion_batch,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_VECTOR_COLUMNS = 3
_COVARIANCE_SYMMETRY_RTOL = 1e-9
_VECTOR_BATCH_VALIDATOR = float64_batch((_VECTOR_COLUMNS,), finite=False)
_OPTIONAL_QUATERNION_BATCH_VALIDATOR = attrs.validators.optional(unit_quaternion_batch)
_OPTIONAL_COVARIANCE_BATCH_VALIDATOR = attrs.validators.optional(
    symmetric_psd_covariance_batch(_VECTOR_COLUMNS, symmetry_rtol=_COVARIANCE_SYMMETRY_RTOL)
)
_OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR = attrs.validators.optional(bool_batch((_VECTOR_COLUMNS,)))


class _HasImuBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    angular_velocity_rad_s: npt.NDArray[np.float64]
    linear_acceleration_m_s2: npt.NDArray[np.float64]
    orientation_quat_xyzw: npt.NDArray[np.float64] | None
    angular_velocity_covariance: npt.NDArray[np.float64] | None
    linear_acceleration_covariance: npt.NDArray[np.float64] | None
    orientation_covariance: npt.NDArray[np.float64] | None
    angular_velocity_valid: npt.NDArray[np.bool_] | None
    linear_acceleration_valid: npt.NDArray[np.bool_] | None
    orientation_valid: npt.NDArray[np.bool_] | None
    angular_velocity_bias_rad_s: npt.NDArray[np.float64] | None
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64] | None
    angular_velocity_bias_valid: npt.NDArray[np.bool_] | None
    linear_acceleration_bias_valid: npt.NDArray[np.bool_] | None
    host_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None
    temperature_c: npt.NDArray[np.float64] | None
    temperature_valid: npt.NDArray[np.bool_] | None


def _optional_temperature_valid(
    instance: _HasImuBatchFields,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional temperature validity mask."""
    optional_bool_array(instance, attribute, value)
    if value is not None and instance.temperature_c is None:
        msg = "temperature_valid requires temperature_c"
        raise ValueError(msg)


def _optional_orientation_valid(
    instance: _HasImuBatchFields,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional orientation validity mask."""
    optional_bool_array(instance, attribute, value)
    if value is not None and instance.orientation_quat_xyzw is None:
        msg = "orientation_valid requires orientation_quat_xyzw"
        raise ValueError(msg)


def _optional_angular_velocity_bias_valid(
    instance: _HasImuBatchFields,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional gyroscope-bias validity mask."""
    _OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR(instance, attribute, value)
    if value is not None and instance.angular_velocity_bias_rad_s is None:
        msg = "angular_velocity_bias_valid requires angular_velocity_bias_rad_s"
        raise ValueError(msg)


def _optional_linear_acceleration_bias_valid(
    instance: _HasImuBatchFields,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional accelerometer-bias validity mask."""
    _OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR(instance, attribute, value)
    if value is not None and instance.linear_acceleration_bias_m_s2 is None:
        msg = "linear_acceleration_bias_valid requires linear_acceleration_bias_m_s2"
        raise ValueError(msg)


def _raw_value_finiteness(
    instance: _HasImuBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate raw IMU payload values against their validity masks."""
    require_finite_or_marked_invalid(
        "angular_velocity_rad_s",
        instance.angular_velocity_rad_s,
        instance.angular_velocity_valid,
    )
    require_finite_or_marked_invalid(
        "linear_acceleration_m_s2",
        instance.linear_acceleration_m_s2,
        instance.linear_acceleration_valid,
    )
    if instance.temperature_c is not None:
        require_finite_or_marked_invalid("temperature_c", instance.temperature_c, instance.temperature_valid)
    if instance.angular_velocity_bias_rad_s is not None:
        require_finite_or_marked_invalid(
            "angular_velocity_bias_rad_s",
            instance.angular_velocity_bias_rad_s,
            instance.angular_velocity_bias_valid,
        )
    if instance.linear_acceleration_bias_m_s2 is not None:
        require_finite_or_marked_invalid(
            "linear_acceleration_bias_m_s2",
            instance.linear_acceleration_bias_m_s2,
            instance.linear_acceleration_bias_valid,
        )


def _batch_lengths(
    instance: _HasImuBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate shared row-count invariants across IMU batch arrays."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "angular_velocity_rad_s": len(instance.angular_velocity_rad_s),
        "linear_acceleration_m_s2": len(instance.linear_acceleration_m_s2),
    }
    optional_fields = (
        "orientation_quat_xyzw",
        "angular_velocity_covariance",
        "linear_acceleration_covariance",
        "orientation_covariance",
        "angular_velocity_valid",
        "linear_acceleration_valid",
        "orientation_valid",
        "angular_velocity_bias_rad_s",
        "linear_acceleration_bias_m_s2",
        "angular_velocity_bias_valid",
        "linear_acceleration_bias_valid",
        "host_timestamps_ns",
        "sequence_counter",
        "temperature_c",
        "temperature_valid",
    )
    for field_name in optional_fields:
        field_value = getattr(instance, field_name)
        if field_value is not None:
            lengths[field_name] = len(field_value)
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class ImuData:
    """IMU point samples stored as structure-of-arrays batches.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    Required vector fields use SI units in the IMU sensor frame. Optional bias
    estimates use the corresponding measurement units and are stored without
    correction; downstream consumers subtract valid biases when appropriate.
    """

    __hash__ = None  # type: ignore[assignment]

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    angular_velocity_rad_s: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )
    linear_acceleration_m_s2: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )

    orientation_quat_xyzw: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_QUATERNION_BATCH_VALIDATOR,
    )
    angular_velocity_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_COVARIANCE_BATCH_VALIDATOR,
    )
    linear_acceleration_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_COVARIANCE_BATCH_VALIDATOR,
    )
    orientation_covariance: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_COVARIANCE_BATCH_VALIDATOR,
    )

    angular_velocity_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR,
    )
    linear_acceleration_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR,
    )
    orientation_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_orientation_valid,
    )

    angular_velocity_bias_rad_s: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.optional(_VECTOR_BATCH_VALIDATOR),
    )
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.optional(_VECTOR_BATCH_VALIDATOR),
    )
    angular_velocity_bias_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_angular_velocity_bias_valid,
    )
    linear_acceleration_bias_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_linear_acceleration_bias_valid,
    )

    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=optional_int64_array,
    )
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=optional_uint64_array,
    )
    temperature_c: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=optional_float64_array,
    )
    # temperature_valid is last so cross-field validators see every optional array.
    temperature_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.and_(
            _optional_temperature_valid,
            _batch_lengths,
            _raw_value_finiteness,
        ),
    )
