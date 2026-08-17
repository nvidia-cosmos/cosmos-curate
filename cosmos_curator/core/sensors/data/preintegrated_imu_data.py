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
"""Data contract for eager IMU preintegration."""

from enum import IntFlag
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    bool_array,
    bool_batch,
    float64_batch,
    int64_array,
    nondecreasing_int64_array,
    strictly_increasing_int64_array,
    uint32_array,
    unit_quaternion_batch,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

MIN_INTEGRATION_SAMPLES = 2

_VECTOR_BATCH_VALIDATOR = float64_batch((3,))
_BOOL_AXIS_BATCH_VALIDATOR = bool_batch((3,))


class ImuIntegrationInvalidReason(IntFlag):
    """Bit mask explaining why a preintegrated interval is invalid."""

    NONE = 0
    FIRST_ALIGNMENT = 1 << 0
    MISSING_BOUNDARY_SUPPORT = 1 << 1
    INVALID_MEASUREMENT = 1 << 2
    INSUFFICIENT_SAMPLES = 1 << 3
    NON_INCREASING_SENSOR_TIME = 1 << 4


def _invalid_reason_array(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint32],
) -> None:
    """Validate a uint32 invalid-reason bit mask."""
    uint32_array(instance, attribute, value)
    allowed_mask = np.uint32(0)
    for reason in ImuIntegrationInvalidReason:
        allowed_mask |= np.uint32(reason.value)
    if np.any(value & ~allowed_mask):
        msg = f"{attribute.name} contains unknown reason bits"
        raise ValueError(msg)


def _validate_batch_lengths(instance: "PreintegratedImuData") -> None:
    """Require every present array to share the alignment row count."""
    expected = len(instance.align_timestamps_ns)
    for field in attrs.fields(type(instance)):
        value = getattr(instance, field.name)
        if value is not None and len(value) != expected:
            msg = f"All arrays must be the same length: align_timestamps_ns={expected} {field.name}={len(value)}"
            raise ValueError(msg)


def _validate_intervals(instance: "PreintegratedImuData") -> None:
    """Validate alignment intervals and physical sensor-clock durations."""
    starts = instance.align_interval_start_timestamps_ns
    ends = instance.align_interval_end_timestamps_ns
    align = instance.align_timestamps_ns
    if np.any(starts > ends):
        msg = "align_interval_start_timestamps_ns must not exceed align_interval_end_timestamps_ns"
        raise ValueError(msg)
    if not np.array_equal(ends, align):
        msg = "align_interval_end_timestamps_ns must equal align_timestamps_ns"
        raise ValueError(msg)
    if len(align) and not np.array_equal(starts[1:], align[:-1]):
        msg = "Each interval must start at the previous alignment timestamp"
        raise ValueError(msg)
    if np.any(instance.integration_duration_ns < 0):
        msg = "integration_duration_ns must be nonnegative"
        raise ValueError(msg)
    if len(align) and starts[0] == ends[0] and instance.integration_duration_ns[0] != 0:
        msg = "The full-grid identity row must have zero integration_duration_ns"
        raise ValueError(msg)
    if len(align) > 1 and not np.array_equal(
        instance.integration_duration_ns[1:],
        np.diff(instance.sensor_timestamps_ns),
    ):
        msg = "integration_duration_ns must equal adjacent sensor timestamp differences"
        raise ValueError(msg)


def _validate_counts_and_quality(instance: "PreintegratedImuData") -> None:
    """Validate sample accounting and validity-reason invariants."""
    total = instance.sample_count_total.astype(np.uint64)
    used = instance.sample_count_used.astype(np.uint64)
    rejected = instance.sample_count_rejected.astype(np.uint64)
    if np.any(used + rejected != total):
        msg = "sample_count_used plus sample_count_rejected must equal sample_count_total"
        raise ValueError(msg)
    if np.any(instance.max_inter_sample_gap_ns < 0):
        msg = "max_inter_sample_gap_ns must be nonnegative"
        raise ValueError(msg)
    reason_is_none = instance.integration_invalid_reason == ImuIntegrationInvalidReason.NONE.value
    if not np.array_equal(instance.integration_valid, reason_is_none):
        msg = "integration_valid must be true exactly when integration_invalid_reason is NONE"
        raise ValueError(msg)
    if np.any(instance.integration_valid & (instance.sample_count_used < MIN_INTEGRATION_SAMPLES)):
        msg = "Valid integration intervals require at least two used samples"
        raise ValueError(msg)
    if np.any(instance.integration_valid & (instance.integration_duration_ns <= 0)):
        msg = "Valid integration intervals require positive integration_duration_ns"
        raise ValueError(msg)


def _validate_initial_row(instance: "PreintegratedImuData") -> None:
    """Validate the identity row when a batch begins at the full-grid origin."""
    if not len(instance.align_timestamps_ns):
        return
    first_reason = ImuIntegrationInvalidReason(int(instance.integration_invalid_reason[0]))
    if instance.align_interval_start_timestamps_ns[0] < instance.align_interval_end_timestamps_ns[0]:
        if ImuIntegrationInvalidReason.FIRST_ALIGNMENT in first_reason:
            msg = "A sliced interval row must not include FIRST_ALIGNMENT"
            raise ValueError(msg)
        return
    if ImuIntegrationInvalidReason.FIRST_ALIGNMENT not in first_reason:
        msg = "The first row invalid reason must include FIRST_ALIGNMENT"
        raise ValueError(msg)
    if instance.integration_valid[0]:
        msg = "The first row must be invalid"
        raise ValueError(msg)
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.array_equal(instance.delta_rotation_quat_xyzw[0], identity):
        msg = "The first row delta rotation must be the identity quaternion"
        raise ValueError(msg)
    if np.any(instance.delta_velocity_m_s[0]) or np.any(instance.delta_position_m[0]):
        msg = "The first row translation deltas must be zero"
        raise ValueError(msg)
    if instance.sample_count_total[0] != 0:
        msg = "The first row must not contain integrated samples"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class PreintegratedImuData:
    """Causal IMU preintegration results aligned to a reference timestamp grid.

    Input measurements, alignment timestamps, biases, and validity come from
    ``ImuData``. Valid bias axes are subtracted before integration; unavailable
    axes use zero bias while their availability remains false in this output.

    Alignment timestamps and alignment interval bounds remain in the external
    reference-clock domain. Sensor timestamps are the corresponding IMU-clock
    endpoints; durations and deltas use that physical sensor timeline. Deltas
    use midpoint integration and are expressed in the starting IMU frame
    without applying gravity.

    A complete grid starts with an invalid zero-duration identity row. A
    window slice may instead begin with a valid interval whose start timestamp
    precedes the slice's first alignment timestamp.
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
    align_interval_start_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=int64_array,
    )
    align_interval_end_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=int64_array,
    )
    integration_duration_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=int64_array,
    )
    delta_rotation_quat_xyzw: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=unit_quaternion_batch,
    )
    delta_velocity_m_s: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )
    delta_position_m: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )
    angular_velocity_bias_used_rad_s: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )
    linear_acceleration_bias_used_m_s2: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_VECTOR_BATCH_VALIDATOR,
    )
    angular_velocity_bias_available: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_BOOL_AXIS_BATCH_VALIDATOR,
    )
    linear_acceleration_bias_available: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_BOOL_AXIS_BATCH_VALIDATOR,
    )
    sample_count_total: npt.NDArray[np.uint32] = attrs.field(
        converter=as_readonly_view,
        validator=uint32_array,
    )
    sample_count_used: npt.NDArray[np.uint32] = attrs.field(
        converter=as_readonly_view,
        validator=uint32_array,
    )
    sample_count_rejected: npt.NDArray[np.uint32] = attrs.field(
        converter=as_readonly_view,
        validator=uint32_array,
    )
    max_inter_sample_gap_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=int64_array,
    )
    integration_valid: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=bool_array,
    )
    integration_invalid_reason: npt.NDArray[np.uint32] = attrs.field(
        converter=as_readonly_view,
        validator=_invalid_reason_array,
    )

    def __attrs_post_init__(self) -> None:
        """Validate cross-field interval, accounting, and identity-row contracts."""
        _validate_batch_lengths(self)
        _validate_intervals(self)
        _validate_counts_and_quality(self)
        _validate_initial_row(self)
