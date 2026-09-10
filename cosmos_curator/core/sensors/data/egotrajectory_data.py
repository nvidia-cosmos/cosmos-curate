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
"""Trajectory data structures for cosmos_curator.core.sensors package.

``EgoTrajectory`` is the SoA partner for ``EgotrajectorySample`` in
``core/sensors/schemas/egotrajectory.proto``. Each row is one sensor-origin
homogeneous transform ``T_worldENU_from_sensorBody`` (clip-local ENU world).
The same batch type also sits in ``AlignedFrame`` for LiDAR motion compensation.
"""

from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_optional_readonly_view, as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    bool_batch,
    nondecreasing_int64_array,
    nonempty_str,
    optional_int64_array,
    optional_uint64_array,
    strictly_increasing_int64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_POSES_TAIL_SHAPE = (4, 4)
_POSES_BATCH_NDIM = 3
_POSES_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_POSES_LAST_ROW_TOLERANCE = 1e-9
_IDENTITY_POSE = np.eye(4, dtype=np.float64)
_POSE_VALID_VALIDATOR = bool_batch(())


class _HasEgoBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    poses: npt.NDArray[np.float64]
    pose_valid: npt.NDArray[np.bool_]
    host_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None
    frame: str


def _poses_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate an ``(N, 4, 4)`` ``float64`` pose batch with a homogeneous last row."""
    if value.dtype != np.float64:
        msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)
    if value.ndim != _POSES_BATCH_NDIM or value.shape[1:] != _POSES_TAIL_SHAPE:
        msg = f"{attribute.name} must have shape (N, 4, 4), got shape={value.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(value)):
        msg = f"{attribute.name} must contain only finite values"
        raise ValueError(msg)
    if value.shape[0] and not np.allclose(
        value[:, 3, :],
        _POSES_LAST_ROW,
        rtol=0.0,
        atol=_POSES_LAST_ROW_TOLERANCE,
    ):
        msg = f"{attribute.name} last row of each (4, 4) must equal [0, 0, 0, 1] within tolerance"
        raise ValueError(msg)


def _ego_batch_lengths(
    instance: _HasEgoBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate the shared row-count ``N`` invariant for :class:`EgoTrajectory`."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "poses": len(instance.poses),
        "pose_valid": len(instance.pose_valid),
    }
    if instance.host_timestamps_ns is not None:
        lengths["host_timestamps_ns"] = len(instance.host_timestamps_ns)
    if instance.sequence_counter is not None:
        lengths["sequence_counter"] = len(instance.sequence_counter)
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


def _invalid_poses_are_identity(
    instance: _HasEgoBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Require identity transforms for every row marked ``pose_valid=false``."""
    invalid = ~instance.pose_valid
    if not np.any(invalid):
        return
    if not np.allclose(
        instance.poses[invalid],
        _IDENTITY_POSE,
        rtol=0.0,
        atol=_POSES_LAST_ROW_TOLERANCE,
    ):
        msg = "poses rows with pose_valid=false must be identity transforms"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class EgoTrajectory:
    """Sensor-origin ego pose batch paired with ``EgotrajectorySample``.

    ``poses`` stores the proto's ``transform_world_enu_from_sensor_body`` as
    ``(N, 4, 4)`` row-major homogeneous transforms. Invalid rows keep an
    identity transform and ``pose_valid=false`` (keep-and-mask).

    ``align_timestamps_ns`` and ``frame`` are sensor-library fields not present
    on the wire schema. ``frame`` names the clip-local ENU world (typically
    ``"world_enu"``).

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
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
    poses: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_poses_batch,
    )
    pose_valid: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_POSE_VALID_VALIDATOR,
    )
    frame: str = attrs.field(validator=nonempty_str)
    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=optional_int64_array,
    )
    # sequence_counter is last so batch and keep-and-mask checks see every field.
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.and_(
            optional_uint64_array,
            _ego_batch_lengths,
            _invalid_poses_are_identity,
        ),
    )
