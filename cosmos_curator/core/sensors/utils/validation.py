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
"""Validation helpers for sensor-library data structures and algorithms."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

AttrsValidator = Callable[[object, AttrsAttribute, Any], None]

INT64_MIN = int(np.iinfo(np.int64).min)
INT64_MAX = int(np.iinfo(np.int64).max)

_QUATERNION_COLUMNS = 4
_QUATERNION_NORM_TOLERANCE = 1e-6
_COVARIANCE_TOLERANCE = 1e-9


def _require_1d(name: str, values: npt.NDArray[Any], dtype: npt.DTypeLike) -> None:
    """Raise if *values* is not a 1-D array with the expected dtype."""
    expected = np.dtype(dtype)
    if values.ndim != 1:
        msg = f"{name} must be 1-D, got ndim={values.ndim}"
        raise ValueError(msg)
    if values.dtype != expected:
        msg = f"{name} must have dtype {expected.name}, got {values.dtype}"
        raise ValueError(msg)


def dtype_array(dtype: npt.DTypeLike) -> AttrsValidator:
    """Build an attrs validator for a 1-D array with *dtype*."""

    def _validator(
        _instance: object,
        attribute: AttrsAttribute,
        value: npt.NDArray[Any],
    ) -> None:
        _require_1d(attribute.name, value, dtype)

    return _validator


int64_array = dtype_array(np.int64)
bool_array = dtype_array(np.bool_)
uint8_array = dtype_array(np.uint8)
uint16_array = dtype_array(np.uint16)
uint32_array = dtype_array(np.uint32)
uint64_array = dtype_array(np.uint64)
float64_array = dtype_array(np.float64)
optional_int64_array = attrs.validators.optional(int64_array)
optional_bool_array = attrs.validators.optional(bool_array)
optional_uint8_array = attrs.validators.optional(uint8_array)
optional_uint16_array = attrs.validators.optional(uint16_array)
optional_uint32_array = attrs.validators.optional(uint32_array)
optional_uint64_array = attrs.validators.optional(uint64_array)
optional_float64_array = attrs.validators.optional(float64_array)


def require_finite_float64_array(name: str, values: npt.NDArray[np.float64]) -> None:
    """Raise if *values* is not a finite ``float64`` array."""
    if values.dtype != np.float64:
        msg = f"{name} must have dtype float64, got {values.dtype}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)


def require_finite_float32_array(name: str, values: npt.NDArray[np.float32]) -> None:
    """Raise if *values* is not a finite ``float32`` array."""
    if values.dtype != np.float32:
        msg = f"{name} must have dtype float32, got {values.dtype}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)


def require_batch_shape(
    name: str,
    values: npt.NDArray[Any],
    trailing_shape: tuple[int, ...],
) -> None:
    """Raise unless an array has a batch dimension followed by ``trailing_shape``."""
    expected_ndim = len(trailing_shape) + 1
    if values.ndim != expected_ndim or values.shape[1:] != trailing_shape:
        dimensions = ", ".join(str(item) for item in trailing_shape)
        msg = f"{name} must have shape (N, {dimensions}), got shape={values.shape}"
        raise ValueError(msg)


def float64_batch(
    trailing_shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> AttrsValidator:
    """Build an attrs validator for float64 batches with a fixed trailing shape."""

    def _validator(
        _instance: object,
        attribute: AttrsAttribute,
        value: npt.NDArray[np.float64],
    ) -> None:
        require_batch_shape(attribute.name, value, trailing_shape)
        if value.dtype != np.float64:
            msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
            raise ValueError(msg)
        if finite and not np.all(np.isfinite(value)):
            msg = f"{attribute.name} must contain only finite values"
            raise ValueError(msg)

    return _validator


def bool_batch(trailing_shape: tuple[int, ...]) -> AttrsValidator:
    """Build an attrs validator for bool batches with a fixed trailing shape."""

    def _validator(
        _instance: object,
        attribute: AttrsAttribute,
        value: npt.NDArray[np.bool_],
    ) -> None:
        require_batch_shape(attribute.name, value, trailing_shape)
        if value.dtype != np.bool_:
            msg = f"{attribute.name} must have dtype bool, got {value.dtype}"
            raise ValueError(msg)

    return _validator


def unit_quaternion_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate finite unit quaternions with shape ``(N, 4)``."""
    require_batch_shape(attribute.name, value, (_QUATERNION_COLUMNS,))
    require_finite_float64_array(attribute.name, value)
    norms = np.linalg.norm(value, axis=1)
    if not np.all(np.isclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_TOLERANCE)):
        msg = f"{attribute.name} quaternion rows must have unit norm within tolerance"
        raise ValueError(msg)


def symmetric_psd_covariance_batch(
    matrix_size: int,
    *,
    symmetry_rtol: float = 0.0,
) -> AttrsValidator:
    """Build an attrs validator for finite symmetric PSD covariance batches."""
    if matrix_size <= 0:
        msg = f"matrix_size must be positive, got {matrix_size}"
        raise ValueError(msg)
    if symmetry_rtol < 0.0:
        msg = f"symmetry_rtol must be nonnegative, got {symmetry_rtol}"
        raise ValueError(msg)

    def _validator(
        _instance: object,
        attribute: AttrsAttribute,
        value: npt.NDArray[np.float64],
    ) -> None:
        require_batch_shape(attribute.name, value, (matrix_size, matrix_size))
        require_finite_float64_array(attribute.name, value)
        if not np.allclose(
            value,
            np.swapaxes(value, 1, 2),
            rtol=symmetry_rtol,
            atol=_COVARIANCE_TOLERANCE,
        ):
            msg = f"{attribute.name} covariance matrices must be symmetric"
            raise ValueError(msg)
        if value.shape[0] and np.min(np.linalg.eigvalsh(value)) < -_COVARIANCE_TOLERANCE:
            msg = f"{attribute.name} covariance matrices must be positive semidefinite"
            raise ValueError(msg)

    return _validator


def require_strictly_increasing(name: str, values: npt.NDArray[np.int64]) -> None:
    """Raise if *values* is not strictly sorted in ascending order."""
    if len(values) > 1 and not np.all(values[:-1] < values[1:]):
        msg = f"{name} must be strictly sorted in ascending order with no duplicates"
        raise ValueError(msg)


def require_nondecreasing(name: str, values: npt.NDArray[np.int64]) -> None:
    """Raise if *values* is not sorted in ascending order allowing duplicates."""
    if len(values) > 1 and not np.all(values[:-1] <= values[1:]):
        msg = f"{name} must be sorted in ascending order"
        raise ValueError(msg)


def strictly_increasing_int64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator for a 1-D strictly increasing ``int64`` array."""
    _require_1d(attribute.name, value, np.int64)
    require_strictly_increasing(attribute.name, value)


def nondecreasing_int64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64],
) -> None:
    """Attrs validator for a 1-D nondecreasing ``int64`` array."""
    _require_1d(attribute.name, value, np.int64)
    require_nondecreasing(attribute.name, value)


def finite_float64_array(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Attrs validator for a finite ``float64`` array."""
    require_finite_float64_array(attribute.name, value)


def finite_float32_vector(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float32],
) -> None:
    """Attrs validator for a finite 1-D ``float32`` array."""
    _require_1d(attribute.name, value, np.float32)
    require_finite_float32_array(attribute.name, value)


def finite_float32_point_batch(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float32],
) -> None:
    """Attrs validator for a finite ``float32`` point batch with shape ``(N, 3)``."""
    point_batch_ndim = 2
    point_dims = 3
    if value.ndim != point_batch_ndim or value.shape[1:] != (point_dims,):
        msg = f"{attribute.name} must have shape (N, 3), got shape={value.shape}"
        raise ValueError(msg)
    require_finite_float32_array(attribute.name, value)


def optional_finite_float32_vector(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float32] | None,
) -> None:
    """Attrs validator for an optional finite 1-D ``float32`` array."""
    if value is None:
        return
    _require_1d(attribute.name, value, np.float32)
    require_finite_float32_array(attribute.name, value)


def require_finite_or_marked_invalid(
    name: str,
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_] | None,
) -> None:
    """Allow non-finite values only when a matching validity mask entry is false."""
    if validity is not None and validity.shape != values.shape:
        msg = f"{name} validity mask must have shape {values.shape}, got {validity.shape}"
        raise ValueError(msg)
    nonfinite = ~np.isfinite(values)
    if not np.any(nonfinite):
        return
    if validity is None:
        msg = f"{name} must contain only finite values when no validity mask is provided"
        raise ValueError(msg)
    if np.any(nonfinite & validity):
        msg = f"{name} non-finite values require matching validity mask entries to be false"
        raise ValueError(msg)


def nonempty_str(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: object,
) -> None:
    """Attrs validator for a non-empty ``str`` (attrs boundary; ``value`` is untyped input)."""
    if not isinstance(value, str):
        msg = f"{attribute.name} must be a string, got {type(value).__name__}"
        raise ValueError(msg)  # noqa: TRY004 -- ValueError matches project convention for invalid attrs input
    if not value:
        msg = f"{attribute.name} must be a non-empty string"
        raise ValueError(msg)


def optional_nonempty_str(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: object,
) -> None:
    """Attrs validator for an optional non-empty ``str`` (attrs boundary; ``value`` is untyped input)."""
    if value is None:
        return
    if not isinstance(value, str):
        msg = f"{attribute.name} must be a string, got {type(value).__name__}"
        raise ValueError(msg)  # noqa: TRY004 -- ValueError matches project convention for invalid attrs input
    if not value:
        msg = f"{attribute.name} must be a non-empty string"
        raise ValueError(msg)


def uint8_frame_batch(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8],
) -> None:
    """Attrs validator for a 4-D ``uint8`` frame batch."""
    frame_ndim = 4
    if value.ndim != frame_ndim:
        msg = f"{attribute.name} must be 4-D with shape (N, H, W, 3), got ndim={value.ndim}"
        raise ValueError(msg)
    if value.dtype != np.uint8:
        msg = f"{attribute.name} must have dtype uint8, got {value.dtype}"
        raise ValueError(msg)


def positive_value(
    instance: object,  # noqa: ARG001
    attribute: AttrsAttribute,
    value: int,
) -> None:
    """Attrs validator ensuring a value is positive."""
    if value <= 0:
        msg = f"{attribute.name} must be positive, got {value=}"
        raise ValueError(msg)
