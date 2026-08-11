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
"""Unit tests for public sensor validation helpers."""

import attrs
import numpy as np
import pytest

from cosmos_curator.core.sensors.utils.validation import (
    bool_array,
    finite_float64_array,
    float64_array,
    int64_array,
    nondecreasing_int64_array,
    optional_bool_array,
    optional_float64_array,
    optional_int64_array,
    optional_uint32_array,
    optional_uint64_array,
    require_finite_or_marked_invalid,
    require_strictly_increasing,
    strictly_increasing_int64_array,
    uint8_array,
    uint8_frame_batch,
    uint32_array,
    uint64_array,
)


def test_require_strictly_increasing_accepts_sorted_values() -> None:
    """Strictly ascending arrays should pass the public ordering helper."""
    values = np.array([0, 10, 20], dtype=np.int64)

    require_strictly_increasing("values", values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 10, 10], dtype=np.int64),
        np.array([0, 20, 10], dtype=np.int64),
    ],
)
def test_require_strictly_increasing_rejects_non_increasing_values(values: np.ndarray) -> None:
    """Duplicate or descending values should raise ValueError."""
    with pytest.raises(ValueError, match="strictly sorted"):
        require_strictly_increasing("values", values)


@attrs.define
class _StrictlyIncreasingArrayHolder:
    """Test fixture for the strictly increasing int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=strictly_increasing_int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([100, 200, 200], dtype=np.int64), "strictly sorted"),
        (np.array([[100, 200, 300]], dtype=np.int64), "1-D"),
        (np.array([100, 200, 300], dtype=np.int32), "dtype int64"),
    ],
)
def test_strictly_increasing_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Strict validator should reject invalid ordering, rank, or dtype."""
    with pytest.raises(ValueError, match=match):
        _StrictlyIncreasingArrayHolder(values=values)


def test_strictly_increasing_int64_array_accepts_strictly_increasing_int64_vector() -> None:
    """Strict validator should accept 1-D int64 arrays with strictly increasing values."""
    values = np.array([100, 200, 300], dtype=np.int64)
    holder = _StrictlyIncreasingArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _NondecreasingArrayHolder:
    """Test fixture for the nondecreasing int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=nondecreasing_int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([100, 300, 200], dtype=np.int64), "sorted in ascending order"),
        (np.array([[100, 200, 300]], dtype=np.int64), "1-D"),
        (np.array([100, 200, 300], dtype=np.int32), "dtype int64"),
    ],
)
def test_nondecreasing_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Nondecreasing validator should reject descending values, rank mismatches, and dtype mismatches."""
    with pytest.raises(ValueError, match=match):
        _NondecreasingArrayHolder(values=values)


def test_nondecreasing_int64_array_accepts_nondecreasing_int64_vector() -> None:
    """Nondecreasing validator should allow duplicate timestamps."""
    values = np.array([100, 200, 200, 300], dtype=np.int64)
    holder = _NondecreasingArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _BoolArrayHolder:
    """Test fixture for the bool array attrs validator."""

    values: np.ndarray = attrs.field(validator=bool_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[True, False]], dtype=np.bool_), "1-D"),
        (np.array([1, 0], dtype=np.int64), "dtype bool"),
    ],
)
def test_bool_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Bool-array validator should reject non-vector or non-bool arrays."""
    with pytest.raises(ValueError, match=match):
        _BoolArrayHolder(values=values)


def test_bool_array_accepts_bool_vector() -> None:
    """Bool-array validator should accept a 1-D bool array."""
    values = np.array([True, False], dtype=np.bool_)
    holder = _BoolArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _OptionalBoolArrayHolder:
    """Test fixture for the optional bool array attrs validator."""

    values: np.ndarray | None = attrs.field(validator=optional_bool_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[True, False]], dtype=np.bool_), "1-D"),
        (np.array([1, 0], dtype=np.int64), "dtype bool"),
    ],
)
def test_optional_bool_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Optional bool-array validator should reject non-vector or non-bool arrays."""
    with pytest.raises(ValueError, match=match):
        _OptionalBoolArrayHolder(values=values)


def test_optional_bool_array_accepts_none_and_bool_vector() -> None:
    """Optional bool-array validator should accept None or a 1-D bool array."""
    assert _OptionalBoolArrayHolder(values=None).values is None
    values = np.array([True, False], dtype=np.bool_)
    holder = _OptionalBoolArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Int64ArrayHolder:
    """Test fixture for the int64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.int64), "1-D"),
        (np.array([1, 2], dtype=np.int32), "dtype int64"),
    ],
)
def test_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Int64-array validator should reject non-vector or non-int64 arrays."""
    with pytest.raises(ValueError, match=match):
        _Int64ArrayHolder(values=values)


def test_int64_array_accepts_int64_vector() -> None:
    """Int64-array validator should accept a 1-D int64 array."""
    values = np.array([1, 2], dtype=np.int64)
    holder = _Int64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _OptionalInt64ArrayHolder:
    """Test fixture for the optional int64 array attrs validator."""

    values: np.ndarray | None = attrs.field(validator=optional_int64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.int64), "1-D"),
        (np.array([1, 2], dtype=np.int32), "dtype int64"),
    ],
)
def test_optional_int64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Optional int64-array validator should reject non-vector or non-int64 arrays."""
    with pytest.raises(ValueError, match=match):
        _OptionalInt64ArrayHolder(values=values)


def test_optional_int64_array_accepts_none_and_int64_vector() -> None:
    """Optional int64-array validator should accept None or a 1-D int64 array."""
    assert _OptionalInt64ArrayHolder(values=None).values is None
    values = np.array([1, 2], dtype=np.int64)
    holder = _OptionalInt64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _OptionalUint64ArrayHolder:
    """Test fixture for the optional uint64 array attrs validator."""

    values: np.ndarray | None = attrs.field(validator=optional_uint64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint64), "1-D"),
        (np.array([1, 2], dtype=np.int64), "dtype uint64"),
    ],
)
def test_optional_uint64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Optional uint64-array validator should reject non-vector or non-uint64 arrays."""
    with pytest.raises(ValueError, match=match):
        _OptionalUint64ArrayHolder(values=values)


def test_optional_uint64_array_accepts_none_and_uint64_vector() -> None:
    """Optional uint64-array validator should accept None or a 1-D uint64 array."""
    assert _OptionalUint64ArrayHolder(values=None).values is None
    values = np.array([1, 2], dtype=np.uint64)
    holder = _OptionalUint64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Float64ArrayHolder:
    """Test fixture for the float64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=float64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1.0, 2.0]], dtype=np.float64), "1-D"),
        (np.array([1.0, 2.0], dtype=np.float32), "dtype float64"),
    ],
)
def test_float64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Float64-array validator should reject non-vector or non-float64 arrays."""
    with pytest.raises(ValueError, match=match):
        _Float64ArrayHolder(values=values)


def test_float64_array_accepts_nonfinite_float64_vector() -> None:
    """Float64-array validator should accept NaN/Inf so keep-and-mask can retain placeholders."""
    values = np.array([1.0, np.nan, np.inf], dtype=np.float64)
    holder = _Float64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _OptionalFloat64ArrayHolder:
    """Test fixture for the optional float64 array attrs validator."""

    values: np.ndarray | None = attrs.field(validator=optional_float64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1.0, 2.0]], dtype=np.float64), "1-D"),
        (np.array([1.0, 2.0], dtype=np.float32), "dtype float64"),
    ],
)
def test_optional_float64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Optional float64-array validator should reject non-vector or non-float64 arrays."""
    with pytest.raises(ValueError, match=match):
        _OptionalFloat64ArrayHolder(values=values)


def test_optional_float64_array_accepts_none_and_float64_vector() -> None:
    """Optional float64-array validator should accept None or a 1-D float64 array."""
    assert _OptionalFloat64ArrayHolder(values=None).values is None
    values = np.array([1.0, np.nan], dtype=np.float64)
    holder = _OptionalFloat64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _OptionalUint32ArrayHolder:
    """Test fixture for the optional uint32 array attrs validator."""

    values: np.ndarray | None = attrs.field(validator=optional_uint32_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint32), "1-D"),
        (np.array([1, 2], dtype=np.uint8), "dtype uint32"),
    ],
)
def test_optional_uint32_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Optional uint32-array validator should reject non-vector or non-uint32 arrays."""
    with pytest.raises(ValueError, match=match):
        _OptionalUint32ArrayHolder(values=values)


def test_optional_uint32_array_accepts_none_and_uint32_vector() -> None:
    """Optional uint32-array validator should accept None or a 1-D uint32 array."""
    assert _OptionalUint32ArrayHolder(values=None).values is None
    values = np.array([1, 2], dtype=np.uint32)
    holder = _OptionalUint32ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


def test_require_finite_or_marked_invalid_allows_masked_nonfinite() -> None:
    """Non-finite values are allowed only where the validity mask is false."""
    values = np.array([1.0, np.nan, np.inf], dtype=np.float64)
    validity = np.array([True, False, False], dtype=np.bool_)
    require_finite_or_marked_invalid("values", values, validity)


def test_require_finite_or_marked_invalid_rejects_unmasked_nonfinite() -> None:
    """Non-finite values with true validity entries should raise."""
    values = np.array([1.0, np.nan], dtype=np.float64)
    validity = np.array([True, True], dtype=np.bool_)
    with pytest.raises(ValueError, match="validity mask"):
        require_finite_or_marked_invalid("values", values, validity)


def test_require_finite_or_marked_invalid_rejects_nonfinite_without_mask() -> None:
    """Without a validity mask, all values must be finite."""
    values = np.array([1.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="no validity mask"):
        require_finite_or_marked_invalid("values", values, None)


@attrs.define
class _Uint8ArrayHolder:
    """Test fixture for the uint8 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint8_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint8), "1-D"),
        (np.array([1, 2], dtype=np.uint32), "dtype uint8"),
    ],
)
def test_uint8_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint8-array validator should reject non-vector or non-uint8 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint8ArrayHolder(values=values)


def test_uint8_array_accepts_uint8_vector() -> None:
    """Uint8-array validator should accept a 1-D uint8 array."""
    values = np.array([1, 2], dtype=np.uint8)
    holder = _Uint8ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Uint32ArrayHolder:
    """Test fixture for the uint32 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint32_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint32), "1-D"),
        (np.array([1, 2], dtype=np.uint8), "dtype uint32"),
    ],
)
def test_uint32_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint32-array validator should reject non-vector or non-uint32 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint32ArrayHolder(values=values)


def test_uint32_array_accepts_uint32_vector() -> None:
    """Uint32-array validator should accept a 1-D uint32 array."""
    values = np.array([1, 2], dtype=np.uint32)
    holder = _Uint32ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _Uint64ArrayHolder:
    """Test fixture for the uint64 array attrs validator."""

    values: np.ndarray = attrs.field(validator=uint64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([[1, 2]], dtype=np.uint64), "1-D"),
        (np.array([1, 2], dtype=np.int64), "dtype uint64"),
    ],
)
def test_uint64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Uint64-array validator should reject non-vector or non-uint64 arrays."""
    with pytest.raises(ValueError, match=match):
        _Uint64ArrayHolder(values=values)


def test_uint64_array_accepts_uint64_vector() -> None:
    """Uint64-array validator should accept a 1-D uint64 array."""
    values = np.array([1, 2], dtype=np.uint64)
    holder = _Uint64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _FiniteFloat64ArrayHolder:
    """Test fixture for the finite float64 attrs validator."""

    values: np.ndarray = attrs.field(validator=finite_float64_array)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([0.0, 1.0], dtype=np.float32), "dtype float64"),
        (np.array([0.0, np.inf], dtype=np.float64), "finite"),
        (np.array([0.0, np.nan], dtype=np.float64), "finite"),
    ],
)
def test_finite_float64_array_rejects_invalid_inputs(values: np.ndarray, match: str) -> None:
    """Finite-float validator should reject non-float64 or non-finite arrays."""
    with pytest.raises(ValueError, match=match):
        _FiniteFloat64ArrayHolder(values=values)


def test_finite_float64_array_accepts_finite_float64_array() -> None:
    """Finite-float validator should accept finite float64 arrays of any rank."""
    values = np.ones((2, 3), dtype=np.float64)
    holder = _FiniteFloat64ArrayHolder(values=values)
    np.testing.assert_array_equal(holder.values, values)


@attrs.define
class _FrameBatchHolder:
    """Test fixture for the uint8 frame batch attrs validator."""

    frames: np.ndarray = attrs.field(validator=uint8_frame_batch)


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (np.zeros((1, 3), dtype=np.uint8), r"4-D with shape \(N, H, W, 3\)"),
        (np.zeros((1, 1, 1, 3), dtype=np.float32), "dtype uint8"),
    ],
)
def test_uint8_frame_batch_rejects_invalid_inputs(frames: np.ndarray, match: str) -> None:
    """Frame-batch validator should reject wrong rank or dtype."""
    with pytest.raises(ValueError, match=match):
        _FrameBatchHolder(frames=frames)


def test_uint8_frame_batch_accepts_uint8_4d_array() -> None:
    """Frame-batch validator should accept a 4-D uint8 array."""
    frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    holder = _FrameBatchHolder(frames=frames)
    np.testing.assert_array_equal(holder.frames, frames)
