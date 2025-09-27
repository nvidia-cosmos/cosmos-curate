# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for data_model utility functions."""

import sys
from collections import deque
from unittest.mock import MagicMock, patch

import attrs
import numpy as np

from cosmos_curate.pipelines.video.utils.data_model import (
    _add_children_to_queue,
    _get_object_size,
    get_major_size,
)


class TestGetObjectSize:
    """Test _get_object_size function."""

    def test_numpy_array(self) -> None:
        """Test size calculation for numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        assert _get_object_size(arr) == arr.nbytes

    def test_numpy_scalar(self) -> None:
        """Test size calculation for numpy scalars."""
        scalar = np.int32(42)
        assert _get_object_size(scalar) == scalar.nbytes

    def test_torch_tensor(self) -> None:
        """Test size calculation for torch tensors."""
        # Mock torch tensor
        mock_tensor = MagicMock()
        element_size = 4
        num_elements = 10
        expected_size = element_size * num_elements
        mock_tensor.element_size.return_value = element_size
        mock_tensor.nelement.return_value = num_elements

        # Create a mock tensor type
        mock_tensor_type = type(mock_tensor)

        with patch("cosmos_curate.pipelines.video.utils.data_model.TensorType", mock_tensor_type):
            result = _get_object_size(mock_tensor)
            assert result == expected_size

    def test_regular_object(self) -> None:
        """Test size calculation for regular Python objects."""
        obj = "test string"
        expected_size = sys.getsizeof(obj)
        assert _get_object_size(obj) == expected_size

    def test_dict(self) -> None:
        """Test size calculation for dictionaries."""
        obj = {"key": "value"}
        # Dictionaries return 0 since we only count contents, not the container
        expected_size = 0
        assert _get_object_size(obj) == expected_size

    def test_list(self) -> None:
        """Test size calculation for lists."""
        obj = [np.array([1, 2, 3, 4, 5])]
        # lists return 0 since we only count contents, not the container
        expected_size = 0
        assert _get_object_size(obj) == expected_size


class TestAddChildrenToQueue:
    """Test _add_children_to_queue function."""

    def test_dict_children(self) -> None:
        """Test adding children from dictionary."""
        obj = {"a": 1, "b": 2}
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 2
        assert len(q) == expected_count
        assert obj["a"] in q
        assert obj["b"] in q

    def test_list_children(self) -> None:
        """Test adding children from list."""
        obj = [1, 2, 3]
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 3
        assert len(q) == expected_count
        assert obj[0] in q
        assert obj[1] in q
        assert obj[2] in q

    def test_tuple_children(self) -> None:
        """Test adding children from tuple."""
        obj = (1, 2, 3)
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 3
        assert len(q) == expected_count
        assert obj[0] in q
        assert obj[1] in q
        assert obj[2] in q

    def test_attrs_object_children(self) -> None:
        """Test adding children from attrs object."""

        @attrs.define
        class TestClass:
            field1: int = 1
            field2: str = "test"
            stage_perf: dict[str, str] = attrs.Factory(dict)  # Should be skipped

        obj = TestClass()
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 2
        assert len(q) == expected_count
        assert 1 in q
        assert "test" in q

    def test_visited_objects_skipped(self) -> None:
        """Test that already visited objects are skipped."""
        shared_obj = "shared"
        obj = {"a": shared_obj, "b": shared_obj}
        q: deque[object] = deque()
        visited: set[int] = {id(shared_obj)}

        _add_children_to_queue(obj, q, visited)

        # Should not add shared_obj since it's already visited
        assert len(q) == 0

    def test_non_attrs_object(self) -> None:
        """Test handling of non-attrs objects."""
        obj = "simple string"
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        # Should not add any children for simple objects
        assert len(q) == 0


class TestEstimateMajorSize:
    """Test get_major_size function."""

    def test_simple_object(self) -> None:
        """Test size estimation for simple objects."""
        obj = "test string"
        expected_size = sys.getsizeof(obj)
        assert get_major_size(obj) == expected_size

    def test_numpy_array(self) -> None:
        """Test size estimation for numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected_size = arr.nbytes
        assert get_major_size(arr) == expected_size

    def test_dict_with_values(self) -> None:
        """Test size estimation for dictionaries with values."""
        obj = {"key1": "value1", "key2": "value2"}
        # Should only include size of values, not the dict itself
        value1_size = sys.getsizeof("value1")
        value2_size = sys.getsizeof("value2")
        expected_size = value1_size + value2_size

        assert get_major_size(obj) == expected_size

    def test_list_with_items(self) -> None:
        """Test size estimation for lists with items."""
        obj = [1, 2, 3]
        # Should only include size of items, not the list itself
        item_size = sys.getsizeof(1)  # All items are same size
        expected_size = item_size * 3

        assert get_major_size(obj) == expected_size

    def test_attrs_object(self) -> None:
        """Test size estimation for attrs objects."""

        @attrs.define
        class TestClass:
            field1: int = 42
            field2: str = "test"
            stage_perf: dict[str, str] = attrs.Factory(dict)  # Should be skipped

        obj = TestClass()
        # Should include size of the attrs object itself plus field1 and field2, but not stage_perf
        obj_size = sys.getsizeof(obj)
        field1_size = sys.getsizeof(42)
        field2_size = sys.getsizeof("test")
        expected_size = obj_size + field1_size + field2_size

        assert get_major_size(obj) == expected_size

    def test_nested_structures(self) -> None:
        """Test size estimation for nested data structures."""
        inner_dict = {"inner": "value"}
        outer_dict = {"outer": inner_dict}

        # Should only include size of the inner value, not the dicts themselves
        value_size = sys.getsizeof("value")
        expected_size = value_size

        assert get_major_size(outer_dict) == expected_size

    def test_circular_reference(self) -> None:
        """Test that circular references don't cause infinite loops."""
        obj: dict[str, object] = {}
        obj["self"] = obj  # Create circular reference

        # Should not hang and should return 0 since dicts return 0
        size = get_major_size(obj)
        assert size == 0
        assert isinstance(size, int)

    def test_mixed_data_types(self) -> None:
        """Test size estimation for mixed data types."""
        arr = np.array([1, 2, 3])
        obj = {"array": arr, "string": "test", "number": 42, "nested": {"inner": "value"}}

        # Should include sizes of all components (not the containers themselves)
        arr_size = arr.nbytes
        string_size = sys.getsizeof("test")
        number_size = sys.getsizeof(42)
        inner_value_size = sys.getsizeof("value")
        expected_size = arr_size + string_size + number_size + inner_value_size

        size = get_major_size(obj)
        assert size == expected_size

    def test_empty_containers(self) -> None:
        """Test size estimation for empty containers."""
        empty_dict: dict[str, str] = {}
        empty_list: list[str] = []
        empty_tuple: tuple[()] = ()
        dict_size = get_major_size(empty_dict)
        list_size = get_major_size(empty_list)
        tuple_size = get_major_size(empty_tuple)

        assert dict_size == 0
        assert list_size == 0
        assert tuple_size == 0

    def test_large_numpy_array(self) -> None:
        """Test size estimation for large numpy arrays."""
        rng = np.random.default_rng(42)
        large_arr = rng.random((1000, 1000))
        expected_size = large_arr.nbytes

        assert get_major_size(large_arr) == expected_size

    def test_torch_tensor_integration(self) -> None:
        """Test size estimation with torch tensors."""
        # Mock torch tensor
        mock_tensor = MagicMock()
        element_size = 4
        num_elements = 100
        expected_size = element_size * num_elements
        mock_tensor.element_size.return_value = element_size
        mock_tensor.nelement.return_value = num_elements

        # Create a mock tensor type
        mock_tensor_type = type(mock_tensor)

        with patch("cosmos_curate.pipelines.video.utils.data_model.TensorType", mock_tensor_type):
            size = get_major_size(mock_tensor)
            assert size == expected_size
