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
"""Unit tests for sensor validation helpers."""

import numpy as np
import pytest

from cosmos_curate.core.sensors.utils.validation import require_strictly_increasing


def test_require_strictly_increasing_accepts_sorted_values() -> None:
    """Strictly ascending arrays should pass validation."""
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
    with pytest.raises(ValueError, match="values must be strictly sorted in ascending order with no duplicates"):
        require_strictly_increasing("values", values)
