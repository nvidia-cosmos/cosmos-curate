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

import numpy as np
import numpy.typing as npt


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
