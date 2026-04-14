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

"""Aligned frame data structures for cosmos_curate.core.sensors package."""

from collections.abc import Mapping
from types import MappingProxyType

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curate.core.sensors.data.sensor_data import SensorData
from cosmos_curate.core.sensors.utils.helpers import make_numpy_fields_readonly
from cosmos_curate.core.sensors.utils.validation import require_strictly_increasing


@attrs.define(hash=False, frozen=True)
class AlignedFrame:
    """Single bundle of sensor data aligned to a reference timeline.

    Attributes:
        timestamps_ns: shared 1-D reference timestamps (ns); each sensor's timestamps_ns must match this array exactly
        sensor_data: a mapping of sensor data by sensor id

    """

    __hash__ = None  # type: ignore[assignment]

    timestamps_ns: npt.NDArray[np.int64]  # sampled timestamps, shape (N,) nanoseconds
    sensor_data: Mapping[str, SensorData]

    def __attrs_post_init__(self) -> None:
        """Post-initialization."""
        if self.timestamps_ns.ndim != 1:
            msg = f"timestamps_ns must be 1-D, got ndim={self.timestamps_ns.ndim}"
            raise ValueError(msg)
        require_strictly_increasing("timestamps_ns", self.timestamps_ns)
        make_numpy_fields_readonly(self)
        expected_len = len(self.timestamps_ns)
        sensor_data = dict(self.sensor_data)
        for sensor_id, data in sensor_data.items():
            if len(data.timestamps_ns) != expected_len:
                msg = (
                    f"sensor {sensor_id!r} timestamps_ns length {len(data.timestamps_ns)} "
                    f"!= aligned frame length {expected_len}"
                )
                raise ValueError(msg)
            if not np.array_equal(data.timestamps_ns, self.timestamps_ns):
                msg = f"sensor {sensor_id!r} timestamps_ns must exactly match aligned frame timestamps_ns"
                raise ValueError(msg)
            if len(data.canonical_timestamps_ns) != expected_len:
                msg = (
                    f"sensor {sensor_id!r} canonical_timestamps_ns length "
                    f"{len(data.canonical_timestamps_ns)} != aligned frame length {expected_len}"
                )
                raise ValueError(msg)

        # Make sensor_data immutable
        object.__setattr__(self, "sensor_data", MappingProxyType(sensor_data))

    def __getitem__(self, key: str) -> SensorData:
        """Get sensor data by key."""
        return self.sensor_data[key]

    def __contains__(self, key: object) -> bool:
        """Check if sensor data exists by key."""
        return key in self.sensor_data
