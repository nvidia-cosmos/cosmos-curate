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
"""Grid-aligned sensor for eagerly preintegrated IMU data."""

from collections.abc import Generator

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.data.preintegrated_imu_data import PreintegratedImuData
from cosmos_curator.core.sensors.preintegration.imu_preintegrator import preintegrate_imu
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NoSamplingPolicy, require_no_sampling_policy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.imu_sensor import ImuSensor


def _slice_preintegrated_imu_data(
    data: PreintegratedImuData,
    row_slice: slice,
) -> PreintegratedImuData:
    """Return a row slice while preserving every structure-of-arrays field."""
    return PreintegratedImuData(
        align_timestamps_ns=data.align_timestamps_ns[row_slice],
        sensor_timestamps_ns=data.sensor_timestamps_ns[row_slice],
        align_interval_start_timestamps_ns=data.align_interval_start_timestamps_ns[row_slice],
        align_interval_end_timestamps_ns=data.align_interval_end_timestamps_ns[row_slice],
        integration_duration_ns=data.integration_duration_ns[row_slice],
        delta_rotation_quat_xyzw=data.delta_rotation_quat_xyzw[row_slice],
        delta_velocity_m_s=data.delta_velocity_m_s[row_slice],
        delta_position_m=data.delta_position_m[row_slice],
        angular_velocity_bias_used_rad_s=data.angular_velocity_bias_used_rad_s[row_slice],
        linear_acceleration_bias_used_m_s2=data.linear_acceleration_bias_used_m_s2[row_slice],
        angular_velocity_bias_available=data.angular_velocity_bias_available[row_slice],
        linear_acceleration_bias_available=data.linear_acceleration_bias_available[row_slice],
        sample_count_total=data.sample_count_total[row_slice],
        sample_count_used=data.sample_count_used[row_slice],
        sample_count_rejected=data.sample_count_rejected[row_slice],
        max_inter_sample_gap_ns=data.max_inter_sample_gap_ns[row_slice],
        integration_valid=data.integration_valid[row_slice],
        integration_invalid_reason=data.integration_invalid_reason[row_slice],
    )


def _window_row_slice(
    align_timestamps_ns: npt.NDArray[np.int64],
    window: SamplingWindow,
) -> slice:
    """Locate one sampling window as a contiguous full-grid row slice."""
    if not len(window.timestamps_ns):
        return slice(0, 0)
    start = int(np.searchsorted(align_timestamps_ns, window.timestamps_ns[0], side="left"))
    stop = start + len(window.timestamps_ns)
    if stop > len(align_timestamps_ns) or not np.array_equal(
        align_timestamps_ns[start:stop],
        window.timestamps_ns,
    ):
        msg = "SamplingWindow timestamps must be a contiguous subset of the preintegrated alignment grid"
        raise ValueError(msg)
    return slice(start, stop)


class PreintegratedImuSensor:
    """Wrap an ``ImuSensor`` with eager full-grid preintegration.

    Raw MCAP data is decoded once and retained in memory. Preintegration is
    cached for the most recently requested alignment timeline, then returned as
    exact per-window slices compatible with ``SensorGroup``.
    """

    def __init__(self, imu_sensor: ImuSensor) -> None:
        """Initialize from an MCAP-backed raw IMU sensor."""
        self._imu_sensor = imu_sensor
        self._raw_imu_data: ImuData | None = None
        self._cached_grid_timestamps_ns: npt.NDArray[np.int64] | None = None
        self._cached_preintegrated_data: PreintegratedImuData | None = None

    @property
    def start_ns(self) -> int:
        """Return the earliest source IMU message time."""
        return self._imu_sensor.start_ns

    @property
    def end_ns(self) -> int:
        """Return the latest source IMU message time."""
        return self._imu_sensor.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return the maximum source IMU message gap."""
        return self._imu_sensor.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Return source MCAP log timestamps."""
        return self._imu_sensor.timestamps_ns

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NoSamplingPolicy)

    def _get_raw_imu_data(self) -> ImuData:
        """Decode and cache the complete source recording."""
        if self._raw_imu_data is None:
            self._raw_imu_data = self._imu_sensor.read_all()
        return self._raw_imu_data

    def _get_preintegrated_data(
        self,
        align_timestamps_ns: npt.NDArray[np.int64],
    ) -> PreintegratedImuData:
        """Return cached full-grid preintegration or compute it once."""
        if self._cached_grid_timestamps_ns is None or not np.array_equal(
            self._cached_grid_timestamps_ns, align_timestamps_ns
        ):
            self._cached_preintegrated_data = preintegrate_imu(
                self._get_raw_imu_data(),
                align_timestamps_ns,
            )
            self._cached_grid_timestamps_ns = np.array(align_timestamps_ns, copy=True)
            self._cached_grid_timestamps_ns.flags.writeable = False
        assert self._cached_preintegrated_data is not None
        return self._cached_preintegrated_data

    def sample(self, spec: SamplingSpec, *, policy: NoSamplingPolicy) -> Generator[PreintegratedImuData]:
        """Yield one exact full-grid preintegration slice per sampling window."""
        require_no_sampling_policy(policy, sensor_name=type(self).__name__)
        full_data = self._get_preintegrated_data(spec.grid.timestamps_ns)
        for window in spec.grid:
            yield _slice_preintegrated_imu_data(
                full_data,
                _window_row_slice(full_data.align_timestamps_ns, window),
            )
