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
"""Tests for the eager preintegrated IMU sensor."""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors import preintegrated_imu_sensor as preintegrated_imu_sensor_module
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from cosmos_curator.core.sensors.sensors.imu_sensor import ImuSensor
from cosmos_curator.core.sensors.sensors.preintegrated_imu_sensor import PreintegratedImuSensor
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    protobuf_descriptor_set_from_proto,
    protobuf_message_class,
    write_protobuf_mcap,
)

_HALF_SECOND_NS = 500_000_000
_ONE_SECOND_NS = 1_000_000_000
_REPO_ROOT = Path(__file__).parents[5]
_REFERENCE_IMU_SCHEMA_NAME = "cosmos_curator.sensors.imu.v1.ImuSample"
_REFERENCE_IMU_PROTO_PATH = _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "schemas" / "imu.proto"
_REFERENCE_IMU_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "imu_protobuf_mapping.yaml"
)


class _FakeImuSensor:
    """In-memory stand-in exposing the wrapped sensor surface."""

    def __init__(self, data: ImuData) -> None:
        self.data = data
        self.read_count = 0

    @property
    def start_ns(self) -> int:
        return int(self.data.align_timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.data.align_timestamps_ns[-1])

    @property
    def max_gap_ns(self) -> int:
        return int(np.max(np.diff(self.data.align_timestamps_ns)))

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        return self.data.align_timestamps_ns

    def read_all(self) -> ImuData:
        """Return the complete recording and count eager decodes."""
        self.read_count += 1
        return self.data


def _raw_imu_data() -> ImuData:
    """Build a complete in-memory constant-motion recording."""
    raw_timestamps = np.arange(0, 2 * _ONE_SECOND_NS + 1, _HALF_SECOND_NS, dtype=np.int64)
    zeros = np.zeros((len(raw_timestamps), 3), dtype=np.float64)
    return ImuData(
        align_timestamps_ns=raw_timestamps,
        sensor_timestamps_ns=raw_timestamps,
        angular_velocity_rad_s=zeros,
        linear_acceleration_m_s2=zeros,
    )


def _multi_window_spec(timestamps_ns: npt.NDArray[np.int64]) -> SamplingSpec:
    """Build two-second windows over a supplied active timestamp grid."""
    return SamplingSpec(
        grid=SamplingGrid(
            start_ns=int(timestamps_ns[0]),
            exclusive_end_ns=3 * _ONE_SECOND_NS,
            timestamps_ns=timestamps_ns,
            stride_ns=2 * _ONE_SECOND_NS,
            duration_ns=2 * _ONE_SECOND_NS,
        ),
    )


def _reference_imu_payload(timestamp_ns: int) -> bytes:
    """Serialize one valid constant-acceleration reference IMU sample."""
    descriptor_set = protobuf_descriptor_set_from_proto(_REFERENCE_IMU_PROTO_PATH)
    message_cls = protobuf_message_class(descriptor_set, _REFERENCE_IMU_SCHEMA_NAME)
    sample = message_cls()
    sample.sensor_timestamp_ns = timestamp_ns
    sample.host_timestamp_ns = timestamp_ns
    sample.linear_acceleration_x_m_s2 = 2.0
    sample.angular_velocity_x_valid = True
    sample.angular_velocity_y_valid = True
    sample.angular_velocity_z_valid = True
    sample.linear_acceleration_x_valid = True
    sample.linear_acceleration_y_valid = True
    sample.linear_acceleration_z_valid = True
    return sample.SerializeToString()


def test_preintegrated_imu_sensor_caches_recording_and_slices_exact_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full recording is decoded once and returned as valid grid slices."""
    fake = _FakeImuSensor(_raw_imu_data())
    sensor = PreintegratedImuSensor(cast("ImuSensor", fake))
    grid_timestamps = np.array([0, _ONE_SECOND_NS, 2 * _ONE_SECOND_NS], dtype=np.int64)
    spec = _multi_window_spec(grid_timestamps)
    integration_count = 0
    original_preintegrate = preintegrated_imu_sensor_module.preintegrate_imu

    def counting_preintegrate(imu_data: ImuData, align_timestamps_ns: npt.NDArray[np.int64]) -> object:
        nonlocal integration_count
        integration_count += 1
        return original_preintegrate(imu_data, align_timestamps_ns)

    monkeypatch.setattr(preintegrated_imu_sensor_module, "preintegrate_imu", counting_preintegrate)

    first_pass = list(sensor.sample(spec))
    second_pass = list(sensor.sample(spec))

    assert fake.read_count == 1
    assert integration_count == 1
    np.testing.assert_array_equal(first_pass[0].align_timestamps_ns, grid_timestamps[:2])
    np.testing.assert_array_equal(first_pass[1].align_timestamps_ns, grid_timestamps[2:])
    assert first_pass[0].integration_valid.tolist() == [False, True]
    assert first_pass[1].integration_valid.tolist() == [True]
    assert first_pass[1].align_interval_start_timestamps_ns.tolist() == [_ONE_SECOND_NS]
    np.testing.assert_array_equal(second_pass[1].align_timestamps_ns, first_pass[1].align_timestamps_ns)


def test_preintegrated_imu_sensor_reintegrates_new_grid_without_redecoding() -> None:
    """Changing the alignment grid reuses the cached complete raw recording."""
    fake = _FakeImuSensor(_raw_imu_data())
    sensor = PreintegratedImuSensor(cast("ImuSensor", fake))

    list(sensor.sample(_multi_window_spec(np.array([0, _ONE_SECOND_NS, 2 * _ONE_SECOND_NS], dtype=np.int64))))
    finer_timestamps = np.array(
        [0, _HALF_SECOND_NS, _ONE_SECOND_NS, 3 * _HALF_SECOND_NS, 2 * _ONE_SECOND_NS],
        dtype=np.int64,
    )
    finer_batches = list(sensor.sample(_multi_window_spec(finer_timestamps)))

    assert fake.read_count == 1
    np.testing.assert_array_equal(finer_batches[0].align_timestamps_ns, finer_timestamps[:4])
    np.testing.assert_array_equal(finer_batches[1].align_timestamps_ns, finer_timestamps[4:])


def test_preintegrated_imu_sensor_works_with_sensor_group() -> None:
    """Window slices satisfy SensorGroup's exact alignment contract."""
    fake = _FakeImuSensor(_raw_imu_data())
    sensor = PreintegratedImuSensor(cast("ImuSensor", fake))
    spec = _multi_window_spec(np.array([0, _ONE_SECOND_NS, 2 * _ONE_SECOND_NS], dtype=np.int64))

    frames = list(SensorGroup({"imu_preintegrated": sensor}).sample(spec))

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].sensor_data["imu_preintegrated"].align_timestamps_ns, [0, _ONE_SECOND_NS])
    np.testing.assert_array_equal(frames[1].sensor_data["imu_preintegrated"].align_timestamps_ns, [2 * _ONE_SECOND_NS])


def test_preintegrated_imu_sensor_decodes_reference_mcap(tmp_path: Path) -> None:
    """The public reference protobuf and YAML mapping drive eager preintegration."""
    mcap_path = tmp_path / "imu.mcap"
    descriptor_set = protobuf_descriptor_set_from_proto(_REFERENCE_IMU_PROTO_PATH)
    timestamps = [0, _HALF_SECOND_NS, _ONE_SECOND_NS]
    write_protobuf_mcap(
        mcap_path,
        [McapSample(timestamp, _reference_imu_payload(timestamp)) for timestamp in timestamps],
        topic="/imu",
        schema_name=_REFERENCE_IMU_SCHEMA_NAME,
        schema_data=descriptor_set.SerializeToString(),
        library="preintegrated imu sensor test",
    )
    raw_sensor = ImuSensor(
        mcap_path,
        schema_name=_REFERENCE_IMU_SCHEMA_NAME,
        protobuf_mapping=_REFERENCE_IMU_MAPPING_PATH,
    )
    sensor = PreintegratedImuSensor(raw_sensor)
    spec = SamplingSpec(
        grid=SamplingGrid(
            start_ns=0,
            exclusive_end_ns=_ONE_SECOND_NS + 1,
            timestamps_ns=np.array([0, _ONE_SECOND_NS], dtype=np.int64),
            stride_ns=2 * _ONE_SECOND_NS,
            duration_ns=2 * _ONE_SECOND_NS,
        ),
    )

    (result,) = list(sensor.sample(spec))

    assert result.integration_valid.tolist() == [False, True]
    np.testing.assert_allclose(result.delta_velocity_m_s[1], [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.delta_position_m[1], [1.0, 0.0, 0.0], atol=1e-12)
