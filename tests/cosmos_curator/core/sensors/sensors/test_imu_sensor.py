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
"""Tests for MCAP protobuf ``ImuSensor``."""

from pathlib import Path

import numpy as np
import pytest
from google.protobuf import descriptor_pb2
from google.protobuf.message import Message

from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sensors.imu_sensor import DEFAULT_TOPIC, ImuSensor
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    one_window_spec,
    protobuf_descriptor_set_from_proto,
    protobuf_message_class,
    write_protobuf_mcap,
)

_FIELD_DESCRIPTOR_PROTO = descriptor_pb2.FieldDescriptorProto
_REPO_ROOT = Path(__file__).parents[5]
_REFERENCE_IMU_SCHEMA_NAME = "cosmos_curator.sensors.imu.v1.ImuSample"
_REFERENCE_IMU_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "imu_protobuf_mapping.yaml"
)
_REFERENCE_IMU_PROTO_PATH = _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "schemas" / "imu.proto"
_CUSTOMER_IMU_SCHEMA_NAME = "test.imu.Message"
_CUSTOMER_IMU_PROTO_FILE_NAME = "test/imu.proto"
_IMU_PROTO_FIELDS = (
    ("sensor_timestamp_ns", 1, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
    ("host_timestamp_ns", 2, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
    ("angular_velocity_x_rad_s", 3, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("angular_velocity_y_rad_s", 4, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("angular_velocity_z_rad_s", 5, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("linear_acceleration_x_m_s2", 6, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("linear_acceleration_y_m_s2", 7, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("linear_acceleration_z_m_s2", 8, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
    ("sequence_counter", 15, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64),
    ("temperature_c", 16, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE),
)


def _reference_imu_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build the descriptor set for the checked-in reference IMU schema."""
    return protobuf_descriptor_set_from_proto(_REFERENCE_IMU_PROTO_PATH)


def _reference_imu_payload(sensor_timestamp_ns: int, **overrides: object) -> bytes:
    """Serialize one reference IMU protobuf sample."""
    message_cls = protobuf_message_class(_reference_imu_descriptor_set(), _REFERENCE_IMU_SCHEMA_NAME)
    sample = message_cls()
    values: dict[str, object] = {
        "sensor_timestamp_ns": sensor_timestamp_ns,
        "host_timestamp_ns": sensor_timestamp_ns + 50,
        "angular_velocity_x_rad_s": 0.1,
        "angular_velocity_y_rad_s": 0.2,
        "angular_velocity_z_rad_s": 0.3,
        "linear_acceleration_x_m_s2": 1.0,
        "linear_acceleration_y_m_s2": 2.0,
        "linear_acceleration_z_m_s2": 9.8,
        "angular_velocity_x_valid": True,
        "angular_velocity_y_valid": True,
        "angular_velocity_z_valid": True,
        "linear_acceleration_x_valid": True,
        "linear_acceleration_y_valid": True,
        "linear_acceleration_z_valid": True,
        "sequence_counter": 1,
        "temperature_c": 47.0,
        "temperature_valid": True,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(sample, name, value)
    return sample.SerializeToString()


def _write_reference_imu_mcap(path: Path, samples: list[McapSample]) -> None:
    """Write an MCAP using the checked-in reference IMU schema contract."""
    write_protobuf_mcap(
        path,
        samples,
        topic=DEFAULT_TOPIC,
        schema_name=_REFERENCE_IMU_SCHEMA_NAME,
        schema_data=_reference_imu_descriptor_set().SerializeToString(),
        library="cosmos_curator reference imu sensor test",
    )


def _reference_imu_sensor(path: Path) -> ImuSensor:
    """Create an IMU sensor using the checked-in reference mapping."""
    return ImuSensor(
        path,
        schema_name=_REFERENCE_IMU_SCHEMA_NAME,
        protobuf_mapping=_REFERENCE_IMU_MAPPING_PATH,
    )


def _customer_imu_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build the protobuf descriptor set embedded in synthetic MCAP fixtures."""
    file_descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = file_descriptor_set.file.add()
    file_descriptor.name = _CUSTOMER_IMU_PROTO_FILE_NAME
    file_descriptor.package = "test.imu"
    file_descriptor.syntax = "proto3"

    message_descriptor = file_descriptor.message_type.add()
    message_descriptor.name = "Message"
    for field_name, field_number, field_type in _IMU_PROTO_FIELDS:
        field = message_descriptor.field.add()
        field.name = field_name
        field.number = field_number
        field.label = _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL
        field.type = field_type
    return file_descriptor_set


def _customer_imu_message_class() -> type[Message]:
    """Create the dynamic message class used by synthetic fixture writers."""
    return protobuf_message_class(_customer_imu_descriptor_set(), _CUSTOMER_IMU_SCHEMA_NAME)


def _vendor_imu_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build a nested customer IMU schema with repeated vectors and numeric validity."""
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = descriptor_set.file.add()
    file_descriptor.name = "vendor/imu.proto"
    file_descriptor.package = "vendor"
    file_descriptor.syntax = "proto3"

    frame = file_descriptor.message_type.add()
    frame.name = "ImuFrame"
    fields = (
        ("sensor_time_us", 1, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64, False),
        ("alignment_time_ns", 2, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64, False),
        ("turnrate", 3, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE, True),
        ("acceleration", 4, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE, True),
        ("turnrate_quality", 5, _FIELD_DESCRIPTOR_PROTO.TYPE_INT32, True),
        ("acceleration_quality", 6, _FIELD_DESCRIPTOR_PROTO.TYPE_INT32, True),
        ("host_time_us", 7, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64, False),
        ("sequence", 8, _FIELD_DESCRIPTOR_PROTO.TYPE_UINT64, False),
        ("temperature", 9, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE, False),
        ("temperature_quality", 10, _FIELD_DESCRIPTOR_PROTO.TYPE_INT32, False),
        ("turnrate_offset", 11, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE, True),
        ("acceleration_offset", 12, _FIELD_DESCRIPTOR_PROTO.TYPE_DOUBLE, True),
        ("turnrate_offset_quality", 13, _FIELD_DESCRIPTOR_PROTO.TYPE_INT32, True),
        ("acceleration_offset_quality", 14, _FIELD_DESCRIPTOR_PROTO.TYPE_INT32, True),
    )
    for name, number, field_type, repeated in fields:
        field = frame.field.add()
        field.name = name
        field.number = number
        field.type = field_type
        field.label = _FIELD_DESCRIPTOR_PROTO.LABEL_REPEATED if repeated else _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL

    envelope = file_descriptor.message_type.add()
    envelope.name = "ImuEnvelope"
    root = envelope.field.add()
    root.name = "imu_frame"
    root.number = 1
    root.label = _FIELD_DESCRIPTOR_PROTO.LABEL_OPTIONAL
    root.type = _FIELD_DESCRIPTOR_PROTO.TYPE_MESSAGE
    root.type_name = ".vendor.ImuFrame"
    return descriptor_set


def _vendor_imu_payload(*, turnrate: tuple[float, ...] = (0.1, 0.2, 0.3)) -> bytes:
    """Serialize one nested customer IMU message."""
    message_cls = protobuf_message_class(_vendor_imu_descriptor_set(), "vendor.ImuEnvelope")
    message = message_cls()
    frame = message.imu_frame
    frame.sensor_time_us = 2
    frame.alignment_time_ns = 100
    frame.turnrate.extend(turnrate)
    frame.acceleration.extend((1.0, 2.0, 9.8))
    frame.turnrate_quality.extend((5, 4, 5))
    frame.acceleration_quality.extend((5, 5, 4))
    frame.host_time_us = 3
    frame.sequence = 7
    frame.temperature = 47.5
    frame.temperature_quality = 5
    frame.turnrate_offset.extend((0.01, 0.02, 0.03))
    frame.acceleration_offset.extend((0.1, 0.2, 0.3))
    frame.turnrate_offset_quality.extend((5, 4, 5))
    frame.acceleration_offset_quality.extend((4, 5, 5))
    return message.SerializeToString()


def _customer_imu_payload(  # noqa: PLR0913
    *,
    sensor_timestamp_ns: int,
    host_timestamp_ns: int,
    sequence_counter: int,
    angular_velocity: tuple[float, float, float] = (0.1, 0.2, 0.3),
    linear_acceleration: tuple[float, float, float] = (1.0, 2.0, 9.8),
    temperature_c: float = 47.0,
) -> bytes:
    """Serialize one synthetic IMU protobuf sample."""
    message_cls = _customer_imu_message_class()
    sample = message_cls()
    fields: dict[str, object] = {
        "sensor_timestamp_ns": sensor_timestamp_ns,
        "host_timestamp_ns": host_timestamp_ns,
        "angular_velocity_x_rad_s": angular_velocity[0],
        "angular_velocity_y_rad_s": angular_velocity[1],
        "angular_velocity_z_rad_s": angular_velocity[2],
        "linear_acceleration_x_m_s2": linear_acceleration[0],
        "linear_acceleration_y_m_s2": linear_acceleration[1],
        "linear_acceleration_z_m_s2": linear_acceleration[2],
        "sequence_counter": sequence_counter,
        "temperature_c": temperature_c,
    }
    for name, value in fields.items():
        setattr(sample, name, value)
    return sample.SerializeToString()


def _sample(sensor_timestamp_ns: int, sequence_counter: int) -> McapSample:
    """Build one synthetic MCAP sample with matching log and sensor timestamps."""
    return McapSample(
        log_time_ns=sensor_timestamp_ns,
        data=_customer_imu_payload(
            sensor_timestamp_ns=sensor_timestamp_ns,
            host_timestamp_ns=sensor_timestamp_ns + 50,
            sequence_counter=sequence_counter,
        ),
    )


def _write_customer_imu_mcap(  # noqa: PLR0913
    path: Path,
    samples: list[McapSample],
    *,
    topic: str = DEFAULT_TOPIC,
    schema_name: str = _CUSTOMER_IMU_SCHEMA_NAME,
    schema_encoding: str = "protobuf",
    message_encoding: str = "protobuf",
    schema_data: bytes | None = None,
) -> None:
    """Write a synthetic MCAP fixture with one IMU-like channel."""
    write_protobuf_mcap(
        path,
        samples,
        topic=topic,
        schema_name=schema_name,
        schema_data=(_customer_imu_descriptor_set().SerializeToString() if schema_data is None else schema_data),
        schema_encoding=schema_encoding,
        message_encoding=message_encoding,
        library="cosmos_curator imu sensor test",
    )


def _custom_mapping(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "sensor_timestamp_ns": {"from": "sensor_timestamp_ns", "type": "timestamp", "unit": "us"},
        "angular_velocity_rad_s": {
            "group": ["angular_velocity_x_rad_s", "angular_velocity_y_rad_s", "angular_velocity_z_rad_s"],
            "type": "float",
        },
        "linear_acceleration_m_s2": {
            "group": ["linear_acceleration_x_m_s2", "linear_acceleration_y_m_s2", "linear_acceleration_z_m_s2"],
            "type": "float",
        },
    }
    fields.update(overrides)
    return {"fields": fields}


def test_imu_sensor_reads_reference_schema_with_checked_in_mapping(tmp_path: Path) -> None:
    """The shipped reference schema and YAML should produce complete IMU data."""
    path = tmp_path / "reference_imu.mcap"
    _write_reference_imu_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_imu_payload(
                    100,
                    angular_velocity_y_valid=False,
                    linear_acceleration_z_valid=False,
                    sequence_counter=10,
                ),
            ),
            McapSample(
                log_time_ns=200,
                data=_reference_imu_payload(
                    200,
                    angular_velocity_x_rad_s=0.4,
                    angular_velocity_y_rad_s=0.5,
                    angular_velocity_z_rad_s=0.6,
                    sequence_counter=11,
                    temperature_c=48.0,
                ),
            ),
        ],
    )

    batch = next(_reference_imu_sensor(path).sample(one_window_spec(100, 300), policy=NoSamplingPolicy()))

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([150, 250], dtype=np.int64))
    np.testing.assert_allclose(batch.angular_velocity_rad_s, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    np.testing.assert_array_equal(
        batch.angular_velocity_valid,
        np.array([[True, False, True], [True, True, True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        batch.linear_acceleration_valid,
        np.array([[True, True, False], [True, True, True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(batch.sequence_counter, np.array([10, 11], dtype=np.uint64))
    np.testing.assert_allclose(batch.temperature_c, np.array([47.0, 48.0]))
    np.testing.assert_array_equal(batch.temperature_valid, np.array([True, True], dtype=np.bool_))


def test_imu_sensor_rejects_nearest_timestamp_policy(tmp_path: Path) -> None:
    """Raw IMU sampling only accepts the explicit no-op sampling policy."""
    path = tmp_path / "reference_imu.mcap"
    _write_reference_imu_mcap(path, [McapSample(log_time_ns=100, data=_reference_imu_payload(100))])
    sensor = _reference_imu_sensor(path)

    with pytest.raises(TypeError, match="ImuSensor requires NoSamplingPolicy"):
        next(sensor.sample(one_window_spec(100, 200), policy=NearestTimestampPolicy()))


def test_imu_reference_mapping_preserves_invalid_values_and_masks(tmp_path: Path) -> None:
    """Reference mapping should preserve values independently from validity masks."""
    path = tmp_path / "invalid_values_imu.mcap"
    _write_reference_imu_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_imu_payload(
                    100,
                    angular_velocity_x_rad_s=np.nan,
                    angular_velocity_x_valid=False,
                    linear_acceleration_z_m_s2=np.nan,
                    linear_acceleration_z_valid=False,
                    temperature_c=np.nan,
                    temperature_valid=False,
                ),
            )
        ],
    )

    batch = next(_reference_imu_sensor(path).sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))

    assert np.isnan(batch.angular_velocity_rad_s[0, 0])
    assert not batch.angular_velocity_valid[0, 0]
    assert np.isnan(batch.linear_acceleration_m_s2[0, 2])
    assert not batch.linear_acceleration_valid[0, 2]
    assert batch.temperature_c is not None
    assert batch.temperature_valid is not None
    assert np.isnan(batch.temperature_c[0])
    assert not batch.temperature_valid[0]


def test_imu_reference_mapping_empty_window_preserves_optional_arrays(tmp_path: Path) -> None:
    """Fully mapped reference optionals should remain present in empty windows."""
    path = tmp_path / "reference_empty_window.mcap"
    _write_reference_imu_mcap(path, [McapSample(log_time_ns=100, data=_reference_imu_payload(100))])

    batch = next(_reference_imu_sensor(path).sample(one_window_spec(200, 300), policy=NoSamplingPolicy()))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.angular_velocity_rad_s.shape == (0, 3)
    assert batch.linear_acceleration_m_s2.shape == (0, 3)
    assert batch.host_timestamps_ns is not None
    assert batch.host_timestamps_ns.shape == (0,)
    assert batch.sequence_counter is not None
    assert batch.sequence_counter.shape == (0,)
    assert batch.temperature_c is not None
    assert batch.temperature_c.shape == (0,)
    assert batch.temperature_valid is not None
    assert batch.temperature_valid.shape == (0,)


def test_imu_sensor_rejects_duplicate_mapped_align_timestamps(tmp_path: Path) -> None:
    """Mapped IMU rows require strictly increasing alignment timestamps."""
    path = tmp_path / "duplicate_timestamps.mcap"
    _write_customer_imu_mcap(path, [_sample(100, 1), _sample(100, 2)])
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match="strictly increasing align_timestamp_ns"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))


def test_imu_sensor_read_all_rejects_recording_wide_duplicate_alignment(tmp_path: Path) -> None:
    """Eager decoding requires one globally strictly increasing source timeline."""
    path = tmp_path / "recording_duplicate_timestamps.mcap"
    _write_customer_imu_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_customer_imu_payload(sensor_timestamp_ns=1, host_timestamp_ns=150, sequence_counter=1),
            ),
            McapSample(
                log_time_ns=200,
                data=_customer_imu_payload(sensor_timestamp_ns=1, host_timestamp_ns=250, sequence_counter=2),
            ),
        ],
    )
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match="strictly increasing align_timestamp_ns"):
        sensor.read_all()


def test_imu_sensor_custom_mapping_uses_sensor_owned_target(tmp_path: Path) -> None:
    """Custom mappings populate the sensor-owned IMU row target."""
    path = tmp_path / "custom_imu.mcap"
    _write_customer_imu_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_customer_imu_payload(sensor_timestamp_ns=2, host_timestamp_ns=150, sequence_counter=7),
            )
        ],
    )
    mapping = _custom_mapping(
        align_timestamp_ns={"from": "$mcap.logtime", "type": "timestamp", "unit": "ns"},
        host_timestamp_ns={"from": "host_timestamp_ns", "type": "timestamp", "unit": "ns"},
        sequence_counter={"from": "sequence_counter", "type": "int"},
        temperature_c={"from": "temperature_c", "type": "float"},
    )
    batch = next(
        ImuSensor(path, schema_name=_CUSTOMER_IMU_SCHEMA_NAME, protobuf_mapping=mapping).sample(
            one_window_spec(100, 200),
            policy=NoSamplingPolicy(),
        )
    )
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([2_000], dtype=np.int64))
    np.testing.assert_allclose(batch.angular_velocity_rad_s, np.array([[0.1, 0.2, 0.3]]))
    np.testing.assert_array_equal(batch.angular_velocity_valid, np.array([[True, True, True]]))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([7], dtype=np.uint64))
    np.testing.assert_array_equal(batch.temperature_valid, np.array([True], dtype=np.bool_))


def test_imu_sensor_custom_mapping_requires_core_fields() -> None:
    """Custom mappings must populate required IMU measurements."""
    with pytest.raises(ValueError, match="linear_acceleration_m_s2"):
        ImuSensor(
            "unused.mcap",
            schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
            protobuf_mapping={
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_ns", "type": "timestamp", "unit": "ns"},
                    "angular_velocity_rad_s": {"default": [0.0, 0.0, 0.0]},
                }
            },
        )


def test_imu_sensor_custom_mapping_preserves_optional_absence_for_all_windows(tmp_path: Path) -> None:
    """Optional custom outputs stay absent for populated and empty windows."""
    path = tmp_path / "custom_optional_imu.mcap"
    _write_customer_imu_mcap(path, [_sample(sensor_timestamp_ns=100, sequence_counter=1)])
    sensor = ImuSensor(path, schema_name=_CUSTOMER_IMU_SCHEMA_NAME, protobuf_mapping=_custom_mapping())

    populated = next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))
    empty = next(sensor.sample(one_window_spec(200, 300), policy=NoSamplingPolicy()))
    for batch in (populated, empty):
        assert batch.host_timestamps_ns is None
        assert batch.sequence_counter is None
        assert batch.temperature_c is None
        assert batch.temperature_valid is None
        assert batch.angular_velocity_bias_rad_s is None
        assert batch.linear_acceleration_bias_m_s2 is None
        assert batch.angular_velocity_bias_valid is None
        assert batch.linear_acceleration_bias_valid is None
    np.testing.assert_array_equal(populated.angular_velocity_valid, np.array([[True, True, True]]))
    np.testing.assert_array_equal(populated.linear_acceleration_valid, np.array([[True, True, True]]))


def test_imu_sensor_rejects_temperature_valid_without_temperature() -> None:
    """Temperature validity cannot be mapped without its paired value."""
    with pytest.raises(ValueError, match=r"temperature_valid.*temperature_c.*not mapped"):
        ImuSensor(
            "unused.mcap",
            schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
            protobuf_mapping=_custom_mapping(temperature_valid={"from": "temperature_valid", "type": "bool"}),
        )


@pytest.mark.parametrize(
    ("validity_name", "source_name"),
    [
        ("angular_velocity_bias_valid", "angular_velocity_x_valid"),
        ("linear_acceleration_bias_valid", "linear_acceleration_x_valid"),
    ],
)
def test_imu_sensor_rejects_bias_validity_without_bias(validity_name: str, source_name: str) -> None:
    """Bias validity cannot be mapped without its corresponding bias value."""
    with pytest.raises(ValueError, match=rf"{validity_name}.*not mapped"):
        ImuSensor(
            "unused.mcap",
            schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
            protobuf_mapping=_custom_mapping(
                **{validity_name: {"group": [source_name] * 3, "type": "bool"}},
            ),
        )


def test_imu_sensor_custom_mapping_reports_malformed_payload_context(tmp_path: Path) -> None:
    """Malformed custom payload errors retain the IMU topic context."""
    path = tmp_path / "malformed_custom_imu.mcap"
    _write_customer_imu_mcap(path, [McapSample(log_time_ns=100, data=b"\x80")])
    sensor = ImuSensor(path, schema_name=_CUSTOMER_IMU_SCHEMA_NAME, protobuf_mapping=_custom_mapping())

    with pytest.raises(ValueError, match=r"failed to parse IMU protobuf message on topic '/imu'"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))


def test_imu_sensor_custom_mapping_rejects_out_of_range_sequence(tmp_path: Path) -> None:
    """Mapped sequence counters must fit the ImuData uint64 dtype."""
    path = tmp_path / "invalid_sequence_imu.mcap"
    _write_customer_imu_mcap(path, [_sample(sensor_timestamp_ns=100, sequence_counter=1)])
    mapping = _custom_mapping(sequence_counter={"default": 1 << 64, "type": "int"})

    with pytest.raises(ValueError, match="sequence_counter values must be present and fit uint64"):
        next(
            ImuSensor(path, schema_name=_CUSTOMER_IMU_SCHEMA_NAME, protobuf_mapping=mapping).sample(
                one_window_spec(100, 200),
                policy=NoSamplingPolicy(),
            )
        )


def _nested_repeated_mapping() -> dict[str, object]:
    return {
        "root": "imu_frame",
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
            "align_timestamp_ns": {"from": "alignment_time_ns", "type": "timestamp", "unit": "ns"},
            "angular_velocity_rad_s": {"from": "turnrate", "type": "float"},
            "linear_acceleration_m_s2": {"from": "acceleration", "type": "float"},
            "angular_velocity_valid": {
                "from": "turnrate_quality",
                "type": "bool",
                "valid_when": {"equals": 5},
            },
            "linear_acceleration_valid": {
                "from": "acceleration_quality",
                "type": "bool",
                "valid_when": {"equals": 5},
            },
            "angular_velocity_bias_rad_s": {"from": "turnrate_offset", "type": "float"},
            "linear_acceleration_bias_m_s2": {"from": "acceleration_offset", "type": "float"},
            "angular_velocity_bias_valid": {
                "from": "turnrate_offset_quality",
                "type": "bool",
                "valid_when": {"equals": 5},
            },
            "linear_acceleration_bias_valid": {
                "from": "acceleration_offset_quality",
                "type": "bool",
                "valid_when": {"equals": 5},
            },
            "host_timestamp_ns": {"from": "host_time_us", "type": "timestamp", "unit": "us"},
            "sequence_counter": {"from": "sequence", "type": "int"},
            "temperature_c": {"from": "temperature", "type": "float"},
            "temperature_valid": {
                "from": "temperature_quality",
                "type": "bool",
                "valid_when": {"equals": 5},
            },
        },
    }


def test_imu_sensor_maps_nested_repeated_customer_schema(tmp_path: Path) -> None:
    """Nested roots, repeated vectors, and numeric validity map into ImuData."""
    path = tmp_path / "nested_repeated_imu.mcap"
    _write_customer_imu_mcap(
        path,
        [McapSample(log_time_ns=100, data=_vendor_imu_payload())],
        schema_name="vendor.ImuEnvelope",
        schema_data=_vendor_imu_descriptor_set().SerializeToString(),
    )
    sensor = ImuSensor(path, schema_name="vendor.ImuEnvelope", protobuf_mapping=_nested_repeated_mapping())

    batch = next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([2_000], dtype=np.int64))
    np.testing.assert_allclose(batch.angular_velocity_rad_s, np.array([[0.1, 0.2, 0.3]]))
    np.testing.assert_allclose(batch.linear_acceleration_m_s2, np.array([[1.0, 2.0, 9.8]]))
    np.testing.assert_array_equal(batch.angular_velocity_valid, np.array([[True, False, True]]))
    np.testing.assert_array_equal(batch.linear_acceleration_valid, np.array([[True, True, False]]))
    np.testing.assert_allclose(batch.angular_velocity_bias_rad_s, np.array([[0.01, 0.02, 0.03]]))
    np.testing.assert_allclose(batch.linear_acceleration_bias_m_s2, np.array([[0.1, 0.2, 0.3]]))
    np.testing.assert_array_equal(batch.angular_velocity_bias_valid, np.array([[True, False, True]]))
    np.testing.assert_array_equal(batch.linear_acceleration_bias_valid, np.array([[False, True, True]]))
    np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([3_000], dtype=np.int64))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([7], dtype=np.uint64))
    np.testing.assert_allclose(batch.temperature_c, np.array([47.5]))
    np.testing.assert_array_equal(batch.temperature_valid, np.array([True]))

    empty = next(sensor.sample(one_window_spec(200, 300), policy=NoSamplingPolicy()))
    assert empty.angular_velocity_bias_rad_s is not None
    assert empty.angular_velocity_bias_rad_s.shape == (0, 3)
    assert empty.linear_acceleration_bias_m_s2 is not None
    assert empty.linear_acceleration_bias_m_s2.shape == (0, 3)
    assert empty.angular_velocity_bias_valid is not None
    assert empty.angular_velocity_bias_valid.shape == (0, 3)
    assert empty.linear_acceleration_bias_valid is not None
    assert empty.linear_acceleration_bias_valid.shape == (0, 3)


def test_imu_sensor_defaults_mapped_bias_validity_to_true(tmp_path: Path) -> None:
    """Mapped bias vectors default to valid when their validity mappings are omitted."""
    path = tmp_path / "default_bias_validity.mcap"
    write_protobuf_mcap(
        path,
        [McapSample(log_time_ns=100, data=_vendor_imu_payload())],
        topic=DEFAULT_TOPIC,
        schema_name="vendor.ImuEnvelope",
        schema_data=_vendor_imu_descriptor_set().SerializeToString(),
        library="cosmos_curator vendor imu sensor test",
    )
    mapping = _nested_repeated_mapping()
    fields = mapping["fields"]
    assert isinstance(fields, dict)
    del fields["angular_velocity_bias_valid"]
    del fields["linear_acceleration_bias_valid"]

    batch = next(
        ImuSensor(path, schema_name="vendor.ImuEnvelope", protobuf_mapping=mapping).sample(
            one_window_spec(100, 200), policy=NoSamplingPolicy()
        )
    )

    np.testing.assert_array_equal(batch.angular_velocity_bias_valid, np.ones((1, 3), dtype=np.bool_))
    np.testing.assert_array_equal(batch.linear_acceleration_bias_valid, np.ones((1, 3), dtype=np.bool_))


def test_imu_sensor_rejects_repeated_vector_with_wrong_length(tmp_path: Path) -> None:
    """Repeated customer vectors must contain exactly three elements."""
    path = tmp_path / "short_vector_imu.mcap"
    _write_customer_imu_mcap(
        path,
        [McapSample(log_time_ns=100, data=_vendor_imu_payload(turnrate=(0.1, 0.2)))],
        schema_name="vendor.ImuEnvelope",
        schema_data=_vendor_imu_descriptor_set().SerializeToString(),
    )
    sensor = ImuSensor(path, schema_name="vendor.ImuEnvelope", protobuf_mapping=_nested_repeated_mapping())

    with pytest.raises(ValueError, match=r"angular_velocity_rad_s.*expected 3 values"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))


def test_imu_sensor_exposes_mcap_topic_timeline_with_mapping(tmp_path: Path) -> None:
    """Timeline properties should remain independent of protobuf mapping."""
    path = tmp_path / "timeline.mcap"
    _write_customer_imu_mcap(path, [_sample(100, 1), _sample(200, 2), _sample(300, 3)])
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    np.testing.assert_array_equal(sensor.timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    assert sensor.timestamps_ns.flags.writeable is False
    assert sensor.max_gap_ns == 0


def test_imu_sensor_rejects_empty_topic_timeline_with_mapping(tmp_path: Path) -> None:
    """Timeline properties should fail clearly when the configured topic has no messages."""
    path = tmp_path / "empty_topic.mcap"
    _write_customer_imu_mcap(path, [])
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match=r"no MCAP messages on topic .*/imu"):
        _ = sensor.timestamps_ns
    with pytest.raises(ValueError, match=r"no MCAP messages on topic .*/imu"):
        _ = sensor.start_ns


def test_imu_sensor_empty_window_yields_mapped_empty_data(tmp_path: Path) -> None:
    """A window with no messages should preserve the mapped output contract."""
    path = tmp_path / "empty_window.mcap"
    _write_customer_imu_mcap(path, [_sample(100, 1)])
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    batch = next(sensor.sample(one_window_spec(200, 300), policy=NoSamplingPolicy()))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.angular_velocity_rad_s.shape == (0, 3)
    assert batch.linear_acceleration_m_s2.shape == (0, 3)
    assert batch.angular_velocity_valid.shape == (0, 3)
    assert batch.linear_acceleration_valid.shape == (0, 3)
    assert batch.host_timestamps_ns is None
    assert batch.sequence_counter is None
    assert batch.temperature_c is None
    assert batch.temperature_valid is None


def test_imu_sensor_rejects_missing_mapped_topic(tmp_path: Path) -> None:
    """The configured mapped topic must exist in the MCAP."""
    path = tmp_path / "missing_topic.mcap"
    _write_customer_imu_mcap(path, [_sample(100, 1)], topic="/other")
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match="no MCAP channel found for topic '/imu'"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))


def test_imu_sensor_rejects_wrong_mapped_schema_name(tmp_path: Path) -> None:
    """The MCAP schema must match the mapped protobuf schema name."""
    path = tmp_path / "wrong_schema.mcap"
    _write_customer_imu_mcap(path, [_sample(100, 1)], schema_name="test.imu.Other")
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match="expected MCAP schema"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))


def test_imu_sensor_rejects_non_protobuf_mapped_channel(tmp_path: Path) -> None:
    """Mapped IMU channels must still use protobuf encoding."""
    path = tmp_path / "wrong_encoding.mcap"
    _write_customer_imu_mcap(
        path,
        [McapSample(log_time_ns=100, data=b"{}")],
        schema_encoding="jsonschema",
        message_encoding="json",
        schema_data=b"{}",
    )
    sensor = ImuSensor(
        path,
        schema_name=_CUSTOMER_IMU_SCHEMA_NAME,
        protobuf_mapping=_custom_mapping(),
    )

    with pytest.raises(ValueError, match="expected protobuf channel"):
        next(sensor.sample(one_window_spec(100, 200), policy=NoSamplingPolicy()))
