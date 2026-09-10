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
"""Tests for the standalone protobuf mapping validator script."""

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import pytest
import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message
from mcap.reader import make_reader
from mcap.writer import CompressionType, IndexType, Writer

from cosmos_curator.core.sensors.scripts.validate_protobuf_mapping import (
    INPUT_ERROR_EXIT_CODE,
    INVALID_MAPPING_EXIT_CODE,
    PASS_EXIT_CODE,
    main,
    validate_protobuf_mapping,
)
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    protobuf_descriptor_set_from_proto,
    write_protobuf_mcap,
)

_FIELD = descriptor_pb2.FieldDescriptorProto
_SCHEMA_NAME = "validator_test.SensorEnvelope"
_TOPIC = "/sensor/data"


def _descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    file_descriptor = descriptor_set.file.add()
    file_descriptor.name = "validator_test.proto"
    file_descriptor.package = "validator_test"
    file_descriptor.syntax = "proto3"
    message = file_descriptor.message_type.add()
    message.name = "SensorEnvelope"
    for name, number, field_type, repeated in (
        ("sensor_time_us", 1, _FIELD.TYPE_UINT64, False),
        ("latitude", 2, _FIELD.TYPE_DOUBLE, False),
        ("longitude", 3, _FIELD.TYPE_DOUBLE, False),
        ("altitude", 4, _FIELD.TYPE_DOUBLE, False),
        ("flag", 5, _FIELD.TYPE_BOOL, False),
        ("gyro_rates", 6, _FIELD.TYPE_DOUBLE, True),
        ("acceleration", 7, _FIELD.TYPE_DOUBLE, True),
    ):
        field = message.field.add()
        field.name = name
        field.number = number
        field.type = field_type
        field.label = _FIELD.LABEL_REPEATED if repeated else _FIELD.LABEL_OPTIONAL
    return descriptor_set


@dataclass(frozen=True)
class _GpsRow:
    sensor_timestamp_ns: int
    align_timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass(frozen=True)
class _VectorRow:
    values: tuple[float, float, float]
    values_valid: tuple[bool, bool, bool] = (True, True, True)


@dataclass(frozen=True)
class _BoolRow:
    valid: bool


@dataclass(frozen=True)
class _TemperatureRow:
    temperature_c: float | None = None
    temperature_valid: bool | None = None


@dataclass(frozen=True)
class _AmbiguousUnitValidityRow:
    value_m: float
    value_deg: float
    value_valid: bool


def _message_class() -> type[Message]:
    pool = descriptor_pool.DescriptorPool()
    for file_descriptor in _descriptor_set().file:
        pool.Add(file_descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_SCHEMA_NAME))


def _nested_message_class() -> type[Message]:
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "nested_validator_test.proto"
    file_descriptor.package = "validator_test"
    file_descriptor.syntax = "proto3"
    fix = file_descriptor.message_type.add()
    fix.name = "Fix"
    for name, number, field_type in (
        ("sensor_time_us", 1, _FIELD.TYPE_UINT64),
        ("latitude", 2, _FIELD.TYPE_DOUBLE),
    ):
        field = fix.field.add()
        field.name = name
        field.number = number
        field.type = field_type
        field.label = _FIELD.LABEL_OPTIONAL
    envelope = file_descriptor.message_type.add()
    envelope.name = "NestedEnvelope"
    root = envelope.field.add()
    root.name = "fix"
    root.number = 1
    root.type = _FIELD.TYPE_MESSAGE
    root.type_name = ".validator_test.Fix"
    root.label = _FIELD.LABEL_OPTIONAL
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("validator_test.NestedEnvelope"))


def _write_mcap(
    path: Path,
    *,
    topic: str = _TOPIC,
    message_encoding: str = "protobuf",
    schema_encoding: str = "protobuf",
    schema_data: bytes | None = None,
    indexed: bool = True,
) -> None:
    data = _descriptor_set().SerializeToString() if schema_data is None else schema_data
    with path.open("wb") as stream:
        writer = Writer(
            stream,
            compression=CompressionType.NONE,
            index_types=IndexType.ALL if indexed else IndexType.NONE,
            repeat_channels=indexed,
            repeat_schemas=indexed,
            use_statistics=indexed,
            use_summary_offsets=indexed,
        )
        writer.start(library="protobuf mapping validator test")
        schema_id = writer.register_schema(name=_SCHEMA_NAME, encoding=schema_encoding, data=data)
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic=topic,
            message_encoding=message_encoding,
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=100,
            publish_time=100,
            sequence=0,
            data=b"\x80",
        )
        writer.finish()  # type: ignore[no-untyped-call]


def _gps_mapping() -> dict[str, object]:
    return {
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
            "latitude_deg": {"from": "latitude", "type": "float"},
            "longitude_deg": {"from": "longitude", "type": "float"},
            "altitude_m": {"from": "altitude", "type": "float"},
        }
    }


def _imu_mapping() -> dict[str, object]:
    return {
        "fields": {
            "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
            "angular_velocity_rad_s": {"from": "gyro_rates", "type": "float"},
            "linear_acceleration_m_s2": {"from": "acceleration", "type": "float"},
        }
    }


def _run(
    path: Path,
    mapping_path: Path,
    *,
    target: str = "gps",
    topic: str = _TOPIC,
    quiet: bool = False,
) -> int:
    args = [
        "--target",
        target,
        "--topic",
        topic,
        "--mapping",
        str(mapping_path),
        "--mcap",
        str(path),
    ]
    if quiet:
        args.append("--quiet")
    return main(args)


@pytest.mark.parametrize(("target", "mapping"), [("gps", _gps_mapping()), ("imu", _imu_mapping())])
def test_script_validates_sensor_target_without_decoding_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    mapping: dict[str, object],
) -> None:
    """Both targets should validate even though the synthetic message payload is malformed."""
    mcap_path = tmp_path / f"{target}.mcap"
    mapping_path = tmp_path / f"{target}.yaml"
    _write_mcap(mcap_path)
    mapping_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    assert _run(mcap_path, mapping_path, target=target) == PASS_EXIT_CODE
    output = capsys.readouterr().out
    assert f"Target: {target}" in output
    assert f"Topic: {_TOPIC}" in output
    assert f"Schema: {_SCHEMA_NAME}" in output
    assert "Root: <message>" in output
    assert "Mappings (* required)" in output
    assert "* sensor_timestamp_ns" in output
    assert "path: sensor_time_us [uint64]" in output
    assert "mapping: timestamp from us to ns" in output
    assert "target: int" in output
    expected_required = 4 if target == "gps" else 3
    assert f"Required fields: {expected_required}/{expected_required} mapped" in output
    if target == "imu":
        assert "* angular_velocity_rad_s" in output
        assert "path: gyro_rates [repeated double]" in output
    assert "PASS" in output


def test_script_validates_unindexed_mcap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The message-scan fallback should match the complete topic name."""
    mcap_path = tmp_path / "unindexed.mcap"
    mapping_path = tmp_path / "gps.yaml"
    _write_mcap(mcap_path, indexed=False)
    mapping_path.write_text(yaml.safe_dump(_gps_mapping()), encoding="utf-8")

    with mcap_path.open("rb") as stream:
        assert make_reader(stream).get_summary() is None

    assert _run(mcap_path, mapping_path) == PASS_EXIT_CODE
    output = capsys.readouterr().out
    assert f"Topic: {_TOPIC}" in output
    assert f"Schema: {_SCHEMA_NAME}" in output
    assert "PASS" in output

    assert _run(mcap_path, mapping_path, topic="missing:topic") == INPUT_ERROR_EXIT_CODE
    assert "no MCAP messages found" in capsys.readouterr().err


def test_script_reports_all_mapping_problems(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script should print independent source, destination, type, and required-field problems."""
    mcap_path = tmp_path / "invalid.mcap"
    mapping_path = tmp_path / "invalid.yaml"
    _write_mcap(mcap_path)
    mapping_path.write_text(
        yaml.safe_dump(
            {
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
                    "latitude_deg": {"from": "missing_latitude", "type": "float"},
                    "longitude_deg": {"from": "flag", "type": "float"},
                    "unknown_destination": {"from": "longitude", "type": "float"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert _run(mcap_path, mapping_path) == INVALID_MAPPING_EXIT_CODE
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "unknown_destination" in output
    assert "altitude_m" in output
    assert "missing_latitude" in output
    assert "Mappings (* required)" in output
    assert "* latitude_deg" in output
    assert "path: missing_latitude [unresolved]" in output
    assert "Required fields: 3/4 mapped" in output
    assert "protobuf type 'bool'" in output
    assert "compatible with 'float'" in output


def test_script_quiet_mode_suppresses_success_and_writes_failures_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Quiet mode should support exit-code-based bulk validation."""
    mcap_path = tmp_path / "quiet.mcap"
    valid_mapping_path = tmp_path / "valid.yaml"
    invalid_mapping_path = tmp_path / "invalid.yaml"
    _write_mcap(mcap_path)
    valid_mapping_path.write_text(yaml.safe_dump(_gps_mapping()), encoding="utf-8")
    invalid_mapping_path.write_text(yaml.safe_dump({"fields": {}}), encoding="utf-8")

    assert _run(mcap_path, valid_mapping_path, quiet=True) == PASS_EXIT_CODE
    assert capsys.readouterr() == ("", "")

    assert _run(mcap_path, invalid_mapping_path, quiet=True) == INVALID_MAPPING_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "FAIL: mapping has" in captured.err
    assert "required target field 'sensor_timestamp_ns' is not mapped" in captured.err


@pytest.mark.parametrize(
    "case",
    [
        ("missing", "protobuf", "protobuf", None, "no MCAP channel"),
        (_TOPIC, "json", "jsonschema", b"{}", "expected protobuf channel"),
        (_TOPIC, "protobuf", "protobuf", b"\x80", "failed to parse MCAP protobuf schema"),
    ],
)
def test_script_reports_mcap_input_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: tuple[str, str, str, bytes | None, str],
) -> None:
    """Topic and schema failures should use the input-error exit code."""
    topic, message_encoding, schema_encoding, schema_data, expected = case
    mcap_path = tmp_path / "input_error.mcap"
    mapping_path = tmp_path / "gps.yaml"
    _write_mcap(
        mcap_path,
        message_encoding=message_encoding,
        schema_encoding=schema_encoding,
        schema_data=schema_data,
    )
    mapping_path.write_text(yaml.safe_dump(_gps_mapping()), encoding="utf-8")

    assert _run(mcap_path, mapping_path, topic=topic) == INPUT_ERROR_EXIT_CODE
    assert expected in capsys.readouterr().err


def test_script_reports_malformed_mcap_as_input_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed MCAP containers should not escape as an unhandled traceback."""
    mcap_path = tmp_path / "malformed.mcap"
    mapping_path = tmp_path / "gps.yaml"
    mcap_path.write_bytes(b"")
    mapping_path.write_text(yaml.safe_dump(_gps_mapping()), encoding="utf-8")

    assert _run(mcap_path, mapping_path) == INPUT_ERROR_EXIT_CODE
    assert "failed to read MCAP" in capsys.readouterr().err


def test_validation_api_accepts_valid_timestamp_mapping() -> None:
    """The importable validator in the script should return a successful result."""
    result = validate_protobuf_mapping(
        _gps_mapping(),
        target_cls=_GpsRow,
        message_cls=_message_class(),
        required_target_names={"sensor_timestamp_ns", "latitude_deg", "longitude_deg", "altitude_m"},
    )

    assert result.is_valid
    assert result.problems == ()
    summary = "\n".join(result.summary_lines)
    assert "Root: <message>" in summary
    assert "Required fields: 4/4 mapped" in summary
    assert "* sensor_timestamp_ns" in summary
    assert "path: sensor_time_us [uint64]" in summary
    assert "mapping: timestamp from us to ns" in summary
    assert "target: int" in summary


def test_validation_api_checks_group_arity() -> None:
    """Grouped sources should match a fixed-width target tuple."""
    result = validate_protobuf_mapping(
        {"fields": {"values": {"group": ["latitude", "longitude"], "type": "float"}}},
        target_cls=_VectorRow,
        message_cls=_message_class(),
    )

    assert any("expected 3 grouped values, got 2" in problem for problem in result.problems)


def test_validation_api_checks_bool_predicate() -> None:
    """Numeric validity should require valid_when while the predicate form passes."""
    direct = validate_protobuf_mapping(
        {"fields": {"valid": {"from": "sensor_time_us", "type": "bool"}}},
        target_cls=_BoolRow,
        message_cls=_message_class(),
    )
    predicate = validate_protobuf_mapping(
        {"fields": {"valid": {"from": "sensor_time_us", "type": "bool", "valid_when": {"equals": 5}}}},
        target_cls=_BoolRow,
        message_cls=_message_class(),
    )

    assert any("use valid_when" in problem for problem in direct.problems)
    assert predicate.is_valid


def test_validation_api_checks_unit_suffixed_validity_pairs() -> None:
    """Validity must map its uniquely paired unit-suffixed value target."""
    result = validate_protobuf_mapping(
        {"fields": {"temperature_valid": {"from": "flag", "type": "bool"}}},
        target_cls=_TemperatureRow,
        message_cls=_message_class(),
    )

    assert any(
        "maps validity for 'temperature_c', but fields.temperature_c is not mapped" in problem
        for problem in result.problems
    )


def test_validation_api_rejects_ambiguous_unit_suffixed_validity_pairs() -> None:
    """Validity must not select between multiple unit-suffixed value targets."""
    result = validate_protobuf_mapping(
        {"fields": {"value_valid": {"from": "flag", "type": "bool"}}},
        target_cls=_AmbiguousUnitValidityRow,
        message_cls=_message_class(),
    )

    assert any("value_valid" in problem and "value_deg, value_m" in problem for problem in result.problems)


def test_validation_api_accepts_repeated_scalar_collection() -> None:
    """A repeated scalar source should be compatible with a tuple target."""
    result = validate_protobuf_mapping(
        {"fields": {"values": {"from": "gyro_rates", "type": "float"}}},
        target_cls=_VectorRow,
        message_cls=_message_class(),
    )

    assert result.is_valid


@pytest.mark.parametrize(
    ("target", "schema_name", "schema_filename", "mapping_filename"),
    [
        ("gps", "cosmos_curator.sensors.gps.v1.GpsSample", "gps.proto", "gps_protobuf_mapping.yaml"),
        ("imu", "cosmos_curator.sensors.imu.v1.ImuSample", "imu.proto", "imu_minimal_protobuf_mapping.yaml"),
        ("imu", "cosmos_curator.sensors.imu.v1.ImuSample", "imu.proto", "imu_protobuf_mapping.yaml"),
        (
            "egotrajectory",
            "cosmos_curator.sensors.egotrajectory.v1.EgotrajectorySample",
            "egotrajectory.proto",
            "egotrajectory_protobuf_mapping.yaml",
        ),
    ],
)
def test_script_validates_checked_in_reference_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    schema_name: str,
    schema_filename: str,
    mapping_filename: str,
) -> None:
    """The public schema and matching generic mapping must validate together."""
    mcap_path = tmp_path / f"{target}.mcap"
    schema_resource = files("cosmos_curator").joinpath("core", "sensors", "schemas", schema_filename)
    mapping_resource = files("cosmos_curator").joinpath("core", "sensors", "examples", mapping_filename)

    with as_file(schema_resource) as schema_path, as_file(mapping_resource) as mapping_path:
        descriptor_set = protobuf_descriptor_set_from_proto(schema_path)
        write_protobuf_mcap(
            mcap_path,
            [McapSample(log_time_ns=100, data=b"")],
            topic=f"/{target}",
            schema_name=schema_name,
            schema_data=descriptor_set.SerializeToString(),
        )

        assert _run(mcap_path, mapping_path, target=target, topic=f"/{target}") == PASS_EXIT_CODE
    assert "PASS: mapping is structurally valid." in capsys.readouterr().out


def test_validation_api_accepts_nested_root() -> None:
    """Source paths should be validated relative to a nested message root."""

    @dataclass(frozen=True)
    class NestedRow:
        sensor_timestamp_ns: int
        align_timestamp_ns: int
        latitude_deg: float

    result = validate_protobuf_mapping(
        {
            "root": "fix",
            "fields": {
                "sensor_timestamp_ns": {"from": "sensor_time_us", "type": "timestamp", "unit": "us"},
                "latitude_deg": {"from": "latitude", "type": "float"},
            },
        },
        target_cls=NestedRow,
        message_cls=_nested_message_class(),
    )

    assert result.is_valid
    assert "  Root: fix" in result.summary_lines
