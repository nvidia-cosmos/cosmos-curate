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
"""Smoke-test a GPS, IMU, or ego-trajectory protobuf YAML mapping against an MCAP schema."""

import argparse
import dataclasses
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Never, cast

from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message
from mcap.exceptions import McapError
from mcap.reader import McapReader, make_reader
from mcap.records import Channel, Schema

from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import (
    REQUIRED_EGO_TRAJECTORY_MAPPING_FIELDS,
    DecodedEgoTrajectorySample,
)
from cosmos_curator.core.sensors.sensors.gps_sensor import (
    REQUIRED_GPS_MAPPING_FIELDS,
    DecodedGpsSample,
)
from cosmos_curator.core.sensors.sensors.imu_sensor import (
    REQUIRED_IMU_MAPPING_FIELDS,
    DecodedImuSample,
)
from cosmos_curator.core.sensors.utils.mcap import (
    McapProtobufMessageResolver,
    channel_for_topic,
    schema_for_channel,
)
from cosmos_curator.core.sensors.utils.protobuf_mapper import (
    MISSING,
    PROTOBUF_NUMERIC_TYPES,
    FieldMapping,
    ProtobufRowMapper,
    load_spec,
    paired_target_name,
    parse_one_mapping,
    parse_optional_str,
    path_segments,
    require_collection_target,
    require_mapping,
    resolve_descriptor_path,
    target_metadata,
    validate_mapping_target_type,
    validate_target_class,
    validate_valid_when_source,
)

_PROTOBUF_TIMESTAMP_TYPES = PROTOBUF_NUMERIC_TYPES - {FieldDescriptor.TYPE_ENUM}


def _invalid(message: str, *, cause: Exception | None = None) -> Never:
    if cause is not None:
        raise ValueError(message) from cause
    raise ValueError(message)


@dataclasses.dataclass(frozen=True)
class ProtobufMappingValidationResult:
    """Collected problems and printable summary from structural validation."""

    problems: tuple[str, ...]
    summary_lines: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether validation found no problems."""
        return not self.problems


class _ValidationCollector:
    def __init__(
        self,
        spec: Mapping[str, Any],
        target_cls: type[Any],
        message_cls: type[Message],
        required_target_names: Collection[str],
    ) -> None:
        self.spec = spec
        self.target_cls = target_cls
        self.message_cls = message_cls
        self.required_target_names = required_target_names
        self.problems: list[str] = []
        self.target_annotations: dict[str, object] = {}
        self.non_init_target_names: set[str] = set()
        self.parsed_mappings: list[FieldMapping] = []
        self.known_mappings: dict[str, FieldMapping] = {}
        self.summary_lines: list[str] = []
        self.root_segments: tuple[str, ...] = ()
        self.root_is_valid = True

    def add_problem(self, location: str, error: str | Exception) -> None:
        self.problems.append(f"{location}: {error}")

    def run(self) -> ProtobufMappingValidationResult:
        self._validate_target()
        self._parse_root()
        self._parse_fields()
        self._validate_required_fields()
        self._validate_validity_pairs()
        root_descriptor = self._resolve_root_descriptor()
        if root_descriptor is not None:
            self._validate_sources(root_descriptor)
        if not self.problems:
            self._run_mapper_smoke_check()
        required_names = set(self.required_target_names)
        required_mapped = len(required_names & set(self.known_mappings))
        header = [
            "Mapping summary",
            f"  Root: {'.'.join(self.root_segments) or '<message>'}",
            f"  Required fields: {required_mapped}/{len(required_names)} mapped",
            f"  Total mappings: {len(self.parsed_mappings)}",
            "",
            "Mappings (* required)",
        ]
        return ProtobufMappingValidationResult(
            problems=tuple(self.problems),
            summary_lines=tuple(header + self.summary_lines),
        )

    def _validate_target(self) -> None:
        try:
            target_cls = validate_target_class(self.target_cls, "target_cls")
            self.target_annotations, _target_defaults, self.non_init_target_names = target_metadata(target_cls)
        except ValueError as e:
            self.add_problem("target_cls", e)

    def _parse_root(self) -> None:
        try:
            root_path = parse_optional_str(self.spec.get("root"), "root") or ""
            self.root_segments = path_segments(root_path, "root") if root_path else ()
        except ValueError as e:
            self.add_problem("root", e)
            self.root_is_valid = False

    def _parse_fields(self) -> None:
        try:
            raw_fields = require_mapping(self.spec.get("fields"), "fields")
        except ValueError as e:
            self.add_problem("fields", e)
            return
        for raw_target_name, raw_mapping in raw_fields.items():
            if not isinstance(raw_target_name, str):
                self.add_problem("fields", "fields keys must be target attribute names")
                continue
            self._parse_field(raw_target_name, raw_mapping)

    def _parse_field(self, target_name: str, raw_mapping: object) -> None:
        location = f"fields.{target_name}"
        target_known = target_name in self.target_annotations
        if target_name in self.non_init_target_names:
            self.add_problem(location, f"target attribute {target_name!r} has init=False")
            target_known = False
        elif not target_known:
            expected = ", ".join(sorted(self.target_annotations))
            self.add_problem(location, f"unknown target attribute {target_name!r}; expected one of: {expected}")
        try:
            mapping = parse_one_mapping(target_name, raw_mapping)
        except ValueError as e:
            self.add_problem(location, e)
            return
        self.parsed_mappings.append(mapping)
        if not target_known:
            return
        self.known_mappings[target_name] = mapping
        try:
            validate_mapping_target_type(self.target_annotations[target_name], mapping)
        except ValueError as e:
            self.add_problem(location, e)

    def _validate_required_fields(self) -> None:
        for required_name in sorted(set(self.required_target_names) - set(self.known_mappings)):
            self.add_problem(
                f"fields.{required_name}",
                f"required target field {required_name!r} is not mapped",
            )

    def _validate_validity_pairs(self) -> None:
        for target_name in self.known_mappings:
            if not target_name.endswith("_valid"):
                continue
            try:
                paired_name = paired_target_name(target_name, self.target_annotations)
            except ValueError as e:
                self.add_problem(f"fields.{target_name}", str(e))
                continue
            if paired_name in self.target_annotations and paired_name not in self.known_mappings:
                self.add_problem(
                    f"fields.{target_name}",
                    f"maps validity for {paired_name!r}, but fields.{paired_name} is not mapped",
                )

    def _resolve_root_descriptor(self) -> Descriptor | None:
        try:
            root_descriptor = self.message_cls.DESCRIPTOR
        except AttributeError:
            self.add_problem("message_cls", "message_cls must be a protobuf Message class with a DESCRIPTOR")
            return None
        if not self.root_is_valid:
            return None
        if not self.root_segments:
            return root_descriptor
        try:
            root_field = resolve_descriptor_path(
                root_descriptor,
                self.root_segments,
                "root",
                allow_terminal_repeated=False,
            )
        except ValueError as e:
            self.add_problem("root", e)
            return None
        if root_field.message_type is None:
            self.add_problem(
                "root",
                f"root path {'.'.join(self.root_segments)!r} does not resolve to a protobuf message",
            )
            return None
        return root_field.message_type

    def _validate_sources(self, root_descriptor: Descriptor) -> None:
        required_names = set(self.required_target_names)
        for mapping in self.parsed_mappings:
            sources = tuple(
                self._validate_source(root_descriptor, mapping, source_name, source_path)
                for source_name, source_path in mapping.sources
            )
            marker = "* " if mapping.target_name in required_names else "  "
            self.summary_lines.extend(["", f"{marker}{mapping.target_name}"])
            if not sources:
                self.summary_lines.append("    path: <default>")
            elif len(sources) == 1:
                self.summary_lines.append(f"    path: {sources[0]}")
            else:
                self.summary_lines.append("    paths:")
                self.summary_lines.extend(f"      - {source}" for source in sources)
            self.summary_lines.extend(
                [
                    f"    mapping: {_conversion_label(mapping)}",
                    f"    target: {_annotation_label(self.target_annotations.get(mapping.target_name))}",
                ]
            )

    def _validate_source(
        self,
        root_descriptor: Descriptor,
        mapping: FieldMapping,
        source_name: str,
        source_path: tuple[str, ...] | None,
    ) -> str:
        location = f"fields.{mapping.target_name}"
        if source_path is None:
            source_type = FieldDescriptor.TYPE_UINT64
            source_field = None
        else:
            try:
                source_field = resolve_descriptor_path(
                    root_descriptor,
                    source_path,
                    location,
                    allow_terminal_repeated=not mapping.is_group,
                )
            except ValueError as e:
                self.add_problem(location, e)
                return f"{source_name} [unresolved]"
            source_type = source_field.type
        try:
            _validate_source_type(mapping, source_name, source_type)
        except ValueError as e:
            self.add_problem(location, e)
        if source_field is not None and source_field.is_repeated:
            try:
                require_collection_target(self.target_annotations.get(mapping.target_name), mapping.target_name)
            except ValueError as e:
                self.add_problem(location, e)
        repeated = "repeated " if source_field is not None and source_field.is_repeated else ""
        return f"{source_name} [{repeated}{_protobuf_type_label(source_type)}]"

    def _run_mapper_smoke_check(self) -> None:
        try:
            ProtobufRowMapper(self.spec, target_cls=self.target_cls, message_cls=self.message_cls)
        except ValueError as e:
            self.add_problem("mapper", e)


def validate_protobuf_mapping(
    yaml_spec: str | Path | Mapping[str, Any],
    *,
    target_cls: type[Any],
    message_cls: type[Message],
    required_target_names: Collection[str] = (),
) -> ProtobufMappingValidationResult:
    """Validate a mapping structurally without decoding a protobuf message."""
    try:
        spec = load_spec(yaml_spec)
    except (OSError, UnicodeError, ValueError) as e:
        return ProtobufMappingValidationResult((f"spec: {e}",))
    if "target" in spec:
        problem = "target: protobuf mapper YAML must not define 'target'; the sensor supplies the target class"
        return ProtobufMappingValidationResult((problem,))
    return _ValidationCollector(spec, target_cls, message_cls, required_target_names).run()


def _protobuf_type_label(source_type: int) -> str:
    label = cast("str", descriptor_pb2.FieldDescriptorProto.Type.Name(source_type))
    return label.removeprefix("TYPE_").lower()


def _annotation_label(annotation: object) -> str:
    if annotation is None:
        return "unknown target"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _conversion_label(mapping: FieldMapping) -> str:
    if mapping.default is not MISSING:
        return f"default value {mapping.default!r}"
    if mapping.valid_when_equals is not MISSING:
        return f"true when value equals {mapping.valid_when_equals!r}"
    if mapping.value_type == "timestamp":
        return f"timestamp from {mapping.unit} to ns"
    return f"{mapping.value_type} conversion" if mapping.value_type is not None else "identity"


def _validate_source_type(mapping: FieldMapping, source_name: str, source_type: int) -> None:
    if mapping.valid_when_equals is not MISSING:
        validate_valid_when_source(mapping, source_name, source_type)
        return
    if (
        (mapping.value_type in {"float", "int"} and source_type in PROTOBUF_NUMERIC_TYPES)
        or (mapping.value_type == "timestamp" and source_type in _PROTOBUF_TIMESTAMP_TYPES)
        or (mapping.value_type == "bool" and source_type == FieldDescriptor.TYPE_BOOL)
    ):
        return
    if mapping.value_type == "bool":
        _invalid(
            f"fields.{mapping.target_name} uses direct bool conversion for non-bool protobuf field "
            f"{source_name!r}; use valid_when"
        )
    else:
        expected = f"a protobuf scalar compatible with {mapping.value_type!r}"
    _invalid(
        f"fields.{mapping.target_name} source {source_name!r} has protobuf type "
        f"{_protobuf_type_label(source_type)!r}; expected {expected}"
    )


PASS_EXIT_CODE = 0
INVALID_MAPPING_EXIT_CODE = 1
INPUT_ERROR_EXIT_CODE = 2


_TARGET_CONFIGS: dict[str, tuple[type[Any], frozenset[str]]] = {
    "egotrajectory": (DecodedEgoTrajectorySample, REQUIRED_EGO_TRAJECTORY_MAPPING_FIELDS),
    "gps": (DecodedGpsSample, REQUIRED_GPS_MAPPING_FIELDS),
    "imu": (DecodedImuSample, REQUIRED_IMU_MAPPING_FIELDS),
}


def _schema_and_channel_for_topic(reader: McapReader, topic: str) -> tuple[Schema, Channel]:
    summary = reader.get_summary()
    if summary is not None:
        channel = channel_for_topic(summary, topic)
        if channel is None:
            msg = f"no MCAP channel found for topic {topic!r}"
            raise ValueError(msg)
        return schema_for_channel(summary, channel, topic), channel

    for schema, channel, _message in reader.iter_messages(topics=[topic]):
        if schema is None:
            msg = f"MCAP message on topic {topic!r} is missing a protobuf schema"
            raise ValueError(msg)
        return schema, channel
    msg = f"no MCAP messages found for topic {topic!r} in an unindexed file"
    raise ValueError(msg)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a GPS, IMU, or ego-trajectory protobuf YAML mapping against the descriptor embedded in "
            "an MCAP without decoding message payloads."
        )
    )
    parser.add_argument("--target", choices=sorted(_TARGET_CONFIGS), required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress successful validation output; write invalid-mapping problems to stderr.",
    )
    return parser


def _write_mapping_summary(result: ProtobufMappingValidationResult) -> None:
    sys.stdout.write("\n".join(("", *result.summary_lines, "")))


def main(argv: Sequence[str] | None = None) -> int:
    """Validate command arguments and return a process exit code."""
    args = _parser().parse_args(argv)
    target_cls, required_target_names = _TARGET_CONFIGS[args.target]
    try:
        yaml_spec = args.mapping.read_text(encoding="utf-8")
        with args.mcap.open("rb") as stream:
            reader = make_reader(stream)
            schema, channel = _schema_and_channel_for_topic(reader, args.topic)
            resolver = McapProtobufMessageResolver(schema.name)
            message_cls = resolver.message_class_for_message(schema, channel, topic=args.topic)
    except McapError as e:
        detail = str(e) or type(e).__name__
        sys.stderr.write(f"ERROR: failed to read MCAP: {detail}\n")
        return INPUT_ERROR_EXIT_CODE
    except (OSError, UnicodeError, ValueError) as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return INPUT_ERROR_EXIT_CODE

    result = validate_protobuf_mapping(
        yaml_spec,
        target_cls=target_cls,
        message_cls=message_cls,
        required_target_names=required_target_names,
    )
    if args.quiet:
        if result.is_valid:
            return PASS_EXIT_CODE
        sys.stderr.write(f"FAIL: mapping has {len(result.problems)} problem(s).\n")
        for problem in result.problems:
            sys.stderr.write(f"- {problem}\n")
        return INVALID_MAPPING_EXIT_CODE

    sys.stdout.write(f"Target: {args.target}\n")
    sys.stdout.write(f"Topic: {args.topic}\n")
    sys.stdout.write(f"Schema: {schema.name}\n")
    _write_mapping_summary(result)
    if result.is_valid:
        sys.stdout.write("\nPASS: mapping is structurally valid.\n")
        return PASS_EXIT_CODE

    sys.stdout.write(f"\nFAIL: mapping has {len(result.problems)} problem(s).\n")
    for problem in result.problems:
        sys.stdout.write(f"- {problem}\n")
    return INVALID_MAPPING_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
