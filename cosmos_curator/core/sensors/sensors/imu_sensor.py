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
"""MCAP protobuf IMU sensor."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from google.protobuf.message import Message
from mcap.reader import McapReader
from mcap.records import Channel, Schema

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.imu_data import ImuData
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NoSamplingPolicy, require_no_sampling_policy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.mcap import (
    McapProtobufMessageResolver,
    McapTopicAccessor,
    parse_protobuf_message,
)
from cosmos_curator.core.sensors.utils.protobuf_mapper import ProtobufRowMapper

DEFAULT_TOPIC = "/imu"
_VECTOR_COLUMNS = 3
REQUIRED_IMU_MAPPING_FIELDS = frozenset({"sensor_timestamp_ns", "angular_velocity_rad_s", "linear_acceleration_m_s2"})


@dataclass(frozen=True)
class DecodedImuSample:
    """Decoded IMU protobuf fields normalized for ``ImuData`` construction."""

    sensor_timestamp_ns: int
    align_timestamp_ns: int
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]
    angular_velocity_valid: tuple[bool, bool, bool] = (True, True, True)
    linear_acceleration_valid: tuple[bool, bool, bool] = (True, True, True)
    angular_velocity_bias_rad_s: tuple[float, float, float] | None = None
    linear_acceleration_bias_m_s2: tuple[float, float, float] | None = None
    angular_velocity_bias_valid: tuple[bool, bool, bool] | None = None
    linear_acceleration_bias_valid: tuple[bool, bool, bool] | None = None
    host_timestamp_ns: int | None = None
    sequence_counter: int | None = None
    temperature_c: float | None = None
    temperature_valid: bool | None = None


def _imu_data_from_samples(  # noqa: C901, PLR0913
    samples: list[DecodedImuSample],
    *,
    host: bool,
    sequence: bool,
    temperature: bool,
    angular_velocity_bias: bool,
    linear_acceleration_bias: bool,
) -> ImuData:
    """Build an ``ImuData`` batch from decoded IMU samples."""
    align = np.array([sample.align_timestamp_ns for sample in samples], dtype=np.int64)
    if np.any(align[1:] <= align[:-1]):
        msg = "IMU samples require strictly increasing align_timestamp_ns"
        raise ValueError(msg)
    seq_values = [sample.sequence_counter for sample in samples]
    if sequence and any(v is None or int(v) < 0 or int(v) > int(np.iinfo(np.uint64).max) for v in seq_values):
        msg = "sequence_counter values must be present and fit uint64"
        raise ValueError(msg)
    host_values = [sample.host_timestamp_ns for sample in samples]
    if host and any(v is None for v in host_values):
        msg = "host_timestamp_ns must be present for every IMU sample"
        raise ValueError(msg)
    temperature_array: npt.NDArray[np.float64] | None = None
    temperature_valid_array: npt.NDArray[np.bool_] | None = None
    if temperature:
        temperature_values = [sample.temperature_c for sample in samples]
        if any(value is None for value in temperature_values):
            msg = "temperature_c must be present for every IMU sample"
            raise ValueError(msg)
        temperature_validity = [sample.temperature_valid for sample in samples]
        if any(value is None for value in temperature_validity):
            msg = "temperature_valid must be present for every IMU sample"
            raise ValueError(msg)
        temperature_array = np.array(temperature_values, dtype=np.float64)
        temperature_valid_array = np.array(temperature_validity, dtype=np.bool_)
    angular_velocity_bias_array: npt.NDArray[np.float64] | None = None
    angular_velocity_bias_valid_array: npt.NDArray[np.bool_] | None = None
    if angular_velocity_bias:
        angular_velocity_bias_values = [sample.angular_velocity_bias_rad_s for sample in samples]
        angular_velocity_bias_validity = [sample.angular_velocity_bias_valid for sample in samples]
        if any(value is None for value in angular_velocity_bias_values) or any(
            value is None for value in angular_velocity_bias_validity
        ):
            msg = "angular velocity bias and validity must be present for every IMU sample"
            raise ValueError(msg)
        angular_velocity_bias_array = np.array(angular_velocity_bias_values, dtype=np.float64)
        angular_velocity_bias_valid_array = np.array(angular_velocity_bias_validity, dtype=np.bool_)
    linear_acceleration_bias_array: npt.NDArray[np.float64] | None = None
    linear_acceleration_bias_valid_array: npt.NDArray[np.bool_] | None = None
    if linear_acceleration_bias:
        linear_acceleration_bias_values = [sample.linear_acceleration_bias_m_s2 for sample in samples]
        linear_acceleration_bias_validity = [sample.linear_acceleration_bias_valid for sample in samples]
        if any(value is None for value in linear_acceleration_bias_values) or any(
            value is None for value in linear_acceleration_bias_validity
        ):
            msg = "linear acceleration bias and validity must be present for every IMU sample"
            raise ValueError(msg)
        linear_acceleration_bias_array = np.array(linear_acceleration_bias_values, dtype=np.float64)
        linear_acceleration_bias_valid_array = np.array(linear_acceleration_bias_validity, dtype=np.bool_)
    return ImuData(
        align_timestamps_ns=align,
        sensor_timestamps_ns=np.array([sample.sensor_timestamp_ns for sample in samples], dtype=np.int64),
        angular_velocity_rad_s=np.array([sample.angular_velocity_rad_s for sample in samples], dtype=np.float64),
        linear_acceleration_m_s2=np.array([sample.linear_acceleration_m_s2 for sample in samples], dtype=np.float64),
        angular_velocity_valid=np.array([sample.angular_velocity_valid for sample in samples], dtype=np.bool_),
        linear_acceleration_valid=np.array([sample.linear_acceleration_valid for sample in samples], dtype=np.bool_),
        angular_velocity_bias_rad_s=angular_velocity_bias_array,
        linear_acceleration_bias_m_s2=linear_acceleration_bias_array,
        angular_velocity_bias_valid=angular_velocity_bias_valid_array,
        linear_acceleration_bias_valid=linear_acceleration_bias_valid_array,
        host_timestamps_ns=np.array(host_values, dtype=np.int64) if host else None,
        sequence_counter=np.array(seq_values, dtype=np.uint64) if sequence else None,
        temperature_c=temperature_array,
        temperature_valid=temperature_valid_array,
    )


def _empty_imu_data(
    *,
    host: bool,
    sequence: bool,
    temperature: bool,
    angular_velocity_bias: bool,
    linear_acceleration_bias: bool,
) -> ImuData:
    """Build an empty raw IMU sample batch."""
    empty_ts = np.empty(0, dtype=np.int64)
    return ImuData(
        align_timestamps_ns=empty_ts,
        sensor_timestamps_ns=empty_ts,
        angular_velocity_rad_s=np.empty((0, _VECTOR_COLUMNS), dtype=np.float64),
        linear_acceleration_m_s2=np.empty((0, _VECTOR_COLUMNS), dtype=np.float64),
        angular_velocity_valid=np.empty((0, _VECTOR_COLUMNS), dtype=np.bool_),
        linear_acceleration_valid=np.empty((0, _VECTOR_COLUMNS), dtype=np.bool_),
        angular_velocity_bias_rad_s=(
            np.empty((0, _VECTOR_COLUMNS), dtype=np.float64) if angular_velocity_bias else None
        ),
        linear_acceleration_bias_m_s2=(
            np.empty((0, _VECTOR_COLUMNS), dtype=np.float64) if linear_acceleration_bias else None
        ),
        angular_velocity_bias_valid=(np.empty((0, _VECTOR_COLUMNS), dtype=np.bool_) if angular_velocity_bias else None),
        linear_acceleration_bias_valid=(
            np.empty((0, _VECTOR_COLUMNS), dtype=np.bool_) if linear_acceleration_bias else None
        ),
        host_timestamps_ns=np.empty(0, dtype=np.int64) if host else None,
        sequence_counter=np.empty(0, dtype=np.uint64) if sequence else None,
        temperature_c=np.empty(0, dtype=np.float64) if temperature else None,
        temperature_valid=np.empty(0, dtype=np.bool_) if temperature else None,
    )


class ImuSensor:
    """MCAP-backed raw IMU sensor.

    Reads mapped protobuf messages from an MCAP topic. ``schema_name`` selects
    the message declared by the MCAP's embedded descriptor, and
    ``protobuf_mapping`` maps that message into ``ImuData`` fields. Optional
    output arrays are present only when their corresponding values are mapped:
    ``host_timestamp_ns``, ``sequence_counter``, ``temperature_c``,
    ``angular_velocity_bias_rad_s``, or ``linear_acceleration_bias_m_s2``.
    Mapping temperature or bias values without their corresponding validity
    fields defaults validity to true. Empty and populated windows preserve the
    same optional-array presence.
    Once mapped, an optional value and its validity mask must be available for
    every message in a populated batch; producers should use the validity mask
    rather than omit a mapped field.
    Every populated batch requires strictly increasing alignment timestamps;
    duplicate or decreasing values raise ``ValueError``. Mappings whose sensor
    clock can repeat should map ``align_timestamp_ns`` from a stable clock.
    """

    def __init__(
        self,
        source: DataSource,
        topic: str = DEFAULT_TOPIC,
        *,
        schema_name: str,
        protobuf_mapping: str | Path | Mapping[str, Any],
        extrinsics: SensorExtrinsics | None = None,
    ) -> None:
        """Initialize the MCAP IMU sensor."""
        self._topic = topic
        self._extrinsics = extrinsics
        self._mcap = McapTopicAccessor(source, topic)
        self._protobuf_resolver = McapProtobufMessageResolver(schema_name, schema_label="IMU protobuf")
        self._protobuf_row_mapper = ProtobufRowMapper(protobuf_mapping, target_cls=DecodedImuSample)
        mapped = self._protobuf_row_mapper.mapped_target_names
        missing = sorted(REQUIRED_IMU_MAPPING_FIELDS - mapped)
        if missing:
            msg = "IMU protobuf mapping must map required field(s): " + ", ".join(map(repr, missing))
            raise ValueError(msg)
        if "temperature_valid" in mapped and "temperature_c" not in mapped:
            msg = "IMU protobuf mapping cannot map 'temperature_valid' without 'temperature_c'"
            raise ValueError(msg)
        if "angular_velocity_bias_valid" in mapped and "angular_velocity_bias_rad_s" not in mapped:
            msg = "IMU protobuf mapping cannot map 'angular_velocity_bias_valid' without 'angular_velocity_bias_rad_s'"
            raise ValueError(msg)
        if "linear_acceleration_bias_valid" in mapped and "linear_acceleration_bias_m_s2" not in mapped:
            msg = (
                "IMU protobuf mapping cannot map 'linear_acceleration_bias_valid' "
                "without 'linear_acceleration_bias_m_s2'"
            )
            raise ValueError(msg)
        self._include_host = "host_timestamp_ns" in mapped
        self._include_sequence = "sequence_counter" in mapped
        self._include_temperature = "temperature_c" in mapped
        self._include_angular_velocity_bias = "angular_velocity_bias_rad_s" in mapped
        self._include_linear_acceleration_bias = "linear_acceleration_bias_m_s2" in mapped
        self._empty_imu_data: ImuData | None = None

    @property
    def start_ns(self) -> int:
        """Earliest IMU message time on this topic, in nanoseconds."""
        return self._mcap.start_ns

    @property
    def end_ns(self) -> int:
        """Latest IMU message time on this topic, in nanoseconds."""
        return self._mcap.end_ns

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return self._mcap.max_gap_ns

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Message times in nanoseconds from raw MCAP ``log_time`` values."""
        return self._mcap.timestamps_ns

    def _resolve_message_class(self, reader: McapReader) -> type[Message] | None:
        """Resolve and cache the IMU message class from the MCAP summary when available."""
        return self._protobuf_resolver.resolve_from_summary(reader, self._topic)

    def _message_class_for_message(self, schema: Schema | None, channel: Channel) -> type[Message]:
        """Resolve the dynamic message class and validate each message channel."""
        return self._protobuf_resolver.message_class_for_message(schema, channel, topic=self._topic)

    def _get_empty_imu_data(self) -> ImuData:
        """Return a cached empty raw IMU sample batch."""
        if self._empty_imu_data is None:
            self._empty_imu_data = _empty_imu_data(
                host=self._include_host,
                sequence=self._include_sequence,
                temperature=self._include_temperature,
                angular_velocity_bias=self._include_angular_velocity_bias,
                linear_acceleration_bias=self._include_linear_acceleration_bias,
            )
        return self._empty_imu_data

    def _read_messages(
        self,
        reader: McapReader,
        start_ns: int,
        exclusive_end_ns: int,
    ) -> ImuData:
        """Read mapped IMU samples in one half-open MCAP log-time range."""
        samples: list[DecodedImuSample] = []
        for schema, channel, message in self._mcap.iter_messages(
            reader,
            start_ns,
            exclusive_end_ns,
            log_time_order=True,
        ):
            resolved_cls = self._message_class_for_message(schema, channel)
            message_obj = parse_protobuf_message(
                resolved_cls,
                message.data,
                topic=self._topic,
                sensor_label="IMU",
            )
            sample = self._protobuf_row_mapper(message_obj, mcap_logtime_ns=int(message.log_time))
            samples.append(sample)

        if not samples:
            return self._get_empty_imu_data()
        return _imu_data_from_samples(
            samples,
            host=self._include_host,
            sequence=self._include_sequence,
            temperature=self._include_temperature,
            angular_velocity_bias=self._include_angular_velocity_bias,
            linear_acceleration_bias=self._include_linear_acceleration_bias,
        )

    def _read_window_messages(
        self,
        reader: McapReader,
        window: SamplingWindow,
    ) -> ImuData:
        """Read all IMU samples whose message time overlaps one sampling window."""
        return self._read_messages(reader, int(window.start_ns), int(window.exclusive_end_ns))

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        return isinstance(policy, NoSamplingPolicy)

    def read_all(self) -> ImuData:
        """Decode every message on the configured IMU topic into one batch."""
        with self._mcap.open_reader() as reader:
            self._resolve_message_class(reader)
            return self._read_messages(reader, self.start_ns, self.end_ns + 1)

    def sample(self, spec: SamplingSpec, *, policy: NoSamplingPolicy) -> Generator[ImuData]:
        """Yield raw IMU sample batches for each sampling window.

        Each yielded batch contains the MCAP messages whose ``log_time`` falls
        inside the current half-open sampling window. ``align_timestamps_ns``
        copies the mapped sensor timestamp unless the mapping explicitly supplies
        ``align_timestamp_ns`` from another clock.
        """
        require_no_sampling_policy(policy, sensor_name=type(self).__name__)
        with self._mcap.open_reader() as reader:
            self._resolve_message_class(reader)
            for window in spec.grid:
                yield self._read_window_messages(reader, window)
