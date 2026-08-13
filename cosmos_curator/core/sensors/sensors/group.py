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

"""SensorGroup: top-level coordinator for aligned multi-sensor sampling."""

from collections.abc import Generator, Iterator, Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.sensor_data import SensorData
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec

# Canonical error for sensors that expose no full-fidelity timeline read. Shared
# so the message stays single-sourced across every non-camera sensor.
STREAM_TIMESTAMPS_CAMERA_ONLY_MSG = "stream_timestamps is only implemented for CameraSensor"


class Sensor(Protocol):  # pragma: no cover
    """Structural interface for all sensor implementations.

    Any object with ``start_ns``, ``end_ns``, ``sample()``, and
    ``stream_timestamps()`` satisfies this protocol; explicit inheritance is not
    required. ``stream_timestamps()`` may raise ``NotImplementedError`` for
    sensor types without a full-fidelity timeline read (see its docstring).
    """

    @property
    def start_ns(self) -> int:
        """Earliest sensor timestamp in nanoseconds."""
        ...

    @property
    def end_ns(self) -> int:
        """Latest sensor timestamp in nanoseconds."""
        ...

    def supports_sampling_policy(self, policy: object) -> bool:
        """Return whether this sensor can sample with *policy*."""
        ...

    def sample(self, spec: SamplingSpec, *, policy: object) -> Generator[SensorData]:
        """Yield one ``SensorData`` per window in ``spec.grid``."""
        ...

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Yield the sensor's own ``int64`` ns timestamps in presentation order.

        Unlike ``sample()``, which resamples onto ``spec``'s grid, this streams
        every timestamp the sensor recorded, in order, with no resampling or
        decimation. Batches are consecutive and non-overlapping, and their union
        covers every timestamp.

        Args:
            batch_size: ``0`` yields all timestamps in a single batch; ``> 0``
                yields consecutive batches of that length (the final batch may be
                shorter). An empty timeline yields no batches in either mode.

        Yields:
            Timestamps in nanoseconds as ``int64`` arrays.

        Raises:
            NotImplementedError: for sensor types without a full-fidelity
                timeline read. Only ``CameraSensor`` implements this today.

        """
        ...


class SensorGroup:
    """Top-level coordinator for aligned multi-sensor sampling.

    ``SensorGroup`` owns a named collection of sensors, exposes aggregate
    ``start_ns`` / ``end_ns`` bounds, and drives all sensor generators in
    lockstep through a single ``.sample(spec, policies=...)`` entry point.

    Partial coverage:
        When a sensor has no data for a window it yields empty
        ``SensorData`` (``len(align_timestamps_ns) == 0``). Such sensors are
        omitted from that window's ``AlignedFrame.sensor_data``.  Windows
        where *every* sensor has no data produce an ``AlignedFrame`` with an
        empty ``sensor_data`` mapping.

    Policy enforcement:
        ``sample()`` requires one concrete policy per sensor id. The mapping is
        validated completely before any sensor sampling iterator is created or
        advanced. A ``ValueError`` raised by any sensor or policy check
        propagates to the caller unchanged.
    """

    def __init__(self, sensors: dict[str, Sensor]) -> None:
        """Initialise with a non-empty mapping of named sensors.

        Args:
            sensors: named sensors; must contain at least one entry.

        Raises:
            ValueError: if ``sensors`` is empty.

        """
        if not sensors:
            msg = "sensors must be non-empty"
            raise ValueError(msg)
        self._sensors = dict(sensors)

    @property
    def start_ns(self) -> int:
        """Minimum ``start_ns`` across all sensors."""
        return min(s.start_ns for s in self._sensors.values())

    @property
    def end_ns(self) -> int:
        """Maximum ``end_ns`` across all sensors."""
        return max(s.end_ns for s in self._sensors.values())

    def _validate_policies(self, policies: Mapping[str, object]) -> dict[str, object]:
        """Validate and return a concrete policy mapping for every sensor."""
        expected = set(self._sensors)
        provided = set(policies)
        errors: list[str] = []

        missing = sorted(expected - provided)
        if missing:
            errors.append(f"missing policy ids: {missing}")

        unknown = sorted(provided - expected)
        if unknown:
            errors.append(f"unknown policy ids: {unknown}")

        for sensor_id in sorted(expected & provided):
            policy = policies[sensor_id]
            if policy is None:
                errors.append(f"policy for {sensor_id!r} must be a concrete policy, got None")
                continue
            supports_policy = getattr(self._sensors[sensor_id], "supports_sampling_policy", None)
            if not callable(supports_policy) or not supports_policy(policy):
                errors.append(
                    f"unsupported policy type for {sensor_id!r}: {type(policy).__name__}",
                )

        if errors:
            msg = "; ".join(errors)
            raise ValueError(msg)

        return {sensor_id: policies[sensor_id] for sensor_id in self._sensors}

    def sample(self, spec: SamplingSpec, *, policies: Mapping[str, object]) -> Generator[AlignedFrame]:
        """Yield one ``AlignedFrame`` per window in ``spec.grid``.

        All sensor generators are started with the same ``spec`` and the
        matching concrete policy, then advanced in lockstep — one step per
        window. Each yielded frame carries
        ``align_timestamps_ns == window.timestamps_ns`` and a ``sensor_data``
        mapping that includes only sensors with data for that window.

        Args:
            spec: sampling specification; the same grid request is passed to
                every sensor generator.
            policies: mapping from sensor id to one concrete policy object for
                that sensor.

        Yields:
            ``AlignedFrame`` for each window in ``spec.grid``.

        Raises:
            ValueError: if the policy mapping is incomplete, contains unknown
                ids, contains ``None``, contains an unsupported policy type, or
                if any sensor's policy check fails.

        """
        policies_by_id = self._validate_policies(policies)
        generators = {name: sensor.sample(spec, policy=policies_by_id[name]) for name, sensor in self._sensors.items()}
        for window in spec.grid:
            sensor_data: dict[str, SensorData] = {}
            for name, gen in generators.items():
                data = next(gen)
                if len(data.align_timestamps_ns) > 0:
                    sensor_data[name] = data
            frame = AlignedFrame(
                align_timestamps_ns=window.timestamps_ns,
                sensor_data=sensor_data,
            )
            yield frame
