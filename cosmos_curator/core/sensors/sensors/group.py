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
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
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


def _advance_one_sensor(
    sensor_id: str,
    generator: Generator[SensorData],
    align_timestamps_ns: npt.NDArray[np.int64],
) -> SensorData:
    """Advance one sensor by a window and return its payload, or raise the window's failure.

    Kept separate from the frame loop so that a failure raises before the next
    sensor's generator is touched. That is what makes "first configured failure"
    deterministic rather than a race between sensors.

    Only an empty batch is judged here, because it is the one failure the group
    can see and ``AlignedFrame`` cannot distinguish: an empty payload would reach
    its validator as a length mismatch, indistinguishable from a broken sensor.
    Every other structural defect is left to ``AlignedFrame``, whose ``ValueError``
    is the right outcome for a sensor that returned something no recording could
    produce.
    """
    try:
        data = next(generator)
    except AlignmentError as error:
        # Attribute the failure to the configured id unconditionally, so
        # sensor_id is always a key into the group's sensors rather than
        # sometimes a key and sometimes a label. Re-raise the same object to keep
        # the original traceback.
        error.sensor_id = sensor_id
        # The timeline is different: sample_window_indices does supply one, and
        # what a sensor was actually asked for is more truthful than what the
        # group is currently iterating. Only fill it in when nothing set it.
        if error.align_timestamps_ns is None:
            error.align_timestamps_ns = align_timestamps_ns
        raise

    returned_ns = data.align_timestamps_ns
    if len(returned_ns) == 0 and len(align_timestamps_ns) > 0:
        # Unbind the payload before raising. This frame lands in the error's
        # traceback, and a caller holding the error would otherwise keep a
        # decoded batch alive with it.
        del data
        raise AlignmentError(
            AlignmentFailureReason.EMPTY_BATCH,
            f"no rows for the {len(align_timestamps_ns)} timestamps this window requested",
            sensor_id=sensor_id,
            align_timestamps_ns=align_timestamps_ns,
        )
    return data


class SensorGroup:
    """Top-level coordinator for aligned multi-sensor sampling.

    ``SensorGroup`` owns a named collection of sensors, exposes aggregate
    ``start_ns`` / ``end_ns`` bounds, and drives all sensor generators in
    lockstep through a single ``.sample(spec, policies=...)`` entry point.

    Required-sensor atomicity:
        Every configured sensor is required for every window. A window produces
        a complete ``AlignedFrame`` — every configured sensor present, one
        logical row per requested timestamp, ``align_timestamps_ns`` exactly
        equal to ``window.timestamps_ns`` — or it fails. No sensor is silently
        omitted and no partial frame is yielded.

        A window that carries no reference timestamps at all is a real window
        with nothing to sample, not a failure: every sensor yields a zero-row
        payload and the frame is empty.

    How a window fails:
        ``AlignmentError`` means the recording could not serve the request: a
        sensor covered none of the window, or its nearest observation was
        further away than the policy allows. Nothing is broken, and the caller's
        job is to drop that input. Callers place one ``try/except
        AlignmentError`` around iteration and own the disposition.

        Anything else keeps its own exception type. A payload with the wrong row
        count or a timeline that was never requested is a defect in the sensor,
        and ``AlignedFrame`` rejects it with ``ValueError`` — which should reach
        someone who can fix the code rather than be quarantined as bad data.

        Not everything outside ``AlignmentError`` is a defect, though. Corrupt
        media and unreadable sources are bad data that no code fix repairs; they
        keep their own types only because this contract does not classify them.
        A caller wanting to drop bad inputs needs a bucket for those too.

    Ordering:
        Sensors are advanced one at a time in configured order and each is
        checked before the next is touched, so the first configured
        ``AlignmentError`` is raised deterministically and later sensors are
        never advanced for that window. Iteration is not resumable: a failed
        window ends the generator, so that window and every later window produce
        no frame.

    Policy enforcement:
        ``sample()`` requires one concrete policy per sensor id. The mapping is
        validated completely before any sensor is advanced, so a bad mapping
        opens no sources and decodes nothing. (Creating a generator is not
        advancing one: ``sample()`` is a generator function, so its body — and
        any resource it acquires — waits for the first advance.) A ``ValueError``
        raised by any sensor or policy check propagates to the caller unchanged.
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
        window, one sensor at a time in configured order. Each yielded frame
        carries ``align_timestamps_ns == window.timestamps_ns`` and a
        ``sensor_data`` mapping containing every configured sensor.

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
            AlignmentError: at the first window the recording cannot serve — a
                configured sensor covered none of it, or its nearest observation
                was outside the policy's tolerance. Iteration stops there.
            ValueError: if a sensor returns a payload no recording could
                produce, such as the wrong row count or a timeline that was
                never requested. Raised by ``AlignedFrame``, and a defect in
                that sensor rather than a property of the data.
            Exception: whatever a sensor raises while being torn down, if
                iteration was otherwise clean. Every sensor is still closed
                first, and a teardown failure never replaces an error already
                in flight.

        """
        policies_by_id = self._validate_policies(policies)
        # ``sample()`` is a generator function, so this acquires nothing and cannot
        # fail — the bodies do not run until the first advance below.
        generators = {name: sensor.sample(spec, policy=policies_by_id[name]) for name, sensor in self._sensors.items()}
        drained = False
        try:
            for window in spec.grid:
                # Built inline rather than through a named local. A raised
                # AlignmentError keeps this frame alive through its traceback, and
                # a local here would pin the previous window's decoded payloads —
                # a whole window of frame buffers per retained error.
                yield AlignedFrame(
                    align_timestamps_ns=window.timestamps_ns,
                    sensor_data={
                        name: _advance_one_sensor(name, gen, window.timestamps_ns) for name, gen in generators.items()
                    },
                )
            drained = True
        finally:
            # Sensors suspend inside their own `with` blocks, holding decoders and
            # MCAP readers. A failed window leaves every generator parked there,
            # and a caller that keeps the AlignmentError keeps this frame — and so
            # these generators — reachable, so refcounting never releases them.
            # Close them here rather than at some later finalization.
            #
            # Teardown can itself fail — a decoder rejecting a stream on flush,
            # say. Every sensor still gets a close attempt, and a teardown error
            # never replaces whatever we were already unwinding: that would
            # silence the AlignmentError callers were told to catch and turn a
            # droppable recording into a dead stage. Only when this iteration
            # ran to completion is there nothing to protect, so the teardown
            # error surfaces then. That is tracked here rather than read from
            # interpreter state, which would also see an exception the caller
            # happened to be handling around the loop.
            teardown_errors: list[Exception] = []
            for generator in generators.values():
                try:
                    generator.close()
                except Exception as error:  # noqa: BLE001
                    teardown_errors.append(error)
            if drained and teardown_errors:
                raise teardown_errors[0]
