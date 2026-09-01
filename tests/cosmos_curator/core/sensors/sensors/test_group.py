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

"""Tests for SensorGroup."""

import gc
import weakref
from collections.abc import Generator, Iterator

import attrs
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import STREAM_TIMESTAMPS_CAMERA_ONLY_MSG, SensorGroup


@attrs.define
class _FakeSensorData:
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]


class _FakeSensor:
    """Minimal in-memory sensor for unit tests: nearest-neighbour sampling, no I/O."""

    def __init__(self, sensor_timestamps_ns: npt.NDArray[np.int64]) -> None:
        self._ts = np.array(sensor_timestamps_ns, dtype=np.int64, copy=True)
        self.sample_started = False
        self.policy_seen: object | None = None

    @property
    def start_ns(self) -> int:
        return int(self._ts[0])

    @property
    def end_ns(self) -> int:
        return int(self._ts[-1])

    def supports_sampling_policy(self, policy: object) -> bool:
        return isinstance(policy, NearestTimestampPolicy)

    def sample(self, spec: SamplingSpec, *, policy: object) -> Generator[_FakeSensorData]:
        self.sample_started = True
        self.policy_seen = policy
        empty = np.empty(0, dtype=np.int64)
        for window in spec.grid:
            if len(window) == 0:
                yield _FakeSensorData(align_timestamps_ns=empty, sensor_timestamps_ns=empty)
                continue
            indices, _counts = sample_window_indices(self._ts, window, policy=policy, dedup=False)
            if len(indices) == 0:
                yield _FakeSensorData(align_timestamps_ns=empty, sensor_timestamps_ns=empty)
                continue
            yield _FakeSensorData(
                align_timestamps_ns=np.array(window.timestamps_ns, dtype=np.int64),
                sensor_timestamps_ns=self._ts[indices],
            )

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Not implemented: the timestamp stream is camera-only for now."""
        del batch_size
        raise NotImplementedError(STREAM_TIMESTAMPS_CAMERA_ONLY_MSG)


class _SensorWithoutPolicySupport:
    """Sensor-shaped test double for legacy objects without policy support checks."""

    sample_started = False

    @property
    def start_ns(self) -> int:
        return 0

    @property
    def end_ns(self) -> int:
        return 1_000

    def sample(self, spec: SamplingSpec, *, policy: object) -> Generator[_FakeSensorData]:
        del spec, policy
        self.sample_started = True
        yield _FakeSensorData(
            align_timestamps_ns=np.array([0], dtype=np.int64),
            sensor_timestamps_ns=np.array([0], dtype=np.int64),
        )

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        del batch_size
        raise NotImplementedError(STREAM_TIMESTAMPS_CAMERA_ONLY_MSG)


class _ScriptedSensor:
    """Sensor test double that replays a fixed script and records how often it was advanced.

    Each script entry is either a ``_FakeSensorData`` to yield or an exception to
    raise. ``advances`` counts how many times the generator body ran, which is how
    the tests assert that a later sensor was never advanced.
    """

    def __init__(self, script: list[object]) -> None:
        self._script = script
        self.advances = 0
        self.sample_started = False
        self.released = False

    @property
    def start_ns(self) -> int:
        return 0

    @property
    def end_ns(self) -> int:
        return 10_000

    def supports_sampling_policy(self, policy: object) -> bool:
        return isinstance(policy, NearestTimestampPolicy)

    def sample(self, spec: SamplingSpec, *, policy: object) -> Generator[_FakeSensorData]:
        del spec, policy
        self.sample_started = True
        # Stands in for the `with decoder_cm as decoder:` a real sensor suspends inside.
        try:
            for index, item in enumerate(self._script):
                self.advances += 1
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, _FakeSensorData)
                # Real sensors build a payload per window and keep no reference to
                # it. Drop ours too, so retention tests measure the group and not
                # this double's script.
                self._script[index] = None
                yield item
        finally:
            self.released = True

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        del batch_size
        raise NotImplementedError(STREAM_TIMESTAMPS_CAMERA_ONLY_MSG)


def _collect_until_failure(frames: Iterator[AlignedFrame]) -> tuple[list[AlignedFrame], AlignmentError]:
    """Drain *frames* and return the frames it produced plus the failure that stopped it."""
    collected: list[AlignedFrame] = []
    with pytest.raises(AlignmentError) as caught:  # noqa: PT012
        for frame in frames:
            collected.append(frame)  # noqa: PERF402
    return collected, caught.value


def _rows(timestamps_ns: list[int]) -> _FakeSensorData:
    """Build a well-formed batch aligned to *timestamps_ns*."""
    aligned = np.array(timestamps_ns, dtype=np.int64)
    return _FakeSensorData(align_timestamps_ns=aligned, sensor_timestamps_ns=aligned)


def _empty_rows() -> _FakeSensorData:
    """Build a zero-row batch, as a window-local reader emits across a dropout."""
    empty = np.empty(0, dtype=np.int64)
    return _FakeSensorData(align_timestamps_ns=empty, sensor_timestamps_ns=empty)


def _make_grid(timestamps_ns: npt.NDArray[np.int64], stride_ns: int, duration_ns: int) -> SamplingGrid:
    return SamplingGrid(
        start_ns=int(timestamps_ns[0]),
        exclusive_end_ns=int(timestamps_ns[-1]) + stride_ns,
        timestamps_ns=timestamps_ns,
        stride_ns=stride_ns,
        duration_ns=duration_ns,
    )


def _nearest_policies(*sensor_ids: str) -> dict[str, NearestTimestampPolicy]:
    return {sensor_id: NearestTimestampPolicy() for sensor_id in sensor_ids}


_TS = np.array([0, 1_000, 2_000, 3_000, 4_000], dtype=np.int64)
_STRIDE = 1_000


def test_single_sensor_yields_one_frame_per_window() -> None:
    """SensorGroup with one sensor yields one AlignedFrame per grid window."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _FakeSensor(_TS)})

    frames = list(group.sample(spec, policies=_nearest_policies("a")))
    windows = list(grid)

    assert len(frames) == len(windows)
    for frame, window in zip(frames, windows, strict=True):
        assert isinstance(frame, AlignedFrame)
        np.testing.assert_array_equal(frame.align_timestamps_ns, window.timestamps_ns)
        assert "a" in frame.sensor_data


def test_multi_sensor_all_present_when_coverage_complete() -> None:
    """All sensors appear in every frame when both cover all windows."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _FakeSensor(_TS), "b": _FakeSensor(_TS)})

    for frame in group.sample(spec, policies=_nearest_policies("a", "b")):
        assert "a" in frame.sensor_data
        assert "b" in frame.sensor_data


def test_start_ns_is_min_across_sensors() -> None:
    """start_ns is the minimum start_ns across all sensors."""
    ts_early = np.array([0, 1_000, 2_000], dtype=np.int64)
    ts_late = np.array([500, 1_500, 2_500], dtype=np.int64)
    group = SensorGroup({"early": _FakeSensor(ts_early), "late": _FakeSensor(ts_late)})
    assert group.start_ns == 0


def test_end_ns_is_max_across_sensors() -> None:
    """end_ns is the maximum end_ns across all sensors."""
    ts_short = np.array([0, 1_000, 2_000], dtype=np.int64)
    ts_long = np.array([0, 1_000, 5_000], dtype=np.int64)
    group = SensorGroup({"short": _FakeSensor(ts_short), "long": _FakeSensor(ts_long)})
    assert group.end_ns == 5_000


def test_policy_max_delta_exceeded_raises_alignment_error() -> None:
    """A nearest match beyond policy.max_delta_ns is an alignment failure, attributed to its sensor."""
    # Window [1000, 2000): eligible sensor ts=[1500], grid ts=[1000], delta=500 > tolerance=100
    sensor_ts = np.array([0, 1_500, 2_000, 3_000, 4_000], dtype=np.int64)
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _FakeSensor(sensor_ts)})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies={"a": NearestTimestampPolicy(max_delta_ns=100)}))

    # The sampler raises without a sensor id; the group supplies its configured one.
    assert caught.value.reason is AlignmentFailureReason.TOLERANCE_EXCEEDED
    assert caught.value.sensor_id == "a"
    assert caught.value.max_delta_ns == 100
    assert caught.value.delta_ns == 500


def test_sensor_group_routes_each_policy_to_matching_sensor() -> None:
    """SensorGroup passes each sensor only its matching concrete policy."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor_a = _FakeSensor(_TS)
    sensor_b = _FakeSensor(_TS)
    policy_a = NearestTimestampPolicy(max_delta_ns=10)
    policy_b = NearestTimestampPolicy()
    group = SensorGroup({"a": sensor_a, "b": sensor_b})

    next(group.sample(spec, policies={"a": policy_a, "b": policy_b}))

    assert sensor_a.policy_seen is policy_a
    assert sensor_b.policy_seen is policy_b


def test_sensor_group_rejects_missing_policy_before_sampling() -> None:
    """Missing sensor ids are rejected before any sensor iterator starts."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor_a = _FakeSensor(_TS)
    sensor_b = _FakeSensor(_TS)
    group = SensorGroup({"a": sensor_a, "b": sensor_b})

    with pytest.raises(ValueError, match="missing policy ids: \\['b'\\]"):
        list(group.sample(spec, policies={"a": NearestTimestampPolicy()}))

    assert not sensor_a.sample_started
    assert not sensor_b.sample_started


def test_sensor_group_rejects_unknown_policy_id_before_sampling() -> None:
    """Unknown sensor ids are rejected before any sensor iterator starts."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _FakeSensor(_TS)
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match="unknown policy ids: \\['unknown'\\]"):
        list(group.sample(spec, policies={"a": NearestTimestampPolicy(), "unknown": NearestTimestampPolicy()}))

    assert not sensor.sample_started


def test_sensor_group_rejects_none_policy_before_sampling() -> None:
    """Bare None policies are rejected before any sensor iterator starts."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _FakeSensor(_TS)
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match="policy for 'a' must be a concrete policy, got None"):
        list(group.sample(spec, policies={"a": None}))

    assert not sensor.sample_started


def test_sensor_group_rejects_unsupported_policy_before_sampling() -> None:
    """Unsupported concrete policy types are rejected before any sensor iterator starts."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _FakeSensor(_TS)
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match="unsupported policy type for 'a': NoSamplingPolicy"):
        list(group.sample(spec, policies={"a": NoSamplingPolicy()}))

    assert not sensor.sample_started


def test_sensor_group_rejects_sensor_without_policy_support_before_sampling() -> None:
    """Sensors without policy support declarations are rejected before sampling."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _SensorWithoutPolicySupport()
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match="unsupported policy type for 'a': NearestTimestampPolicy"):
        list(group.sample(spec, policies={"a": NearestTimestampPolicy()}))

    assert not sensor.sample_started


def test_sensor_past_its_coverage_snaps_to_its_last_timestamp() -> None:
    """A sensor whose timeline ends early keeps emitting rows, snapped to its last timestamp.

    Window bounds do not restrict which sensor timestamps may be selected, so a
    sensor is only omitted from a frame when it has no timestamps at all for that
    window. Rejecting a snap that reaches too far is ``max_delta_ns``'s job.
    """
    ts_short = np.array([0, 1_000], dtype=np.int64)
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"full": _FakeSensor(_TS), "short": _FakeSensor(ts_short)})

    frames = list(group.sample(spec, policies=_nearest_policies("full", "short")))

    # ts_short covers windows [0,1000) and [1000,2000) exactly.
    assert frames[0].sensor_data["short"].sensor_timestamps_ns.tolist() == [0]
    assert frames[1].sensor_data["short"].sensor_timestamps_ns.tolist() == [1_000]
    assert len(frames) == len(_TS)

    # Windows [2000,5000) are past ts_short's range, so every row snaps to 1000.
    for frame in frames[2:]:
        assert frame.sensor_data["short"].sensor_timestamps_ns.tolist() == [1_000]
        assert "full" in frame.sensor_data


def test_empty_batch_from_a_configured_sensor_raises_empty_batch() -> None:
    """A window-local reader that finds no messages fails the window."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _ScriptedSensor([_empty_rows()])
    group = SensorGroup({"a": sensor})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    assert caught.value.reason is AlignmentFailureReason.EMPTY_BATCH
    assert caught.value.sensor_id == "a"
    np.testing.assert_array_equal(caught.value.align_timestamps_ns, [0])
    # The other three are documented as tolerance-only; a dropout has no delta.
    assert caught.value.sensor_timestamps_ns is None
    assert caught.value.delta_ns is None
    assert caught.value.max_delta_ns is None


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        pytest.param(
            _rows([0, 1_000]),
            "align_timestamps_ns length 2",
            id="too_many_rows",
        ),
        pytest.param(
            _FakeSensorData(
                align_timestamps_ns=np.array([0], dtype=np.int64),
                sensor_timestamps_ns=np.empty(0, dtype=np.int64),
            ),
            "sensor_timestamps_ns length 0",
            id="sensor_timeline_wrong_length",
        ),
        pytest.param(
            _rows([7]),
            "must exactly match",
            id="wrong_timeline",
        ),
    ],
)
def test_a_structurally_impossible_payload_is_not_an_alignment_error(
    payload: _FakeSensorData,
    expected_message: str,
) -> None:
    """A payload no recording could produce is a sensor defect, so it keeps its own type.

    ``AlignmentError`` is a quarantine signal: it says this input could not be
    served and the pipeline should drop it. A sensor that returns the wrong row
    count or a timeline it was never asked for is broken in a way no data can
    cause, and quarantining would discard good recordings across a whole dataset
    while hiding the bug. ``AlignedFrame`` already rejects these, and that
    ``ValueError`` is the correct outcome rather than a gap to be papered over.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _ScriptedSensor([payload])
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match=expected_message) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    assert not isinstance(caught.value, AlignmentError)
    # Cleanup is not conditional on which kind of failure ended iteration.
    assert sensor.released


def test_a_sensor_that_ends_early_is_not_reported_as_an_alignment_failure() -> None:
    """Early exhaustion is left unclassified here, pending its own investigation.

    A short generator can mean the recording had nothing left, or that the sensor
    gave up silently, and those want opposite responses. Rather than guess, this
    keeps the pre-existing behaviour: the bare ``next()`` raises ``StopIteration``
    inside a generator, which PEP 479 turns into ``RuntimeError``. It is a poor
    diagnostic, but it is the one already shipped, and classifying it is tracked
    separately.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _ScriptedSensor([_rows([0])])})

    frames = []
    with pytest.raises(RuntimeError, match="generator raised StopIteration") as caught:  # noqa: PT012
        for frame in group.sample(spec, policies=_nearest_policies("a")):
            frames.append(frame)  # noqa: PERF402

    assert len(frames) == 1
    assert not isinstance(caught.value, AlignmentError)


def test_modality_alignment_error_is_stamped_with_the_configured_sensor_id() -> None:
    """A modality that raises without a sensor id gets the group's configured one."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    raised = AlignmentError(AlignmentFailureReason.TOLERANCE_EXCEEDED, "selection failed")
    group = SensorGroup({"a": _ScriptedSensor([raised])})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    assert caught.value is raised
    assert caught.value.reason is AlignmentFailureReason.TOLERANCE_EXCEEDED
    assert caught.value.sensor_id == "a"


def test_a_propagated_modality_error_identifies_the_window_that_failed() -> None:
    """Every raise path must say which window failed, including the propagation path.

    A modality knows what it was asked for but not that the request came from a
    window, so it can raise without a timeline. Without this, a caller can locate
    a group-detected failure in time but not a modality-raised one, and would
    need a parallel window counter to tell them apart.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    raised = AlignmentError(AlignmentFailureReason.TOLERANCE_EXCEEDED, "selection failed")
    # Fails on the third window, so a correct answer cannot be the first one.
    group = SensorGroup({"a": _ScriptedSensor([_rows([0]), _rows([1_000]), raised])})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    np.testing.assert_array_equal(caught.value.align_timestamps_ns, [2_000])


def test_a_modality_error_keeps_the_timeline_it_supplied_itself() -> None:
    """A modality that attached its own timeline keeps it; the group does not overwrite."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    raised = AlignmentError(
        AlignmentFailureReason.TOLERANCE_EXCEEDED,
        "selection failed",
        align_timestamps_ns=np.array([42], dtype=np.int64),
    )
    group = SensorGroup({"a": _ScriptedSensor([raised])})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    np.testing.assert_array_equal(caught.value.align_timestamps_ns, [42])


def test_a_stamped_timeline_is_memory_the_error_owns() -> None:
    """The group's stamping path must copy, exactly as construction does.

    Iterating a grid builds one timestamp array and hands each window a slice of
    it, so an error that kept the slice would pin every window's timestamps for
    as long as a caller holds the error. Construction copies through the field
    converter; this path assigns after the fact, and only copies because
    ``attrs.define`` runs converters on ``setattr`` too. ``attrs.setters.NO_OP``,
    or a move to ``attrs.frozen``, drops that silently.

    Ownership rather than mutation is what this asserts, unlike the construction
    case in ``test_attached_timelines_are_copies_the_error_owns``. Grid windows
    are read-only, so no caller can write through to a stamped timeline; what a
    kept slice costs here is retention, not a changing answer.
    """
    spec = SamplingSpec(grid=_make_grid(_TS, _STRIDE, _STRIDE))
    raised = AlignmentError(AlignmentFailureReason.TOLERANCE_EXCEEDED, "selection failed")
    # Fails on the second window, so the stamped timeline is one the group chose.
    group = SensorGroup({"a": _ScriptedSensor([_rows([0]), raised])})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    stamped = caught.value.align_timestamps_ns
    np.testing.assert_array_equal(stamped, [1_000])
    assert stamped.base is None, "stamped timeline is a view, so it pins the grid's window timestamps"


def test_a_propagated_error_is_always_attributed_to_the_configured_sensor_id() -> None:
    """``sensor_id`` is the group's key for the failing sensor, whatever the error arrived with.

    Keeping that unconditional is what lets a caller use it as a lookup —
    ``frame.sensor_data[error.sensor_id]`` — rather than as a label that is
    sometimes a key and sometimes not.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    raised = AlignmentError(
        AlignmentFailureReason.TOLERANCE_EXCEEDED,
        "selection failed",
        sensor_id="some.sub_stream",
    )
    group = SensorGroup({"a": _ScriptedSensor([raised])})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a")))

    assert caught.value.sensor_id == "a"


def test_first_configured_failure_wins_and_later_sensors_are_not_advanced() -> None:
    """Two failures in one window resolve deterministically to the first configured sensor."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    first = _ScriptedSensor([_empty_rows()])
    second = _ScriptedSensor([_rows([0, 1_000])])
    group = SensorGroup({"first": first, "second": second})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("first", "second")))

    assert caught.value.sensor_id == "first"
    assert caught.value.reason is AlignmentFailureReason.EMPTY_BATCH
    assert first.advances == 1
    assert second.advances == 0


def test_failure_on_a_middle_window_terminates_the_iterator() -> None:
    """A failed window ends the iteration; later windows produce nothing."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _ScriptedSensor([_rows([0]), _empty_rows(), _rows([2_000]), _rows([3_000])])
    group = SensorGroup({"a": sensor})

    iterator = group.sample(spec, policies=_nearest_policies("a"))
    frames, _error = _collect_until_failure(iterator)

    assert len(frames) == 1
    assert sensor.advances == 2
    with pytest.raises(StopIteration):
        next(iterator)


def test_a_failed_window_releases_every_sensor_even_when_the_caller_keeps_the_error() -> None:
    """Sensors suspend inside their decoder's `with` block, so a failed window must close them.

    Keeping the error is the documented pattern, and it keeps the traceback — and
    therefore the generators — reachable, so refcounting alone never releases
    them. Without an explicit close this leaks an open decoder per sensor per
    failed window.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    healthy = _ScriptedSensor([_rows([0]), _rows([1_000]), _rows([2_000])])
    failing = _ScriptedSensor([_rows([0]), _empty_rows()])
    group = SensorGroup({"healthy": healthy, "failing": failing})

    held: AlignmentError | None = None
    try:
        for _frame in group.sample(spec, policies=_nearest_policies("healthy", "failing")):
            pass
    except AlignmentError as error:
        held = error

    assert held is not None, "the failing sensor should have ended iteration"
    assert healthy.released, "a sibling sensor was left suspended holding its decoder"
    assert failing.released, "the failing sensor was left suspended holding its decoder"


def test_a_retained_error_does_not_pin_the_previous_window_payloads() -> None:
    """A held error keeps ``sample()``'s frame alive; it must not keep a window of decode with it.

    Payloads are the largest thing in flight — a window of frame buffers across
    every camera. If the frame holds them, each retained error costs that much
    memory until the caller drops it.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    first_window = _rows([0])
    tracked = weakref.ref(first_window)
    group = SensorGroup({"a": _ScriptedSensor([first_window, _empty_rows()])})

    held: AlignmentError | None = None
    frames = group.sample(spec, policies=_nearest_policies("a"))
    try:
        for _frame in frames:
            del _frame  # the caller's own reference is not what is under test
    except AlignmentError as error:
        held = error

    del first_window
    gc.collect()

    assert held is not None
    assert held.__traceback__ is not None, "this test is meaningless if the traceback was already dropped"
    assert tracked() is None, "the previous window's payload is still pinned by the retained traceback"


@pytest.mark.parametrize(
    ("timestamps_ns", "expected_reason"),
    [(None, AlignmentFailureReason.EMPTY_BATCH)],
    ids=["empty_batch"],
)
def test_a_retained_error_does_not_pin_the_payload_that_failed(
    timestamps_ns: list[int] | None,
    expected_reason: AlignmentFailureReason,
) -> None:
    """The rejected payload must not ride along in the raising frame's locals.

    Every reason that inspects a payload is covered, because the routine one —
    a dropout producing an empty batch — is a payload object too, and a sensor
    that does not cache its empty batch would keep a decode alive through it.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    # Built here, not in the decorator: pytest keeps parametrize arguments alive
    # for the whole session, which would pin the payload regardless of the code.
    payload = _empty_rows() if timestamps_ns is None else _rows(timestamps_ns)
    tracked = weakref.ref(payload)
    group = SensorGroup({"a": _ScriptedSensor([payload])})

    held: AlignmentError | None = None
    frames = group.sample(spec, policies=_nearest_policies("a"))
    try:
        for _frame in frames:
            del _frame
    except AlignmentError as error:
        held = error

    del payload, frames
    gc.collect()

    assert held is not None
    assert held.reason is expected_reason
    assert held.__traceback__ is not None, "this test is meaningless if the traceback was already dropped"
    assert tracked() is None, "the rejected payload is still pinned by the retained traceback"


class _SensorWithFailingTeardown:
    """Raises while unwinding, as a decoder rejecting a stream on flush does."""

    start_ns, end_ns = 0, 10_000

    def __init__(self, script: list[object]) -> None:
        self._script = script

    def supports_sampling_policy(self, policy: object) -> bool:
        return isinstance(policy, NearestTimestampPolicy)

    def sample(self, spec: SamplingSpec, *, policy: object) -> Generator[_FakeSensorData]:
        del spec, policy
        try:
            for item in self._script:
                assert isinstance(item, _FakeSensorData)
                yield item
        finally:
            msg = "decoder rejected the stream on flush"
            raise ValueError(msg)


def test_a_failing_teardown_does_not_mask_the_alignment_failure() -> None:
    """A decoder that fails to close must not turn a droppable input into a hard failure.

    The caller was told one ``except AlignmentError`` around iteration is enough.
    If a teardown error replaces the in-flight failure, that handler stops firing
    and a recording that should have been quarantined kills the stage instead.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    # "a" fails the second window; "b" raises while being torn down; "c" must
    # still be closed despite "b" raising first.
    good_then_empty = _ScriptedSensor([_rows([0]), _empty_rows()])
    bad_teardown = _SensorWithFailingTeardown([_rows([0]), _rows([1_000])])
    trailing = _ScriptedSensor([_rows([0]), _rows([1_000])])
    group = SensorGroup({"a": good_then_empty, "b": bad_teardown, "c": trailing})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies=_nearest_policies("a", "b", "c")))

    assert caught.value.reason is AlignmentFailureReason.EMPTY_BATCH
    assert trailing.released, "a sensor after the failing teardown was left unclosed"


def test_a_failing_teardown_surfaces_when_nothing_else_went_wrong() -> None:
    """On a clean run there is no failure to protect, so the teardown error is the news."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _SensorWithFailingTeardown([_rows([ts]) for ts in _TS.tolist()])
    group = SensorGroup({"a": sensor})

    with pytest.raises(ValueError, match="decoder rejected the stream on flush"):
        list(group.sample(spec, policies=_nearest_policies("a")))


def test_a_failing_teardown_surfaces_even_when_the_caller_is_handling_something_else() -> None:
    """Whether a teardown error is reported must depend on this iteration, not on the caller.

    The documented quarantine idiom puts work inside ``except AlignmentError``, so
    draining a group from within an exception handler is an ordinary shape. Any
    check of ambient interpreter state would make the same call behave differently
    depending on what the caller happened to be doing.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _SensorWithFailingTeardown([_rows([ts]) for ts in _TS.tolist()])
    group = SensorGroup({"a": sensor})

    earlier_failure = KeyError("an earlier, already-handled failure")
    try:
        raise earlier_failure
    except KeyError:
        with pytest.raises(ValueError, match="decoder rejected the stream on flush"):
            list(group.sample(spec, policies=_nearest_policies("a")))


def test_abandoning_iteration_early_releases_every_sensor() -> None:
    """A caller that stops after one frame must not strand the remaining generators.

    Unlike the retained-error case above, this path does not depend on the
    explicit close: no traceback is kept, so the ``sample()`` frame dies and
    refcounting closes the sensor generators on its own. Verified by mutation —
    this test still passes with the close removed. It guards the abandonment
    guarantee itself, not the fix.
    """
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    sensor = _ScriptedSensor([_rows([0]), _rows([1_000]), _rows([2_000])])
    group = SensorGroup({"a": sensor})

    frames = group.sample(spec, policies=_nearest_policies("a"))
    next(frames)
    frames.close()

    assert sensor.released


def test_unexpected_exception_type_is_preserved() -> None:
    """Decode and programming errors keep their own type instead of becoming alignment failures."""
    grid = _make_grid(_TS, _STRIDE, _STRIDE)
    spec = SamplingSpec(grid=grid)
    group = SensorGroup({"a": _ScriptedSensor([RuntimeError("decoder blew up")])})

    with pytest.raises(RuntimeError, match="decoder blew up"):
        list(group.sample(spec, policies=_nearest_policies("a")))


def test_window_with_no_reference_timestamps_yields_an_empty_frame() -> None:
    """An empty window is a real half-open window with nothing to sample, not a failure."""
    # stride 1000 over timestamps [0, 3000] leaves windows [1000,2000) and [2000,3000) empty.
    timestamps = np.array([0, 3_000], dtype=np.int64)
    grid = SamplingGrid(
        start_ns=0,
        exclusive_end_ns=4_000,
        timestamps_ns=timestamps,
        stride_ns=1_000,
        duration_ns=1_000,
    )
    spec = SamplingSpec(grid=grid)
    empty_windows = [index for index, window in enumerate(grid) if len(window) == 0]
    assert empty_windows, "grid must contain at least one empty window for this test to mean anything"
    group = SensorGroup({"a": _FakeSensor(timestamps)})

    frames = list(group.sample(spec, policies=_nearest_policies("a")))

    for index in empty_windows:
        assert len(frames[index].align_timestamps_ns) == 0
        assert frames[index].sensor_data["a"].align_timestamps_ns.size == 0


def test_empty_sensors_raises() -> None:
    """Constructing SensorGroup with no sensors raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        SensorGroup({})
