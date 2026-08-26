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

"""Unit tests for the session runner (run_stream + run_session)."""

import io
import pathlib
import threading
import time
from collections.abc import Callable, Iterator
from fractions import Fraction
from types import SimpleNamespace
from typing import BinaryIO

import numpy as np
import pytest
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    ExpectedHzSource,
    OverallStatus,
    StreamResult,
)
from cosmos_curator.next.recipes.data_integrity import session_runner
from cosmos_curator.next.recipes.data_integrity.session_runner import run_session, run_stream

HZ_100_PERIOD_NS = 10_000_000  # one sample every 10 ms at 100 Hz


class _FakeSensor:
    """Minimal stand-in satisfying the shared engine's IntegritySensor surface."""

    def __init__(
        self,
        timestamps: list[int],
        *,
        has_bframes: bool = False,
        codec_name: str = "h264",
        avg_frame_rate: Fraction = Fraction(100, 1),
    ) -> None:
        self._ts = np.array(timestamps, dtype=np.int64)
        self.has_bframes = has_bframes
        self.codec_name = codec_name
        self.video_metadata = SimpleNamespace(avg_frame_rate=avg_frame_rate)

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        return self._ts

    @property
    def start_ns(self) -> int:
        return int(self._ts[0]) if len(self._ts) else 0

    @property
    def end_ns(self) -> int:
        return int(self._ts[-1]) if len(self._ts) else 0

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        step = batch_size or len(self._ts) or 1
        for start in range(0, len(self._ts), step):
            yield self._ts[start : start + step]


def _perfect_cadence(n: int = 10) -> list[int]:
    return [i * HZ_100_PERIOD_NS for i in range(n)]


def _statuses(result: StreamResult) -> dict[str, CheckStatus]:
    return {m.name: m.status for m in result.metrics}


def test_run_stream_perfect_cadence_all_pass() -> None:
    """A strictly-increasing, on-cadence, B-frame-free stream passes every metric."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    assert result.num_samples == 10
    assert result.status is OverallStatus.PASS
    assert set(_statuses(result).values()) == {CheckStatus.PASS}


def test_run_stream_records_user_supplied_expected_hz() -> None:
    """An explicit rate is carried onto the result so the session report can show its origin."""
    result = run_stream(_FakeSensor(_perfect_cadence()), source="x", expected_hz=100.0)
    assert result.expected_hz == 100.0
    assert result.expected_hz_source is ExpectedHzSource.USER


def test_run_stream_records_header_expected_hz() -> None:
    """With no explicit rate, the stream's own header rate is resolved and recorded."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(100, 1))
    result = run_stream(sensor, source="x")
    assert result.expected_hz == 100.0
    assert result.expected_hz_source is ExpectedHzSource.HEADER


def test_run_stream_records_unavailable_expected_hz() -> None:
    """A stream with no usable header rate records UNAVAILABLE, explaining the SKIPs."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(0, 1))
    result = run_stream(sensor, source="x")
    assert result.expected_hz is None
    assert result.expected_hz_source is ExpectedHzSource.UNAVAILABLE
    assert _statuses(result)["rate"] is CheckStatus.SKIPPED


def test_run_stream_bframes_fail_the_stream() -> None:
    """A frame-reordering flag fails the frame-reordering metric and the stream."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=True)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    assert _statuses(result)["frame_reordering_present"] is CheckStatus.FAIL
    assert result.status is OverallStatus.FAIL


def test_run_stream_single_sample_marks_timing_metrics_skipped() -> None:
    """One timestamp cannot define any timing metric; they are SKIPPED, not evaluated."""
    sensor = _FakeSensor([0], has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    statuses = _statuses(result)
    for name in ("timestamp_ordering", "rate", "timestamp_gap", "jitter"):
        assert statuses[name] is CheckStatus.SKIPPED
    # Frame reordering only needs the flag, so it is still defined and passes.
    assert statuses["frame_reordering_present"] is CheckStatus.PASS
    # Skipped metrics do not fail the stream.
    assert result.status is OverallStatus.PASS


def test_run_stream_falls_back_to_avg_frame_rate() -> None:
    """With no explicit expected_hz, the nominal avg_frame_rate drives the cadence metrics."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(100, 1))
    result = run_stream(sensor, source="x")
    assert _statuses(result)["rate"] is CheckStatus.PASS


def test_run_stream_no_rate_leaves_cadence_metrics_skipped() -> None:
    """Without an explicit or nominal rate, cadence metrics are SKIPPED but ordering still runs."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(0, 1))
    result = run_stream(sensor, source="x")
    statuses = _statuses(result)
    assert statuses["rate"] is CheckStatus.SKIPPED
    assert statuses["timestamp_gap"] is CheckStatus.SKIPPED
    assert statuses["jitter"] is CheckStatus.SKIPPED
    assert statuses["timestamp_ordering"] is CheckStatus.PASS


def test_run_stream_batched_matches_one_shot() -> None:
    """Streaming in small batches yields the same verdict as one batch."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0, batch_size=3)
    assert result.num_samples == 10
    assert result.status is OverallStatus.PASS


def _canned(source: str, status: CheckStatus) -> StreamResult:
    return StreamResult(
        source=source,
        codec_name="h264",
        has_bframes=False,
        num_samples=10,
        start_ns=0,
        end_ns=1,
        metrics=[CheckResult("m", status, "m=ok", measurement=None, evaluation=None)],
    )


def test_run_session_aggregates_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_session preserves discovery order and rolls the per-stream verdicts up."""
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["x", "y"])
    canned = {"x": _canned("x", CheckStatus.PASS), "y": _canned("y", CheckStatus.FAIL)}
    monkeypatch.setattr(session_runner, "run_one_stream", lambda source, **_k: canned[source])

    report = run_session("sess")

    assert report.session_path == "sess"
    assert [s.source for s in report.streams] == ["x", "y"]
    assert report.status is OverallStatus.FAIL


def test_run_session_captures_open_errors(tmp_path: pathlib.Path) -> None:
    """A file that is not a decodable video becomes a per-stream ERROR, not a crash."""
    (tmp_path / "a.mp4").write_bytes(b"not a real video")
    report = run_session(str(tmp_path))
    assert len(report.streams) == 1
    assert report.streams[0].status is OverallStatus.ERROR
    assert report.streams[0].error is not None
    assert report.status is OverallStatus.ERROR


def test_run_session_empty_dir_is_error(tmp_path: pathlib.Path) -> None:
    """A session with no discoverable streams reports ERROR and no streams."""
    report = run_session(str(tmp_path))
    assert report.streams == []
    assert report.status is OverallStatus.ERROR


@pytest.mark.parametrize("bad_hz", [0.0, -1.0, float("nan"), float("inf")])
def test_run_stream_rejects_invalid_expected_hz(bad_hz: float) -> None:
    """run_stream fails fast on a non-positive / non-finite expected_hz."""
    sensor = _FakeSensor(_perfect_cadence())
    with pytest.raises(ValueError, match="expected_hz"):
        run_stream(sensor, source="x", expected_hz=bad_hz)


def test_run_stream_rejects_negative_batch_size() -> None:
    """run_stream fails fast on a negative batch_size."""
    sensor = _FakeSensor(_perfect_cadence())
    with pytest.raises(ValueError, match="batch_size"):
        run_stream(sensor, source="x", expected_hz=100.0, batch_size=-1)


def test_run_session_rejects_invalid_expected_hz_before_io(tmp_path: pathlib.Path) -> None:
    """run_session validates expected_hz at the boundary, even for an empty session (no I/O)."""
    with pytest.raises(ValueError, match="expected_hz"):
        run_session(str(tmp_path), expected_hz=-5.0)


def test_run_session_rejects_negative_batch_size_before_io(tmp_path: pathlib.Path) -> None:
    """run_session validates batch_size at the boundary, even for an empty session (no I/O)."""
    with pytest.raises(ValueError, match="batch_size"):
        run_session(str(tmp_path), batch_size=-1)


@pytest.mark.parametrize("bad_workers", [0, -1])
def test_run_session_rejects_non_positive_max_workers_before_io(tmp_path: pathlib.Path, bad_workers: int) -> None:
    """run_session validates max_workers at the boundary, even for an empty session (no I/O)."""
    with pytest.raises(ValueError, match="max_workers"):
        run_session(str(tmp_path), max_workers=bad_workers)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_run_session_keeps_discovery_order_regardless_of_completion_order(
    monkeypatch: pytest.MonkeyPatch, max_workers: int
) -> None:
    """Concurrency must not reorder the report: slow streams still land in discovery order."""
    sources = ["a", "b", "c", "d"]
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: sources)

    def _one(source: str, **_kwargs: object) -> StreamResult:
        # Reverse the completion order relative to submission.
        time.sleep(0.02 * (len(sources) - sources.index(source)))
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    report = run_session("sess", max_workers=max_workers)

    assert [s.source for s in report.streams] == sources


def test_run_session_overlaps_streams_when_given_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streams genuinely run at the same time, rather than merely reporting as if they had.

    Each stream waits on a barrier that only trips once all four are inside it, so this
    cannot pass unless four are in flight together.
    """
    workers = 4
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["a", "b", "c", "d"])
    barrier = threading.Barrier(workers, timeout=30)

    def _one(source: str, **_kwargs: object) -> StreamResult:
        barrier.wait()
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    report = run_session("sess", max_workers=workers)

    assert report.status is OverallStatus.PASS
    assert len(report.streams) == workers


def test_cancelling_aborts_in_flight_reads_instead_of_waiting_for_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled session stops at its next read boundary, not at the end of the download.

    A thread inside libav cannot be preempted, so without a check on the read path the
    pool's shutdown waits for every stream in flight -- on a wide session that is the
    whole tail of a dozen large downloads, arriving long after the operator asked to
    stop. The reader wrapper is what bounds that wait to one read.
    """
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["a", "b", "c", "d"])
    cancel = threading.Event()
    reads_after_cancel = 0

    class _Endless(io.RawIOBase):
        """A source that never runs out, standing in for a large object."""

        def read(self, _size: int | None = -1) -> bytes:
            nonlocal reads_after_cancel
            if cancel.is_set():
                reads_after_cancel += 1
            return b"\0" * 4096

        def readable(self) -> bool:
            return True

    def _one(source: str, *, stream_wrapper: object = None, **_kwargs: object) -> StreamResult:
        assert callable(stream_wrapper), "run_session must install a cancellation wrapper"
        reader = stream_wrapper(_Endless())  # type: ignore[operator]
        assert reader.read(4096) != b"", "reads must pass through until cancelled"
        cancel.set()  # stand in for the SIGINT landing mid-read
        # EOF, rather than an exception, is what stops the demuxer quietly.
        assert reader.read(4096) == b""
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    with pytest.raises(KeyboardInterrupt):
        run_session("sess", max_workers=2, cancel=cancel)

    assert reads_after_cancel == 0, "kept reading the source after the cancellation"


def test_cancelling_wraps_the_callers_own_stream_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation composes with a caller's wrapper rather than displacing it.

    The session CLI supplies a byte-counting reader for its progress display, so a
    wrapper that replaced it would silently drop the download counter.
    """
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["a"])
    counted: list[int] = []

    def _make_counter(_index: int, _total: int, _source: str) -> Callable[[BinaryIO], BinaryIO]:
        def _wrap(stream: BinaryIO) -> BinaryIO:
            counted.append(id(stream))
            return stream

        return _wrap

    def _one(source: str, *, stream_wrapper: object = None, **_kwargs: object) -> StreamResult:
        assert callable(stream_wrapper)
        stream_wrapper(io.BytesIO(b"x"))  # type: ignore[operator]
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    run_session("sess", cancel=threading.Event(), make_stream_wrapper=_make_counter)

    assert len(counted) == 1, "the caller's wrapper was not applied"


def test_a_cancelled_stream_is_not_logged_as_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """An aborted read propagates instead of being diagnosed as this stream's failure.

    Cancelling makes the reader report EOF, which libav raises as a decode error, so
    without this every stream still in flight logs an annotated traceback and buries the
    interrupt message: a real Ctrl-C on a 12-stream session produced ~450 lines of them.
    """
    cancel = threading.Event()
    cancel.set()
    logged: list[str] = []

    def _explode(*_a: object, **_k: object) -> tuple[object, object, object]:
        msg = "Invalid data found when processing input"
        raise ValueError(msg)

    monkeypatch.setattr(session_runner, "run_checks", _explode)
    monkeypatch.setattr(session_runner.logger, "opt", lambda **_k: SimpleNamespace(debug=logged.append))

    with pytest.raises(KeyboardInterrupt):
        session_runner.run_one_stream(
            "s3://b/k.mp4",
            expected_hz=None,
            thresholds=DEFAULT_THRESHOLDS,
            batch_size=0,
            s3_profile_name=None,
            azure_profile_name="default",
            endpoint_url=None,
            cancel=cancel,
        )

    assert logged == [], "diagnosed an operator abort as a failed run"


def test_interrupt_stops_starting_queued_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt propagates and the pool abandons its backlog rather than draining it.

    This is a property of ``Executor.map``, which cancels its outstanding futures as it
    unwinds; pinning it here guards the behaviour operators actually feel, since
    dispatching the same work by hand (``submit`` plus ``as_completed``) would instead
    download the entire backlog before exiting and look like a hang.
    """
    sources = ["boom", *[f"s{i}" for i in range(199)]]
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: sources)
    started: list[str] = []
    lock = threading.Lock()

    def _one(source: str, **_kwargs: object) -> StreamResult:
        with lock:
            started.append(source)
        if source == "boom":
            raise KeyboardInterrupt
        # Slow enough that burning through the backlog would take far longer than the
        # microseconds the caller needs to flag the cancellation.
        time.sleep(0.05)
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    with pytest.raises(KeyboardInterrupt):
        run_session("sess", max_workers=2)

    # Only streams already running when the interrupt landed may have started; the
    # remaining ~190 are cancelled while still queued.
    assert len(started) <= 10, f"kept starting streams after the interrupt: {started}"


def _count_attempts(
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
    *,
    max_attempts: int,
    succeed_after: int = 0,
) -> tuple[StreamResult, int]:
    """Run one stream whose first attempts raise ``exc``, and count the attempts."""
    attempts = 0

    def _flaky(*_a: object, **_k: object) -> tuple[object, object, object]:
        nonlocal attempts
        attempts += 1
        if succeed_after and attempts > succeed_after:
            return [], SimpleNamespace(), SimpleNamespace()
        raise exc

    monkeypatch.setattr(session_runner, "run_checks", _flaky)
    monkeypatch.setattr(session_runner, "stream_result", lambda source, *_a, **_k: _canned(source, CheckStatus.PASS))
    monkeypatch.setattr(session_runner.time, "sleep", lambda _s: None)

    result = session_runner.run_one_stream(
        "s3://b/k.mp4",
        expected_hz=None,
        thresholds=DEFAULT_THRESHOLDS,
        batch_size=0,
        s3_profile_name=None,
        azure_profile_name="default",
        endpoint_url=None,
        max_attempts=max_attempts,
    )
    return result, attempts


def test_a_single_session_does_not_retry_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``di-session`` behaviour is unchanged: one attempt, then the errored result."""
    result, attempts = _count_attempts(monkeypatch, TimeoutError("read timed out"), max_attempts=1)

    assert attempts == 1
    assert result.error == "read timed out"


def test_a_transport_hiccup_is_retried_until_it_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over thousands of cloud streams, a read timeout is not a data-quality finding."""
    result, attempts = _count_attempts(
        monkeypatch,
        TimeoutError("read timed out"),
        max_attempts=3,
        succeed_after=2,
    )

    assert attempts == 3
    assert result.error is None


def test_a_bad_stream_fails_on_its_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed file is not a transport problem, and re-reading it wastes the budget."""
    result, attempts = _count_attempts(
        monkeypatch,
        ValueError("Invalid data found when processing input"),
        max_attempts=3,
    )

    assert attempts == 1
    assert result.error == "Invalid data found when processing input"


def test_a_missing_file_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OSError`` is deliberately not on the transient list; a missing path stays missing."""
    _, attempts = _count_attempts(monkeypatch, FileNotFoundError("no such object"), max_attempts=3)

    assert attempts == 1


def test_an_exhausted_retry_budget_still_returns_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last attempt's failure is recorded, so the stream is never silently dropped."""
    result, attempts = _count_attempts(monkeypatch, TimeoutError("read timed out"), max_attempts=3)

    assert attempts == 3
    assert result.error == "read timed out"


def test_an_interrupt_during_the_backoff_is_not_held_for_the_whole_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel landing mid-backoff ends the stream at once, and buys it no further attempt.

    The wait is on the cancel event rather than the clock, so an operator's Ctrl-C is not
    held for the remaining backoff -- which, on a wide session, would be paid once per
    stream still in flight.
    """
    cancel = threading.Event()
    attempts = 0
    # Long enough that a plain sleep would dominate this test, so passing it means the
    # wait really was interruptible.
    backoff_s = 30.0
    monkeypatch.setattr(session_runner, "_RETRY_BACKOFF_FACTOR", backoff_s)
    monkeypatch.setattr(session_runner, "_RETRY_MAX_WAIT_S", backoff_s)

    def _timeout(*_a: object, **_k: object) -> tuple[object, object, object]:
        nonlocal attempts
        attempts += 1
        threading.Timer(0.05, cancel.set).start()  # stands in for the SIGINT landing
        msg = "read timed out"
        raise TimeoutError(msg)

    monkeypatch.setattr(session_runner, "run_checks", _timeout)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        session_runner.run_one_stream(
            "s3://b/k.mp4",
            expected_hz=None,
            thresholds=DEFAULT_THRESHOLDS,
            batch_size=0,
            s3_profile_name=None,
            azure_profile_name="default",
            endpoint_url=None,
            cancel=cancel,
            max_attempts=2,
        )
    elapsed = time.monotonic() - started

    assert attempts == 1, "an interrupt bought the aborted read another attempt"
    assert elapsed < backoff_s / 2, f"waited out the backoff before noticing the interrupt ({elapsed:.1f}s)"


def test_a_nonpositive_attempt_budget_is_rejected() -> None:
    """Zero attempts would return nothing at all, so it cannot mean "no retries"."""
    with pytest.raises(ValueError, match="max_attempts"):
        session_runner.run_one_stream(
            "s3://b/k.mp4",
            expected_hz=None,
            thresholds=DEFAULT_THRESHOLDS,
            batch_size=0,
            s3_profile_name=None,
            azure_profile_name="default",
            endpoint_url=None,
            max_attempts=0,
        )


def test_run_session_serial_does_not_use_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default keeps the work on the caller's thread, so hooks stay where callers expect."""
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["a"])
    threads: list[str] = []

    def _one(source: str, **_kwargs: object) -> StreamResult:
        threads.append(threading.current_thread().name)
        return _canned(source, CheckStatus.PASS)

    monkeypatch.setattr(session_runner, "run_one_stream", _one)

    run_session("sess")

    assert threads == [threading.current_thread().name]
