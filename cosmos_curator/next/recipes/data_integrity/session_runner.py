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

"""Run the data-integrity engine over the streams of a single session.

The per-stream work -- wiring the metric kernel and judging each measurement -- is
the shared engine in :mod:`cosmos_curator.core.sensors.data_integrity.engine`,
reached through :mod:`.sources`, which opens the source it is pointed at. This
module adds only the session layer on top of them: :func:`run_session` discovers a
session's streams, runs the shared engine on each, classifies open/decode failures
as a per-stream ``ERROR``, and aggregates the results into a
:class:`SessionReport`.

:func:`run_one_stream` is the per-stream unit :func:`run_session` is built from --
open one source, measure it, and turn any failure into that stream's ``ERROR`` rather
than the run's. It is public because the Ray Data pipeline distributes exactly that
unit and must not grow a second copy of the failure contract.

:func:`run_stream` is a thin convenience over
:func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics` for an
already-open sensor, packaging its output as a :class:`StreamResult`.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO

from loguru import logger

from cosmos_curator.core.sensors.data_integrity.engine import (
    run_metrics,
    validate_expected_hz,
    validate_non_negative_int,
    validate_positive_int,
)
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    IntegritySensor,
    SessionReport,
    StreamResult,
    stream_result,
)
from cosmos_curator.next.recipes.data_integrity.cli_support import cancellable_reader, raise_if_interrupted
from cosmos_curator.next.recipes.data_integrity.discovery import discover_streams
from cosmos_curator.next.recipes.data_integrity.sources import run_checks


def run_stream(
    sensor: IntegritySensor,
    *,
    source: str,
    expected_hz: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int = 0,
) -> StreamResult:
    """Run all metrics over one already-open sensor and package a :class:`StreamResult`.

    Convenience wrapper over
    :func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics`;
    performs no I/O (the sensor is already open), so it never returns an ``ERROR``
    result.

    Args:
        sensor: an opened sensor exposing the ``IntegritySensor`` surface.
        source: the stream's source path/URI, used to label the result.
        expected_hz: expected sample rate in Hz; ``None`` uses the nominal rate.
        thresholds: pass/fail policy (see ``Thresholds``).
        batch_size: window size for streaming timestamps; ``0`` = one batch.

    Raises:
        ValueError: if ``expected_hz`` or ``batch_size`` is invalid (validated by
            :func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics`).

    """
    metrics, video_info, resolved_cfg = run_metrics(
        sensor, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size
    )
    return stream_result(source, metrics, video_info, resolved_cfg)


#: Exception class names, matched anywhere in the raised exception's class hierarchy,
#: that mean "the transport hiccuped" rather than "this stream is bad".
#:
#: Named rather than imported because the types live in botocore, urllib3 and
#: http.client, and this module should not import a transport stack to describe one.
#: Matching the hierarchy (not just the concrete class) is what makes the short list
#: sufficient: ``ConnectionResetError`` arrives via ``ConnectionError``, and a socket
#: stall via ``TimeoutError``.
#:
#: Bare ``OSError`` is deliberately absent even though Ray's own list carries it (see
#: ``next/core/ray_runtime.py``): Ray matches a formatted string and so cannot see the
#: hierarchy, while here it would drag in every deterministic filesystem failure --
#: ``FileNotFoundError``, ``PermissionError`` -- and spend the whole retry budget
#: re-confirming that a path is still missing. ``botocore.exceptions.ClientError`` is
#: absent for the same reason: it is the base of every AWS HTTP response, 4xx included.
_TRANSIENT_TRANSPORT_ERRORS = frozenset(
    {
        "ConnectionError",
        "TimeoutError",
        "EndpointConnectionError",
        "ReadTimeoutError",
        "ConnectionClosedError",
        "IncompleteRead",
    }
)
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_MAX_WAIT_S = 8.0


def _is_transient_transport_error(exc: BaseException) -> bool:
    """Report whether ``exc`` names a transport failure worth another attempt."""
    return any(cls.__name__ in _TRANSIENT_TRANSPORT_ERRORS for cls in type(exc).__mro__)


def run_one_stream(  # noqa: PLR0913
    source: str,
    *,
    expected_hz: float | None,
    thresholds: Thresholds,
    batch_size: int,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
    cancel: threading.Event | None = None,
    max_attempts: int = 1,
) -> StreamResult:
    """Open a single stream and run the integrity metrics, capturing failures as ERROR.

    Args:
        source: the stream's path or URI.
        expected_hz: expected sample rate; ``None`` uses the stream's nominal rate.
        thresholds: pass/fail policy.
        batch_size: window size for streaming timestamps; ``0`` = one batch.
        s3_profile_name: AWS profile for ``s3://`` sources.
        azure_profile_name: Azure profile for ``az://`` sources.
        endpoint_url: S3 endpoint override for S3-compatible stores.
        stream_wrapper: optional wrapper around the opened byte stream.
        cancel: optional event that aborts the read.
        max_attempts: how many times to open the stream when the failure looks like a
            transport hiccup rather than a bad stream. ``1`` (default) never retries,
            which is what a single session wants: the caller is watching, and the
            failure is one line of its report. A run over thousands of cloud streams
            wants more, because there a read timeout would otherwise be persisted as
            an unreadable input -- a data-quality finding manufactured by the network.
            Only the errors in :data:`_TRANSIENT_TRANSPORT_ERRORS` are retried; a
            malformed file fails on its first attempt as it always did.

    Returns:
        The stream's result, carrying ``error`` when it could not be measured.

    Raises:
        KeyboardInterrupt: If *cancel* was set, rather than reporting the aborted read
            as this stream's verdict.
        ValueError: If ``max_attempts`` is not positive.

    """
    max_attempts = validate_positive_int("max_attempts", max_attempts)
    attempt = 1
    while True:
        try:
            metrics, video_info, resolved_cfg = run_checks(
                source,
                expected_hz=expected_hz,
                thresholds=thresholds,
                batch_size=batch_size,
                s3_profile_name=s3_profile_name,
                azure_profile_name=azure_profile_name,
                endpoint_url=endpoint_url,
                stream_wrapper=stream_wrapper,
            )
        # A session is a batch: one unreadable or malformed stream must not abandon the
        # ones behind it, and the set of exceptions PyAV, botocore, and smart_open can
        # raise is too broad to enumerate safely. So the catch stays wide and the stream
        # is reported as ERROR. The traceback is logged at DEBUG so a genuine bug in the
        # engine is still diagnosable rather than flattened into a one-line message.
        except Exception as exc:  # noqa: BLE001 - see above; per-stream isolation is the contract
            # An abort is a casualty, not a diagnosis: cancelling makes the reader report EOF,
            # which libav raises as a decode error, so every stream still in flight would log
            # an annotated traceback and bury the interrupt message that follows. Ahead of the
            # retry, too -- an interrupt must never buy the aborted read another attempt.
            raise_if_interrupted(cancel)
            if attempt < max_attempts and _is_transient_transport_error(exc):
                wait_s = min(_RETRY_BACKOFF_FACTOR**attempt, _RETRY_MAX_WAIT_S)
                logger.warning(
                    "data-integrity attempt {}/{} for {} hit a transport error ({}); retrying in {}s",
                    attempt,
                    max_attempts,
                    source,
                    exc,
                    wait_s,
                )
                attempt += 1
                # Waiting on the event rather than sleeping keeps the backoff from holding
                # an interrupt for the whole wait, once per stream still in flight.
                if cancel is None:
                    time.sleep(wait_s)
                else:
                    cancel.wait(wait_s)
                raise_if_interrupted(cancel)
                continue
            logger.opt(exception=True).debug("data-integrity run failed for {}", source)
            return StreamResult(
                source=source,
                codec_name=None,
                has_bframes=None,
                num_samples=None,
                start_ns=None,
                end_ns=None,
                metrics=[],
                error=str(exc),
            )
        return stream_result(source, metrics, video_info, resolved_cfg)


def run_session(  # noqa: PLR0913
    session_path: str,
    *,
    expected_hz: float | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int = 0,
    limit: int = 0,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    max_workers: int = 1,
    cancel: threading.Event | None = None,
    on_stream_start: Callable[[int, int, str], None] | None = None,
    on_stream_finish: Callable[[int, int, StreamResult], None] | None = None,
    make_stream_wrapper: Callable[[int, int, str], Callable[[BinaryIO], BinaryIO]] | None = None,
) -> SessionReport:
    """Discover and run integrity metrics over every stream in one session.

    Args:
        session_path: local directory, local file, or ``s3://`` / ``az://`` prefix
            for a single session (typically one ``clips/<uuid>/``).
        expected_hz: expected sample rate; ``None`` uses each stream's nominal rate.
        thresholds: pass/fail policy applied to every stream.
        batch_size: window size for streaming timestamps; ``0`` = one batch.
        limit: cap on the number of streams (``0`` = all), for sampling large sessions.
        s3_profile_name: optional AWS profile for ``s3://`` sources.
        azure_profile_name: Azure profile for ``az://`` sources.
        endpoint_url: optional S3 endpoint override for S3-compatible stores.
        max_workers: how many streams to check concurrently. ``1`` (default) keeps
            everything on the calling thread. Higher values overlap the per-stream
            waits, which dominate on cloud sources. Hooks are then called from
            worker threads and must be thread-safe; results stay in discovery order
            either way.
        cancel: optional event that asks the session to stop. Cloud reads abort at
            their next boundary once it is set, and no further stream is opened; the
            run then ends by raising ``KeyboardInterrupt``, since a session stopped
            part-way has no verdict to report. CLIs pass the event that their SIGINT
            handler sets (see
            :func:`~cosmos_curator.next.recipes.data_integrity.cli_support.interrupt_guard`).
        on_stream_start: optional progress hook called *before* each stream is
            opened, with ``(index, total, source)`` (``index`` is 1-based). This is
            the point at which a slow open/decode begins; callers use it to show
            live progress. With ``max_workers`` above 1 several streams are in
            flight at once, so these interleave with ``on_stream_finish``.
        on_stream_finish: optional progress hook called *after* each stream, with
            ``(index, total, result)``.
        make_stream_wrapper: optional factory called with ``(index, total, source)``
            that returns a stream wrapper for that stream (e.g. a byte-counting
            reader for download progress), for local paths as well as cloud URIs.

    Returns:
        A :class:`SessionReport` with one :class:`StreamResult` per discovered stream.

    Raises:
        ValueError: if ``expected_hz``, ``batch_size``, ``limit``, or ``max_workers``
            is invalid.
        KeyboardInterrupt: on cancellation, so the caller owns the exit status rather
            than receiving a report for a session that never finished.

    """
    # Fail fast at the public boundary, before any discovery / I/O, so an invalid
    # config errors deterministically even for a session with zero streams.
    expected_hz = validate_expected_hz(expected_hz)
    batch_size = validate_non_negative_int("batch_size", batch_size)
    max_workers = validate_positive_int("max_workers", max_workers)
    sources = discover_streams(
        session_path,
        limit=limit,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
    )
    total = len(sources)

    def _wrapper_for(index: int, source: str) -> Callable[[BinaryIO], BinaryIO] | None:
        """Compose the caller's stream wrapper, if any, with cancellation."""
        caller_wrapper = make_stream_wrapper(index, total, source) if make_stream_wrapper is not None else None
        if cancel is None:
            return caller_wrapper
        # Cancellation goes outermost so its check runs before the caller's wrapper
        # forwards the read, and applies whether or not a caller supplied one.
        return lambda stream: cancellable_reader(
            caller_wrapper(stream) if caller_wrapper is not None else stream, cancel
        )

    def _check(numbered: tuple[int, str]) -> StreamResult:
        index, source = numbered
        # Raised from worker threads too: the exception is stored on that stream's
        # future, and ``Executor.map`` re-raises it on the caller while abandoning
        # the backlog.
        raise_if_interrupted(cancel)
        if on_stream_start is not None:
            on_stream_start(index, total, source)
        result = run_one_stream(
            source,
            expected_hz=expected_hz,
            thresholds=thresholds,
            batch_size=batch_size,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            stream_wrapper=_wrapper_for(index, source),
            cancel=cancel,
        )
        # Checked again because the stream may have *succeeded* despite the abort: libav
        # can swallow it and finish off a truncated source, and that verdict is no more
        # reportable than the aborted one, nor are the streams still queued behind it.
        raise_if_interrupted(cancel)
        if on_stream_finish is not None:
            on_stream_finish(index, total, result)
        return result

    work = list(enumerate(sources, start=1))
    if max_workers == 1:
        # Kept distinct from the pool so the default stays free of worker threads and
        # the hooks keep firing on the caller's own thread.
        streams = [_check(item) for item in work]
    else:
        # Threads, not processes: the time goes to network waits and to libav, both of
        # which release the GIL, and each stream builds its own cloud client and
        # decoder rather than sharing one. ``map`` yields in submission order, so the
        # report stays deterministic regardless of completion order.
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="di-stream") as pool:
            streams = list(pool.map(_check, work))
    return SessionReport(session_path=session_path, streams=streams)
