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

"""Tests for the per-process OTel tracing hook (``tracing_hook.py``).

Exercises the full tracing lifecycle: env-var config parsing, real
``TracerProvider`` setup, ``.jsonl`` span file creation on disk,
the resilient OTLP exporter wrapper, and the module-level convenience
functions.

::

    What we test                          How we test it
    +----------------------------------+  +--------------------------------------+
    | TracingConfig.from_env()         |  | monkeypatch env vars, assert fields  |
    | _TracingBackend file creation    |  | Real backend in tmp_path -> .jsonl   |
    | setup_provider + flush -> spans  |  | Real provider, create span, flush    |
    |                                  |  |   -> assert span name in .jsonl file |
    | flush() idempotency              |  | Call twice, assert _flushed=True     |
    | setup_tracing() re-entrancy      |  | Call twice -> exactly one .jsonl     |
    | flush_tracing / propagate no-op  |  | _current_backend=None, no raise      |
    | _is_connection_error chain walk  |  | Direct/wrapped/OSError/unrelated exc |
    | _ResilientOtlpExporter           |  | Mock delegate: success, conn error,  |
    |   suppress + disable + re-raise  |  |   wrapped error, non-conn re-raise   |
    +----------------------------------+  +--------------------------------------+

Test setup:
    Two autouse fixtures isolate each test:

    1. ``_reset_tracing_singleton`` -- saves and restores the
       module-level ``_current_backend`` singleton so ``setup_tracing()``
       tests don't pollute each other.
    2. ``_reset_tracer_provider`` -- resets OTel's set-once guard and
       installs a fresh no-op ``TracerProvider`` before and after each
       test, preventing the recursion bug in OTel's default proxy.

    Trace directories use pytest's ``tmp_path``.  OTLP endpoints are
    disabled (empty string) to avoid network calls.

    ``_ResilientOtlpExporter`` tests use a ``MagicMock`` delegate
    exporter to simulate connection failures without requiring a
    real OTLP collector.

    No Ray cluster, no network, no GPU required.
"""

import errno
import pathlib
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult

import cosmos_curate.core.utils.infra.tracing_hook as _hook_module  # module-level singleton access
from cosmos_curate.core.utils.infra.tracing_hook import (
    TracingConfig,
    _is_connection_error,
    _ResilientOtlpExporter,
    _TracingBackend,
    flush_tracing,
    propagate_trace_context,
    setup_tracing,
)


def _reset_otel_provider_guard() -> None:
    """Reset OTel's set-once guard so tests can swap providers freely."""
    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE._done = False


@pytest.fixture(autouse=True)
def _reset_tracing_singleton() -> Generator[None, None, None]:
    """Reset the module-level _current_backend before each test.

    This prevents cross-test pollution from setup_tracing() which sets
    a module-level singleton.  After the test, we also reset it.
    """
    original = _hook_module._current_backend
    _hook_module._current_backend = None
    yield
    _hook_module._current_backend = original


@pytest.fixture(autouse=True)
def _reset_tracer_provider() -> Generator[None, None, None]:
    """Reset the global OTel TracerProvider before and after each test.

    Ensures each test starts with a fresh SDK provider (avoiding the
    recursion that occurs when the default proxy provider delegates
    to itself after the set-once guard is reset).
    """
    _reset_otel_provider_guard()
    trace.set_tracer_provider(TracerProvider())
    yield
    _reset_otel_provider_guard()
    trace.set_tracer_provider(TracerProvider())


class TestTracingConfig:
    """Verify TracingConfig reads environment variables correctly."""

    def test_defaults_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a clean env, defaults are used for all fields."""
        # Clear all relevant env vars to ensure clean state.
        for var in (
            "COSMOS_CURATE_TRACE_DIR",
            "COSMOS_CURATE_ARTIFACTS_STAGING_DIR",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_SERVICE_NAME",
            "COSMOS_CURATE_TRACEPARENT",
        ):
            monkeypatch.delenv(var, raising=False)

        config = TracingConfig.from_env()
        assert config.trace_dir == "/tmp/cosmos_curate_traces"  # noqa: S108
        assert config.service_name == "cosmos_curate"
        assert config.otlp_endpoint == "http://localhost:4318"

    def test_trace_dir_derived_from_staging_env(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When COSMOS_CURATE_ARTIFACTS_STAGING_DIR is set, trace_dir = <staging>/traces."""
        monkeypatch.setenv("COSMOS_CURATE_ARTIFACTS_STAGING_DIR", str(tmp_path))
        monkeypatch.delenv("COSMOS_CURATE_TRACE_DIR", raising=False)

        config = TracingConfig.from_env()
        assert config.trace_dir == str(tmp_path / "traces")

    def test_service_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OTEL_SERVICE_NAME overrides the default service name."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my_custom_svc")
        config = TracingConfig.from_env()
        assert config.service_name == "my_custom_svc"


class TestTracingBackendFileLifecycle:
    """Verify _TracingBackend creates trace files and captures spans."""

    def test_creates_trace_file_on_init(self, tmp_path: pathlib.Path) -> None:
        """Backend __init__ creates a .jsonl file on disk."""
        config = TracingConfig(trace_dir=str(tmp_path))
        backend = _TracingBackend(config)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1, f"Expected 1 .jsonl file, got {jsonl_files}"

        # Clean up the file handle.
        backend._file_handle.close()

    def test_setup_and_flush_produces_span_data(self, tmp_path: pathlib.Path) -> None:
        """After setup_provider + span creation + flush, the .jsonl file contains span data."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",  # Disable OTLP to avoid network calls.
        )
        backend = _TracingBackend(config)

        # Reset the set-once guard immediately before setup_provider()
        # so the backend can install its TracerProvider as the global.
        _reset_otel_provider_guard()
        backend.setup_provider()

        # Create a real span using the global tracer.
        tracer = trace.get_tracer("cosmos_curate")
        span = tracer.start_span("test.hook.span")
        span.end()

        backend.flush()

        # The .jsonl file should contain the span name.
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "test.hook.span" in content, f"Span name not found in trace file. Content: {content[:500]}"

    def test_flush_is_idempotent(self, tmp_path: pathlib.Path) -> None:
        """Calling flush() twice does not raise; _flushed is True after the first."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)

        # First flush.
        backend.flush()
        assert backend._flushed is True

        # Second flush (no-op).
        backend.flush()
        assert backend._flushed is True


class TestSetupTracingReentrancy:
    """Verify setup_tracing() re-entrancy guard prevents double init."""

    def test_second_call_is_noop(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling setup_tracing() twice creates exactly one .jsonl file."""
        monkeypatch.setenv("COSMOS_CURATE_TRACE_DIR", str(tmp_path))
        # Disable OTLP to avoid network calls.
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

        setup_tracing()
        setup_tracing()  # Should be a no-op.

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1, f"Expected exactly 1 .jsonl file, got {len(jsonl_files)}"


class TestSetupTracingProviderFailure:
    """Verify setup_tracing() closes the file handle when setup_provider() raises.

    If ``setup_provider()`` raises, the backend is never installed as
    the module singleton and the ``atexit`` handler is never registered.
    Without explicit cleanup in the except block, the file handle
    opened in ``__init__()`` would leak until garbage collection.
    """

    def test_file_handle_closed_on_setup_provider_failure(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If setup_provider() raises, the .jsonl file handle must be closed."""
        monkeypatch.setenv("COSMOS_CURATE_TRACE_DIR", str(tmp_path))
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

        # Capture the backend instance created inside setup_tracing()
        # so we can directly inspect its file handle after the failure.
        captured: list[_TracingBackend] = []

        def failing_setup(self: _TracingBackend) -> None:
            captured.append(self)
            msg = "simulated OTel failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(_TracingBackend, "setup_provider", failing_setup)

        with pytest.raises(RuntimeError, match="simulated OTel failure"):
            setup_tracing()

        # The singleton must NOT be installed.
        assert _hook_module._current_backend is None

        # The backend instance must have been created and captured.
        assert len(captured) == 1
        backend = captured[0]

        # The file handle must be closed by the try/except in setup_tracing().
        assert backend._file_handle.closed, "File handle should be closed after setup_provider() failure"


class TestFlushAndPropagateNoBackend:
    """Verify flush_tracing and propagate_trace_context are safe when no backend."""

    def test_flush_tracing_noop_when_no_backend(self) -> None:
        """flush_tracing() must not raise when _current_backend is None."""
        _hook_module._current_backend = None
        flush_tracing()  # Should not raise.

    def test_propagate_noop_when_no_backend(self) -> None:
        """propagate_trace_context() must not raise when _current_backend is None."""
        _hook_module._current_backend = None
        propagate_trace_context()  # Should not raise.


class TestIsConnectionError:
    """Verify _is_connection_error walks the exception chain correctly."""

    def test_direct_connection_error(self) -> None:
        """A bare ConnectionError is detected."""
        assert _is_connection_error(ConnectionError("refused")) is True

    def test_direct_connection_refused_error(self) -> None:
        """A bare ConnectionRefusedError is detected."""
        assert _is_connection_error(ConnectionRefusedError()) is True

    def test_wrapped_connection_error_in_cause(self) -> None:
        """ConnectionError nested in __cause__ is detected.

        This matches the real-world pattern where requests.ConnectionError
        wraps urllib3.MaxRetryError which wraps a ConnectionRefusedError.
        """
        inner = ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
        middle = OSError("MaxRetryError")
        middle.__cause__ = inner
        outer = RuntimeError("export failed")
        outer.__cause__ = middle
        assert _is_connection_error(outer) is True

    def test_os_error_with_econnrefused_errno(self) -> None:
        """OSError with errno=ECONNREFUSED is detected."""
        exc = OSError(errno.ECONNREFUSED, "Connection refused")
        assert _is_connection_error(exc) is True

    def test_unrelated_error_returns_false(self) -> None:
        """A ValueError with no connection error in the chain returns False."""
        assert _is_connection_error(ValueError("bad value")) is False

    def test_runtime_error_with_unrelated_cause(self) -> None:
        """A chain of non-connection errors returns False."""
        inner = TypeError("wrong type")
        outer = RuntimeError("export failed")
        outer.__cause__ = inner
        assert _is_connection_error(outer) is False


class TestResilientOtlpExporter:
    """Verify _ResilientOtlpExporter suppresses connection errors gracefully."""

    @staticmethod
    def _make_mock_exporter(*, side_effect: object = None) -> MagicMock:
        """Create a mock SpanExporter with optional side_effect on export()."""
        mock = MagicMock()
        mock.export.return_value = SpanExportResult.SUCCESS
        if side_effect is not None:
            mock.export.side_effect = side_effect
        return mock

    @staticmethod
    def _dummy_spans() -> Sequence[ReadableSpan]:
        """Return an empty span sequence for test calls."""
        return ()

    def test_delegates_to_real_exporter_on_success(self) -> None:
        """When the delegate succeeds, its result is returned."""
        delegate = self._make_mock_exporter()
        wrapper = _ResilientOtlpExporter(delegate)

        result = wrapper.export(self._dummy_spans())

        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_called_once()

    def test_suppresses_connection_error_and_disables(self) -> None:
        """On ConnectionError, the wrapper logs once and disables itself."""
        delegate = self._make_mock_exporter(
            side_effect=ConnectionError("Connection refused"),
        )
        wrapper = _ResilientOtlpExporter(delegate)

        # First call: ConnectionError is caught and suppressed.
        result = wrapper.export(self._dummy_spans())
        assert result == SpanExportResult.SUCCESS
        assert wrapper._disabled is True

        # Second call: delegate is never called (disabled).
        delegate.export.reset_mock()
        result = wrapper.export(self._dummy_spans())
        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_not_called()

    def test_suppresses_wrapped_connection_error(self) -> None:
        """A ConnectionRefusedError wrapped in RuntimeError is suppressed."""
        inner = ConnectionRefusedError(111, "Connection refused")
        outer = RuntimeError("OTLP export failed")
        outer.__cause__ = inner

        delegate = self._make_mock_exporter(side_effect=outer)
        wrapper = _ResilientOtlpExporter(delegate)

        result = wrapper.export(self._dummy_spans())
        assert result == SpanExportResult.SUCCESS
        assert wrapper._disabled is True

    def test_reraises_non_connection_error(self) -> None:
        """Non-connection errors (e.g. ValueError) are re-raised."""
        delegate = self._make_mock_exporter(
            side_effect=ValueError("bad data"),
        )
        wrapper = _ResilientOtlpExporter(delegate)

        with pytest.raises(ValueError, match="bad data"):
            wrapper.export(self._dummy_spans())

        # The wrapper should NOT be disabled for non-connection errors.
        assert wrapper._disabled is False

    def test_shutdown_delegates(self) -> None:
        """shutdown() is forwarded to the delegate."""
        delegate = self._make_mock_exporter()
        wrapper = _ResilientOtlpExporter(delegate)

        wrapper.shutdown()
        delegate.shutdown.assert_called_once()

    def test_force_flush_skipped_when_disabled(self) -> None:
        """force_flush() returns True without calling delegate when disabled."""
        delegate = self._make_mock_exporter()
        wrapper = _ResilientOtlpExporter(delegate)
        wrapper._disabled = True

        assert wrapper.force_flush() is True
        delegate.force_flush.assert_not_called()


class TestFlushInsideTracedSpanCausesExportError:
    """Reproduce flush_tracing() inside traced_span closing the file.

    This test exercises the **underlying mechanism** of the bug
    reported in the error trace:

    ::

        traced_span("Stage.setup")     <-- span created
          |
          except BaseException:
          |   flush_tracing()           <-- closes .jsonl file handle
          |   raise
          |
          exit traced_span             <-- OTel ends span, on_end() fires
               ConsoleSpanExporter.export()
                 self.out.write(...)    <-- writes to closed file
                 --> ValueError: I/O operation on closed file

    The fix defers flush_tracing() until after the traced_span exits,
    so the ConsoleSpanExporter can write the span before the file
    is closed.

    The test validates both the broken sequence (simulated inline
    flush) and the correct sequence (deferred flush) to prove the
    fix is necessary and effective.
    """

    @staticmethod
    def _simulate_inline_flush(traced_span_fn: object, flush_fn: object) -> None:
        """Simulate the buggy pattern: flush inside traced_span on error."""
        with traced_span_fn("Stage.setup_INLINE_FLUSH"):  # type: ignore[operator]
            flush_fn()  # type: ignore[operator]
            msg = "simulated setup failure"
            raise RuntimeError(msg)

    def test_inline_flush_inside_traced_span_raises_on_export(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Inline flush closes the file before OTel exports the ending span.

        This test demonstrates the root cause of the bug: the
        flush -> close -> span.end() -> export -> write-to-closed-file
        sequence.
        """
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        _hook_module._current_backend = backend

        # Simulate the buggy pattern: flush inside traced_span on error.
        # OTel's SimpleSpanProcessor fires on_end() synchronously when
        # the traced_span context exits.  After flush_tracing() closes
        # the file, ConsoleSpanExporter.export() writes to a closed
        # file, producing "Exception while exporting Span" with a
        # ValueError.  OTel catches this internally and prints it to
        # stderr rather than raising, so we detect the failure by
        # checking that the span data is NOT in the file (because the
        # export was aborted).
        from cosmos_curate.core.utils.infra.tracing import traced_span  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="simulated setup failure"):
            self._simulate_inline_flush(traced_span, flush_tracing)

        # The span file should be empty or missing the span name because
        # the ConsoleSpanExporter could not write to the closed file.
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "Stage.setup_INLINE_FLUSH" not in content, (
            "Span should NOT appear in the file -- the export should have "
            "failed because flush_tracing() closed the file handle before "
            "the traced_span exited."
        )

    @staticmethod
    def _simulate_deferred_flush(traced_span_fn: object, flush_fn: object) -> None:
        """Simulate the correct pattern: flush after traced_span exits."""
        try:
            with traced_span_fn("Stage.setup_DEFERRED_FLUSH"):  # type: ignore[operator]
                msg = "simulated setup failure"
                raise RuntimeError(msg)
        finally:
            flush_fn()  # type: ignore[operator]

    def test_deferred_flush_after_traced_span_preserves_span_data(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Deferred flush lets OTel export the span before closing the file.

        This validates the fix: deferring flush to after span export
        allows ConsoleSpanExporter to write the span data to the
        still-open file, then flush_tracing() closes and persists it.
        """
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        _hook_module._current_backend = backend

        from cosmos_curate.core.utils.infra.tracing import traced_span  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="simulated setup failure"):
            self._simulate_deferred_flush(traced_span, flush_tracing)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "Stage.setup_DEFERRED_FLUSH" in content, (
            "Span SHOULD appear in the file -- flush_tracing() ran after "
            "the traced_span exited, so ConsoleSpanExporter wrote to the "
            "still-open file before it was closed."
        )
