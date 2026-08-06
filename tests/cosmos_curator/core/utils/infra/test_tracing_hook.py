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
    | flush() idempotency              |  | Call twice, assert file still open   |
    | setup_tracing() re-entrancy      |  | Call twice -> exactly one .jsonl     |
    | flush_tracing / propagate no-op  |  | _current_backend=None, no raise      |
    | _is_connection_error chain walk  |  | Direct/wrapped/OSError/unrelated exc |
    | _ResilientOtlpExporter           |  | Mock delegate: success, conn error,  |
    |   suppress + disable + re-raise  |  |   wrapped error, non-conn re-raise   |
    | _reset_tracing_after_fork()      |  | Call directly, assert same backend,  |
    |   dup2 fd redirect to new file   |  |   unique filename, fd writes to new  |
    | _inject_span_context() log hook  |  | Fake span -> hex ids, JSON-native    |
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
import json
import logging
import os
import pathlib
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

import cosmos_curator.core.utils.infra.tracing_hook as _hook_module  # module-level singleton access
from cosmos_curator.core.utils.infra.tracing_hook import (
    TracingConfig,
    _inject_span_context,
    _is_connection_error,
    _ResilientOtlpExporter,
    _TracingBackend,
    attach_remote_parent,
    flush_tracing,
    propagate_trace_context,
    setup_tracing,
)


def _reset_otel_provider_guard() -> None:
    """Reset OTel's set-once guard so tests can swap providers freely."""
    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE._done = False


@pytest.fixture(autouse=True)
def _reset_tracing_singleton() -> Generator[None]:
    """Reset the module-level _current_backend before each test.

    This prevents cross-test pollution from setup_tracing() which sets
    a module-level singleton.  After the test, we also reset it.
    """
    original = _hook_module._current_backend
    _hook_module._current_backend = None
    yield
    _hook_module._current_backend = original


@pytest.fixture(autouse=True)
def _reset_tracer_provider() -> Generator[None]:
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
            "COSMOS_CURATOR_TRACE_DIR",
            "COSMOS_CURATOR_ARTIFACTS_STAGING_DIR",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_SERVICE_NAME",
            "COSMOS_CURATOR_TRACEPARENT",
        ):
            monkeypatch.delenv(var, raising=False)

        config = TracingConfig.from_env()
        assert config.trace_dir == "/tmp/cosmos_curator_traces"  # noqa: S108
        assert config.service_name == "cosmos_curator"
        assert config.otlp_endpoint == ""

    def test_trace_dir_derived_from_staging_env(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When COSMOS_CURATOR_ARTIFACTS_STAGING_DIR is set, trace_dir = <staging>/traces."""
        monkeypatch.setenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", str(tmp_path))
        monkeypatch.delenv("COSMOS_CURATOR_TRACE_DIR", raising=False)

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

    def test_resource_includes_run_attributes_when_enabled(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TracerProvider resource carries Slurm metadata when enabled."""
        monkeypatch.delenv("COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES", raising=False)
        monkeypatch.setenv("SLURM_JOB_ID", "9001")
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
            include_run_attributes=True,
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()
        resource_attrs = dict(trace.get_tracer_provider().resource.attributes)
        assert resource_attrs.get("slurm_job_id") == "9001"
        backend.shutdown()

    def test_resource_omits_run_attributes_when_disabled(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``include_run_attributes=False`` keeps Slurm keys off the resource."""
        monkeypatch.setenv("SLURM_JOB_ID", "9001")
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
            include_run_attributes=False,
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()
        resource_attrs = dict(trace.get_tracer_provider().resource.attributes)
        assert "slurm_job_id" not in resource_attrs
        backend.shutdown()

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
        tracer = trace.get_tracer("cosmos_curator")
        span = tracer.start_span("test.hook.span")
        span.end()

        backend.flush()

        # File must still be open after flush.
        assert not backend._file_handle.closed, "File must stay open after flush"

        # Shut down to close the file for reading.
        backend.shutdown()

        # The .jsonl file should contain the span name.
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "test.hook.span" in content, f"Span name not found in trace file. Content: {content[:500]}"

    def test_flush_is_idempotent(self, tmp_path: pathlib.Path) -> None:
        """Calling flush() twice does not raise; file stays open."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)

        backend.flush()
        assert not backend._file_handle.closed, "File must stay open after flush"

        backend.flush()
        assert not backend._file_handle.closed, "File must stay open after second flush"

        # Clean up.
        backend.close_file_handle()


class TestSetupTracingReentrancy:
    """Verify setup_tracing() re-entrancy guard prevents double init."""

    def test_second_call_is_noop(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling setup_tracing() twice creates exactly one .jsonl file."""
        monkeypatch.setenv("COSMOS_CURATOR_TRACE_DIR", str(tmp_path))
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
        monkeypatch.setenv("COSMOS_CURATOR_TRACE_DIR", str(tmp_path))
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


class TestEnableTracingEnv:
    """Verify tracing endpoint env-var wiring."""

    def test_enable_tracing_pins_traces_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting tracing OTLP endpoint pins both generic and traces-specific vars."""
        monkeypatch.setattr(_hook_module, "setup_tracing", lambda: None)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

        _hook_module.enable_tracing(otlp_endpoint="http://traces.example:4318")

        assert __import__("os").environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "http://traces.example:4318"
        assert __import__("os").environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "http://traces.example:4318"


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


class TestFlushKeepsFileOpen:
    """Verify flush_tracing() no longer closes the file handle.

    With the lifecycle fix, ``flush()`` only pushes buffered data to
    the OS kernel -- the file stays open.  This means:

    1. Inline flush (inside a ``traced_span``) is safe: the span
       that ends after flush can still be exported to the open file.
    2. Late-arriving spans (e.g. async operations failing during
       teardown) are written successfully.
    3. Only ``shutdown()`` closes the file, after disabling the
       provider.

    ::

        traced_span("Stage.setup")     <-- span created
          |
          except BaseException:
          |   flush_tracing()           <-- flushes buffer, file stays OPEN
          |   raise
          |
          exit traced_span             <-- OTel ends span, on_end() fires
               ConsoleSpanExporter.export()
                 self.out.write(...)    <-- writes to OPEN file -> OK
    """

    @staticmethod
    def _simulate_inline_flush(traced_span_fn: object, flush_fn: object) -> None:
        """Flush inside traced_span on error -- previously caused ValueError."""
        with traced_span_fn("Stage.setup_INLINE_FLUSH"):  # type: ignore[operator]
            flush_fn()  # type: ignore[operator]
            msg = "simulated setup failure"
            raise RuntimeError(msg)

    def test_inline_flush_inside_traced_span_exports_span(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Inline flush no longer closes the file -- span IS exported.

        Previously, flush_tracing() closed the file via _persist(), so
        the span ending after the flush would fail to export.  With
        the lifecycle fix, flush() keeps the file open and the span
        is successfully written.
        """
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        _hook_module._current_backend = backend

        from cosmos_curator.core.utils.infra.tracing import traced_span  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="simulated setup failure"):
            self._simulate_inline_flush(traced_span, flush_tracing)

        # File must still be open -- flush does not close it.
        assert not backend._file_handle.closed, "File must stay open after flush_tracing()"

        # Shut down to finalize the file.
        backend.shutdown()

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "Stage.setup_INLINE_FLUSH" in content, (
            "Span SHOULD appear in the file -- flush_tracing() no longer "
            "closes the file handle, so ConsoleSpanExporter can write the "
            "span when the traced_span exits."
        )

    @staticmethod
    def _simulate_deferred_flush(traced_span_fn: object, flush_fn: object) -> None:
        """Flush after traced_span exits -- always worked, still works."""
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
        """Deferred flush still works -- span is exported before flush runs."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        _hook_module._current_backend = backend

        from cosmos_curator.core.utils.infra.tracing import traced_span  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="simulated setup failure"):
            self._simulate_deferred_flush(traced_span, flush_tracing)

        # Shut down to finalize the file.
        backend.shutdown()

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "Stage.setup_DEFERRED_FLUSH" in content, (
            "Span SHOULD appear in the file -- flush_tracing() ran after "
            "the traced_span exited, so ConsoleSpanExporter wrote to the "
            "still-open file."
        )

    def test_late_span_after_flush_is_exported(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """A span created and ended after flush() is still written to the file.

        This is the direct regression test for the EngineDeadError bug:
        an async operation fails after flush has been called, its
        traced_span ends, and the span must still be exported because
        the file handle is open.
        """
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        # Create and end an early span, then flush.
        tracer = trace.get_tracer("cosmos_curator")
        early = tracer.start_span("early.span")
        early.end()
        backend.flush()

        # Create and end a LATE span -- simulates an async operation
        # failing after flush_tracing() has already been called.
        late = tracer.start_span("late.span.after_flush")
        late.end()

        # Shut down to close the file.
        backend.shutdown()

        content = (tmp_path / backend._filename).read_text(encoding="utf-8")
        assert "early.span" in content, "Early span should be in the file"
        assert "late.span.after_flush" in content, (
            "Late span should ALSO be in the file -- flush() keeps the "
            "file open so late-arriving spans can still be exported."
        )

    def test_shutdown_closes_file_after_provider(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """shutdown() closes the file handle after disabling the provider."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        tracer = trace.get_tracer("cosmos_curator")
        span = tracer.start_span("test.shutdown.span")
        span.end()

        assert not backend._file_handle.closed, "File must be open before shutdown"

        backend.shutdown()

        assert backend._file_handle.closed, "File must be closed after shutdown"

        content = (tmp_path / backend._filename).read_text(encoding="utf-8")
        assert "test.shutdown.span" in content, "Span should be persisted before file close"


class TestAttachRemoteParent:
    """Verify attach_remote_parent() sets the OTel context correctly.

    The function parses a W3C ``traceparent`` and attaches a remote span
    context so subsequent spans share the same ``trace_id``, inherit the
    root's sampling decision, and become children of the remote parent.
    The flagless legacy ``"trace_id_hex:span_id_hex"`` form is still
    accepted.

    ::

        attach_remote_parent("00-abc123-def456-01")
            |
            v
        trace.get_current_span().get_span_context()
            -> trace_id = 0xabc123
            -> span_id  = 0xdef456 (parent of next span)
            -> sampled  = True

        tracer.start_span("child")
            -> parent_span_id = 0xdef456  (linked to remote parent)
    """

    def test_noop_for_empty_string(self) -> None:
        """Empty traceparent is silently ignored (no crash, no context change)."""
        span_before = trace.get_current_span()
        attach_remote_parent("")
        span_after = trace.get_current_span()
        assert span_before is span_after

    def test_attaches_correct_trace_id(self) -> None:
        """After attachment, the current span context has the expected trace_id."""
        trace_id_hex = "0000000000000000000000000000abcd"
        span_id_hex = "00000000000000ef"
        attach_remote_parent(f"{trace_id_hex}:{span_id_hex}")

        current = trace.get_current_span().get_span_context()
        assert current.trace_id == int(trace_id_hex, 16)
        assert current.span_id == int(span_id_hex, 16)
        assert current.is_remote is True

    def test_child_span_inherits_trace_id(self, tmp_path: pathlib.Path) -> None:
        """A span created after attachment shares the remote trace_id as parent."""
        config = TracingConfig(
            trace_dir=str(tmp_path),
            otlp_endpoint="",
        )
        backend = _TracingBackend(config)
        _reset_otel_provider_guard()
        backend.setup_provider()

        trace_id_hex = "00000000000000000000000000001234"
        span_id_hex = "0000000000005678"
        attach_remote_parent(f"{trace_id_hex}:{span_id_hex}")

        tracer = trace.get_tracer("cosmos_curator")
        child = tracer.start_span("test.child.span")
        child_ctx = child.get_span_context()

        assert child_ctx.trace_id == int(trace_id_hex, 16), "Child must inherit the remote trace_id"

        # The parent of the child span should be the remote span_id.
        # ReadableSpan exposes the parent via .parent attribute.
        assert hasattr(child, "parent"), "Child must expose a parent attribute"
        assert child.parent is not None, "Child must have a parent"
        assert child.parent.span_id == int(span_id_hex, 16), "Parent span_id must match the remote span_id"

        child.end()
        backend.shutdown()

    def test_malformed_traceparent_does_not_crash(self) -> None:
        """A malformed string does not crash -- warning is logged via loguru."""
        # Should not raise -- the function catches the parse error
        # and logs a warning via loguru (which goes to stderr, not
        # Python's logging module).
        attach_remote_parent("not-a-valid-traceparent")

    def test_w3c_traceparent_preserves_unsampled_flag(self) -> None:
        """An unsampled root stays unsampled across the process boundary.

        ``ParentBased`` samplers defer to the parent, so losing this flag
        would force every worker span to sample regardless of
        ``--profile-tracing-sampling``.
        """
        trace_id_hex = "0000000000000000000000000000abce"
        span_id_hex = "00000000000000f0"
        attach_remote_parent(f"00-{trace_id_hex}-{span_id_hex}-00")

        current = trace.get_current_span().get_span_context()
        assert current.trace_id == int(trace_id_hex, 16)
        assert current.span_id == int(span_id_hex, 16)
        assert current.is_remote is True
        assert current.trace_flags.sampled is False

    def test_w3c_traceparent_preserves_sampled_flag(self) -> None:
        """A sampled root is propagated as sampled."""
        trace_id_hex = "0000000000000000000000000000abcf"
        span_id_hex = "00000000000000f1"
        attach_remote_parent(f"00-{trace_id_hex}-{span_id_hex}-01")

        current = trace.get_current_span().get_span_context()
        assert current.trace_id == int(trace_id_hex, 16)
        assert current.trace_flags.sampled is True

    def test_legacy_format_is_assumed_sampled(self) -> None:
        """The flagless legacy form keeps the behaviour it had when it was the only format."""
        trace_id_hex = "0000000000000000000000000000abd0"
        span_id_hex = "00000000000000f2"
        attach_remote_parent(f"{trace_id_hex}:{span_id_hex}")

        current = trace.get_current_span().get_span_context()
        assert current.trace_id == int(trace_id_hex, 16)
        assert current.trace_flags.sampled is True

    @pytest.mark.parametrize("sampled", [True, False])
    def test_round_trip_through_propagate_context(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        sampled: bool,  # noqa: FBT001
    ) -> None:
        """propagate_context() writes a traceparent that attach_remote_parent() reads back intact."""
        monkeypatch.delenv("COSMOS_CURATOR_TRACEPARENT", raising=False)
        backend = _TracingBackend(TracingConfig(trace_dir=str(tmp_path), otlp_endpoint=""))
        _reset_otel_provider_guard()
        backend.setup_provider()

        trace_id = int("0000000000000000000000000000abd1", 16)
        span_id = int("00000000000000f3", 16)
        flags = TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)
        remote = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=flags)

        token = context.attach(trace.set_span_in_context(NonRecordingSpan(remote)))
        try:
            backend.propagate_context()
        finally:
            context.detach(token)

        written = os.environ["COSMOS_CURATOR_TRACEPARENT"]
        assert written == f"00-{trace_id:032x}-{span_id:016x}-{int(flags):02x}"

        attach_remote_parent(written)
        current = trace.get_current_span().get_span_context()
        assert current.trace_id == trace_id
        assert current.span_id == span_id
        assert current.trace_flags.sampled is sampled

        backend.shutdown()


class TestInjectSpanContext:
    """Verify the ``LoggingInstrumentor`` log hook stamps correlation fields.

    The hook is what makes a structured (``PYTHON_LOG_FORMAT=json``) log line
    resolvable to a trace, so the emitted values must be zero-padded lowercase
    hex of the canonical widths and JSON-native types.
    """

    @staticmethod
    def _span(*, trace_id: int, span_id: int, sampled: bool) -> trace.Span:
        flags = trace.TraceFlags(trace.TraceFlags.SAMPLED if sampled else trace.TraceFlags.DEFAULT)
        return trace.NonRecordingSpan(
            trace.SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=flags)
        )

    @staticmethod
    def _record() -> logging.LogRecord:
        return logging.LogRecord(
            name="cosmos_curator.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="traced",
            args=(),
            exc_info=None,
        )

    @pytest.mark.parametrize("sampled", [True, False])
    def test_stamps_zero_padded_hex_ids(self, *, sampled: bool) -> None:
        """Ids are padded to the 32/16 hex-char widths Tempo/Jaeger expect."""
        record = self._record()

        _inject_span_context(self._span(trace_id=0xABCD, span_id=0xEF, sampled=sampled), record)

        assert record.__dict__["trace_id"] == "0000000000000000000000000000abcd"
        assert record.__dict__["span_id"] == "00000000000000ef"
        assert record.__dict__["trace_sampled"] is sampled

    def test_values_are_json_native(self) -> None:
        """Ray's JSONFormatter calls json.dumps without a ``default=``, so no OTel objects."""
        record = self._record()
        span = self._span(trace_id=0x0AF7651916CD43DD8448EB211C80319C, span_id=0xB7AD6B7169203331, sampled=True)

        _inject_span_context(span, record)

        fields = {key: record.__dict__[key] for key in ("trace_id", "span_id", "trace_sampled")}
        assert json.loads(json.dumps(fields)) == {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "b7ad6b7169203331",
            "trace_sampled": True,
        }
