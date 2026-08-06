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
"""Tests for the launcher structured-logging helper used by onto_nvcf/onto_slurm."""

import json
import logging

import pytest

from cosmos_curator.core.utils.misc.json_logging import (
    StructuredJsonFormatter,
    configure_stdlib_logging,
    wants_json_logs,
)
from cosmos_curator.scripts import onto_nvcf


def _record(msg: str, *, level: int = logging.INFO, exc_info: object = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="cosmos_curator.scripts.onto_nvcf",
        level=level,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=exc_info,  # type: ignore[arg-type]
        func="some_func",
    )


def test_onto_nvcf_uses_shared_helper() -> None:
    """onto_nvcf must delegate to the shared helper so the toggle stays consistent."""
    assert onto_nvcf.configure_stdlib_logging is configure_stdlib_logging


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("", False), ("text", False), ("json", True), ("JSON", True), (" Json ", True)],
)
def test_wants_json_logs_toggle(monkeypatch: pytest.MonkeyPatch, value: str | None, *, expected: bool) -> None:
    """PYTHON_LOG_FORMAT selects JSON case-insensitively and defaults to text."""
    if value is None:
        monkeypatch.delenv("PYTHON_LOG_FORMAT", raising=False)
    else:
        monkeypatch.setenv("PYTHON_LOG_FORMAT", value)
    assert wants_json_logs() is expected


def test_json_formatter_emits_valid_json_with_identity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON mode emits flat, Ray-aligned JSON enriched with pod/replica/pid/run_id and a timestamp."""
    monkeypatch.setenv("POD_NAME", "cosmos-curator-5")
    monkeypatch.setenv("CURATOR_RUN_ID", "req-123")
    fmt = StructuredJsonFormatter()

    obj = json.loads(fmt.format(_record("hello json")))

    assert obj["message"] == "hello json"
    assert obj["levelname"] == "INFO"
    assert obj["name"] == "cosmos_curator.scripts.onto_nvcf"
    assert obj["pod"] == "cosmos-curator-5"
    assert obj["replica"] == "5"
    assert obj["run_id"] == "req-123"
    assert isinstance(obj["pid"], int)
    assert isinstance(obj["timestamp_ns"], int)
    assert "asctime" in obj


def test_json_formatter_seq_is_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-process seq counter is gap-free and monotonic across records."""
    monkeypatch.setenv("POD_NAME", "pod-0")
    fmt = StructuredJsonFormatter()
    seqs = [json.loads(fmt.format(_record(f"m{i}")))["seq"] for i in range(3)]
    assert seqs == [0, 1, 2]


def test_json_formatter_pod_prefers_pod_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The k8s POD_NAME wins over the SLURM/host fallbacks."""
    monkeypatch.setenv("POD_NAME", "cosmos-curator-2")
    monkeypatch.setenv("SLURMD_NODENAME", "slurm-node-9")
    fmt = StructuredJsonFormatter()
    obj = json.loads(fmt.format(_record("x")))
    assert obj["pod"] == "cosmos-curator-2"
    assert obj["replica"] == "2"


def test_json_formatter_pod_falls_back_to_slurm_nodename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off k8s, pod falls back to SLURMD_NODENAME so SLURM nodes stay distinguishable.

    replica stays empty off-k8s: it is a k8s StatefulSet ordinal sourced only from
    POD_NAME, so a numeric-suffixed node name must not be misread as a replica.
    """
    monkeypatch.delenv("POD_NAME", raising=False)
    monkeypatch.setenv("SLURMD_NODENAME", "slurm-node-9")
    fmt = StructuredJsonFormatter()
    obj = json.loads(fmt.format(_record("x")))
    assert obj["pod"] == "slurm-node-9"
    assert obj["replica"] == ""


def test_json_formatter_pod_falls_back_to_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither POD_NAME nor SLURMD_NODENAME, pod falls back to the hostname."""
    monkeypatch.delenv("POD_NAME", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    monkeypatch.delenv("CURATOR_RUN_ID", raising=False)
    monkeypatch.setattr(
        "cosmos_curator.core.utils.misc.json_logging.socket.gethostname",
        lambda: "host-xyz",
    )
    fmt = StructuredJsonFormatter()
    obj = json.loads(fmt.format(_record("nopod")))
    assert obj["pod"] == "host-xyz"
    assert obj["run_id"] == ""


@pytest.mark.parametrize("sampled", [True, False])
def test_json_formatter_includes_span_fields(monkeypatch: pytest.MonkeyPatch, *, sampled: bool) -> None:
    """Trace-correlation attributes stamped by OTel's log hook reach the JSON line."""
    monkeypatch.setenv("POD_NAME", "pod-3")
    fmt = StructuredJsonFormatter()
    rec = _record("traced")
    rec.__dict__.update(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        span_id="b7ad6b7169203331",
        trace_sampled=sampled,
    )

    obj = json.loads(fmt.format(rec))

    assert obj["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
    assert obj["span_id"] == "b7ad6b7169203331"
    assert obj["trace_sampled"] is sampled


def test_json_formatter_omits_span_fields_when_untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an active span the keys are absent, so log UIs cannot build a dead link."""
    monkeypatch.setenv("POD_NAME", "pod-4")
    fmt = StructuredJsonFormatter()

    obj = json.loads(fmt.format(_record("untraced")))

    assert not {"trace_id", "span_id", "trace_sampled"} & obj.keys()


def test_json_formatter_includes_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception info is preserved under the top-level exception field in JSON mode."""
    monkeypatch.setenv("POD_NAME", "pod-1")
    fmt = StructuredJsonFormatter()
    error = ValueError("boom")
    try:
        raise error
    except ValueError as exc:
        rec = _record("failed", level=logging.ERROR, exc_info=(type(exc), exc, exc.__traceback__))
    obj = json.loads(fmt.format(rec))
    assert "ValueError: boom" in obj["exception"]


def _snapshot_root() -> tuple[list[logging.Handler], int]:
    root = logging.getLogger()
    return list(root.handlers), root.level


def _restore_root(state: tuple[list[logging.Handler], int]) -> None:
    root = logging.getLogger()
    root.handlers[:] = state[0]
    root.setLevel(state[1])


def test_configure_text_mode_installs_plain_formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text mode does not install the JSON formatter (behavior unchanged)."""
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "text")
    state = _snapshot_root()
    logging.getLogger().handlers.clear()  # let basicConfig install a fresh handler
    try:
        configure_stdlib_logging(text_format="%(asctime)s - %(levelname)s - %(message)s")
        formatters = [h.formatter for h in logging.getLogger().handlers if h.formatter is not None]
        assert not any(isinstance(f, StructuredJsonFormatter) for f in formatters)
    finally:
        _restore_root(state)


def test_configure_json_mode_installs_json_formatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON mode installs a StructuredJsonFormatter on the root handler."""
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    state = _snapshot_root()
    logging.getLogger().handlers.clear()
    try:
        configure_stdlib_logging(text_format="%(asctime)s - %(levelname)s - %(message)s")
        formatters = [h.formatter for h in logging.getLogger().handlers]
        assert any(isinstance(f, StructuredJsonFormatter) for f in formatters)
    finally:
        _restore_root(state)
