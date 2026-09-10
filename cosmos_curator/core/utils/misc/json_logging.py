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

"""Shared stdlib-logging helpers for the container launcher scripts.

The launcher scripts (``onto_nvcf.py`` / ``onto_slurm.py``) run *outside* Ray and
use the standard library ``logging`` module rather than loguru. This module gives
them the same ``PYTHON_LOG_FORMAT`` toggle used by ``cosmos_xenna.utils.python_log``:

- ``text`` (default): human-readable output, unchanged from before.
- ``json``: one flat JSON object per line, aligned with Ray's structured-logging
  schema (top-level ``levelname`` / ``message`` / ``timestamp_ns`` plus flat identity
  fields: pod, replica, pid, run_id, seq). Because launcher logs land in the same
  stream as the Ray pipeline logs, matching Ray's field names lets a single
  log-shipper mapping parse, sort, and correlate every source.

When distributed tracing is also enabled, OTel's logging instrumentation stamps
``trace_id`` / ``span_id`` / ``trace_sampled`` onto records emitted inside a span
(see ``core/utils/infra/tracing_hook.py``); those fields are passed through here.

This module deliberately does not import anything from ``cosmos_xenna`` -- nor from
OpenTelemetry -- so the launcher scripts stay lightweight and xenna remains
self-contained.
"""

import itertools
import json
import logging
import os
import socket
from typing import Any


def wants_json_logs() -> bool:
    """Return True when PYTHON_LOG_FORMAT selects structured (JSON) logging."""
    return os.getenv("PYTHON_LOG_FORMAT", "").strip().lower() == "json"


def _node_identity() -> str:
    """Best-effort identifier for the emitting node/instance across platforms.

    Prefers the k8s ``POD_NAME`` (downward API), then SLURM's ``SLURMD_NODENAME``,
    then the container/host name. This keeps the ``pod`` field populated and
    distinguishable off-k8s (SLURM/NVCF/local), where ``POD_NAME`` is absent.
    """
    return os.getenv("POD_NAME") or os.getenv("SLURMD_NODENAME") or socket.gethostname() or ""


def _replica_from_pod_name(pod: str) -> str:
    """Return the trailing ``-N`` ordinal of a pod name, or "" when absent."""
    _, sep, tail = pod.rpartition("-")
    return tail if sep and tail.isdigit() else ""


# Trace-correlation attributes stamped onto records by OTel's logging
# instrumentation. Absent unless tracing is enabled and a span is active.
_SPAN_FIELDS = ("trace_id", "span_id", "trace_sampled")


class StructuredJsonFormatter(logging.Formatter):
    """Format stdlib ``LogRecord`` objects as flat, Ray-aligned JSON lines.

    Emits the same flat top-level fields (``levelname`` / ``message`` /
    ``timestamp_ns`` plus flat identity fields ``pod`` / ``replica`` / ``pid`` /
    ``run_id`` / ``seq``) that Ray's ``JSONFormatter`` produces for pipeline logs, so
    one log-shipper mapping can parse, sort, and correlate launcher and pipeline logs
    together. Identity fields are read once from the environment; ``seq`` is a
    gap-free per-process counter that tiebreaks equal timestamps. Trace-correlation
    fields (``trace_id`` / ``span_id`` / ``trace_sampled``) are emitted only when
    present on the record.
    """

    def __init__(self) -> None:
        """Capture per-process identity fields once and start the seq counter."""
        super().__init__()
        self._pod = _node_identity()
        # ``replica`` is the k8s StatefulSet ordinal, so source it strictly from
        # POD_NAME: it is populated only on k8s/NVCF and stays "" off-k8s (SLURM/local)
        # rather than misreading a node name like ``pool0-0218`` as a replica ordinal.
        self._replica = _replica_from_pod_name(os.getenv("POD_NAME", ""))
        self._run_id = os.getenv("CURATOR_RUN_ID", "")
        self._seq = itertools.count()

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as a single flat, Ray-aligned JSON line."""
        payload: dict[str, Any] = {
            "levelname": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "process": record.process,
            "asctime": self.formatTime(record),
            "timestamp_ns": int(record.created * 1_000_000_000),
            "pod": self._pod,
            "replica": self._replica,
            "pid": os.getpid(),
            "run_id": self._run_id,
            "seq": next(self._seq),
        }
        # Omit span fields entirely when untraced: an absent key keeps log UIs from
        # rendering a link to a nonexistent trace.
        for field in _SPAN_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_stdlib_logging(*, text_format: str, datefmt: str | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger for a launcher script, honoring PYTHON_LOG_FORMAT.

    In text mode this is equivalent to ``logging.basicConfig(format=text_format,
    datefmt=datefmt, level=level)`` (byte-for-byte the prior behavior). In json mode
    it installs a single stderr handler using :class:`StructuredJsonFormatter`.

    Like ``basicConfig``, this is a no-op if the root logger already has handlers.
    """
    if wants_json_logs():
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(level=level, format=text_format, datefmt=datefmt)
