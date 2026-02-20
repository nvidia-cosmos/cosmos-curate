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

r"""Replay OTel trace ``.jsonl`` files to an OTLP endpoint.

Standalone CLI tool.  Run after a pipeline execution to re-export the
``.jsonl`` span files produced by ``--profile-tracing`` into Jaeger,
Grafana Tempo, VictoriaTraces, or any OTLP-compatible backend.

The script appends ``/v1/traces`` to the URL you pass.  Backends that use
a different path (e.g. VictoriaTraces) require that path in the base URL.

Usage::

    # Replay to a local Jaeger (standard OTLP path)
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:4318

    # Replay to VictoriaTraces (path is /insert/opentelemetry/v1/traces)
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:10428/insert/opentelemetry

    # Filter by span name
    python -m benchmarks.replay_traces ./profiles/traces/ \
        --replay-to http://localhost:4318 --filter-name RemuxStage

"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

from loguru import logger


def _discover_jsonl(directory: pathlib.Path) -> list[pathlib.Path]:
    """Glob ``.jsonl`` span files in *directory*, sorted by name."""
    if not directory.is_dir():
        msg = f"Directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    return sorted(directory.glob("*.jsonl"))


def _parse_spans(path: pathlib.Path) -> list[dict]:
    """Parse all spans from a single ``.jsonl`` file."""
    spans: list[dict] = []
    with path.open() as f:
        for line_no, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                spans.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.warning(f"  Skipping {path.name}:{line_no}: {e}")
    return spans


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse an OTel ISO-8601 timestamp string."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _hex_to_int(hex_str: str | None) -> int:
    """Convert a ``0x...`` hex string to int, or 0 if falsy."""
    if not hex_str:
        return 0
    return int(hex_str, 16)


def _iso_to_ns(ts: str | None) -> int | None:
    """Convert ISO-8601 timestamp to nanoseconds since epoch."""
    dt = _parse_timestamp(ts)
    if dt is None:
        return None
    return int(dt.timestamp() * 1_000_000_000)


def _json_to_readable_span(span_json: dict) -> object | None:
    """Convert a single span JSON dict to an OTel ``ReadableSpan``.

    Returns ``None`` if the span is missing required context IDs.

    """
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import ReadableSpan  # noqa: PLC0415
    from opentelemetry.trace import SpanContext, SpanKind, StatusCode, TraceFlags  # noqa: PLC0415
    from opentelemetry.trace.status import Status  # noqa: PLC0415

    status_map = {
        "OK": StatusCode.OK,
        "ERROR": StatusCode.ERROR,
        "UNSET": StatusCode.UNSET,
    }

    kind_map: dict[str | int, SpanKind] = {
        "SpanKind.INTERNAL": SpanKind.INTERNAL,
        "SpanKind.SERVER": SpanKind.SERVER,
        "SpanKind.CLIENT": SpanKind.CLIENT,
        "SpanKind.PRODUCER": SpanKind.PRODUCER,
        "SpanKind.CONSUMER": SpanKind.CONSUMER,
        0: SpanKind.INTERNAL,
        1: SpanKind.SERVER,
        2: SpanKind.CLIENT,
        3: SpanKind.PRODUCER,
        4: SpanKind.CONSUMER,
    }

    ctx = span_json.get("context", {})
    trace_id = _hex_to_int(ctx.get("trace_id"))
    span_id = _hex_to_int(ctx.get("span_id"))

    if trace_id == 0 or span_id == 0:
        return None

    parent_id = _hex_to_int(span_json.get("parent_id"))
    # TraceFlags.SAMPLED (0x01) is required so that BatchSpanProcessor and
    # downstream backends treat the span as sampled rather than silently
    # dropping it. Without this flag the replay appears to succeed but no
    # spans show up in the collector / viewer.
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    parent_context = (
        SpanContext(trace_id=trace_id, span_id=parent_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        if parent_id
        else None
    )

    # Status
    status_data = span_json.get("status", {})
    status_code_str = status_data.get("status_code", "UNSET") if isinstance(status_data, dict) else "UNSET"
    status_desc = status_data.get("description") if isinstance(status_data, dict) else None
    status = Status(status_map.get(status_code_str, StatusCode.UNSET), status_desc)

    # Resource
    resource_data = span_json.get("resource", {})
    resource_attrs = resource_data.get("attributes", {}) if isinstance(resource_data, dict) else {}

    return ReadableSpan(
        name=span_json.get("name", "<unnamed>"),
        context=span_context,
        parent=parent_context,
        resource=Resource(attributes=resource_attrs),
        attributes=span_json.get("attributes", {}) or {},
        kind=kind_map.get(span_json.get("kind"), SpanKind.INTERNAL),
        status=status,
        start_time=_iso_to_ns(span_json.get("start_time")),
        end_time=_iso_to_ns(span_json.get("end_time")),
    )


def _replay_to_otlp(all_spans: list[dict], endpoint: str) -> None:
    """Replay parsed spans to an OTLP HTTP endpoint.

    Converts raw span dicts into ``ReadableSpan`` objects and exports
    them synchronously using ``OTLPSpanExporter.export()``.

    ::

        .jsonl dicts --> _json_to_readable_span --> OTLPSpanExporter.export()
                                                         |
                                                         v
                                                   OTLP endpoint

    A synchronous (direct) export is used instead of ``BatchSpanProcessor``
    because replay is a one-shot operation: we want immediate confirmation
    that all spans were delivered and clear error reporting on failure.

    Requires ``opentelemetry-exporter-otlp-proto-http``.

    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: PLC0415

    readable_spans = []
    skipped = 0

    for span_json in all_spans:
        try:
            readable = _json_to_readable_span(span_json)
            if readable is None:
                skipped += 1
                continue
            readable_spans.append(readable)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"  Skipping span: {e}")
            skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} malformed/empty span(s)")

    if not readable_spans:
        logger.warning("No valid spans to replay")
        return

    logger.info(f"Replaying {len(readable_spans)} span(s) to {endpoint}")

    # OTLPSpanExporter uses the endpoint parameter as-is (no auto-appending
    # of /v1/traces), so we must include the full path ourselves.
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")

    result = exporter.export(readable_spans)
    if result == SpanExportResult.SUCCESS:
        logger.info(f"Replay complete: {len(readable_spans)} span(s) exported successfully")
    else:
        logger.error(f"Replay failed: exporter returned {result}")

    exporter.shutdown()


def main(argv: list[str] | None = None) -> None:
    """Entry point for the trace replay CLI."""
    parser = argparse.ArgumentParser(
        description="Replay OTel trace .jsonl files to an OTLP endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", type=pathlib.Path, help="Directory containing .jsonl span files.")
    parser.add_argument(
        "--replay-to",
        type=str,
        required=True,
        metavar="URL",
        help=(
            "OTLP HTTP endpoint base URL. /v1/traces is appended. "
            "Jaeger/Tempo: http://localhost:4318 "
            "VictoriaTraces: http://localhost:10428/insert/opentelemetry"
        ),
    )
    parser.add_argument(
        "--filter-name",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Only include spans whose name contains this substring.",
    )

    args = parser.parse_args(argv)

    # 1. Discover
    paths = _discover_jsonl(args.directory)
    if not paths:
        logger.error(f"No .jsonl files found in {args.directory}")
        sys.exit(1)
    logger.info(f"Found {len(paths)} span file(s)")

    # 2. Parse
    all_spans: list[dict] = []
    for p in paths:
        all_spans.extend(_parse_spans(p))

    if not all_spans:
        logger.error("No spans found")
        sys.exit(1)

    # 3. Filter
    if args.filter_name:
        before = len(all_spans)
        all_spans = [s for s in all_spans if args.filter_name in s.get("name", "")]
        logger.info(f"Filter '{args.filter_name}': {before} -> {len(all_spans)} span(s)")
        if not all_spans:
            logger.error(f"No spans match filter '{args.filter_name}'")
            sys.exit(1)

    logger.info(f"Total: {len(all_spans)} span(s)")

    # 4. Replay
    _replay_to_otlp(all_spans, args.replay_to)


if __name__ == "__main__":
    main()
