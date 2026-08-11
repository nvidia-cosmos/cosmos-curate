# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in collector integration test for the chart OTLP log parser.

The JSON ``timestamp_ns`` fixture value is larger than 2^53, so collector
``ParseJSON`` decodes it through float64 and the emitted ``timeUnixNano`` has a
deterministic 64 ns rounding offset.
"""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "charts" / "cosmos-curator" / "templates" / "otlp-log-collector-config.yaml").exists():
            return parent
    message = "could not find repository root"
    raise RuntimeError(message)


REPO_ROOT = _repo_root()
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "otel-log-parser"
CHART_COLLECTOR_TEMPLATE = REPO_ROOT / "charts" / "cosmos-curator" / "templates" / "otlp-log-collector-config.yaml"
DEFAULT_IMAGE = "ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector-contrib:0.157.0"
DEFAULT_TIMEOUT_SECONDS = 20
CONTAINER_RUNTIMES = frozenset({"docker", "podman"})

pytestmark = [
    pytest.mark.otel_collector,
    pytest.mark.skipif(
        os.environ.get("COSMOS_CURATOR_RUN_OTEL_COLLECTOR_TESTS") != "1",
        reason="set COSMOS_CURATOR_RUN_OTEL_COLLECTOR_TESTS=1 to run collector integration tests",
    ),
]


def _extract_chart_relay_config() -> str:
    lines = CHART_COLLECTOR_TEMPLATE.read_text().splitlines()
    try:
        start = lines.index("  relay.yaml: |") + 1
    except ValueError as exc:
        message = f"could not find relay.yaml block in {CHART_COLLECTOR_TEMPLATE}"
        raise RuntimeError(message) from exc

    relay_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("    "):
            relay_lines.append(line[4:])
        elif line == "":
            relay_lines.append("")
        else:
            break
    return "\n".join(relay_lines).rstrip() + "\n"


def _drop_top_level_sections(config: str, section_names: set[str]) -> str:
    lines = config.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line and not line.startswith(" ") and line.endswith(":") and line[:-1] in section_names:
            index += 1
            while index < len(lines) and (lines[index].startswith(" ") or lines[index] == ""):
                index += 1
            continue
        kept.append(line)
        index += 1
    return "\n".join(kept).rstrip() + "\n"


def _drop_processor(config: str, processor_name: str) -> str:
    lines = config.splitlines()
    kept: list[str] = []
    index = 0
    start_marker = f"  {processor_name}:"
    while index < len(lines):
        line = lines[index]
        if line == start_marker:
            index += 1
            while index < len(lines) and not (lines[index].startswith("  ") and not lines[index].startswith("    ")):
                index += 1
            continue
        kept.append(line)
        index += 1
    return "\n".join(kept).rstrip() + "\n"


def _use_fixture_input(config: str) -> str:
    lines = config.splitlines()
    rewritten: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() == "include:":
            indent = line[: len(line) - len(line.lstrip())]
            rewritten.extend([line, f"{indent}  - /input/input.log"])
            index += 1
            while index < len(lines) and lines[index].startswith(f"{indent}  - "):
                index += 1
            continue
        if line.lstrip().startswith("start_at:"):
            indent = line[: len(line) - len(line.lstrip())]
            rewritten.append(f"{indent}start_at: beginning")
        else:
            rewritten.append(line)
        index += 1
    return "\n".join(rewritten).rstrip() + "\n"


def _chart_pipeline_processors(config: str) -> list[str]:
    match = re.search(r"^\s+processors:\s*\[([^\]]+)\]", config, flags=re.MULTILINE)
    if not match:
        message = "could not find logs pipeline processors in chart collector config"
        raise RuntimeError(message)
    return [
        processor.strip()
        for processor in match.group(1).split(",")
        if processor.strip() not in {"memory_limiter", "resource"}
    ]


def _local_collector_config() -> str:
    chart_config = _extract_chart_relay_config()
    processors = _chart_pipeline_processors(chart_config)
    local_config = _use_fixture_input(chart_config)
    local_config = _drop_top_level_sections(local_config, {"exporters", "extensions", "service"})
    local_config = _drop_processor(local_config, "memory_limiter")
    local_config = _drop_processor(local_config, "resource")

    return (
        local_config
        + "\n"
        + "\n".join(
            [
                "extensions:",
                "  file_storage/ray:",
                "    directory: /storage",
                "",
                "exporters:",
                "  file/logs:",
                "    path: /output/otel-output.json",
                "    format: json",
                "",
                "service:",
                "  pipelines:",
                "    logs:",
                "      receivers: [file_log/ray]",
                f"      processors: [{', '.join(processors)}]",
                "      exporters: [file/logs]",
                "  extensions: [file_storage/ray]",
                "  telemetry:",
                "    logs:",
                '      level: "info"',
                "",
            ]
        )
    )


def _load_expected(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fixture_cases() -> list[object]:
    cases = [
        pytest.param(path, id=path.name)
        for path in sorted(FIXTURE_ROOT.iterdir())
        if path.is_dir() and (path / "input.log").is_file() and (path / "expected.jsonl").is_file()
    ]
    if not cases:
        message = f"no OTLP log parser fixtures found under {FIXTURE_ROOT}"
        raise RuntimeError(message)
    return cases


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)  # noqa: S603


def _require_container_runtime(runtime: str) -> None:
    if runtime not in CONTAINER_RUNTIMES:
        pytest.skip(f"{runtime!r} is not an allowed test container runtime")
    if shutil.which(runtime) is None:
        pytest.skip(f"{runtime!r} not found on PATH")
    result = _run([runtime, "version"], timeout=10)
    if result.returncode != 0:
        pytest.skip(f"{runtime} is not usable: {result.stderr or result.stdout}")


def _run_collector(runtime: str, image: str, workdir: Path, fixture_dir: Path, timeout_seconds: int) -> Path:
    config = workdir / "collector.yaml"
    input_dir = workdir / "input"
    output_dir = workdir / "output"
    storage_dir = workdir / "storage"
    input_dir.mkdir()
    output_dir.mkdir()
    storage_dir.mkdir()
    output_dir.chmod(0o777)
    storage_dir.chmod(0o777)
    config.write_text(_local_collector_config())
    shutil.copyfile(fixture_dir / "input.log", input_dir / "input.log")

    container_name = f"curator-otel-log-parser-{uuid.uuid4().hex}"
    run_cmd = [
        runtime,
        "run",
        "--name",
        container_name,
        "-d",
        "-v",
        f"{config}:/etc/otelcol/config.yaml:ro",
        "-v",
        f"{input_dir}:/input:ro",
        "-v",
        f"{output_dir}:/output",
        "-v",
        f"{storage_dir}:/storage",
        "-e",
        "OTELCOL_TELEMETRY_LOG_LEVEL=info",
        image,
        "--config=/etc/otelcol/config.yaml",
    ]
    result = _run(run_cmd, timeout=timeout_seconds)
    assert result.returncode == 0, result.stderr or result.stdout

    output_path = output_dir / "otel-output.json"
    deadline = time.monotonic() + timeout_seconds
    last_size = -1
    stable_since: float | None = None
    try:
        while time.monotonic() < deadline:
            if output_path.exists():
                size = output_path.stat().st_size
                if size > 0:
                    if size == last_size:
                        stable_since = time.monotonic() if stable_since is None else stable_since
                        if time.monotonic() - stable_since >= 1.0:
                            return output_path
                    else:
                        last_size = size
                        stable_since = None
            time.sleep(0.2)
        logs = _run([runtime, "logs", container_name], timeout=10)
        message = f"collector did not write {output_path}:\n{logs.stderr}{logs.stdout}"
        raise AssertionError(message)
    finally:
        _run([runtime, "stop", "-t", "2", container_name], timeout=10)
        _run([runtime, "rm", "-f", container_name], timeout=10)


def _iter_json_documents(text: str) -> list[dict[str, object]]:
    decoder = json.JSONDecoder()
    docs: list[dict[str, object]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        doc, index = decoder.raw_decode(text, index)
        assert isinstance(doc, dict)
        docs.append(doc)
    return docs


def _field_value(field: dict[str, object]) -> object:
    if "stringValue" in field:
        return field["stringValue"]
    if "intValue" in field:
        return int(field["intValue"])
    for key in ("doubleValue", "boolValue", "bytesValue"):
        if key in field:
            return field[key]
    if "arrayValue" in field:
        array_value = field["arrayValue"]
        return array_value.get("values", []) if isinstance(array_value, dict) else []
    if "kvlistValue" in field:
        kvlist_value = field["kvlistValue"]
        return kvlist_value.get("values", []) if isinstance(kvlist_value, dict) else []
    return None


def _attrs(kv: list[dict[str, object]]) -> dict[str, object]:
    return {item["key"]: _field_value(item["value"]) for item in kv}


def _body(log_record: dict[str, object]) -> object:
    value = log_record.get("body", {})
    return _field_value(value) if isinstance(value, dict) else value


def _normalize(output_path: Path) -> list[dict[str, object]]:
    docs = _iter_json_documents(output_path.read_text())
    records: list[dict[str, object]] = []
    keep = {"code.filepath", "curator.log.parser", "job_id", "lineno", "message", "name", "span_id", "trace_id"}
    for doc in docs:
        for resource_log in doc.get("resourceLogs", []):
            for scope_log in resource_log.get("scopeLogs", []):
                for log_record in scope_log.get("logRecords", []):
                    attrs = _attrs(log_record.get("attributes", []))
                    record = {
                        "body": _body(log_record),
                        "attributes": {key: attrs[key] for key in sorted(keep) if key in attrs},
                    }
                    if "timeUnixNano" in log_record:
                        record["time_unix_nano"] = log_record["timeUnixNano"]
                    if "severityNumber" in log_record:
                        record["severity_number"] = log_record["severityNumber"]
                    if "severityText" in log_record:
                        record["severity_text"] = log_record["severityText"]
                    if "traceId" in log_record:
                        record["trace_id"] = log_record["traceId"]
                    if "spanId" in log_record:
                        record["span_id"] = log_record["spanId"]
                    records.append(record)
    return records


def _pretty_json(records: list[dict[str, object]]) -> str:
    return json.dumps(records, indent=2, sort_keys=True)


@pytest.mark.parametrize("fixture_dir", _fixture_cases())
def test_chart_otlp_log_parser_handles_fixture_logs(tmp_path: Path, fixture_dir: Path) -> None:
    """Run focused sample logs through the chart collector config."""
    runtime = os.environ.get("OTEL_COLLECTOR_RUNTIME", "docker")
    image = os.environ.get("OTEL_COLLECTOR_IMAGE", DEFAULT_IMAGE)
    timeout_seconds = int(os.environ.get("OTEL_COLLECTOR_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    _require_container_runtime(runtime)

    output_path = _run_collector(runtime, image, tmp_path, fixture_dir, timeout_seconds)
    actual = _normalize(output_path)
    expected = _load_expected(fixture_dir / "expected.jsonl")

    assert actual == expected, f"normalized collector output for {fixture_dir.name}:\n" + _pretty_json(actual)
