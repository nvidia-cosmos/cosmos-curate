# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Helm render checks for chart OTLP observability wiring."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "charts" / "cosmos-curator" / "Chart.yaml").exists():
            return parent
    message = "could not find repository root"
    raise RuntimeError(message)


REPO_ROOT = _repo_root()
CHART_DIR = REPO_ROOT / "charts" / "cosmos-curator"
JOB_DRIVER_LOG_GLOB = "/tmp/ray/session_*/logs/job-driver*"  # noqa: S108 - chart path under test

pytestmark = pytest.mark.helm

requires_collector_runtime = pytest.mark.skipif(
    os.environ.get("COSMOS_CURATOR_RUN_OTEL_COLLECTOR_TESTS") != "1",
    reason="set COSMOS_CURATOR_RUN_OTEL_COLLECTOR_TESTS=1 to run collector validation tests",
)


def _yaml() -> ModuleType:
    return pytest.importorskip("yaml")


def _run_command(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True, timeout=timeout)  # noqa: S603


def _render_chart(values_file: Path, show_only: list[str]) -> list[dict[str, object]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")
    args = ["helm", "template", "diag", str(CHART_DIR), "-f", str(values_file)]
    for template in show_only:
        args.extend(["--show-only", template])

    try:
        result = _run_command(args, timeout=30)
    except subprocess.TimeoutExpired as exc:
        message = "helm template timed out after 30s"
        raise AssertionError(message) from exc
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return [doc for doc in _yaml().safe_load_all(result.stdout) if doc]


def _object(docs: list[dict[str, object]], kind: str, name: str) -> dict[str, object]:
    for doc in docs:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    message = f"rendered chart did not include {kind}/{name}"
    raise AssertionError(message)


def _resource_attributes(relay_config: str) -> dict[str, str]:
    relay = _yaml().safe_load(relay_config)
    attributes = relay["processors"]["resource"]["attributes"]
    return {attr["key"]: attr["value"] for attr in attributes}


def _collector_relay(config_map: dict[str, object]) -> dict[str, object]:
    data = config_map["data"]
    assert isinstance(data, dict)
    return _yaml().safe_load(data["relay"])


def _collector_image() -> str:
    values = _yaml().safe_load((CHART_DIR / "values.yaml").read_text())
    collector_image = values["opentelemetry-collector"]["image"]
    return f"{collector_image['repository']}:{collector_image['tag']}"


def _write_test_tls_files(cert_dir: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    cert_dir.mkdir()
    result = _run_command(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(cert_dir / "tls.key"),
            "-out",
            str(cert_dir / "tls.crt"),
            "-sha256",
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=collector-validate.test",
        ],
        timeout=30,
    )
    if result.returncode != 0:
        message = result.stderr or result.stdout
        raise AssertionError(message)
    (cert_dir / "ca.crt").write_text((cert_dir / "tls.crt").read_text())


CONTAINER_RUNTIMES = frozenset({"docker", "podman"})


def _collector_runtime() -> str:
    runtime = os.environ.get("OTEL_COLLECTOR_RUNTIME", "docker")
    if runtime not in CONTAINER_RUNTIMES:
        pytest.skip(f"{runtime!r} is not an allowed test container runtime")
    if shutil.which(runtime) is None:
        pytest.skip(f"{runtime!r} not found on PATH")
    version_result = _run_command([runtime, "version"], timeout=10)
    if version_result.returncode != 0:
        pytest.skip(f"{runtime} is not usable: {version_result.stderr or version_result.stdout}")
    return runtime


def _validate_collector_config(
    config_file: Path,
    *,
    mounts: list[tuple[Path, str]] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    runtime = _collector_runtime()

    args = [
        runtime,
        "run",
        "--rm",
        "-v",
        f"{config_file}:/etc/otelcol/config.yaml:ro",
    ]
    for source, target in mounts or []:
        args.extend(["-v", f"{source}:{target}"])
    collector_env = {
        "MY_POD_IP": "127.0.0.1",
        "POD_IP": "127.0.0.1",
        "POD_NS": "test-namespace",
        "POD_NAMESPACE": "test-namespace",
        "POD_NAME": "collector-validate",
        "POD_UID": "00000000-0000-0000-0000-000000000000",
        "NODE_NAME": "test-node",
        "SERVICE_ACCOUNT": "test-service-account",
        "POD_INDEX": "0",
        "OTELCOL_TELEMETRY_LOG_LEVEL": "info",
    }
    collector_env.update(env or {})
    for key, value in collector_env.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend([_collector_image(), "validate", "--config=/etc/otelcol/config.yaml"])

    result = _run_command(args, timeout=120)
    if result.returncode != 0:
        message = result.stderr or result.stdout
        raise AssertionError(message)


def _log_collector_relay(config_map: dict[str, object]) -> dict[str, object]:
    data = config_map["data"]
    assert isinstance(data, dict)
    return _yaml().safe_load(data["relay.yaml"])


def _collector_metrics_exporter(values_file: Path) -> dict[str, object]:
    docs = _render_chart(values_file, ["templates/otel-collector-config.yaml"])
    collector = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))
    exporters = collector["exporters"]
    assert isinstance(exporters, dict)
    exporter = exporters["otlp_http/metrics"]
    assert isinstance(exporter, dict)
    return exporter


def _container(pod_spec: dict[str, object], container_name: str) -> dict[str, object]:
    for container in pod_spec["containers"]:
        if container["name"] == container_name:
            return container
    message = f"pod spec did not include container {container_name}"
    raise AssertionError(message)


def _mount_names(pod_spec: dict[str, object], container_name: str) -> set[str]:
    return {mount["name"] for mount in _container(pod_spec, container_name).get("volumeMounts", [])}


def _otlp_secret_pod_spec(tmp_path: Path, *, extra_values: str = "") -> dict[str, object]:
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        f"""
otlp:
  endpoint: https://otlp.example
  tls:
    secret:
      enabled: true
logging:
  otlp:
    enabled: true
{extra_values}
""".lstrip()
    )
    docs = _render_chart(values_file, ["templates/statefulset.yaml"])
    return _object(docs, "StatefulSet", "cosmos-curator")["spec"]["template"]["spec"]


@pytest.mark.parametrize(
    ("log_collector_values", "expected_job_driver_excludes"),
    [
        ("", set()),
        ("    collectJobDriverLogs: false\n", {JOB_DRIVER_LOG_GLOB}),
    ],
)
def test_helm_log_collector_job_driver_source_is_explicit(
    tmp_path: Path, log_collector_values: str, expected_job_driver_excludes: set[str]
) -> None:
    """Driver files stay enabled by default and can be excluded explicitly."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        f"""
otlp:
  endpoint: https://otlp.example
logging:
  otlp:
    enabled: true
{log_collector_values}""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])
    relay = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    excludes = relay["receivers"]["file_log/ray"]["exclude"]

    assert set(excludes) & {JOB_DRIVER_LOG_GLOB} == expected_job_driver_excludes


def test_helm_otlp_client_certs_skip_curator_container_without_in_process_exporters(tmp_path: Path) -> None:
    """Logging-only deployments keep the OTLP client key out of the curator container."""
    pod_spec = _otlp_secret_pod_spec(tmp_path)

    assert "otlp-cert-store" in {volume["name"] for volume in pod_spec["volumes"]}
    assert "otlp-cert-store" in _mount_names(pod_spec, "otlp-log-collector")
    assert "otlp-cert-store" not in _mount_names(pod_spec, "cosmos-curator")


def test_helm_logging_with_args_override_still_enables_entrypoint_tee(tmp_path: Path) -> None:
    """Overriding image CMD args leaves the image entrypoint wrapper active."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
logging:
  otlp:
    enabled: true
args:
  - pixi
  - run
  - custom-task
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/statefulset.yaml"])
    pod_spec = _object(docs, "StatefulSet", "cosmos-curator")["spec"]["template"]["spec"]
    curator = _container(pod_spec, "cosmos-curator")
    env = {item["name"]: item.get("value") for item in curator["env"]}

    assert curator["args"] == ["pixi", "run", "custom-task"]
    assert env["COSMOS_CURATOR_TEE_STDOUT_STDERR"] == "true"


def test_helm_otlp_client_certs_mount_into_curator_container_for_tracing(tmp_path: Path) -> None:
    """In-process exporters do need the client key in the curator container."""
    pod_spec = _otlp_secret_pod_spec(
        tmp_path,
        extra_values="tracing:\n  otlp:\n    enabled: true\n",
    )

    assert "otlp-cert-store" in _mount_names(pod_spec, "cosmos-curator")
    assert "otlp-cert-store" in _mount_names(pod_spec, "otlp-log-collector")


def test_helm_explicit_otlp_cert_paths_mount_extra_volume_into_log_sidecar(tmp_path: Path) -> None:
    """Explicit OTLP cert paths can be backed by operator-managed pod volumes."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    certPath: /var/run/curator-otlp/tls.crt
    keyPath: /var/run/curator-otlp/tls.key
    caPath: /var/run/curator-otlp/ca.crt
logging:
  otlp:
    enabled: true
extraVolumes:
  - name: cert-manager-otlp-certs
    csi:
      driver: csi.cert-manager.io
      readOnly: true
extraVolumeMounts:
  - name: cert-manager-otlp-certs
    mountPath: /var/run/curator-otlp
    readOnly: true
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/statefulset.yaml",
            "templates/otlp-log-collector-config.yaml",
        ],
    )
    pod_spec = _object(docs, "StatefulSet", "cosmos-curator")["spec"]["template"]["spec"]
    assert "cert-manager-otlp-certs" in _mount_names(pod_spec, "cosmos-curator")
    assert "cert-manager-otlp-certs" in _mount_names(pod_spec, "otlp-log-collector")

    relay = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    assert relay["exporters"]["otlp_http/logs"]["tls"] == {
        "cert_file": "/var/run/curator-otlp/tls.crt",
        "key_file": "/var/run/curator-otlp/tls.key",
        "ca_file": "/var/run/curator-otlp/ca.crt",
        "insecure_skip_verify": False,
    }


def test_helm_shared_otlp_rejects_inherited_signal_path_endpoint(tmp_path: Path) -> None:
    """A legacy full metrics URL must not become the base endpoint for pod-local consumers."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
logging:
  otlp:
    enabled: true
metrics:
  otlp:
    endpoint: https://legacy.example/v1/metrics
""".lstrip()
    )

    with pytest.raises(AssertionError, match="need a base endpoint without /v1/metrics"):
        _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])


def test_helm_shared_otlp_rejects_explicit_signal_path_endpoint(tmp_path: Path) -> None:
    """The same mistake made directly on otlp.endpoint is rejected too."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example/v1/logs
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    with pytest.raises(AssertionError, match="need a base endpoint without /v1/logs"):
        _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])


def test_helm_metrics_collector_still_accepts_full_metrics_endpoint(tmp_path: Path) -> None:
    """Without pod-local consumers the collector keeps taking a full metrics URL."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
metrics:
  enabled: true
  extractNVCFSecrets: false
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
    endpoint: https://legacy.example/v1/metrics
""".lstrip()
    )

    exporter = _collector_metrics_exporter(values_file)
    assert exporter["metrics_endpoint"] == "https://legacy.example/v1/metrics"


def test_helm_shared_otlp_rejects_legacy_env_endpoint_default(tmp_path: Path) -> None:
    """Shared OTLP consumers require a real endpoint, not the legacy collector env default."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    with pytest.raises(AssertionError, match="OTLP logs, traces, or in-process metrics push require"):
        _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])


def test_helm_metrics_otlp_preserves_legacy_env_endpoint_default(tmp_path: Path) -> None:
    """The standalone metrics collector keeps the 2.3 env-substituted endpoint default."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
metrics:
  extractNVCFSecrets: false
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
""".lstrip()
    )

    exporter = _collector_metrics_exporter(values_file)
    assert exporter["metrics_endpoint"] == "${env:OTEL_EXPORTER_OTLP_METRICS_ENDPOINT}"


def test_helm_curator_config_disables_tqdm_by_default_with_override(tmp_path: Path) -> None:
    """Chart pods suppress tqdm progress bars unless custom env overrides it."""
    default_values = tmp_path / "default-values.yaml"
    default_values.write_text("{}\n")
    default_docs = _render_chart(default_values, ["templates/curator-configmap.yaml"])
    default_config = _object(default_docs, "ConfigMap", "curator-config")["data"]
    assert default_config["TQDM_DISABLE"] == "1"

    override_values = tmp_path / "override-values.yaml"
    override_values.write_text(
        """
customEnvVars:
  TQDM_DISABLE: "0"
""".lstrip()
    )
    override_docs = _render_chart(override_values, ["templates/curator-configmap.yaml"])
    override_config = _object(override_docs, "ConfigMap", "curator-config")["data"]
    assert override_config["TQDM_DISABLE"] == "0"


def test_helm_legacy_metrics_otlp_values_feed_shared_otlp_defaults(tmp_path: Path) -> None:
    """Legacy metrics.otlp endpoint remains a fallback when it has no collector-scoped certs."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
logging:
  otlp:
    enabled: true
tracing:
  otlp:
    enabled: true
metrics:
  extractNVCFSecrets: false
  otlp:
    enabled: true
    endpoint: https://legacy-otlp.example
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/curator-configmap.yaml",
            "templates/otlp-log-collector-config.yaml",
            "templates/otel-collector-config.yaml",
        ],
    )

    curator_config = _object(docs, "ConfigMap", "curator-config")["data"]
    assert curator_config["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://legacy-otlp.example"
    assert "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE" not in curator_config
    assert "OTEL_EXPORTER_OTLP_CLIENT_KEY" not in curator_config

    log_collector = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    logs_exporter = log_collector["exporters"]["otlp_http/logs"]
    assert logs_exporter["endpoint"] == "https://legacy-otlp.example"
    assert "tls" not in logs_exporter

    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["metrics_endpoint"] == "https://legacy-otlp.example"
    assert "tls" not in metrics_exporter


def test_helm_shared_otlp_rejects_legacy_metrics_mtls_values(tmp_path: Path) -> None:
    """Shared pod-local OTLP consumers cannot use collector-scoped metrics cert paths."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
logging:
  otlp:
    enabled: true
metrics:
  extractNVCFSecrets: false
  otlp:
    endpoint: https://legacy-otlp.example
    certPath: /legacy/tls.crt
    keyPath: /legacy/tls.key
""".lstrip()
    )

    with pytest.raises(AssertionError, match=r"cannot use legacy metrics\.otlp certPath/keyPath"):
        _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])


def test_helm_otlp_secret_extraction_requires_pod_local_otlp_consumer(tmp_path: Path) -> None:
    """NVCF OTLP extraction should not render orphan pod volumes/initContainers."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  extractNVCFSecrets: true
metrics:
  extractNVCFSecrets: false
  remoteWrite:
    enabled: false
  otlp:
    enabled: false
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        ["templates/statefulset.yaml"],
    )

    pod_spec = _object(docs, "StatefulSet", "cosmos-curator")["spec"]["template"]["spec"]
    assert "initContainers" not in pod_spec
    volume_names = {volume["name"] for volume in pod_spec["volumes"]}
    assert "otlp-secret-extractor-script" not in volume_names
    assert "otlp-cert-store" not in volume_names


def test_helm_top_level_otlp_values_feed_shared_otlp_consumers(tmp_path: Path) -> None:
    """New top-level OTLP values configure logs, traces, and collector metrics."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://shared-otlp.example
  tls:
    certPath: /shared/tls.crt
    keyPath: /shared/tls.key
logging:
  otlp:
    enabled: true
tracing:
  otlp:
    enabled: true
metrics:
  extractNVCFSecrets: false
  otlp:
    enabled: true
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/curator-configmap.yaml",
            "templates/otlp-log-collector-config.yaml",
            "templates/otel-collector-config.yaml",
        ],
    )

    curator_config = _object(docs, "ConfigMap", "curator-config")["data"]
    assert curator_config["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://shared-otlp.example"
    assert curator_config["OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE"] == "/shared/tls.crt"
    assert curator_config["OTEL_EXPORTER_OTLP_CLIENT_KEY"] == "/shared/tls.key"

    log_collector = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    logs_exporter = log_collector["exporters"]["otlp_http/logs"]
    assert logs_exporter["endpoint"] == "https://shared-otlp.example"
    assert logs_exporter["tls"]["cert_file"] == "/shared/tls.crt"
    assert logs_exporter["tls"]["key_file"] == "/shared/tls.key"

    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["endpoint"] == "https://shared-otlp.example"
    assert metrics_exporter["tls"]["cert_file"] == "/shared/tls.crt"
    assert metrics_exporter["tls"]["key_file"] == "/shared/tls.key"


def test_helm_top_level_otlp_tolerates_null_metrics_otlp_endpoint(tmp_path: Path) -> None:
    """A YAML null metrics.otlp.endpoint should not break top-level OTLP defaults."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://shared-otlp.example
metrics:
  extractNVCFSecrets: false
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
    endpoint:
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otel-collector-config.yaml"])
    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["endpoint"] == "https://shared-otlp.example"


def test_helm_top_level_otlp_values_override_legacy_metrics_otlp_defaults(tmp_path: Path) -> None:
    """Top-level OTLP values win over legacy metrics.otlp fallback values."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://shared-otlp.example
  tls:
    certPath: /shared/tls.crt
    keyPath: /shared/tls.key
logging:
  otlp:
    enabled: true
tracing:
  otlp:
    enabled: true
metrics:
  extractNVCFSecrets: false
  otlp:
    enabled: true
    endpoint: https://legacy-otlp.example
    certPath: /legacy/tls.crt
    keyPath: /legacy/tls.key
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/curator-configmap.yaml",
            "templates/otlp-log-collector-config.yaml",
            "templates/otel-collector-config.yaml",
        ],
    )

    curator_config = _object(docs, "ConfigMap", "curator-config")["data"]
    assert curator_config["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://shared-otlp.example"
    assert curator_config["OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE"] == "/shared/tls.crt"
    assert curator_config["OTEL_EXPORTER_OTLP_CLIENT_KEY"] == "/shared/tls.key"

    log_collector = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    logs_exporter = log_collector["exporters"]["otlp_http/logs"]
    assert logs_exporter["endpoint"] == "https://shared-otlp.example"
    assert logs_exporter["tls"]["cert_file"] == "/shared/tls.crt"
    assert logs_exporter["tls"]["key_file"] == "/shared/tls.key"

    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["metrics_endpoint"] == "https://legacy-otlp.example"
    assert metrics_exporter["tls"]["cert_file"] == "/legacy/tls.crt"
    assert metrics_exporter["tls"]["key_file"] == "/legacy/tls.key"


def test_helm_otlp_render_carries_run_attributes_to_traces_and_logs(tmp_path: Path) -> None:
    """Chart values should feed app trace attrs and log collector resource attrs."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
tracing:
  otlp:
    enabled: true
logging:
  otlp:
    enabled: true
    includeMetricsExternalLabels: true
    extraLabels:
      site_owner: test-owner
metrics:
  extractNVCFSecrets: false
  extraExternalLabels:
    backend: test-backend
    function_id: test-function
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/curator-configmap.yaml",
            "templates/otlp-log-collector-config.yaml",
        ],
    )

    curator_config = _object(docs, "ConfigMap", "curator-config")["data"]
    trace_attrs = json.loads(curator_config["COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES"])
    expected_attrs = {
        "backend": "test-backend",
        "function_id": "test-function",
    }
    for key, value in expected_attrs.items():
        assert trace_attrs[key] == value

    log_collector = _object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config")["data"]
    resource_attrs = _resource_attributes(log_collector["relay.yaml"])
    for key, value in expected_attrs.items():
        assert resource_attrs[key] == value
    assert resource_attrs["site_owner"] == "test-owner"


def test_helm_legacy_metrics_insecure_skip_verify_stays_collector_scoped(tmp_path: Path) -> None:
    """Legacy metrics.otlp.insecureSkipVerify must not relax TLS for pod-local consumers."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://shared-otlp.example
logging:
  otlp:
    enabled: true
metrics:
  enabled: true
  extractNVCFSecrets: false
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
    endpoint: https://legacy-otlp.example
    insecureSkipVerify: true
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/otlp-log-collector-config.yaml",
            "templates/otel-collector-config.yaml",
        ],
    )

    log_collector = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    logs_exporter = log_collector["exporters"]["otlp_http/logs"]
    assert logs_exporter.get("tls", {}).get("insecure_skip_verify", False) is False

    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["tls"]["insecure_skip_verify"] is True


def test_helm_shared_insecure_skip_verify_applies_to_every_otlp_consumer(tmp_path: Path) -> None:
    """Shared otlp.tls.insecureSkipVerify relaxes TLS for the sidecar and the collector."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://shared-otlp.example
  tls:
    insecureSkipVerify: true
logging:
  otlp:
    enabled: true
metrics:
  enabled: true
  extractNVCFSecrets: false
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/otlp-log-collector-config.yaml",
            "templates/otel-collector-config.yaml",
        ],
    )

    log_collector = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    assert log_collector["exporters"]["otlp_http/logs"]["tls"]["insecure_skip_verify"] is True

    metrics_exporter = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))["exporters"][
        "otlp_http/metrics"
    ]
    assert metrics_exporter["tls"]["insecure_skip_verify"] is True


def test_helm_remote_write_only_pipeline_keeps_memory_limiter(tmp_path: Path) -> None:
    """Remote write without OTLP still sheds load before batching."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
metrics:
  enabled: true
  remoteWrite:
    enabled: true
    endpoint: https://remote-write.example/api/v1/receive
  otlp:
    enabled: false
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otel-collector-config.yaml"])
    collector = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))
    assert collector["service"]["pipelines"]["metrics/remote-write"]["processors"] == ["memory_limiter", "batch"]


def test_helm_metrics_otlp_does_not_infer_ca_file(tmp_path: Path) -> None:
    """The collector exporter should not infer CA files from secret-key defaults."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
metrics:
  extractNVCFSecrets: false
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
""".lstrip()
    )

    exporter = _collector_metrics_exporter(values_file)
    assert "ca_file" not in exporter.get("tls", {})


def test_helm_metrics_otlp_honors_explicit_shared_ca_path(tmp_path: Path) -> None:
    """The collector exporter should render an explicit shared CA path."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    caPath: /etc/curator-otlp/certs/ca.crt
metrics:
  extractNVCFSecrets: false
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
""".lstrip()
    )

    exporter = _collector_metrics_exporter(values_file)
    assert exporter["tls"]["ca_file"] == "/etc/curator-otlp/certs/ca.crt"


def test_helm_metrics_outputs_keep_independent_collector_config(tmp_path: Path) -> None:
    """Remote write, collector OTLP, and ServiceMonitor are independent metrics outputs.

    This preserves the chart behavior where Prometheus remote write and
    Prometheus Operator scraping keep their original values, while the new OTLP
    metrics exporter can be added without reusing remote-write TLS settings.
    """
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    certPath: /etc/curator-otlp/certs/tls.crt
    keyPath: /etc/curator-otlp/certs/tls.key
    caPath: /etc/curator-otlp/certs/ca.crt
metrics:
  extractNVCFSecrets: false
  enabled: true
  path: /custom-metrics
  remoteWrite:
    enabled: true
    endpoint: https://remote-write.example/api/v1/receive
    certPath: /etc/curator-remote-write/certs/tls.crt
    keyPath: /etc/curator-remote-write/certs/tls.key
  otlp:
    enabled: true
  serviceMonitor:
    enabled: true
    interval: 45s
""".lstrip()
    )

    docs = _render_chart(
        values_file,
        [
            "templates/otel-collector-config.yaml",
            "templates/servicemonitor.yaml",
        ],
    )

    collector = _collector_relay(_object(docs, "ConfigMap", "otel-collector-config"))
    scrape_config = collector["receivers"]["prometheus"]["config"]["scrape_configs"][0]
    assert scrape_config["job_name"] == "ray-service-metrics"
    assert scrape_config["static_configs"] == [{"targets": ["cosmos-curator-0.cosmos-curator-headless:9002"]}]
    assert scrape_config["metric_relabel_configs"][0] == {
        "source_labels": ["__name__"],
        "regex": "ray_tasks",
        "action": "drop",
    }

    exporters = collector["exporters"]
    assert isinstance(exporters, dict)
    remote_write = exporters["prometheus_remote_write"]
    otlp_metrics = exporters["otlp_http/metrics"]
    assert remote_write["endpoint"] == "https://remote-write.example/api/v1/receive"
    assert remote_write["tls"] == {
        "cert_file": "/etc/curator-remote-write/certs/tls.crt",
        "key_file": "/etc/curator-remote-write/certs/tls.key",
    }
    assert remote_write["external_labels"]["namespace"] == "${env:POD_NS}"
    assert otlp_metrics["endpoint"] == "https://otlp.example"
    assert otlp_metrics["tls"] == {
        "cert_file": "/etc/curator-otlp/certs/tls.crt",
        "key_file": "/etc/curator-otlp/certs/tls.key",
        "ca_file": "/etc/curator-otlp/certs/ca.crt",
        "insecure_skip_verify": False,
    }

    pipelines = collector["service"]["pipelines"]
    assert sorted(pipelines) == ["metrics/otlp", "metrics/remote-write"]
    expected_processors = ["memory_limiter", "attributes/external-labels", "batch"]
    assert pipelines["metrics/remote-write"]["processors"] == expected_processors
    assert pipelines["metrics/otlp"]["processors"] == expected_processors
    assert pipelines["metrics/remote-write"]["exporters"] == ["prometheus_remote_write"]
    assert pipelines["metrics/otlp"]["exporters"] == ["otlp_http/metrics"]

    actions = collector["processors"]["attributes/external-labels"]["actions"]
    namespace_actions = [action for action in actions if action["key"] == "namespace"]
    assert namespace_actions == [{"key": "namespace", "value": "${env:POD_NS}", "action": "upsert"}]

    service_monitor = _object(docs, "ServiceMonitor", "cosmos-curator")
    endpoint = service_monitor["spec"]["endpoints"][0]
    assert endpoint["path"] == "/custom-metrics"
    assert endpoint["interval"] == "45s"
    assert endpoint["metricRelabelings"][0] == {
        "sourceLabels": ["__name__"],
        "regex": "ray_tasks",
        "action": "drop",
    }


@pytest.mark.otel_collector
@requires_collector_runtime
@pytest.mark.parametrize(
    ("remote_write_enabled", "metrics_otlp_enabled", "extract_nvcf_secrets"),
    [
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_helm_metrics_collector_config_validates_with_pinned_collector(
    tmp_path: Path,
    *,
    remote_write_enabled: bool,
    metrics_otlp_enabled: bool,
    extract_nvcf_secrets: bool,
) -> None:
    """The rendered metrics collector relay should parse in the chart's collector image."""
    cert_dir = tmp_path / "certs"
    _write_test_tls_files(cert_dir)
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        f"""
otlp:
  endpoint: https://otlp.example
  tls:
    certPath: /certs/tls.crt
    keyPath: /certs/tls.key
    caPath: /certs/ca.crt
metrics:
  enabled: true
  extractNVCFSecrets: {str(extract_nvcf_secrets).lower()}
  remoteWrite:
    enabled: {str(remote_write_enabled).lower()}
    endpoint: https://remote-write.example/api/v1/receive
    certPath: /certs/tls.crt
    keyPath: /certs/tls.key
  otlp:
    enabled: {str(metrics_otlp_enabled).lower()}
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otel-collector-config.yaml"])
    collector_config = _object(docs, "ConfigMap", "otel-collector-config")
    data = collector_config["data"]
    assert isinstance(data, dict)
    config_file = tmp_path / "collector.yaml"
    config_file.write_text(data["relay"])

    _validate_collector_config(config_file, mounts=[(cert_dir, "/certs")])


@pytest.mark.otel_collector
@requires_collector_runtime
def test_helm_log_collector_config_validates_with_pinned_collector(tmp_path: Path) -> None:
    """The rendered log sidecar relay should parse in the chart's collector image."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])
    collector_config = _object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config")
    data = collector_config["data"]
    assert isinstance(data, dict)
    config_file = tmp_path / "collector.yaml"
    config_file.write_text(data["relay.yaml"])
    storage_dir = tmp_path / "file_storage"
    storage_dir.mkdir()

    _validate_collector_config(config_file, mounts=[(storage_dir, "/var/lib/otelcol/file_storage")])


# Rendered pod shape for every combination of the OTLP certificate sources:
# otlp.extractNVCFSecrets, otlp.tls.secret.enabled, and an otlp.tls.caPath
# pointing inside the shared cert directory.
_CERT_SOURCE_CASES = [
    # extract, secret, ca_path, expect_init_container, expect_ca_key_env, expect_cert_store
    pytest.param(False, False, False, False, False, False, id="no-cert-source"),
    pytest.param(False, True, False, False, False, True, id="secret"),
    pytest.param(False, True, True, False, False, True, id="secret-with-ca"),
    pytest.param(True, False, False, True, False, True, id="extract"),
    pytest.param(True, False, True, True, True, True, id="extract-with-ca"),
    pytest.param(True, True, False, True, False, True, id="extract-and-secret"),
    pytest.param(True, True, True, True, True, True, id="extract-and-secret-with-ca"),
]


def _extractor_env(pod_spec: dict[str, object]) -> dict[str, str] | None:
    init_containers = pod_spec.get("initContainers") or []
    for container in init_containers:
        if container["name"] == "otlp-secret-extractor":
            return {item["name"]: item.get("value") for item in container.get("env", [])}
    return None


@pytest.mark.parametrize(
    ("extract", "secret", "ca_path", "expect_init_container", "expect_ca_key_env", "expect_cert_store"),
    _CERT_SOURCE_CASES,
)
def test_helm_otlp_cert_sources_render_expected_pod_shape(
    tmp_path: Path,
    *,
    extract: bool,
    secret: bool,
    ca_path: bool,
    expect_init_container: bool,
    expect_ca_key_env: bool,
    expect_cert_store: bool,
) -> None:
    """Each certificate-source combination renders exactly the plumbing it needs."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        f"""
otlp:
  endpoint: https://otlp.example
  extractNVCFSecrets: {str(extract).lower()}
  tls:
    caPath: "{"/etc/curator-otlp/certs/ca.crt" if ca_path else ""}"
    secret:
      enabled: {str(secret).lower()}
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/statefulset.yaml"])
    pod_spec = _object(docs, "StatefulSet", "cosmos-curator")["spec"]["template"]["spec"]

    extractor_env = _extractor_env(pod_spec)
    assert (extractor_env is not None) is expect_init_container
    # A configured CA key makes the extractor hard-fail when the NVCF secret has
    # no CA, so it is only requested when otlp.tls.caPath gives it a consumer.
    assert (extractor_env is not None and "SECRET_EXTRACTOR_CA_KEY" in extractor_env) is expect_ca_key_env

    volume_names = {volume["name"] for volume in pod_spec["volumes"]}
    assert ("otlp-cert-store" in volume_names) is expect_cert_store
    assert ("otlp-cert-store" in _mount_names(pod_spec, "otlp-log-collector")) is expect_cert_store
    # Logging alone never needs the key in the curator container.
    assert "otlp-cert-store" not in _mount_names(pod_spec, "cosmos-curator")


def test_helm_ca_path_without_client_cert_source_is_rejected(tmp_path: Path) -> None:
    """A CA path alone gives the sidecar no mounted client cert source."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    caPath: /etc/curator-otlp/certs/ca.crt
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    with pytest.raises(AssertionError, match="must include certPath and keyPath"):
        _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])


def test_helm_ca_path_with_secret_source_renders(tmp_path: Path) -> None:
    """A mounted Secret gives the sidecar a real CA file."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    caPath: /etc/curator-otlp/certs/ca.crt
    secret:
      enabled: true
logging:
  otlp:
    enabled: true
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/otlp-log-collector-config.yaml"])
    relay = _log_collector_relay(_object(docs, "ConfigMap", "cosmos-curator-otlp-log-collector-config"))
    assert relay["exporters"]["otlp_http/logs"]["tls"]["ca_file"] == "/etc/curator-otlp/certs/ca.crt"


def test_helm_ca_path_allowed_with_external_log_collector_config(tmp_path: Path) -> None:
    """An operator-supplied collector config owns its own file layout."""
    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
otlp:
  endpoint: https://otlp.example
  tls:
    caPath: /etc/curator-otlp/certs/ca.crt
logging:
  otlp:
    enabled: true
    configMap:
      existingName: my-collector-config
""".lstrip()
    )

    docs = _render_chart(values_file, ["templates/statefulset.yaml"])
    assert _object(docs, "StatefulSet", "cosmos-curator")["kind"] == "StatefulSet"
