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
"""Container entrypoints for a managed Ray head or worker Slurm job.

The ``cleanup`` and ``probe-head`` roles run outside the container on a compute node, so this module is held to
the same Python floor as its peers. See :mod:`.` for what that floor is and why.
"""

# Deferred annotations keep PEP 585 and PEP 604 syntax out of the runtime; see the package docstring.
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

if TYPE_CHECKING or __package__:
    from cosmos_curator.client.slurm_cli.managed_ray.onnode import slurm_ray_state as _state
else:
    # Executed as a plain script from a run directory, where the state module is a flat sibling file.
    import slurm_ray_state as _state  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

RUNTIME_MODULE_FILENAME = "slurm_ray_runtime.py"
WORKER_HEAD_JOB_ID_ENV = "COSMOS_CURATOR_SLURM_RAY_HEAD_JOB_ID"
WORKER_LANE_ENV = "COSMOS_CURATOR_SLURM_RAY_LANE"

# Fixed ports for a worker, which owns its node exclusively and so can keep predictable ones. The head shares a
# node with whatever else the CPU partition scheduled there, including another run's head, so it reserves free
# ports at startup instead and publishes them in the bootstrap record.
_OBJECT_MANAGER_PORT = 8076
_NODE_MANAGER_PORT = 8077
_DASHBOARD_AGENT_GRPC_PORT = 52366
_RUNTIME_ENV_AGENT_PORT = 20267
_METRICS_EXPORT_PORT = 9002
# This module also runs as a standalone Python 3.8 script, so these mirror the
# shared Curator resource contract instead of importing the package there.
_CURATOR_IO_RESOURCE_NAME = "curator_io"
_DEFAULT_IO_SLOTS_PER_NODE = 16
# Every port ``ray start --head`` pre-selects, all of them named by the head rather than left to a Ray default.
# Ray refuses to start when two pre-selected ports collide, and three of its defaults are fixed values a shared
# node cannot use: the client server at 10001, the dashboard agent's HTTP listener at 52365, and the whole
# 10002-19999 worker range. The worker range is the one that bites even with a single head, because a node whose
# ``ip_local_port_range`` starts below 20000 hands out reserved ports inside it; the head asks for random worker
# ports instead, which costs nothing because it runs no tasks.
_HEAD_PORT_NAMES = (
    "gcs",
    "dashboard",
    "object_manager",
    "node_manager",
    "dashboard_agent_grpc",
    "dashboard_agent_http",
    "runtime_env_agent",
    "metrics_export",
    "ray_client_server",
)
_RANDOM_WORKER_PORTS = "0"
# How often the head publishes a Ray observation. The client derives its staleness threshold from this.
STATUS_INTERVAL_SECONDS = 10
_POLL_INTERVAL_SECONDS = 1
_TERMINATED_STATUS = 143
# How long head cleanup waits for canceled lanes to leave the queue. Exclusive accelerator nodes can sit in
# COMPLETING for a while; if they outlast this, the run stays STOPPING until a later status reconciles it.
_LANE_DRAIN_TIMEOUT_SECONDS = 180
# How hard a starting lane tries to reach slurmctld before it gives up on confirming its head. Long enough to
# outlast a controller busy under a scheduling storm, far short of the walltime it would otherwise waste.
_HEAD_PROBE_ATTEMPTS = 5
_HEAD_PROBE_RETRY_SECONDS = 15


def _configure_logging() -> None:
    """Use Curator's structured logging when available, with a standalone fallback."""
    try:
        from cosmos_curator.core.utils.misc.json_logging import configure_stdlib_logging  # noqa: PLC0415
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        configure_stdlib_logging(text_format="%(asctime)s - %(levelname)s - %(message)s")


def _reserve_head_ports() -> dict[str, int]:
    """Reserve one free TCP port per head service.

    The head shares its node, so it cannot assume the well-known Ray ports are free: a second run's head may
    already have them. Every socket is held open until all of them are assigned, which is what keeps the ports
    distinct from each other. A port lost between closing the socket and Ray binding fails the head at startup.
    """
    sockets = []
    try:
        for _ in _HEAD_PORT_NAMES:
            reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reserved.bind(("", 0))
            sockets.append(reserved)
        return dict(zip(_HEAD_PORT_NAMES, (reserved.getsockname()[1] for reserved in sockets)))
    finally:
        for reserved in sockets:
            reserved.close()


def _ray_head_command(head_node: str, ports: dict[str, int], temp_dir: Path | None) -> list[str]:
    command = [
        "ray",
        "start",
        "--head",
        "--block",
        "--node-ip-address",
        head_node,
        "--port",
        str(ports["gcs"]),
        "--num-cpus",
        "0",
        "--num-gpus",
        "0",
        "--object-manager-port",
        str(ports["object_manager"]),
        "--node-manager-port",
        str(ports["node_manager"]),
        "--runtime-env-agent-port",
        str(ports["runtime_env_agent"]),
        "--metrics-export-port",
        str(ports["metrics_export"]),
        "--dashboard-host",
        "127.0.0.1",
        "--dashboard-port",
        str(ports["dashboard"]),
        "--dashboard-agent-grpc-port",
        str(ports["dashboard_agent_grpc"]),
        "--dashboard-agent-listen-port",
        str(ports["dashboard_agent_http"]),
        "--ray-client-server-port",
        str(ports["ray_client_server"]),
        "--min-worker-port",
        _RANDOM_WORKER_PORTS,
        "--max-worker-port",
        _RANDOM_WORKER_PORTS,
        "--disable-usage-stats",
    ]
    if temp_dir is not None:
        command.extend(["--temp-dir", str(temp_dir)])
    return command


def _ray_worker_command(
    address: str,
    temp_dir: Path | None,
    io_slots_per_node: int = _DEFAULT_IO_SLOTS_PER_NODE,
) -> list[str]:
    command = [
        "ray",
        "start",
        "--block",
        "--address",
        address,
        "--node-ip-address",
        socket.gethostname(),
        "--object-manager-port",
        str(_OBJECT_MANAGER_PORT),
        "--node-manager-port",
        str(_NODE_MANAGER_PORT),
        "--runtime-env-agent-port",
        str(_RUNTIME_ENV_AGENT_PORT),
        "--metrics-export-port",
        str(_METRICS_EXPORT_PORT),
        "--dashboard-agent-grpc-port",
        str(_DASHBOARD_AGENT_GRPC_PORT),
        "--disable-usage-stats",
        "--resources",
        json.dumps({_CURATOR_IO_RESOURCE_NAME: io_slots_per_node}),
    ]
    if temp_dir is not None:
        command.extend(["--temp-dir", str(temp_dir)])
    return command


class _FatalStartupError(Exception):
    """Raised by a startup wait that cannot succeed, so :func:`_poll_for` must not retry it.

    Deliberately not a :class:`RuntimeError`: the waits below tolerate those while a resource comes up.
    """


def _poll_for(
    attempt: Callable[[], _T | None],
    *,
    timeout_seconds: int,
    description: str,
    tolerate: tuple[type[Exception], ...] = (),
) -> _T:
    """Call ``attempt`` until it returns a value, treating ``None`` and ``tolerate`` as not-ready-yet.

    Anything else propagates, which is how a wait says it can never succeed rather than burning its timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = attempt()
        except tolerate as exc:
            last_error = exc
        else:
            if value is not None:
                return value
        time.sleep(_POLL_INTERVAL_SECONDS)

    detail = f": {last_error}" if last_error is not None else ""
    msg = f"Timed out waiting for {description} after {timeout_seconds} seconds{detail}"
    raise TimeoutError(msg)


def _wait_for_submitted_manifest(manifest_path: Path, run_id: str, timeout_seconds: int) -> _state.SlurmRayManifest:
    """Wait for submission to finish recording every Slurm job it created."""

    def submitted_manifest() -> _state.SlurmRayManifest | None:
        manifest = _state.read_manifest(manifest_path)
        if manifest["run_id"] != run_id:
            msg = f"Manifest run ID does not match {run_id!r}"
            raise _FatalStartupError(msg)
        if _state.run_state(manifest) == "SUBMITTING":
            return None
        if _state.run_state(manifest) != "STARTING":
            msg = f"Run is already {manifest['state']}; refusing to start Ray"
            raise _FatalStartupError(msg)
        return manifest

    return _poll_for(
        submitted_manifest,
        timeout_seconds=timeout_seconds,
        description="submission to complete",
        tolerate=(OSError, ValueError, TypeError),
    )


def _mutate(manifest_path: Path, run_id: str, mutation: _state.JsonObject) -> _state.SlurmRayManifest:
    """Apply one named manifest mutation under the run lock."""
    return _state.mutate_manifest(manifest_path, mutation, run_id=run_id)


def _prepare_ray_temp_dir(root: str | None, role: str) -> Path | None:
    if root is None:
        return None
    job_id = os.getenv("SLURM_JOB_ID", "unknown")
    restart_count = os.getenv("SLURM_RESTART_COUNT", "0")
    # The Slurm allocation and restart identify one Ray node while Ray's own ``session_<timestamp>_<pid>``
    # directory distinguishes starts within it. Omitting the longer run ID keeps AF_UNIX socket paths below
    # Linux's 107-byte limit without making cleanup shared between active Ray nodes.
    path = Path(root).expanduser() / f"{role}-{job_id}-{restart_count}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _connect_to_ray(address: str, head_process: subprocess.Popen[bytes], timeout_seconds: int) -> Any:  # noqa: ANN401
    import ray  # noqa: PLC0415

    host, _, port = address.rpartition(":")

    def connected() -> Any | None:  # noqa: ANN401
        if head_process.poll() is not None:
            msg = f"Ray head exited during startup with status {head_process.returncode}"
            raise _FatalStartupError(msg)
        socket.create_connection((host, int(port)), timeout=1).close()
        ray.init(address=address, logging_level=logging.WARNING)
        return ray

    return _poll_for(
        connected,
        timeout_seconds=timeout_seconds,
        description="the Ray head",
        tolerate=(ConnectionError, OSError, RuntimeError, ValueError),
    )


def _live_ray_nodes(ray_module: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    return [node for node in ray_module.nodes() if node.get("Alive") is True]


def _wait_for_first_worker(
    ray_module: Any,  # noqa: ANN401
    *,
    head_process: subprocess.Popen[bytes],
    stop_signal: threading.Event,
) -> None:
    while not stop_signal.is_set():
        if head_process.poll() is not None:
            msg = f"Ray head exited while waiting for a worker with status {head_process.returncode}"
            raise RuntimeError(msg)
        for node in _live_ray_nodes(ray_module):
            resources = node.get("Resources", {})
            if isinstance(resources, dict) and (
                float(resources.get("CPU", 0.0)) > 0.0 or float(resources.get("GPU", 0.0)) > 0.0
            ):
                return
        stop_signal.wait(_POLL_INTERVAL_SECONDS)


class _StatusWriter:
    """Periodically persist a timestamped observation of Ray state.

    The supervisor assigns :attr:`driver_state` as the run progresses; each snapshot reports the latest value.
    """

    def __init__(
        self,
        *,
        ray_module: Any,  # noqa: ANN401
        status_path: Path,
        head_node: str,
    ) -> None:
        self.driver_state = "WAITING_FOR_WORKER"
        self._ray = ray_module
        self._status_path = status_path
        self._head_node = head_node
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="slurm-ray-status", daemon=True)

    def start(self) -> None:
        """Start the background writer."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the writer and wait briefly for it. Safe to call more than once."""
        self._stop.set()
        self._thread.join(timeout=2)

    def _snapshot(self) -> _state.JsonObject:
        nodes = _live_ray_nodes(self._ray)
        return {
            "timestamp": _state.utc_now(),
            "driver_state": self.driver_state,
            "head_node": self._head_node,
            "live_ray_node_ids": [str(node.get("NodeID", "")) for node in nodes],
            "total_resources": self._ray.cluster_resources(),
            "available_resources": self._ray.available_resources(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                _state.atomic_write_json(self._status_path, self._snapshot())
            except Exception:
                logger.exception("Failed to write Ray status snapshot")
            self._stop.wait(STATUS_INTERVAL_SECONDS)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _stop_ray(ray_module: Any, head_process: subprocess.Popen[bytes]) -> None:  # noqa: ANN401
    try:
        ray_module.shutdown()
    except Exception:
        logger.exception("Failed to disconnect the supervisor from Ray")
    try:
        subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)  # noqa: S607
    except subprocess.TimeoutExpired:
        logger.exception("Timed out while stopping local Ray processes")
    _stop_process(head_process)


def _run_head(args: argparse.Namespace) -> int:
    """Supervise one managed Ray head and its pipeline driver."""
    manifest_path = Path(args.manifest)
    try:
        with ExitStack() as teardown:
            status = _supervise_head(args, manifest_path, teardown)
    except Exception as exc:
        logger.exception("Managed Ray head failed")
        with suppress(Exception):
            _mutate(manifest_path, args.run_id, {"operation": "record-failure", "error": str(exc)})
        return 1
    else:
        return status


def _supervise_head(args: argparse.Namespace, manifest_path: Path, teardown: ExitStack) -> int:
    """Start Ray, run the driver once a worker joins, and persist its outcome.

    Head resources are registered with ``teardown`` as they are created. The manifest stays ``STOPPING`` until those
    resources have been released and the outer batch wrapper has canceled the worker lanes.
    """
    head_node = socket.gethostname()
    ports = _reserve_head_ports()
    address = "{}:{}".format(head_node, ports["gcs"])
    stop_signal = _install_stop_handlers()

    manifest = _wait_for_submitted_manifest(manifest_path, args.run_id, args.startup_timeout_seconds)
    head_job_id = _verified_head_job_id(manifest)

    temp_dir = _prepare_ray_temp_dir(args.temp_dir, "head")
    if temp_dir is not None:
        teardown.callback(shutil.rmtree, temp_dir, ignore_errors=True)

    logger.info("Starting Ray head at %s", address)
    head_process = subprocess.Popen(_ray_head_command(head_node, ports, temp_dir))  # noqa: S603
    teardown.callback(_stop_process, head_process)
    ray_module = _connect_to_ray(address, head_process, args.startup_timeout_seconds)
    teardown.callback(_stop_ray, ray_module, head_process)
    _state.atomic_write_json(
        manifest_path.parent / _state.BOOTSTRAP_FILENAME,
        _bootstrap_record(args.run_id, head_job_id, address, ports),
    )

    status_writer = _StatusWriter(
        ray_module=ray_module,
        status_path=manifest_path.parent / _state.RAY_STATUS_FILENAME,
        head_node=head_node,
    )
    status_writer.start()
    teardown.callback(status_writer.stop)

    _wait_for_first_worker(ray_module, head_process=head_process, stop_signal=stop_signal)
    if stop_signal.is_set():
        return _TERMINATED_STATUS
    if _state.run_state(_mutate(manifest_path, args.run_id, {"operation": "activate"})) != "ACTIVE":
        logger.info("Run left STARTING before driver start; stopping the head")
        return _TERMINATED_STATUS

    status_writer.driver_state = "RUNNING"
    driver_status, head_exited = _run_driver(args.command, address, head_process, stop_signal)
    status_writer.driver_state = "STOPPING"

    if head_exited:
        _mutate(
            manifest_path,
            args.run_id,
            {
                "operation": "record-failure",
                "exit_status": driver_status,
                "error": f"Ray head exited unexpectedly with status {head_process.returncode}",
            },
        )
        return 1

    _mutate(manifest_path, args.run_id, {"operation": "record-driver-exit", "exit_status": driver_status})
    return driver_status


def _install_stop_handlers() -> threading.Event:
    """Convert termination signals into a stop event the supervisor polls."""
    stop_signal = threading.Event()

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("Received signal %s", signum)
        stop_signal.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    return stop_signal


def _verified_head_job_id(manifest: _state.SlurmRayManifest) -> str:
    """Confirm this allocation is the head job the manifest recorded."""
    expected = manifest["head_job_id"]
    if expected is None:
        msg = "Sealed manifest does not record a head job ID"
        raise RuntimeError(msg)
    actual = os.getenv("SLURM_JOB_ID")
    if actual is not None and expected != actual:
        msg = f"Head job ID mismatch: expected {expected}, running as {actual}"
        raise RuntimeError(msg)
    return expected


def _bootstrap_record(run_id: str, head_job_id: str, address: str, ports: dict[str, int]) -> _state.JsonObject:
    """Record what a worker needs to join, and what an operator needs to reach the head it joined."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "head_job_id": head_job_id,
        "ray_address": address,
        "ports": dict(ports),
        "created_at": _state.utc_now(),
    }


def _run_driver(
    command: list[str],
    address: str,
    head_process: subprocess.Popen[bytes],
    stop_signal: threading.Event,
) -> tuple[int, bool]:
    """Run the pipeline command to completion, returning its status and whether Ray died under it."""
    environment = os.environ.copy()
    environment["RAY_ADDRESS"] = address
    logger.info("Starting pipeline command: %s", command)
    driver_process = subprocess.Popen(command, env=environment)  # noqa: S603

    head_exited = False
    try:
        while driver_process.poll() is None:
            if head_process.poll() is not None:
                head_exited = True
                break
            if stop_signal.wait(_POLL_INTERVAL_SECONDS):
                break
    finally:
        _stop_process(driver_process)
    return driver_process.wait(), head_exited


def _wait_for_bootstrap(path: Path, run_id: str, head_job_id: str, timeout_seconds: int) -> _state.JsonObject:
    def matching_bootstrap() -> _state.JsonObject | None:
        bootstrap = _state.read_json(path)
        matches = bootstrap.get("run_id") == run_id and str(bootstrap.get("head_job_id")) == head_job_id
        return bootstrap if matches else None

    return _poll_for(
        matching_bootstrap,
        timeout_seconds=timeout_seconds,
        description="a matching Ray bootstrap record",
        tolerate=(OSError, ValueError, TypeError),
    )


def _run_worker(args: argparse.Namespace) -> int:
    bootstrap = _wait_for_bootstrap(
        Path(args.bootstrap),
        args.run_id,
        args.head_job_id,
        args.startup_timeout_seconds,
    )
    address = bootstrap.get("ray_address")
    if not isinstance(address, str) or not address:
        msg = "Bootstrap record does not contain a Ray address"
        raise RuntimeError(msg)

    temp_dir = _prepare_ray_temp_dir(args.temp_dir, f"lane-{args.lane}")
    try:
        logger.info(
            "Starting lane %s incarnation %s for run %s against %s",
            args.lane,
            os.getenv("SLURM_RESTART_COUNT", "0"),
            args.run_id,
            address,
        )
        completed = subprocess.run(  # noqa: S603
            _ray_worker_command(address, temp_dir, args.io_slots_per_node),
            check=False,
        )
        return completed.returncode
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _queued_job_states() -> dict[str, str] | None:
    """Return the caller's queued jobs and their states, or ``None`` when squeue could not be reached.

    Keeping "the scheduler did not answer" distinct from "the job is not queued" is the point of the ``None``:
    both are the same nonzero exit from a direct job query, and only an answer the scheduler actually gave may
    condemn a run.

    Array tasks are collapsed onto their array job so that a lane's queued tail (``<id>_[3-41%1]``) still matches
    the plain array ID the manifest recorded; without that a renewing lane would never appear to drain. A lane
    with several reported tasks takes the state of its liveliest one, ranked by :func:`_state.slurm_state_rank`.
    """
    try:
        # ``getpass`` covers a batch environment that exports LOGNAME but not USER; it can fail outright on a
        # host with no passwd entry, which is one more way to have received no answer.
        user = os.environ.get("USER") or getpass.getuser()
        result = subprocess.run(  # noqa: S603
            ["squeue", "--noheader", "--user", user, "--format=%i|%T"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, KeyError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        reported_id, separator, state = line.strip().partition("|")
        if not separator:
            continue
        job_id = _state.base_job_id(reported_id)
        normalized = _state.normalize_slurm_state(state)
        if job_id not in states or _state.slurm_state_rank(normalized) > _state.slurm_state_rank(states[job_id]):
            states[job_id] = normalized
    return states


def _wait_for_lanes_to_drain(job_ids: list[str], timeout_seconds: int) -> bool:
    """Wait until Slurm no longer queues any recorded lane, tolerating transient query failures."""
    remaining = set(job_ids)
    deadline = time.monotonic() + timeout_seconds
    while remaining and time.monotonic() < deadline:
        queued = _queued_job_states()
        if queued is not None:
            remaining &= set(queued)
        if remaining:
            time.sleep(_POLL_INTERVAL_SECONDS)
    return not remaining


def _run_probe_head(args: argparse.Namespace) -> int:
    """Refuse to let a lane join a run whose head the scheduler says is already gone.

    A lane runs this before it starts anything, so a run that has ended does not collect workers that will only
    wait out their walltime. An unreachable controller is retried rather than treated as an answer.
    """
    for attempt in range(1, _HEAD_PROBE_ATTEMPTS + 1):
        queued = _queued_job_states()
        if queued is not None:
            state = queued.get(args.job_id)
            if state in _state.ACTIVE_HEAD_SLURM_STATES:
                return 0
            logger.error("Head job %s is %s; refusing to join stale run", args.job_id, state or "not queued")
            return 1
        logger.warning("Slurm did not answer on attempt %d; retrying", attempt)
        if attempt < _HEAD_PROBE_ATTEMPTS:
            time.sleep(_HEAD_PROBE_RETRY_SECONDS)

    logger.error("Slurm never answered; refusing to join without confirming the head job is active")
    return 1


def _run_cleanup(args: argparse.Namespace) -> int:
    """Cancel every recorded lane and publish the run's terminal state once Slurm has released them.

    The head's batch wrapper runs this outside the container as its exit trap, so it still runs when the
    supervisor itself was killed. Leaving the run ``STOPPING`` is safe: a later ``status`` reconciles it.
    """
    manifest_path = Path(args.manifest)
    outcome = "Head job exited before recording a terminal outcome"

    # Fence the run into STOPPING before taking the lane snapshot, so a scale still in flight is refused rather
    # than recording a lane behind the set about to be canceled.
    with suppress(OSError, TypeError, ValueError, RuntimeError):
        _mutate(manifest_path, args.run_id, {"operation": "begin-teardown", "error": outcome})

    lane_job_ids = _state.manifest_lane_job_ids(manifest_path)
    if lane_job_ids:
        logger.info("Canceling %d recorded worker lanes", len(lane_job_ids))
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603
                ["scancel", "--quiet", *lane_job_ids],  # noqa: S607
                check=False,
                timeout=60,
            )

    if not _wait_for_lanes_to_drain(lane_job_ids, _LANE_DRAIN_TIMEOUT_SECONDS):
        logger.error("Worker lanes have not reached terminal Slurm state; leaving run STOPPING")
        return 1

    _mutate(manifest_path, args.run_id, {"operation": "finalize", "fallback_error": outcome})
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = "--io-slots-per-node must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    head = subparsers.add_parser("head")
    head.add_argument("--run-id", required=True)
    head.add_argument("--manifest", required=True)
    head.add_argument("--startup-timeout-seconds", required=True, type=int)
    head.add_argument("--temp-dir")
    head.add_argument("command", nargs=argparse.REMAINDER)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--head-job-id", default=os.getenv(WORKER_HEAD_JOB_ID_ENV))
    worker.add_argument("--lane", type=int, default=os.getenv(WORKER_LANE_ENV))
    worker.add_argument("--bootstrap", required=True)
    worker.add_argument("--startup-timeout-seconds", required=True, type=int)
    worker.add_argument("--io-slots-per-node", type=_positive_int, default=_DEFAULT_IO_SLOTS_PER_NODE)
    worker.add_argument("--temp-dir")

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--manifest", required=True)

    probe_head = subparsers.add_parser("probe-head")
    probe_head.add_argument("--job-id", required=True)
    return parser


def main() -> None:
    """Run the selected managed cluster role."""
    _configure_logging()
    args = _parser().parse_args()
    if args.role == "head":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            msg = "The head role requires a pipeline command after '--'"
            raise ValueError(msg)
        status = _run_head(args)
    elif args.role == "cleanup":
        status = _run_cleanup(args)
    elif args.role == "probe-head":
        status = _run_probe_head(args)
    else:
        if args.head_job_id is None or args.lane is None:
            msg = f"The worker role requires {WORKER_HEAD_JOB_ID_ENV} and {WORKER_LANE_ENV}"
            raise ValueError(msg)
        status = _run_worker(args)
    sys.exit(status)


if __name__ == "__main__":
    main()
