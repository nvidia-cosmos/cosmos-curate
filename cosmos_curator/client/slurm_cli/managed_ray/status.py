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
"""The combined manifest, Slurm, and Ray view one ``status`` command reports.

Its one write is a reconciliation conditional on the manifest revision it observed; everything that deliberately
changes a run lives in :mod:`.lifecycle`.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from cosmos_curator.client.slurm_cli.managed_ray.config import DEFAULT_STATE_DIR
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import STATUS_INTERVAL_SECONDS
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    BOOTSTRAP_FILENAME,
    LOG_DIR_NAME,
    NONTERMINAL_SLURM_STATES,
    RAY_STATUS_FILENAME,
    TERMINAL_RUN_STATES,
    JsonObject,
    SlurmRayLane,
    SlurmRayManifest,
    run_state,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    RemoteManifestRevisionConflictError,
    RemoteRun,
    SlurmRayOperationError,
    mutate_remote_manifest,
    open_remote_run,
    read_remote_json,
)
from cosmos_curator.client.slurm_cli.managed_ray.scheduler import (
    SlurmJobState,
    query_job_states,
    unknown_job_state,
)
from cosmos_curator.client.slurm_cli.slurm_submit import ConnectionProtocol, remote_path_exists

# Three missed writes: long enough to ride out a slow shared filesystem, short enough that a head that stopped
# publishing is not still reported as live.
_RAY_STATUS_STALE_SECONDS = 3 * STATUS_INTERVAL_SECONDS


class SlurmRayHeadStatus(SlurmJobState):
    """Slurm state for the head allocation, or ``UNKNOWN`` when the manifest records none yet."""

    job_id: str | None


class SlurmRayLaneStatus(SlurmJobState):
    """Slurm state for one lane, keyed by the lane number and array job ID the manifest recorded."""

    lane: int
    job_id: str


class SlurmRayRayStatus(TypedDict):
    """The latest Ray observation, classified by how much of it can still be believed."""

    state: str
    age_seconds: float | None
    address: str | None
    snapshot: NotRequired[JsonObject]


class SlurmRayStatus(TypedDict):
    """The three views ``status`` combines without conflating: manifest, Slurm, and Ray.

    This is the ``--json`` contract.
    """

    run_id: str
    slurm_cluster_name: str
    state: str
    pipeline_exit_status: int | None
    driver_state: str
    error: str | None
    head: SlurmRayHeadStatus
    lane_allocations: int
    lanes: list[SlurmRayLaneStatus]
    ray: SlurmRayRayStatus
    manifest_path: str
    log_dir: str


def status_slurm_ray_run(
    run_id: str,
    *,
    login_node: str,
    username: str | None = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> SlurmRayStatus:
    """Combine manifest, Slurm allocation state, and the latest Ray observation."""
    with open_remote_run(
        run_id,
        login_node=login_node,
        username=username,
        state_dir=state_dir,
    ) as remote_run:
        connection = remote_run.connection
        manifest = remote_run.manifest
        lanes = manifest["lanes"]
        head_job_id = manifest["head_job_id"]
        job_ids = [lane["job_id"] for lane in lanes]
        if head_job_id is not None:
            job_ids.insert(0, head_job_id)
        states = query_job_states(connection, job_ids)

        head_status = states.get(head_job_id, unknown_job_state()) if head_job_id is not None else unknown_job_state()
        manifest = _reconcile_terminal_state(remote_run, head_state=head_status["state"], states=states)
        manifest_state = run_state(manifest)
        run_is_terminal = manifest_state in TERMINAL_RUN_STATES

        directory = remote_run.manifest_path.parent
        snapshot, snapshot_age = _read_remote_snapshot(connection, directory / RAY_STATUS_FILENAME)
        ray_status = _ray_status(
            snapshot,
            snapshot_age,
            address=_ray_address(connection, directory),
            run_is_terminal=run_is_terminal,
        )

        return {
            "run_id": run_id,
            "slurm_cluster_name": manifest["slurm_cluster_name"],
            "state": manifest_state,
            "pipeline_exit_status": manifest["pipeline_exit_status"],
            "driver_state": _driver_state(
                snapshot, ray_is_live=ray_status["state"] == "live", terminal=run_is_terminal
            ),
            "error": manifest["error"],
            "head": {
                "job_id": head_job_id,
                **head_status,
            },
            "lane_allocations": manifest["lane_allocations"],
            "lanes": [
                {"lane": lane["lane"], "job_id": lane["job_id"], **states[lane["job_id"]]}
                for lane in sorted(lanes, key=_lane_number)
            ],
            "ray": ray_status,
            "manifest_path": str(remote_run.manifest_path),
            "log_dir": str(directory / LOG_DIR_NAME),
        }


def _lane_number(lane: SlurmRayLane) -> int:
    return lane["lane"]


def _snapshot_age_seconds(snapshot: JsonObject) -> float | None:
    timestamp = snapshot.get("timestamp")
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - parsed).total_seconds())


def _read_remote_snapshot(
    connection: ConnectionProtocol,
    path: Path,
) -> tuple[JsonObject | None, float | None]:
    if not remote_path_exists(connection, path):
        return None, None
    snapshot = read_remote_json(connection, path)
    return snapshot, _snapshot_age_seconds(snapshot)


def _driver_state(snapshot: JsonObject | None, *, ray_is_live: bool, terminal: bool) -> str:
    """Report the driver state the head last observed, or what the run's own state implies."""
    if ray_is_live and snapshot is not None:
        observed = snapshot.get("driver_state")
        return observed if isinstance(observed, str) else "UNKNOWN"
    return "EXITED" if terminal else "UNKNOWN"


def _reconcile_terminal_state(
    remote_run: RemoteRun,
    *,
    head_state: str,
    states: dict[str, SlurmJobState],
) -> SlurmRayManifest:
    """Finish a stranded transition once Slurm confirms every recorded allocation is gone.

    Conditional on the revision the Slurm query was built from, so a run that changed under it is left for a
    later call to observe coherently.
    """
    manifest = remote_run.manifest
    observed = [head_state, *(states[lane["job_id"]]["state"] for lane in manifest["lanes"])]
    if run_state(manifest) in TERMINAL_RUN_STATES or any(
        state in NONTERMINAL_SLURM_STATES or state == "UNKNOWN" for state in observed
    ):
        return manifest
    try:
        return mutate_remote_manifest(
            remote_run.connection,
            remote_run.manifest_path,
            {
                "operation": "finalize",
                "fallback_error": f"Head job entered terminal Slurm state {head_state}",
            },
            run_id=manifest["run_id"],
            expected_revision=manifest["revision"],
        )
    except RemoteManifestRevisionConflictError:
        return manifest


def _ray_address(connection: ConnectionProtocol, directory: Path) -> str | None:
    bootstrap_path = directory / BOOTSTRAP_FILENAME
    if not remote_path_exists(connection, bootstrap_path):
        return None
    try:
        address = read_remote_json(connection, bootstrap_path).get("ray_address")
    except SlurmRayOperationError:
        return None
    return address if isinstance(address, str) else None


def _ray_status(
    snapshot: JsonObject | None,
    snapshot_age: float | None,
    *,
    address: str | None,
    run_is_terminal: bool,
) -> SlurmRayRayStatus:
    """Classify the latest Ray observation as stopped, unknown, stale, or live."""
    if run_is_terminal:
        return {"state": "stopped", "age_seconds": snapshot_age, "address": address}
    if snapshot is None:
        return {"state": "unknown", "age_seconds": None, "address": address}
    if snapshot_age is None or snapshot_age > _RAY_STATUS_STALE_SECONDS:
        return {"state": "stale", "age_seconds": snapshot_age, "address": address}
    # Only a live observation carries its contents.
    return {"state": "live", "age_seconds": snapshot_age, "address": address, "snapshot": snapshot}
