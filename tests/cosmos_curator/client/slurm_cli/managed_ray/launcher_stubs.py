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
"""Shared stubs and manifest fixtures for the managed Ray launcher tests."""

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.config import (
    SlurmRayConfig,
    resolve_slurm_ray_config_data,
)
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    MANIFEST_SCHEMA_VERSION,
    MINIMUM_PYTHON_VERSION,
    _apply_manifest_mutation,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    RemoteRun,
)

MODULE = "cosmos_curator.client.slurm_cli.managed_ray.lifecycle"
REMOTE_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.remote"
STATUS_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.status"


class FakeResult:
    """The subset of an invoke ``Result`` the launcher reads back."""

    def __init__(self, *, stdout: str = "", stderr: str = "", ok: bool = True, exited: int | None = None) -> None:
        """Record what one fake command returned."""
        self.stdout = stdout
        self.stderr = stderr
        self.ok = ok
        self.exited = (0 if ok else 1) if exited is None else exited


class FakeConnection:
    """A login node that answers the handful of commands submission issues before it reaches sbatch.

    Every command is recorded, so a test can assert on what the launcher actually asked the cluster to do.
    """

    host = "login"

    def __init__(self) -> None:
        """Start with an empty command log and an open connection."""
        self.commands: list[str] = []
        self.closed = False

    def run(self, command: str, **_kwargs: object) -> FakeResult:
        """Log one command and answer the few that submission cannot proceed without."""
        self.commands.append(command)
        if command == 'printf "%s" "$HOME"':
            return FakeResult(stdout="/remote/home")
        if "sys.version_info" in command:
            # Submission probes the interpreter that will run the modules it uploads.
            return FakeResult(stdout="{}.{}\n".format(*MINIMUM_PYTHON_VERSION))
        if command == "scontrol show config":
            return FakeResult(stdout="ClusterName = test-cluster\n")
        return FakeResult()

    def put(self, _local: str, remote: str) -> None:
        """Accept an upload without writing anything."""
        del remote

    def close(self) -> None:
        """Mark the connection closed so a test can assert it was not leaked."""
        self.closed = True


def apply_test_manifest_mutation(
    manifest: dict[str, object],
    mutation: dict[str, object],
) -> dict[str, object]:
    """Apply one real mutation in-process, bumping the revision the way the remote writer does."""
    before = copy.deepcopy(manifest)
    _apply_manifest_mutation(manifest, mutation)
    if manifest != before:
        manifest["revision"] = int(manifest["revision"]) + 1  # type: ignore[arg-type]
    return manifest


def mock_submission_manifest(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: list[dict[str, object]],
) -> None:
    """Keep a submission's manifest in memory, appending every published revision to ``snapshots``."""
    state: dict[str, dict[str, object]] = {}

    def upload_json(_connection: object, path: Path, value: dict[str, object], **_kwargs: object) -> None:
        if path.name == "manifest.json":
            state["manifest"] = copy.deepcopy(value)

    def fake_mutate(
        _connection: object,
        _path: Path,
        mutation: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        manifest = apply_test_manifest_mutation(state["manifest"], mutation)
        snapshots.append(copy.deepcopy(manifest))
        return manifest

    monkeypatch.setattr(f"{MODULE}.atomic_upload_json", upload_json)
    monkeypatch.setattr(f"{MODULE}.mutate_remote_manifest", fake_mutate)


def record_test_mutations(
    manifest: dict[str, object],
    snapshots: list[dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Return a mutate_remote_manifest stand-in that applies real mutations and records each result."""

    def fake_mutate(
        _connection: object,
        _path: Path,
        mutation: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        result = apply_test_manifest_mutation(manifest, mutation)
        snapshots.append(copy.deepcopy(result))
        return result

    return fake_mutate


def make_config(*, worker_lanes: int = 2, environment: list[str] | None = None) -> SlurmRayConfig:
    """Build a resolved config with the fields most tests need already filled in."""
    return resolve_slurm_ray_config_data(
        {
            "schema_version": 1,
            "job_name": "curator",
            "worker_lanes": worker_lanes,
            "slurm": {
                "account": "acct",
                "head": {"partition": "cpu", "qos": "normal", "time": "1-00:00:00"},
                "worker": {
                    "partition": "gpu",
                    "qos": "backfill",
                    "time": "2-00:00:00",
                    "gpus": 8,
                },
            },
            "runtime": {
                "mount_s3_creds": False,
                "mounts": [{"source": "/shared/data", "destination": "/data", "mode": "ro"}],
                "environment": environment or [],
            },
            "ray": {"startup_timeout": "5m"},
        }
    )


def make_active_manifest(config: SlurmRayConfig) -> dict[str, object]:
    """Build an active manifest for a run that has a head and two recorded lanes."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "cc-ray-deadbeef",
        "job_name": "curator",
        "slurm_cluster_name": "test-cluster",
        "state": "ACTIVE",
        "revision": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:00Z",
        "stop_requested": False,
        "pipeline_exit_status": None,
        "error": None,
        "command": ["python", "-m", "pipeline"],
        "config": config.model_dump(mode="json"),
        "runtime_paths": {
            "run_dir": "/state/cc-ray-deadbeef",
            "container_image": "/images/curator.sqsh",
            "mounts": [
                {
                    "source": "/state/cc-ray-deadbeef",
                    "destination": "/run/cosmos-curator/slurm-ray",
                    "mode": "rw",
                }
            ],
            "prepare_directories": [],
            "forwarded_environment_keys": [],
            "ray_temp_dir": None,
        },
        "head_job_id": "100",
        "lane_allocations": 1,
        "lanes": [
            {
                "lane": 0,
                "job_id": "101",
                "submitted_at": "2026-01-01T00:00:00Z",
            },
            {
                "lane": 1,
                "job_id": "102",
                "submitted_at": "2026-01-01T00:00:00Z",
            },
        ],
    }


def mock_remote_run(
    monkeypatch: pytest.MonkeyPatch,
    connection: FakeConnection,
    manifest: dict[str, object],
) -> None:
    """Point the lifecycle and status modules at one in-memory manifest reached over a fake connection."""
    manifest_path = Path("/state/cc-ray-deadbeef/manifest.json")

    @contextmanager
    def open_remote_run(*_args: object, **_kwargs: object) -> Iterator[RemoteRun]:
        try:
            yield RemoteRun(
                connection=connection,
                manifest_path=manifest_path,
                manifest=manifest,  # type: ignore[arg-type]
            )
        finally:
            connection.close()

    monkeypatch.setattr(f"{MODULE}.open_remote_run", open_remote_run)
    monkeypatch.setattr(f"{STATUS_MODULE}.open_remote_run", open_remote_run)

    def fake_mutate(
        _connection: object,
        _path: Path,
        mutation: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        return apply_test_manifest_mutation(manifest, mutation)

    # Both the module-level callers and ``RemoteRun.mutate``, which resolves it in the remote module.
    for module in (MODULE, STATUS_MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", fake_mutate)
