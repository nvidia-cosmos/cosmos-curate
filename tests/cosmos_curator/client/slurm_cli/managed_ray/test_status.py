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
"""Tests for the combined manifest, Slurm, and Ray view one status command reports."""

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.remote import RemoteManifestRevisionConflictError
from cosmos_curator.client.slurm_cli.managed_ray.status import status_slurm_ray_run
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import (
    FakeConnection,
    make_active_manifest,
    make_config,
    mock_remote_run,
    record_test_mutations,
)

MODULE = "cosmos_curator.client.slurm_cli.managed_ray.status"
REMOTE_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.remote"


def test_status_treats_old_ray_snapshot_as_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slurm remains visible while an old Ray observation is not reported as live."""
    config = make_config()
    manifest = make_active_manifest(config)
    connection = FakeConnection()

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "RUNNING", "restart_count": 0},
            "101": {"state": "RUNNING", "restart_count": 1},
            "102": {"state": "PENDING", "restart_count": 0},
        },
    )
    monkeypatch.setattr(
        f"{MODULE}._read_remote_snapshot",
        lambda *_args, **_kwargs: ({"timestamp": "2026-01-01T00:00:00Z"}, 31.0),
    )
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)

    result = status_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert result["state"] == "ACTIVE"
    assert result["ray"] == {"state": "stale", "age_seconds": 31.0, "address": None}
    assert result["lanes"][0]["restart_count"] == 1  # type: ignore[index]
    assert result["log_dir"] == "/state/cc-ray-deadbeef/logs"


def test_status_recovers_success_after_terminal_manifest_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed head reconciles a persisted successful pipeline outcome."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    manifest["pipeline_exit_status"] = 0
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETED", "restart_count": 0},
            "101": {"state": "CANCELLED", "restart_count": 0},
            "102": {"state": "CANCELLED", "restart_count": 0},
        },
    )
    monkeypatch.setattr(f"{MODULE}._read_remote_snapshot", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)
    writer = record_test_mutations(manifest, snapshots)
    for module in (MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", writer)

    result = status_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert result["state"] == "SUCCEEDED"
    assert result["driver_state"] == "EXITED"
    assert result["pipeline_exit_status"] == 0
    assert snapshots[-1]["state"] == "SUCCEEDED"
    assert snapshots[-1]["error"] is None


def test_status_does_not_finalize_after_its_manifest_snapshot_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduler observations are discarded when another writer advances the manifest revision."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    manifest["pipeline_exit_status"] = 0
    connection = FakeConnection()
    expected_revisions: list[int | None] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETED", "restart_count": 0},
            "101": {"state": "CANCELLED", "restart_count": 0},
            "102": {"state": "CANCELLED", "restart_count": 0},
        },
    )
    monkeypatch.setattr(f"{MODULE}._read_remote_snapshot", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)

    def conflicting_mutation(*_args: object, **kwargs: object) -> dict[str, object]:
        expected_revisions.append(kwargs.get("expected_revision"))  # type: ignore[arg-type]
        message = "injected revision race"
        raise RemoteManifestRevisionConflictError(message)

    monkeypatch.setattr(f"{MODULE}.mutate_remote_manifest", conflicting_mutation)

    result = status_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert expected_revisions == [0]
    assert result["state"] == "STOPPING"


def test_status_waits_for_every_allocation_before_finalizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal head is insufficient while a recorded worker lane remains in Slurm."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "STOPPING"
    manifest["pipeline_exit_status"] = 0
    connection = FakeConnection()
    snapshots: list[dict[str, object]] = []

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETED", "restart_count": 0},
            "101": {"state": "COMPLETING", "restart_count": 0},
            "102": {"state": "CANCELLED", "restart_count": 0},
        },
    )
    monkeypatch.setattr(f"{MODULE}._read_remote_snapshot", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(f"{MODULE}.remote_path_exists", lambda *_args, **_kwargs: False)
    writer = record_test_mutations(manifest, snapshots)
    for module in (MODULE, REMOTE_MODULE):
        monkeypatch.setattr(f"{module}.mutate_remote_manifest", writer)

    result = status_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert result["state"] == "STOPPING"
    assert snapshots == []


def test_status_reports_terminal_ray_as_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed run does not mislabel its final Ray observation as stale."""
    manifest = make_active_manifest(make_config())
    manifest["state"] = "SUCCEEDED"
    manifest["pipeline_exit_status"] = 0
    connection = FakeConnection()

    mock_remote_run(monkeypatch, connection, manifest)
    monkeypatch.setattr(
        f"{MODULE}.query_job_states",
        lambda *_args, **_kwargs: {
            "100": {"state": "COMPLETED", "restart_count": 0},
            "101": {"state": "CANCELLED", "restart_count": 0},
            "102": {"state": "CANCELLED", "restart_count": 0},
        },
    )
    monkeypatch.setattr(
        f"{MODULE}._read_remote_snapshot",
        lambda *_args, **_kwargs: ({"timestamp": "2026-01-01T00:00:00Z"}, 31.0),
    )
    monkeypatch.setattr(
        f"{MODULE}.remote_path_exists",
        lambda _connection, path: path.name == "bootstrap.json",
    )
    monkeypatch.setattr(
        f"{MODULE}.read_remote_json",
        lambda *_args, **_kwargs: {"ray_address": "cpu-0001:6379"},
    )

    result = status_slurm_ray_run("cc-ray-deadbeef", login_node="login", username="user")

    assert result["ray"] == {
        "state": "stopped",
        "age_seconds": 31.0,
        "address": "cpu-0001:6379",
    }
    assert result["driver_state"] == "EXITED"
    assert result["pipeline_exit_status"] == 0
