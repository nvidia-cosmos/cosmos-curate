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
"""Tests for the cosmos-curator slurm ray command group."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.slurm_cli.managed_ray.lifecycle import (
    SlurmRayPartialOperationError,
    SlurmRayScale,
    SlurmRayStop,
    SlurmRaySubmission,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    SlurmRayOperationError,
    open_remote_run,
)
from cosmos_curator.client.slurm_cli.slurm_submit import LocalConnection

REMOTE_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.remote"
runner = CliRunner()


def test_unknown_run_names_the_path_it_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown run ID and a wrong state directory are indistinguishable, so report where we looked."""
    connection = LocalConnection("localhost", "user")
    connection._context.config.run.in_stream = False
    monkeypatch.setattr(f"{REMOTE_MODULE}.connect", lambda *_args, **_kwargs: connection)

    with (
        pytest.raises(SlurmRayOperationError, match=r"No managed Ray run state at .*--state-dir"),
        open_remote_run(
            "cc-ray-deadbeef",
            login_node="localhost",
            username="user",
            state_dir=str(tmp_path),
        ),
    ):
        pass  # pragma: no cover - the context manager must not open


def test_submit_and_status_human_output_show_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle command output makes the run-scoped log directory discoverable."""
    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(
        yaml.safe_dump({"schema_version": 1, "runtime": {"mount_s3_creds": False}}),
        encoding="utf-8",
    )
    submission = SlurmRaySubmission(
        run_id="cc-ray-deadbeef",
        slurm_cluster_name="test-cluster",
        state_dir="/state",
        manifest_path="/state/cc-ray-deadbeef/manifest.json",
        log_dir="/state/cc-ray-deadbeef/logs",
        head_job_id="100",
        lane_job_ids=["101"],
    )
    submit_kwargs: dict[str, object] = {}

    def fake_submit(*_args: object, **kwargs: object) -> SlurmRaySubmission:
        submit_kwargs.update(kwargs)
        return submission

    monkeypatch.setattr(
        "cosmos_curator.client.slurm_cli.managed_ray.cli.submit_slurm_ray_run",
        fake_submit,
    )

    submit_result = runner.invoke(
        cosmos_curator,
        ["slurm", "ray", "submit", str(config_path), "--state-dir", "/state", "--", "python", "-m", "pipeline"],
    )

    assert submit_result.exit_code == 0
    assert submit_kwargs["state_dir"] == "/state"
    assert "Slurm cluster: test-cluster" in submit_result.stdout
    assert "Logs: /state/cc-ray-deadbeef/logs" in submit_result.stdout

    status_result_payload = {
        "run_id": "cc-ray-deadbeef",
        "slurm_cluster_name": "test-cluster",
        "state": "ACTIVE",
        "head": {"job_id": "100", "state": "RUNNING", "reason": None},
        "driver_state": "RUNNING",
        "lane_allocations": 1,
        "lanes": [
            {
                "lane": 0,
                "job_id": "101",
                "state": "RUNNING",
                "restart_count": 0,
                "reason": None,
                "array_task_id": None,
            }
        ],
        "ray": {"state": "live", "address": "head:6379"},
        "log_dir": "/state/cc-ray-deadbeef/logs",
    }
    monkeypatch.setattr(
        "cosmos_curator.client.slurm_cli.managed_ray.cli.status_slurm_ray_run",
        lambda *_args, **_kwargs: status_result_payload,
    )

    status_result = runner.invoke(cosmos_curator, ["slurm", "ray", "status", "cc-ray-deadbeef"])

    assert status_result.exit_code == 0
    assert "on test-cluster" in status_result.stdout
    assert "Logs: /state/cc-ray-deadbeef/logs" in status_result.stdout


def test_mutation_result_json_flag_changes_the_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scale and stop default to readable output while --json remains structured."""
    scale_result = SlurmRayScale(
        run_id="cc-ray-deadbeef",
        target_worker_lanes=3,
        previous_worker_lanes=2,
        submitted_job_ids=["103"],
        canceled_job_ids=[],
    )
    monkeypatch.setattr(
        "cosmos_curator.client.slurm_cli.managed_ray.cli.scale_slurm_ray_run",
        lambda *_args, **_kwargs: scale_result,
    )

    human = runner.invoke(cosmos_curator, ["slurm", "ray", "scale", "cc-ray-deadbeef", "--workers", "3"])

    assert human.exit_code == 0
    assert "Run cc-ray-deadbeef: 3 worker lanes (previously 2, target 3)" in human.stdout
    assert "Submitted jobs: 103" in human.stdout
    assert "Canceled jobs: none" in human.stdout

    structured = runner.invoke(
        cosmos_curator,
        ["slurm", "ray", "scale", "cc-ray-deadbeef", "--workers", "3", "--json"],
    )

    assert json.loads(structured.stdout) == scale_result.json_payload()

    stop_result = SlurmRayStop(
        run_id="cc-ray-deadbeef",
        state="STOPPED",
        canceled_lane_job_ids=["101", "102"],
        canceled_head_job_ids=["100"],
    )
    monkeypatch.setattr(
        "cosmos_curator.client.slurm_cli.managed_ray.cli.stop_slurm_ray_run",
        lambda *_args, **_kwargs: stop_result,
    )

    human = runner.invoke(cosmos_curator, ["slurm", "ray", "stop", "cc-ray-deadbeef"])

    assert human.exit_code == 0
    assert "Run cc-ray-deadbeef: STOPPED" in human.stdout
    assert "Canceled worker jobs: 101, 102" in human.stdout
    assert "Canceled head jobs: 100" in human.stdout

    structured = runner.invoke(cosmos_curator, ["slurm", "ray", "stop", "cc-ray-deadbeef", "--json"])

    assert json.loads(structured.stdout) == stop_result.json_payload()


def test_launcher_bugs_are_not_reported_as_operator_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only deliberate failures become a one-line message; a launcher bug keeps its traceback."""

    def raise_bug(*_args: object, **_kwargs: object) -> None:
        msg = "unsupported operand type(s)"
        raise TypeError(msg)

    monkeypatch.setattr("cosmos_curator.client.slurm_cli.managed_ray.cli.status_slurm_ray_run", raise_bug)

    result = runner.invoke(cosmos_curator, ["slurm", "ray", "status", "cc-ray-deadbeef"])

    assert isinstance(result.exception, TypeError)


def test_scale_result_reports_what_changed_not_what_was_asked() -> None:
    """A partial reconciliation is described by the jobs it actually moved."""
    partial = SlurmRayScale(
        run_id="cc-ray-deadbeef",
        target_worker_lanes=5,
        previous_worker_lanes=2,
        submitted_job_ids=["103"],
        canceled_job_ids=[],
    )

    assert partial.worker_lanes == 3
    assert partial.json_payload()["worker_lanes"] == 3


def test_scale_cli_includes_partial_result_in_json_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automation can reconcile exact Slurm changes after a nonzero scale result."""
    partial_result: dict[str, object] = {
        "run_id": "cc-ray-deadbeef",
        "target_worker_lanes": 3,
        "previous_worker_lanes": 1,
        "worker_lanes": 2,
        "submitted_job_ids": ["102"],
        "canceled_job_ids": [],
    }

    def fail_scale(*_args: object, **_kwargs: object) -> None:
        msg = "injected partial scale failure"
        raise SlurmRayPartialOperationError(msg, result=partial_result)

    monkeypatch.setattr("cosmos_curator.client.slurm_cli.managed_ray.cli.scale_slurm_ray_run", fail_scale)

    result = runner.invoke(
        cosmos_curator,
        ["slurm", "ray", "scale", "cc-ray-deadbeef", "--workers", "3", "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error"] == "scale_failed"
    assert payload["details"][0]["result"] == partial_result
