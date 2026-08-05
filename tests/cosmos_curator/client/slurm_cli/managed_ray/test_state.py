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
"""Tests for the standalone run-state module shipped with managed Slurm-Ray runs."""

from pathlib import Path

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    MANIFEST_SCHEMA_VERSION,
    STATE_MODULE_FILENAME,
    ManifestRevisionConflictError,
    atomic_write_json,
    manifest_lane_job_ids,
    mutate_manifest,
    read_json,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    RemoteManifestRevisionConflictError,
    mutate_remote_manifest,
    upload_state_module,
)
from cosmos_curator.client.slurm_cli.slurm_submit import LocalConnection

RUN_ID = "cc-ray-deadbeef"


def _manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "job_name": "curator",
        "slurm_cluster_name": "test-cluster",
        "state": "ACTIVE",
        "revision": 0,
        "updated_at": "2026-01-01T00:00:00Z",
        "started_at": None,
        "runtime_paths": {"run_dir": "/state/cc-ray-deadbeef"},
        "lanes": [],
        "head_job_id": None,
        "lane_allocations": 1,
        "stop_requested": False,
        "pipeline_exit_status": None,
        "error": None,
    }
    manifest.update(overrides)
    return manifest


def _write_manifest(tmp_path: Path, **overrides: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, _manifest(**overrides))
    return manifest_path


def _local_connection() -> LocalConnection:
    connection = LocalConnection("localhost", "user")
    connection._context.config.run.in_stream = False
    return connection


def test_uploaded_state_module_is_private_and_executable(tmp_path: Path) -> None:
    """The exact standalone source uploaded to a run is private and runnable."""
    upload_state_module(_local_connection(), tmp_path)

    module_path = tmp_path / STATE_MODULE_FILENAME
    assert module_path.is_file()
    assert module_path.stat().st_mode & 0o777 == 0o700


def test_manifest_mutation_is_locked_and_bound_to_one_run(tmp_path: Path) -> None:
    """A mutation reads fresh state under the run lock and refuses to touch another run."""
    manifest_path = _write_manifest(tmp_path)
    lane = {
        "lane": 0,
        "job_id": "101",
        "submitted_at": "2026-01-01T00:00:00Z",
    }

    result = mutate_manifest(
        manifest_path,
        {"operation": "append-lane", "lane": lane},
        run_id=RUN_ID,
    )

    assert result["revision"] == 1
    assert result["lanes"] == [lane]
    assert (tmp_path / "manifest.lock").stat().st_mode & 0o777 == 0o600
    assert manifest_path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(RuntimeError, match="belongs to another run"):
        mutate_manifest(
            manifest_path,
            {"operation": "finish-submission"},
            run_id="cc-ray-00000000",
        )


def test_repeated_mutation_is_idempotent_and_does_not_bump_revision(tmp_path: Path) -> None:
    """Replaying a mutation after a lost response leaves state and revision unchanged."""
    manifest_path = _write_manifest(tmp_path)
    mutation = {"operation": "record-head", "job_id": "100"}

    first = mutate_manifest(manifest_path, mutation, run_id=RUN_ID)
    second = mutate_manifest(manifest_path, mutation, run_id=RUN_ID)

    assert first["head_job_id"] == "100"
    assert second["revision"] == first["revision"]


def test_conditional_mutation_rejects_an_obsolete_manifest_revision(tmp_path: Path) -> None:
    """A status observation cannot finalize state that changed after its scheduler query."""
    manifest_path = _write_manifest(tmp_path)
    changed = mutate_manifest(manifest_path, {"operation": "request-stop"}, run_id=RUN_ID)

    with pytest.raises(ManifestRevisionConflictError, match=r"0 to 1"):
        mutate_manifest(
            manifest_path,
            {"operation": "finalize"},
            run_id=RUN_ID,
            expected_revision=0,
        )

    assert changed["revision"] == 1
    assert read_json(manifest_path)["state"] == "STOPPING"


def test_remote_mutation_wrapper_invokes_the_uploaded_module(tmp_path: Path) -> None:
    """The SSH-facing wrapper sends one command and decodes the resulting state."""
    connection = _local_connection()
    manifest_path = _write_manifest(tmp_path)
    upload_state_module(connection, tmp_path)

    result = mutate_remote_manifest(
        connection,
        manifest_path,
        {"operation": "record-head", "job_id": "100"},
        run_id=RUN_ID,
    )

    assert result["head_job_id"] == "100"
    assert result["revision"] == 1


def test_remote_mutation_reports_a_revision_conflict(tmp_path: Path) -> None:
    """The uploaded writer gives compare-and-set races a distinct client-side error."""
    connection = _local_connection()
    manifest_path = _write_manifest(tmp_path)
    upload_state_module(connection, tmp_path)
    mutate_manifest(manifest_path, {"operation": "request-stop"}, run_id=RUN_ID)

    with pytest.raises(RemoteManifestRevisionConflictError, match="current revision is 1"):
        mutate_remote_manifest(
            connection,
            manifest_path,
            {"operation": "finalize"},
            run_id=RUN_ID,
            expected_revision=0,
        )


def test_finishing_submission_advances_before_any_worker_is_scheduled(tmp_path: Path) -> None:
    """A run whose lanes are all still queued reports STARTING, not SUBMITTING."""
    manifest_path = _write_manifest(tmp_path, state="SUBMITTING")

    submitted = mutate_manifest(manifest_path, {"operation": "finish-submission"}, run_id=RUN_ID)
    assert submitted["state"] == "STARTING"
    assert submitted["started_at"] is None

    # Replaying after a lost response is idempotent and does not regress a run that already started.
    assert (
        mutate_manifest(manifest_path, {"operation": "finish-submission"}, run_id=RUN_ID)["revision"]
        == submitted["revision"]
    )

    started = mutate_manifest(manifest_path, {"operation": "activate"}, run_id=RUN_ID)
    assert started["state"] == "ACTIVE"
    assert started["started_at"] is not None
    assert mutate_manifest(manifest_path, {"operation": "finish-submission"}, run_id=RUN_ID)["state"] == "ACTIVE"


def test_damaged_state_is_rejected_rather_than_indexed_into(tmp_path: Path) -> None:
    """Validation covers every field a caller indexes directly, so readers fail with a message, not a KeyError."""
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, _manifest(runtime_paths={}))

    with pytest.raises(TypeError, match="invalid runtime paths"):
        mutate_manifest(manifest_path, {"operation": "finish-submission"}, run_id=RUN_ID)


def test_finalize_without_an_outcome_falls_back_to_failed(tmp_path: Path) -> None:
    """A head that vanished without recording a result is reconciled as a failure."""
    manifest_path = _write_manifest(tmp_path, state="ACTIVE")

    result = mutate_manifest(
        manifest_path,
        {"operation": "finalize", "fallback_error": "Head job entered terminal Slurm state FAILED"},
        run_id=RUN_ID,
    )

    assert result["state"] == "FAILED"
    assert result["error"] == "Head job entered terminal Slurm state FAILED"


def test_lane_job_listing_is_safe_for_cleanup(tmp_path: Path) -> None:
    """Head cleanup gets only valid IDs out of a manifest that may be damaged."""
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, _manifest(lanes=[{"job_id": "101"}, {"job_id": None}, "invalid"]))
    assert manifest_lane_job_ids(manifest_path) == ["101"]


def test_a_stopping_run_refuses_new_lanes(tmp_path: Path) -> None:
    """Teardown and scale are ordered by the manifest, because neither holds anything across its round trips.

    A scale already in flight can reach the manifest after teardown fenced the run, so the write has to refuse it.
    """
    manifest_path = _write_manifest(tmp_path, state="ACTIVE")
    lane = {"lane": 0, "job_id": "101", "submitted_at": "2026-01-01T00:00:00Z"}
    assert mutate_manifest(manifest_path, {"operation": "append-lane", "lane": lane}, run_id=RUN_ID)["lanes"] == [lane]

    mutate_manifest(manifest_path, {"operation": "request-stop"}, run_id=RUN_ID)

    late = {"lane": 1, "job_id": "102", "submitted_at": "2026-01-01T00:00:00Z"}
    with pytest.raises(RuntimeError, match="Cannot add a worker lane to a run that is STOPPING"):
        mutate_manifest(manifest_path, {"operation": "append-lane", "lane": late}, run_id=RUN_ID)
    assert read_json(manifest_path)["lanes"] == [lane]


def test_cleanup_fences_a_run_without_overwriting_a_recorded_outcome(tmp_path: Path) -> None:
    """Head cleanup fences with one unconditional mutation, so it must leave a normal stop alone."""
    fence = {"operation": "begin-teardown", "error": "head vanished"}

    abrupt = _write_manifest(tmp_path / "abrupt", state="ACTIVE")
    fenced = mutate_manifest(abrupt, fence, run_id=RUN_ID)
    assert fenced["state"] == "STOPPING"
    assert fenced["error"] == "head vanished"

    normal = _write_manifest(tmp_path / "normal", state="ACTIVE")
    mutate_manifest(normal, {"operation": "record-driver-exit", "exit_status": 0}, run_id=RUN_ID)
    unchanged = mutate_manifest(normal, fence, run_id=RUN_ID)
    assert unchanged["error"] is None
    assert mutate_manifest(normal, {"operation": "finalize"}, run_id=RUN_ID)["state"] == "SUCCEEDED"
