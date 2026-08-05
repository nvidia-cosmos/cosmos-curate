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
"""Tests for the private run state a managed Ray run keeps on the login node."""

from pathlib import Path

import pytest

from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    SlurmRayOperationError,
    list_run_manifest_paths,
    open_remote_run,
)
from cosmos_curator.client.slurm_cli.slurm_submit import LocalConnection
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import (
    FakeConnection,
    FakeResult,
    make_active_manifest,
    make_config,
)

REMOTE_MODULE = "cosmos_curator.client.slurm_cli.managed_ray.remote"


def test_home_relative_state_directory_expands_on_the_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default state directory is home-relative and resolves against the cluster account."""
    connection = FakeConnection()
    manifest = make_active_manifest(make_config())

    monkeypatch.setattr(f"{REMOTE_MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{REMOTE_MODULE}._read_remote_manifest", lambda *_args, **_kwargs: manifest)

    with open_remote_run(
        "cc-ray-deadbeef",
        login_node="login",
        username="user",
        state_dir="~/slurm-ray",
    ) as remote_run:
        assert remote_run.manifest_path == Path("/remote/home/slurm-ray/cc-ray-deadbeef/manifest.json")


def test_lifecycle_commands_reject_a_manifest_from_another_slurm_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared storage cannot make cluster-local job IDs actionable through the wrong controller."""

    class OtherClusterConnection(FakeConnection):
        def run(self, command: str, **kwargs: object) -> FakeResult:
            if command == "scontrol show config":
                return FakeResult(stdout="ClusterName = another-cluster\n")
            return super().run(command, **kwargs)

    connection = OtherClusterConnection()
    manifest = make_active_manifest(make_config())
    monkeypatch.setattr(f"{REMOTE_MODULE}.connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(f"{REMOTE_MODULE}._read_remote_manifest", lambda *_args, **_kwargs: manifest)

    with (
        pytest.raises(SlurmRayOperationError, match=r"belongs to Slurm cluster 'test-cluster'.*'another-cluster'"),
        open_remote_run(
            "cc-ray-deadbeef",
            login_node="login",
            username="user",
            state_dir="/state",
        ),
    ):
        pass


def test_damaged_run_state_is_reported_as_a_lifecycle_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest validation raises TypeError, which the client must translate rather than leak as a bug."""
    connection = LocalConnection("localhost", "user")
    connection._context.config.run.in_stream = False
    run_dir = tmp_path / "cc-ray-deadbeef"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"schema_version": 99}', encoding="utf-8")
    monkeypatch.setattr(f"{REMOTE_MODULE}.connect", lambda *_args, **_kwargs: connection)

    with (
        pytest.raises(SlurmRayOperationError, match="Invalid managed Ray run state"),
        open_remote_run("cc-ray-deadbeef", login_node="login", username="user", state_dir=str(tmp_path)),
    ):
        pass


def test_run_listing_enumerates_a_state_directory_against_a_real_shell(tmp_path: Path) -> None:
    """Runs are discovered from the state directory itself, ignoring anything that is not a run."""
    connection = LocalConnection("localhost", "user")
    connection._context.config.run.in_stream = False
    for run_id in ("cc-ray-deadbeef", "cc-ray-00000000"):
        (tmp_path / run_id).mkdir()
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert list_run_manifest_paths(connection, str(tmp_path)) == {
        "cc-ray-deadbeef": tmp_path / "cc-ray-deadbeef" / "manifest.json",
        "cc-ray-00000000": tmp_path / "cc-ray-00000000" / "manifest.json",
    }
    assert list_run_manifest_paths(connection, str(tmp_path / "absent")) == {}
