# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test the slurm module."""

import pathlib
import unittest
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import Mock, patch

import invoke
import pytest

from cosmos_curate.client.slurm_cli.slurm import (
    _START_RAY,
    ContainerSpec,
    MountSpec,
    SlurmJobSpec,
    _get_username,
    _parse_job_id,
    _render_sbatch_script,
    connect,
    curator_submit,
    submit_cli,
    upload_text,
)
from cosmos_curate.scripts.onto_slurm import SlurmEnv

MODULE_NAME = "cosmos_curate.client.slurm_cli.slurm"
GRES = "gpu:8"


@pytest.mark.parametrize(
    ("command", "raises"),
    [
        (["echo", "test"], nullcontext()),
        ([], pytest.raises(ValueError, match="A command must be provided")),
    ],
)
@patch(f"{MODULE_NAME}.curator_submit")
def test_submit_cmd(mock_curator_submit: Mock, command: list[str], raises: AbstractContextManager[Any]) -> None:
    """Test that the launch command executes without errors."""
    with raises:
        submit_cli(
            command=command,
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            num_nodes=1,
            container_mounts=None,  # default
            environment=None,  # default
            remote_files_path=pathlib.Path("/remote/files"),
        )

    if isinstance(raises, nullcontext):
        mock_curator_submit.assert_called_once()
    else:
        mock_curator_submit.assert_not_called()


@pytest.mark.parametrize(
    ("exclude_nodes"),
    [
        (None),
        (["node1", "node2"]),
    ],
)
def test_render_sbatch_script(exclude_nodes: list[str] | None) -> None:
    """Test that the render sbatch script function returns the correct sbatch script."""
    job_spec = SlurmJobSpec(
        login_node="login_node",
        container=ContainerSpec(
            squashfs_path="test_path", command=[str(_START_RAY), "arg1", "arg2"], mounts=[], environment=[]
        ),
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        username="test_user",
        num_nodes=1,
        gres=GRES,
        exclusive=True,
        remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
        time_limit="01:00:00",
        log_dir=pathlib.Path("/logs"),
        stop_retries_after=100,
        exclude_nodes=exclude_nodes,
        comment="test_comment",
    )
    sbatch_script = _render_sbatch_script(job_spec)
    expected_exclude_nodes = ",".join(job_spec.exclude_nodes) if job_spec.exclude_nodes else None
    assert "test_job" in sbatch_script
    assert "test_account" in sbatch_script
    assert "test_partition" in sbatch_script
    assert str(_START_RAY) in sbatch_script
    assert "arg1" in sbatch_script
    assert "arg2" in sbatch_script
    assert f"--gres={GRES}" in sbatch_script
    assert f"--time={job_spec.time_limit}" in sbatch_script
    assert f"STOP_RETRIES_AFTER={job_spec.stop_retries_after}" in sbatch_script
    if exclude_nodes:
        assert f"--exclude={expected_exclude_nodes}" in sbatch_script
    else:
        assert "--exclude=" not in sbatch_script
    assert f"--output={job_spec.log_dir!s}" in sbatch_script
    assert f'--comment="{job_spec.comment}"' in sbatch_script


class TestSubmitCmd(unittest.TestCase):
    """Test the submit command."""

    def test_get_username(self) -> None:
        """Test that the get_username function returns the correct username."""
        with patch("os.getuid") as mock_getuid:
            mock_getuid.return_value = 123
            with patch("pwd.getpwuid") as mock_getpwuid:
                mock_getpwuid.return_value = Mock(pw_name="test_user")
                username = _get_username()
                assert username == "test_user"

    def test_mount_spec_from_str(self) -> None:
        """Test that the mount spec from string function returns the correct mount spec."""
        mount_str = "/src:/dst:rw"
        mount_spec = MountSpec.from_str(mount_str)
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_slurm_job_spec(self) -> None:
        """Test that the slurm job spec function returns the correct slurm job spec."""
        job_spec = SlurmJobSpec(
            login_node="login_node",
            container=ContainerSpec(squashfs_path="test_path", command=["cmd"], mounts=[], environment=[]),
            job_name="test_job",
            account="test_account",
            partition="test_partition",
            username="test_user",
            num_nodes=1,
            gres=GRES,
            exclusive=True,
            remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
            log_dir=pathlib.Path("/logs"),
        )
        assert job_spec.job_name == "test_job"
        assert job_spec.account == "test_account"
        assert job_spec.partition == "test_partition"
        assert job_spec.username == "test_user"
        assert job_spec.num_nodes == 1
        assert job_spec.gres == GRES
        assert job_spec.exclusive
        assert job_spec.remote_job_path == pathlib.Path("/remote/files") / "test_job.20250611"
        assert job_spec.log_dir == pathlib.Path("/logs")

    def test_parse_job_id(self) -> None:
        """Test that the parse job id function returns the correct job id."""
        output = "Submitted batch job 12345"
        job_id = _parse_job_id(output)
        assert job_id == "12345"

    def test_parse_job_id_with_dots_and_underscores(self) -> None:
        """Test that the parse job id function returns the correct job id with dots and underscores."""
        output = "Submitted batch job job_123.45"
        job_id = _parse_job_id(output)
        assert job_id == "job_123.45"

    def test_parse_job_id_missing_job_id(self) -> None:
        """Test that the parse job id function raises an error if the job id is missing."""
        output = "Submitted batch job"
        with pytest.raises(
            ValueError,
            match="Output 'Submitted batch job' does not contain 'Submitted batch job' followed by a job ID.",
        ):
            _parse_job_id(output)

    def test_parse_job_id_invalid_output(self) -> None:
        """Test that the parse job id function raises an error if the output is invalid."""
        output = "Invalid output"
        with pytest.raises(
            ValueError, match="Output 'Invalid output' does not contain 'Submitted batch job' followed by a job ID."
        ):
            _parse_job_id(output)

    def test_parse_job_id_empty_string(self) -> None:
        """Test that the parse job id function raises an error if the output is empty."""
        output = ""
        with pytest.raises(ValueError, match="Output '' does not contain 'Submitted batch job' followed by a job ID."):
            _parse_job_id(output)

    @patch("fabric.Connection")
    def test_connect_login_creates_connection(self, mock_connection: Mock) -> None:
        """Test that the connect function creates a connection with correct params."""
        conn = connect(remote_host="test_host", user="test_user")
        mock_connection.assert_called_once_with("test_host", user="test_user")
        assert conn == mock_connection.return_value

    @patch("fabric.Connection")
    def test_connect_verifies_connection_works(self, mock_connection: Mock) -> None:
        """Test that the connect function verifies the connection by running 'ls'."""
        mock_conn = mock_connection.return_value
        connect(remote_host="test_host", user="test_user")
        mock_conn.run.assert_called_once_with("ls", hide=True)

    def test_upload_text(self) -> None:
        """Test that the upload text function uploads the correct files."""
        connection = Mock()
        files = [("text1", pathlib.Path("/remote/path1"), 0o644), ("text2", pathlib.Path("/remote/path2"), 0o755)]
        upload_text(connection, files)
        EXPECTED_CALL_COUNT = 2
        assert connection.put.call_count == EXPECTED_CALL_COUNT
        assert connection.run.call_count == EXPECTED_CALL_COUNT

    def test_upload_text_empty_list(self) -> None:
        """Test that the upload text function raises an error if the list of files is empty."""
        connection = Mock()
        files: list[tuple[str, pathlib.Path, int]] = []
        with pytest.raises(ValueError, match="Must upload at least one file"):
            upload_text(connection, files)

    def test_upload_text_file_mode_too_low(self) -> None:
        """Test that the upload text function raises an error if the file mode is too low."""
        connection = Mock()
        files = [("text", pathlib.Path("/remote/path"), -1)]
        with pytest.raises(ValueError, match="Invalid octal file mode: -0o1"):
            upload_text(connection, files)

    def test_upload_text_file_mode_too_high(self) -> None:
        """Test that the upload text function raises an error if the file mode is too high."""
        connection = Mock()
        files = [("text", pathlib.Path("/remote/path"), 0o7777777)]
        with pytest.raises(ValueError, match="Invalid octal file mode: 0o7777777"):
            upload_text(connection, files)


class TestSubmitCurationJob:
    """Test the submit curation job function."""

    @pytest.fixture
    def job_spec(self) -> SlurmJobSpec:
        """Test that the submit curation job function returns the correct job spec."""
        return SlurmJobSpec(
            login_node="login_node",
            container=ContainerSpec(squashfs_path="test_path", command=["cmd"], mounts=[], environment=[]),
            job_name="test_job",
            account="test_account",
            partition="test_partition",
            username="test_user",
            num_nodes=1,
            gres=GRES,
            exclusive=True,
            remote_job_path=pathlib.Path("/remote/files") / "test_job.20250611",
            time_limit="01:00:00",
            log_dir=pathlib.Path("/logs"),
        )

    def test_curator_submit(self, mock_connection: Mock, job_spec: SlurmJobSpec) -> None:
        """Test that the submit curation job function submits the correct job."""
        conn = mock_connection.return_value

        failed_result = Mock()
        failed_result.exited = 1

        # Create an exception that will be raised on first call
        unexpected_exit = invoke.exceptions.UnexpectedExit(result=failed_result)

        # Create a mock for successful run with job ID for sbatch command
        success_result = Mock()
        success_result.stdout = "Submitted batch job 12345"

        # Configure the run method to raise exception when checking that the remote dir exists
        conn.run.side_effect = [
            Mock(),  # ls call succeeds
            unexpected_exit,  # directory check should fail as expected (test -e)
            Mock(),  # mkdir call succeeds
            Mock(),  # chmod job dir
            Mock(),  # chmod sbatch script
            Mock(),  # chmod prometheus service discovery script
            success_result,  # sbatch command returns job ID
        ]

        job_id = curator_submit(job_spec)

        assert job_id == "12345"
        sbatch_calls = [
            call[0][0]
            for call in conn.run.call_args_list
            if isinstance(call[0][0], str) and call[0][0].startswith("sbatch")
        ]
        EXPECTED_SBATCH_CALL_COUNT = 1
        assert len(sbatch_calls) == EXPECTED_SBATCH_CALL_COUNT


class TestMountSpec:
    """Test the mount spec class."""

    def test_mount_spec_can_be_created_with_source_and_dest(self) -> None:
        """Test that the mount spec can be created with source and dest."""
        mount_spec = MountSpec(source="/src", dest="/dst")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_mount_spec_can_be_created_with_source_dest_and_mode(self) -> None:
        """Test that the mount spec can be created with source, dest, and mode."""
        mount_spec = MountSpec(source="/src", dest="/dst", mode="ro")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "ro"

    def test_mount_spec_from_str(self) -> None:
        """Test that the mount spec can be created from a string."""
        mount_spec = MountSpec.from_str("/src:/dst")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "rw"

    def test_mount_spec_from_str_with_mode(self) -> None:
        """Test that the mount spec can be created from a string with mode."""
        mount_spec = MountSpec.from_str("/src:/dst:ro")
        assert mount_spec.source == "/src"
        assert mount_spec.dest == "/dst"
        assert mount_spec.mode == "ro"

    def test_mount_spec_str(self) -> None:
        """Test that the mount spec can be converted to a string."""
        mount_spec = MountSpec(source="/src", dest="/dst", mode="ro")
        assert str(mount_spec) == "/src:/dst:ro"

    def test_mount_spec_from_str_with_invalid_format(self) -> None:
        """Test that the mount spec raises an error if the format is invalid."""
        with pytest.raises(ValueError, match="`/src` must have at least 2 or colon separated parts"):
            MountSpec.from_str("/src")

    def test_mount_spec_from_str_with_too_many_parts(self) -> None:
        """Test that the mount spec raises an error if the format has too many parts."""
        with pytest.raises(ValueError, match="`/src:/dst:ro:extra` must have at least 2 or colon separated parts"):
            MountSpec.from_str("/src:/dst:ro:extra")

    def test_mount_spec_valid_mode(self) -> None:
        """Test that the mount spec can be created with valid modes."""
        MountSpec(source="/src", dest="/dst", mode="rw")
        MountSpec(source="/src", dest="/dst", mode="ro")

    def test_mount_spec_invalid_mode(self) -> None:
        """Test that the mount spec raises an error if the mode is invalid."""
        with pytest.raises(ValueError):  # noqa: PT011
            MountSpec(source="/src", dest="/dst", mode="rx")


class TestContainerSpec:
    """Test the container spec class."""

    @pytest.mark.parametrize("missing_fields", [[], ["command"], ["mounts"], ["environment"], ["squashfs_path"]])
    def test_container_spec_creation(self, missing_fields: list[str]) -> None:
        """Test that the container spec can be created with the correct fields."""
        args: dict[str, Any] = {}
        mounts = [MountSpec(source="/src", dest="/dst")]
        command = ["python", "script.py"]
        squashfs_path = "/path/to/image.sqsh"
        environment = ["a", "b"]

        if "mounts" not in missing_fields:
            args["mounts"] = mounts

        if "command" not in missing_fields:
            args["command"] = command

        if "squashfs_path" not in missing_fields:
            args["squashfs_path"] = squashfs_path

        if "environment" not in missing_fields:
            args["environment"] = environment

        ctx = nullcontext() if len(missing_fields) == 0 else pytest.raises(TypeError)
        with ctx:
            container_spec = ContainerSpec(**args)

        if len(missing_fields) == 0:
            assert container_spec.mounts == mounts
            assert container_spec.command == command
            assert container_spec.squashfs_path == squashfs_path
            assert container_spec.environment == environment


class TestLaunch:
    """Test the launch function."""

    @pytest.fixture
    def mock_curator_submit(self, mocker: Mock) -> Any:  # noqa: ANN401
        """Test that the launch function launches the correct job."""
        return mocker.patch(f"{MODULE_NAME}.curator_submit")

    def test_launch(self, mock_curator_submit: Mock) -> None:
        """Test that the launch function launches the correct job."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_launch_container_mounts(self, mock_curator_submit: Mock) -> None:
        """Test that the launch function launches the correct job with container mounts."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            container_mounts="src0:dst0,src1:dst1",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_launch_environment(self, mock_curator_submit: Mock) -> None:
        """Test that the launch function launches the correct job with environment variables."""
        submit_cli(
            command=[str(_START_RAY), "arg1", "arg2"],
            login_node="login_node",
            account="test_account",
            partition="test_partition",
            container_image="test_image",
            environment="VA1,VA2",
            num_nodes=1,
            remote_files_path=pathlib.Path("/remote/files"),
            gres=GRES,
            exclusive=True,
        )
        mock_curator_submit.assert_called_once()

    def test_launch_invalid_mounts(self, mock_curator_submit: Mock) -> None:
        """Test that the launch function raises an error if the container mounts are invalid."""
        with pytest.raises(ValueError, match="(?i).*must have at least 2 or colon separated parts.*"):
            submit_cli(
                command=[str(_START_RAY), "arg1", "arg2"],
                login_node="login_node",
                account="test_account",
                partition="test_partition",
                container_image="test_image",
                num_nodes=1,
                remote_files_path=pathlib.Path("/remote/files"),
                gres=GRES,
                exclusive=True,
                container_mounts="invalid_mounts",
            )
        mock_curator_submit.assert_not_called()


@pytest.mark.parametrize(
    ("num_nodes", "head_node", "nodename", "procid", "stop_retries_after", "is_head_node"),
    [
        (1, "head_node", "head_node", 0, 100, True),
        (1, "head_node", "worker_node", 1, 100, False),
    ],
)
def test_head_node_is_head_node(  # noqa: PLR0913
    num_nodes: int, head_node: str, nodename: str, procid: int, stop_retries_after: int, *, is_head_node: bool
) -> None:
    """Test that the head node is the head node."""
    slurm_env = SlurmEnv(
        num_nodes=num_nodes,
        head_node=head_node,
        nodename=nodename,
        procid=procid,
        stop_retries_after=stop_retries_after,
    )
    assert slurm_env.is_head_node() == is_head_node
