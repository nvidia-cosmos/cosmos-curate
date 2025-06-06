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
"""Remote SLURM job submission and management utilities."""

from __future__ import annotations

import logging
import os
import pwd
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import attrs
import fabric  # type: ignore[import-untyped]
import invoke
import jinja2
import typer
from attrs import field, validators
from typer import Argument, Option

from cosmos_curate.core.utils import environment

logger = logging.getLogger(__name__)

_SBATCH_TEMPLATE_PATH = Path("sbatch.sh.j2")
_START_RAY = environment.CONTAINER_PATHS_CODE_DIR / "cosmos_curate" / "scripts" / "onto_slurm.py"
_MAX_FILE_MODE = 0o7777
_HOME_DIR = Path(os.getenv("REMOTE_HOME_DIR", Path.home()))


def _get_username() -> str:
    """Retrieve the username of the current user.

    Returns:
        str: The username of the current user.

    Raises:
        KeyError: If the user ID is not found in the password database.
        OSError: If an operating system error occurs while retrieving the user ID or username.

    """
    uid = os.getuid()
    return pwd.getpwuid(uid).pw_name


def _get_user_dir(user_dir: Path | None = None) -> Path:
    """Get the user's directory."""
    if user_dir is not None:
        return user_dir

    return _HOME_DIR


def _get_log_dir(log_dir: Path | None = None, user_dir: Path | None = None) -> Path:
    """Get the default log directory.

    If the SLURM_LOG_DIR environment variable is set, it will be used. Otherwise, it will be placed into
    the user's directory.

    Args:
        log_dir: The path to the log directory.
        user_dir: The path to the user's directory.

    Returns:
        The path to the log directory.

    """
    if log_dir is not None:
        return log_dir

    log_dir_str = os.environ.get("SLURM_LOG_DIR")
    if log_dir_str is not None:
        return Path(log_dir_str)

    return _get_user_dir(user_dir) / "job_logs"


@attrs.define
class MountSpec:
    """Represents a mount and its mount point."""

    source: str
    dest: str
    mode: str = field(default="rw", validator=validators.in_(["rw", "ro"]))

    @staticmethod
    def from_str(mount_str: str) -> MountSpec:
        """Create a MountSpec instance from a string.

        The string must have the format:
        source:dest:mode or source:dest

        Args:
            mount_str: The mount string formatted as one of:
                source:dest
                source:dest:mode

                mode is either "ro" or "rw"

        Returns:
            A MountSpec instance

        """
        _MIN_PARTS = 2
        _MAX_PARTS = 3

        parts = mount_str.split(":")
        if len(parts) < _MIN_PARTS or len(parts) > _MAX_PARTS:
            error_message = f"`{mount_str}` must have at least {_MIN_PARTS} or colon separated parts"
            raise ValueError(error_message)

        source = parts[0]
        dest = parts[1]
        mode = "rw" if len(parts) == _MIN_PARTS else parts[2]

        return MountSpec(source, dest, mode)

    def __str__(self) -> str:
        """Return a string representation of the mount suitable for use with Docker or Enroot."""
        return f"{self.source}:{self.dest}:{self.mode}"


@attrs.define
class ContainerSpec:
    """Configuration for a container to run in the SLURM job."""

    command: list[str]
    squashfs_path: str
    mounts: list[MountSpec]
    environment: list[str]


@attrs.define
class SlurmJobSpec:
    """Configuration for a SLURM job."""

    login_node: str
    username: str
    account: str
    partition: str
    job_name: str
    remote_files_path: Path
    log_dir: Path
    num_nodes: int
    exclusive: bool
    container: ContainerSpec
    gres: str | None = None
    time_limit: str | None = None
    stop_retries_after: int = 600
    exclude_nodes: list[str] | None = None


def _render_sbatch_script(spec: SlurmJobSpec) -> str:
    """Render a Slurm batch script using the provided job specification and cluster configuration.

    Args:
        spec (SlurmJobSpec): Job specification, including name, account,
            partition, and command.

    Returns:
        str: Rendered Slurm batch script as a string.

    Notes:
        This function assumes the _SBATCH_TEMPLATE_PATH template file
        exists and contains necessary template variables.

    """
    container_mounts = ",".join(str(x) for x in spec.container.mounts) if spec.container.mounts is not None else None
    command = " ".join(spec.container.command)
    template_dir = Path(__file__).parent
    sbatch_template = template_dir / _SBATCH_TEMPLATE_PATH

    env_vars: dict[str, str] = {}
    if spec.container.environment is not None:
        env_vars = {
            env_var: os.environ.get(env_var, "")
            for env_var in spec.container.environment
            if os.environ.get(env_var) is not None
        }

    return jinja2.Template(sbatch_template.read_text()).render(
        job_name=spec.job_name,
        account=spec.account,
        partition=spec.partition,
        num_nodes=spec.num_nodes,
        command=command,
        gres=spec.gres,
        exclusive=spec.exclusive,
        container_image=spec.container.squashfs_path,
        container_mounts=container_mounts,
        env_vars=env_vars,
        time_limit_string=spec.time_limit,
        stop_retries_after=spec.stop_retries_after,
        exclude_nodes=spec.exclude_nodes,
        log_dir=str(spec.log_dir),
    )


def _parse_job_id(output: str) -> str:
    """Parse the job ID from a string that contains the submission confirmation of a batch job.

    Args:
        output: The output from a job submission command, expected to contain
            "Submitted batch job <job_id>".

    Returns:
        The job ID parsed from the output string.

    Raises:
        ValueError: If the job ID cannot be found or is not valid

    """
    # job_ids are not always an integer, they can contain dots and underscores
    pattern = r"Submitted batch job (.*)"
    match = re.search(pattern, output)

    if match:
        return match.group(1)

    error_message = f"Output '{output}' does not contain 'Submitted batch job' followed by a job ID."
    raise ValueError(error_message)


def connect(remote_host: str, user: str) -> fabric.Connection:
    """Connect to a SLURM cluster host.

    Args:
        remote_host (str): the hostname to connect to
        user (str): the username to connect with

    Returns:
        A fabric.Connection object if successful

    Raises:
        typer.Abort: if connection could not be established

    """
    logger.info("Connecting to %s as %s...", remote_host, user)
    conn = fabric.Connection(remote_host, user=user)
    conn.run("ls", hide=True)
    return conn


def upload_text(connection: fabric.Connection, files: list[tuple[str, Path, int]]) -> None:
    """Upload multiple text strings the provided connection.

    Args:
        connection (fabric.Connection): An established connection
        files (list[tuple[str, Path, int]]): A list of tuples containing
            the text to upload, the remote path, and the octal file mode.

    Returns:
        None

    Raises:
        ValueError: If the number of files is not greater than zero.
        ValueError: If the octal file mode is not a valid integer.

    """
    if len(files) == 0:
        error_message = "Must upload at least one file"
        raise ValueError(error_message)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for text, remote_path, file_mode in files:
            if not isinstance(file_mode, int) or file_mode < 0 or file_mode > _MAX_FILE_MODE:
                error_message = f"Invalid octal file mode: {oct(file_mode)}"
                raise ValueError(error_message)

            tmp_file = Path(tmp_dir) / remote_path.name
            tmp_file.write_text(text)
            logger.debug("Uploading %s to %s", tmp_file, remote_path)

            connection.put(str(tmp_file), remote=str(remote_path))
            connection.run(f"chmod {file_mode:o} {remote_path}")


def remote_path_exists(connection: fabric.Connection, path: Path) -> bool:
    """Check if a path exists on the remote host.

    Args:
        connection (fabric.Connection): An ssh connection to the login node
        path (Path): The path to check

    Returns:
        bool: True if the directory exists, False otherwise

    """
    dir_exists = False
    try:
        dir_exists = connection.run(f"test -e {path}", hide=True).ok
    except invoke.exceptions.UnexpectedExit as e:
        if e.result.exited != 1:  # test returns 1 when file doesn't exist
            # If exit code is something other than 1, there's another issue
            raise

    return dir_exists


def create_remote_path(connection: fabric.Connection, path: Path, mode: int = 0o700) -> None:
    """Create a remote path on the cluster.

    Args:
        connection (fabric.Connection): An ssh connection to the login node
        path (Path): The path to create
        mode (int): The mode to set for the path

    """
    connection.run(f"mkdir -p {path}")
    connection.run(f"chmod {mode:o} {path}")


def create_remote_files_path(connection: fabric.Connection, job_spec: SlurmJobSpec) -> Path:
    """Create a remote files path for the job on the cluster.

    Args:
        connection (fabric.Connection): An ssh connection to the login node
        job_spec (SlurmJobSpec): The job specification, including job name,
            account, partition, and command.

    Returns:
        The remote files path

    Raises:
        ValueError: If the remote files path already exists

    """
    remote_files_path = (
        job_spec.remote_files_path / f"{job_spec.job_name}.{datetime.now().strftime('%Y%m%dT%H%M%S.%f')}"  # noqa: DTZ005
    )

    # If the directory exists, raise an error because there might be a race condition
    if remote_path_exists(connection, remote_files_path):
        error_message = f"Remote files path already exists: {remote_files_path}"
        raise ValueError(error_message)

    create_remote_path(connection, remote_files_path)
    return remote_files_path


def curator_submit(slurm_job_spec: SlurmJobSpec) -> str:
    """Submit a curator pipeline batch job to the cluster.

    Args:
        slurm_job_spec: The job specification

    Returns:
        The slurm job id

    Raises:
        ValueError: If the job ID cannot be parsed from the submission
            output.

    """
    connection = connect(slurm_job_spec.login_node, slurm_job_spec.username)
    remote_files_path = create_remote_files_path(connection, slurm_job_spec)

    slurm_job_spec.container.mounts += [MountSpec(source=str(remote_files_path), dest="/remote_files")]
    logger.debug("Container mounts: %s", slurm_job_spec.container.mounts)
    remote_sbatch_path = remote_files_path / "sbatch.sh"

    remote_files = [
        (
            _render_sbatch_script(slurm_job_spec),
            remote_sbatch_path,
            0o700,
        ),
    ]

    upload_text(connection, remote_files)
    out = connection.run(f"sbatch {remote_sbatch_path}")
    return _parse_job_id(out.stdout)


def remote_find_job_log_file(connection: fabric.Connection, log_dir: Path, job_id: str) -> Path:
    """Find a log file for a given job ID in the log directory.

    Args:
        connection: SSH connection to the remote host
        log_dir: Directory to search for log files
        job_id: The job ID to search for

    Returns:
        Path to the log file

    Raises:
        FileNotFoundError: If the log directory doesn't exist or no log file is found

    """
    if not remote_path_exists(connection, log_dir):
        error_message = f"Directory `{log_dir}` does not exist on {connection.host}"
        raise FileNotFoundError(error_message)

    find_result = connection.run(f"find {log_dir} -name '*_{job_id}.log' -type f | head -n 1", hide=True)
    if not find_result.stdout.strip():
        error_message = f"No log file found for job ID {job_id} in {log_dir}"
        raise FileNotFoundError(error_message)

    return Path(find_result.stdout.strip())


def remote_tail(connection: fabric.Connection, file_path: Path) -> None:
    """Tail a file on a remote host using the provided connection.

    Args:
        connection: SSH connection to the remote host
        file_path: Path to the file to tail

    """
    cmd: list[str] = ["tail", "-f", str(file_path)]
    cmd_str = " ".join(cmd)
    logger.info("Running `%s`, press Ctrl+C to stop", cmd_str)
    connection.run(cmd_str)


def job_log(hostname: str, username: str, job_id: str, log_dir: Path | None = None) -> None:
    """Connect to a login node and tails the log for a specific job ID.

    Args:
        hostname: The hostname of the node that has access to the logs
        username: The username to use for the connection
        job_id: The Slurm job ID to find logs for
        log_dir: The path to the log directory

    Raises:
        ValueError: If no log file is found for the given job ID

    """
    connection = connect(hostname, username)
    _log_dir = _get_log_dir(log_dir)
    log_file = remote_find_job_log_file(connection, _log_dir, job_id)
    remote_tail(connection, log_file)


def job_log_cli(
    *,
    job_id: Annotated[str, Option(help="The Slurm job ID to find logs for", rich_help_panel="common")],
    login_node: Annotated[str, Option(help="Hostname of SLURM login node to use", rich_help_panel="common")],
    username: Annotated[
        str, Option(help=("Optional cluster username"), rich_help_panel="common")
    ] = f"{_get_username()}",
    log_dir: Annotated[Path | None, Option(help="Path to the log directory", rich_help_panel="common")] = None,
) -> None:
    """View the logs for a specific job running on the cluster.

    Either slurm_config or login_node must be provided, but not both.

    """
    job_log(login_node, username, job_id, log_dir)


def submit_cli(  # noqa: PLR0913
    command: Annotated[list[str], Argument(help="The command to run", rich_help_panel="common")],
    *,
    login_node: Annotated[
        str,
        Option(
            help="Hostname of SLURM login node to run command on",
            rich_help_panel="cluster",
        ),
    ],
    account: Annotated[
        str,
        Option(
            help="Name of account for billing",
            rich_help_panel="cluster",
        ),
    ],
    partition: Annotated[
        str,
        Option(
            help=("The slurm partition to use"),
            rich_help_panel="cluster",
        ),
    ],
    remote_files_path: Annotated[
        Path,
        Option(
            help="Path to remote files directory, where the sbatch script will be placed",
            rich_help_panel="cluster",
        ),
    ] = _HOME_DIR / "curator_launch_files",
    container_image: Annotated[
        str,
        Option(help=("Canonical path to the .sqsh container image"), rich_help_panel="container"),
    ],
    container_mounts: Annotated[
        str | None,
        Option(help="Comma separated container mounts, `src0:dst0,src1:dst1`", rich_help_panel="container"),
    ] = None,
    environment: Annotated[
        str | None,
        Option(
            help="Comma separated list of environment variables to set in the container", rich_help_panel="container"
        ),
    ] = None,
    username: Annotated[
        str,
        Option(
            help=("Optional cluster username"),
            rich_help_panel="misc",
        ),
    ] = f"{_get_username()}",
    job_name: Annotated[
        str,
        Option(
            help="Name of the job",
            rich_help_panel="cluster",
        ),
    ] = "cosmos_curate",
    num_nodes: Annotated[
        int,
        Option(
            help="Number of nodes to use on the cluster",
            rich_help_panel="cluster",
        ),
    ] = 1,
    gres: Annotated[
        str | None,
        Option(
            help="Optional GPU specification, e.g. 'gpu:8'",
            rich_help_panel="cluster",
        ),
    ] = None,
    exclusive: Annotated[
        bool,
        Option(
            help="Whether to use nodes exclusively",
            rich_help_panel="cluster",
        ),
    ] = True,
    time: Annotated[
        str | None,
        Option(
            help="Time limit for the job, e.g. 01:00:00 for 1 hour. See sbatch --time for more details.",
            rich_help_panel="cluster",
        ),
    ] = None,
    stop_retries_after: Annotated[
        int,
        Option(
            help="Maximum time in secondsto wait before `ray start` retries end",
            rich_help_panel="cluster",
        ),
    ] = 600,
    exclude_nodes: Annotated[
        str | None,
        Option(help="Comma separated list of nodes to exclude", rich_help_panel="cluster"),
    ] = None,
    log_dir: Annotated[
        Path | None,
        Option(help="Path to the log directory", rich_help_panel="cluster"),
    ] = None,
) -> None:
    """Submit a job to a SLURM cluster."""
    if not command:
        error_message = "A command must be provided"
        raise ValueError(error_message)

    container_mount_specs: list[MountSpec] = []
    if container_mounts is not None:
        container_mount_specs = [MountSpec.from_str(x) for x in container_mounts.split(",")]

    env_list = environment.split(",") if environment is not None else []
    exclude_nodes_list = exclude_nodes.split(",") if exclude_nodes is not None else None

    container_spec = ContainerSpec(
        command=[str(_START_RAY), *command],
        squashfs_path=container_image,
        environment=env_list,
        mounts=container_mount_specs,
    )

    slurm_job_spec = SlurmJobSpec(
        login_node=login_node,
        username=username,
        remote_files_path=remote_files_path,
        log_dir=_get_log_dir(log_dir),
        job_name=job_name,
        account=account,
        partition=partition,
        num_nodes=num_nodes,
        container=container_spec,
        gres=gres,
        exclusive=exclusive,
        time_limit=time,
        stop_retries_after=stop_retries_after,
        exclude_nodes=exclude_nodes_list,
    )

    job_id = curator_submit(slurm_job_spec)
    logger.info("Job submitted with ID: %s", job_id)


slurm_cli = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

slurm_cli.command("submit", no_args_is_help=True)(submit_cli)
slurm_cli.command("job-log", no_args_is_help=True)(job_log_cli)


if __name__ == "__main__":
    slurm_cli()
