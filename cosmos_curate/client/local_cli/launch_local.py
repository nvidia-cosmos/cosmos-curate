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

"""CLI to launch commands."""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import psutil
import tomli
import typer
from loguru import logger
from typer import Argument, Option

from cosmos_curate.client.environment import (
    AZURE_PROFILE_PATH,
    CONTAINER_PATHS_CODE_DIR,
    CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE,
    CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR,
    LOCAL_AWS_CREDENTIALS_FILE,
    LOCAL_AZURE_CREDENTIALS_FILE,
    LOCAL_COSMOS_CURATOR_CONFIG_FILE,
    LOCAL_DOCKER_ENV_VAR_NAME,
    LOCAL_WORKSPACE_PATH,
    S3_PROFILE_PATH,
)
from cosmos_curate.client.image_cli.image_app import get_image_label

cc_client_local = typer.Typer(
    help="Commands for building container image.",
    no_args_is_help=True,
)


@dataclass
class LaunchDocker:
    """Configuration class for launching Docker containers with specified parameters.

    This class holds the configuration needed to launch a Docker container, including
    image details, paths, and credential mounting options.
    """

    image_label: str
    curator_path: str | None
    mount_xenna: bool
    command: str
    gpus: str | None
    mount_s3_creds: bool
    mount_azure_creds: bool


@cc_client_local.command(no_args_is_help=True)
def launch(  # noqa: PLR0913
    *,
    command: Annotated[list[str], Argument(help="The command to run", rich_help_panel="common")],
    image_name: Annotated[
        str,
        Option(
            help=("The docker image name string to use."),
            rich_help_panel="container-image",
        ),
    ] = "cosmos-curate",
    image_tag: Annotated[
        str,
        Option(
            help=("The docker image tag to use."),
            rich_help_panel="container-image",
        ),
    ] = "1.0.0",
    curator_path: Annotated[
        str | None,
        Option(
            help=("Path to the cosmos-curate repo directory; set to mount local curator code into the container."),
            rich_help_panel="local-docker",
        ),
    ] = None,
    mount_xenna: Annotated[
        bool,
        Option(
            help=(
                "Mount the local cosmos_xenna into the container; python code & default env only. "
                "WARNING: very hacky, for local development only."
            ),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = False,
    gpus: Annotated[
        str | None,
        Option(
            help=("The GPUs to use for local-docker mode, e.g. `1 or 0,1`. If not specified, defaults to all."),
            rich_help_panel="local-docker",
        ),
    ] = None,
    mount_s3_creds: Annotated[
        bool,
        Option(
            help=("Skip mounting the AWS credentials file into the container."),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = True,
    mount_azure_creds: Annotated[
        bool,
        Option(
            help=("Mount the Azure credentials file into the container."),
            rich_help_panel="local-docker",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """Launch video-curation pipeline in local docker container.

    The function supports mounting AWS S3 and Azure credentials into the container,
    which can be controlled independently with the mount_s3_creds and mount_azure_creds
    flags. This allows using either S3 storage, Azure storage, or both together.
    """
    command_str = " ".join(command)

    opts = LaunchDocker(
        image_label=get_image_label(image_name, image_tag),
        curator_path=curator_path,
        mount_xenna=mount_xenna,
        command=command_str,
        gpus=gpus,
        mount_s3_creds=mount_s3_creds,
        mount_azure_creds=mount_azure_creds,
    )
    return _launch_in_docker_container(opts)


def _verify_local_path_exists(local_paths: list[Path]) -> None:
    for local_path in local_paths:
        if not local_path.exists():
            logger.error(f"Local path {local_path} does not exist")
            sys.exit(1)


def _pause_for_warnings(timeout: int = 5) -> None:
    logger.info(f"Pausing for {timeout} seconds to show above warnings")
    time.sleep(timeout)


def _get_s3_creds_mount_strings(opts: LaunchDocker) -> list[str]:
    """Handle S3 credentials mounting."""
    s3_creds_strings = []
    if LOCAL_AWS_CREDENTIALS_FILE.exists():
        s3_creds_strings += [
            "-v",
            f"{LOCAL_AWS_CREDENTIALS_FILE}:{S3_PROFILE_PATH}",
        ]
    elif opts.mount_s3_creds:
        logger.warning(f"No AWS creds file found at {LOCAL_AWS_CREDENTIALS_FILE}; S3 operations will not work")
        _pause_for_warnings()
    return s3_creds_strings


def _get_azure_creds_mount_strings(opts: LaunchDocker) -> list[str]:
    """Handle Azure credentials mounting."""
    azure_creds_strings = []
    if LOCAL_AZURE_CREDENTIALS_FILE.exists():
        azure_creds_strings += [
            "-v",
            f"{LOCAL_AZURE_CREDENTIALS_FILE}:{AZURE_PROFILE_PATH}",
        ]
    elif opts.mount_azure_creds:
        logger.warning("No Azure creds file found at {LOCAL_AZURE_CREDENTIALS_FILE}; Azure operations will not work")
        _pause_for_warnings()
    return azure_creds_strings


def _get_config_file_mount_strings(*, is_model_cli: bool) -> list[str]:
    """Handle cosmos-curate config file mounting."""
    config_file_strings = []
    if LOCAL_COSMOS_CURATOR_CONFIG_FILE.exists():
        config_file_strings += [
            "-v",
            f"{LOCAL_COSMOS_CURATOR_CONFIG_FILE}:{CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE}",
        ]
    else:
        logger.warning(f"No config file found at {LOCAL_COSMOS_CURATOR_CONFIG_FILE}")
        logger.warning("Model download and database operation will not work")
        if is_model_cli:
            _verify_local_path_exists([LOCAL_COSMOS_CURATOR_CONFIG_FILE])
        else:
            _pause_for_warnings()
    return config_file_strings


def _get_python_version_from_pixi_toml(curator_path: Path) -> str | None:
    pixi_toml_path = curator_path / Path("pixi.toml")
    if not pixi_toml_path.exists():
        return None
    try:
        with pixi_toml_path.open("rb") as fp:
            pixi_toml = tomli.load(fp)
        python_version = pixi_toml["dependencies"]["python"]
    except (tomli.TOMLDecodeError, KeyError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse python version from pixi.toml: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error reading pixi.toml: {e}")
        return None
    else:
        # validate the version format
        if not python_version or not isinstance(python_version, str):
            logger.warning("Python version not found or invalid in pixi.toml")
            return None
        # strip the possible ">=" or "<=" prefix
        if python_version.startswith((">=", "<=")):
            python_version = python_version[2:]
        # strip the minor version
        version_parts = python_version.split(".")
        _TARGET_PYTHON_VERSION_PARTS = 2
        if len(version_parts) < _TARGET_PYTHON_VERSION_PARTS:
            logger.warning(f"Invalid python version format in pixi.toml: {python_version}")
            return None
        return ".".join(version_parts[:_TARGET_PYTHON_VERSION_PARTS])


def _get_code_mount_strings(opts: LaunchDocker) -> list[str]:
    code_path_strings = []
    if opts.curator_path is not None:
        curator_code_path = Path(opts.curator_path) / Path("cosmos_curate")
        pipeline_code_path = curator_code_path / Path("pipelines")
        if not pipeline_code_path.exists():
            logger.error(f"Curator pipelines code does not exist at {pipeline_code_path}")
            sys.exit(1)
        code_path_strings += ["-v", f"{curator_code_path.absolute()}:{CONTAINER_PATHS_CODE_DIR}/cosmos_curate"]

        if opts.mount_xenna:
            xenna_path = (Path(opts.curator_path) / Path("cosmos-xenna") / Path("cosmos_xenna")).absolute()
            _python_version = _get_python_version_from_pixi_toml(Path(opts.curator_path))
            if xenna_path.exists() and _python_version is not None:
                xenna_lib_path = (
                    CONTAINER_PATHS_CODE_DIR
                    / Path(".pixi/envs/default/lib")
                    / f"python{_python_version}"
                    / Path("site-packages/cosmos_xenna")
                )
                for python_module in ["pipelines", "ray_utils", "utils"]:
                    code_path_strings += ["-v", f"{xenna_path / python_module}:{xenna_lib_path / python_module}"]

        tests_path = Path(opts.curator_path) / Path("tests") / Path("cosmos_curate")
        code_path_strings += ["-v", f"{tests_path.absolute()}:{CONTAINER_PATHS_CODE_DIR}/tests/cosmos_curate"]
    return code_path_strings


def _get_system_memory_gb() -> float:
    mem = psutil.virtual_memory()
    return mem.total / (1024**3)


def _get_shm_size_str() -> str:
    default_proportion = 0.4
    mem_proportion_str = os.environ.get("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", str(default_proportion))
    try:
        fraction = float(mem_proportion_str)
    except ValueError:
        logger.warning(
            f"Found RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION in env, but value must be a float. "
            f"Got: {mem_proportion_str}. Using default 0.4."
        )
        fraction = default_proportion
    return f"{_get_system_memory_gb() * fraction:.2f}gb"


def _launch_in_docker_container(opts: LaunchDocker) -> None:
    """Launch the command inside a local Docker container."""
    if not LOCAL_WORKSPACE_PATH.exists():
        Path(LOCAL_WORKSPACE_PATH).mkdir()
    is_model_cli = "model_cli" in opts.command
    is_postgres_cli = "postgres_cli" in opts.command

    gpus_string = f'"device={opts.gpus}"' if opts.gpus else "all"

    user_strings = ["-u", f"{os.getuid()}:{os.getgid()}"] if is_model_cli else []
    interactive_strings = ["-i"] if is_postgres_cli else []

    docker_command = [
        "docker",
        "run",
        "--rm",
        f"--gpus={gpus_string}",
        "--device=/dev/dri:/dev/dri",
        f"--shm-size={_get_shm_size_str()}",
        "--network=host",
        "--cap-add=SYS_ADMIN",
        "-e",
        f"{LOCAL_DOCKER_ENV_VAR_NAME}=1",
        "-e",
        "NVCF_REQUEST_STATUS=false",
    ]
    docker_command.extend(user_strings)
    docker_command.extend(
        [
            "-v",
            f"{LOCAL_WORKSPACE_PATH}:{CONTAINER_PATHS_DEFAULT_WORKSPACE_DIR}",
        ]
    )
    docker_command.extend(_get_code_mount_strings(opts))
    docker_command.extend(_get_s3_creds_mount_strings(opts))
    docker_command.extend(_get_azure_creds_mount_strings(opts))
    docker_command.extend(_get_config_file_mount_strings(is_model_cli=is_model_cli))
    docker_command.extend(interactive_strings)
    docker_command.extend(
        [
            "-t",
            f"{opts.image_label}",
            "bash",
            "-c",
        ]
    )

    docker_command_to_print = " ".join([*docker_command, f'"{opts.command}"'])
    logger.info(f"Docker command:\n{docker_command_to_print}")

    docker_command.append(f"{opts.command}")

    result = subprocess.call(  # noqa: S603
        docker_command,
        shell=False,
    )
    if result != 0:
        logger.error("Failed to run command via docker")
        sys.exit(1)
