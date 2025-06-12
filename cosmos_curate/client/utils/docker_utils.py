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
"""Simple utilities for working with docker.

These are used to help automate docker building/pushing/running.
"""

import pathlib
import subprocess
import tempfile

import attrs
import jinja2
from loguru import logger

from cosmos_curate.client.utils import conda_envs


def generate_dockerfile(  # noqa: PLR0913
    *,
    curator_path: pathlib.Path,
    dockerfile_template_path: pathlib.Path,
    envs_paths: list[pathlib.Path],
    conda_env_names: list[str],
    code_paths: list[str],
    dockerfile_output_path: pathlib.Path | None = None,
    verbose: bool = False,
) -> pathlib.Path:
    """Generate a Dockerfile based on the provided template and parameters.

    Args:
        curator_path (pathlib.Path): The path to the cosmos-curate repo directory.
        dockerfile_template_path (pathlib.Path): The path to the Dockerfile template.
        envs_paths: (list[pathlib.Path]): List of environment paths
        conda_env_names (List[str]): The list of conda environment names to include in the Docker image.
        code_paths (List[conda_envs.CodePath]): The list of code paths to include in the Docker image.
        dockerfile_output_path (Optional[pathlib.Path]): The path to write the rendered Dockerfile.
                                                         If None, writes to Dockerfile.
        verbose (bool): If True, logs detailed information.

    Returns:
        pathlib.Path: The path to the generated Dockerfile.

    """
    # Retrieve conda environments
    if conda_env_names:
        all_envs = {x.name: x for x in conda_envs.get_conda_envs(curator_path, envs_paths, conda_env_names)}
        available_env_names = set(all_envs.keys())
        not_found_env_names = set(conda_env_names) - available_env_names
        if not_found_env_names:
            error_msg = (
                f"Was unable to find env name(s) {sorted(not_found_env_names)}. "
                f"Available env names = {sorted(available_env_names)}"
            )
            raise ValueError(error_msg)

        env_list = sorted([all_envs[x] for x in conda_env_names], key=lambda x: x.name)
    else:
        env_list = []

    # Read and render the Dockerfile template
    with pathlib.Path(dockerfile_template_path).open() as f:
        template = jinja2.Template(f.read())
    common_template_params = conda_envs.CommonTemplateParams.make(curator_path)
    contents = template.render(
        envs=env_list,
        code_paths=code_paths,
        **attrs.asdict(common_template_params),
    )
    if verbose:
        logger.info(f"Generated Dockerfile content:\n{contents}")

    # Write the rendered Dockerfile to disk
    if not dockerfile_output_path:
        dockerfile_output_path = pathlib.Path(tempfile.gettempdir()) / "Dockerfile"  # Default output
    dockerfile_output_path.write_text(contents)
    logger.info(f"Dockerfile written to: {dockerfile_output_path}")
    return dockerfile_output_path


def build(  # noqa: PLR0913
    *,
    curator_path: pathlib.Path,
    dockerfile_path: pathlib.Path,
    image: str | None = None,
    cache_from: list[str] | None = None,
    cache_to: str | None = None,
    verbose: bool = False,
) -> None:
    """Build a Docker image variables and a Dockerfile template.

    Args:
        curator_path (pathlib.Path): The path to the curator directory.
        dockerfile_path (pathlib.Path): The path to the Dockerfile.
        image (Optional[str]): The name and tag of the Docker image. Default is None.
        cache_from (Optional[str]): The image to use as a cache source. Default is None.
        cache_to (Optional[str]): The image to use as a cache destination. Default is None.
        verbose (bool): If True, logs detailed information. Default is False.

    """
    cmd = ["docker"]
    if cache_from or cache_to:
        cmd.append("buildx")
    cmd.append("build")
    if cache_from is not None:
        for cache_from_src in cache_from:
            cmd.extend(["--cache-from", cache_from_src])
    if cache_to:
        cmd.extend(["--cache-to", cache_to, "--push"])
    elif cache_from:
        cmd.extend(["--load"])
    cmd.extend(
        [
            f"--progress={'plain' if verbose else 'auto'}",
            "--network=host",
            "-f",
            str(dockerfile_path),
            "-t",
            str(image),
            ".",
        ],
    )
    logger.info(f"Running command from {curator_path}: {' '.join(cmd)}")
    subprocess.check_call(  # noqa: S603
        cmd,
        cwd=curator_path.as_posix(),
    )
