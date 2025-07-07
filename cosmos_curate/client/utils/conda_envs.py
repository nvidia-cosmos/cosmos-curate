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
"""Stores information about the available conda envs and contains utilities for building them."""

from pathlib import Path

import attrs
import jinja2

from cosmos_curate.client.environment import CONTAINER_PATHS_CODE_DIR

_BUILD_FILE_NAME = "build_steps.dockerfile.j2"


@attrs.define
class CosmosCuratorDeps:
    """Gets the core cosmos-curate dependencies.

    These need to be installed into the main docker image as well as any conda envs.
    """

    core: list[str]
    regular: list[str]

    @classmethod
    def make(cls, curator_path: Path) -> "CosmosCuratorDeps":
        """Create CosmosCuratorDeps instance by loading dependencies from curator path.

        Args:
            curator_path: Path to the cosmos-curate repo directory.

        Returns:
            A CosmosCuratorDeps instance with the dependencies loaded from the requirements.txt file.

        """
        requirements_file_path = curator_path / "package" / "cosmos_curate"
        video_deps = {}
        for variant in ("core", "regular"):
            requirements_file = requirements_file_path / f"requirements-{variant}.txt"
            if not requirements_file.exists():
                error_msg = f"Could not find requirements file at {requirements_file}"
                raise FileNotFoundError(error_msg)
            with Path(requirements_file).open() as f:
                video_deps[variant] = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        return CosmosCuratorDeps(sorted(video_deps["core"]), sorted(video_deps["regular"]))


@attrs.define
class CondaEnv:
    """A conda env that cosmos-curate can be built with."""

    name: str
    # The rendered build_env.sh.j2 file contents
    build_steps_str: str


_INSTALL_COSMOS_CURATOR_DEPS_TEMPLATE = jinja2.Template(
    """
RUN {{cache_mount_str}} pip install {{cosmos_curator_deps_string}}
""".strip(),
)


@attrs.define
class CommonTemplateParams:
    """Template params that are in common between env templates."""

    # Core deps
    core_cosmos_curator_deps_string: str
    # Regular deps
    regular_cosmos_curator_deps_string: str

    install_core_cosmos_curator_deps_str: str
    install_regular_cosmos_curator_deps_str: str

    # This is a string that is needed to put after "RUN" when installing things via conda or pip.
    # It gets conda and pip to cache to a cache mount to speed up future builds.
    cache_mount_str: str = (
        "--mount=type=cache,target=/cosmos_curate/conda_envs/pkgs --mount=type=cache,target=/root/.cache/"
    )
    code_dir_str: str = CONTAINER_PATHS_CODE_DIR.as_posix()

    @classmethod
    def make(cls, curator_path: Path) -> "CommonTemplateParams":
        """Create CommonTemplateParams instance by loading dependencies from cosmos-curate path.

        Args:
            curator_path: Path to the cosmos-curate repo directory.

        Returns:
            CommonTemplateParams instance with loaded dependencies.

        """
        deps = CosmosCuratorDeps.make(curator_path)
        core_cosmos_curator_deps_string = " ".join([f'"{x}"' for x in deps.core])
        regular_cosmos_curator_deps_string = " ".join([f'"{x}"' for x in deps.regular])

        out = CommonTemplateParams(
            core_cosmos_curator_deps_string,
            regular_cosmos_curator_deps_string,
            "",
            "",
        )
        out.install_core_cosmos_curator_deps_str = _INSTALL_COSMOS_CURATOR_DEPS_TEMPLATE.render(
            cache_mount_str=out.cache_mount_str,
            cosmos_curator_deps_string=out.core_cosmos_curator_deps_string,
        )
        out.install_regular_cosmos_curator_deps_str = _INSTALL_COSMOS_CURATOR_DEPS_TEMPLATE.render(
            cache_mount_str=out.cache_mount_str,
            cosmos_curator_deps_string=out.regular_cosmos_curator_deps_string,
        )
        return out


def get_potential_envs(envs_paths: list[Path]) -> list[str]:
    """Get list of available conda environment names from the provided paths.

    Args:
        envs_paths: List of paths to search for conda environment configurations.

    Returns:
        List of conda environment names found in the paths.

    """
    potential_envs = []
    for path in envs_paths:
        potential_envs.extend([str(x.parent.name) for x in path.glob(f"*/{_BUILD_FILE_NAME}")])
    return potential_envs


def get_conda_envs(curator_path: Path, envs_paths: list[Path], envs_to_build: list[str]) -> list[CondaEnv]:
    """Return all the available cosmos-curate conda envs.

    This is used when building docker images which contain conda envs.

    For each env, we look for a directory at envs_path/{env_name} and a file at
    envs_path/{env_name}/build_steps.dockerfile.j2.

    We will then render the build_steps.dockerfile.j2 file using the parameters in CommonTemplateParams.
    Finally, we return the conda name and the rendered docker file contents. This file will be used to build
    the dockerfile we will launch with.
    """
    if not envs_to_build:
        return []

    available_env_names = set(get_potential_envs(envs_paths))
    not_found_env_names = set(envs_to_build) - available_env_names
    if not_found_env_names:
        error_msg = (
            f"Was unable to find env name(s) {sorted(not_found_env_names)}. "
            f"Available env names = {sorted(available_env_names)}"
        )
        raise ValueError(error_msg)

    template_params = CommonTemplateParams.make(curator_path)
    out = []
    for env_name in envs_to_build:
        found_env = False
        for path in envs_paths:
            d = path / env_name
            if not d.exists():
                continue
            found_env = True
            break
        if not found_env:
            error_msg = (
                f"Expected directory to exist in one of {envs_paths} because we were told to build env={env_name}, "
                "but it does not exist."
            )
            raise ValueError(error_msg)
        steps_file = d / _BUILD_FILE_NAME
        if not steps_file.exists():
            error_msg = f"Expected {steps_file} to exists because directory {d} exists."
            raise ValueError(error_msg)
        contents = jinja2.Template(steps_file.read_text()).render(**attrs.asdict(template_params))
        out.append(CondaEnv(env_name, contents))
    return out
