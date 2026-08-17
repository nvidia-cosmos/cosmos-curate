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

"""Cosmos Curator package setup.

This script handles the setup and installation process for the Cosmos Curator package.
It reads metadata from pyproject.toml, prepares the build directory structure,
and configures the package for distribution.
"""

import shutil
import sys
from pathlib import Path

import tomli
from setuptools import find_namespace_packages, setup


def load_project_name() -> str:
    """Load the package name from pyproject.toml.

    Package metadata, including the dynamic version, is supplied by pyproject.toml
    and setuptools-scm. This script only needs the name for build-directory paths.

    Returns:
        Package name.

    """
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["name"]


name = load_project_name()

build_dir = "build"
pkg_path = Path(build_dir) / name
src_package_dir = Path(name)


def build_package() -> None:
    """Copy the source package into the non-editable build tree."""
    # Recreate only the package staging tree. Distribution output belongs to
    # the build frontend and must survive metadata-only setup invocations.
    shutil.rmtree(pkg_path, ignore_errors=True)

    Path(build_dir).mkdir(exist_ok=True)
    if not src_package_dir.is_dir():
        msg = f"Missing source package: {src_package_dir}"
        raise FileNotFoundError(msg)
    shutil.copytree(
        src_package_dir,
        pkg_path,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
    )


is_editable_install = any(command in sys.argv for command in ("develop", "editable_wheel"))
if is_editable_install:
    package_search_dir = "."
    package_dir = {}
else:
    build_package()
    package_search_dir = build_dir
    package_dir = {"": build_dir}

setup(
    name=name,
    packages=find_namespace_packages(where=package_search_dir, include=[name, f"{name}.*"]),
    include_package_data=True,
    package_dir=package_dir,
)
