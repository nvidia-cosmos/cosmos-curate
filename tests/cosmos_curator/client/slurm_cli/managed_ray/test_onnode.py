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
"""Contract tests for the modules uploaded into a run directory and executed on cluster nodes."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_curator.client.slurm_cli.managed_ray import onnode
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_runtime import RUNTIME_MODULE_FILENAME
from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    MANIFEST_SCHEMA_VERSION,
    STATE_MODULE_FILENAME,
)

ONNODE_DIR = Path(onnode.__file__).parent
SHIPPED_MODULES = sorted(path for path in ONNODE_DIR.glob("*.py") if path.name != "__init__.py")
# Oldest python3 found on a supported Slurm login or compute node; see the onnode package docstring.
FLOOR_VERSION = (3, 8)
FLOOR_PYTHON_ENV_VAR = "COSMOS_CURATOR_ONNODE_FLOOR_PYTHON"
FLOOR_SKIP_REASON = f"no Python {FLOOR_VERSION[0]}.{FLOOR_VERSION[1]} here; set ${FLOOR_PYTHON_ENV_VAR}"


_FLOOR_MANIFEST = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "run_id": "cc-ray-0123456789ab",
    "job_name": "cosmos_curator",
    "slurm_cluster_name": "test-cluster",
    "state": "SUBMITTING",
    "revision": 0,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "started_at": None,
    "stop_requested": False,
    "pipeline_exit_status": None,
    "error": None,
    "command": ["true"],
    "config": {},
    "runtime_paths": {"run_dir": "/tmp/run"},  # noqa: S108
    "head_job_id": "1",
    "lane_allocations": 1,
    "lanes": [],
}


def _stage_run_directory(destination: Path) -> None:
    """Copy the shipped modules out of the package the way submission copies them into a run directory."""
    for shipped in SHIPPED_MODULES:
        (destination / shipped.name).write_text(shipped.read_text(encoding="utf-8"), encoding="utf-8")


def _floor_interpreter() -> str | None:
    """Locate an interpreter at the on-node floor, if this machine happens to have one.

    The AST checks below need no interpreter and always run; executing on the floor is an opportunistic bonus.
    Point ``$COSMOS_CURATOR_ONNODE_FLOOR_PYTHON`` at a 3.8, or have ``python3.8`` on ``PATH``.
    """
    return os.environ.get(FLOOR_PYTHON_ENV_VAR) or shutil.which("python{}.{}".format(*FLOOR_VERSION))


def _module_scope_imports(source: str) -> set[str]:
    """Return the root package of every import executed when the module is loaded."""
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        # Only module-scope imports run on a host without a Cosmos Curator install; the head and worker roles
        # import heavier dependencies lazily inside the functions that need them.
        if not isinstance(node, ast.Import | ast.ImportFrom) or _is_nested(tree, node):
            continue
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif node.level == 0 and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots


def _is_nested(tree: ast.Module, target: ast.stmt) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and any(
            child is target for child in ast.walk(node)
        ):
            return True
    return False


def test_shipped_modules_are_discovered() -> None:
    """The contract below covers every file submission uploads."""
    assert {path.name for path in SHIPPED_MODULES} == {STATE_MODULE_FILENAME, RUNTIME_MODULE_FILENAME}


@pytest.mark.parametrize("module_path", SHIPPED_MODULES, ids=lambda path: path.name)
def test_onnode_imports_only_the_standard_library(module_path: Path) -> None:
    """A run directory is executed by a host with no Cosmos Curator install and no Pixi environment."""
    imported = _module_scope_imports(module_path.read_text(encoding="utf-8"))
    # A shipped module may import a peer by flat name, and may import cosmos_curator under a TYPE_CHECKING or
    # in-package guard, because neither is resolved on a bare host.
    siblings = {path.stem for path in SHIPPED_MODULES}
    third_party = imported - sys.stdlib_module_names - siblings - {"cosmos_curator"}

    assert not third_party, f"{module_path.name} imports {sorted(third_party)} at module scope"


@pytest.mark.parametrize("module_path", SHIPPED_MODULES, ids=lambda path: path.name)
def test_onnode_modules_run_as_standalone_scripts(module_path: Path, tmp_path: Path) -> None:
    """Each module works when copied out of the package and run by path, as a run directory does."""
    _stage_run_directory(tmp_path)
    subprocess.run(  # noqa: S603
        [sys.executable, str(tmp_path / module_path.name), "--help"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )


@pytest.mark.parametrize("module_path", SHIPPED_MODULES, ids=lambda path: path.name)
def test_onnode_defers_annotations(module_path: Path) -> None:
    """Deferred annotations are what keep PEP 585 and PEP 604 syntax off the on-node floor."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    deferred = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(a.name == "annotations" for a in node.names)
        for node in tree.body
    )
    assert deferred, f"{module_path.name} must declare `from __future__ import annotations`"


@pytest.mark.parametrize("module_path", SHIPPED_MODULES, ids=lambda path: path.name)
def test_onnode_type_aliases_avoid_runtime_generics(module_path: Path) -> None:
    """A module-level alias is an ordinary expression, so deferred annotations do not cover it."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    builtin_generics = {"dict", "list", "set", "frozenset", "tuple", "type"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Subscript):
            continue
        base = node.value.value
        assert not (isinstance(base, ast.Name) and base.id in builtin_generics), (
            f"{module_path.name} subscripts the builtin {getattr(base, 'id', '?')!r} in a module-level alias, "
            f"which fails on Python {FLOOR_VERSION[0]}.{FLOOR_VERSION[1]}; use the typing equivalent"
        )


@pytest.mark.skipif(_floor_interpreter() is None, reason=FLOOR_SKIP_REASON)
def test_onnode_state_operations_run_on_the_floor_interpreter(tmp_path: Path) -> None:
    """The bare-host entry points work under the oldest python3 a supported cluster provides."""
    interpreter = _floor_interpreter()
    assert interpreter is not None
    _stage_run_directory(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(_FLOOR_MANIFEST), encoding="utf-8")

    def state(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [interpreter, str(tmp_path / STATE_MODULE_FILENAME), *arguments],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

    # Exercises the run lock and the manifest state machine, not just module import.
    result = state(
        "mutate",
        str(tmp_path / "manifest.json"),
        "cc-ray-0123456789ab",
        json.dumps({"operation": "finish-submission"}),
    )

    assert json.loads(result.stdout)["state"] == "STARTING"


@pytest.mark.skipif(_floor_interpreter() is None, reason=FLOOR_SKIP_REASON)
def test_floor_interpreter_is_actually_the_floor() -> None:
    """A skip is acceptable here; silently testing the development interpreter twice is not."""
    interpreter = _floor_interpreter()
    assert interpreter is not None
    reported = subprocess.run(  # noqa: S603
        [interpreter, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert reported == f"{FLOOR_VERSION[0]}.{FLOOR_VERSION[1]}"
