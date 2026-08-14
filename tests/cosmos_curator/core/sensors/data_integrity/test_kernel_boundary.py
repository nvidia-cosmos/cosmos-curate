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

"""Enforce that the data-integrity kernel stays a kernel.

The reusable half of data integrity is what the sensor library itself can depend on,
which only holds while it stays free of the things a workflow drags in: an argument
parser, a storage format, a cloud client. Those belong to
``cosmos_curator.next.recipes.data_integrity``, and each has a specific cost here --
``argparse`` would mean the metrics could only be reached through a CLI's idea of
validation, ``lance`` / ``pyarrow`` would put a columnar format in the import path of
a plain in-memory check, and a cloud client would contradict the backend-agnostic
contract documented in ``cosmos_curator/core/sensors/utils/io.py`` (the library takes
an open stream, never a URI).

The complementary direction -- no module under ``core.sensors`` importing
``cosmos_curator.next`` or anything else outside the sensors package -- is covered by
``tests/cosmos_curator/core/sensors/test_import_boundary.py``.
"""

import ast
from pathlib import Path

_KERNEL_ROOT = Path(__file__).parents[5] / "cosmos_curator" / "core" / "sensors" / "data_integrity"

#: Third-party modules whose presence would mean the kernel had grown a workflow
#: concern. Matched on the top-level name, so ``pyarrow.compute`` is caught too.
_FORBIDDEN_TOP_LEVEL = frozenset({"argparse", "boto3", "botocore", "lance", "pyarrow", "smart_open"})

#: The cloud helpers the recipe uses. Importable from here under the sensors-package
#: rule, which is exactly why it needs saying: the kernel must not reach for them.
_FORBIDDEN_PREFIX = "cosmos_curator.core.sensors.scripts"


def _is_forbidden(dotted: str) -> bool:
    """Whether an absolute dotted path names something the kernel must not import."""
    return bool(dotted) and (
        dotted.split(".", maxsplit=1)[0] in _FORBIDDEN_TOP_LEVEL or dotted.startswith(_FORBIDDEN_PREFIX)
    )


def _absolute_target(node: ast.ImportFrom, package: str) -> str:
    """Resolve what an ``ImportFrom`` names to an absolute dotted path.

    A relative import has to be resolved against the importing module's own package
    before it can be matched: ``from ..scripts._cli_cloud import ...`` carries the
    module as ``scripts._cli_cloud``, which matches no absolute rule here and would
    otherwise slip past. One level means the package itself, each further level
    strips a trailing component.
    """
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def _forbidden_imports(path: Path) -> list[str]:
    """Return any workflow-only imports found in *path*."""
    rel = path.relative_to(_KERNEL_ROOT.parents[3])
    package = ".".join(rel.parent.parts)
    violations: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        match node:
            case ast.Import(names=names):
                violations.extend(f"{rel}: import {alias.name}" for alias in names if _is_forbidden(alias.name))
            case ast.ImportFrom():
                target = _absolute_target(node, package)
                # The imported names are checked as submodules too, so that reaching the
                # package rather than the module (``from ...sensors import scripts``) is
                # caught alongside the direct spelling.
                if _is_forbidden(target) or any(_is_forbidden(f"{target}.{a.name}") for a in node.names):
                    violations.append(f"{rel}: from {target} import ...")
    return violations


def test_the_kernel_imports_no_cli_store_or_cloud_client() -> None:
    """Every module under core/sensors/data_integrity/ must stay free of workflow dependencies."""
    violations: list[str] = []
    for py_file in sorted(_KERNEL_ROOT.rglob("*.py")):
        violations.extend(_forbidden_imports(py_file))

    assert violations == [], (
        "the data-integrity kernel imports a workflow-only dependency; it belongs in "
        "cosmos_curator.next.recipes.data_integrity instead:\n" + "\n".join(f"  {v}" for v in violations)
    )
