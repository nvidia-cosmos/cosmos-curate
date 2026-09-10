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

"""Import-layer guards for the embeddings package.

Two invariants keep the package CPU-testable and its layering one-way:

1. Importing ``cosmos_curator.next.embeddings`` must not pull in the heavy
   inference/decoding stack (``torch`` / ``transformers`` /
   ``sentence_transformers`` / ``av``). Those are imported lazily inside the
   actor bodies; if that discipline rots, the "pure" suite silently starts
   needing a GPU environment. Checked in a *fresh* interpreter so another test
   importing torch first cannot mask a regression.
2. No module under ``next/embeddings`` imports from ``next/recipes`` - the
   component layer must stay reusable and free of the recipe layer, which is
   the one plausible circular-import risk in the design.
"""

import ast
import importlib.util
import pathlib
import subprocess
import sys

import cosmos_curator.next.embeddings as embeddings_pkg

_HEAVY_MODULES = ("torch", "transformers", "sentence_transformers", "av")


def _module_package(path: pathlib.Path, package_root: pathlib.Path, root_module: str) -> str:
    """Return the dotted ``__package__`` a relative import in ``path`` resolves against.

    Both a regular module (``pkg/a.py``) and a package (``pkg/__init__.py``) have
    ``__package__`` equal to the dotted path of the file's parent directory, so a
    single rule anchors relative imports in either.
    """
    relative_parent = path.parent.relative_to(package_root)
    if not relative_parent.parts:
        return root_module
    return f"{root_module}." + ".".join(relative_parent.parts)


def _import_from_module(node: ast.ImportFrom, package: str) -> str:
    """Resolve an ``ast.ImportFrom`` to the absolute module name it imports from.

    A relative import (``node.level > 0``, e.g. ``from ..recipes import x``) is
    normalized against ``package`` so the recipe-layer prefix check sees the same
    absolute name Python would import; an absolute import passes through unchanged.
    """
    if not node.level:
        return node.module or ""
    relative = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative, package)


def _import_from_targets(node: ast.ImportFrom, package: str) -> list[str]:
    """Return the module imported FROM plus each name imported out of it.

    Recording the imported names (not just the base module) is what catches the
    submodule-by-name form: ``from cosmos_curator.next import recipes`` (or the
    relative ``from .. import recipes``) resolves the base to
    ``cosmos_curator.next``, so only ``base + ".recipes"`` reveals the forbidden
    recipe-layer import.
    """
    base = _import_from_module(node, package)
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def test_importing_package_does_not_load_heavy_deps(repo_root: pathlib.Path) -> None:
    """A fresh import of the package leaves torch/transformers/etc. unloaded."""
    code = (
        "import sys, importlib;"
        "importlib.import_module('cosmos_curator.next.embeddings');"
        "print(' '.join(m for m in "
        f"{_HEAVY_MODULES!r}"
        " if m in sys.modules))"
    )
    result = subprocess.run(  # noqa: S603 - fixed args, our own interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )
    assert result.returncode == 0, f"importing the package failed:\n{result.stderr}"
    loaded = result.stdout.split()
    assert loaded == [], f"heavy deps imported at package load time: {loaded}"


def test_no_embeddings_module_imports_the_recipe_layer() -> None:
    """The component layer never imports ``next.recipes`` (one-way dependency)."""
    package_root = pathlib.Path(embeddings_pkg.__file__).parent
    root_module = embeddings_pkg.__name__
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        package = _module_package(path, package_root, root_module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = _import_from_targets(node, package)
            if any(name.startswith("cosmos_curator.next.recipes") for name in names):
                offenders.append(path.name)
    assert offenders == [], f"embeddings modules importing the recipe layer: {sorted(set(offenders))}"


def test_import_from_submodule_by_name_is_detected() -> None:
    """Importing the recipe layer as a named submodule is caught, absolute or relative.

    ``from cosmos_curator.next import recipes`` and ``from .. import recipes``
    both name the recipe *package* rather than a symbol inside it, so the scanner
    must resolve the imported name (not just the base module) to the recipe-layer
    prefix.
    """
    absolute = ast.parse("from cosmos_curator.next import recipes").body[0]
    assert isinstance(absolute, ast.ImportFrom)
    assert "cosmos_curator.next.recipes" in _import_from_targets(absolute, "cosmos_curator.next.embeddings")

    relative = ast.parse("from .. import recipes").body[0]
    assert isinstance(relative, ast.ImportFrom)
    assert "cosmos_curator.next.recipes" in _import_from_targets(relative, "cosmos_curator.next.embeddings")
