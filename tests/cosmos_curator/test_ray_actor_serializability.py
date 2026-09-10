# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every Ray actor class in this package must survive cloudpickle.

``@ray.remote`` rebinds the module attribute to an ``ActorClass``, so
cloudpickle's by-reference lookup for the underlying methods fails and the class
is serialized *by value* -- which pickles every global those methods reference.
Anything unpicklable reachable that way (locks, sockets, file handles, thread
pools, loaded models) breaks actor creation at runtime inside the deployed
function, with nothing failing at build time.

The check is deliberately generic: it dumps each actor class and reports
whatever comes back. The fixture installs the JSON logging sink shape so the
sweep exercises deployed logging configuration rather than whatever handlers
the test process happens to have.
"""

import ast
import contextlib
import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

cloudpickle = pytest.importorskip("ray.cloudpickle")
ray = pytest.importorskip("ray")

_PACKAGE = "cosmos_curator"


def _references_ray_remote(tree: ast.AST) -> bool:
    """Return whether the parsed module refers to ``ray.remote`` in any form.

    Covers ``@ray.remote``, ``@ray.remote(...)``, ``@remote`` after
    ``from ray import remote``, and the plain call form ``Foo = ray.remote(Foo)``.
    A substring search would miss the aliased and call forms and would also match
    comments and docstrings.
    """
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "ray"
        for alias in node.names
        if alias.name == "remote"
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "remote":
            value = node.value
            if isinstance(value, ast.Name) and value.id == "ray":
                return True
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
    return False


def _modules_declaring_actors() -> list[str]:
    """Return module names in this package that reference ``ray.remote``.

    Discovery is static so the sweep does not have to import the entire package
    (most of which pulls in heavy optional dependencies). Enumeration of the
    actual actor classes is dynamic in the test body.
    """
    root = Path(importlib.import_module(_PACKAGE).__file__).parent
    modules: list[str] = []
    for source_path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - a broken file is another test's failure
            continue
        if not _references_ray_remote(tree):
            continue
        parts = source_path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join((_PACKAGE, *parts)))
    return modules


@contextlib.contextmanager
def _json_mode_sink() -> Iterator[None]:
    """Use Xenna's deployed JSON logging configuration while serializing actors.

    This runs after the module under test is imported because module imports can
    configure Xenna logging and replace sinks installed earlier.
    """
    previous_format = os.environ.get("PYTHON_LOG_FORMAT")
    os.environ["PYTHON_LOG_FORMAT"] = "json"
    python_log = pytest.importorskip("cosmos_xenna.utils.python_log")

    python_log.ensure_configured(force=True)
    try:
        yield
    finally:
        if previous_format is None:
            os.environ.pop("PYTHON_LOG_FORMAT", None)
        else:
            os.environ["PYTHON_LOG_FORMAT"] = previous_format
        python_log.ensure_configured(force=True)


@pytest.mark.parametrize("module_name", _modules_declaring_actors())
def test_ray_actor_classes_are_serializable(module_name: str) -> None:
    """Actor classes in this module must pickle without unpicklable globals."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 an unimportable module is an environment gap
        pytest.skip(f"cannot import {module_name}: {type(exc).__name__}: {exc}")

    actor_classes = {name: obj for name, obj in vars(module).items() if isinstance(obj, ray.actor.ActorClass)}
    if not actor_classes:
        pytest.skip(f"{module_name} references ray.remote but exposes no actor class at import time")

    failures: list[str] = []
    with _json_mode_sink():
        for name, actor_class in actor_classes.items():
            try:
                cloudpickle.dumps(actor_class.__ray_metadata__.modified_class)
            except Exception as exc:  # noqa: BLE001 report every serialization failure
                failures.append(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")

    assert not failures, f"actor classes in {module_name} are not serializable:\n" + "\n".join(failures)
