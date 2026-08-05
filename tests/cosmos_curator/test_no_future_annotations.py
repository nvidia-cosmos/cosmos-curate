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

"""Guard against reintroducing postponed annotation evaluation imports."""

from pathlib import Path

_FORBIDDEN_IMPORT = "from __future__ import annotations"
_REPO_ROOT = Path(__file__).parents[2]
_SCAN_ROOTS = (
    _REPO_ROOT / "cosmos_curator",
    _REPO_ROOT / "tests",
)
# The rule assumes Python 3.12, where native annotations always evaluate. These modules run on a node's own
# python3, so deferred annotations are what keeps PEP 585 and PEP 604 out of their runtime, and their own
# contract tests require the import this one forbids. See the onnode package docstring.
_EXEMPT_DIR = _REPO_ROOT / "cosmos_curator" / "client" / "slurm_cli" / "managed_ray" / "onnode"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        files.extend(path for path in root.rglob("*.py") if path.parent != _EXEMPT_DIR)
    return sorted(files)


def test_exemption_covers_only_the_on_node_modules() -> None:
    """The exemption is a directory, so it must not quietly widen as the launcher grows."""
    exempt = sorted(path.name for path in _EXEMPT_DIR.glob("*.py"))
    assert exempt == ["__init__.py", "slurm_ray_runtime.py", "slurm_ray_state.py"]


def _code_before_comment(line: str) -> str:
    return line.split("#", maxsplit=1)[0].strip()


def test_code_before_comment_strips_trailing_comments() -> None:
    """Verify the guard catches future imports with trailing comments."""
    assert _code_before_comment("from __future__ import annotations  # noqa") == _FORBIDDEN_IMPORT
    assert _code_before_comment("from __future__ import annotations  # type: ignore") == _FORBIDDEN_IMPORT
    assert _code_before_comment("\tfrom __future__ import annotations  # type: ignore") == _FORBIDDEN_IMPORT


def test_future_annotations_import_is_not_used() -> None:
    """Verify first-party Python files do not import future annotations."""
    violations: list[str] = []

    for path in _python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _code_before_comment(line) == _FORBIDDEN_IMPORT:
                rel_path = path.relative_to(_REPO_ROOT)
                violations.append(f"{rel_path}:{line_number}")

    assert violations == [], f"{_FORBIDDEN_IMPORT!r} is not allowed in Cosmos Curate Python files:\n" + "\n".join(
        f"  {violation}" for violation in violations
    )
