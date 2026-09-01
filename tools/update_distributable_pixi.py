# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
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

"""Generate the redistributable Pixi manifest from the main manifest."""

import argparse
import copy
import difflib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_PATH = REPO_ROOT / "pixi.toml"
TARGET_MANIFEST_PATH = REPO_ROOT / "distributable" / "pixi.toml"
TARGET_LOCK_PATH = REPO_ROOT / "distributable" / "pixi.lock"
UPDATE_COMMAND = "pixi run -e tools python tools/update_distributable_pixi.py"

IMAGE_ENVIRONMENTS = (
    "cuml",
    "default",
    "legacy-transformers",
    "model-download",
    "paddle-ocr",
    "seedvr",
    "sam3",
    "style-transfer",
)
EXCLUDED_FEATURES = {"tools", "cluster", "dev"}
MEDIA_CONDA_PACKAGES = {"av", "ffmpeg", "libopencv", "opencv", "py-opencv"}
MEDIA_PYPI_PACKAGES = {
    "av": "==17.0.0",
    "opencv-python-headless": "*",
    # av==17.0.0 builds from source here: PyPI's prebuilt wheels bundle ffmpeg's H.264
    # codecs, which this image can't redistribute (royalty encumbrance), so the build
    # links against our own ffmpeg instead. That source build uses --no-build-isolation,
    # so it compiles against whatever Cython is already resolved in this environment.
    # Cython >=3.3 got stricter about redeclaring a Cython-typed local inside a
    # conditional (av/container/pyio.py:40), which av 17.0.0 hits and fails to build.
    # Cap below that until av's fix lands in a release compatible with this pin.
    "cython": "<3.3",
}
TERMINAL_TABLE_NAMES = {
    "dependencies",
    "pypi-dependencies",
    "tasks",
    "system-requirements",
    "environments",
    "workspace",
    "env",
    "dependency-overrides",
}


def main() -> int:
    """Run the distributable Pixi manifest generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if distributable/pixi.toml or distributable/pixi.lock is not up to date.",
    )
    args = parser.parse_args()

    generated = generate_manifest()
    if args.check:
        current = TARGET_MANIFEST_PATH.read_text(encoding="utf-8") if TARGET_MANIFEST_PATH.exists() else ""
        if current != generated:
            _print_manifest_diff(current, generated)
            _print_update_hint(f"{_repo_relative(TARGET_MANIFEST_PATH)} is generated and out of date.")
            return 1
        return _run_pixi_lock(check=True)

    TARGET_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_MANIFEST_PATH.write_text(generated, encoding="utf-8")
    return _run_pixi_lock(check=False)


def generate_manifest() -> str:
    """Return the generated distributable Pixi manifest text."""
    source = _load_toml(SOURCE_MANIFEST_PATH)
    distributable = _build_distributable_config(source)
    return _render_manifest(distributable)


def _print_manifest_diff(current: str, generated: str) -> None:
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=str(TARGET_MANIFEST_PATH),
        tofile=f"{TARGET_MANIFEST_PATH} (generated)",
    )
    sys.stderr.writelines(diff)


def _run_pixi_lock(*, check: bool) -> int:
    pixi = shutil.which("pixi")
    if pixi is None:
        sys.stderr.write("ERROR: pixi executable not found on PATH; cannot validate the distributable lockfile.\n")
        action = "checked" if check else "regenerated"
        _print_update_hint(f"{_repo_relative(TARGET_LOCK_PATH)} could not be {action}.")
        return 1

    command = [pixi, "lock", "--manifest-path", str(_repo_relative(TARGET_MANIFEST_PATH))]
    if check:
        command.append("--check")

    result = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603
    if result.returncode != 0:
        _print_update_hint(f"{_repo_relative(TARGET_LOCK_PATH)} is out of date or failed Pixi validation.")
    return result.returncode


def _print_update_hint(reason: str) -> None:
    sys.stderr.write(
        f"\n{reason}\n\n"
        "Regenerate the redistributable Pixi files with:\n"
        f"  {UPDATE_COMMAND}\n\n"
        f"That command updates {_repo_relative(TARGET_MANIFEST_PATH)} and {_repo_relative(TARGET_LOCK_PATH)}.\n",
    )


def _repo_relative(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        msg = f"{path} did not parse as a TOML table"
        raise TypeError(msg)
    return data


def _build_distributable_config(source: dict[str, Any]) -> dict[str, Any]:
    environments = {
        name: copy.deepcopy(features) for name, features in source["environments"].items() if name in IMAGE_ENVIRONMENTS
    }
    missing_envs = sorted(set(IMAGE_ENVIRONMENTS) - set(environments))
    if missing_envs:
        msg = f"missing image environments in {SOURCE_MANIFEST_PATH}: {missing_envs}"
        raise ValueError(msg)

    included_features = {
        feature_name
        for feature_names in environments.values()
        for feature_name in feature_names
        if feature_name not in EXCLUDED_FEATURES
    }

    workspace = copy.deepcopy(source["workspace"])
    workspace.pop("conda-pypi-map", None)
    workspace["name"] = f"{workspace['name']}-distributable"

    features = {}
    for feature_name, feature in source["feature"].items():
        if feature_name not in included_features:
            continue
        feature_config = copy.deepcopy(feature)
        if feature_name == "media":
            feature_config = _redistributable_media_feature(feature_config)
        features[feature_name] = feature_config

    return {
        "workspace": workspace,
        "activation": copy.deepcopy(source.get("activation", {})),
        "dependencies": copy.deepcopy(source["dependencies"]),
        "feature": features,
        "environments": environments,
    }


def _redistributable_media_feature(feature_config: dict[str, Any]) -> dict[str, Any]:
    conda_deps = feature_config.get("dependencies", {})
    if not isinstance(conda_deps, dict):
        msg = "feature.media.dependencies must be a table"
        raise TypeError(msg)
    for package_name in MEDIA_CONDA_PACKAGES:
        conda_deps.pop(package_name, None)
    if conda_deps:
        feature_config["dependencies"] = conda_deps
    else:
        feature_config.pop("dependencies", None)

    pypi_deps = feature_config.setdefault("pypi-dependencies", {})
    if not isinstance(pypi_deps, dict):
        msg = "feature.media.pypi-dependencies must be a table"
        raise TypeError(msg)
    pypi_deps.update(MEDIA_PYPI_PACKAGES)
    return feature_config


def _render_manifest(config: dict[str, Any]) -> str:
    lines = [
        "# This file is generated by tools/update_distributable_pixi.py. Do not edit by hand.",
        "",
    ]
    _append_table(lines, ("workspace",), config["workspace"])
    if config.get("activation"):
        _append_table(lines, ("activation",), config["activation"])
    _append_table(lines, ("dependencies",), config["dependencies"])

    for feature_name, feature_config in config["feature"].items():
        _append_table(lines, ("feature", feature_name), feature_config)

    _append_table(lines, ("environments",), config["environments"])
    return "\n".join(lines).rstrip() + "\n"


def _append_table(lines: list[str], path: tuple[str, ...], table: dict[str, Any]) -> None:
    entries: list[tuple[str, Any]] = []
    subtables: list[tuple[str, dict[str, Any]]] = []

    for key, value in table.items():
        if isinstance(value, dict) and _should_descend(path):
            subtables.append((key, value))
        else:
            entries.append((key, value))

    if entries:
        lines.append(f"[{'.'.join(path)}]")
        for key, value in entries:
            lines.append(f"{key} = {_format_value(value)}")
        lines.append("")

    for key, value in subtables:
        _append_table(lines, (*path, key), value)


def _should_descend(path: tuple[str, ...]) -> bool:
    return not path or path[-1] not in TERMINAL_TABLE_NAMES


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_format_value(item)}" for key, item in value.items()) + " }"
    msg = f"unsupported TOML value: {value!r}"
    raise TypeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
