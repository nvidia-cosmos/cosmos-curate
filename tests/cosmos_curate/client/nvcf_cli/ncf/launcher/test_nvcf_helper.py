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

"""Test the NVCF Helper Module."""

import json
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from cosmos_curate.client.nvcf_cli.ncf.launcher.nvcf_helper import (  # type: ignore[import-untyped]
    NvcfHelper,
    _raise_runtime_err,
    _raise_timeout_err,
)


def test_raise_runtime_err() -> None:
    """Test that _raise_runtime_err raises a RuntimeError with the given message."""
    with pytest.raises(RuntimeError):
        _raise_runtime_err("test message")

    with pytest.raises(RuntimeError):
        _raise_runtime_err({"error": "test message"})


def test_raise_timeout_err() -> None:
    """Test that _raise_timeout_err raises a TimeoutError with the given message."""
    with pytest.raises(TimeoutError):
        _raise_timeout_err("test message")

    with pytest.raises(TimeoutError):
        _raise_timeout_err({"error": "test message"})


def test_load_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that load_ids returns the expected dictionary."""
    config_dir = tmp_path / ".config/cosmos_curate"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", timeout=15)
    assert nvcf_helper.load_ids() == {"name": "test_name", "id": "test_id", "version": "test_version"}


def test_store_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that store_ids stores the given dictionary to the file."""
    config_dir = tmp_path / ".config/cosmos_curate"
    config_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", timeout=15)
    nvcf_helper.store_ids({"name": "test_name", "id": "test_id", "version": "test_version"})
    assert nvcf_helper.load_ids() == {"name": "test_name", "id": "test_id", "version": "test_version"}


def test_cleanup_ids(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that cleanup_ids removes the file."""
    config_dir = tmp_path / ".config/cosmos_curate"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", timeout=15)
    nvcf_helper.cleanup_ids()
    assert not fname.exists()


def test_id_version(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that id_version returns the expected tuple."""
    config_dir = tmp_path / ".config/cosmos_curate"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "funcid.json"
    Path.open(fname, "w")

    # Check empty file first
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nvcf_helper = NvcfHelper(url="", nvcf_url="", key="", org="", timeout=15)
    assert nvcf_helper.id_version(None, None) == (False, None, None)
    assert nvcf_helper.id_version(None, "test_version") == (False, None, "test_version")
    assert nvcf_helper.id_version("test_id", "test_version") == (True, "test_id", "test_version")

    # File file with fake data
    with Path.open(fname, "w") as f:
        json.dump({"name": "test_name", "id": "test_id", "version": "test_version"}, f)

    assert nvcf_helper.id_version(None, "test_version") == (True, "test_id", "test_version")
    assert nvcf_helper.id_version("test_id", "test_version") == (True, "test_id", "test_version")
