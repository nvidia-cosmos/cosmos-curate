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
"""Test nvcf base functionality."""

from pathlib import Path
from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curate.client.nvcf_cli.ncf.common import base_callback  # type: ignore[import-untyped]

runner = CliRunner()

_TIMEOUT = 15


def test_base_callback(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test that base_callback function sets up config defaults.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Define a temporary directory for the config file
    config_dir = tmp_path / ".cosmos_curate"
    config_dir.mkdir(parents=True, exist_ok=True)
    fname = config_dir / "config.json"
    Path.open(fname, "w")

    fake_ctx = MagicMock()
    fake_ctx.command.name = "config"
    fake_ctx.obj = {
        "url": None,
        "nvcf_url": None,
        "key": "BASE",
        "org": "NVIDIA",
        "timeout": None,
        "config": None,
        "nvcfHdl": None,
        "test": True,
    }

    base_callback(ctx=fake_ctx)
    # assert that values have been changed to defaults
    assert fake_ctx.obj["url"] == "https://api.ngc.nvidia.com"
    assert fake_ctx.obj["nvcf_url"] == "https://api.nvcf.nvidia.com"
    assert fake_ctx.obj["key"] is None
    assert fake_ctx.obj["org"] is None
    assert fake_ctx.obj["timeout"] == _TIMEOUT
