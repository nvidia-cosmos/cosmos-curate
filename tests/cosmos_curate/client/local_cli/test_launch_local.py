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
"""Test cosmos-curate local commands."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curate.client.cli import cosmos_curator  # type: ignore[import-untyped]
from cosmos_curate.client.local_cli.launch_local import (  # type: ignore[import-untyped]
    _get_config_file_mount_strings,
    _verify_local_path_exists,
)

runner = CliRunner()


def test_launch_command() -> None:
    """Test that docker run command forms.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    args = [
        "local",
        "launch",
        "--image-name",
        "cosmos-curate",
        "--image-tag",
        "hello-world",
        "--curator-path",
        ".",
        "--",
        "python3",
        "-m",
        "cosmos_curate.pipelines.examples.hello_world_pipeline",
    ]
    with patch("cosmos_curate.client.local_cli.launch_local.subprocess.call") as mock_call:
        mock_call.return_value = 0

        result = runner.invoke(cosmos_curator, args)

        assert result.exit_code == 0

        # Extract the actual command that would be run
        called_args = mock_call.call_args[0][0]

    # check docker command is formed
    assert called_args[:3] == ["docker", "run", "--rm"]


def test_verify_local_path_exists(monkeypatch: MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test verify_local_path_exists function.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.
        caplog: LogCaptureFixture object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mock_path_one = tmp_path / "test1"
    mock_path_one.touch()
    mock_path_two = tmp_path / "test2"
    mock_path_two.touch()
    mock_path_fail = tmp_path / "failure"

    with caplog.at_level(logging.DEBUG):
        mock_paths = [Path(str(mock_path_one)), Path(str(mock_path_two))]
        mock_paths_fail = [Path(str(mock_path_fail))]
        _verify_local_path_exists(mock_paths)

        assert "ERROR" not in caplog.text

        with pytest.raises(SystemExit):
            _verify_local_path_exists(mock_paths_fail)


def test_get_config_file_mount_string(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test get_config_file_mount_string function.

    Args:
        monkeypatch: The monkeypatch object.
        tmp_path: The tmp_path object.

    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mock_file = tmp_path / "mock_config.yaml"
    mock_file.write_text("anything")

    # check correctness when file exists
    with patch("cosmos_curate.client.local_cli.launch_local.LOCAL_COSMOS_CURATOR_CONFIG_FILE", mock_file):
        result = _get_config_file_mount_strings(is_model_cli=True)

    assert str(mock_file) in result[1]

    # check correctness when file does not exist
    mock_file_fail = tmp_path / "mock_fail.yaml"

    with patch("cosmos_curate.client.local_cli.launch_local.LOCAL_COSMOS_CURATOR_CONFIG_FILE", mock_file_fail):  # noqa: SIM117
        with pytest.raises(SystemExit):
            _get_config_file_mount_strings(is_model_cli=True)
