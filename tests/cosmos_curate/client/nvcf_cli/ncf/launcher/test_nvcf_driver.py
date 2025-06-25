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
"""Test nvcf driver commands."""

import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cosmos_curate.client.cli import cosmos_curator  # type: ignore[import-not-found]

runner = CliRunner()


def test_deploy_function_no_id_version(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that  deploy-function fails with no id.

    Args:
        caplog: To capture the logs.
        tmp_path: A temporary path for tests.

    """
    mock_func = MagicMock()

    # A mock data file to pass to the command
    mock_func.id_version.return_value = (False, 1234, 5678)
    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    with caplog.at_level(logging.DEBUG):
        runner.invoke(cosmos_curator, args)

    assert "ERROR" in caplog.text
    assert "Could not deploy function" in caplog.text


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_no_backend_gpu_instance(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with no backened, gpu, or instance.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # A mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # A mock data file to pass to the command
    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    # Mock return values to test backend = None
    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, None, None, None)

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: backend is required")
    assert result.exit_code != 0

    # Mock return values to test gpu = None
    mock_instance.get_cluster.return_value = (True, True, None, None)
    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: gpu is required")
    assert result.exit_code != 0

    # Mock return values to test instance = None
    mock_instance.get_cluster.return_value = (True, True, True, None)
    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: instance is required")
    assert result.exit_code != 0


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_success(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function can succeeed.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    # Mock file to pass as arguement
    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, True, True, True)

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.console.print.assert_called_with(
        "Function with id '1234' and version '5678' is being deployed; "
        "to check status: cosmos-curate nvcf function get-deployment-detail"
    )
    assert result.exit_code == 0


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_exception(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with an exception.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary Path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)
    mock_instance.get_cluster.return_value = (True, True, True, True)
    mock_instance.nvcf_helper_deploy_function.side_effect = RuntimeError("mock exception")

    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not deploy function: mock exception")
    assert result.exit_code == 1


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_deploy_function_validation_failures(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that deploy-function fails with invalid arguments.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"

    # file does not exist
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)
    args = ["nvcf", "function", "deploy-function", "--data-file", str(fname), "-version", "badversion"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    # negatitive number
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
        "--min-instances",
        "-5",
    ]
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_no_id_version(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails with no id.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    mock_instance.id_version.return_value = (False, 1234, 5678)

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with("Could not invoke function: id and version are required")
    assert result.exit_code == 1


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_bad_assetid(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails with a bad assetid.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)
    # A mock uuid
    my_uuid = uuid.uuid4()

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
        "--assetid",
        str(my_uuid),
        "--assetfile",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.logger.error.assert_called_with(
        "Could not invoke function: assetid and assetfile are mutually exclusive"
    )
    assert result.exit_code == 1


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_test_wait(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function call correct function with wait.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"
    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)

    args = [
        "nvcf",
        "function",
        "invoke-function",
        "--data-file",
        str(fname),
        "--wait",
    ]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.nvcf_helper_invoke_wait_retry_function.assert_called()
    assert result.exit_code == 0

    args = ["nvcf", "function", "invoke-function", "--data-file", str(fname), "--no-wait"]

    result = runner.invoke(cosmos_curator, args)
    mock_instance.nvcf_helper_invoke_function.assert_called()
    mock_instance.console.print.assert_called()
    assert result.exit_code == 0


@patch("cosmos_curate.client.nvcf_cli.ncf.common.nvcf_base.cc_client_instances")
def test_invoke_function_validation_failures(mock_cc: MagicMock, tmp_path: Path) -> None:
    """Test that invoke-function fails validation errors.

    Args:
        mock_cc: A mock to return a fake helper.
        tmp_path: A temporary path.

    """
    # Mock NvfcHelper
    mock_instance = MagicMock()
    mock_func = MagicMock(return_value=mock_instance)
    mock_cc.return_value = {"function": {"help": "A fake function", "type": mock_func}}

    fname = tmp_path / "fake.json"

    # file does not exist
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
    ]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    Path.open(fname, "w")

    mock_instance.id_version.return_value = (True, 1234, 5678)
    args = ["nvcf", "function", "deploy-function", "--data-file", str(fname), "-assetid", "badassetid"]

    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0

    # negatitive number
    args = [
        "nvcf",
        "function",
        "deploy-function",
        "--data-file",
        str(fname),
        "--retry_cnt",
        "-5",
    ]
    result = runner.invoke(cosmos_curator, args)
    assert result.exit_code != 0
