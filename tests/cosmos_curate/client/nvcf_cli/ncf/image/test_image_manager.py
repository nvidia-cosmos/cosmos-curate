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
"""Test nvcf image commands."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from cosmos_curate.client.cli import cosmos_curator  # type: ignore[import-untyped]

runner = CliRunner()


def test_success_list_every_image() -> None:
    """Test that all images are displayed correctly.

    Args:
        mock_client:  Mock the client for tests.


    """
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)

    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items

    # show all images
    args = ["nvcf", "image", "list-images", "--all-images"]
    with patch("cosmos_curate.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)

    output = result.stdout

    # check that table appears
    assert "All Images that My Org Has Access to:" in output
    assert "Name" in output
    assert "Tag" in output
    assert "Size" in output

    assert "Org" in output
    assert "Teams" in output

    # check that image1 appears
    assert "image1" in output
    assert "v1" in output

    # check that image2 appears
    assert "image2" in output
    assert "v2" in output


def test_success_list_own_image() -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "list-images",
    ]
    with patch("cosmos_curate.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
        result = runner.invoke(cosmos_curator, args)
    output = result.stdout
    # check that table appears
    assert "All Images from My Org:" in output
    assert "Name" in output
    assert "Tag" in output
    assert "Size" in output
    assert "Org" in output
    assert "Teams" in output
    # check that image1 appears
    # assert "image1" in output
    assert "v1" in output
    # check that image2 does not appear
    assert "image2" not in output
    assert "v2" not in output


def test_upload_image(monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
    ]
    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.list.return_value = fake_items
    mock_image.create.return_value = 1
    mock_image.push.return_value = 1

    # fake image to upload
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    fname = tmp_path / "fakeimage.json"
    with Path.open(fname, "w") as f:
        json.dump({"image": "testimage", "tag": "testtag"}, f)

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "upload-image",
        "--data-file",
        str(fname),
    ]
    with caplog.at_level(logging.DEBUG):  # noqa: SIM117
        with patch("cosmos_curate.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
            runner.invoke(cosmos_curator, args)

    # assert upload message appears
    assert "testimage" in caplog.text
    assert "testtag" in caplog.text


def test_delete_image(caplog: pytest.LogCaptureFixture) -> None:
    """Test that your org images are displayed correctly."""
    # Define some fake images
    mock_image = Mock()
    fake_items = [
        [Mock(toDict=lambda: {"name": "image1", "latestTag": "v1", "sharedWithOrgs": "1"})],
        [Mock(toDict=lambda: {"name": "image2", "latestTag": "v2", "sharedWithOrgs": "2"})],
        [Mock(toDict=lambda: {"name": "1/todelete", "latestTag": "v3", "sharedWithOrgs": "1"})],
    ]

    def remove_mock_by_name(pattern: str, *, default_yes: bool = True) -> None:
        """Fake remove function.

        Args:
            pattern: identify images to remove.
            default_yes: remove the image.

        """
        if default_yes:
            nonlocal fake_items
            fake_items[:] = [image for image in fake_items if image[0].toDict()["name"] != pattern]

    mock_registry = Mock(image=mock_image)
    # Mock the client instance
    mock_client_instance = Mock(registry=mock_registry)
    mock_image.remove = remove_mock_by_name
    mock_image.list.return_value = fake_items

    # show all images
    args = [
        "nvcf",
        "image",
        "--org",
        "1",
        "list-images",
    ]

    args2 = [
        "nvcf",
        "image",
        "--org",
        "1",
        "delete-image",
        "--iname",
        "todelete",
    ]
    with caplog.at_level(logging.DEBUG):  # noqa: SIM117
        with patch("cosmos_curate.client.nvcf_cli.ncf.image.image_manager.Client", return_value=mock_client_instance):
            result = runner.invoke(cosmos_curator, args)
            runner.invoke(cosmos_curator, args2)
            result_del = runner.invoke(cosmos_curator, args)

    assert "image1" in result.output
    assert "todel" in result.output

    # after deletion
    assert "image1" in result_del.output
    assert "todel" not in result_del.output
