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
"""Test nvcf response and nvcf client functionality."""

import json
from unittest.mock import MagicMock

import pytest

from cosmos_curate.client.nvcf_cli.ncf.common.nvcf_client import (  # type: ignore[import-untyped]
    NvcfClient,
    NVCFResponse,
)

_OK = 200
_NOT_FOUND = 404
_UNAUTHORIZED = 401
_FORBIDDEN = 403
_PRECONDITION_FAILED = 412
_TOO_MANY_REQUESTS = 429
_INTERNAL_SERVER_ERROR = 500
_TIMEOUT = 504


def test_nvcf_response_properties() -> None:
    """Test nvcf client properties."""
    response = NVCFResponse({"status": 200, "body": "test"})
    assert response.status == _OK
    assert response.has_status
    assert not response.is_not_found
    assert not response.is_not_authorized
    assert not response.is_error
    assert not response.is_timeout

    response = NVCFResponse({"status": 404, "body": "test"})
    assert response.status == _NOT_FOUND
    assert response.has_status
    assert response.is_not_found
    assert not response.is_not_authorized
    assert response.is_error

    response = NVCFResponse({"status": 401, "body": "test"})
    assert response.status == _UNAUTHORIZED
    assert response.has_status
    assert not response.is_not_found
    assert response.is_not_authorized
    assert response.is_error

    response = NVCFResponse()
    assert response.status == _INTERNAL_SERVER_ERROR


def test_nvcf_response_get_term_status() -> None:
    """Test nvcf response get term status."""
    response = NVCFResponse({"term-status": "test"})
    assert response.get_term_status() == "test"

    response = NVCFResponse({"body-status": "test"})
    assert response.get_term_status() == "test"

    response = NVCFResponse()
    assert response.get_term_status() == ""


def test_nvcf_response_get_detail() -> None:
    """Test nvcf response get detail."""
    response = NVCFResponse({"detail": "test"})
    assert response.get_detail() == "test"

    response = NVCFResponse({"issue": {"detail": "test"}})
    assert response.get_detail() == "test"

    response = NVCFResponse()
    assert response.get_detail() is None


def test_nvcf_response_get_error() -> None:
    """Test nvcf response get error."""
    response = NVCFResponse({"status": 400})
    assert response.get_error("test") == "invalid request for test"
    response = NVCFResponse({"status": 401})
    assert response.get_error("test") == "operation not authorized for test"
    response = NVCFResponse({"status": 403})
    assert response.get_error("test") == "operation not allowed for test"
    response = NVCFResponse({"status": 404})
    assert response.get_error("test") == "test not found"
    response = NVCFResponse({"status": 412})
    assert response.get_error("test") == "test precondition failed"
    response = NVCFResponse({"status": 429})
    assert response.get_error("test") == "test too many requests"
    response = NVCFResponse({"status": 500})
    assert response.get_error("test") == "unknown error occurred: status=500"

    response = NVCFResponse()
    assert response.get_error("test") == "unexpected empty response"


def test_get_success() -> None:
    """Test nvcf client get."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _OK
    mock_response.json.return_value = {"status": _OK}
    mock_response.text = json.dumps({"status": _OK})
    mock_response.headers = {}

    client.ses.get = MagicMock(return_value=mock_response)

    response = client.get("/test")
    assert response is not None
    assert response.status == _OK


def test_client_error() -> None:
    """Test nvcf client get error."""
    client = NvcfClient(logger=MagicMock(), url="")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _UNAUTHORIZED
    mock_response.json.return_value = {"status": _UNAUTHORIZED}
    mock_response.text = json.dumps({"status": _UNAUTHORIZED})

    client.ses.get = MagicMock(return_value=mock_response)
    client._handle_client_error = MagicMock(return_value=NVCFResponse({"status": _UNAUTHORIZED}))  # noqa: SLF001

    response = client.get("/test")
    assert response is not None
    assert response.status == _UNAUTHORIZED
    client._handle_client_error.assert_called()  # noqa: SLF001


def test_server_error() -> None:
    """Test nvcf client get server error."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _INTERNAL_SERVER_ERROR
    mock_response.json.return_value = {"status": _INTERNAL_SERVER_ERROR}
    mock_response.text = json.dumps({"status": _INTERNAL_SERVER_ERROR})

    client.ses.get = MagicMock(return_value=mock_response)

    with pytest.raises(RuntimeError):
        client.get("/test")


def test_get_timeout() -> None:
    """Test nvcf client timeout response."""
    client = NvcfClient(logger=MagicMock(), url="testurl")
    assert client is not None

    # Mock the session's get method
    mock_response = MagicMock()
    mock_response.status_code = _TIMEOUT
    mock_response.json.return_value = {"status": _TIMEOUT}

    client.ses.get = MagicMock(return_value=mock_response)

    response = client.get("/test")
    assert response is not None
    assert response.status == _TIMEOUT
    assert response.is_timeout
