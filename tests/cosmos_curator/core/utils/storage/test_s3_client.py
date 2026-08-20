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
"""Tests for S3 client listing semantics."""

import subprocess
import sys
from typing import Any

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3ClientConfig, S3Prefix


class _FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.last_paginate_kwargs: dict[str, object] | None = None

    def paginate(self, **kwargs: object) -> list[dict[str, Any]]:
        self.last_paginate_kwargs = kwargs
        return self._pages


class _FakeS3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.paginator = _FakePaginator(pages)

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return self.paginator


def test_list_recursive_respects_limit_within_large_page() -> None:
    """Trim results to exact limit when a single page contains more entries than requested."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        }
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_respects_limit_across_pages() -> None:
    """Trim to exact limit when overflow occurs after reading a subsequent page."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
            ]
        },
        {
            "Contents": [
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        },
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_without_limit_returns_all_pages() -> None:
    """Return all objects when no limit is specified."""
    pages = [
        {"Contents": [{"Key": "root/a.mp4"}]},
        {"Contents": [{"Key": "root/b.mp4"}]},
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=0)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_client_uses_configured_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile region must reach boto3 and win over the ambient environment."""
    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    client = S3Client(
        S3ClientConfig(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret",  # noqa: S106
            region="us-west-2",
        )
    )

    assert client.session.region_name == "us-west-2"
    assert client.s3.meta.region_name == "us-west-2"


def test_client_without_configured_region_defers_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No configured region leaves boto3's own resolution chain untouched."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    client = S3Client(
        S3ClientConfig(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret",  # noqa: S106
        )
    )

    assert client.s3.meta.region_name == "eu-central-1"


# Blocks ``ray`` at the import system level, then does what the client CLI does: import the
# module and validate an S3 location with ``S3Prefix``. Run as a subprocess so the guard is
# meaningful in environments that *do* have ray (``dev``, ``default``) and so a partially
# imported module cannot leak into the rest of the test session.
_IMPORT_WITHOUT_RAY = """
import sys


class _RayBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "ray" or name.startswith("ray."):
            msg = "ray is unavailable in the client-only 'tools' environment"
            raise ImportError(msg)
        return None


sys.meta_path.insert(0, _RayBlocker())

from cosmos_curator.core.utils.storage.s3_client import S3Prefix

assert S3Prefix("s3://some-bucket/some/key").bucket == "some-bucket"
assert "ray" not in sys.modules, "importing s3_client pulled in ray"
"""


def test_module_imports_without_ray() -> None:
    """Importing this module must not require ray.

    ``S3Prefix`` is pure string validation, and the client CLI uses it to validate ``s3://``
    locations in ``cosmos-curator pipeline validate``. The client-only ``tools`` pixi
    environment has no ray, so a module-scope ``nvcf_utils`` import (which imports ray) makes
    validating any S3 config fail with ``ModuleNotFoundError: No module named 'ray'``.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _IMPORT_WITHOUT_RAY],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"s3_client is not importable without ray:\n{result.stderr}"
