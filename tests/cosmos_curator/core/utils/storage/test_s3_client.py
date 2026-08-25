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
"""Tests for S3 client listing and download semantics."""

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cosmos_curator.core.utils.storage import s3_client
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
        self.uploads: list[tuple[str, str, str]] = []

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return self.paginator

    def upload_file(self, local_path: str, bucket: str, prefix: str, **_kwargs: object) -> None:
        self.uploads.append((local_path, bucket, prefix))


class _TraceLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def trace(self, message: str) -> None:
        self.messages.append(message)


class _FakeStreamingBody:
    """Botocore StreamingBody stand-in that records its read and close calls.

    ``raise_on_read`` makes ``read`` fail the way a connection dropped mid-stream
    would, so a test can assert the stream is still closed on that path.
    """

    def __init__(self, payload: bytes, *, raise_on_read: bool = False) -> None:
        self._payload = payload
        self._raise_on_read = raise_on_read
        self.read_count = 0
        self.closed = False

    def read(self) -> bytes:
        self.read_count += 1
        if self._raise_on_read:
            msg = "connection reset mid-stream"
            raise OSError(msg)
        return self._payload

    def close(self) -> None:
        self.closed = True


class _FakeDownloadS3:
    """Single-object S3 stand-in that records which download shape was used.

    Passing ``report_content_length=False`` omits ``ContentLength`` from the
    ``get_object`` response, leaving the object unmeasured. Passing
    ``raise_on_read=True`` makes the served body fail when it is read.
    """

    def __init__(self, payload: bytes, *, report_content_length: bool = True, raise_on_read: bool = False) -> None:
        self._payload = payload
        self._report_content_length = report_content_length
        self._raise_on_read = raise_on_read
        self.bodies: list[_FakeStreamingBody] = []
        self.get_object_calls: list[dict[str, object]] = []
        self.download_fileobj_calls: list[tuple[str, str]] = []

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        self.get_object_calls.append(kwargs)
        body = _FakeStreamingBody(self._payload, raise_on_read=self._raise_on_read)
        self.bodies.append(body)
        response: dict[str, Any] = {"Body": body}
        if self._report_content_length:
            response["ContentLength"] = len(self._payload)
        return response

    def download_fileobj(self, bucket: str, key: str, fileobj: io.BytesIO, **_kwargs: object) -> None:
        self.download_fileobj_calls.append((bucket, key))
        fileobj.write(self._payload)


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


def test_upload_file_emits_one_trace_instead_of_per_object_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bulk uploads stay quiet at normal log levels while retaining opt-in detail."""
    fake_s3 = _FakeS3([])
    trace_logger = _TraceLogger()
    client = object.__new__(S3Client)
    client.s3 = fake_s3
    client.can_overwrite = True
    monkeypatch.setattr(s3_client, "logger", trace_logger)
    local_path = str(tmp_path / "clip.mp4")

    client.upload_file(local_path, S3Prefix("s3://bucket/clips/clip.mp4"))

    assert fake_s3.uploads == [(local_path, "bucket", "clips/clip.mp4")]
    assert trace_logger.messages == [f"Uploaded {local_path} to s3://bucket/clips/clip.mp4"]


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


DownloadClientFactory = Callable[..., tuple[S3Client, _FakeDownloadS3]]


@pytest.fixture
def make_download_client() -> DownloadClientFactory:
    """Return a closure building a client wired to a fresh fake S3 for one object."""

    def _factory(
        payload: bytes, *, report_content_length: bool = True, raise_on_read: bool = False
    ) -> tuple[S3Client, _FakeDownloadS3]:
        fake = _FakeDownloadS3(payload, report_content_length=report_content_length, raise_on_read=raise_on_read)
        client = object.__new__(S3Client)
        client.s3 = fake
        return client, fake

    return _factory


def test_small_object_is_served_by_a_single_get_object(make_download_client: DownloadClientFactory) -> None:
    """Serve an object below the multipart threshold from one get_object response."""
    payload = b"below-threshold-payload"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert data == payload
    assert fake.get_object_calls == [{"Bucket": "bucket", "Key": "root/a.mp4"}]
    assert fake.download_fileobj_calls == []


def test_large_object_is_served_by_the_managed_transfer(make_download_client: DownloadClientFactory) -> None:
    """Serve an object above the multipart threshold through the managed transfer."""
    payload = b"above-threshold-payload"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) - 1)

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]


def test_object_sized_exactly_at_the_threshold_is_served_by_the_managed_transfer(
    make_download_client: DownloadClientFactory,
) -> None:
    """Treat the threshold as exclusive: an object of exactly that size takes the managed transfer."""
    payload = b"exactly-at-threshold"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload))

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]


def test_unread_body_is_closed_when_the_managed_transfer_is_used(
    make_download_client: DownloadClientFactory,
) -> None:
    """Close the get_object probe's stream instead of leaking it when the transfer takes over."""
    payload = b"above-threshold-payload"
    client, fake = make_download_client(payload)

    client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) - 1)

    assert len(fake.bodies) == 1
    assert fake.bodies[0].closed
    assert fake.bodies[0].read_count == 0


def test_body_is_closed_when_the_small_path_read_fails(
    make_download_client: DownloadClientFactory,
) -> None:
    """Close the stream even when reading it raises, so a failed small read leaks no connection.

    A caller that retries re-enters this method on every attempt, so a stream left
    open on the failing path would leak once per attempt rather than once per run.
    """
    payload = b"below-threshold-payload"
    client, fake = make_download_client(payload, raise_on_read=True)

    with pytest.raises(OSError, match="connection reset mid-stream"):
        client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert fake.bodies[0].closed


def test_response_without_content_length_falls_back_to_the_managed_transfer(
    make_download_client: DownloadClientFactory,
) -> None:
    """Hand an unmeasured object to the managed transfer rather than reading its body blind."""
    payload = b"unmeasured-payload"
    client, fake = make_download_client(payload, report_content_length=False)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]
    assert fake.bodies[0].read_count == 0


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
