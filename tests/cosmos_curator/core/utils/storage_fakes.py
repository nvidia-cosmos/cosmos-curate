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
"""Backend fakes shared by the two storage characterization suites.

Consumed by ``test_storage_cli`` alongside this file and by
``tests/cosmos_curator/next/recipes/data_integrity/test_storage_io``, which pins the
recipe's policy layer over the same helpers. Exactly the fakes both suites drive live
here; anything reachable from only one of them stays in that suite, so neither file
carries a fake it does not use.

The fakes are SDK-shaped, not storage-shaped: the request each Curator call makes on
the wire is what those suites exist to pin, so :func:`s3_client` / :func:`azure_client`
wrap them in a real client rather than the fakes standing in for one. There is no
cloud-mocking library in any pixi environment (no ``moto``, no ``botocore.stub``),
which is why they are hand-rolled. Nothing here performs network I/O.

Not named ``test_*`` so pytest does not collect it, and not a ``conftest`` because the
two suites share no directory that would put it in scope for both without also putting
it in scope for every other test under ``tests/cosmos_curator``.
"""

import datetime
from collections.abc import Iterator
from typing import Any, cast

from cosmos_curator.core.utils.storage.azure_client import AzureClient
from cosmos_curator.core.utils.storage.s3_client import S3Client

#: The instant every fake reports as a last-modified time, so a stat or a listing can be
#: compared against a literal rather than against whatever the clock said.
WHEN = datetime.datetime(2026, 8, 1, 12, 0, tzinfo=datetime.UTC)


def s3_client(sdk_client: object, *, can_overwrite: bool = False) -> S3Client:
    """Wrap a boto3-shaped fake in a real ``S3Client``, resolving no credentials.

    ``__new__`` rather than ``__init__``: the constructor builds a boto3 session and
    client, which would need credentials for tests that are about neither. The result is
    a genuine ``S3Client``, which matters because the smart_open bridge dispatches on the
    type.

    ``can_overwrite`` defaults off, matching the factory it stands in for. A caller that
    hardcoded it on would make the upload path unconditionally skip its existence check,
    so a write that had lost the flag would still look correct here.
    """
    client = S3Client.__new__(S3Client)
    client.s3 = cast("Any", sdk_client)
    client.can_overwrite = can_overwrite
    return client


def azure_client(sdk_client: object) -> AzureClient:
    """Wrap a ``BlobServiceClient``-shaped fake in a real ``AzureClient``.

    Same reasoning as :func:`s3_client`; here the constructor would reach for a
    connection string or a credential object.
    """
    client = AzureClient.__new__(AzureClient)
    client.service_client = cast("Any", sdk_client)
    client.can_overwrite = True
    return client


class FakeBlobProperties:
    """The three fields a stat reads off an Azure blob, each independently absent-able."""

    def __init__(self, size: int | None, last_modified: datetime.datetime | None, etag: str | None) -> None:
        """Hold one canned properties answer."""
        self.size = size
        self.last_modified = last_modified
        self.etag = etag


class FakeBlob:
    """Listing entry carrying the properties a recursive listing reads off a blob.

    A name ending in the delimiter gets size 0, which is what a hierarchical-namespace
    account reports for a directory placeholder, so the unfiltered listing sees the
    same shape it would in production.
    """

    def __init__(self, name: str) -> None:
        """Derive the properties a listing would report for the blob called ``name``."""
        self.name = name
        self.size = 0 if name.endswith("/") else 1024
        self.last_modified = WHEN
        self.etag = '"listed"'


class FakeBlobClient:
    """Blob stand-in serving one canned properties answer and counting its calls."""

    def __init__(self, props: FakeBlobProperties | None = None, error: Exception | None = None) -> None:
        """Hold the properties this blob will serve, or the error it will raise instead."""
        self._props = props
        self._error = error
        self.property_calls = 0

    def get_blob_properties(self) -> FakeBlobProperties:
        """Serve the canned answer, or raise, recording that one request was made."""
        self.property_calls += 1
        if self._error is not None:
            raise self._error
        assert self._props is not None
        return self._props


class FakeAzureContainer:
    """Container stand-in whose listing records how many entries were consumed.

    Azure's ``list_blobs`` is a lazily paged iterator, so the consumed count is what
    shows an early-exiting listing stopped fetching rather than filtering afterwards.
    """

    def __init__(self, blobs: list[FakeBlob] | None = None, blob_client: FakeBlobClient | None = None) -> None:
        """Hold the blobs a listing will yield and the blob client a stat will reach."""
        self._blobs = blobs or []
        self._blob_client = blob_client
        self.blobs_yielded = 0
        self.name_starts_with: str | None = None
        self.requested_blob: str | None = None

    def list_blobs(self, name_starts_with: str = "") -> Iterator[FakeBlob]:
        """Record the prefix the listing asked for and hand back a lazy iterator."""
        self.name_starts_with = name_starts_with
        return self._iter_blobs()

    def _iter_blobs(self) -> Iterator[FakeBlob]:
        for blob in self._blobs:
            self.blobs_yielded += 1
            yield blob

    def get_blob_client(self, blob: str) -> FakeBlobClient:
        """Record which blob was addressed and hand back the canned blob client."""
        self.requested_blob = blob
        assert self._blob_client is not None
        return self._blob_client


class FakeAzureService:
    """``BlobServiceClient`` stand-in handing out one container client."""

    def __init__(self, container: FakeAzureContainer) -> None:
        """Hold the single container client this service resolves every name to."""
        self._container = container
        self.requested_container: str | None = None

    def get_container_client(self, container: str) -> FakeAzureContainer:
        """Record which container was addressed and hand back its client."""
        self.requested_container = container
        return self._container

    def get_blob_client(self, container: str, blob: str) -> FakeBlobClient:
        """Address a single blob container-first, as ``BlobServiceClient`` does.

        ``AzureClient`` reaches a blob through the service client rather than through a
        container client, so this delegation is what keeps the container fake's record
        of which blob was asked for.
        """
        self.requested_container = container
        return self._container.get_blob_client(blob)
