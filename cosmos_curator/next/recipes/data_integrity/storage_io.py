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

"""The three storage operations this recipe needs on its own terms.

Each one inverts something the storage layer is deliberately strict about, and each
inversion is a recipe policy rather than a storage capability -- which is why they
live here and not in :mod:`cosmos_curator.core.utils.storage_cli`:

* :func:`object_stat` never raises. ``StorageClient.stat`` follows ``os.stat`` and
  reports absence as an error, which is right for a caller that needs the object;
  its two callers here are a progress display and a provenance record, and neither
  may break the check it decorates.
* :func:`read_text` and :func:`write_text` are ``s3://``-only. They back the store's
  ``manifest.json``, and the store has no Azure implementation (see
  ``storage_cli.get_lance_storage_options``), so an ``az://`` URI is a caller mistake
  to report rather than a backend to reach.

``storage_cli`` is imported as a module, not by name, so a test that replaces its
client factory is seen by these functions too.
"""

from cosmos_curator.core.utils import storage_cli
from cosmos_curator.core.utils.storage.s3_client import S3Client, is_s3path
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StorageStat
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage.storage_utils import read_text as read_storage_text


def object_stat(
    source: str,
    *,
    client: StorageClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> StorageStat | None:
    """Return what one metadata lookup learned about ``source``, or ``None`` if nothing.

    Best-effort: a single ``HEAD`` (S3) / ``get_blob_properties`` (Azure). Any failure
    -- a local path, a missing object, a credential error, a malformed URI -- yields
    ``None`` rather than raising.

    ``None`` is what the callers branch on, and it deliberately also covers the store
    that answered while reporting no metadata at all. Both callers pass what they get
    on as fact: the progress display as a byte total, the store as recorded content
    identity. A response with nothing in it is worth no more than a failed request, and
    conflating the two here is what keeps a null ETag from being persisted for an object
    that was merely unlucky -- the store simply looks again for itself.

    Size, ETag and last-modified all come out of that one response, so recording
    content identity alongside a progress total costs no extra round trip.

    Args:
        source: ``s3://`` or ``az://`` URI of the object.
        client: Pre-built storage client (overrides the profile arguments).
        s3_profile_name: Optional AWS profile used when ``client`` is absent.
        azure_profile_name: Azure profile used when ``client`` is absent.
        endpoint_url: Optional S3 endpoint override used when ``client`` is absent.

    """
    if not is_remote_path(source):
        return None
    try:
        if client is None:
            client = storage_cli.make_storage_client(
                source,
                s3_profile_name=s3_profile_name,
                azure_profile_name=azure_profile_name,
                endpoint_url=endpoint_url,
            )
        stat = client.stat(storage_cli.storage_prefix(source))
    except Exception:  # noqa: BLE001 - advisory only; never fail the caller over it
        return None
    return stat if (stat.size_bytes, stat.last_modified, stat.etag) != (None, None, None) else None


def _s3_client_for(
    uri: str,
    s3_profile_name: str | None,
    endpoint_url: str | None,
    *,
    can_overwrite: bool = False,
) -> S3Client:
    """Build the client the whole-document helpers use, refusing a non-S3 URI first.

    Raises:
        StorageCliError: If ``uri`` is not an ``s3://`` URI.

    """
    if not is_s3path(uri):
        msg = f"expected an s3:// URI, got {uri!r}"
        raise storage_cli.StorageCliError(msg)
    return storage_cli.make_s3_client(uri, s3_profile_name, endpoint_url, can_overwrite=can_overwrite)


def read_text(uri: str, *, s3_profile_name: str | None = None, endpoint_url: str | None = None) -> str:
    """Read an ``s3://`` object as UTF-8 text.

    Single-attempt on purpose: the store reads its manifest to find out whether there
    is one, so an absent object is a normal answer and must come back promptly rather
    than after the read path's minutes of backoff.

    Raises:
        StorageCliError: If ``uri`` is not an ``s3://`` URI, or names a bucket the
            backend's own rules reject.

    """
    client = _s3_client_for(uri, s3_profile_name, endpoint_url)
    return read_storage_text(storage_cli.storage_prefix(uri), client, max_attempts=1)


def write_text(uri: str, text: str, *, s3_profile_name: str | None = None, endpoint_url: str | None = None) -> None:
    """Write ``text`` to an ``s3://`` URI as UTF-8, overwriting unconditionally.

    Small-document helper for the store's manifest, which is rewritten on every run;
    a whole-object upload, so not for anything that should be streamed.

    Raises:
        StorageCliError: If ``uri`` is not an ``s3://`` URI, or names a bucket the
            backend's own rules reject.

    """
    client = _s3_client_for(uri, s3_profile_name, endpoint_url, can_overwrite=True)
    client.upload_bytes(storage_cli.storage_prefix(uri), text.encode())
