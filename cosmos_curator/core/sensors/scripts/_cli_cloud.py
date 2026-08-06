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
"""Shared CLI helpers for sensor-library scripts that accept cloud sources.

Centralises ``s3://`` / ``az://`` URI detection, credential resolution, and
the actual ``smart_open`` open used by ``check_video_index``,
``camera_sensor_benchmark``, and ``cloud_io_benchmark``.

The sensor library itself is backend-agnostic and never accepts URIs (see
``cosmos_curator/core/sensors/types/types.py``). This helper module is the
single carve-out under ``cosmos_curator/core/sensors/`` that imports
``smart_open``, on behalf of the in-tree scripts only.

The helpers are split into two layers so instrumentation-heavy callers
(e.g. ``cloud_io_benchmark``) can build a boto3 client, attach event hooks
to it, and *then* hand it into :func:`open_cloud_source`:

* :func:`make_s3_client` / :func:`make_azure_client` resolve credentials
  and return the underlying SDK client.
* :func:`open_cloud_source` is a context manager that yields a seekable
  :class:`typing.BinaryIO` for an ``s3://`` / ``az://`` URI, using a
  caller-provided client when present and otherwise constructing one from
  the supplied profile-name arguments.

Credentials are loaded via ``boto3`` (S3) and ``azure.storage.blob`` /
``azure.identity`` (Azure) directly rather than through
``cosmos_curator.core.utils.storage`` to keep the sensor-package boundary
intact.
"""

import argparse
import configparser
import dataclasses
import datetime
import os
import pathlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, BinaryIO, cast

import boto3
import smart_open  # type: ignore[import-untyped]
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, NoCredentialsError, ProfileNotFound

S3_CREDENTIALS_HINT = (
    "Use --s3-profile-name to select an AWS profile, or configure standard AWS credentials "
    "with AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an IAM role."
)
AZURE_CREDENTIALS_HINT = (
    "Use --azure-profile-name to select an Azure profile, or populate the Azure credentials file "
    "(default: /dev/shm/azure_creds_file, override with COSMOS_AZURE_PROFILE_PATH) with one of "
    "azure_connection_string, azure_account_name+azure_account_key, or azure_use_managed_identity."
)


class CloudCliError(Exception):
    """Actionable user-facing error from cloud-source CLI helpers."""


def is_s3_uri(source: str) -> bool:
    """Return True if ``source`` is an ``s3://`` URI."""
    return source.startswith("s3://")


def is_azure_uri(source: str) -> bool:
    """Return True if ``source`` is an ``az://`` URI."""
    return source.startswith("az://")


def is_cloud_uri(source: str) -> bool:
    """Return True if ``source`` is a supported cloud URI."""
    return is_s3_uri(source) or is_azure_uri(source)


def validate_source(source: str) -> None:
    """Validate that ``source`` is a local file path or a supported cloud URI.

    Raises:
        CloudCliError: If ``source`` uses an unsupported scheme or refers to a
            local path that does not exist.

    """
    if is_cloud_uri(source):
        return
    if "://" in source:
        msg = f"unsupported source URI {source!r}; use a local file path or an s3:// or az:// URI"
        raise CloudCliError(msg)
    if not pathlib.Path(source).is_file():
        msg = f"source is not a file: {source}"
        raise CloudCliError(msg)


def resolve_s3_endpoint_url(explicit: str | None = None) -> str | None:
    """Resolve the S3 endpoint URL to use, or ``None`` for the default AWS endpoint.

    boto3 does not read the ``endpoint_url`` that the ``awscli_plugin_endpoint``
    plugin nests under the ``s3`` / ``s3api`` sections of ``~/.aws/config`` (that
    is a CLI-only plugin feature), so an S3-compatible store reached only through
    such a profile needs the endpoint supplied here.

    Resolution order (first non-empty wins):

    1. ``explicit`` — typically a ``--endpoint-url`` CLI argument
    2. ``AWS_ENDPOINT_URL_S3`` environment variable (S3-specific)
    3. ``AWS_ENDPOINT_URL`` environment variable (all services)
    4. ``None`` — boto3's default AWS endpoint
    """
    for candidate in (explicit, os.getenv("AWS_ENDPOINT_URL_S3"), os.getenv("AWS_ENDPOINT_URL")):
        if candidate:
            return candidate
    return None


def make_s3_client(source: str, s3_profile_name: str | None, endpoint_url: str | None = None) -> BaseClient:
    """Build a credentialled boto3 S3 client for an ``s3://`` source.

    The returned client is the place to attach botocore event hooks (e.g.
    ``before-send.s3.GetObject``) before handing it into
    :func:`open_cloud_source`.

    Args:
        source: The ``s3://`` URI the client will be used for (for error messages).
        s3_profile_name: Optional AWS profile; ``None`` uses boto3's default chain.
        endpoint_url: Optional S3 endpoint override. Passed through
            :func:`resolve_s3_endpoint_url`, so ``None`` still honours the
            ``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL`` environment variables
            before falling back to boto3's default AWS endpoint.

    Raises:
        CloudCliError: When boto3 cannot construct a credentialled S3 client.

    """
    endpoint_url = resolve_s3_endpoint_url(endpoint_url)
    try:
        session = boto3.Session(profile_name=s3_profile_name) if s3_profile_name else boto3.Session()
        credentials = session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CloudCliError(msg) from e

    if credentials is None:
        msg = f"could not configure S3 access for {source!r}: {NoCredentialsError()}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg)

    try:
        return cast("BaseClient", session.client("s3", endpoint_url=endpoint_url))
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CloudCliError(msg) from e


def _azure_profile_path() -> pathlib.Path:
    """Return the on-disk Azure credentials file path.

    Mirrors the default used by ``cosmos_curator.core.utils.environment`` so
    operators with an existing profile file get the same lookup behaviour
    here as elsewhere in the codebase, without taking a hard import on the
    storage package (which the sensor library is not allowed to depend on).
    """
    return pathlib.Path(os.getenv("COSMOS_AZURE_PROFILE_PATH", "/dev/shm/azure_creds_file"))  # noqa: S108


def _load_azure_profile_section(profile_name: str) -> configparser.SectionProxy:
    """Locate ``profile_name`` in the on-disk Azure credentials file."""
    path = _azure_profile_path()
    if not path.exists():
        msg = f"Azure profile file {path} does not exist"
        raise CloudCliError(msg)

    parser = configparser.ConfigParser()
    parser.read(path)

    section_lookup_len = 2
    for section in parser.sections():
        if section == profile_name:
            return parser[section]
        if section.startswith("profile "):
            parts = section.split()
            if len(parts) == section_lookup_len and parts[1] == profile_name:
                return parser[section]

    msg = f"Azure profile {profile_name!r} not found in {path}"
    raise CloudCliError(msg)


def _build_azure_service_client(profile_name: str) -> BlobServiceClient:
    """Construct an Azure ``BlobServiceClient`` from the profile file.

    Supports the same three credential modes as the storage-package helper:
    connection string, account name + key, and managed identity.
    """
    section = _load_azure_profile_section(profile_name)

    connection_string = section.get("azure_connection_string", None)
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    if section.getboolean("azure_use_managed_identity", False):
        account_url = section.get("azure_account_url", None)
        if not account_url:
            msg = f"Azure profile {profile_name!r}: azure_use_managed_identity set but azure_account_url missing"
            raise CloudCliError(msg)
        return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())

    account_name = section.get("azure_account_name", None)
    account_key = section.get("azure_account_key", None)
    if account_name and account_key:
        account_url = section.get("azure_account_url", None) or f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(
            account_url=account_url,
            credential={"account_name": account_name, "account_key": account_key},
        )

    msg = (
        f"Azure profile {profile_name!r} has no usable credentials "
        "(need one of azure_connection_string, azure_account_name+azure_account_key, "
        "or azure_use_managed_identity+azure_account_url)"
    )
    raise CloudCliError(msg)


def make_azure_client(source: str, azure_profile_name: str) -> BlobServiceClient:
    """Build a credentialled Azure ``BlobServiceClient`` for an ``az://`` source.

    Raises:
        CloudCliError: When the Azure profile is missing or invalid.

    """
    try:
        return _build_azure_service_client(azure_profile_name)
    except CloudCliError:
        raise
    except Exception as e:
        msg = f"could not configure Azure access for {source!r}: {e}\n{AZURE_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e


@contextmanager
def open_cloud_source(  # noqa: PLR0913
    source: str,
    *,
    s3_client: BaseClient | None = None,
    azure_client: BlobServiceClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> Generator[BinaryIO]:
    """Open an ``s3://`` or ``az://`` URI as a seekable :class:`BinaryIO`.

    If a pre-built SDK client is supplied via ``s3_client`` / ``azure_client``
    it is used as-is, which preserves any caller-attached event hooks (e.g.
    botocore's ``before-send.s3.GetObject``). Otherwise the corresponding
    client is constructed from the profile-name arguments.

    Args:
        source: ``s3://`` or ``az://`` URI to open. Local paths are rejected.
        s3_client: Pre-built boto3 S3 client (overrides ``s3_profile_name``).
        azure_client: Pre-built Azure ``BlobServiceClient`` (overrides
            ``azure_profile_name``).
        s3_profile_name: Optional AWS profile used when ``s3_client`` is not
            provided. ``None`` falls back to boto3's default credential chain.
        azure_profile_name: Azure profile used when ``azure_client`` is not
            provided.
        endpoint_url: Optional S3 endpoint override used only when ``s3_client``
            is not supplied (see :func:`resolve_s3_endpoint_url`). Ignored for
            Azure, whose endpoint is baked into the account URL.

    Yields:
        A seekable :class:`BinaryIO` opened in binary read mode via
        ``smart_open``. Ownership stays with this context manager; the
        caller must not close the stream.

    Raises:
        CloudCliError: If ``source`` is not a supported cloud URI.

    """
    transport_params: dict[str, Any]
    if is_s3_uri(source):
        s3 = s3_client if s3_client is not None else make_s3_client(source, s3_profile_name, endpoint_url)
        transport_params = {"client": s3}
    elif is_azure_uri(source):
        azure = azure_client if azure_client is not None else make_azure_client(source, azure_profile_name)
        transport_params = {"client": azure}
    else:
        msg = f"open_cloud_source requires an s3:// or az:// URI, got {source!r}"
        raise CloudCliError(msg)

    with smart_open.open(source, "rb", transport_params=transport_params) as stream:
        yield stream


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/prefix`` into ``(bucket, key_prefix)``."""
    rest = uri[len("s3://") :]
    bucket, _, key_prefix = rest.partition("/")
    if not bucket:
        msg = f"malformed s3 URI (no bucket): {uri!r}"
        raise CloudCliError(msg)
    return bucket, key_prefix


def _split_azure_uri(uri: str) -> tuple[str, str]:
    """Split ``az://container/blob/prefix`` into ``(container, blob_prefix)``."""
    rest = uri[len("az://") :]
    container, _, blob_prefix = rest.partition("/")
    if not container:
        msg = f"malformed az URI (no container): {uri!r}"
        raise CloudCliError(msg)
    return container, blob_prefix


def _matches_suffix(key: str, suffixes: tuple[str, ...] | None) -> bool:
    """Whether ``key`` ends with one of ``suffixes`` (case-insensitive); no filter when ``None``."""
    if suffixes is None:
        return True
    lowered = key.lower()
    return lowered.endswith(suffixes)


def _list_s3_objects(prefix: str, s3_client: BaseClient, limit: int, suffixes: tuple[str, ...] | None) -> list[str]:
    """List object URIs under an ``s3://`` prefix, stopping once ``limit`` matches are found."""
    bucket, key_prefix = _split_s3_uri(prefix)
    uris: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip the zero-byte "directory" placeholder keys some tools create.
            if key.endswith("/") or not _matches_suffix(key, suffixes):
                continue
            uris.append(f"s3://{bucket}/{key}")
            if 0 < limit <= len(uris):
                return uris
    return uris


def _list_azure_objects(
    prefix: str, azure_client: BlobServiceClient, limit: int, suffixes: tuple[str, ...] | None
) -> list[str]:
    """List blob URIs under an ``az://`` prefix, stopping once ``limit`` matches are found."""
    container, blob_prefix = _split_azure_uri(prefix)
    container_client = azure_client.get_container_client(container)
    uris: list[str] = []
    for blob in container_client.list_blobs(name_starts_with=blob_prefix):
        if blob.name.endswith("/") or not _matches_suffix(blob.name, suffixes):
            continue
        uris.append(f"az://{container}/{blob.name}")
        if 0 < limit <= len(uris):
            return uris
    return uris


def list_cloud_objects(  # noqa: PLR0913
    prefix: str,
    *,
    s3_client: BaseClient | None = None,
    azure_client: BlobServiceClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    limit: int = 0,
    suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """List object URIs under an ``s3://`` or ``az://`` prefix.

    Returns fully-qualified URIs (``s3://bucket/key`` / ``az://container/blob``)
    suitable for handing straight back to :func:`open_cloud_source`. Zero-byte
    directory-placeholder keys are skipped. Listing stops as soon as ``limit``
    matching objects are collected so a caller can cheaply sample a huge bucket
    without paging it in full.

    Args:
        prefix: ``s3://`` or ``az://`` prefix to list under.
        s3_client: Pre-built boto3 S3 client (overrides ``s3_profile_name`` /
            ``endpoint_url``).
        azure_client: Pre-built Azure ``BlobServiceClient`` (overrides
            ``azure_profile_name``).
        s3_profile_name: Optional AWS profile used when ``s3_client`` is absent.
        azure_profile_name: Azure profile used when ``azure_client`` is absent.
        endpoint_url: Optional S3 endpoint override used when ``s3_client`` is
            absent (see :func:`resolve_s3_endpoint_url`).
        limit: Maximum number of objects to return; ``0`` (default) means all.
        suffixes: Optional tuple of lowercased suffixes (for example
            ``(".mp4", ".mkv")``); only matching keys are returned and counted
            toward ``limit``. ``None`` (default) returns every object.

    Raises:
        CloudCliError: If ``prefix`` is not a supported cloud URI.

    """
    if is_s3_uri(prefix):
        s3 = s3_client if s3_client is not None else make_s3_client(prefix, s3_profile_name, endpoint_url)
        return _list_s3_objects(prefix, s3, limit, suffixes)
    if is_azure_uri(prefix):
        azure = azure_client if azure_client is not None else make_azure_client(prefix, azure_profile_name)
        return _list_azure_objects(prefix, azure, limit, suffixes)
    msg = f"list_cloud_objects requires an s3:// or az:// URI, got {prefix!r}"
    raise CloudCliError(msg)


@dataclasses.dataclass(frozen=True)
class CloudObjectStat:
    """What one ``HEAD`` tells us about a cloud object.

    Attributes:
        size_bytes: object size, or ``None`` if the backend did not report one.
        etag: the backend's entity tag. Treat it as an **opaque change token, not a
            checksum**: S3 returns an MD5 for a single-part upload but a
            ``<hash>-<part-count>`` composite for a multipart one, so identical bytes
            can carry different tags depending on how they were uploaded. A changed
            tag reliably means something happened; an unchanged tag alongside an
            unchanged size is good evidence nothing did.
        last_modified: server-side modification time, timezone-aware.

    """

    size_bytes: int | None = None
    etag: str | None = None
    last_modified: datetime.datetime | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this stat learned nothing at all.

        The lookup never raises, so an all-``None`` stat is how a failed ``HEAD``
        arrives. Callers need to tell that apart from a real response: passing an
        empty one along as if it were a fact would record a null ETag for an object
        that was merely unlucky the first time.
        """
        return self.size_bytes is None and self.etag is None and self.last_modified is None


def _s3_object_stat(source: str, s3: BaseClient) -> CloudObjectStat:
    bucket, key = _split_s3_uri(source)
    # head_object is a dynamically generated botocore method (not on BaseClient's stub).
    head = cast("Any", s3).head_object(Bucket=bucket, Key=key)
    size = head.get("ContentLength")
    etag = head.get("ETag")
    return CloudObjectStat(
        size_bytes=int(size) if size is not None else None,
        # boto3 hands back the tag quoted, as it appears on the wire. Unquoted here so
        # a stored value compares cleanly against one from another client.
        etag=str(etag).strip('"') if etag is not None else None,
        last_modified=head.get("LastModified"),
    )


def _azure_object_stat(source: str, azure: BlobServiceClient) -> CloudObjectStat:
    container, blob = _split_azure_uri(source)
    props = azure.get_container_client(container).get_blob_client(blob).get_blob_properties()
    return CloudObjectStat(
        size_bytes=int(props.size) if props.size is not None else None,
        etag=str(props.etag).strip('"') if props.etag is not None else None,
        last_modified=props.last_modified,
    )


def get_cloud_object_stat(  # noqa: PLR0913
    source: str,
    *,
    s3_client: BaseClient | None = None,
    azure_client: BlobServiceClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> CloudObjectStat:
    """Return what one ``HEAD`` says about a cloud object; empty when it can't be determined.

    Best-effort: issues a single ``HEAD`` (S3) / ``get_blob_properties`` (Azure) for
    ``source``. Any failure (non-cloud URI, missing object, credential error) yields
    an all-``None`` :class:`CloudObjectStat` rather than raising, because both callers
    -- a progress display and a staleness record -- are enrichment and must never
    break the actual check.

    Size, ETag and last-modified all come out of that same one response, so recording
    content identity alongside the progress total costs no extra round trip.

    Args:
        source: ``s3://`` or ``az://`` URI of the object.
        s3_client: pre-built boto3 S3 client (overrides ``s3_profile_name`` / ``endpoint_url``).
        azure_client: pre-built Azure ``BlobServiceClient`` (overrides ``azure_profile_name``).
        s3_profile_name: optional AWS profile used when ``s3_client`` is absent.
        azure_profile_name: Azure profile used when ``azure_client`` is absent.
        endpoint_url: optional S3 endpoint override used when ``s3_client`` is absent.

    """
    try:
        if is_s3_uri(source):
            s3 = s3_client if s3_client is not None else make_s3_client(source, s3_profile_name, endpoint_url)
            return _s3_object_stat(source, s3)
        if is_azure_uri(source):
            azure = azure_client if azure_client is not None else make_azure_client(source, azure_profile_name)
            return _azure_object_stat(source, azure)
    except Exception:  # noqa: BLE001 - advisory only; never fail the caller over it
        return CloudObjectStat()
    return CloudObjectStat()


def get_cloud_object_size(  # noqa: PLR0913
    source: str,
    *,
    s3_client: BaseClient | None = None,
    azure_client: BlobServiceClient | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> int | None:
    """Return the byte size of a single cloud object, or ``None`` if it can't be determined.

    Thin wrapper over :func:`get_cloud_object_stat` for callers that only want the
    size (the progress display); same one ``HEAD``, same never-raises contract.
    """
    return get_cloud_object_stat(
        source,
        s3_client=s3_client,
        azure_client=azure_client,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
    ).size_bytes


def put_cloud_text(
    uri: str,
    text: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    """Write ``text`` to an ``s3://`` URI as UTF-8.

    Small-document helper for the store's manifest; a whole-object ``PUT``, so it is
    not for anything that should be streamed.

    Raises:
        CloudCliError: if ``uri`` is not an ``s3://`` URI.

    """
    if not is_s3_uri(uri):
        msg = f"expected an s3:// URI, got {uri!r}"
        raise CloudCliError(msg)
    bucket, key = _split_s3_uri(uri)
    client = make_s3_client(uri, s3_profile_name, endpoint_url)
    cast("Any", client).put_object(Bucket=bucket, Key=key, Body=text.encode())


def get_cloud_text(
    uri: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """Read an ``s3://`` object as UTF-8 text.

    Raises:
        CloudCliError: if ``uri`` is not an ``s3://`` URI.

    """
    if not is_s3_uri(uri):
        msg = f"expected an s3:// URI, got {uri!r}"
        raise CloudCliError(msg)
    bucket, key = _split_s3_uri(uri)
    client = make_s3_client(uri, s3_profile_name, endpoint_url)
    body = cast("Any", client).get_object(Bucket=bucket, Key=key)["Body"].read()
    return str(body.decode())


def get_lance_storage_options(
    uri: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, str] | None:
    """Build Lance ``storage_options`` for ``uri``, or ``None`` for a local path.

    A boundary-respecting equivalent of
    ``cosmos_curator.core.utils.storage.storage_utils.get_lance_storage_options``,
    which modules under ``cosmos_curator/core/sensors/`` may not import (see the
    self-containment rule in ``cosmos_curator/core/sensors/__init__.py``). Same
    option names, resolved from the same boto3 session the rest of this module uses,
    so a store written here is readable by the pipeline side and vice versa.

    Azure raises rather than silently writing an unauthenticated store: Lance's Azure
    options are a different set of keys, and a wrong guess would surface as an opaque
    permission error at write time.

    Raises:
        CloudCliError: if ``uri`` is an ``az://`` URI, or S3 credentials cannot be
            resolved.

    """
    if is_azure_uri(uri):
        msg = f"the data-integrity store does not support az:// yet: {uri!r}; use a local path or an s3:// URI"
        raise CloudCliError(msg)
    if not is_s3_uri(uri):
        return None

    endpoint_url = resolve_s3_endpoint_url(endpoint_url)
    try:
        session = boto3.Session(profile_name=s3_profile_name) if s3_profile_name else boto3.Session()
        credentials = session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {uri!r}: {e}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg) from e
    if credentials is None:
        msg = f"could not configure S3 access for {uri!r}: {NoCredentialsError()}\n{S3_CREDENTIALS_HINT}"
        raise CloudCliError(msg)

    # frozen_credentials resolves whatever the chain produced (static keys, SSO, an
    # assumed role) into concrete values, which is what Lance's object store needs --
    # it cannot call back into boto3 to refresh them.
    frozen = credentials.get_frozen_credentials()
    options = {
        "aws_access_key_id": frozen.access_key,
        "aws_secret_access_key": frozen.secret_key,
        "aws_session_token": frozen.token,
        "aws_region": session.region_name,
        "aws_endpoint": endpoint_url,
    }
    return {key: value for key, value in options.items() if value} or None


def add_cloud_credential_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--s3-profile-name`` / ``--azure-profile-name`` / ``--endpoint-url`` flags to ``parser``."""
    parser.add_argument(
        "--s3-profile-name",
        default=None,
        help="Optional AWS profile name used for s3:// sources. If omitted, boto3's default credential chain is used.",
    )
    parser.add_argument(
        "--azure-profile-name",
        default="default",
        help="Azure profile name used for az:// sources (default: 'default').",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help=(
            "Optional S3 endpoint URL for s3:// sources on S3-compatible stores. "
            "Falls back to AWS_ENDPOINT_URL_S3 / AWS_ENDPOINT_URL, "
            "then boto3's default AWS endpoint. Ignored for az:// sources."
        ),
    )
