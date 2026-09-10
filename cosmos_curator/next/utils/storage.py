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

"""Shared storage utilities for Curator Next recipes."""

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from botocore.exceptions import ClientError
from loguru import logger

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, is_s3path
from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    is_missing_object_error,
    is_remote_path,
    path_to_prefix,
)

_PRECONDITION_FAILED = 412


def artifact_uri(location: str) -> str:
    """Return the normalized durable URI recorded in manifests, rows, and receipts."""
    if is_s3path(location):
        return S3Prefix(location).path
    if location.startswith("file://"):
        return _file_uri_to_path(location).resolve().as_uri()
    return Path(location).expanduser().resolve().as_uri()


def write_media(location: str, data: bytes, *, storage_profile: str = "default") -> None:
    """Atomically replace a deterministic local/S3 media object with complete bytes."""
    if is_s3path(location):
        client = get_storage_client(location, profile_name=storage_profile, can_overwrite=True)
        if client is None:
            msg = f"Could not create an S3 client for {location}"
            raise RuntimeError(msg)
        client.upload_bytes(S3Prefix(location), data)
        return

    # ``artifact_uri`` in this same module hands back file:// URIs, and recipes
    # accept a file:// output prefix, so a location can arrive as one. ``Path``
    # would read the scheme as a directory name and write the object into a
    # relative ``file:/`` tree beside whatever the worker's cwd was -- quietly,
    # so the run reports success and the bytes are somewhere else.
    destination = local_path(location)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_media_if_present(location: str, *, storage_profile: str = "default") -> bytes | None:
    """Read a media object's bytes, or ``None`` when it does not exist.

    The read-direction counterpart of :func:`write_media` and the same
    absence-is-a-no-op contract as :func:`remove_object`: a missing target returns
    ``None`` so a best-effort artifact (for example a Curate run report) collapses
    to "unavailable" instead of crashing, while a permission, network, or any
    other non-absence failure propagates. Local paths and remote object stores
    (``s3://`` / ``az://``) are handled through the same storage client as the
    other helpers in this module.

    Args:
        location: Local path or remote (``s3://`` / ``az://``) object URI.
        storage_profile: Credential profile for the remote read.

    Returns:
        The object's bytes when present, otherwise ``None``.

    Raises:
        ValueError: If ``location`` is remote but no storage client can be built.

    """
    if is_remote_path(location):
        client = get_storage_client(location, profile_name=storage_profile)
        if client is None:
            msg = f"no storage client available to read remote object {location}"
            raise ValueError(msg)
        prefix = path_to_prefix(location)
        # object_exists maps a remote 404 to False and lets every other error
        # (permission, network) propagate, so absence is the only path to None -
        # a raw download would raise a backend-specific not-found exception that
        # the recipe layer must not know about.
        if not client.object_exists(prefix):
            return None
        try:
            return client.download_object_as_bytes(prefix)
        except Exception as e:
            if is_missing_object_error(e):
                return None
            raise

    try:
        return Path(location).read_bytes()
    except FileNotFoundError:
        # Same contract as the removal helpers: an absent object is not an error,
        # every other failure (PermissionError, IsADirectoryError) propagates.
        return None


def remove_prefix(uri: str, *, storage_profile: str = "default") -> None:
    """Delete everything under a prefix, local or remote; absence is a no-op.

    Removes a whole staging directory / object-store prefix: a recursive object
    delete under ``uri`` for remote stores, a directory ``rmtree`` for a local
    path.

    Args:
        uri: Local path or remote (``s3://`` / ``az://``) prefix to remove.
        storage_profile: Credential profile for remote deletion.

    Raises:
        ValueError: If ``uri`` is remote but no delete-capable storage client
            can be built for it.

    """
    if is_remote_path(uri):
        client = get_storage_client(uri, profile_name=storage_profile, can_delete=True)
        if client is None:
            msg = f"no storage client available to remove remote prefix {uri}"
            raise ValueError(msg)
        deleted = 0
        for obj in client.list_recursive_directory(path_to_prefix(uri)):
            try:
                client.delete_object(obj)
            except Exception as e:
                if is_missing_object_error(e):
                    continue
                raise
            deleted += 1
        logger.info(f"Removed remote prefix {uri} ({deleted} object(s)).")
        return

    try:
        shutil.rmtree(uri)
    except FileNotFoundError:
        # Absence is not an error: a re-run that already cleaned up is a no-op.
        # Every OTHER filesystem failure - a PermissionError, or a partially
        # deleted directory - must propagate rather than be logged as a
        # successful removal, which is why ignore_errors is NOT used here.
        return
    logger.info(f"Removed local prefix {uri}.")


def remove_object(uri: str, *, storage_profile: str = "default") -> None:
    """Delete a single object/file, local or remote; absence is a no-op.

    Removes one durable artifact (for example a stale report or a copied-out
    centroid file), never a tree - use :func:`remove_prefix` for a whole staging
    prefix. A missing target is not an error; a local path that is a directory,
    or any other non-absence failure, propagates.

    Args:
        uri: Local path or remote (``s3://`` / ``az://``) object URI.
        storage_profile: Credential profile for remote deletion.

    Raises:
        ValueError: If ``uri`` is remote but no delete-capable storage client
            can be built for it.

    """
    if is_remote_path(uri):
        client = get_storage_client(uri, profile_name=storage_profile, can_delete=True)
        if client is None:
            msg = f"no storage client available to remove remote object {uri}"
            raise ValueError(msg)
        prefix = path_to_prefix(uri)
        # object_exists maps a remote 404 to False and lets every other error
        # propagate, so absence is the only path to a no-op - Azure DELETE on a
        # missing blob raises ResourceNotFoundError whereas S3 DELETE is idempotent.
        if not client.object_exists(prefix):
            return
        try:
            client.delete_object(prefix)
        except Exception as e:
            if is_missing_object_error(e):
                return
            raise
        logger.info(f"Removed remote object {uri}.")
        return

    try:
        Path(uri).unlink()
    except FileNotFoundError:
        # Same contract as remove_prefix: an absent object is a no-op, but a
        # PermissionError or IsADirectoryError must surface, not be swallowed.
        return
    logger.info(f"Removed local object {uri}.")


def _write_local_if_absent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    while True:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        break

    try:
        with os.fdopen(descriptor, "wb") as artifact:
            artifact.write(data)
            artifact.flush()
            os.fsync(artifact.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_s3_if_absent(location: str, data: bytes, *, storage_profile: str) -> None:
    prefix = S3Prefix(location)
    client = get_storage_client(location, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for {location}"
        raise TypeError(msg)
    try:
        client.s3.put_object(Bucket=prefix.bucket, Key=prefix.prefix, Body=data, IfNoneMatch="*")
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
        if status == _PRECONDITION_FAILED or code in {"PreconditionFailed", "ConditionalRequestConflict"}:
            return
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def local_path(location: str) -> Path:
    """Return the filesystem path a non-S3 location names, whether URI or path."""
    if location.startswith("file://"):
        return _file_uri_to_path(location)
    return Path(location)


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or not parsed.path:
        msg = f"Unsupported local file URI: {uri}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))
