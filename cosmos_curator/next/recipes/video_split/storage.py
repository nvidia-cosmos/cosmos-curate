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

"""S3 URI and worker-side storage helpers for ``video-split``."""

from functools import lru_cache

from botocore.exceptions import BotoCoreError, ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, get_s3_client_config, is_s3path

# Botocore already retries transient HTTP responses. The outer retry is for
# transport failures that escape that layer; retrying every ``ClientError``
# would also repeat deterministic responses such as AccessDenied and NoSuchKey.
RETRYABLE_STORAGE_ERRORS = (BotoCoreError, OSError)

# All failures raised by the storage boundary. Callers decide whether an
# exhausted transport error or deterministic service response is an item
# outcome (source reads) or an operational run failure (media writes).
STORAGE_ERRORS = (*RETRYABLE_STORAGE_ERRORS, ClientError)


def download_file(source_uri: str, destination_path: str, *, storage_profile: str) -> None:
    """Stream one S3 object to an atomic worker-local file."""
    client = _s3_client(source_uri, storage_profile=storage_profile)
    client.download_to_path(S3Prefix(source_uri), destination_path)


def upload_file(source_path: str, destination_uri: str, *, storage_profile: str) -> None:
    """Atomically replace one deterministic S3 object from a worker-local file."""
    client = _s3_client(destination_uri, storage_profile=storage_profile, can_overwrite=True)
    client.upload_file(source_path, S3Prefix(destination_uri))


def _s3_client(location: str, *, storage_profile: str, can_overwrite: bool = False) -> S3Client:
    if not is_s3path(location):
        msg = f"Could not create an S3 client for {location}"
        raise TypeError(msg)
    return _cached_s3_client(storage_profile, can_overwrite=can_overwrite)


@lru_cache(maxsize=8)
def _cached_s3_client(storage_profile: str, *, can_overwrite: bool) -> S3Client:
    """Build one client per worker process instead of one per clip.

    A client is fully determined by profile and permission, so the target path
    is not part of the key. Building one costs a profile-file read plus a boto3
    session and client, which is far too much to pay twice per clip. The
    tradeoff is that a cached client holds the credentials it was built with,
    so profile rotation is not picked up mid-run.
    """
    return S3Client(get_s3_client_config(storage_profile, can_overwrite=can_overwrite))
