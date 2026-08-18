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

"""Deterministic pre-Ray S3 source selection for ``video-split``."""

from pathlib import PurePosixPath

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client
from cosmos_curator.next.recipes.video_split.config import VideoSplitInputConfig
from cosmos_curator.next.recipes.video_split.uris import normalize_s3_mp4_uri


def resolve_input_selection(config: VideoSplitInputConfig, *, storage_profile: str = "default") -> tuple[str, ...]:
    """Realize, normalize, deduplicate, and sort one exact source selection."""
    if config.uris is not None:
        # Explicit objects become source outcomes even when a later HEAD/probe
        # proves one missing or unreadable.
        return tuple(sorted(set(config.uris)))

    if config.root_uri is None:  # guarded by config validation
        msg = "input must set uris or root_uri"
        raise ValueError(msg)
    return _discover_root(config.root_uri, storage_profile=storage_profile)


def _discover_root(root_uri: str, *, storage_profile: str) -> tuple[str, ...]:
    root = S3Prefix(root_uri)
    client = get_storage_client(root.path, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for input root: {root_uri}"
        raise TypeError(msg)

    # Treat a non-bucket root as a directory boundary, not a raw lexical
    # prefix, so ``raw-old/`` is never selected for root ``raw``.
    listing_key = f"{root.prefix.rstrip('/')}/" if root.prefix else ""
    listing_root = S3Prefix(f"s3://{root.bucket}/{listing_key}")

    realized: set[str] = set()
    for candidate in client.list_recursive_directory(listing_root):
        if not isinstance(candidate, S3Prefix):
            msg = f"S3 discovery returned an unexpected object reference: {candidate!r}"
            raise TypeError(msg)
        if candidate.bucket != root.bucket or (listing_key and not candidate.prefix.startswith(listing_key)):
            msg = f"S3 discovery returned an object outside {listing_root.path}: {candidate.path}"
            raise ValueError(msg)
        if PurePosixPath(candidate.prefix).suffix.lower() != ".mp4":
            continue
        realized.add(normalize_s3_mp4_uri(candidate.path))
    return tuple(sorted(realized))
