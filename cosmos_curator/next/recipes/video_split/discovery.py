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

from dataclasses import dataclass
from pathlib import PurePosixPath

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client
from cosmos_curator.next.recipes.video_split.config import VideoSplitInputConfig
from cosmos_curator.next.recipes.video_split.uris import normalize_s3_mp4_uri


@dataclass(frozen=True)
class ResolvedInputSelection:
    """One logical source set and its independent execution priority."""

    canonical_uris: tuple[str, ...]
    scheduled_uris: tuple[str, ...]


def resolve_input_selection(
    config: VideoSplitInputConfig,
    *,
    storage_profile: str = "default",
) -> ResolvedInputSelection:
    """Realize one canonical source set plus its source-task submission order."""
    if config.uris is not None:
        # Explicit objects become source errors when a later read/probe proves
        # one missing or unreadable. Avoid one HEAD request per explicit
        # object merely to estimate task cost, so this form stays URI-ordered.
        canonical_uris = tuple(sorted(set(config.uris)))
        return ResolvedInputSelection(canonical_uris=canonical_uris, scheduled_uris=canonical_uris)

    if config.root_uri is None:  # guarded by config validation
        msg = "input must set uris or root_uri"
        raise ValueError(msg)
    return _discover_root(config.root_uri, storage_profile=storage_profile)


def _discover_root(root_uri: str, *, storage_profile: str) -> ResolvedInputSelection:
    root = S3Prefix(root_uri)
    client = get_storage_client(root.path, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for input root: {root_uri}"
        raise TypeError(msg)

    # Treat a non-bucket root as a directory boundary, not a raw lexical
    # prefix, so ``raw-old/`` is never selected for root ``raw``.
    listing_key = f"{root.prefix.rstrip('/')}/" if root.prefix else ""
    listing_root = S3Prefix(f"s3://{root.bucket}/{listing_key}")

    size_by_uri: dict[str, int | None] = {}
    for metadata in client.list_recursive(listing_root):
        if not isinstance(metadata, dict) or not isinstance(metadata.get("Key"), str):
            msg = f"S3 discovery returned unexpected object metadata: {metadata!r}"
            raise TypeError(msg)
        candidate = S3Prefix(f"s3://{root.bucket}/{metadata['Key']}")
        if listing_key and not candidate.prefix.startswith(listing_key):
            msg = f"S3 discovery returned an object outside {listing_root.path}: {candidate.path}"
            raise ValueError(msg)
        if PurePosixPath(candidate.prefix).suffix.lower() != ".mp4":
            continue
        source_uri = normalize_s3_mp4_uri(candidate.path)
        raw_size = metadata.get("Size")
        size_bytes = (
            raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size >= 0 else None
        )
        previous_size = size_by_uri.get(source_uri)
        if source_uri not in size_by_uri or (
            size_bytes is not None and (previous_size is None or size_bytes > previous_size)
        ):
            size_by_uri[source_uri] = size_bytes

    canonical_uris = tuple(sorted(size_by_uri))

    def schedule_key(uri: str) -> tuple[bool, int, str]:
        size_bytes = size_by_uri[uri]
        return size_bytes is None, 0 if size_bytes is None else -size_bytes, uri

    scheduled_uris = tuple(sorted(canonical_uris, key=schedule_key))
    return ResolvedInputSelection(canonical_uris=canonical_uris, scheduled_uris=scheduled_uris)
