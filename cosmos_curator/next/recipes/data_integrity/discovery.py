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

"""Discover the video streams that make up a single data-integrity session.

A session is one recording -- for example a ``clips/<uuid>/`` prefix whose
per-camera videos live under ``recorder-XX/`` subprefixes. ``discover_streams``
finds those video files under a session path (a local directory, a cloud prefix,
or a single file) so the runner can open each as a ``CameraSensor``. Non-video
session files (rig calibration JSON, upload markers) are ignored by the
extension filter.

Multi-session grouping (running many ``<uuid>`` sessions under a parent prefix)
is intentionally not handled here: it is a thin loop over the single-session
runner, not part of discovery.
"""

import pathlib

from cosmos_curator.core.sensors.data_integrity.engine import validate_non_negative_int
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    is_cloud_uri,
    list_cloud_objects,
)

# Container suffixes CameraSensor can open (MP4 family + MKV). Lowercased for
# case-insensitive matching. MPEG-TS is intentionally excluded (rejected by the
# sensor library).
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".mkv")


def _is_video(name: str) -> bool:
    return name.lower().endswith(VIDEO_SUFFIXES)


def discover_streams(
    session_path: str,
    *,
    limit: int = 0,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> list[str]:
    """Discover the video streams under a single session path.

    Args:
        session_path: A local directory, a local file, or an ``s3://`` / ``az://``
            prefix (typically one ``clips/<uuid>/``). A lone video file is also
            accepted and returned as a single-element list.
        limit: Maximum number of streams to return; ``0`` (default) means all.
            For cloud prefixes the cap is pushed into the listing so a large
            bucket is not paged in full.
        s3_profile_name: Optional AWS profile for ``s3://`` sources.
        azure_profile_name: Azure profile for ``az://`` sources.
        endpoint_url: Optional S3 endpoint override for S3-compatible stores
            (see ``_cli_cloud.resolve_s3_endpoint_url``).

    Returns:
        Fully-qualified stream paths/URIs in sorted (discovery) order. Cloud URIs
        can be handed straight to ``open_cloud_source``; local paths are absolute
        filesystem paths.

    Raises:
        FileNotFoundError: If a local ``session_path`` does not exist.
        ValueError: If ``limit`` is negative.

    """
    limit = validate_non_negative_int("limit", limit)
    if is_cloud_uri(session_path):
        streams = list_cloud_objects(
            session_path,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            limit=limit,
            suffixes=VIDEO_SUFFIXES,
        )
        return sorted(streams)

    path = pathlib.Path(session_path)
    if path.is_file():
        # Apply the same video-suffix filter as directory discovery so an explicit
        # non-video file (e.g. rig_meta.json) does not become a bogus stream.
        return [str(path)] if _is_video(path.name) else []
    if path.is_dir():
        streams = sorted(str(p) for p in path.rglob("*") if p.is_file() and _is_video(p.name))
        return streams[:limit] if limit > 0 else streams

    msg = f"session path does not exist: {session_path}"
    raise FileNotFoundError(msg)
