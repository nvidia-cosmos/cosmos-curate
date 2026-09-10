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
"""Derive an absolute (epoch) capture start time for a source video from its path.

MCAP ``log_time`` is meant to be a real clock reading, but nothing upstream in the
split pipeline carries one: ``VideoMetadata`` has no creation-time field,
``extract_video_metadata`` discards the container tags ffprobe returns, and the
mpegts remux (``remux_stages``) copies streams without ``-map_metadata``, so a
container ``creation_time`` would not survive it anyway. Capture recordings name
their start instead -- either in the enclosing directory
(``.../375edge/01/2026-08-18-09-00/3.mp4``) or in the file stem
(``2026-08-10-09-00-bullet-far.mp4``, ``2018-03-05.13-10-00.13-15-00.bus.mp4``) --
which is the same basis the reference ``375mcap`` recordings use ("capture folder
minute; the source video carries no creation_time").

Those names are local wall-clock readings with no zone, so the caller supplies the
zone (``--mcap-timezone``). Paths that name no time at all (``3PANEL.mp4``) yield
``None``, and the writer falls back to a 0-based timeline.
"""

import datetime
import posixpath
import re

from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND

# YYYY-MM-DD<sep>HH-MM[-SS], where <sep> is "-" (2026-08-18-09-00) or "." as used by
# the dotted convention (2018-03-05.13-10-00.13-15-00...). The digit guards keep a
# longer run of digits from matching a shorter field.
_PATH_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[-.](\d{2})-(\d{2})(?:-(\d{2}))?(?!\d)")

CAPTURE_START_SOURCE_NONE = "none"


def _match_to_epoch_ns(match: re.Match[str], tz: datetime.tzinfo) -> int | None:
    year, month, day, hour, minute, second = match.groups()
    try:
        # fold=0 resolves a DST-ambiguous local reading to the earlier of the two instants.
        moment = datetime.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second) if second is not None else 0,
            tzinfo=tz,
            fold=0,
        )
    except ValueError:
        # A digit run that looks like a date but is not one (e.g. month 13).
        return None
    return round(moment.timestamp()) * NS_PER_SECOND


def parse_capture_start_ns(input_path: str, tz: datetime.tzinfo) -> tuple[int, str] | None:
    """Parse the capture start of *input_path* as epoch nanoseconds.

    Args:
        input_path: Source video path or URI (``s3://bucket/.../3.mp4``, ``/data/x.mp4``).
        tz: Zone the path's wall-clock reading is expressed in.

    Returns:
        ``(epoch_ns, matched_text)``, or ``None`` when the path names no usable time.
        The enclosing directory is preferred over the file stem, because a directory
        naming a time is a capture folder while a stem may also carry an unrelated
        run label.

    """
    normalized = input_path.rstrip("/")
    file_name = posixpath.basename(normalized)
    parent_name = posixpath.basename(posixpath.dirname(normalized))
    for candidate in (parent_name, file_name):
        if not candidate:
            continue
        for match in _PATH_TIMESTAMP_RE.finditer(candidate):
            epoch_ns = _match_to_epoch_ns(match, tz)
            if epoch_ns is not None:
                return epoch_ns, match.group(0)
    return None
