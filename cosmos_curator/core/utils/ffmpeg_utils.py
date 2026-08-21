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
"""Runtime checks against the FFmpeg binary on PATH."""

import subprocess


def assert_ffmpeg_supports_h264() -> None:
    """Fail fast if the runtime FFmpeg lacks an H.264 decoder.

    The video pipelines still have internal H.264 dependencies (transcoding,
    intermediate clip storage), so a no-H.264 FFmpeg silently skips every
    input. Until those dependencies are removed, surface the mismatch as a
    clear startup error instead.
    """
    command = ["ffmpeg", "-hide_banner", "-decoders"]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        msg = f"Failed to query FFmpeg decoders: {e}"
        raise RuntimeError(msg) from e
    for line in result.stdout.splitlines():
        tokens = line.split(maxsplit=2)
        if len(tokens) > 1 and tokens[1] == "h264":
            return
    msg = (
        "FFmpeg in this environment does not expose an H.264 decoder. The video "
        "pipeline currently requires H.264 support; the runtime image was likely "
        "built with --redistributable. Rebuild the non-redistributable image, "
        "or mount an FFmpeg build that supports H.264."
    )
    raise RuntimeError(msg)
