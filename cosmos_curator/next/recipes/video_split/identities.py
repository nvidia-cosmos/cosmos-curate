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

"""Canonical SHA-256 identities for ``video-split`` sources and clips."""

import hashlib
import json
from typing import Any

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split.config import TranscodeConfig
from cosmos_curator.next.recipes.video_split.contracts import MEDIA_CONTRACT_VERSION


def canonical_digest(value: Any) -> str:  # noqa: ANN401
    """Hash sorted compact UTF-8 JSON with non-finite values forbidden."""
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_source_id(source_uri: str) -> str:
    """Identify immutable source content by its normalized URI."""
    return canonical_digest(source_uri)


def make_clip_id(source_id: str, span: Span, transcode: TranscodeConfig) -> str:
    """Identify one logical clip independently of scheduling and run order."""
    return canonical_digest(
        {
            "source_id": source_id,
            "start_ns": span.start_ns,
            "end_ns": span.end_ns,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "transcode": transcode.model_dump(mode="json"),
        }
    )
