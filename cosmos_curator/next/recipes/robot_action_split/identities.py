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

"""Stable SHA-256 identifiers for ``robot-action-split`` entities."""

from cosmos_curator.next.recipes.robot_action_split.contracts import ACTION_CONTRACT_VERSION, MEDIA_CONTRACT_VERSION
from cosmos_curator.next.utils.identity import canonical_digest


def make_source_id(dataset_root_uri: str) -> str:
    """Identify a dataset root under the immutable-normalized-URI contract."""
    return canonical_digest(dataset_root_uri.rstrip("/"))


def make_span_group_id(source_id: str, episode_id: str, subtask_index: int, frame_start: int) -> str:
    """Identify one logical subtask span; shared across all camera views.

    Deliberately omits ``frame_end``: two discovery runs over the same
    ``(source_id, episode_id, subtask_index, frame_start)`` are expected to
    agree on where the span ends too, since span boundaries are derived
    deterministically from the same immutable source parquet (see
    ``make_source_id``'s immutable-normalized-URI contract). If that
    assumption is ever violated — e.g. an upstream parquet revision that
    changes a subtask's end frame while its start frame is unchanged — this
    identity would not detect it, since recovery only diffs by ``clip_id``,
    not by re-comparing span geometry the way ``video_split.recovery`` does
    for its stored source fields. Tracked as a known limitation rather than
    fixed here, since it is a pre-existing property of this identity, not
    something this change introduced.
    """
    return canonical_digest(
        {
            "episode_id": episode_id,
            "frame_start": frame_start,
            "source_id": source_id,
            "subtask_index": subtask_index,
        }
    )


def make_action_id(span_group_id: str, action_format: str, source_dataset: str) -> str:
    """Identify one span's action data under the v1 action contract.

    Encodes the span identity, serialization format, dataset spec, and action
    contract version so that a change in any of these produces a new action ID.
    """
    return canonical_digest(
        {
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "action_format": action_format,
            "source_dataset": source_dataset,
            "span_group_id": span_group_id,
        }
    )


def make_clip_id(span_group_id: str, view_name: str, video_bitrate: str, *, source_is_multiview: bool) -> str:
    """Identify one view's clip under the v1 media contract.

    Always includes ``media_contract_version``, ``span_group_id``, and
    ``video_bitrate`` in the digest so that a change in encoding parameters
    produces a new clip identity.  Multi-view sources additionally fold in
    ``view_name`` so each camera gets a distinct, stable clip_id.
    """
    payload: dict[str, object] = {
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "span_group_id": span_group_id,
        "video_bitrate": video_bitrate,
    }
    if source_is_multiview:
        payload["view_name"] = view_name
    return canonical_digest(payload)
