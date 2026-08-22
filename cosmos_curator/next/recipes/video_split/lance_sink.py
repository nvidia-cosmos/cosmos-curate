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

"""Distributed fragment writes and one atomic clip-snapshot commit."""

import json

import lance
import pyarrow as pa
from lance.fragment import write_fragments

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_split.contracts import LANCE_DATA_STORAGE_VERSION
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA


def write_clip_fragments(clips: pa.Table, *, uri: str, storage_profile: str) -> list[str]:
    """Write one canonical clip batch as an uncommitted Lance fragment."""
    if not clips.schema.equals(CLIP_SCHEMA):
        msg = "Clip fragment input does not match the canonical clip schema"
        raise ValueError(msg)
    if clips.num_rows == 0:
        return []
    fragments = write_fragments(
        clips,
        uri,
        schema=CLIP_SCHEMA,
        mode="overwrite",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
    )
    return [json.dumps(fragment.to_json()) for fragment in fragments]


def commit_clip_snapshot(clip_fragments: list[str], *, uri: str, storage_profile: str) -> int:
    """Commit every worker-written fragment as the complete clip snapshot."""
    fragments = [lance.FragmentMetadata.from_json(fragment) for fragment in clip_fragments]
    transaction = lance.Transaction(
        read_version=0,
        operation=lance.LanceOperation.Overwrite(CLIP_SCHEMA, fragments),
        transaction_properties={"kind": "video-split", "snapshot": "clips"},
    )
    dataset = lance.LanceDataset.commit(
        uri,
        transaction,
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
        enable_v2_manifest_paths=True,
    )
    return int(dataset.version)
