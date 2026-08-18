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

"""Distributed clip fragment writes and ordered overwrite commits.

Clip data never passes through the driver. Workers write Lance fragments
directly to storage and return only their metadata; the driver commits every
fragment in one atomic ``Overwrite``. Source outcomes are one row per source,
so they are small enough to build and write on the driver.
"""

import json
from dataclasses import dataclass
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import write_fragments

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_split.config import VideoSplitOutputConfig
from cosmos_curator.next.recipes.video_split.contracts import LANCE_DATA_STORAGE_VERSION
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA, SOURCE_SCHEMA, clip_table, source_table


@dataclass(frozen=True)
class PublishedSnapshots:
    """Exact versions committed by one operationally successful run."""

    clips_version: int
    sources_version: int


def write_clip_fragments(work_records: pa.Table, *, uri: str, storage_profile: str) -> list[str]:
    """Write one batch's published clips as one uncommitted fragment on a worker.

    Fragment size is the publish batch size: a fragment cannot span batches, so
    a separate row cap could only ever subdivide one further.
    """
    published = work_records.filter(
        pc.and_(
            pc.equal(work_records["record_type"], "clip_outcome"),
            pc.equal(work_records["status"], "success"),
        )
    )
    if published.num_rows == 0:
        return []
    fragments = write_fragments(
        clip_table(published),
        uri,
        schema=CLIP_SCHEMA,
        mode="overwrite",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
    )
    return [json.dumps(fragment.to_json()) for fragment in fragments]


def publish_snapshots(
    clip_fragments: list[str],
    source_rows: list[dict[str, Any]],
    *,
    output: VideoSplitOutputConfig,
    storage_profile: str,
) -> PublishedSnapshots:
    """Commit clips first, then bind and commit all source outcomes."""
    clips_version = commit_clip_snapshot(
        clip_fragments,
        uri=output.clips_lance_uri,
        storage_profile=storage_profile,
    )
    bound_source_rows = [
        {
            **row,
            "clips_lance_uri": output.clips_lance_uri,
            "clips_lance_version": clips_version,
        }
        for row in source_rows
    ]
    sources_version = write_source_snapshot(
        bound_source_rows,
        uri=output.sources_lance_uri,
        clips_lance_uri=output.clips_lance_uri,
        clips_lance_version=clips_version,
        storage_profile=storage_profile,
    )
    return PublishedSnapshots(clips_version=clips_version, sources_version=sources_version)


def commit_clip_snapshot(clip_fragments: list[str], *, uri: str, storage_profile: str) -> int:
    """Commit every worker-written fragment as the complete clip snapshot.

    Workers assign fragment ids independently, so they collide across batches.
    ``Overwrite`` reassigns them at commit, which is also what makes an empty
    fragment list a valid complete snapshot.
    """
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


def write_source_snapshot(
    rows: list[dict[str, Any]],
    *,
    uri: str,
    clips_lance_uri: str,
    clips_lance_version: int,
    storage_profile: str,
) -> int:
    """Overwrite source outcomes bound to the exact committed clip version."""
    dataset = lance.write_dataset(
        source_table(rows),
        uri,
        schema=SOURCE_SCHEMA,
        mode="overwrite",
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        enable_v2_manifest_paths=True,
        transaction_properties={
            "kind": "video-split",
            "snapshot": "sources",
            "clips_lance_uri": clips_lance_uri,
            "clips_lance_version": str(clips_lance_version),
        },
    )
    return int(dataset.version)
