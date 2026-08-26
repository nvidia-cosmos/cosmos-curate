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

"""Schema bootstrap, distributed fragment staging, and incremental appends."""

import json
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import write_fragments
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_split.contracts import LANCE_DATA_STORAGE_VERSION
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA
from cosmos_curator.next.utils.lance_utils import open_dataset

_TRANSACTION_PROPERTIES = {"kind": "video-split", "operation": "append-clips"}


def open_or_create_clip_table(*, uri: str, storage_profile: str) -> lance.LanceDataset:
    """Open the canonical table or atomically bootstrap its zero-row schema."""
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        empty = pa.Table.from_batches([], schema=CLIP_SCHEMA)
        try:
            dataset = lance.write_dataset(
                empty,
                uri,
                schema=CLIP_SCHEMA,
                mode="create",
                data_storage_version=LANCE_DATA_STORAGE_VERSION,
                storage_options=storage_options,
                enable_v2_manifest_paths=True,
                transaction_properties={"kind": "video-split", "operation": "bootstrap"},
            )
        except OSError:  # A storage failure may be an ambiguous successful create.
            dataset = open_dataset(uri, storage_options=storage_options)
            if dataset is None:
                raise

    validate_clip_table(dataset, uri=uri)
    return dataset


def validate_clip_table(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Require the splitting-owned schema while allowing nullable curation fields."""
    data_storage_version = getattr(dataset, "data_storage_version", None)
    if data_storage_version != LANCE_DATA_STORAGE_VERSION:
        msg = (
            f"Existing Lance table {uri} has data_storage_version={data_storage_version!r}; "
            f"expected {LANCE_DATA_STORAGE_VERSION!r}"
        )
        raise ValueError(msg)

    expected_by_name = {field.name: field for field in CLIP_SCHEMA}
    actual_by_name = {field.name: field for field in dataset.schema}
    missing = [name for name in expected_by_name if name not in actual_by_name]
    if missing:
        msg = f"Existing Lance table {uri} is missing splitting-owned field(s): {', '.join(missing)}"
        raise ValueError(msg)

    mismatched = [
        name
        for name, expected in expected_by_name.items()
        if not actual_by_name[name].equals(expected, check_metadata=True)
    ]
    if mismatched:
        msg = f"Existing Lance table {uri} has incompatible splitting-owned field(s): {', '.join(mismatched)}"
        raise ValueError(msg)

    non_nullable_extensions = [
        field.name for field in dataset.schema if field.name not in expected_by_name and not field.nullable
    ]
    if non_nullable_extensions:
        msg = f"Existing Lance table {uri} has non-nullable curation field(s): {', '.join(non_nullable_extensions)}"
        raise ValueError(msg)


def write_clip_fragment(clips: pa.Table, *, uri: str, storage_profile: str) -> str | None:
    """Stage one canonical clip batch and serialize its recovery candidate."""
    if not clips.schema.equals(CLIP_SCHEMA):
        msg = "Clip fragment input does not match the canonical clip schema"
        raise ValueError(msg)
    if clips.num_rows == 0:
        return None

    clip_ids = [str(value) for value in clips["clip_id"].to_pylist()]
    if len(set(clip_ids)) != len(clip_ids):
        msg = "Clip fragment contains duplicate clip_id values"
        raise ValueError(msg)

    # ``overwrite`` applies only to uncommitted worker-local fragment ID allocation.
    # The coordinator publishes this descriptor exclusively with ``Append``.
    fragments = write_fragments(
        clips,
        uri,
        schema=CLIP_SCHEMA,
        mode="overwrite",
        max_rows_per_file=clips.num_rows,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
    )
    if len(fragments) != 1:
        msg = f"One publication batch must stage exactly one Lance fragment, got {len(fragments)}"
        raise RuntimeError(msg)
    fragment = fragments[0]
    if fragment.num_rows != len(clip_ids):
        msg = f"Staged fragment contains {fragment.num_rows} rows for {len(clip_ids)} candidate clip IDs"
        raise RuntimeError(msg)
    data_files = tuple(str(data_file.path) for data_file in fragment.files)
    logger.info(
        "Staged Lance fragment with {} clip row(s) for {}: data_files={}",
        len(clip_ids),
        uri,
        data_files,
    )

    return json.dumps(
        {"fragment": fragment.to_json(), "clip_ids": clip_ids},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def append_clip_fragment(
    candidate_json: str,
    *,
    uri: str,
    storage_profile: str,
    attempts: int,
) -> int:
    """Idempotently append one staged fragment and return a version containing it."""
    if attempts < 1:
        msg = f"Append attempts must be positive, got {attempts}"
        raise ValueError(msg)
    fragment, clip_ids = _parse_candidate(candidate_json)
    data_files = tuple(str(data_file.path) for data_file in fragment.files)
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)

    for attempt in range(1, attempts + 1):
        dataset, present = _candidate_presence(clip_ids, uri=uri, storage_options=storage_options)
        if len(present) == len(clip_ids):
            logger.info(
                "Staged Lance fragment with {} clip row(s) is already canonical in {} at version {}; skipping append: "
                "data_files={}",
                len(clip_ids),
                uri,
                dataset.version,
                data_files,
            )
            return int(dataset.version)
        if present:
            msg = (
                f"Atomic append invariant violated for {uri}: {len(present)} of "
                f"{len(clip_ids)} candidate clip IDs are canonical"
            )
            raise RuntimeError(msg)

        transaction = lance.Transaction(
            read_version=dataset.version,
            operation=lance.LanceOperation.Append([fragment]),
            transaction_properties=_TRANSACTION_PROPERTIES,
        )
        logger.info(
            "Committing staged Lance fragment with {} clip row(s) to {} from version {} (attempt {}/{}): data_files={}",
            len(clip_ids),
            uri,
            dataset.version,
            attempt,
            attempts,
            data_files,
        )
        try:
            committed = lance.LanceDataset.commit(
                uri,
                transaction,
                storage_options=storage_options,
                enable_v2_manifest_paths=True,
            )
        except Exception as exc:  # All failed commit responses are potentially ambiguous.
            latest, present = _candidate_presence(clip_ids, uri=uri, storage_options=storage_options)
            if len(present) == len(clip_ids):
                logger.info(
                    "Confirmed staged Lance fragment with {} clip row(s) in {} at version {} after an ambiguous "
                    "commit response: data_files={}",
                    len(clip_ids),
                    uri,
                    latest.version,
                    data_files,
                )
                return int(latest.version)
            if present:
                msg = (
                    f"Atomic append invariant violated for {uri}: {len(present)} of "
                    f"{len(clip_ids)} candidate clip IDs are canonical"
                )
                raise RuntimeError(msg) from exc
            if attempt == attempts:
                raise
            continue
        logger.info(
            "Committed staged Lance fragment with {} clip row(s) to {} at version {}: data_files={}",
            len(clip_ids),
            uri,
            committed.version,
            data_files,
        )
        return int(committed.version)

    msg = "Append attempt loop exited unexpectedly"
    raise AssertionError(msg)


def _parse_candidate(candidate_json: str) -> tuple[lance.FragmentMetadata, tuple[str, ...]]:
    candidate: Any = json.loads(candidate_json)
    if not isinstance(candidate, dict) or not isinstance(candidate.get("fragment"), dict):
        msg = "Invalid staged fragment candidate payload"
        raise TypeError(msg)
    raw_clip_ids = candidate.get("clip_ids")
    if (
        not isinstance(raw_clip_ids, list)
        or not raw_clip_ids
        or not all(isinstance(value, str) for value in raw_clip_ids)
    ):
        msg = "Staged fragment candidate must contain nonempty string clip IDs"
        raise ValueError(msg)
    clip_ids = tuple(raw_clip_ids)
    if len(set(clip_ids)) != len(clip_ids):
        msg = "Staged fragment candidate contains duplicate clip IDs"
        raise ValueError(msg)
    fragment = lance.FragmentMetadata.from_json(json.dumps(candidate["fragment"]))
    if fragment.num_rows != len(clip_ids):
        msg = f"Staged fragment has {fragment.num_rows} rows for {len(clip_ids)} candidate clip IDs"
        raise ValueError(msg)
    return fragment, clip_ids


def _candidate_presence(
    clip_ids: tuple[str, ...],
    *,
    uri: str,
    storage_options: dict[str, str] | None,
) -> tuple[lance.LanceDataset, set[str]]:
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        msg = f"Canonical Lance table disappeared while resolving an append: {uri}"
        raise RuntimeError(msg)
    validate_clip_table(dataset, uri=uri)
    candidate_filter = pc.field("clip_id").isin(pa.array(clip_ids, type=pa.string()))
    committed = dataset.to_table(columns=["clip_id"], filter=candidate_filter)
    committed_ids = [str(value) for value in committed["clip_id"].to_pylist()]
    if len(set(committed_ids)) != len(committed_ids):
        msg = "Canonical table contains duplicate candidate clip IDs"
        raise RuntimeError(msg)
    return dataset, set(committed_ids)
