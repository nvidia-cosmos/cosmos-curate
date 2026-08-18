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

"""Atomic source-level recovery results for ``video-split``."""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

import pyarrow as pa

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client, path_to_prefix
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig
from cosmos_curator.next.recipes.video_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
    RECOVERY_RECORD_SCHEMA_VERSION,
    SOURCE_RECORD_SCHEMA_VERSION,
)
from cosmos_curator.next.recipes.video_split.identities import canonical_digest, make_clip_id
from cosmos_curator.next.recipes.video_split.records import SOURCE_MEDIA_FIELDS, WORK_RECORD_SCHEMA, work_record_table
from cosmos_curator.next.recipes.video_split.storage import RETRYABLE_STORAGE_ERRORS, STORAGE_ERRORS
from cosmos_curator.next.recipes.video_split.uris import join_s3_uri

_RECOVERY_ROOT = "_recovery/video-split"
_RESULT_FILE = "result.arrow"
_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 30.0

logger = logging.getLogger(__name__)

_SOURCE_IDENTITY_FIELDS = (
    "media_contract_version",
    "source_id",
    "source_uri",
    "source_media_known",
    *SOURCE_MEDIA_FIELDS,
)


def recovery_contract_id(config: ResolvedVideoSplitConfig) -> str:
    """Identify reusable work without including execution-only tuning."""
    return canonical_digest(
        {
            "recovery_record_schema_version": RECOVERY_RECORD_SCHEMA_VERSION,
            "clip_record_schema_version": CLIP_RECORD_SCHEMA_VERSION,
            "source_record_schema_version": SOURCE_RECORD_SCHEMA_VERSION,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "split": config.split.model_dump(mode="json"),
            "transcode": config.transcode.model_dump(mode="json"),
        }
    )


def restore_source_result(
    source_uri: str,
    source_id: str,
    *,
    config: ResolvedVideoSplitConfig,
) -> list[dict[str, Any]] | None:
    """Return one complete saved result, treating an unavailable cache as a miss."""
    try:
        return _retry(
            lambda: _restore_source_result(source_uri, source_id, config=config),
            config=config,
            name="recovery-read",
        )
    except STORAGE_ERRORS as exc:
        logger.warning("Recovery cache read failed for %s; processing the source again: %s", source_uri, exc)
        return None


def _restore_source_result(
    source_uri: str,
    source_id: str,
    *,
    config: ResolvedVideoSplitConfig,
) -> list[dict[str, Any]] | None:
    client = _client(config.output.media_root, config.execution.storage_profile)
    uri = _result_uri(config, source_id)
    prefix = path_to_prefix(uri)
    if not client.object_exists(prefix):
        return None

    payload = client.download_object_as_bytes(prefix)
    try:
        records = _decode_records(payload, location=uri)
        _validate_result(records, source_uri=source_uri, source_id=source_id, config=config)
    except ValueError as exc:
        logger.warning("Ignoring invalid recovery cache entry %s; processing the source again: %s", uri, exc)
        return None
    return records


def write_source_result(records: list[dict[str, Any]], *, config: ResolvedVideoSplitConfig) -> None:
    """Best-effort checkpoint one fully successful source and all of its clips."""
    if not records:
        msg = "Cannot checkpoint an empty source result"
        raise ValueError(msg)
    source_outcome = records[0]
    source_uri = str(source_outcome["source_uri"])
    source_id = str(source_outcome["source_id"])
    _validate_result(records, source_uri=source_uri, source_id=source_id, config=config)
    uri = _result_uri(config, source_id)
    payload = _encode_records(records)
    try:
        _retry(
            lambda: _client(config.output.media_root, config.execution.storage_profile).upload_bytes(
                path_to_prefix(uri), payload
            ),
            config=config,
            name="recovery-write",
        )
    except STORAGE_ERRORS as exc:
        logger.warning("Recovery cache write failed for %s; continuing without a checkpoint: %s", source_uri, exc)


def _validate_result(
    records: list[dict[str, Any]],
    *,
    source_uri: str,
    source_id: str,
    config: ResolvedVideoSplitConfig,
) -> None:
    if not records:
        msg = f"Recovery result for {source_uri} must start with one source_outcome record"
        raise ValueError(msg)
    source_outcome = _validate_source_outcome(records[0], source_uri=source_uri, source_id=source_id)

    outcomes = records[1:]
    planned = int(source_outcome["planned_clip_count"])
    published = int(source_outcome["published_clip_count"])
    failed = int(source_outcome["failed_clip_count"])
    if published != planned or failed != 0:
        msg = f"Recovery result for {source_uri} does not describe a complete source"
        raise ValueError(msg)
    if len(outcomes) != published:
        msg = f"Recovery result for {source_uri} declares {published} published clip(s) but contains {len(outcomes)}"
        raise ValueError(msg)

    expected_clip_ids = _expected_clip_ids(
        source_outcome,
        source_uri=source_uri,
        source_id=source_id,
        planned=planned,
        config=config,
    )

    clip_ids: set[str] = set()
    for row in outcomes:
        _validate_record_versions(row, expected_schema_version=CLIP_RECORD_SCHEMA_VERSION)
        if row.get("record_type") != "clip_outcome" or row.get("status") != "success":
            msg = f"Recovery result for {source_uri} contains a non-success clip outcome"
            raise ValueError(msg)
        if any(row[field] != source_outcome[field] for field in _SOURCE_IDENTITY_FIELDS):
            msg = f"Recovery result for {source_uri} contains a clip for another source"
            raise ValueError(msg)

        span = Span(start_ns=int(row["start_ns"]), end_ns=int(row["end_ns"]))
        clip_id = str(row["clip_id"])
        expected_clip_id = make_clip_id(source_id, span, config.transcode)
        expected_clip_uri = join_s3_uri(config.output.media_root, "clips", f"{expected_clip_id}.mp4")
        if not clip_id or clip_id in clip_ids:
            msg = f"Recovery result for {source_uri} contains an empty or duplicate clip ID"
            raise ValueError(msg)
        if clip_id != expected_clip_id or row.get("clip_uri") != expected_clip_uri:
            msg = f"Recovery result for {source_uri} contains a clip outside its media contract"
            raise ValueError(msg)
        clip_ids.add(clip_id)
    if clip_ids != expected_clip_ids:
        msg = f"Recovery result for {source_uri} does not match the expected clip plan"
        raise ValueError(msg)


def _expected_clip_ids(
    source_outcome: dict[str, Any],
    *,
    source_uri: str,
    source_id: str,
    planned: int,
    config: ResolvedVideoSplitConfig,
) -> set[str]:
    split = config.split
    clip_ids = {
        make_clip_id(source_id, span, config.transcode)
        for span in fixed_stride_spans(
            int(source_outcome["source_duration_ns"]),
            duration_ns=seconds_to_nanoseconds(split.duration_s),
            stride_ns=seconds_to_nanoseconds(split.stride_s),
            min_duration_ns=seconds_to_nanoseconds(split.min_duration_s),
        )
    }
    if planned != len(clip_ids):
        msg = f"Recovery result for {source_uri} does not match the expected clip plan"
        raise ValueError(msg)
    return clip_ids


def _validate_source_outcome(
    source_outcome: dict[str, Any],
    *,
    source_uri: str,
    source_id: str,
) -> dict[str, Any]:
    if source_outcome.get("record_type") != "source_outcome":
        msg = f"Recovery result for {source_uri} must start with one source_outcome record"
        raise ValueError(msg)
    _validate_record_versions(source_outcome, expected_schema_version=SOURCE_RECORD_SCHEMA_VERSION)
    if source_outcome.get("status") != "success":
        msg = f"Recovery result for {source_uri} does not contain a successful source"
        raise ValueError(msg)
    if not source_outcome.get("source_media_known"):
        msg = f"Recovery result for {source_uri} does not contain source media properties"
        raise ValueError(msg)
    if source_outcome.get("source_uri") != source_uri or source_outcome.get("source_id") != source_id:
        msg = f"Recovery result identity does not match {source_uri}"
        raise ValueError(msg)
    return source_outcome


def _validate_record_versions(row: dict[str, Any], *, expected_schema_version: int) -> None:
    if int(row["record_schema_version"]) != expected_schema_version:
        msg = "Recovery record has an incompatible record schema version"
        raise ValueError(msg)
    if int(row["media_contract_version"]) != MEDIA_CONTRACT_VERSION:
        msg = "Recovery record has an incompatible media contract version"
        raise ValueError(msg)


def _encode_records(records: list[dict[str, Any]]) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, WORK_RECORD_SCHEMA) as writer:
        writer.write_table(work_record_table(records))
    return cast("bytes", sink.getvalue().to_pybytes())


def _decode_records(payload: bytes, *, location: str) -> list[dict[str, Any]]:
    try:
        table = pa.ipc.open_file(pa.BufferReader(payload)).read_all()
    except (pa.ArrowInvalid, pa.ArrowIOError) as exc:
        msg = f"Could not read video-split recovery record {location}"
        raise ValueError(msg) from exc
    if not table.schema.equals(WORK_RECORD_SCHEMA, check_metadata=True):
        msg = f"Video-split recovery record has an incompatible schema: {location}"
        raise ValueError(msg)
    return cast("list[dict[str, Any]]", table.to_pylist())


def _result_uri(config: ResolvedVideoSplitConfig, source_id: str) -> str:
    return join_s3_uri(
        config.output.media_root,
        _RECOVERY_ROOT,
        recovery_contract_id(config),
        "sources",
        source_id,
        _RESULT_FILE,
    )


@lru_cache(maxsize=8)
def _client(media_root: str, storage_profile: str) -> StorageClient:
    client = get_storage_client(media_root, profile_name=storage_profile, can_overwrite=True)
    if client is None:
        msg = f"Video-split recovery requires remote storage, got {media_root}"
        raise TypeError(msg)
    return client


def _retry[T](operation: Callable[[], T], *, config: ResolvedVideoSplitConfig, name: str) -> T:
    return do_with_retries(
        operation,
        RETRYABLE_STORAGE_ERRORS,
        max_attempts=config.execution.storage_attempts,
        backoff_factor=_BACKOFF_FACTOR,
        max_wait_time_s=_MAX_BACKOFF_S,
        name=name,
    )
