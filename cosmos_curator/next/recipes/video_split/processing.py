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

"""Source-granular download, planning, transcoding, and recovery."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.media.ffmpeg import (
    MediaError,
    ProbeError,
    TranscodeError,
    VideoMetadata,
    assert_video_encoder_available,
    probe_video_path,
    transcode_span_to_path,
)
from cosmos_curator.next.media.spans import Span, fixed_stride_spans, seconds_to_nanoseconds
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig
from cosmos_curator.next.recipes.video_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
    SOURCE_RECORD_SCHEMA_VERSION,
    UNKNOWN_FRAME_COUNT,
)
from cosmos_curator.next.recipes.video_split.identities import make_clip_id, make_source_id
from cosmos_curator.next.recipes.video_split.recovery import restore_source_result, write_source_result
from cosmos_curator.next.recipes.video_split.storage import (
    RETRYABLE_STORAGE_ERRORS,
    STORAGE_ERRORS,
    download_file,
    upload_file,
)
from cosmos_curator.next.recipes.video_split.uris import join_s3_uri

# FFmpeg reports transient transport stalls and permanently unreadable media
# through the same exit code, so both are retried. A deterministic media failure
# therefore costs its attempts before it becomes a failure row; the backoff below
# matters more, because a tight retry loop amplifies S3 throttling instead of
# letting it clear. UnsupportedMediaError is deliberately absent: FFprobe
# answered, so another attempt gets the same answer.
_RETRYABLE_MEDIA_ERRORS = (ProbeError, TranscodeError)

_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 30.0


def process_source(row: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Restore or fully process one source from a single worker-local copy."""
    source_uri = str(row["source_uri"])
    source_id = make_source_id(source_uri)
    execution = config.execution

    restored = restore_source_result(source_uri, source_id, config=config)
    if restored is not None:
        return restored

    with tempfile.TemporaryDirectory(prefix="curator_next_video_split_") as tmp_dir:
        root = Path(tmp_dir)
        source_path = root / "source.mp4"
        try:
            _retry(
                lambda: download_file(
                    source_uri,
                    str(source_path),
                    storage_profile=execution.storage_profile,
                ),
                RETRYABLE_STORAGE_ERRORS,
                attempts=execution.storage_attempts,
                name="source-read",
            )
            size_bytes = source_path.stat().st_size
        except STORAGE_ERRORS as exc:
            return [_source_failure(source_uri, source_id, stage="source-read", error=exc)]

        try:
            metadata = _retry(
                lambda: probe_video_path(source_path, timeout_s=execution.probe_timeout_s),
                _RETRYABLE_MEDIA_ERRORS,
                attempts=execution.probe_attempts,
                name="source-probe",
            )
        except MediaError as exc:
            return [_source_failure(source_uri, source_id, stage="source-probe", error=exc)]

        source_fields = _source_fields(source_uri, source_id, size_bytes, metadata)
        clip_work = _plan_clips(source_fields, config=config)
        if clip_work:
            # A worker whose FFmpeg build lacks the encoder is a broken
            # environment, not a bad source, so it fails the task.
            assert_video_encoder_available(config.transcode.video_encoder)

        outcomes: list[dict[str, Any]] = []
        for clip in clip_work:
            clip_path = root / "clips" / f"{clip['clip_id']}.mp4"
            outcomes.append(
                _process_clip(
                    clip,
                    source_path=source_path,
                    clip_path=clip_path,
                    config=config,
                )
            )
            clip_path.unlink(missing_ok=True)

        source_outcome = _source_outcome(source_fields, outcomes)
        result = [source_outcome, *(outcome for outcome in outcomes if outcome["status"] == "success")]
        if source_outcome["status"] == "success":
            write_source_result(result, config=config)
        return result


def _plan_clips(source_fields: dict[str, Any], *, config: ResolvedVideoSplitConfig) -> list[dict[str, Any]]:
    """Build deterministic clip work after the source has been probed."""
    split = config.split
    spans = fixed_stride_spans(
        int(source_fields["source_duration_ns"]),
        duration_ns=seconds_to_nanoseconds(split.duration_s),
        stride_ns=seconds_to_nanoseconds(split.stride_s),
        min_duration_ns=seconds_to_nanoseconds(split.min_duration_s),
    )
    records: list[dict[str, Any]] = []
    for span in spans:
        clip_id = make_clip_id(str(source_fields["source_id"]), span, config.transcode)
        records.append(
            _clip_work_record(
                source_fields,
                span=span,
                clip_id=clip_id,
                clip_uri=join_s3_uri(config.output.media_root, "clips", f"{clip_id}.mp4"),
            )
        )
    return records


def _process_clip(
    row: dict[str, Any],
    *,
    source_path: Path,
    clip_path: Path,
    config: ResolvedVideoSplitConfig,
) -> dict[str, Any]:
    span = Span(start_ns=int(row["start_ns"]), end_ns=int(row["end_ns"]))
    execution = config.execution
    try:
        _retry(
            lambda: transcode_span_to_path(
                source_path,
                clip_path,
                span,
                config.transcode,
                encoder_threads=execution.encoder_threads,
                timeout_s=execution.transcode_timeout_s,
            ),
            _RETRYABLE_MEDIA_ERRORS,
            attempts=execution.transcode_attempts,
            name="transcode",
        )
    except MediaError as exc:
        return _clip_failure(row, stage="transcode", error=exc)

    try:
        clip_metadata = probe_video_path(clip_path, timeout_s=execution.probe_timeout_s)
        clip_size_bytes = clip_path.stat().st_size
    except MediaError as exc:
        return _clip_failure(row, stage="clip-probe", error=exc)

    # The media destination is shared run configuration. An exhausted upload
    # failure must fail the task/run rather than turn a broken destination into
    # thousands of item failures and publish an empty successful snapshot.
    _retry(
        lambda: upload_file(
            str(clip_path),
            str(row["clip_uri"]),
            storage_profile=execution.storage_profile,
        ),
        RETRYABLE_STORAGE_ERRORS,
        attempts=execution.storage_attempts,
        name="media-write",
    )

    result = dict(row)
    result.update(
        {
            "record_type": "clip_outcome",
            "status": "success",
            "clip_size_bytes": clip_size_bytes,
            "clip_duration_ns": clip_metadata.duration_ns,
            "clip_width": clip_metadata.width,
            "clip_height": clip_metadata.height,
            "clip_frame_rate": clip_metadata.frame_rate,
            "clip_frame_count": _frame_count(clip_metadata),
            "clip_video_codec": clip_metadata.video_codec,
            "error_stage": "",
            "error_message": "",
        }
    )
    return result


def _source_fields(source_uri: str, source_id: str, size_bytes: int, metadata: VideoMetadata) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_uri": source_uri,
        "source_media_known": True,
        "source_size_bytes": size_bytes,
        "source_duration_ns": metadata.duration_ns,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "source_frame_rate": metadata.frame_rate,
        "source_frame_count": _frame_count(metadata),
        "source_video_codec": metadata.video_codec,
    }


def _source_failure(source_uri: str, source_id: str, *, stage: str, error: Exception) -> dict[str, Any]:
    """Build the final outcome for a source that never reached span generation.

    The media columns hold placeholders rather than real values; the publication
    projection reads ``source_media_known`` and publishes nulls for them, so
    nothing downstream mistakes a zero here for a measurement.
    """
    return _base_record(
        {
            "source_id": source_id,
            "source_uri": source_uri,
            "source_media_known": False,
            "source_size_bytes": 0,
            "source_duration_ns": 0,
            "source_width": 0,
            "source_height": 0,
            "source_frame_rate": 0.0,
            "source_frame_count": UNKNOWN_FRAME_COUNT,
            "source_video_codec": "",
        },
        record_type="source_outcome",
        status="failed",
    ) | {
        "record_schema_version": SOURCE_RECORD_SCHEMA_VERSION,
        "error_stage": stage,
        "error_message": _error_message(error),
    }


def _base_record(source_fields: dict[str, Any], *, record_type: str, status: str) -> dict[str, Any]:
    """Build the columns every work record carries.

    Source outcomes and clip rows share one dataset, so both must carry every
    column. The clip-side columns start at placeholders that only a clip row
    goes on to fill in.
    """
    return {
        "record_type": record_type,
        "record_schema_version": CLIP_RECORD_SCHEMA_VERSION,
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        **source_fields,
        # Owned by the source outcome. Publication never reads these counts from
        # clip rows, so carrying per-clip copies would only invite divergence.
        "planned_clip_count": 0,
        "published_clip_count": 0,
        "failed_clip_count": 0,
        "start_ns": 0,
        "end_ns": 0,
        "clip_id": "",
        "clip_uri": "",
        "clip_size_bytes": 0,
        "clip_duration_ns": 0,
        "clip_width": 0,
        "clip_height": 0,
        "clip_frame_rate": 0.0,
        "clip_frame_count": UNKNOWN_FRAME_COUNT,
        "clip_video_codec": "",
        "status": status,
        "error_stage": "",
        "error_message": "",
    }


def _source_outcome(source_fields: dict[str, Any], clip_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one fully processed source before its rows leave the worker."""
    planned = len(clip_outcomes)
    published = sum(outcome["status"] == "success" for outcome in clip_outcomes)
    first_failure = next((outcome for outcome in clip_outcomes if outcome["status"] == "failed"), None)
    return _base_record(
        source_fields,
        record_type="source_outcome",
        status="failed" if first_failure is not None else "success",
    ) | {
        "record_schema_version": SOURCE_RECORD_SCHEMA_VERSION,
        "planned_clip_count": planned,
        "published_clip_count": published,
        "failed_clip_count": planned - published,
        "error_stage": "" if first_failure is None else first_failure["error_stage"],
        "error_message": "" if first_failure is None else first_failure["error_message"],
    }


def _clip_work_record(
    source_fields: dict[str, Any],
    *,
    span: Span,
    clip_id: str,
    clip_uri: str,
) -> dict[str, Any]:
    """Build one pending clip work row."""
    return _base_record(source_fields, record_type="clip_work", status="pending") | {
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "clip_id": clip_id,
        "clip_uri": clip_uri,
    }


def _clip_failure(row: dict[str, Any], *, stage: str, error: Exception) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "record_type": "clip_outcome",
            "status": "failed",
            "error_stage": stage,
            "error_message": _error_message(error),
        }
    )
    return result


def _retry[T](
    operation: Callable[[], T],
    exceptions: tuple[type[Exception], ...],
    *,
    attempts: int,
    name: str,
) -> T:
    """Retry one worker-local operation with the house backoff policy."""
    return do_with_retries(
        operation,
        exceptions,
        max_attempts=attempts,
        backoff_factor=_BACKOFF_FACTOR,
        max_wait_time_s=_MAX_BACKOFF_S,
        name=name,
    )


def _frame_count(metadata: VideoMetadata) -> int:
    return metadata.frame_count if metadata.frame_count is not None else UNKNOWN_FRAME_COUNT


def _error_message(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:8192]
