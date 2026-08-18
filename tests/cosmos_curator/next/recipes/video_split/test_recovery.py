# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for atomic source-level recovery results."""

import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split import recovery
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config_data
from cosmos_curator.next.recipes.video_split.identities import make_clip_id, make_source_id
from cosmos_curator.next.recipes.video_split.processing import _clip_work_record, _source_outcome

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"


class _MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def object_exists(self, dest: StoragePrefix) -> bool:
        return str(dest) in self.objects

    def upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        self.objects[str(dest)] = data

    def download_object_as_bytes(self, uri: StoragePrefix) -> bytes:
        return self.objects[str(uri)]


def _config(*, transcode_cpus: float = 1.0, duration_s: float = 10.0) -> ResolvedVideoSplitConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": {"uris": [_SOURCE_URI]},
            "split": {"duration_s": duration_s, "stride_s": 10.0, "min_duration_s": 2.0},
            "output": {"media_root": "s3://example-bucket/output"},
            "execution": {"transcode_cpus": transcode_cpus, "storage_attempts": 1},
        }
    )


def _result(config: ResolvedVideoSplitConfig, *, clip_count: int = 2) -> list[dict[str, Any]]:
    source_id = make_source_id(_SOURCE_URI)
    fields = {
        "source_id": source_id,
        "source_uri": _SOURCE_URI,
        "source_media_known": True,
        "source_size_bytes": 1_024,
        "source_duration_ns": clip_count * 10_000_000_000,
        "source_width": 1_920,
        "source_height": 1_080,
        "source_frame_rate": 30.0,
        "source_frame_count": clip_count * 300,
        "source_video_codec": "h264",
    }
    outcomes: list[dict[str, Any]] = []
    for index in range(clip_count):
        span = Span(start_ns=index * 10_000_000_000, end_ns=(index + 1) * 10_000_000_000)
        clip_id = make_clip_id(source_id, span, config.transcode)
        work = _clip_work_record(
            fields,
            span=span,
            clip_id=clip_id,
            clip_uri=f"{config.output.media_root}/clips/{clip_id}.mp4",
        )
        outcomes.append(
            work
            | {
                "record_type": "clip_outcome",
                "status": "success",
                "clip_size_bytes": 512,
                "clip_duration_ns": 10_000_000_000,
                "clip_width": 1_920,
                "clip_height": 1_080,
                "clip_frame_rate": 30.0,
                "clip_frame_count": 300,
                "clip_video_codec": "h264",
            }
        )
    return [_source_outcome(fields, outcomes), *outcomes]


@pytest.fixture
def memory_storage(monkeypatch: pytest.MonkeyPatch) -> _MemoryStorage:
    """Replace remote recovery storage with an in-memory object store."""
    storage = _MemoryStorage()
    monkeypatch.setattr(recovery, "_client", lambda *_args, **_kwargs: storage)
    return storage


def test_recovery_contract_ignores_execution_tuning() -> None:
    """Changing resource shape must not invalidate semantically identical media."""
    assert recovery.recovery_contract_id(_config(transcode_cpus=0.5)) == recovery.recovery_contract_id(
        _config(transcode_cpus=8.0)
    )
    assert recovery.recovery_contract_id(_config(duration_s=5.0)) != recovery.recovery_contract_id(
        _config(duration_s=10.0)
    )


def test_complete_result_round_trips_as_one_object(memory_storage: _MemoryStorage) -> None:
    """One result contains everything needed to rebuild both snapshots."""
    config = _config()
    result = _result(config)
    source_id = make_source_id(_SOURCE_URI)

    assert recovery.restore_source_result(_SOURCE_URI, source_id, config=config) is None

    recovery.write_source_result(result, config=config)

    assert recovery.restore_source_result(_SOURCE_URI, source_id, config=config) == result
    assert len(memory_storage.objects) == 1
    assert next(iter(memory_storage.objects)).endswith(f"/sources/{source_id}/result.arrow")


def test_valid_zero_clip_source_is_a_complete_result(memory_storage: _MemoryStorage) -> None:
    """A successfully probed short source does not need clip rows to recover."""
    config = _config()
    result = _result(config, clip_count=0)

    recovery.write_source_result(result, config=config)

    assert recovery.restore_source_result(_SOURCE_URI, make_source_id(_SOURCE_URI), config=config) == result
    assert len(memory_storage.objects) == 1


def test_unavailable_recovery_cache_is_treated_as_a_miss(
    memory_storage: _MemoryStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cache service error does not prevent fresh source processing."""
    error = ClientError({"Error": {"Code": "SlowDown", "Message": "retry later"}}, "HeadObject")

    def fail_read(_dest: StoragePrefix) -> bool:
        raise error

    monkeypatch.setattr(memory_storage, "object_exists", fail_read)

    with caplog.at_level(logging.WARNING):
        restored = recovery.restore_source_result(_SOURCE_URI, make_source_id(_SOURCE_URI), config=_config())

    assert restored is None
    assert "processing the source again" in caplog.text


def test_invalid_recovery_cache_entry_is_treated_as_a_miss(
    memory_storage: _MemoryStorage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated checkpoint cannot make an otherwise usable source fatal."""
    config = _config()
    source_id = make_source_id(_SOURCE_URI)
    memory_storage.objects[recovery._result_uri(config, source_id)] = b"truncated"

    with caplog.at_level(logging.WARNING):
        restored = recovery.restore_source_result(_SOURCE_URI, source_id, config=config)

    assert restored is None
    assert "Ignoring invalid recovery cache entry" in caplog.text


def test_self_consistent_incomplete_recovery_cache_is_treated_as_a_miss(
    memory_storage: _MemoryStorage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cached counters cannot hide a clip omitted from the current split plan."""
    config = _config()
    source_id = make_source_id(_SOURCE_URI)
    result = _result(config)
    result.pop()
    result[0]["planned_clip_count"] = 1
    result[0]["published_clip_count"] = 1
    memory_storage.objects[recovery._result_uri(config, source_id)] = recovery._encode_records(result)

    with caplog.at_level(logging.WARNING):
        restored = recovery.restore_source_result(_SOURCE_URI, source_id, config=config)

    assert restored is None
    assert "does not match the expected clip plan" in caplog.text


def test_recovery_clip_ids_must_match_the_current_split_plan() -> None:
    """A valid clip identity for an unplanned span cannot replace a planned clip."""
    config = _config()
    source_id = make_source_id(_SOURCE_URI)
    result = _result(config)
    span = Span(start_ns=5_000_000_000, end_ns=15_000_000_000)
    clip_id = make_clip_id(source_id, span, config.transcode)
    result[-1].update(
        {
            "start_ns": span.start_ns,
            "end_ns": span.end_ns,
            "clip_id": clip_id,
            "clip_uri": f"{config.output.media_root}/clips/{clip_id}.mp4",
        }
    )

    with pytest.raises(ValueError, match="does not match the expected clip plan"):
        recovery._validate_result(result, source_uri=_SOURCE_URI, source_id=source_id, config=config)


def test_recovery_cache_write_failure_does_not_discard_fresh_results(
    memory_storage: _MemoryStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Checkpoint storage is optional after a source has completed successfully."""
    error = ClientError({"Error": {"Code": "SlowDown", "Message": "retry later"}}, "PutObject")

    def fail_write(_dest: StoragePrefix, _data: bytes) -> None:
        raise error

    monkeypatch.setattr(memory_storage, "upload_bytes", fail_write)

    with caplog.at_level(logging.WARNING):
        recovery.write_source_result(_result(_config()), config=_config())

    assert not memory_storage.objects
    assert "continuing without a checkpoint" in caplog.text


def test_incomplete_source_cannot_be_checkpointed(memory_storage: _MemoryStorage) -> None:
    """A failed clip keeps the whole source outside the recovery cache."""
    config = _config()
    result = _result(config)
    result[1] |= {"status": "failed", "error_stage": "transcode", "error_message": "broken GOP"}

    with pytest.raises(ValueError, match="non-success"):
        recovery.write_source_result(result, config=config)

    assert not memory_storage.objects
