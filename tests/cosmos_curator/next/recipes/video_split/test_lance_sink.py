# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical table bootstrap and incremental fragment appends."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_split import lance_sink
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA, clip_table

_SOURCE_URI = "s3://example-bucket/raw/source.mp4"
_UNKNOWN_FRAME_COUNT = -1


def _record(clip_id: str, *, frame_count: int = 300) -> dict[str, Any]:
    return {
        "record_type": "clip",
        "record_schema_version": 1,
        "media_contract_version": 1,
        "source_id": make_source_id(_SOURCE_URI),
        "source_uri": _SOURCE_URI,
        "source_size_bytes": 1024,
        "source_duration_ns": 30_000_000_000,
        "source_width": 1920,
        "source_height": 1080,
        "source_frame_rate": 30.0,
        "source_frame_count": frame_count,
        "source_video_codec": "h264",
        "start_ns": 0,
        "end_ns": 10_000_000_000,
        "clip_id": clip_id,
        "clip_uri": f"s3://example-bucket/output/clips/{clip_id}.mp4",
        "clip_size_bytes": 512,
        "clip_duration_ns": 10_000_000_000,
        "clip_width": 1920,
        "clip_height": 1080,
        "clip_frame_rate": 30.0,
        "clip_frame_count": frame_count,
        "clip_video_codec": "h264",
    }


def _clips(*clip_ids: str, frame_count: int = 300) -> pa.Table:
    return clip_table(pa.Table.from_pylist([_record(clip_id, frame_count=frame_count) for clip_id in clip_ids]))


def _bootstrap(uri: str) -> lance.LanceDataset:
    return lance_sink.open_or_create_clip_table(uri=uri, storage_profile="default")


def _stage(uri: str, *clip_ids: str) -> str:
    candidate = lance_sink.write_clip_fragment(_clips(*clip_ids), uri=uri, storage_profile="default")
    assert candidate is not None
    return candidate


def _append(uri: str, candidate: str, *, attempts: int = 1) -> int:
    return lance_sink.append_clip_fragment(
        candidate,
        uri=uri,
        storage_profile="default",
        attempts=attempts,
    )


def test_absent_table_is_bootstrapped_once_with_zero_rows(tmp_path: Path) -> None:
    """The durable schema exists before the first clip and is never recreated."""
    uri = str(tmp_path / "clips.lance")

    created = _bootstrap(uri)
    reopened = _bootstrap(uri)

    assert created.version == reopened.version == 1
    assert reopened.count_rows() == 0
    assert reopened.schema == CLIP_SCHEMA


def test_ambiguous_bootstrap_reopens_the_created_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lost create response is resolved from the canonical destination."""
    uri = str(tmp_path / "clips.lance")
    real_write_dataset = lance.write_dataset

    def create_then_raise(*args: object, **kwargs: object) -> lance.LanceDataset:
        real_write_dataset(*args, **kwargs)
        message = "response lost"
        raise OSError(message)

    monkeypatch.setattr(lance, "write_dataset", create_then_raise)

    dataset = _bootstrap(uri)

    assert dataset.count_rows() == 0
    assert dataset.schema == CLIP_SCHEMA


def test_existing_nullable_curation_fields_are_preserved_on_append(tmp_path: Path) -> None:
    """Split fragments omit enrichment fields and Lance presents them as null."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri).add_columns(pa.field("caption__test_v1", pa.string(), nullable=True))

    version = _append(uri, _stage(uri, "a"))

    dataset = lance.dataset(uri)
    expected = _record("a")
    expected.pop("record_type")
    assert dataset.version == version
    assert dataset.schema.field("caption__test_v1").nullable
    assert dataset.to_table().to_pylist() == [expected | {"caption__test_v1": None}]


def test_existing_table_must_contain_compatible_splitting_fields(tmp_path: Path) -> None:
    """A table with a different canonical row contract cannot be reused."""
    uri = str(tmp_path / "clips.lance")
    incompatible_schema = CLIP_SCHEMA.remove(CLIP_SCHEMA.get_field_index("clip_id"))
    lance.write_dataset(
        pa.Table.from_batches([], schema=incompatible_schema),
        uri,
        mode="create",
        data_storage_version="2.2",
    )

    with pytest.raises(ValueError, match=r"missing splitting-owned field.*clip_id"):
        _bootstrap(uri)


def test_existing_curation_fields_must_be_nullable(tmp_path: Path) -> None:
    """Every extension must accept null from future splitting-only fragments."""
    uri = str(tmp_path / "clips.lance")
    incompatible_schema = CLIP_SCHEMA.append(pa.field("caption__test_v1", pa.string(), nullable=False))
    lance.write_dataset(
        pa.Table.from_batches([], schema=incompatible_schema),
        uri,
        mode="create",
        data_storage_version="2.2",
    )

    with pytest.raises(ValueError, match=r"non-nullable curation field.*caption__test_v1"):
        _bootstrap(uri)


def test_every_staged_fragment_appends_as_a_new_version(tmp_path: Path) -> None:
    """The schema bootstrap and each fragment have separate visible versions."""
    uri = str(tmp_path / "clips.lance")
    bootstrap_version = _bootstrap(uri).version

    first_version = _append(uri, _stage(uri, "a", "b"))
    second_version = _append(uri, _stage(uri, "c"))

    dataset = lance.dataset(uri)
    assert first_version == bootstrap_version + 1
    assert second_version == first_version + 1
    assert len(dataset.get_fragments()) == 2
    assert sorted(dataset.to_table(columns=["clip_id"])["clip_id"].to_pylist()) == ["a", "b", "c"]


def test_staging_and_committing_emit_fragment_lifecycle_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker staging and driver commit boundaries are visible in operational logs."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    fake_logger = Mock()
    monkeypatch.setattr(lance_sink, "logger", fake_logger)

    _append(uri, _stage(uri, "a", "b"))

    templates = [str(call.args[0]) for call in fake_logger.info.call_args_list]
    assert templates == [
        "Staged Lance fragment with {} clip row(s) for {}: data_files={}",
        "Committing staged Lance fragment with {} clip row(s) to {} from version {} (attempt {}/{}): data_files={}",
        "Committed staged Lance fragment with {} clip row(s) to {} at version {}: data_files={}",
    ]


def test_unknown_frame_counts_are_published_as_null(tmp_path: Path) -> None:
    """Worker sentinels retain the nullable canonical representation."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    clips = _clips("a", frame_count=_UNKNOWN_FRAME_COUNT)
    candidate = lance_sink.write_clip_fragment(clips, uri=uri, storage_profile="default")
    assert candidate is not None

    _append(uri, candidate)

    published = lance.dataset(uri).to_table().to_pylist()[0]
    assert published["clip_frame_count"] is None
    assert published["source_frame_count"] is None


def test_ambiguous_success_is_detected_without_duplicate_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost successful response is accepted from complete candidate presence."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    candidate = _stage(uri, "a", "b")
    real_commit = lance.LanceDataset.commit

    def commit_then_raise(*args: object, **kwargs: object) -> lance.LanceDataset:
        real_commit(*args, **kwargs)
        message = "response lost"
        raise OSError(message)

    monkeypatch.setattr(lance.LanceDataset, "commit", commit_then_raise)

    version = _append(uri, candidate, attempts=2)

    dataset = lance.dataset(uri)
    assert dataset.version == version
    assert sorted(dataset.to_table(columns=["clip_id"])["clip_id"].to_pylist()) == ["a", "b"]
    assert len(dataset.get_fragments()) == 1


def test_successful_append_replay_is_idempotent(tmp_path: Path) -> None:
    """A descriptor replay skips clip IDs that are already canonical."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    candidate = _stage(uri, "a", "b")

    first_version = _append(uri, candidate)
    replay_version = _append(uri, candidate)

    dataset = lance.dataset(uri)
    assert replay_version == first_version == dataset.version
    assert sorted(dataset.to_table(columns=["clip_id"])["clip_id"].to_pylist()) == ["a", "b"]
    assert len(dataset.get_fragments()) == 1


def test_definite_failed_append_retries_the_same_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No candidate presence permits the already staged fragment to retry."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    candidate = _stage(uri, "a")
    real_commit = lance.LanceDataset.commit
    calls = 0

    def fail_then_commit(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "commit rejected before publication"
            raise OSError(message)
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "commit", fail_then_commit)

    _append(uri, candidate, attempts=2)

    assert calls == 2
    assert lance.dataset(uri).to_table(columns=["clip_id"])["clip_id"].to_pylist() == ["a"]


def test_partial_candidate_presence_fails_instead_of_appending_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight rejects partial visibility before attempting another append."""
    uri = str(tmp_path / "clips.lance")
    _bootstrap(uri)
    _append(uri, _stage(uri, "a"))
    candidate = _stage(uri, "a", "b")

    commit = Mock(side_effect=AssertionError("partial candidate unexpectedly reached commit"))
    monkeypatch.setattr(lance.LanceDataset, "commit", commit)

    with pytest.raises(RuntimeError, match="1 of 2 candidate clip IDs"):
        _append(uri, candidate, attempts=2)

    commit.assert_not_called()
    assert lance.dataset(uri).to_table(columns=["clip_id"])["clip_id"].to_pylist() == ["a"]


def test_fragment_writer_requires_the_canonical_schema(tmp_path: Path) -> None:
    """Internal work records cannot leak into the canonical table."""
    with pytest.raises(ValueError, match="canonical clip schema"):
        lance_sink.write_clip_fragment(
            pa.table({"clip_id": ["a"]}),
            uri=str(tmp_path / "clips.lance"),
            storage_profile="default",
        )
