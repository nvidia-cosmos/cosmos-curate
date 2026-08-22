# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic S3 source realization."""

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.recipes.video_split import discovery
from cosmos_curator.next.recipes.video_split.config import VideoSplitInputConfig, resolve_config_data


def _resolved_input(raw_input: dict[str, object]) -> VideoSplitInputConfig:
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-split",
            "input": raw_input,
            "output": {"media_root": "s3://example-bucket/output"},
        }
    ).input


def test_explicit_selection_needs_no_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit URIs are already a complete realized source set."""
    monkeypatch.setattr(
        discovery,
        "get_storage_client",
        lambda *_args, **_kwargs: pytest.fail("explicit selection must not list S3"),
    )

    selected = discovery.resolve_input_selection(
        _resolved_input(
            {
                "uris": [
                    "s3://example-bucket/b.mp4",
                    "s3://example-bucket/a.mp4",
                ]
            }
        )
    )

    expected = ("s3://example-bucket/a.mp4", "s3://example-bucket/b.mp4")
    assert selected.canonical_uris == expected
    assert selected.scheduled_uris == expected


def test_root_discovery_is_canonicalized_and_scheduled_largest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Root listing size prioritizes execution without changing canonical URI order."""
    seen: list[str] = []

    class FakeS3Client:
        def list_recursive(self, root: S3Prefix) -> list[dict[str, object]]:
            seen.append(root.path)
            return [
                {"Key": "raw/nested/z.MP4", "Size": 300},
                {"Key": "raw/readme.txt", "Size": 1_000},
                {"Key": "raw/a.mp4", "Size": 100},
                {"Key": "raw/a.mp4", "Size": 100},
            ]

    monkeypatch.setattr(discovery, "S3Client", FakeS3Client)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: FakeS3Client())

    selected = discovery.resolve_input_selection(
        _resolved_input({"root_uri": "s3://example-bucket/raw"}),
        storage_profile="training",
    )

    assert seen == ["s3://example-bucket/raw/"]
    assert selected.canonical_uris == (
        "s3://example-bucket/raw/a.mp4",
        "s3://example-bucket/raw/nested/z.MP4",
    )
    assert selected.scheduled_uris == (
        "s3://example-bucket/raw/nested/z.MP4",
        "s3://example-bucket/raw/a.mp4",
    )


def test_root_schedule_is_deterministic_for_ties_and_missing_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown sizes follow known sizes and every tie falls back to URI order."""

    class FakeS3Client:
        def list_recursive(self, _root: S3Prefix) -> list[dict[str, object]]:
            return [
                {"Key": "raw/d.mp4"},
                {"Key": "raw/c.mp4", "Size": "unknown"},
                {"Key": "raw/b.mp4", "Size": 100},
                {"Key": "raw/a.mp4", "Size": 100},
            ]

    monkeypatch.setattr(discovery, "S3Client", FakeS3Client)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: FakeS3Client())

    selected = discovery.resolve_input_selection(_resolved_input({"root_uri": "s3://example-bucket/raw"}))

    assert selected.scheduled_uris == (
        "s3://example-bucket/raw/a.mp4",
        "s3://example-bucket/raw/b.mp4",
        "s3://example-bucket/raw/c.mp4",
        "s3://example-bucket/raw/d.mp4",
    )


def test_root_discovery_rejects_client_results_outside_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A buggy/custom S3 backend cannot expand a lexical sibling into the selection."""

    class FakeS3Client:
        def list_recursive(self, _root: S3Prefix) -> list[dict[str, object]]:
            return [{"Key": "raw-old/a.mp4", "Size": 100}]

    monkeypatch.setattr(discovery, "S3Client", FakeS3Client)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: FakeS3Client())

    with pytest.raises(ValueError, match="outside"):
        discovery.resolve_input_selection(_resolved_input({"root_uri": "s3://example-bucket/raw"}))
