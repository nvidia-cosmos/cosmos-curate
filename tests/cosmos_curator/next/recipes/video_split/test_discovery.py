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

    assert selected == ("s3://example-bucket/a.mp4", "s3://example-bucket/b.mp4")


def test_root_discovery_is_recursive_filtered_and_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A root is listed once with a directory boundary and only MP4 descendants survive."""
    seen: list[str] = []

    class FakeS3Client:
        def list_recursive_directory(self, root: S3Prefix) -> list[S3Prefix]:
            seen.append(root.path)
            return [
                S3Prefix("s3://example-bucket/raw/nested/z.MP4"),
                S3Prefix("s3://example-bucket/raw/readme.txt"),
                S3Prefix("s3://example-bucket/raw/a.mp4"),
                S3Prefix("s3://example-bucket/raw/a.mp4"),
            ]

    monkeypatch.setattr(discovery, "S3Client", FakeS3Client)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: FakeS3Client())

    selected = discovery.resolve_input_selection(
        _resolved_input({"root_uri": "s3://example-bucket/raw"}),
        storage_profile="training",
    )

    assert seen == ["s3://example-bucket/raw/"]
    assert selected == (
        "s3://example-bucket/raw/a.mp4",
        "s3://example-bucket/raw/nested/z.MP4",
    )


def test_root_discovery_rejects_client_results_outside_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A buggy/custom S3 backend cannot expand a lexical sibling into the selection."""

    class FakeS3Client:
        def list_recursive_directory(self, _root: S3Prefix) -> list[S3Prefix]:
            return [S3Prefix("s3://example-bucket/raw-old/a.mp4")]

    monkeypatch.setattr(discovery, "S3Client", FakeS3Client)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: FakeS3Client())

    with pytest.raises(ValueError, match="outside"):
        discovery.resolve_input_selection(_resolved_input({"root_uri": "s3://example-bucket/raw"}))
