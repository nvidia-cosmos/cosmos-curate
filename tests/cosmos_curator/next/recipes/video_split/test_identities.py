# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for what clip identity does and does not depend on."""

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.video_split.config import resolve_config_data
from cosmos_curator.next.recipes.video_split.identities import make_clip_id, make_source_id

_SPAN = Span(start_ns=0, end_ns=10_000_000_000)


def _config(**overrides: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "video-split",
        "input": {"uris": ["s3://example-bucket/raw/a.mp4"]},
        "output": {"media_root": "s3://example-bucket/output/"},
        **overrides,
    }


def test_clip_id_ignores_encoder_thread_count() -> None:
    """Thread count is a scheduling knob, so it must not relocate published clips."""
    single = resolve_config_data(_config(execution={"encoder_threads": 1}))
    many = resolve_config_data(_config(execution={"encoder_threads": 8}))
    source_id = make_source_id("s3://example-bucket/raw/a.mp4")

    assert make_clip_id(source_id, _SPAN, single.transcode) == make_clip_id(source_id, _SPAN, many.transcode)


def test_clip_id_tracks_media_contract_settings() -> None:
    """Settings that change the output media do change clip identity."""
    default = resolve_config_data(_config())
    rebitrated = resolve_config_data(_config(transcode={"video_bitrate": "8M"}))
    source_id = make_source_id("s3://example-bucket/raw/a.mp4")

    assert make_clip_id(source_id, _SPAN, default.transcode) != make_clip_id(source_id, _SPAN, rebitrated.transcode)


def test_clip_id_tracks_the_requested_span() -> None:
    """Two spans of one source are distinct clips under identical settings."""
    config = resolve_config_data(_config())
    source_id = make_source_id("s3://example-bucket/raw/a.mp4")
    other_span = Span(start_ns=10_000_000_000, end_ns=20_000_000_000)

    assert make_clip_id(source_id, _SPAN, config.transcode) != make_clip_id(source_id, other_span, config.transcode)
    assert make_clip_id(source_id, _SPAN, config.transcode) == make_clip_id(
        source_id, Span(start_ns=0, end_ns=10_000_000_000), config.transcode
    )
