# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact dispatch tests for Curator Next versus legacy video split."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator
from cosmos_curator.client.pipeline_cli import pipeline_runtime
from cosmos_curator.client.pipeline_cli.pipeline_config import load_pipeline_kind_name
from cosmos_curator.next.recipes.video_split import pipeline as next_video_split

runner = CliRunner()


def _write_next_config(path: Path) -> Path:
    path.write_text(
        """schema_version: 1
kind: video-split
input:
  uris: [s3://example-bucket/raw/a.mp4]
output:
  media_root: s3://example-bucket/output
""",
        encoding="utf-8",
    )
    return path


def test_kind_loading_does_not_normalize_video_split_hyphen(tmp_path: Path) -> None:
    """The two discriminator spellings remain different strategies."""
    next_config = _write_next_config(tmp_path / "next.yaml")
    legacy_config = tmp_path / "legacy.yaml"
    legacy_config.write_text("kind: video_split\n", encoding="utf-8")

    assert load_pipeline_kind_name(next_config) == "video-split"
    assert load_pipeline_kind_name(legacy_config) == "video_split"


def test_cli_exposes_distinct_templates() -> None:
    """Template lookup uses the exact requested kind."""
    next_result = runner.invoke(cosmos_curator, ["pipeline", "template", "video-split"])
    legacy_result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split"])

    assert next_result.exit_code == 0
    assert legacy_result.exit_code == 0
    assert yaml.safe_load(next_result.stdout)["kind"] == "video-split"
    assert yaml.safe_load(legacy_result.stdout)["kind"] == "video_split"


def test_runtime_dispatches_hyphenated_kind_to_next_recipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run-pipeline delegates video-split to the new snapshot recipe."""
    expected = {
        "sources": 1,
        "sources_succeeded": 1,
        "sources_failed": 0,
        "clips_planned": 2,
        "clips_published": 2,
        "clips_failed": 0,
        "clips_lance_uri": "s3://example-bucket/output/lance/clips.lance",
        "clips_lance_version": 1,
        "sources_lance_uri": "s3://example-bucket/output/lance/sources.lance",
        "sources_lance_version": 1,
    }
    monkeypatch.setattr(next_video_split, "run_config", lambda _config: expected)

    pipeline_runtime.main(_write_next_config(tmp_path / "next.yaml"), set_overrides=None, json_output=True)

    assert json.loads(capsys.readouterr().out) == expected
