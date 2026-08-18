# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot-action-split config resolution."""

from pathlib import Path

import yaml

from cosmos_curator.next.recipes.robot_action_split.config import resolve_config


def test_shared_dotted_overrides_apply_before_validation(tmp_path: Path) -> None:
    """Robot-action config uses the shared typed override contract."""
    config_path = tmp_path / "robot-action-split.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "robot-action-split",
                "input": {
                    "uris": ["s3://example-bucket/robot-data/dataset/"],
                    "source_dataset": "dataset",
                },
                "output": {
                    "media_root": "s3://example-bucket/output/",
                    "lance_uri": "s3://example-bucket/output/clips.lance",
                },
            }
        ),
        encoding="utf-8",
    )

    config = resolve_config(
        config_path,
        overrides=["execution.progress=true", "split.max_duration_s=null"],
    )

    assert config.execution.progress is True
    assert config.split.max_duration_s is None
