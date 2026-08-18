# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in pipeline-kind composition root."""

import subprocess
import sys

import pytest

from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS


def test_registered_kind_names_are_derived_from_concrete_objects() -> None:
    """The composition root never restates discriminator strings as mapping keys."""
    assert BUILTIN_PIPELINE_KINDS.names() == (
        "caption_judge",
        "robot-action-split",
        "video-split",
        "video_split",
    )
    assert tuple(kind.name for kind in BUILTIN_PIPELINE_KINDS) == BUILTIN_PIPELINE_KINDS.names()


def test_importing_composition_root_defers_config_and_runtime_modules() -> None:
    """CLI startup may import adapters, but not recipe models or execution dependencies."""
    probe = (
        "import sys;"
        "from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS;"
        "BUILTIN_PIPELINE_KINDS.names();"
        "print([m for m in sys.modules if m == 'pydantic' or m == 'ray' or m.endswith('.config')])"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"


def test_robot_action_split_underscore_spelling_is_not_supported() -> None:
    """Only the recipe's canonical hyphenated discriminator is registered."""
    with pytest.raises(ValueError, match="Unknown pipeline kind 'robot_action_split'"):
        BUILTIN_PIPELINE_KINDS.get("robot_action_split")


def test_unknown_kind_names_the_valid_ones() -> None:
    """Lookup failures advertise names derived from the registered objects."""
    with pytest.raises(ValueError, match="Unknown pipeline kind 'nope'") as excinfo:
        BUILTIN_PIPELINE_KINDS.get("nope")

    assert "robot-action-split" in str(excinfo.value)
