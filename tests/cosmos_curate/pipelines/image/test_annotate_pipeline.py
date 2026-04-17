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

"""Tests for the builder-based image annotate pipeline."""

import argparse
import pathlib

from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curate.pipelines.image.annotate_pipeline import _assemble_stages, add_annotate_command


def test_add_annotate_command_registers_subcommand(tmp_path: pathlib.Path) -> None:
    """The annotate CLI should parse the basic required image pipeline arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)

    args = parser.parse_args(["annotate", "--input-image-path", str(tmp_path / "in"), "--output-path", str(tmp_path)])
    assert args.command == "annotate"
    assert callable(args.func)


def test_assemble_stages_without_captioning_returns_ingest_and_output(tmp_path: pathlib.Path) -> None:
    """Disabling captions should leave only ingest and output stage specs."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--no-generate-captions",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 2
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)


def test_assemble_stages_with_captioning_returns_four_specs(tmp_path: pathlib.Path) -> None:
    """Enabling captions should return ingest, prep, caption, and output stage specs."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_annotate_command(subparsers)
    args = parser.parse_args(
        [
            "annotate",
            "--input-image-path",
            str(tmp_path / "in"),
            "--output-path",
            str(tmp_path / "out"),
            "--captioning-algorithm",
            "qwen",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 4
    assert all(isinstance(stage, CuratorStageSpec) for stage in stages)
