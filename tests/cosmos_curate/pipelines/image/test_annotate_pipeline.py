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

import pytest

from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curate.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curate.pipelines.image.annotate_pipeline import _assemble_stages, add_annotate_command
from cosmos_curate.pipelines.image.captioning import image_api_caption_stages
from cosmos_curate.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
)
from cosmos_curate.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curate.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage


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


def test_assemble_stages_with_gemini_captioning_returns_three_specs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini should build ingest, Gemini caption, and output stages."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData(gemini=Gemini(api_key="k")))
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
            "gemini",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 3
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageGeminiCaptionStage)


def test_assemble_stages_with_openai_captioning_returns_four_specs(tmp_path: pathlib.Path) -> None:
    """OpenAI should default to ingest, prep, caption, and output stages."""
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
            "openai",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 4
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageOpenAIPrepStage)
    assert isinstance(stages[2].stage, ImageOpenAICaptionStage)


def test_assemble_stages_with_openai_raw_image_skips_prep(tmp_path: pathlib.Path) -> None:
    """OpenAI raw-image mode should not include the local preprocessing stage."""
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
            "openai",
            "--openai-caption-raw-image",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 3
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageOpenAICaptionStage)


def test_assemble_stages_with_local_semantic_filter_returns_filter_specs(tmp_path: pathlib.Path) -> None:
    """Local semantic filtering should run before normal captioning."""
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
            "--semantic-filter",
            "enable",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 7
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageVllmPrepStage)
    assert isinstance(stages[2].stage, ImageVllmCaptionStage)
    assert isinstance(stages[3].stage, ImageSemanticFilterStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[4].stage, ImageVllmPrepStage)
    assert isinstance(stages[5].stage, ImageVllmCaptionStage)
    assert stages[2].stage._result_target == "filter_caption"
    assert stages[5].stage._result_target == "caption"


def test_assemble_stages_with_local_classifier_returns_classifier_specs(tmp_path: pathlib.Path) -> None:
    """Local image classifier should run before normal captioning."""
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
            "--image-classifier",
            "enable",
            "--image-classifier-use-custom-categories",
            "--image-classifier-allow",
            "planet_earth",
        ]
    )

    stages = _assemble_stages(args)
    assert len(stages) == 7
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[3], CuratorStageSpec)
    assert isinstance(stages[1].stage, ImageVllmPrepStage)
    assert isinstance(stages[2].stage, ImageVllmCaptionStage)
    assert isinstance(stages[3].stage, ImageClassifierStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[4].stage, ImageVllmPrepStage)
    assert isinstance(stages[5].stage, ImageVllmCaptionStage)
    assert stages[2].stage._result_target == "filter_caption"
    assert stages[5].stage._result_target == "caption"
