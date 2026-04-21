# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image semantic filter/classifier stage builders."""

import pytest

from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curate.pipelines.image.captioning.image_vllm_stages import ImageVllmCaptionStage, ImageVllmPrepStage
from cosmos_curate.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curate.pipelines.image.filtering.filtering_builders import (
    ImageClassifierConfig,
    ImageSemanticFilterConfig,
    build_image_filter_classifier_stages,
)


def test_build_image_filter_stages_returns_local_vllm_and_shared_postprocess() -> None:
    """Semantic filter builder should return local vLLM prep/inference plus shared postprocess."""
    stages = build_image_filter_classifier_stages(filter_config=ImageSemanticFilterConfig(enabled=True))

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageVllmPrepStage)
    assert isinstance(stages[1].stage, ImageVllmCaptionStage)
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert stages[1].stage._result_target == "filter_caption"


def test_build_image_classifier_stages_returns_local_vllm_and_shared_postprocess() -> None:
    """Classifier builder should return local vLLM prep/inference plus shared postprocess."""
    stages = build_image_filter_classifier_stages(
        classifier_config=ImageClassifierConfig(enabled=True, custom_categories=True, type_allow="planet_earth")
    )

    assert len(stages) == 3
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageVllmPrepStage)
    assert isinstance(stages[1].stage, ImageVllmCaptionStage)
    assert isinstance(stages[2].stage, ImageClassifierStage)
    assert stages[1].stage._result_target == "filter_caption"


def test_build_image_filter_classifier_requires_a_config() -> None:
    """Builder should require at least one filter or classifier config."""
    with pytest.raises(ValueError, match="At least one of filter_config or classifier_config is required"):
        build_image_filter_classifier_stages()


def test_build_image_filter_classifier_stages_orders_filter_before_classifier() -> None:
    """Semantic filter stages should run before classifier stages and use distinct caption keys."""
    stages = build_image_filter_classifier_stages(
        filter_config=ImageSemanticFilterConfig(enabled=True, model_variant="qwen"),
        classifier_config=ImageClassifierConfig(enabled=True, model_variant="qwen"),
    )

    assert len(stages) == 6
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, ImageSemanticFilterStage)
    assert isinstance(stages[4], CuratorStageSpec)
    assert isinstance(stages[5], CuratorStageSpec)
    assert isinstance(stages[5].stage, ImageClassifierStage)
    assert stages[1].stage._result_key == "semantic:qwen"
    assert stages[2].stage._filter_caption_key == "semantic:qwen"
    assert stages[4].stage._result_key == "classifier:qwen"
    assert stages[5].stage._filter_caption_key == "classifier:qwen"
