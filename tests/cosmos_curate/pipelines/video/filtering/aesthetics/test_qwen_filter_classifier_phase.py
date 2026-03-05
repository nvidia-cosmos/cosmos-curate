# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for QwenFilterClassifierPhase (unified filter/classifier phase)."""

import pytest

from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curate.pipelines.video.filtering.aesthetics.phases import (
    QwenFilterClassifierPhase,
    QwenFilterConfig,
    QwenVideoClassifierConfig,
)
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_stages import (
    QwenFilteringStage,
    QwenInputPreparationStageFiltering,
    QwenVideoClassifierStage,
)


def test_qwen_filter_classifier_phase_requires_at_least_one_config() -> None:
    """Phase must receive at least one of filter_config or classifier_config."""
    with pytest.raises(ValueError, match="At least one of filter_config or classifier_config is required"):
        QwenFilterClassifierPhase()


def test_qwen_filter_classifier_phase_name_filter_only() -> None:
    """Phase name is qwen_filter when only filter config is set."""
    phase = QwenFilterClassifierPhase(filter_config=QwenFilterConfig())
    assert phase.name == "qwen_filter"


def test_qwen_filter_classifier_phase_name_classifier_only() -> None:
    """Phase name is qwen_video_classifier when only classifier config is set."""
    phase = QwenFilterClassifierPhase(classifier_config=QwenVideoClassifierConfig())
    assert phase.name == "qwen_video_classifier"


def test_qwen_filter_classifier_phase_name_both() -> None:
    """Phase name is qwen_filter_and_classifier when both configs are set."""
    phase = QwenFilterClassifierPhase(
        filter_config=QwenFilterConfig(),
        classifier_config=QwenVideoClassifierConfig(),
    )
    assert phase.name == "qwen_filter_and_classifier"


def test_qwen_filter_classifier_phase_requires_and_populates() -> None:
    """Phase requires transcoded and populates qwen_filtered."""
    phase = QwenFilterClassifierPhase(filter_config=QwenFilterConfig())
    assert phase.requires == frozenset({"transcoded"})
    assert phase.populates == frozenset({"qwen_filtered"})


def test_qwen_filter_classifier_phase_build_stages_filter_only() -> None:
    """Filter-only phase builds prep + filter stages (2 stages)."""
    phase = QwenFilterClassifierPhase(filter_config=QwenFilterConfig())
    stages = phase.build_stages()
    assert len(stages) == 2
    assert isinstance(stages[0], QwenInputPreparationStageFiltering)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, QwenFilteringStage)


def test_qwen_filter_classifier_phase_build_stages_classifier_only() -> None:
    """Classifier-only phase builds prep + classifier stages (2 stages)."""
    phase = QwenFilterClassifierPhase(classifier_config=QwenVideoClassifierConfig())
    stages = phase.build_stages()
    assert len(stages) == 2
    assert isinstance(stages[0], QwenInputPreparationStageFiltering)
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, QwenVideoClassifierStage)


def test_qwen_filter_classifier_phase_build_stages_both() -> None:
    """Both configs: single prep, then classifier, then filter (3 stages, no double prep)."""
    phase = QwenFilterClassifierPhase(
        filter_config=QwenFilterConfig(),
        classifier_config=QwenVideoClassifierConfig(),
    )
    stages = phase.build_stages()
    assert len(stages) == 3
    # Single prep (feeds both classifier and filter via extra_outputs)
    assert isinstance(stages[0], QwenInputPreparationStageFiltering)
    # Classifier
    assert isinstance(stages[1], CuratorStageSpec)
    assert isinstance(stages[1].stage, QwenVideoClassifierStage)
    # Filter (reads from same prep under model_input_key)
    assert isinstance(stages[2], CuratorStageSpec)
    assert isinstance(stages[2].stage, QwenFilteringStage)
