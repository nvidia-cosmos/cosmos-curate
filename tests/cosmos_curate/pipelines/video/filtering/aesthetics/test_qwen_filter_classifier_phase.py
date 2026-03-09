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

"""Unit tests for QwenFilterClassifierPhase (unified filter/classifier phase) and classifier metadata."""

import pathlib
import uuid

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
from cosmos_curate.pipelines.video.utils.data_model import Clip, Video, Window


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


def test_qwen_video_classifier_stage_custom_categories_requires_allow_or_block() -> None:
    """custom_categories=True requires at least one of type_allow or type_block."""
    with pytest.raises(ValueError, match="custom_categories=True requires at least one of type_allow or type_block"):
        QwenVideoClassifierStage(custom_categories=True)


def test_qwen_video_classifier_stage_custom_categories_builds_valid_labels() -> None:
    """With custom_categories=True, allow+block union becomes the only categories."""
    stage = QwenVideoClassifierStage(
        custom_categories=True,
        type_allow="penguins",
        type_block="polar_bears",
    )
    assert stage._valid_type_labels == ("penguins", "polar_bears")


def test_qwen_filter_classifier_phase_build_stages_custom_categories() -> None:
    """Phase with custom_categories passes filter_categories to prep so prompt uses only those."""
    phase = QwenFilterClassifierPhase(
        classifier_config=QwenVideoClassifierConfig(
            custom_categories=True,
            type_allow="penguins",
            type_block="polar_bears",
        ),
    )
    stages = phase.build_stages()
    assert len(stages) == 2
    prep = stages[0]
    assert isinstance(prep, QwenInputPreparationStageFiltering)
    assert prep._filter_categories == "penguins,polar_bears"


def test_qwen_filter_classifier_phase_build_stages_default_categories_prep_gets_none() -> None:
    """When custom_categories=False, prep gets filter_categories=None (27 default labels)."""
    phase = QwenFilterClassifierPhase(classifier_config=QwenVideoClassifierConfig())
    stages = phase.build_stages()
    prep = stages[0]
    assert isinstance(prep, QwenInputPreparationStageFiltering)
    assert prep._filter_categories is None


def test_qwen_filter_classifier_phase_custom_categories_union_only_allow() -> None:
    """Custom categories with only allow list uses that list for prep."""
    phase = QwenFilterClassifierPhase(
        classifier_config=QwenVideoClassifierConfig(
            custom_categories=True,
            type_allow="planet_earth,mountains",
            type_block=None,
        ),
    )
    stages = phase.build_stages()
    assert stages[0]._filter_categories == "mountains,planet_earth"


def test_qwen_filter_classifier_phase_custom_categories_union_only_block() -> None:
    """Custom categories with only block list uses that list for prep."""
    phase = QwenFilterClassifierPhase(
        classifier_config=QwenVideoClassifierConfig(
            custom_categories=True,
            type_allow=None,
            type_block="space",
        ),
    )
    stages = phase.build_stages()
    assert stages[0]._filter_categories == "space"


def _make_video_with_one_clip_one_window() -> tuple[Video, int]:
    """Minimal Video with one clip and one filter_window for _type_mode_filter_clip."""
    window = Window(start_frame=0, end_frame=124)
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video="test.mp4",
        span=(0.0, 5.0),
    )
    clip.filter_windows.append(window)
    video = Video(input_video=pathlib.Path("test.mp4"), clips=[clip])
    return video, 0


def test_qwen_video_classifier_rejection_reasons_only_allow_no_and_block_yes() -> None:
    """Rejection reasons must only include allow-list 'no' and block-list 'yes', not block 'no'."""
    stage = QwenVideoClassifierStage(
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    result_json = '{"planet_earth": "no", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues, rejected_windows)
    cap = video.clips[clip_idx].filter_windows[0].caption["qwen_rejection_reasons"]
    assert "planet_earth" in cap
    assert "no" in cap
    assert "mountains" in cap
    assert "space" not in cap


def test_qwen_video_classifier_rejection_reasons_block_yes_included() -> None:
    """Rejection reasons include block-list categories when model says yes."""
    stage = QwenVideoClassifierStage(
        custom_categories=True,
        type_allow="planet_earth",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    result_json = '{"planet_earth": "no", "space": "yes"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues, rejected_windows)
    cap = video.clips[clip_idx].filter_windows[0].caption["qwen_rejection_reasons"]
    assert "space" in cap
    assert "yes" in cap
    assert "planet_earth" in cap
    assert "no" in cap


def test_qwen_video_classifier_custom_categories_empty_gets_unclassified() -> None:
    """When custom_categories=True and no category is yes, classification is ['unclassified']."""
    stage = QwenVideoClassifierStage(
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    result_json = '{"planet_earth": "no", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues, rejected_windows)
    assert video.clips[clip_idx].qwen_type_classification == ["unclassified"]


def test_qwen_video_classifier_custom_categories_match_not_other() -> None:
    """When custom_categories=True and at least one category is yes, classification lists them."""
    stage = QwenVideoClassifierStage(
        custom_categories=True,
        type_allow="planet_earth,mountains",
        type_block="space",
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    result_json = '{"planet_earth": "yes", "mountains": "no", "space": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues, rejected_windows)
    assert video.clips[clip_idx].qwen_type_classification == ["planet_earth"]


def test_qwen_video_classifier_default_categories_empty_not_unclassified() -> None:
    """When custom_categories=False and no category is yes, classification is [] (no default fallback)."""
    stage = QwenVideoClassifierStage(
        custom_categories=False,
        type_allow="nature_environment",
        type_block=None,
    )
    video, clip_idx = _make_video_with_one_clip_one_window()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    result_json = '{"nature_environment": "no"}'
    stage._type_mode_filter_clip(video, clip_idx, [(0, result_json)], all_issues, rejected_windows)
    assert video.clips[clip_idx].qwen_type_classification == []
