# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared image semantic filter/classifier postprocess stages."""

import pathlib

import numpy as np

from cosmos_curate.core.utils.data.lazy_data import LazyData
from cosmos_curate.pipelines.image.captioning.image_vllm_stages import _collect_caption_inputs
from cosmos_curate.pipelines.image.filtering.filter_stages import ImageClassifierStage, ImageSemanticFilterStage
from cosmos_curate.pipelines.image.utils.data_model import Image, ImagePipeTask


def _make_task() -> ImagePipeTask:
    """Create a minimal image task for shared postprocess-stage tests."""
    image = Image(
        input_image=pathlib.Path("example.jpg"),
        relative_path="example.jpg",
        encoded_data=LazyData.coerce(np.frombuffer(b"\xff\xd8\xff", dtype=np.uint8)),
    )
    return ImagePipeTask(session_id="example", image=image)


def test_image_semantic_filter_stage_drops_filtered_tasks() -> None:
    """Semantic postprocess should mark an image filtered when a rejection criterion matches."""
    task = _make_task()
    task.image.filter_captions["qwen"] = '{"synthetic_image": "yes"}'
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image")

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "semantic"


def test_image_classifier_stage_sets_classification_and_keeps_allowed_task() -> None:
    """Classifier postprocess should record labels and keep an allowed image."""
    task = _make_task()
    task.image.filter_captions["qwen"] = '{"planet_earth": "yes", "space": "no"}'
    stage = ImageClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
        type_block="space",
    )

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is False
    assert task.image.qwen_type_classification == ["planet_earth"]


def test_image_semantic_filter_stage_score_only_keeps_missing_caption_as_error() -> None:
    """Score-only semantic postprocess should keep images with missing filter captions."""
    task = _make_task()
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image", score_only=True)

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is False
    assert task.image.errors.get("qwen") == "all_windows_failed_preparation"
    assert task.image.qwen_rejection_stage is None


def test_image_semantic_filter_stage_score_only_writes_rejection_reasons() -> None:
    """Score-only mode must still write qwen_rejection_reasons so callers can see the scores."""
    task = _make_task()
    task.image.filter_captions["qwen"] = '{"synthetic_image": "yes"}'
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image", score_only=True)

    stage.process_data([task])

    assert task.image.is_filtered is False
    assert task.image.qwen_rejection_reasons is not None
    assert "synthetic image" in task.image.qwen_rejection_reasons


def test_image_semantic_filter_stage_passing_image_has_no_rejection_reasons() -> None:
    """A passing image in non-score-only mode must not have qwen_rejection_reasons written."""
    task = _make_task()
    task.image.filter_captions["qwen"] = '{"synthetic_image": "no"}'
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image")

    stage.process_data([task])

    assert task.image.is_filtered is False
    assert task.image.qwen_rejection_reasons is None


def test_image_semantic_filter_stage_marks_malformed_model_output_as_error() -> None:
    """Malformed semantic-filter output should filter the image and store the error."""
    task = _make_task()
    task.image.filter_captions["qwen"] = "not valid json"
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image")

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "semantic"
    assert task.image.errors.get("qwen") == "malformed_model_output"


def test_image_semantic_filter_stage_malformed_score_only_keeps_image() -> None:
    """Malformed semantic-filter output with score_only should keep the image."""
    task = _make_task()
    task.image.filter_captions["qwen"] = "not valid json"
    stage = ImageSemanticFilterStage(model_variant="qwen", user_prompt="synthetic image", score_only=True)

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is False
    assert task.image.qwen_rejection_stage is None
    assert task.image.errors.get("qwen") == "malformed_model_output"


def test_image_classifier_stage_marks_malformed_model_output_as_error() -> None:
    """Malformed classifier output should filter the image and store the error."""
    task = _make_task()
    task.image.filter_captions["qwen"] = "not valid json"
    stage = ImageClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
    )

    result = stage.process_data([task])

    assert result is not None
    assert len(result) == 1
    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "classifier"
    assert task.image.errors.get("qwen") == "malformed_model_output"


def test_collect_caption_inputs_skips_filtered_images() -> None:
    """Already-filtered images must be excluded from filter-caption collection to avoid wasted GPU work."""
    filtered = _make_task()
    filtered.image.is_filtered = True
    filtered.image.model_input["qwen"] = {"data": "something"}

    passing = _make_task()
    passing.image.model_input["qwen"] = {"data": "something"}

    _, valid_indices = _collect_caption_inputs([filtered, passing], "qwen", result_target="filter_caption")

    assert valid_indices == [1]


def test_semantic_filter_and_classifier_accumulate_rejection_reasons() -> None:
    """Reasons from semantic filter (score_only) and classifier rejection must both appear in qwen_rejection_reasons."""
    task = _make_task()
    task.image.filter_captions["semantic:qwen"] = '{"synthetic_image": "yes"}'
    task.image.filter_captions["classifier:qwen"] = '{"planet_earth": "no", "space": "no"}'

    semantic_stage = ImageSemanticFilterStage(
        model_variant="qwen",
        filter_caption_key="semantic:qwen",
        user_prompt="synthetic image",
        score_only=True,
    )
    classifier_stage = ImageClassifierStage(
        model_variant="qwen",
        filter_caption_key="classifier:qwen",
        custom_categories=True,
        type_allow="planet_earth",
        type_block="space",
    )

    semantic_stage.process_data([task])
    classifier_stage.process_data([task])

    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "classifier"
    assert task.image.qwen_rejection_reasons is not None
    assert "synthetic image" in task.image.qwen_rejection_reasons
    assert "planet_earth" in task.image.qwen_rejection_reasons


def test_image_classifier_stage_allow_list_rejection_includes_reasons() -> None:
    """Allow-list rejection reasons must appear in qwen_rejection_reasons, not just qwen_rejection_stage."""
    task = _make_task()
    task.image.filter_captions["qwen"] = '{"planet_earth": "no", "space": "no"}'
    stage = ImageClassifierStage(
        model_variant="qwen",
        custom_categories=True,
        type_allow="planet_earth",
        type_block="space",
    )

    result = stage.process_data([task])

    assert result is not None
    assert task.image.is_filtered is True
    assert task.image.qwen_rejection_stage == "classifier"
    assert task.image.qwen_rejection_reasons is not None
    assert "planet_earth" in task.image.qwen_rejection_reasons
