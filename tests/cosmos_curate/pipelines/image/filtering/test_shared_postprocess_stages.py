# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared image semantic filter/classifier postprocess stages."""

import pathlib

import numpy as np

from cosmos_curate.core.utils.data.lazy_data import LazyData
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
