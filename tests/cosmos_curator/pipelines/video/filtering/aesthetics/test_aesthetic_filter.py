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
"""Tests for aesthetic score filtering stages.

This test verifies the aesthetic scoring and filtering stages using a sample video.
The expected aesthetic score values were obtained by running the aesthetic filter pipeline
on the sample video (ForBiggerBlazes.mp4) and capturing the actual values produced.
These values serve as a regression test to ensure the aesthetic scoring algorithm
maintains consistency across code changes.
"""

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import torch

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage
from cosmos_curator.pipelines.video.filtering.aesthetics import aesthetic_filter_stages
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import (
    AestheticFilterStage,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature

EXPECTED_AESTHETIC_SCORE_MEAN: float = 4.8575
EXPECTED_AESTHETIC_SCORE_MIN: float = 3.7989
TOLERANCE: float = 0.002


def _frame_signature(fps: float) -> str:
    return FrameExtractionSignature(
        extraction_policy=FrameExtractionPolicy.sequence,
        target_fps=fps,
    ).to_str()


def _make_aesthetic_task(target_fps: list[float]) -> SplitPipeTask:
    extracted_frames = {_frame_signature(fps): np.full((2, 2, 2, 3), round(fps), dtype=np.uint8) for fps in target_fps}
    clip = Clip(
        uuid=UUID("12345678-1234-5678-1234-567812345678"),
        source_video="sample_video.mp4",
        span=(0.0, 2.0),
        encoded_data=np.ones(1, dtype=np.uint8),
        extracted_frames=LazyData(
            value=extracted_frames,
            nbytes=sum(frames.nbytes for frames in extracted_frames.values()),
        ),
    )
    return SplitPipeTask(
        session_id="test-session",
        video=Video(input_video=Path("sample_video.mp4"), clips=[clip]),
    )


def _set_aesthetic_score(
    monkeypatch: pytest.MonkeyPatch,
    stage: AestheticFilterStage,
    score: float,
) -> None:
    monkeypatch.setattr(stage, "_model", lambda _frames: torch.tensor([score]))


@pytest.mark.env("default")
def test_aesthetic_filter_setup() -> None:
    """Test that the aesthetic filter stage can be set up properly."""
    aesthetic_filter_stage = AestheticFilterStage(
        score_threshold=0.0,  # Set to 0 to ensure no filtering happens during score testing
        reduction="mean",
        log_stats=True,
    )
    # Set up the stage
    aesthetic_filter_stage.stage_setup()

    # Verify the model is set up
    assert aesthetic_filter_stage.model is not None

    # Clean up
    aesthetic_filter_stage.destroy()


@pytest.mark.env("default")
def test_aesthetic_score_calculation_mean(
    sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that aesthetic scores are calculated correctly with mean reduction.

    Args:
        sample_filtering_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = AestheticFilterStage(
        score_threshold=0.0,
        reduction="mean",
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip (since threshold is 0.0)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure aesthetic score attribute is present
    assert hasattr(clip, "aesthetic_score")

    assert clip.aesthetic_score == pytest.approx(EXPECTED_AESTHETIC_SCORE_MEAN, abs=TOLERANCE)

    # Verify stage performance stats were recorded
    assert "AestheticFilterStage" in result_task.stage_perf


@pytest.mark.env("default")
def test_aesthetic_score_calculation_min(
    sample_filtering_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test that aesthetic scores are calculated correctly with min reduction.

    Args:
        sample_filtering_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = AestheticFilterStage(
        score_threshold=0.0,
        reduction="min",
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip (since threshold is 0.0)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure aesthetic score attribute is present
    assert hasattr(clip, "aesthetic_score")

    assert clip.aesthetic_score == pytest.approx(EXPECTED_AESTHETIC_SCORE_MIN, abs=TOLERANCE)

    # Verify stage performance stats were recorded
    assert "AestheticFilterStage" in result_task.stage_perf


@pytest.mark.env("default")
@pytest.mark.parametrize(
    ("score_threshold", "should_be_filtered"),
    [
        # Threshold higher than expected score - clip should be filtered
        (9.0, True),
        # Threshold lower than expected score - clip should NOT be filtered
        (1.0, False),
    ],
)
def test_end_to_end_aesthetic_processing(
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
    score_threshold: float,
    *,
    should_be_filtered: bool,
) -> None:
    """Test the complete aesthetic processing pipeline end-to-end with different thresholds.

    This parameterized test verifies the filtering behavior with various thresholds:
    - When actual aesthetic scores are below the threshold, the clip should be filtered out
    - When actual aesthetic scores are above the threshold, the clip should be kept

    Args:
        sample_filtering_task: The sample task fixture
        sequential_runner: Runner for sequential test execution
        score_threshold: The aesthetic score threshold to test
        should_be_filtered: Whether the clip should be filtered given the threshold

    """
    stage = AestheticFilterStage(
        score_threshold=score_threshold,
        reduction="mean",
        target_fps=1.0,
        verbose=True,
        log_stats=True,
    )
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], [stage], runner=sequential_runner)

    # Verify the result
    video = result_tasks[0].video

    # Check that we have clips in either the main list or filtered list
    total_clips: int = len(video.clips) + len(video.filtered_clips)
    assert total_clips == 1  # We started with 1 clip

    if should_be_filtered:
        assert len(video.filtered_clips) == 1
        assert len(video.clips) == 0
    else:
        assert len(video.filtered_clips) == 0
        assert len(video.clips) == 1


def test_aesthetic_filter_consumes_frames_without_downstream_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default lifecycle should keep destructive cleanup for standalone aesthetics."""
    task = _make_aesthetic_task([1.0])
    clip = task.video.clips[0]
    stage = AestheticFilterStage(score_threshold=0.0)
    _set_aesthetic_score(monkeypatch, stage, 5.0)

    stage.process_data([task])

    assert task.video.clips == [clip]
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


def test_aesthetic_filter_preserves_passing_shared_frames_for_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing clips should retain an intentionally shared frame entry for embedding."""
    task = _make_aesthetic_task([1.0])
    clip = task.video.clips[0]
    original_frames = clip.extracted_frames.resolve()
    stage = AestheticFilterStage(score_threshold=0.0, preserve_extracted_frames=True)
    _set_aesthetic_score(monkeypatch, stage, 5.0)

    stage.process_data([task])

    assert task.video.clips == [clip]
    assert clip.extracted_frames.resolve() is original_frames
    assert set(original_frames or {}) == {_frame_signature(1.0)}


@pytest.mark.parametrize("embedding_fps", [2.0, 1.5], ids=["default-rate", "fractional-rate"])
def test_aesthetic_filter_consumes_own_entry_for_distinct_embedding_rate(
    embedding_fps: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct-signature ownership should remove only the aesthetics entry."""
    task = _make_aesthetic_task([1.0, embedding_fps])
    clip = task.video.clips[0]
    stage = AestheticFilterStage(score_threshold=0.0)
    _set_aesthetic_score(monkeypatch, stage, 5.0)

    stage.process_data([task])

    assert task.video.clips == [clip]
    assert set(clip.extracted_frames.resolve() or {}) == {_frame_signature(embedding_fps)}


@pytest.mark.parametrize(
    ("target_fps", "preserve_extracted_frames"),
    [([1.0, 2.0], False), ([1.0], True)],
    ids=["distinct-signatures", "shared-signature"],
)
def test_aesthetic_filter_drops_complete_frame_map_on_rejection(
    target_fps: list[float],
    *,
    preserve_extracted_frames: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected clips should release frames regardless of signature ownership."""
    task = _make_aesthetic_task(target_fps)
    clip = task.video.clips[0]
    stage = AestheticFilterStage(
        score_threshold=2.0,
        preserve_extracted_frames=preserve_extracted_frames,
    )
    _set_aesthetic_score(monkeypatch, stage, 1.0)

    stage.process_data([task])

    assert task.video.clips == []
    assert task.video.filtered_clips == [clip]
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


@pytest.mark.parametrize(("score_threshold", "is_filtered"), [(0.0, True), (-1.0, False)])
def test_aesthetic_filter_missing_signature_follows_threshold_semantics(
    score_threshold: float,
    *,
    is_filtered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing signature should release only clips rejected by the existing threshold rule."""
    task = _make_aesthetic_task([2.0])
    clip = task.video.clips[0]
    stage = AestheticFilterStage(score_threshold=score_threshold)
    messages: list[str] = []
    monkeypatch.setattr(aesthetic_filter_stages, "logger", SimpleNamespace(error=messages.append))

    stage.process_data([task])

    assert len(messages) == 1
    assert clip.aesthetic_score == -1.0
    if is_filtered:
        assert task.video.clips == []
        assert task.video.filtered_clips == [clip]
        assert clip.extracted_frames.resolve() is None
        assert clip.extracted_frames.nbytes == 0
    else:
        assert task.video.clips == [clip]
        assert task.video.filtered_clips == []
        assert set(clip.extracted_frames.resolve() or {}) == {_frame_signature(2.0)}


@pytest.mark.parametrize(("score_threshold", "is_filtered"), [(0.0, True), (-1.0, False)])
def test_aesthetic_filter_missing_encoded_data_follows_threshold_semantics(
    score_threshold: float,
    *,
    is_filtered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing encoded data should release only clips rejected by the existing threshold rule."""
    task = _make_aesthetic_task([1.0])
    clip = task.video.clips[0]
    clip.encoded_data.drop()
    stage = AestheticFilterStage(
        score_threshold=score_threshold,
        preserve_extracted_frames=True,
    )
    messages: list[str] = []
    monkeypatch.setattr(aesthetic_filter_stages, "logger", SimpleNamespace(warning=messages.append))

    stage.process_data([task])

    assert len(messages) == 1
    assert clip.errors["encoded_data"] == "empty"
    assert clip.aesthetic_score == -1.0
    if is_filtered:
        assert task.video.clips == []
        assert task.video.filtered_clips == [clip]
        assert clip.extracted_frames.resolve() is None
        assert clip.extracted_frames.nbytes == 0
    else:
        assert task.video.clips == [clip]
        assert task.video.filtered_clips == []
        assert set(clip.extracted_frames.resolve() or {}) == {_frame_signature(1.0)}


def test_aesthetic_and_embedding_consumers_share_one_fps_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passing clip should retain a shared entry until embedding consumes it."""
    task = _make_aesthetic_task([1.0])
    clip = task.video.clips[0]
    aesthetic_stage = AestheticFilterStage(score_threshold=0.0, preserve_extracted_frames=True)
    embedding_stage = OpenAIEmbeddingStage(model_name="test-model", target_fps=1.0)
    _set_aesthetic_score(monkeypatch, aesthetic_stage, 5.0)
    monkeypatch.setattr(
        embedding_stage,
        "_generate_embedding",
        lambda _frames: np.ones(4, dtype=np.float32),
    )

    aesthetic_stage.process_data([task])

    assert set(clip.extracted_frames.resolve() or {}) == {_frame_signature(1.0)}
    assert f"frames-{_frame_signature(1.0)}" not in clip.errors

    embedding_stage.process_data([task])

    assert np.array_equal(clip.openai_embedding, np.ones(4, dtype=np.float32))
    assert "openai_embedding" not in clip.errors
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0
