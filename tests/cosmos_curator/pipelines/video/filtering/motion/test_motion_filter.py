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
"""Functional test for CameraSensor motion-vector export and motion filtering stages.

Motion vectors are exported by the CameraSensor clip-extraction path (the sole source since the
legacy PyAV decode path was removed) and scored by ``MotionFilterStage`` using a sample video. The
expected motion score values were captured from that original path; asserting the CameraSensor path
reproduces them serves as both an equivalence anchor and a regression test to ensure the motion
detection algorithm maintains consistency across code changes.
"""

from uuid import uuid4

import numpy as np
import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.core.sensors.data.camera_data import MotionVectorData, MotionVectorFrameData
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import (
    CameraSensorMotionVectorConfig,
    ClipFrameExtractionStage,
)
from cosmos_curator.pipelines.video.filtering.motion.motion_builders import (
    MotionFilterConfig,
    build_motion_filter_stages,
)
from cosmos_curator.pipelines.video.filtering.motion.motion_filter_stages import (
    MotionFilterStage,
)
from cosmos_curator.pipelines.video.filtering.motion.motion_vector_backend import (
    sensor_motion_vector_data_to_legacy_matrices,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

# Golden values for motion scores
EXPECTED_MOTION_GLOBAL_MEAN: float = 0.002402
EXPECTED_MOTION_PER_PATCH_MIN_256: float = 0.001553
TOLERANCE: float = 0.000001


def test_sensor_motion_vector_data_to_legacy_matrices_projects_each_frame() -> None:
    """Sensor motion-vector adapter should return one legacy matrix per sensor frame."""
    frame = MotionVectorFrameData(
        source=np.array([-1], dtype=np.int32),
        w=np.array([16], dtype=np.int32),
        h=np.array([8], dtype=np.int32),
        src_x=np.array([1], dtype=np.int32),
        src_y=np.array([2], dtype=np.int32),
        dst_x=np.array([3], dtype=np.int32),
        dst_y=np.array([4], dtype=np.int32),
        flags=np.array([5], dtype=np.int64),
        motion_x=np.array([6], dtype=np.int32),
        motion_y=np.array([7], dtype=np.int32),
        motion_scale=np.array([8], dtype=np.int32),
    )
    motion_vectors = MotionVectorData(frames=(frame, MotionVectorFrameData.empty()))

    legacy_matrices = sensor_motion_vector_data_to_legacy_matrices(motion_vectors)

    assert len(legacy_matrices) == 2
    assert legacy_matrices[0].dtype == np.float64
    np.testing.assert_array_equal(
        legacy_matrices[0],
        np.array([[-1, 16, 8, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float64),
    )
    assert legacy_matrices[1].shape == (0, 11)


def test_motion_filter_builder_builds_only_filter_stage() -> None:
    """The builder should construct only the scoring/filter stage; motion vectors arrive upstream."""
    stages = build_motion_filter_stages(MotionFilterConfig())

    assert len(stages) == 1
    assert isinstance(stages[0], MotionFilterStage)


def test_motion_filter_drops_extracted_frames_from_filtered_clips() -> None:
    """Clips rejected after early frame extraction should not retain frame payloads."""
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    clip.extracted_frames.value = {"sequence-2000": frames}
    clip.extracted_frames.nbytes = frames.nbytes
    task = SplitPipeTask(session_id="session-a", video=Video(input_video="video.mp4", clips=[clip]))
    stage = MotionFilterStage(
        score_only=False,
        global_mean_threshold=0.00098,
        per_patch_min_256_threshold=0.000001,
    )

    result = stage.process_data([task])

    assert result == [task]
    assert task.video.clips == []
    assert task.video.filtered_clips == [clip]
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


@pytest.fixture
def camera_sensor_motion_stage() -> ClipFrameExtractionStage:
    """Fixture to create the CameraSensor clip-extraction stage in motion-export mode.

    This is the sole motion-vector source: it populates ``clip.decoded_motion_data`` from the
    camera-sensor decode, replacing the removed ``MotionVectorDecodeStage``.

    Returns:
        ClipFrameExtractionStage: Extraction stage configured to export motion vectors only.

    """
    return ClipFrameExtractionStage(
        target_fps=[],
        motion_vectors=CameraSensorMotionVectorConfig(target_fps=2.0, target_duration_ratio=0.5),
        log_stats=True,
    )


@pytest.fixture
def motion_filter_stage() -> MotionFilterStage:
    """Fixture to create a motion filter stage.

    Returns:
        MotionFilterStage: Configured instance of the filter stage

    """
    return MotionFilterStage(
        score_only=True,
        global_mean_threshold=-999,
        per_patch_min_256_threshold=-999,
        log_stats=True,
    )


@pytest.mark.env("default")
def test_camera_sensor_motion_extraction(
    camera_sensor_motion_stage: ClipFrameExtractionStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Test that motion vectors are exported from the sample video via the CameraSensor path.

    Args:
        camera_sensor_motion_stage: The CameraSensor extraction stage to test
        sample_filtering_task: Sample task with video data
        sequential_runner: Sequential runner fixture

    """
    result_tasks: list[SplitPipeTask] = run_pipeline(
        [sample_filtering_task], [camera_sensor_motion_stage], runner=sequential_runner
    )

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    video = result_task.video
    # Verify the video has one clip
    assert len(video.clips) == 1

    clip = video.clips[0]
    # Check that motion data was decoded
    assert clip.decoded_motion_data is not None
    # Check that we have frames
    assert len(clip.decoded_motion_data.frames) > 0

    # Verify stage performance stats were recorded
    assert "ClipFrameExtractionStage" in result_task.stage_perf


@pytest.mark.env("default")
def test_motion_filter_calculation(
    camera_sensor_motion_stage: ClipFrameExtractionStage,
    motion_filter_stage: MotionFilterStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Test that motion scores are calculated correctly and filtering works as expected.

    The golden scores were captured from the (removed) legacy PyAV decode path; asserting the
    CameraSensor path reproduces them is the durable equivalence anchor for that migration.

    Args:
        camera_sensor_motion_stage: The CameraSensor extraction stage to use
        motion_filter_stage: The filter stage to test
        sample_filtering_task: Sample task with video data
        sequential_runner: Sequential runner fixture

    """
    stages = [camera_sensor_motion_stage, motion_filter_stage]
    tasks_after_filter: list[SplitPipeTask] = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    # Verify there's one task returned
    assert len(tasks_after_filter) == 1

    result_task = tasks_after_filter[0]
    video = result_task.video
    # Verify the video has one clip (since we're using score_only=True)
    assert len(video.clips) == 1

    clip = video.clips[0]

    # Ensure motion score attributes are present
    assert hasattr(clip, "motion_score_global_mean")
    assert hasattr(clip, "motion_score_per_patch_min_256")

    # Check that the scores are within expected range
    assert clip.motion_score_global_mean == pytest.approx(EXPECTED_MOTION_GLOBAL_MEAN, abs=TOLERANCE)
    assert clip.motion_score_per_patch_min_256 == pytest.approx(EXPECTED_MOTION_PER_PATCH_MIN_256, abs=TOLERANCE)

    # Verify that decoded_motion_data was cleared to save memory
    assert clip.decoded_motion_data is None

    # Verify stage performance stats were recorded
    assert "MotionFilterStage" in result_task.stage_perf


@pytest.mark.env("default")
@pytest.mark.parametrize(
    ("global_threshold", "patch_threshold", "should_be_filtered"),
    [
        # Both thresholds higher than actual values - clip should be filtered
        (0.003, 0.002, True),
        # Global threshold higher, patch threshold lower - clip should be filtered
        (0.003, 0.001, True),
        # Global threshold lower, patch threshold higher - clip should be filtered
        (0.002, 0.002, True),
        # Both thresholds lower than actual values - clip should NOT be filtered
        (0.002, 0.001, False),
    ],
)
def test_end_to_end_motion_processing(  # noqa: PLR0913 - parametrized test with multiple fixtures
    camera_sensor_motion_stage: ClipFrameExtractionStage,
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
    global_threshold: float,
    patch_threshold: float,
    *,
    should_be_filtered: bool,
) -> None:
    """Test the complete motion processing pipeline end-to-end with different thresholds.

    This parameterized test verifies the filtering behavior with various threshold combinations:
    - When actual motion values are below the thresholds, the clip should be filtered out
    - When actual motion values are above the thresholds, the clip should be kept

    Args:
        camera_sensor_motion_stage: The CameraSensor extraction stage fixture
        sample_filtering_task: The sample task fixture
        sequential_runner: Sequential runner fixture
        global_threshold: The global mean threshold to test
        patch_threshold: The per-patch min 256 threshold to test
        should_be_filtered: Whether the clip should be filtered given the thresholds

    """
    # Instantiate a fresh filter stage with the desired thresholds
    stages = [
        camera_sensor_motion_stage,
        MotionFilterStage(
            score_only=False,
            global_mean_threshold=global_threshold,
            per_patch_min_256_threshold=patch_threshold,
            log_stats=True,
        ),
    ]

    # Run through extraction and filter stages
    tasks: list[SplitPipeTask] = run_pipeline([sample_filtering_task], stages, runner=sequential_runner)

    # Verify the result
    result_task = tasks[0]
    video = result_task.video

    # Check that we have clips in either the main list or filtered list
    total_clips: int = len(video.clips) + len(video.filtered_clips)
    assert total_clips == 1  # We started with 1 clip

    if should_be_filtered:
        assert len(video.filtered_clips) == 1
        assert len(video.clips) == 0
    else:
        assert len(video.filtered_clips) == 0
        assert len(video.clips) == 1
