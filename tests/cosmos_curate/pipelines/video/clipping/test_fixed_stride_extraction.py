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
"""Functional test for fixed stride clip extraction stage.

This test verifies the fixed stride clip extraction stage using a sample video.
The expected results were obtained by running the fixed stride extraction pipeline
on the sample video and capturing the actual values produced.
These values serve as a regression test to ensure the clip extraction algorithm
maintains consistency across code changes.
"""

import copy
import pathlib

import pytest

from cosmos_curate.core.interfaces.pipeline_interface import PipelineExecutionError, run_pipeline
from cosmos_curate.core.interfaces.runner_interface import RunnerInterface
from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video


def test_fixed_stride_extractor_setup() -> None:
    """Test that the fixed stride extractor stage can be set up properly."""
    # Default parameter values for FixedStrideExtractorStage
    default_clip_length = 10
    default_clip_stride = 10
    default_min_clip_length = 10
    default_limit_clips = 0
    default_cpu_count = 1.0
    default_gpu_count = 0

    stage = FixedStrideExtractorStage()

    # Verify basic properties
    assert stage.clip_len_s == default_clip_length
    assert stage.clip_stride_s == default_clip_stride
    assert stage.min_clip_length_s == default_min_clip_length
    assert stage._limit_clips == default_limit_clips
    assert stage._verbose is False
    assert stage._log_stats is False

    # Verify resource requirements (CPU-only stage)
    assert stage.resources.gpus == default_gpu_count
    assert stage.resources.cpus == default_cpu_count  # Default to 1 CPU per worker


def test_fixed_stride_extractor_custom_parameters() -> None:
    """Test that the fixed stride extractor stage accepts custom parameters."""
    # Custom parameter values for testing
    custom_clip_length = 5.0
    custom_clip_stride = 3.0
    custom_min_clip_length = 2.0
    custom_limit_clips = 10

    stage = FixedStrideExtractorStage(
        clip_len_s=custom_clip_length,
        clip_stride_s=custom_clip_stride,
        min_clip_length_s=custom_min_clip_length,
        limit_clips=custom_limit_clips,
        verbose=True,
        log_stats=True,
    )

    # Verify custom parameters are set
    assert stage.clip_len_s == custom_clip_length
    assert stage.clip_stride_s == custom_clip_stride
    assert stage.min_clip_length_s == custom_min_clip_length
    assert stage._limit_clips == custom_limit_clips
    assert stage._verbose is True
    assert stage._log_stats is True


@pytest.mark.env("cosmos-curate")
def test_fixed_stride_extraction_default_parameters(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with default parameters.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Expected clip extraction results for regression testing
    # These values are based on the sample video metadata:
    # - Duration: ~47 seconds (WeAreGoingOnBullrun.mp4)
    # - With default parameters (10s clips, 10s stride, min_clip_length_s=10),
    # - we expect clips: 0-10s, 10-20s, 20-30s, 30-40s (last clip 40-47s is filtered out as it's < 10s)
    expected_clips_default = 4
    expected_clip_spans_default = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]

    stage = FixedStrideExtractorStage(log_stats=True)

    # Process the task
    result_tasks: list[SplitPipeTask] = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)

    # Verify there's one task returned
    assert len(result_tasks) == 1

    result_task = result_tasks[0]
    # Verify the task has one video
    video = result_task.video

    # Verify clips were extracted
    assert hasattr(video, "clips")
    assert len(video.clips) == expected_clips_default

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_default):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)

    # Verify stage performance stats were recorded
    assert "FixedStrideExtractorStage" in result_task.stage_perf


@pytest.mark.env("cosmos-curate")
def test_fixed_stride_extraction_5s_stride(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with 5-second clips and stride.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 5s clips and 5s stride, we expect clips every 5 seconds: 0-5s, 5-10s, ..., 40-45s
    # (last clip 45-47s is filtered out as it's < 5s)
    expected_clips_5s_stride = 9
    expected_clip_spans_5s_stride = [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
        (30.0, 35.0),
        (35.0, 40.0),
        (40.0, 45.0),
    ]

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Verify clips were extracted
    assert len(video.clips) == expected_clips_5s_stride

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_5s_stride):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curate")
def test_fixed_stride_extraction_overlapping_clips(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with overlapping clips (stride < clip_len).

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 3s clips and 2s stride, we expect many overlapping clips
    expected_clips_3s_2s_stride = 23
    expected_clip_spans_3s_2s_stride = [
        (0.0, 3.0),
        (2.0, 5.0),
        (4.0, 7.0),
        (6.0, 9.0),
        (8.0, 11.0),
        (10.0, 13.0),
        (12.0, 15.0),
        (14.0, 17.0),
        (16.0, 19.0),
        (18.0, 21.0),
        (20.0, 23.0),
        (22.0, 25.0),
        (24.0, 27.0),
        (26.0, 29.0),
        (28.0, 31.0),
        (30.0, 33.0),
        (32.0, 35.0),
        (34.0, 37.0),
        (36.0, 39.0),
        (38.0, 41.0),
        (40.0, 43.0),
        (42.0, 45.0),
        (44.0, 47.0),
    ]

    stage = FixedStrideExtractorStage(
        clip_len_s=3.0,
        clip_stride_s=2.0,
        min_clip_length_s=3.0,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Verify clips were extracted
    assert len(video.clips) == expected_clips_3s_2s_stride

    # Verify clip spans match expected values
    for i, expected_span in enumerate(expected_clip_spans_3s_2s_stride):
        if i < len(video.clips):
            clip_span = video.clips[i].span
            assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
            assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curate")
def test_fixed_stride_extraction_with_limit(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with clip limit.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # With 3s clips and 2s stride, we expect many overlapping clips, only list the first 2
    expected_clip_spans_3s_2s_stride = [
        (0.0, 3.0),
        (2.0, 5.0),
    ]

    limit = 2
    stage = FixedStrideExtractorStage(
        clip_len_s=3.0,
        clip_stride_s=2.0,
        min_clip_length_s=3.0,
        limit_clips=limit,
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Verify only the limited number of clips were extracted
    assert len(video.clips) == limit

    # Verify the first clips match expected values
    expected_limited_spans = expected_clip_spans_3s_2s_stride
    for i, expected_span in enumerate(expected_limited_spans):
        clip_span = video.clips[i].span
        assert clip_span[0] == pytest.approx(expected_span[0], abs=0.01)
        assert clip_span[1] == pytest.approx(expected_span[1], abs=0.01)


@pytest.mark.env("cosmos-curate")
def test_fixed_stride_extraction_min_clip_length(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface
) -> None:
    """Test fixed stride extraction with minimum clip length filtering.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Should extract clips: 0-2s, 10-12s, 20-22s, 30-32s, 40-42s (5 clips total for 47s video)
    expected_clips_with_large_stride = 5

    stage = FixedStrideExtractorStage(
        clip_len_s=2.0,
        clip_stride_s=10.0,  # Large stride to get clips at end of video
        min_clip_length_s=1.5,  # Allow shorter clips
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Should extract clips: 0-2s, 10-12s, 20-22s, 30-32s, 40-42s (5 clips total for 47s video)
    assert len(video.clips) == expected_clips_with_large_stride

    # Verify all clips meet minimum length requirement
    for clip in video.clips:
        clip_duration = clip.span[1] - clip.span[0]
        assert clip_duration >= stage.min_clip_length_s - 0.01  # Small tolerance


def test_fixed_stride_extraction_no_clips_short_video(sequential_runner: RunnerInterface) -> None:
    """Test fixed stride extraction with a video shorter than minimum clip length."""
    # Create a task with a very short video duration
    video = Video(
        input_video=pathlib.Path("short_video.mp4"),
        encoded_data=b"dummy_bytes",  # We'll mock the metadata
    )

    # Mock metadata for a very short video
    video.metadata = type(
        "MockMetadata",
        (),
        {
            "num_frames": 30,
            "framerate": 30.0,
            "duration": 1.0,  # 1 second video
            "bit_rate_k": 1000,
            "height": 720,
            "width": 1280,
            "video_codec": "h264",  # Added required attribute
            "pixel_format": "yuv420p",  # Added for potential is_10_bit_color() check
            "audio_codec": None,  # Added for completeness
            "size": None,  # Added for completeness
        },
    )()

    task = SplitPipeTask(video=video, stage_perf={})

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        min_clip_length_s=5.0,  # Longer than video duration
        log_stats=True,
    )

    # Process the task
    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Should extract no clips since video is too short
    assert len(video.clips) == 0


def test_error_handling_no_encoded_data(sequential_runner: RunnerInterface) -> None:
    """Test error handling when video has no source bytes."""
    video = Video(
        input_video=pathlib.Path("no_bytes_video.mp4"),
        encoded_data=None,  # No encoded_data
    )
    task = SplitPipeTask(video=video, stage_perf={})

    stage = FixedStrideExtractorStage(log_stats=True)

    # Should raise PipelineExecutionError (wrapping ValueError) for missing encoded_data
    with pytest.raises(PipelineExecutionError, match="Please load video bytes!"):
        run_pipeline([task], [stage], runner=sequential_runner)


def test_error_handling_incomplete_metadata(sequential_runner: RunnerInterface) -> None:
    """Test error handling when video has incomplete metadata."""
    video = Video(
        input_video=pathlib.Path("incomplete_metadata_video.mp4"),
        encoded_data=b"dummy_bytes",
    )

    # Don't extract metadata, leaving it incomplete
    task = SplitPipeTask(video=video, stage_perf={})

    stage = FixedStrideExtractorStage(log_stats=True)

    # Process the task
    result_tasks = run_pipeline([task], [stage], runner=sequential_runner)
    result_task = result_tasks[0]
    video = result_task.video

    # Should record metadata error and skip processing
    assert "metadata" in video.errors
    assert video.errors["metadata"] == "incomplete"
    assert len(video.clips) == 0


@pytest.mark.env("cosmos-curate")
def test_clip_uuid_generation(sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test that clip UUIDs are generated consistently and uniquely.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    # Process the task twice with separate copies to avoid accumulating clips
    task_copy_1 = copy.deepcopy(sample_splitting_task)
    task_copy_2 = copy.deepcopy(sample_splitting_task)

    result_tasks_1 = run_pipeline([task_copy_1], [stage], runner=sequential_runner)
    result_tasks_2 = run_pipeline([task_copy_2], [stage], runner=sequential_runner)

    video_1 = result_tasks_1[0].video
    video_2 = result_tasks_2[0].video

    # UUIDs should be consistent across runs
    assert len(video_1.clips) == len(video_2.clips)
    for clip_1, clip_2 in zip(video_1.clips, video_2.clips, strict=False):
        assert clip_1.uuid == clip_2.uuid

    # UUIDs should be unique within a video
    uuids = [clip.uuid for clip in video_1.clips]
    assert len(uuids) == len(set(uuids))  # All UUIDs are unique


@pytest.mark.env("cosmos-curate")
def test_clip_properties(sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface) -> None:
    """Test that extracted clips have correct properties.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution

    """
    # Span tuple contains start and end times
    expected_span_tuple_length = 2

    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        min_clip_length_s=5.0,
    )

    # Process the task
    result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)
    video = result_tasks[0].video

    # Verify clip properties
    for clip in video.clips:
        # Check that clip has required attributes
        assert hasattr(clip, "uuid")
        assert hasattr(clip, "encoded_data")
        assert hasattr(clip, "span")

        # Check that source_video matches input
        assert clip.source_video == str(video.input_video)

        # Check that span is valid
        assert len(clip.span) == expected_span_tuple_length
        assert clip.span[0] >= 0
        assert clip.span[1] > clip.span[0]

        # Check that clip duration matches expected length (within tolerance)
        clip_duration = clip.span[1] - clip.span[0]
        assert clip_duration <= stage.clip_len_s + 0.01  # Small tolerance


@pytest.mark.env("cosmos-curate")
def test_verbose_logging(
    sample_splitting_task: SplitPipeTask, sequential_runner: RunnerInterface, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that verbose logging works correctly.

    Args:
        sample_splitting_task: Sample task with video data
        sequential_runner: Runner for sequential test execution
        caplog: Pytest fixture for capturing log output

    """
    stage = FixedStrideExtractorStage(
        clip_len_s=5.0,
        clip_stride_s=5.0,
        verbose=True,
        log_stats=True,
    )

    # Process the task
    with caplog.at_level("INFO"):
        result_tasks = run_pipeline([sample_splitting_task], [stage], runner=sequential_runner)

    # Verify that some logging occurred (exact messages may vary)
    # The stage itself doesn't have explicit verbose logging, but this tests the parameter
    assert len(result_tasks) == 1
