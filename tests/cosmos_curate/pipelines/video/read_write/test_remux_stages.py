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
"""Test remuxing stages for video pipelines."""

# ruff: noqa: ARG002 ok to pass unused arguments for testing, mocks may not be used, but the args are needed

import io
from fractions import Fraction
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import av
import numpy as np
import pytest

from cosmos_curate.pipelines.video.read_write.remux_stages import (
    RemuxStage,
    remux_if_needed,
    remux_to_mp4,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video


def _make_synthetic_video(format_name: str) -> io.BytesIO:
    """Create a synthetic video in the specified format.

    Args:
        format_name: Container format (e.g., "mp4", "mpegts", "avi")

    Returns:
        BytesIO buffer containing the synthetic video

    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format=format_name)
    fps = 30

    # Set up video stream
    if format_name == "mp4":
        stream = container.add_stream(
            "h264",
            rate=fps,
            options={
                "crf": "23",  # Good quality
                "preset": "fast",  # Fast encoding
            },
        )
    else:
        stream = container.add_stream("h264", rate=fps)

    stream.width = 4
    stream.height = 4
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, fps)

    # Create 5 frames
    for i in range(5):
        array = np.full((stream.height, stream.width, 3), i * 50, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    buffer.seek(0)
    return buffer


@pytest.fixture
def synthetic_mp4_video() -> io.BytesIO:
    """Create a synthetic MP4 video in memory."""
    return _make_synthetic_video("mp4")


@pytest.fixture
def synthetic_mpegts_video() -> io.BytesIO:
    """Create a synthetic MPEG-TS video in memory."""
    return _make_synthetic_video("mpegts")


@pytest.fixture
def synthetic_avi_video() -> io.BytesIO:
    """Create a synthetic AVI video in memory (unsupported format for remuxing)."""
    return _make_synthetic_video("avi")


@pytest.fixture
def synthetic_mkv_video() -> io.BytesIO:
    """Create a synthetic MKV video in memory."""
    return _make_synthetic_video("matroska")


@pytest.mark.env("cosmos-curate")
def test_remux_bad_source_bytes() -> None:
    """Test that remuxing fails if the source bytes are not set."""
    with pytest.raises(RuntimeError):
        remux_to_mp4(b"", threads=1)


@pytest.mark.env("cosmos-curate")
def test_remux_if_needed_mp4_no_change(synthetic_mp4_video: io.BytesIO) -> None:
    """Test that MP4 videos are not remuxed."""
    # Arrange
    original_bytes = synthetic_mp4_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.mp4")
    video.populate_metadata()

    # Act
    remux_if_needed(video, threads=1)

    # Assert
    assert video.source_bytes == original_bytes  # Should be unchanged


@pytest.mark.env("cosmos-curate")
def test_remux_if_needed_mpegts_to_mp4(synthetic_mpegts_video: io.BytesIO) -> None:
    """Test that MPEG-TS videos are remuxed to MP4."""
    # Arrange
    original_bytes = synthetic_mpegts_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.ts")
    video.populate_metadata()

    # Verify it starts as MPEG-TS
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert video.metadata.format_name == "mpegts"

    # Act
    remux_if_needed(video, threads=1)

    # Assert
    assert video.source_bytes != original_bytes  # Should be changed
    assert video.source_bytes is not None

    # Verify the result is valid MP4
    result_stream = io.BytesIO(video.source_bytes)
    with av.open(result_stream) as container:
        assert "mp4" in video.metadata.format_name.lower()
        assert "mp4" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curate")
def test_remux_if_needed_avi_to_mp4(synthetic_avi_video: io.BytesIO) -> None:
    """Test that AVI videos are unchanged."""
    # Arrange
    original_bytes = synthetic_avi_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.avi")
    video.populate_metadata()

    # Verify it starts as AVI
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert "avi" in video.metadata.format_name.lower()

    # Act
    remux_if_needed(video, threads=1)

    # Assert
    assert video.source_bytes == original_bytes  # Should be changed
    assert video.source_bytes is not None

    # Verify the result remains AVI (unchanged)
    result_stream = io.BytesIO(video.source_bytes)
    with av.open(result_stream) as container:
        assert "avi" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curate")
def test_remux_if_needed_mkv_to_mp4(synthetic_mkv_video: io.BytesIO) -> None:
    """Test that MKV videos are unchanged."""
    # Arrange
    original_bytes = synthetic_mkv_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.mkv")
    video.populate_metadata()

    # Verify it starts as MKV
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert "matroska,webm" in video.metadata.format_name.lower()

    # Act
    remux_if_needed(video, threads=1)

    # Assert
    assert video.source_bytes == original_bytes  # Should be unchanged
    assert video.source_bytes is not None

    # Verify the result remains MKV (unchanged)
    result_stream = io.BytesIO(video.source_bytes)
    with av.open(result_stream) as container:
        assert "matroska,webm" in container.format.name.lower()
        assert len(container.streams.video) > 0


@pytest.mark.env("cosmos-curate")
def test_remux_if_needed_preserves_video_content(synthetic_mpegts_video: io.BytesIO) -> None:
    """Test that remuxing preserves the actual video content."""

    def _frame_count(data: bytes) -> int:
        with av.open(io.BytesIO(data)) as container:
            in_container = cast("av.container.InputContainer", container)
            return sum(1 for _ in in_container.decode(video=0))
        msg = "Shouldn't get here, adding to make mypy happy"
        raise RuntimeError(msg)

    # Arrange
    original_bytes = synthetic_mpegts_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.ts")
    video.populate_metadata()

    # Get original frame count
    original_stream = io.BytesIO(original_bytes)
    with av.open(original_stream) as container:
        original_width = container.streams.video[0].width
        original_height = container.streams.video[0].height

    original_frame_count = _frame_count(original_bytes)

    # Act
    remux_if_needed(video, threads=1)

    # Assert
    assert video.source_bytes is not None

    result_frame_count = _frame_count(video.source_bytes)
    assert result_frame_count == original_frame_count

    # Verify the result is valid MP4
    result_stream = io.BytesIO(video.source_bytes)
    with av.open(result_stream) as container:
        result_width = container.streams.video[0].width
        result_height = container.streams.video[0].height

        # Video properties should be preserved
        assert result_frame_count == original_frame_count
        assert result_width == original_width
        assert result_height == original_height


@pytest.mark.env("cosmos-curate")
def test_remux_stage_integration(synthetic_mpegts_video: io.BytesIO) -> None:
    """Test RemuxStage end-to-end with real video data."""
    # Arrange
    stage = RemuxStage()

    original_bytes = synthetic_mpegts_video.getvalue()
    video = Video(source_bytes=original_bytes, input_video="test.ts")
    video.populate_metadata()

    task = Mock(spec=SplitPipeTask)
    task.video = video
    task.get_major_size.return_value = len(original_bytes)
    task.stage_perf = {}

    tasks = [task]

    # Verify starts as non-MP4
    assert video.metadata is not None
    assert video.metadata.format_name is not None
    assert "mp4" not in video.metadata.format_name.lower()

    # Act
    result = stage.process_data(tasks)

    # Assert
    assert result == tasks
    assert video.source_bytes != original_bytes  # Should be remuxed
    assert video.source_bytes is not None

    # Verify result is MP4
    result_stream = io.BytesIO(video.source_bytes)
    with av.open(result_stream) as container:
        assert "mp4" in container.format.name.lower()


class TestRemuxStage:
    """Test the RemuxStage class."""

    @pytest.mark.env("cosmos-curate")
    def test_init(self) -> None:
        """Test RemuxStage initialization."""
        # Test that RemuxStage can be initialized with default parameters
        stage = RemuxStage()
        assert stage is not None

        # Test that RemuxStage can be initialized with custom parameters
        stage = RemuxStage(verbose=True, log_stats=True)
        assert stage is not None

    @pytest.mark.env("cosmos-curate")
    def test_resources(self) -> None:
        """Test that RemuxStage returns correct resource requirements."""
        stage = RemuxStage()
        resources = stage.resources
        assert resources.cpus == 1

    @pytest.mark.env("cosmos-curate")
    @patch("cosmos_curate.pipelines.video.read_write.remux_stages.remux_if_needed")
    def test_process_data_success(self, mock_remux_if_needed: MagicMock) -> None:
        """Test successful processing of tasks."""
        # Arrange
        stage = RemuxStage()

        # Create mock tasks
        task1 = Mock(spec=SplitPipeTask)
        task1.video = Mock(spec=Video)
        task1.get_major_size.return_value = 1000
        task1.stage_perf = {}

        task2 = Mock(spec=SplitPipeTask)
        task2.video = Mock(spec=Video)
        task2.get_major_size.return_value = 2000
        task2.stage_perf = {}

        tasks = [task1, task2]

        # Act
        result = stage.process_data(tasks)

        # Assert
        assert result == tasks

    @pytest.mark.env("cosmos-curate")
    @patch("cosmos_curate.pipelines.video.read_write.remux_stages.remux_if_needed")
    def test_process_data_with_logging(self, mock_remux_if_needed: MagicMock) -> None:
        """Test processing with performance logging enabled."""
        # Arrange
        stage = RemuxStage(log_stats=True)

        # Create mock task
        task = Mock(spec=SplitPipeTask)
        task.video = Mock(spec=Video)
        task.get_major_size.return_value = 1000
        task.stage_perf = {}

        tasks = [task]

        # Act
        result = stage.process_data(tasks)

        # Assert
        assert result == tasks
        # Verify that stage_perf was populated (timer should add entries)
        assert len(task.stage_perf) > 0

    @pytest.mark.env("cosmos-curate")
    @patch("cosmos_curate.pipelines.video.read_write.remux_stages.remux_if_needed")
    def test_process_data_empty_list(self, mock_remux_if_needed: MagicMock) -> None:
        """Test processing with empty task list."""
        # Arrange
        stage = RemuxStage()
        tasks: list[SplitPipeTask] = []

        # Act
        result = stage.process_data(tasks)

        # Assert
        assert result == []
        mock_remux_if_needed.assert_not_called()

    @pytest.mark.env("cosmos-curate")
    @patch("cosmos_curate.pipelines.video.read_write.remux_stages.remux_to_mp4")
    def test_process_data_error_handling(self, mock_remux_to_mp4: MagicMock) -> None:
        """Test that remux errors are handled gracefully and don't stop pipeline."""
        # Arrange
        stage = RemuxStage()

        # Create mock tasks - one that will fail, one that will succeed
        failing_task = Mock(spec=SplitPipeTask)
        failing_task.video = Mock(spec=Video)
        failing_task.video.input_video = "failing_video.ts"
        failing_task.video.source_bytes = b"failing_video_data"
        failing_task.video.metadata = Mock()
        failing_task.video.metadata.format_name = "mpegts"  # Needs remuxing
        failing_task.video.errors = {}  # Add errors dict to mock
        failing_task.get_major_size.return_value = 1000
        failing_task.stage_perf = {}

        succeeding_task = Mock(spec=SplitPipeTask)
        succeeding_task.video = Mock(spec=Video)
        succeeding_task.video.input_video = "good_video.mp4"
        succeeding_task.video.source_bytes = b"good_video_data"
        succeeding_task.video.metadata = Mock()
        succeeding_task.video.metadata.format_name = "mov,mp4,m4a,3gp,3g2,mj2"  # Already MP4
        succeeding_task.video.errors = {}  # Add errors dict to mock
        succeeding_task.get_major_size.return_value = 2000
        succeeding_task.stage_perf = {}

        tasks = [failing_task, succeeding_task]

        # Mock remux_to_mp4 to fail for the first video
        mock_remux_to_mp4.side_effect = RuntimeError("ffmpeg failed")

        # Act
        result = stage.process_data(tasks)

        # Assert
        assert result == tasks  # Both tasks should be returned

        # Failing task should have error set in video.errors
        assert "remux" in failing_task.video.errors
        assert isinstance(failing_task.video.errors["remux"], str)

        # Succeeding task should not be affected
        assert succeeding_task.video.errors == {}

        # remux_to_mp4 should only be called for the failing task (the succeeding one is already MP4)
        mock_remux_to_mp4.assert_called_once_with(b"failing_video_data", threads=1)
