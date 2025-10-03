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
"""Test vllm_caption_stage.py."""

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.vllm_model_ids import _VLLM_MODELS
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, VllmConfig, Window

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import (
        VllmModelInterface,
        _get_video_from_task,
        _get_windows_from_tasks,
    )

# Test UUIDs for deterministic testing
UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
UUID_3 = UUID("00000000-0000-0000-0000-000000000003")


@pytest.mark.env("unified")
def test_get_video_from_task_success() -> None:
    """Test get_video_from_task."""
    task = SplitPipeTask(video=Video(input_video=Path("test.mp4")))
    video = _get_video_from_task(task)
    assert video.input_video == Path("test.mp4")


@pytest.mark.env("unified")
def test_get_video_from_task_fail() -> None:
    """Test get_video_from_task."""
    task = 10
    with pytest.raises(TypeError, match=r".*"):
        _get_video_from_task(task)  # type: ignore[type-var]


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("config_variant", "raises"),
    [(k, nullcontext()) for k in _VLLM_MODELS] + [("_fail_model", pytest.raises(ValueError, match=r".*"))],
)
def test_vllm_model_interface_model_id_names(config_variant: str, raises: AbstractContextManager[Any]) -> None:
    """Validate model_id_names are strings for each configured plugin variant."""
    vllm_config = VllmConfig(model_variant=config_variant)
    vllm_model_interface = VllmModelInterface(vllm_config)

    with raises:
        model_id_names = vllm_model_interface.model_id_names
        for model_id_name in model_id_names:
            assert isinstance(model_id_name, str)


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("tasks", "expected_windows", "raises"),
    [
        # Empty tasks list
        ([], [], nullcontext()),
        # Single task with no clips
        ([SplitPipeTask(video=Video(input_video=Path("test.mp4")))], [], nullcontext()),
        # Single task with clip but no windows
        (
            [
                SplitPipeTask(
                    video=Video(
                        input_video=Path("test.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test.mp4",
                                span=(0.0, 1.0),
                            )
                        ],
                    )
                )
            ],
            [],
            nullcontext(),
        ),
        # Single task with clip and windows
        (
            [
                SplitPipeTask(
                    video=Video(
                        input_video=Path("test.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test.mp4",
                                span=(0.0, 1.0),
                                windows=[
                                    Window(
                                        start_frame=0,
                                        end_frame=10,
                                    ),
                                    Window(
                                        start_frame=10,
                                        end_frame=20,
                                    ),
                                ],
                            )
                        ],
                    )
                )
            ],
            [
                (Window(start_frame=0, end_frame=10), UUID_1),
                (Window(start_frame=10, end_frame=20), UUID_1),
            ],
            nullcontext(),
        ),
        # Multiple tasks with mixed scenarios
        (
            [
                SplitPipeTask(
                    video=Video(
                        input_video=Path("test1.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_1,
                                source_video="test1.mp4",
                                span=(0.0, 1.0),
                                windows=[
                                    Window(
                                        start_frame=0,
                                        end_frame=10,
                                    )
                                ],
                            )
                        ],
                    )
                ),
                SplitPipeTask(
                    video=Video(
                        input_video=Path("test2.mp4"),
                        clips=[
                            Clip(
                                uuid=UUID_2,
                                source_video="test2.mp4",
                                span=(0.0, 1.0),
                            ),  # No windows
                            Clip(
                                uuid=UUID_3,
                                source_video="test2.mp4",
                                span=(1.0, 2.0),
                                windows=[
                                    Window(
                                        start_frame=20,
                                        end_frame=30,
                                    )
                                ],
                            ),
                        ],
                    )
                ),
            ],
            [
                (Window(start_frame=0, end_frame=10), UUID_1),
                (Window(start_frame=20, end_frame=30), UUID_3),
            ],
            nullcontext(),
        ),
        # Task with invalid video type
        ([SplitPipeTask(video="invalid")], [], pytest.raises(TypeError, match=r".*")),
    ],
)
def test_get_windows_from_tasks(
    tasks: list[Any], expected_windows: list[tuple[Window, UUID]], raises: AbstractContextManager[Any]
) -> None:
    """Test _gather_windows_from_tasks function."""
    with raises:
        windows, clip_uuids = _get_windows_from_tasks(tasks)
        assert len(windows) == len(expected_windows)
        assert len(clip_uuids) == len(expected_windows)
        for (actual_window, actual_clip_uuid), (expected_window, expected_clip_uuid) in zip(
            zip(windows, clip_uuids, strict=True), expected_windows, strict=True
        ):
            assert actual_clip_uuid == str(expected_clip_uuid)
            assert actual_window.start_frame == expected_window.start_frame
            assert actual_window.end_frame == expected_window.end_frame
