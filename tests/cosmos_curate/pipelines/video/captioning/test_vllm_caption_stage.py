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
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.models.vllm_model_ids import _VLLM_MODELS
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VllmConfig,
    Window,
    WindowConfig,
)

if conda_utils.is_running_in_env("unified"):
    import torch

    from cosmos_curate.models.vllm_interface import _VLLM_PLUGINS
    from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import (
        VllmModelInterface,
        VllmPrepStage,
        _free_vllm_inputs,
        _get_stage2_prompts,
        _get_windows_from_tasks,
        _scatter_captions,
    )
    from cosmos_curate.pipelines.video.utils.data_model import get_video_from_task

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:
    VALID_VARIANTS = []


# Test UUIDs for deterministic testing
UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
UUID_3 = UUID("00000000-0000-0000-0000-000000000003")


@pytest.mark.env("unified")
def test_get_video_from_task_success() -> None:
    """Test get_video_from_task."""
    task = SplitPipeTask(video=Video(input_video=Path("test.mp4")))
    video = get_video_from_task(task)
    assert video.input_video == Path("test.mp4")


@pytest.mark.env("unified")
def test_get_video_from_task_fail() -> None:
    """Test get_video_from_task."""
    task = 10
    with pytest.raises(TypeError, match=r".*"):
        get_video_from_task(task)  # type: ignore[type-var]


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
    ],
)
def test_get_windows_from_tasks(
    tasks: list[Any], expected_windows: list[tuple[Window, UUID]], raises: AbstractContextManager[Any]
) -> None:
    """Test _get_windows_from_tasks function."""
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


@pytest.mark.env("unified")
@pytest.mark.parametrize("keep_mp4", [False, True])
def test_free_vllm_inputs_clears_inputs_and_optionally_mp4(*, keep_mp4: bool) -> None:
    """Validate model inputs are removed and mp4_bytes handled per flag."""
    model_variant = "test_variant"
    other_variant = "other_variant"
    w1 = Window(
        start_frame=0,
        end_frame=10,
        mp4_bytes=b"a",
        model_input={model_variant: {"x": 1}, other_variant: {"z": 3}},
    )
    w2 = Window(
        start_frame=10,
        end_frame=20,
        mp4_bytes=b"b",
        model_input={model_variant: {"y": 2}, other_variant: {"k": 4}},
    )

    original_bytes = [w1.mp4_bytes, w2.mp4_bytes]
    _free_vllm_inputs([w1, w2], model_variant, keep_mp4=keep_mp4)

    for idx, w in enumerate([w1, w2]):
        assert model_variant not in w.model_input
        # Ensure other variant remains untouched
        assert other_variant in w.model_input
        assert set(w.model_input.keys()) == {other_variant}
        if keep_mp4:
            assert w.mp4_bytes == original_bytes[idx]
        else:
            assert w.mp4_bytes is None


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("windows_count", "frames_count", "raises"),
    [
        (2, 2, nullcontext()),
        (1, 2, pytest.raises(ValueError, match=r".*")),
        (3, 1, pytest.raises(ValueError, match=r".*")),
    ],
)
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curate.pipelines.video.captioning.vllm_caption_stage.windowing_utils.make_windows_for_video")
@patch("cosmos_curate.models.vllm_interface.make_model_inputs")
def test_prep_windows_model_input_assignment(  # noqa: PLR0913
    mock_make_model_inputs: MagicMock,
    mock_make_windows: MagicMock,
    model_variant: str,
    windows_count: int,
    frames_count: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Validate VllmPrepStage._prep_windows assigns inputs and enforces strict zipping."""
    config = VllmConfig(model_variant=model_variant)
    window_config = WindowConfig()
    stage = VllmPrepStage(config, window_config, keep_mp4=False)
    # Inject a fake processor since stage_setup isn't called here
    stage._processor = MagicMock()  # type: ignore[attr-defined]

    prompt = "test prompt"
    video = Video(input_video=Path("test.mp4"))

    # Create test data returned by windowing util
    windows = [Window(start_frame=0, end_frame=frames_count) for _ in range(windows_count)]
    frames = [torch.randn(frames_count, 3, 224, 224) for _ in range(frames_count)]
    mock_make_windows.return_value = (windows, frames)
    mock_make_model_inputs.return_value = [{"test": "data"} for _ in range(frames_count)]

    with raises:
        stage._prep_windows(video, prompt)

        for window in windows:
            llm_input = window.model_input.get(model_variant)
            assert isinstance(llm_input, dict)


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curate.pipelines.video.captioning.vllm_caption_stage.windowing_utils.make_windows_for_video")
def test_prep_windows_raises_without_processor(mock_make_windows: MagicMock, model_variant: str) -> None:
    """_prep_windows should raise RuntimeError if self._processor is not set."""
    config = VllmConfig(model_variant=model_variant)
    stage = VllmPrepStage(config, WindowConfig(), keep_mp4=False)
    # Do NOT set stage._processor here

    video = Video(input_video=Path("test.mp4"))
    windows = [Window(start_frame=0, end_frame=10)]
    frames: list[object] = [object()]
    mock_make_windows.return_value = (windows, frames)

    with pytest.raises(RuntimeError, match=r".*processor.*"):
        stage._prep_windows(video, "prompt")


@pytest.mark.env("unified")
@pytest.mark.parametrize(
    ("stage2_prompt_text", "stage2_caption"),
    [
        ("test prompt", True),
        (None, False),
        (None, True),
    ],
)
def test_get_stage2_prompts(stage2_prompt_text: str | None, *, stage2_caption: bool) -> None:
    """Test _get_stage2_prompts."""
    vllm_config = VllmConfig(
        model_variant="test_variant", stage2_caption=stage2_caption, stage2_prompt_text=stage2_prompt_text
    )
    num_windows = 10
    stage2_prompts = _get_stage2_prompts(vllm_config, num_windows=num_windows)

    assert len(stage2_prompts) == num_windows

    if stage2_caption:
        for prompt in stage2_prompts:
            if stage2_prompt_text is None:
                assert isinstance(prompt, str)
            else:
                assert prompt == stage2_prompt_text
    else:
        for prompt in stage2_prompts:
            assert prompt is None


@pytest.mark.env("unified")
@pytest.mark.parametrize("verbose", [True, False])
def test_scatter_captions(*, verbose: bool) -> None:
    """Test _scatter_captions."""
    windows = [Window(start_frame=0, end_frame=10), Window(start_frame=10, end_frame=20)]
    captions = ["caption 1", "caption 2"]
    clip_uuids = ["clip_uuid_1", "clip_uuid_2"]
    model_variant = "test_variant"
    _scatter_captions(windows, captions, clip_uuids, model_variant, verbose=verbose)
    for window, caption, _ in zip(windows, captions, clip_uuids, strict=True):
        assert window.caption[model_variant] == caption
