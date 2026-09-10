# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for extracted-frame cleanup by embedding consumers."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.video.embedding import (
    cosmos_embed1_stages,
    internvideo2_stages,
    openai_embedding_stage,
)
from cosmos_curator.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curator.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2FrameCreationStage,
)
from cosmos_curator.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature


def _frame_signature(fps: float) -> str:
    return FrameExtractionSignature(
        extraction_policy=FrameExtractionPolicy.sequence,
        target_fps=fps,
    ).to_str()


def _make_task(frame_fps: float) -> SplitPipeTask:
    frames = np.ones((2, 2, 2, 3), dtype=np.uint8)
    clip = Clip(
        uuid=UUID("12345678-1234-5678-1234-567812345678"),
        source_video="sample_video.mp4",
        span=(0.0, 2.0),
        encoded_data=np.ones(4, dtype=np.uint8),
        extracted_frames=LazyData(
            value={_frame_signature(frame_fps): frames},
            nbytes=frames.nbytes,
        ),
    )
    return SplitPipeTask(
        session_id="test-session",
        video=Video(input_video=Path("sample_video.mp4"), clips=[clip]),
    )


@pytest.mark.parametrize(
    ("stage_type", "stage_module"),
    [
        (CosmosEmbed1FrameCreationStage, cosmos_embed1_stages),
        (InternVideo2FrameCreationStage, internvideo2_stages),
    ],
    ids=["cosmos-embed1", "internvideo2"],
)
def test_frame_creation_missing_signature_drops_extracted_frames(
    monkeypatch: pytest.MonkeyPatch,
    stage_type: type[CosmosEmbed1FrameCreationStage | InternVideo2FrameCreationStage],
    stage_module: ModuleType,
) -> None:
    """A final local embedding consumer should release frames when its signature is missing."""
    task = _make_task(1.0)
    clip = task.video.clips[0]
    stage = stage_type(target_fps=2.0)
    messages: list[str] = []
    monkeypatch.setattr(stage_module, "logger", SimpleNamespace(error=messages.append))

    stage.process_data([task])

    assert len(messages) == 1
    assert clip.errors[f"frames-{_frame_signature(2.0)}"] == "missing"
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


@pytest.mark.parametrize(
    "stage_type",
    [CosmosEmbed1FrameCreationStage, InternVideo2FrameCreationStage],
    ids=["cosmos-embed1", "internvideo2"],
)
def test_frame_creation_missing_encoded_data_drops_extracted_frames(
    stage_type: type[CosmosEmbed1FrameCreationStage | InternVideo2FrameCreationStage],
) -> None:
    """A final local embedding consumer should release frames when encoded data is missing."""
    task = _make_task(2.0)
    clip = task.video.clips[0]
    clip.encoded_data.drop()
    stage = stage_type(target_fps=2.0)

    stage.process_data([task])

    assert clip.errors["encoded_data"] == "empty"
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


@pytest.mark.parametrize(
    ("stage_type", "output_attribute"),
    [
        (CosmosEmbed1FrameCreationStage, "cosmos_embed1_frames"),
        (InternVideo2FrameCreationStage, "intern_video_2_frames"),
    ],
    ids=["cosmos-embed1", "internvideo2"],
)
def test_frame_creation_success_drops_extracted_frames(
    monkeypatch: pytest.MonkeyPatch,
    stage_type: type[CosmosEmbed1FrameCreationStage | InternVideo2FrameCreationStage],
    output_attribute: str,
) -> None:
    """Successful local frame formulation should release the source frame map."""
    task = _make_task(2.0)
    clip = task.video.clips[0]
    stage = stage_type(target_fps=2.0)
    formulated_frames = np.ones((1, 2, 2), dtype=np.float32)
    model = SimpleNamespace(
        get_target_num_frames=lambda: 1,
        formulate_input_frames=lambda _frames: formulated_frames,
    )
    monkeypatch.setattr(stage, "_model", model)

    stage.process_data([task])

    output_frames = getattr(clip, output_attribute)
    assert np.array_equal(output_frames.resolve(), formulated_frames)
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


@pytest.mark.parametrize(
    "stage_type",
    [CosmosEmbed1FrameCreationStage, InternVideo2FrameCreationStage],
    ids=["cosmos-embed1", "internvideo2"],
)
def test_frame_creation_formulation_exception_retains_extracted_frames(
    monkeypatch: pytest.MonkeyPatch,
    stage_type: type[CosmosEmbed1FrameCreationStage | InternVideo2FrameCreationStage],
) -> None:
    """A formulation exception should propagate without mutating retry input."""
    task = _make_task(2.0)
    clip = task.video.clips[0]
    frame_map = clip.extracted_frames.resolve()
    original_nbytes = clip.extracted_frames.nbytes
    stage = stage_type(target_fps=2.0)

    def fail_formulation(_frames: list[npt.NDArray[np.uint8]]) -> None:
        error_msg = "formulation failed"
        raise RuntimeError(error_msg)

    model = SimpleNamespace(
        get_target_num_frames=lambda: 1,
        formulate_input_frames=fail_formulation,
    )
    monkeypatch.setattr(stage, "_model", model)

    with pytest.raises(RuntimeError, match="formulation failed"):
        stage.process_data([task])

    assert clip.extracted_frames.resolve() is frame_map
    assert clip.extracted_frames.nbytes == original_nbytes


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing-frames", "extracted frames missing"),
        ("api-error", "request failed"),
    ],
)
def test_openai_embedding_failure_drops_extracted_frames(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_error: str,
) -> None:
    """OpenAI should release the complete frame map after a handled embedding failure."""
    task = _make_task(1.0 if failure == "missing-frames" else 2.0)
    clip = task.video.clips[0]
    stage = OpenAIEmbeddingStage(model_name="test-model", target_fps=2.0, max_concurrent_requests=1)
    messages: list[str] = []
    monkeypatch.setattr(
        openai_embedding_stage,
        "logger",
        SimpleNamespace(error=messages.append, warning=messages.append),
    )

    if failure == "api-error":

        def fail_request(_frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            error_msg = "request failed"
            raise RuntimeError(error_msg)

        monkeypatch.setattr(stage, "_generate_embedding", fail_request)

    stage.process_data([task])

    assert len(messages) == 1
    assert clip.errors["openai_embedding"] == expected_error
    assert clip.extracted_frames.resolve() is None
    assert clip.extracted_frames.nbytes == 0


def test_openai_resolve_exception_retains_extracted_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected resolve failure should propagate without mutating retry input."""
    task = _make_task(2.0)
    clip = task.video.clips[0]
    frame_map = clip.extracted_frames.value
    original_nbytes = clip.extracted_frames.nbytes
    stage = OpenAIEmbeddingStage(model_name="test-model", target_fps=2.0, max_concurrent_requests=1)

    def fail_resolve(_lazy_data: LazyData[object]) -> None:
        error_msg = "resolve failed"
        raise RuntimeError(error_msg)

    monkeypatch.setattr(LazyData, "resolve", fail_resolve)

    with pytest.raises(RuntimeError, match="resolve failed"):
        stage.process_data([task])

    assert clip.extracted_frames.value is frame_map
    assert clip.extracted_frames.nbytes == original_nbytes
