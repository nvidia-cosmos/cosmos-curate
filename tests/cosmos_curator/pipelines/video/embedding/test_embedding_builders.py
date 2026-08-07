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
"""Tests for video embedding stage configuration and construction."""

from typing import cast

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.pipelines.video.embedding.cosmos_embed1_stages import CosmosEmbed1FrameCreationStage
from cosmos_curator.pipelines.video.embedding.embedding_builders import (
    CosmosEmbed1Config,
    EmbeddingBackendConfig,
    EmbeddingConfig,
    InternVideo2Config,
    OpenAIEmbeddingConfig,
    build_embedding_stages,
)
from cosmos_curator.pipelines.video.embedding.internvideo2_stages import InternVideo2FrameCreationStage
from cosmos_curator.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature


def _frame_signature(fps: float) -> str:
    return FrameExtractionSignature(
        extraction_policy=FrameExtractionPolicy.sequence,
        target_fps=fps,
    ).to_str()


def test_embedding_config_defaults_to_two_fps() -> None:
    """Programmatic callers should retain the existing 2-FPS default."""
    assert EmbeddingConfig().target_fps == 2.0


@pytest.mark.parametrize("value", [0, -1, 0.0009, float("nan"), float("inf"), "invalid"])
def test_embedding_config_rejects_invalid_sampling_fps(value: float | str) -> None:
    """The config boundary should reject values outside the supported numeric domain."""
    with pytest.raises(ValueError, match=r"greater than or equal to 0\.001"):
        EmbeddingConfig(target_fps=value)


@pytest.mark.parametrize("value", [True, False])
def test_embedding_config_rejects_boolean_sampling_fps(*, value: bool) -> None:
    """Booleans are types, not numeric sampling rates."""
    with pytest.raises(TypeError, match=r"greater than or equal to 0\.001"):
        EmbeddingConfig(target_fps=value)


def test_embedding_config_normalizes_sampling_fps_to_float() -> None:
    """Config-file string values should be retained as normalized floats."""
    assert EmbeddingConfig(target_fps="1.5").target_fps == 1.5


@pytest.mark.parametrize(
    ("backend", "lookup_stage_type"),
    [
        (CosmosEmbed1Config(variant="336p"), CosmosEmbed1FrameCreationStage),
        (InternVideo2Config(), InternVideo2FrameCreationStage),
        (OpenAIEmbeddingConfig(), OpenAIEmbeddingStage),
    ],
    ids=["cosmos-embed1", "internvideo2", "openai"],
)
def test_embedding_builders_use_configured_sampling_signature(
    monkeypatch: pytest.MonkeyPatch,
    backend: EmbeddingBackendConfig,
    lookup_stage_type: type[CuratorStage],
) -> None:
    """Every backend lookup stage should use the configured extraction signature."""
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.embedding.embedding_builders.CosmosEmbed1EmbeddingStage",
        lambda *_args, **_kwargs: CuratorStage(),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.embedding.embedding_builders.InternVideo2EmbeddingStage",
        lambda *_args, **_kwargs: CuratorStage(),
    )
    config = EmbeddingConfig(backend=backend, target_fps=1.5)

    stages = build_embedding_stages(config)

    lookup_stage = cast(
        "CosmosEmbed1FrameCreationStage | InternVideo2FrameCreationStage | OpenAIEmbeddingStage",
        next(stage for stage in stages if isinstance(stage, lookup_stage_type)),
    )
    assert lookup_stage._frame_extraction_signature == _frame_signature(1.5)
