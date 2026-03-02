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

"""Tests for the CaptioningPhase OpenAI dispatch paths in video_curation_phases."""

import pytest

from cosmos_curate.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage
from cosmos_curate.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curate.pipelines.video.utils.data_model import WindowConfig
from cosmos_curate.pipelines.video.video_curation_phases import (
    CaptioningConfig,
    CaptioningPhase,
    OpenAIConfig,
)


def _default_openai_config() -> CaptioningConfig:
    """Return a minimal CaptioningConfig for the openai algorithm."""
    return CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=OpenAIConfig(),
    )


# ---------------------------------------------------------------------------
# _build_prep_stage
# ---------------------------------------------------------------------------


def test_build_prep_stage_openai_returns_api_prep_stage() -> None:
    """The openai prep stage should be an ApiPrepStage."""
    phase = CaptioningPhase(_default_openai_config())
    stage = phase._build_prep_stage()
    assert isinstance(stage, ApiPrepStage)


def test_build_prep_stage_openai_uses_configured_cpus() -> None:
    """ApiPrepStage should receive num_cpus_for_prepare from OpenAIConfig."""
    cfg = CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=OpenAIConfig(num_cpus_for_prepare=5.0),
    )
    phase = CaptioningPhase(cfg)
    stage = phase._build_prep_stage()
    assert isinstance(stage, ApiPrepStage)
    assert stage._num_cpus_for_prepare == 5.0


def test_build_prep_stage_openai_missing_config_raises() -> None:
    """ValueError should be raised when openai_config is None."""
    cfg = CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=None,
    )
    phase = CaptioningPhase(cfg)
    with pytest.raises(ValueError, match="openai_config required"):
        phase._build_prep_stage()


# ---------------------------------------------------------------------------
# _build_caption_stage
# ---------------------------------------------------------------------------


def test_build_caption_stage_openai_returns_openai_stage() -> None:
    """The openai caption stage should be an OpenAICaptionStage."""
    phase = CaptioningPhase(_default_openai_config())
    stage = phase._build_caption_stage()
    assert isinstance(stage, OpenAICaptionStage)


def test_build_caption_stage_openai_forwards_config_params() -> None:
    """OpenAICaptionStage should receive parameters from OpenAIConfig."""
    cfg = CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=OpenAIConfig(
            model_name="my-custom-model",
            max_output_tokens=4096,
            caption_retries=5,
            retry_delay_seconds=2.0,
        ),
    )
    phase = CaptioningPhase(cfg)
    stage = phase._build_caption_stage()
    assert isinstance(stage, OpenAICaptionStage)
    assert stage._model_name == "my-custom-model"
    assert stage._max_output_tokens == 4096
    assert stage._max_caption_retries == 5
    assert stage._retry_delay_seconds == 2.0


def test_build_caption_stage_openai_missing_config_raises() -> None:
    """ValueError should be raised when openai_config is None for caption stage."""
    cfg = CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=None,
    )
    phase = CaptioningPhase(cfg)
    with pytest.raises(ValueError, match="openai_config required"):
        phase._build_caption_stage()


def test_build_caption_stage_unknown_algo_raises() -> None:
    """NotImplementedError for an unrecognized caption algorithm."""
    cfg = CaptioningConfig(
        caption_algo="unknown_algo",
        window_config=WindowConfig(),
    )
    phase = CaptioningPhase(cfg)
    with pytest.raises(NotImplementedError, match="Unknown caption algorithm"):
        phase._build_caption_stage()


# ---------------------------------------------------------------------------
# build_stages (full pipeline)
# ---------------------------------------------------------------------------


def test_build_stages_base_count() -> None:
    """Base openai build_stages should produce exactly 2 stages: prep + caption."""
    phase = CaptioningPhase(_default_openai_config())
    stages = phase.build_stages()
    assert len(stages) == 2
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[1], OpenAICaptionStage)


def test_build_stages_with_previews() -> None:
    """Enabling previews should add a PreviewStage (3 total)."""
    cfg = CaptioningConfig(
        caption_algo="openai",
        window_config=WindowConfig(),
        openai_config=OpenAIConfig(),
        generate_previews=True,
    )
    phase = CaptioningPhase(cfg)
    stages = phase.build_stages()
    assert len(stages) == 3
    # Preview is inserted between prep and caption
    assert isinstance(stages[0], ApiPrepStage)
    assert isinstance(stages[2], OpenAICaptionStage)


def test_phase_metadata() -> None:
    """CaptioningPhase should expose correct name, requires, and populates."""
    phase = CaptioningPhase(_default_openai_config())
    assert phase.name == "captioning"
    assert phase.requires == frozenset({"transcoded"})
    assert phase.populates == frozenset({"captioned"})
