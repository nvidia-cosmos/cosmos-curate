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

"""Focused tests for vLLM async phase-level config validation."""

import pytest

from cosmos_curate.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curate.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage
from cosmos_curate.pipelines.video.captioning.phases import (
    CaptioningConfig,
    CaptioningPhase,
    VllmAsyncCaptionConfig,
)
from cosmos_curate.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig
from cosmos_curate.pipelines.video.captioning.vllm_async_stage import (
    VllmAsyncPrepStage,
    VllmAsyncPromptRenderStage,
)
from cosmos_curate.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curate.pipelines.video.utils.data_model import WindowConfig


def test_vllm_async_caption_config_default_uses_auto_mode_sentinel() -> None:
    """Default config should use `0` as the auto-derive sentinel."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.max_concurrent_requests == 0


def test_vllm_async_caption_config_accepts_zero_concurrency_for_auto_mode() -> None:
    """`max_concurrent_requests=0` should be valid (auto-derive mode)."""
    cfg = VllmAsyncCaptionConfig(max_concurrent_requests=0)
    assert cfg.max_concurrent_requests == 0


def test_vllm_async_caption_config_rejects_negative_concurrency() -> None:
    """Negative concurrency should fail validation."""
    with pytest.raises(ValueError, match="must be >= 0"):
        VllmAsyncCaptionConfig(max_concurrent_requests=-1)


def test_num_workers_per_node_default_is_zero() -> None:
    """Default num_workers_per_node should be 0 (autoscale)."""
    cfg = VllmAsyncCaptionConfig()
    assert cfg.num_workers_per_node == 0


class TestPhasesNumWorkersPerNode:
    """Verify phases.py sets num_workers_per_node based on mode and config."""

    @staticmethod
    def _build_phase(
        data_parallel_size: int = 1,
        num_gpus: int = 1,
        num_workers_per_node: int = 0,
    ) -> CaptioningPhase:
        serve_config = VllmAsyncConfig(
            model_variant="qwen",
            num_gpus=num_gpus,
            data_parallel_size=data_parallel_size,
        )
        vllm_async_config = VllmAsyncCaptionConfig(
            serve_config=serve_config,
            num_workers_per_node=num_workers_per_node,
        )
        cfg = CaptioningConfig(
            caption_algo="vllm_async",
            window_config=WindowConfig(),
            vllm_async_config=vllm_async_config,
        )
        return CaptioningPhase(cfg)

    def test_default_autoscale(self) -> None:
        """Default (0) -> Xenna autoscale with OPF=1.5."""
        phase = self._build_phase(num_workers_per_node=0)
        result = phase._build_caption_stage()
        assert isinstance(result, CuratorStageSpec)
        assert result.num_workers_per_node is None
        assert result.over_provision_factor == 1.5

    def test_explicit_workers(self) -> None:
        """Explicit positive value -> exact worker count."""
        phase = self._build_phase(num_workers_per_node=7)
        result = phase._build_caption_stage()
        assert isinstance(result, CuratorStageSpec)
        assert result.num_workers_per_node == 7

    def test_dp_mode_ignores_num_workers(self) -> None:
        """DP mode (dp=7): always 1 worker, num_workers_per_node ignored."""
        phase = self._build_phase(data_parallel_size=7, num_workers_per_node=5)
        result = phase._build_caption_stage()
        assert isinstance(result, CuratorStageSpec)
        assert result.num_workers_per_node == 1

    def test_prep_stage_spec_config(self) -> None:
        """VllmAsyncPrepStage should have OPF=2.0 and default slots_per_actor."""
        phase = self._build_phase()
        result = phase._build_vllm_async_prep_stage()
        assert isinstance(result, CuratorStageSpec)
        assert result.slots_per_actor is None
        assert result.over_provision_factor == 2.0

    def test_prep_stage_receives_windowing_fields(self) -> None:
        """VllmAsyncPrepConfig should receive windowing fields from CaptioningConfig."""
        phase = self._build_phase()
        result = phase._build_vllm_async_prep_stage()
        assert isinstance(result, CuratorStageSpec)
        stage = result.stage
        assert isinstance(stage, VllmAsyncPrepStage)
        assert stage._prep_config.window_size == 256
        assert stage._prep_config.remainder_threshold == 128
        assert stage._prep_config.keep_mp4 is False

    def test_build_stages_vllm_async_prep_is_first(self) -> None:
        """vllm_async build_stages should produce Prep + Render + Caption (3 stages)."""
        phase = self._build_phase()
        stages = phase.build_stages()
        assert len(stages) == 3
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmAsyncPrepStage)
        assert isinstance(stages[1], CuratorStageSpec)
        assert isinstance(stages[1].stage, VllmAsyncPromptRenderStage)

    def test_build_stages_vllm_async_no_api_prep_stage(self) -> None:
        """vllm_async pipeline should NOT include ApiPrepStage."""
        phase = self._build_phase()
        stages = phase.build_stages()
        for s in stages:
            stage_obj = s.stage if isinstance(s, CuratorStageSpec) else s
            assert not isinstance(stage_obj, ApiPrepStage)

    def test_build_stages_vllm_async_with_previews(self) -> None:
        """Enabling previews should produce Prep + Preview + Render + Caption (4 stages)."""
        serve_config = VllmAsyncConfig(model_variant="qwen")
        vllm_async_config = VllmAsyncCaptionConfig(serve_config=serve_config)
        cfg = CaptioningConfig(
            caption_algo="vllm_async",
            window_config=WindowConfig(),
            vllm_async_config=vllm_async_config,
            generate_previews=True,
        )
        phase = CaptioningPhase(cfg)
        stages = phase.build_stages()
        assert len(stages) == 4
        assert isinstance(stages[0], CuratorStageSpec)
        assert isinstance(stages[0].stage, VllmAsyncPrepStage)
        assert isinstance(stages[1], PreviewStage)
        assert isinstance(stages[2], CuratorStageSpec)
        assert isinstance(stages[2].stage, VllmAsyncPromptRenderStage)

    def test_prep_config_keep_mp4_from_generate_previews(self) -> None:
        """keep_mp4 should be True when generate_previews is enabled."""
        serve_config = VllmAsyncConfig(model_variant="qwen")
        vllm_async_config = VllmAsyncCaptionConfig(serve_config=serve_config)
        cfg = CaptioningConfig(
            caption_algo="vllm_async",
            window_config=WindowConfig(),
            vllm_async_config=vllm_async_config,
            generate_previews=True,
        )
        phase = CaptioningPhase(cfg)
        result = phase._build_vllm_async_prep_stage()
        assert isinstance(result, CuratorStageSpec)
        stage = result.stage
        assert isinstance(stage, VllmAsyncPrepStage)
        assert stage._prep_config.keep_mp4 is True
