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
"""CurationPhase implementations for captioning and T5 encoding stages."""

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.captioning.captioning_stages import (
    EnhanceCaptionStage,
    T5StageForSplit,
)
from cosmos_curate.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curate.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage, VllmPrepStage
from cosmos_curate.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curate.pipelines.video.utils.data_model import VllmConfig, WindowConfig

VLLM_CAPTION_ALGOS: frozenset[str] = frozenset(
    {"nemotron", "phi4", "qwen", "qwen3_vl_30b", "qwen3_vl_30b_fp8", "qwen3_vl_235b", "qwen3_vl_235b_fp8"}
    | {"cosmos_r1", "cosmos_r2"}
)


@attrs.define(frozen=True)
class EnhanceCaptionConfig:
    """Configuration for caption enhancement via a language model."""

    model_variant: str = "qwen_lm"
    batch_size: int = 32
    openai_model: str = "gpt-5.1-20251113"
    fp8_enable: bool = False
    max_output_tokens: int = 2048
    prompt_variant: str = "default"
    prompt_text: str | None = None
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class GeminiConfig:
    """Configuration specific to the Gemini API captioning path."""

    model_name: str = "models/gemini-2.5-pro"
    max_output_tokens: int = 512
    prompt_variant: str = "default"
    prompt_text: str | None = None
    caption_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_inline_video_bytes: int = 20 * 1024 * 1024
    num_cpus_for_prepare: float = 3.0


@attrs.define(frozen=True)
class OpenAIConfig:
    """Configuration specific to the OpenAI-compatible API captioning path."""

    model_name: str = "qwen3.5-397b-a17b-fp8"
    max_output_tokens: int = 8192
    prompt_variant: str = "default"
    prompt_text: str | None = None
    caption_retries: int = 3
    retry_delay_seconds: float = 1.0
    num_cpus_for_prepare: float = 3.0


@attrs.define(frozen=True)
class CaptioningConfig:
    """Configuration for the captioning phase (prep + caption + optional enhance)."""

    caption_algo: str
    window_config: WindowConfig
    vllm_config: VllmConfig | None = None
    gemini_config: GeminiConfig | None = None
    openai_config: OpenAIConfig | None = None
    keep_mp4: bool = False
    generate_previews: bool = False
    preview_target_fps: int = 1
    preview_target_height: int = 240
    inflight_batching: bool = True
    enhance_config: EnhanceCaptionConfig | None = None
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class T5Config:
    """Configuration for T5 encoding of captions."""

    caption_fields: list[str]
    verbose: bool = False
    perf_profile: bool = False


class CaptioningPhase(CurationPhase):
    """Prepare windows, generate captions, and optionally enhance them."""

    def __init__(self, config: CaptioningConfig) -> None:
        """Initialise the captioning phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "captioning"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"captioned"})

    def _build_prep_stage(self) -> CuratorStage:
        """Build the prep stage for the configured caption algorithm."""
        cfg = self._cfg
        caption_algo = cfg.caption_algo.lower()

        if caption_algo == "gemini":
            if cfg.gemini_config is None:
                msg = "gemini_config required for caption_algo='gemini'"
                raise ValueError(msg)
            return ApiPrepStage(
                window_config=cfg.window_config,
                model_variant=caption_algo,
                num_cpus_for_prepare=cfg.gemini_config.num_cpus_for_prepare,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )

        if caption_algo == "openai":
            if cfg.openai_config is None:
                msg = "openai_config required for caption_algo='openai'"
                raise ValueError(msg)
            return ApiPrepStage(
                window_config=cfg.window_config,
                model_variant=caption_algo,
                num_cpus_for_prepare=cfg.openai_config.num_cpus_for_prepare,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )

        if cfg.vllm_config is None:
            msg = "vllm_config required for VLLM captioning"
            raise ValueError(msg)
        vllm_cfg_prepare = attrs.evolve(cfg.vllm_config, copy_weights_to=None)
        return VllmPrepStage(
            vllm_config=vllm_cfg_prepare,
            window_config=cfg.window_config,
            keep_mp4=cfg.keep_mp4,
            verbose=cfg.verbose,
            log_stats=cfg.perf_profile,
        )

    def _build_caption_stage(self) -> CuratorStage | CuratorStageSpec:
        """Build the caption stage for the configured caption algorithm."""
        cfg = self._cfg
        caption_algo = cfg.caption_algo.lower()

        if caption_algo in VLLM_CAPTION_ALGOS:
            if cfg.vllm_config is None:
                msg = f"vllm_config required for caption_algo={caption_algo!r}"
                raise ValueError(msg)
            return CuratorStageSpec(
                VllmCaptionStage(
                    vllm_config=cfg.vllm_config,
                    verbose=cfg.verbose,
                    keep_mp4=cfg.keep_mp4,
                    log_stats=cfg.perf_profile,
                    inflight_batching=cfg.inflight_batching,
                ),
                num_setup_attempts_python=None,
            )

        if caption_algo == "gemini":
            if cfg.gemini_config is None:
                msg = "gemini_config required for caption_algo='gemini'"
                raise ValueError(msg)
            gcfg = cfg.gemini_config
            return GeminiCaptionStage(
                model_variant=caption_algo,
                model_name=gcfg.model_name,
                prompt_variant=gcfg.prompt_variant,
                prompt_text=gcfg.prompt_text,
                max_output_tokens=gcfg.max_output_tokens,
                max_caption_retries=gcfg.caption_retries,
                retry_delay_seconds=gcfg.retry_delay_seconds,
                max_video_size_bytes=gcfg.max_inline_video_bytes,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )

        if caption_algo == "openai":
            if cfg.openai_config is None:
                msg = "openai_config required for caption_algo='openai'"
                raise ValueError(msg)
            ocfg = cfg.openai_config
            return OpenAICaptionStage(
                model_name=ocfg.model_name,
                model_variant=caption_algo,
                prompt_variant=ocfg.prompt_variant,
                prompt_text=ocfg.prompt_text,
                max_output_tokens=ocfg.max_output_tokens,
                max_caption_retries=ocfg.caption_retries,
                retry_delay_seconds=ocfg.retry_delay_seconds,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )

        msg = f"Unknown caption algorithm: {caption_algo!r}"
        raise NotImplementedError(msg)

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the prep, optional preview, caption, and enhance stages."""
        cfg = self._cfg
        stages: list[CuratorStage | CuratorStageSpec] = [self._build_prep_stage()]

        if cfg.generate_previews:
            stages.append(
                PreviewStage(
                    target_fps=cfg.preview_target_fps,
                    target_height=cfg.preview_target_height,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                )
            )

        stages.append(self._build_caption_stage())

        if cfg.enhance_config is not None:
            ecfg = cfg.enhance_config
            stages.append(
                EnhanceCaptionStage(
                    model_variant=ecfg.model_variant,
                    batch_size=ecfg.batch_size,
                    openai_model=ecfg.openai_model,
                    fp8_enable=ecfg.fp8_enable,
                    max_output_tokens=ecfg.max_output_tokens,
                    prompt_variant=ecfg.prompt_variant,
                    prompt_text=ecfg.prompt_text,
                    verbose=ecfg.verbose,
                    log_stats=ecfg.perf_profile,
                )
            )

        return stages


class T5Phase(CurationPhase):
    """Encode captions with T5-XXL to produce training embeddings."""

    def __init__(self, config: T5Config) -> None:
        """Initialise the T5 encoding phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "t5_encode"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"captioned"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"t5_encoded"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the T5 encoding stage."""
        cfg = self._cfg
        return [
            CuratorStageSpec(
                T5StageForSplit(
                    caption_fields=cfg.caption_fields,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
            ),
        ]
