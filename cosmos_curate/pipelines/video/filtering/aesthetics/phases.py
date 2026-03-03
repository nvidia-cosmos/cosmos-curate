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
"""CurationPhase implementations for aesthetic and Qwen-based content filtering."""

from typing import Literal

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import AestheticFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_stages import (
    QwenFilteringStage,
    QwenInputPreparationStageFiltering,
)


@attrs.define(frozen=True)
class AestheticFilterConfig:
    """Configuration for aesthetic quality filtering."""

    score_threshold: float
    reduction: Literal["mean", "min"] = "min"
    gpus_per_worker: float = 0.25
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class QwenFilterConfig:
    """Configuration for Qwen-based content filtering."""

    score_only: bool = False
    model_variant: str = "qwen"
    filter_categories: str | None = None
    prompt_variant: str = "default"
    rejection_threshold: float = 0.5
    batch_size: int = 16
    fp8_enable: bool = False
    max_output_tokens: int = 512
    use_mmcache: bool = False
    # window params shared with captioning
    sampling_fps: float = 2.0
    window_size: int = 256
    remainder_threshold: int = 128
    preprocess_dtype: str = "float16"
    model_does_preprocess: bool = False
    generate_previews: bool = False
    verbose: bool = False
    perf_profile: bool = False


class AestheticFilterPhase(CurationPhase):
    """Score and optionally filter clips by aesthetic quality."""

    def __init__(self, config: AestheticFilterConfig) -> None:
        """Initialise the aesthetic filter phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "aesthetic_filter"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"frames_extracted"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"aesthetics_scored"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the aesthetic filter stage."""
        cfg = self._cfg
        return [
            AestheticFilterStage(
                score_threshold=cfg.score_threshold,
                reduction=cfg.reduction,
                num_gpus_per_worker=cfg.gpus_per_worker,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            ),
        ]


class QwenFilterPhase(CurationPhase):
    """Prepare and run Qwen-based content filtering."""

    def __init__(self, config: QwenFilterConfig) -> None:
        """Initialise the Qwen filter phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "qwen_filter"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"qwen_filtered"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the Qwen input preparation and filtering stages."""
        cfg = self._cfg
        return [
            QwenInputPreparationStageFiltering(
                model_variant=cfg.model_variant,
                filter_categories=cfg.filter_categories,
                prompt_variant=cfg.prompt_variant,
                sampling_fps=cfg.sampling_fps,
                window_size=cfg.window_size,
                remainder_threshold=cfg.remainder_threshold,
                preprocess_dtype=cfg.preprocess_dtype,
                model_does_preprocess=cfg.model_does_preprocess,
                generate_previews=cfg.generate_previews,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            ),
            CuratorStageSpec(
                QwenFilteringStage(
                    model_variant=cfg.model_variant,
                    filter_variant=cfg.prompt_variant,
                    rejection_threshold=cfg.rejection_threshold,
                    user_prompt=cfg.filter_categories,
                    batch_size=cfg.batch_size,
                    fp8_enable=cfg.fp8_enable,
                    max_output_tokens=cfg.max_output_tokens,
                    disable_mmcache=not cfg.use_mmcache,
                    score_only=cfg.score_only,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
            ),
        ]
