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
    QwenVideoClassifierStage,
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
    """Configuration for Qwen-based semantic (criteria) content filtering."""

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


@attrs.define(frozen=True)
class QwenVideoClassifierConfig:
    """Configuration for Qwen-based video-type (allow/block list) filtering.

    Uses the 27 VIDEO_TYPE_LABELS (imaginaire VideoTypeClassifier taxonomy).
    """

    model_variant: str = "qwen"
    rejection_threshold: float = 0.5
    batch_size: int = 16
    fp8_enable: bool = False
    max_output_tokens: int = 512
    use_mmcache: bool = False
    sampling_fps: float = 2.0
    window_size: int = 256
    remainder_threshold: int = 128
    preprocess_dtype: str = "float16"
    model_does_preprocess: bool = False
    generate_previews: bool = False
    verbose: bool = False
    perf_profile: bool = False
    type_allow: str | None = None
    type_block: str | None = None


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


class QwenFilterClassifierPhase(CurationPhase):
    """Unified Qwen phase: run semantic filter and/or video classifier.

    Pass filter_config and/or classifier_config (at least one required). When both
    are set, classifier runs first so every clip gets qwen_type_classification
    before any are filtered by the semantic stage. Clips must pass all enabled
    stages to be kept.
    """

    def __init__(
        self,
        filter_config: QwenFilterConfig | None = None,
        classifier_config: QwenVideoClassifierConfig | None = None,
    ) -> None:
        """Initialise the phase with optional filter and/or classifier config (at least one required)."""
        if filter_config is None and classifier_config is None:
            msg = "At least one of filter_config or classifier_config is required"
            raise ValueError(msg)
        self._filter_cfg = filter_config
        self._classifier_cfg = classifier_config

    @property
    def name(self) -> str:
        """Return the phase name (qwen_filter, qwen_video_classifier, or qwen_filter_and_classifier)."""
        if self._filter_cfg is not None and self._classifier_cfg is not None:
            return "qwen_filter_and_classifier"
        if self._filter_cfg is not None:
            return "qwen_filter"
        return "qwen_video_classifier"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"qwen_filtered"})

    def _prep_stage(
        self,
        config: QwenVideoClassifierConfig | QwenFilterConfig,
        prompt_variant: str,
        filter_categories: str | None,
        *,
        extra_outputs: list[tuple[str, str] | tuple[str, str, str | None]] | None = None,
    ) -> QwenInputPreparationStageFiltering:
        """Build a single prep stage; both config types share the same window/sampling attrs."""
        return QwenInputPreparationStageFiltering(
            model_variant=config.model_variant,
            prompt_variant=prompt_variant,
            filter_categories=filter_categories,
            sampling_fps=config.sampling_fps,
            window_size=config.window_size,
            remainder_threshold=config.remainder_threshold,
            preprocess_dtype=config.preprocess_dtype,
            model_does_preprocess=config.model_does_preprocess,
            generate_previews=config.generate_previews,
            verbose=config.verbose,
            log_stats=config.perf_profile,
            extra_outputs=extra_outputs,
        )

    def _classifier_stage_spec(
        self, cc: QwenVideoClassifierConfig, *, clear_model_input_after: bool = True
    ) -> CuratorStageSpec:
        """Build classifier stage spec from config."""
        return CuratorStageSpec(
            QwenVideoClassifierStage(
                model_variant=cc.model_variant,
                batch_size=cc.batch_size,
                rejection_threshold=cc.rejection_threshold,
                type_allow=cc.type_allow,
                type_block=cc.type_block,
                fp8_enable=cc.fp8_enable,
                max_output_tokens=cc.max_output_tokens,
                disable_mmcache=not cc.use_mmcache,
                verbose=cc.verbose,
                log_stats=cc.perf_profile,
                model_does_preprocess=cc.model_does_preprocess,
                clear_model_input_after=clear_model_input_after,
            ),
        )

    def _filter_stage_spec(self, fc: QwenFilterConfig, *, model_input_key: str | None = None) -> CuratorStageSpec:
        """Build filter stage spec from config."""
        return CuratorStageSpec(
            QwenFilteringStage(
                model_variant=fc.model_variant,
                filter_variant=fc.prompt_variant,
                rejection_threshold=fc.rejection_threshold,
                user_prompt=fc.filter_categories,
                batch_size=fc.batch_size,
                fp8_enable=fc.fp8_enable,
                max_output_tokens=fc.max_output_tokens,
                disable_mmcache=not fc.use_mmcache,
                score_only=fc.score_only,
                verbose=fc.verbose,
                log_stats=fc.perf_profile,
                model_does_preprocess=fc.model_does_preprocess,
                model_input_key=model_input_key,
            ),
        )

    def _classifier_stages(self, cc: QwenVideoClassifierConfig) -> list[CuratorStage | CuratorStageSpec]:
        return [
            self._prep_stage(cc, "type", None),
            self._classifier_stage_spec(cc),
        ]

    def _filter_stages(self, fc: QwenFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
        return [
            self._prep_stage(fc, fc.prompt_variant, fc.filter_categories),
            self._filter_stage_spec(fc),
        ]

    def _stages_both(
        self, cc: QwenVideoClassifierConfig, fc: QwenFilterConfig
    ) -> list[CuratorStage | CuratorStageSpec]:
        """Single prep for both classifier and filter; classifier then filter, no double prep."""
        return [
            self._prep_stage(
                cc,
                "type",
                None,
                extra_outputs=[(fc.prompt_variant, "qwen_filter", fc.filter_categories)],
            ),
            self._classifier_stage_spec(cc, clear_model_input_after=False),
            self._filter_stage_spec(fc, model_input_key="qwen_filter"),
        ]

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Build and return the stage list (classifier then filter when both are enabled)."""
        fc = self._filter_cfg
        cc = self._classifier_cfg
        if cc is not None and fc is not None:
            return self._stages_both(cc, fc)
        if cc is not None:
            return self._classifier_stages(cc)
        assert fc is not None
        return self._filter_stages(fc)
