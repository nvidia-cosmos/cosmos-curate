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
"""Stage builders for aesthetic and Qwen-based content filtering."""

from typing import Literal

import attrs

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import AestheticFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.artificial_text_filter_stage import ArtificialTextFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_stages import (
    QwenFilteringStage,
    QwenInputPreparationStageFiltering,
    QwenVideoClassifierStage,
    custom_categories_union,
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
class ArtificialTextFilterConfig:
    """Configuration for artificial text (overlay/post-production) filtering."""

    gpus_per_worker: float = 0.25
    use_corner_detection: bool = True
    frame_interval: int = 3
    min_duration_frames: int = 10
    min_duration_frames_corner_ratio: float = 0.1
    stability_iou_threshold: float = 0.9
    ignore_corner_region: bool = False
    corner_x_margin_norm: float = 0.1
    corner_y_margin_norm: float = 0.1
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

    By default uses the 27 VIDEO_TYPE_LABELS (imaginaire VideoTypeClassifier taxonomy).
    When custom_categories is True, type_allow and type_block define the full set of
    categories (plus an "unclassified" fallback when none match); the model is prompted
    only for those, and allow/block logic applies.
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
    custom_categories: bool = False


def build_aesthetic_filter_stages(config: AestheticFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the aesthetic filter stage."""
    return [
        AestheticFilterStage(
            score_threshold=config.score_threshold,
            reduction=config.reduction,
            num_gpus_per_worker=config.gpus_per_worker,
            verbose=config.verbose,
            log_stats=config.perf_profile,
        ),
    ]


def build_artificial_text_filter_stages(config: ArtificialTextFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the artificial text filter stage."""
    return [
        CuratorStageSpec(
            ArtificialTextFilterStage(
                num_gpus_per_worker=config.gpus_per_worker,
                use_corner_detection=config.use_corner_detection,
                frame_interval=config.frame_interval,
                min_duration_frames=config.min_duration_frames,
                min_duration_frames_corner_ratio=config.min_duration_frames_corner_ratio,
                stability_iou_threshold=config.stability_iou_threshold,
                ignore_corner_region=config.ignore_corner_region,
                corner_x_margin_norm=config.corner_x_margin_norm,
                corner_y_margin_norm=config.corner_y_margin_norm,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
        ),
    ]


def _build_prep_stage(
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


def _build_classifier_stage_spec(
    cc: QwenVideoClassifierConfig, *, clear_model_input_after: bool = True
) -> CuratorStageSpec:
    """Build classifier stage spec from config."""
    return CuratorStageSpec(
        QwenVideoClassifierStage(
            model_variant=cc.model_variant,
            batch_size=cc.batch_size,
            rejection_threshold=cc.rejection_threshold,
            type_allow=cc.type_allow,
            type_block=cc.type_block,
            custom_categories=cc.custom_categories,
            fp8_enable=cc.fp8_enable,
            max_output_tokens=cc.max_output_tokens,
            disable_mmcache=not cc.use_mmcache,
            verbose=cc.verbose,
            log_stats=cc.perf_profile,
            model_does_preprocess=cc.model_does_preprocess,
            clear_model_input_after=clear_model_input_after,
        ),
    )


def _build_filter_stage_spec(fc: QwenFilterConfig, *, model_input_key: str | None = None) -> CuratorStageSpec:
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


def _build_classifier_stages(cc: QwenVideoClassifierConfig) -> list[CuratorStage | CuratorStageSpec]:
    return [
        _build_prep_stage(
            cc,
            "type",
            custom_categories_union(cc.type_allow, cc.type_block) if cc.custom_categories else None,
        ),
        _build_classifier_stage_spec(cc),
    ]


def _build_filter_stages(fc: QwenFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    return [
        _build_prep_stage(fc, fc.prompt_variant, fc.filter_categories),
        _build_filter_stage_spec(fc),
    ]


def _build_stages_both(cc: QwenVideoClassifierConfig, fc: QwenFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Single prep for both classifier and filter; classifier then filter, no double prep."""
    return [
        _build_prep_stage(
            cc,
            "type",
            custom_categories_union(cc.type_allow, cc.type_block) if cc.custom_categories else None,
            extra_outputs=[(fc.prompt_variant, "qwen_filter", fc.filter_categories)],
        ),
        _build_classifier_stage_spec(cc, clear_model_input_after=False),
        _build_filter_stage_spec(fc, model_input_key="qwen_filter"),
    ]


def build_qwen_filter_classifier_stages(
    filter_config: QwenFilterConfig | None = None,
    classifier_config: QwenVideoClassifierConfig | None = None,
) -> list[CuratorStage | CuratorStageSpec]:
    """Build stages for Qwen semantic filter and/or video classifier.

    Pass filter_config and/or classifier_config (at least one required). When both
    are set, classifier runs first so every clip gets qwen_type_classification
    before any are filtered by the semantic stage.
    """
    if filter_config is None and classifier_config is None:
        msg = "At least one of filter_config or classifier_config is required"
        raise ValueError(msg)
    if classifier_config is not None and filter_config is not None:
        return _build_stages_both(classifier_config, filter_config)
    if classifier_config is not None:
        return _build_classifier_stages(classifier_config)
    assert filter_config is not None
    return _build_filter_stages(filter_config)
