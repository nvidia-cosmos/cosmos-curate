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
from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage, VllmPrepStage
from cosmos_curate.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import AestheticFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.artificial_text_filter_stage import ArtificialTextFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.semantic_filter_prompts import (
    get_qwen_filter_prompt,  # type: ignore[import-untyped]
)
from cosmos_curate.pipelines.video.filtering.aesthetics.semantic_filter_stages import (  # type: ignore[import-untyped]
    VllmFilteringStage,
    VllmVideoClassifierStage,
    custom_categories_union,
    read_categories_file,  # used by _build_vllm_classifier_stages
)
from cosmos_curate.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig, WindowConfig


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

    use_gpu: bool = True
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
class VlmFilterConfig:
    """Configuration for VLM-based semantic (criteria) content filtering."""

    score_only: bool = False
    model_variant: str = "qwen"
    filter_categories: str | None = None
    prompt_variant: str = "default"
    rejection_threshold: float = 0.5
    batch_size: int = 16
    fp8_enable: bool = False
    max_output_tokens: int = 8192
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
class VideoClassifierConfig:
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
    max_output_tokens: int = 8192
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
    type_allow_file: str | None = None
    type_block_file: str | None = None


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
                num_gpus_per_worker=config.gpus_per_worker if config.use_gpu else 0.0,
                use_gpu=config.use_gpu,
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


def _make_vllm_config(config: VlmFilterConfig | VideoClassifierConfig, prompt_text: str) -> VllmConfig:
    """Build a VllmConfig from a filter/classifier config."""
    return VllmConfig(
        model_variant=config.model_variant,
        prompt_text=prompt_text,
        fp8=config.fp8_enable,
        preprocess=config.model_does_preprocess,
        disable_mmcache=not config.use_mmcache,
        num_gpus=1,
        batch_size=config.batch_size,
        sampling_config=VllmSamplingConfig(max_tokens=config.max_output_tokens),
    )


def _make_window_config(config: VlmFilterConfig | VideoClassifierConfig) -> WindowConfig:
    """Build a WindowConfig from a legacy Qwen filter/classifier config."""
    return WindowConfig(
        sampling_fps=config.sampling_fps,
        window_size=config.window_size,
        remainder_threshold=config.remainder_threshold,
        model_does_preprocess=config.model_does_preprocess,
        preprocess_dtype=config.preprocess_dtype,
    )


def _build_vllm_classifier_stages(cc: VideoClassifierConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build prep + inference + classifier post-processing stages using VllmCaptionStage."""
    type_allow = cc.type_allow or read_categories_file(cc.type_allow_file)
    type_block = cc.type_block or read_categories_file(cc.type_block_file)
    has_custom = cc.custom_categories or bool(cc.type_allow_file or cc.type_block_file)
    prompt_text = get_qwen_filter_prompt(
        "type",
        custom_categories_union(type_allow, type_block) if has_custom else None,
    )
    vllm_config = _make_vllm_config(cc, prompt_text)
    window_config = _make_window_config(cc)
    return [
        VllmPrepStage(
            vllm_config,
            window_config,
            use_filter_windows=True,
            keep_mp4=cc.generate_previews,
            verbose=cc.verbose,
            log_stats=cc.perf_profile,
        ),
        CuratorStageSpec(
            VllmCaptionStage(
                vllm_config,
                use_filter_windows=True,
                keep_mp4=cc.generate_previews,
                verbose=cc.verbose,
                log_stats=cc.perf_profile,
            )
        ),
        CuratorStageSpec(
            VllmVideoClassifierStage(
                model_variant=cc.model_variant,
                rejection_threshold=cc.rejection_threshold,
                type_allow=type_allow,
                type_block=type_block,
                custom_categories=has_custom,
                verbose=cc.verbose,
                log_stats=cc.perf_profile,
            )
        ),
    ]


def _build_vllm_filter_stages(fc: VlmFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build prep + inference + filter post-processing stages using VllmCaptionStage."""
    prompt_text = get_qwen_filter_prompt(fc.prompt_variant, fc.filter_categories)
    vllm_config = _make_vllm_config(fc, prompt_text)
    window_config = _make_window_config(fc)
    return [
        VllmPrepStage(
            vllm_config,
            window_config,
            use_filter_windows=True,
            keep_mp4=fc.generate_previews,
            verbose=fc.verbose,
            log_stats=fc.perf_profile,
        ),
        CuratorStageSpec(
            VllmCaptionStage(
                vllm_config,
                use_filter_windows=True,
                keep_mp4=fc.generate_previews,
                verbose=fc.verbose,
                log_stats=fc.perf_profile,
            )
        ),
        CuratorStageSpec(
            VllmFilteringStage(
                model_variant=fc.model_variant,
                user_prompt=fc.filter_categories,
                filter_variant=fc.prompt_variant,
                rejection_threshold=fc.rejection_threshold,
                score_only=fc.score_only,
                verbose=fc.verbose,
                log_stats=fc.perf_profile,
            )
        ),
    ]


def _build_vllm_stages_both(cc: VideoClassifierConfig, fc: VlmFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build classifier + filter stages using VllmCaptionStage; two separate prep+inference passes."""
    return [
        *_build_vllm_classifier_stages(cc),
        *_build_vllm_filter_stages(fc),
    ]


def build_vllm_filter_classifier_stages(
    filter_config: VlmFilterConfig | None = None,
    classifier_config: VideoClassifierConfig | None = None,
) -> list[CuratorStage | CuratorStageSpec]:
    """Build stages for vLLM-based semantic filter and/or video classifier.

    Uses VllmPrepStage + VllmCaptionStage for inference, supporting all vLLM model
    variants (Qwen2.5, Qwen3, Nemotron, etc.) and VLM endpoints. Pass filter_config
    and/or classifier_config (at least one required). When both are set, the classifier
    runs first so every clip gets qwen_type_classification before semantic filtering.
    """
    if filter_config is None and classifier_config is None:
        msg = "At least one of filter_config or classifier_config is required"
        raise ValueError(msg)
    if classifier_config is not None and filter_config is not None:
        return _build_vllm_stages_both(classifier_config, filter_config)
    if classifier_config is not None:
        return _build_vllm_classifier_stages(classifier_config)
    assert filter_config is not None
    return _build_vllm_filter_stages(filter_config)
