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
"""Concrete CurationPhase implementations for the video splitting pipeline.

Each phase wraps one or more CuratorStage instances that form a logical group.
Phases declare their field-token dependencies via requires/populates so that
PipelineBuilder can validate composition order at construction time.

Field tokens used in this module
---------------------------------
  remuxed          — video bytes have been downloaded and remuxed
  split            — video has been split into clips
  transcoded       — clips have been transcoded to H.264
  motion_scored    — clips have been scored for motion (optional)
  frames_extracted — per-clip frame arrays are available in clip.extracted_frames
  aesthetics_scored — clips carry aesthetic_score (optional)
  qwen_filtered    — clips have been scored/filtered by Qwen (optional)
  embedded         — clips carry embedding arrays (optional)
  captioned        — clips carry caption windows (optional)
  t5_encoded       — caption windows carry T5 embeddings (optional)
"""

from typing import Literal, cast

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.models.all_models import get_all_models_by_id
from cosmos_curate.pipelines.video.captioning.captioning_stages import (
    EnhanceCaptionStage,
    T5StageForSplit,
)
from cosmos_curate.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curate.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage, VllmPrepStage
from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curate.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curate.pipelines.video.clipping.frame_extraction_stages import VideoFrameExtractionStage
from cosmos_curate.pipelines.video.clipping.transnetv2_extraction_stages import TransNetV2ClipExtractionStage
from cosmos_curate.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curate.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2EmbeddingStage,
    InternVideo2FrameCreationStage,
)
from cosmos_curate.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import AestheticFilterStage
from cosmos_curate.pipelines.video.filtering.aesthetics.qwen_filter_stages import (
    QwenFilteringStage,
    QwenInputPreparationStageFiltering,
)
from cosmos_curate.pipelines.video.filtering.motion.motion_filter_stages import (
    MotionFilterStage,
    MotionVectorDecodeStage,
)
from cosmos_curate.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curate.pipelines.video.read_write.download_stages import VideoDownloader
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curate.pipelines.video.read_write.remux_stages import RemuxStage
from cosmos_curate.pipelines.video.utils.data_model import VllmConfig, WindowConfig
from cosmos_curate.pipelines.video.utils.decoder_utils import FrameExtractionPolicy

VLLM_CAPTION_ALGOS: frozenset[str] = frozenset(
    {"nemotron", "phi4", "qwen", "qwen3_vl_30b", "qwen3_vl_30b_fp8", "qwen3_vl_235b", "qwen3_vl_235b_fp8"}
    | {"cosmos_r1", "cosmos_r2"}
)

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class IngestConfig:
    """Configuration for the ingest phase (download + remux)."""

    input_path: str
    num_workers_per_node: int = 4
    num_run_attempts: int = 5
    input_s3_profile_name: str = "default"
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class TransNetV2SplitConfig:
    """Configuration for TransNetV2-based scene splitting."""

    threshold: float = 0.4
    min_length_s: float = 2.0
    min_length_frames: int = 48
    max_length_s: float = 60.0
    max_length_mode: Literal["truncate", "stride"] = "stride"
    crop_s: float = 0.5
    num_gpus_per_worker: float = 0.25
    decoder_mode: str = "ffmpeg_cpu"
    num_decode_cpus_per_worker: float = 3.0
    raise_on_pynvc_error: bool = False
    limit_clips: int = 0
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class FixedStrideSplitConfig:
    """Configuration for fixed-stride clip splitting."""

    clip_len_s: int = 10
    clip_stride_s: int = 10
    min_clip_length_s: float = 2.0
    limit_clips: int = 0
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class TranscodeConfig:
    """Configuration for clip transcoding."""

    num_cpus_per_worker: float = 5.0
    encoder: str = "libopenh264"
    encoder_threads: int = 1
    encode_batch_size: int = 16
    use_hwaccel: bool = False
    use_input_bit_rate: bool = False
    num_clips_per_chunk: int = 32
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class MotionFilterConfig:
    """Configuration for motion filtering."""

    score_only: bool = False
    global_mean_threshold: float = 0.00098
    per_patch_min_256_threshold: float = 0.000001
    decode_cpus_per_worker: float = 2.0
    decode_target_fps: float = 2.0
    decode_target_duration_ratio: float = 0.5
    score_gpus_per_worker: float = 0.5
    score_batch_size: int = 64
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class FrameExtractionConfig:
    """Configuration for shared per-clip frame extraction (used by aesthetics and embedding)."""

    target_fps: list[float | int]
    target_res: int = -1
    cpus_per_worker: float = 3.0
    perf_profile: bool = False


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


_COSMOS_EMBED1_VARIANTS: frozenset[str] = frozenset({"224p", "336p", "448p"})


@attrs.define(frozen=True)
class EmbeddingConfig:
    """Configuration for clip embedding generation."""

    algorithm: str = "internvideo2"
    gpus_per_worker: float = 0.25
    batch_size: int = 8
    verbose: bool = False
    perf_profile: bool = False
    # Populated and validated in __attrs_post_init__; None for non-cosmos-embed1 algorithms.
    cosmos_embed1_variant: Literal["224p", "336p", "448p"] | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        """Parse and validate the cosmos-embed1 variant from the algorithm string."""
        if self.algorithm.startswith("cosmos-embed1-"):
            suffix = self.algorithm.split("-")[-1]
            if suffix not in _COSMOS_EMBED1_VARIANTS:
                msg = f"Invalid cosmos-embed1 variant {suffix!r}; expected one of {sorted(_COSMOS_EMBED1_VARIANTS)}"
                raise ValueError(msg)
            object.__setattr__(self, "cosmos_embed1_variant", cast("Literal['224p', '336p', '448p']", suffix))


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


@attrs.define(frozen=True)
class OutputConfig:
    """Configuration for the output/writer phase."""

    output_path: str
    input_path: str
    output_s3_profile_name: str = "default"
    upload_clips: bool = True
    upload_clip_info_in_chunks: bool = False
    upload_clip_info_in_lance: bool = False
    upload_cds_parquet: bool = False
    dry_run: bool = False
    generate_embeddings: bool = True
    embedding_algorithm: str = "internvideo2"
    embedding_model_version: str = "unspecified"
    generate_previews: bool = False
    caption_models: list[str] = attrs.Factory(list)
    enhanced_caption_models: list[str] = attrs.Factory(list)
    generate_cosmos_predict_dataset: str = "disable"
    num_workers_per_node: int = 8
    num_run_attempts: int = 5
    verbose: bool = False
    perf_profile: bool = False


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


class IngestPhase(CurationPhase):
    """Download and remux input videos."""

    def __init__(self, config: IngestConfig) -> None:
        """Initialise the ingest phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "ingest"

    @property
    def requires(self) -> frozenset[str]:
        """Return an empty set (no prior phase required)."""
        return frozenset()

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"remuxed"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the download and remux stages."""
        cfg = self._cfg
        return [
            CuratorStageSpec(
                VideoDownloader(
                    input_path=cfg.input_path,
                    input_s3_profile_name=cfg.input_s3_profile_name,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
                num_workers_per_node=cfg.num_workers_per_node,
                num_run_attempts_python=cfg.num_run_attempts,
            ),
            RemuxStage(
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            ),
        ]


class TransNetV2SplitPhase(CurationPhase):
    """Split videos into scene-coherent clips using TransNetV2."""

    def __init__(self, config: TransNetV2SplitConfig) -> None:
        """Initialise the TransNetV2 split phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "split/transnetv2"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"remuxed"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"split"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the frame extraction and scene detection stages."""
        cfg = self._cfg
        return [
            CuratorStageSpec(
                VideoFrameExtractionStage(
                    decoder_mode=cfg.decoder_mode,
                    num_cpus_per_worker=cfg.num_decode_cpus_per_worker,
                    raise_on_pynvc_error_without_cpu_fallback=cfg.raise_on_pynvc_error,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
            ),
            CuratorStageSpec(
                TransNetV2ClipExtractionStage(
                    threshold=cfg.threshold,
                    min_length_s=cfg.min_length_s,
                    min_length_frames=cfg.min_length_frames,
                    max_length_s=cfg.max_length_s,
                    max_length_mode=cfg.max_length_mode,
                    crop_s=cfg.crop_s,
                    num_gpus_per_worker=cfg.num_gpus_per_worker,
                    limit_clips=cfg.limit_clips,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
                over_provision_factor=2.0,
            ),
        ]


class FixedStrideSplitPhase(CurationPhase):
    """Split videos into fixed-duration clips."""

    def __init__(self, config: FixedStrideSplitConfig) -> None:
        """Initialise the fixed-stride split phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "split/fixed-stride"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"remuxed"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"split"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the fixed-stride extractor stage."""
        cfg = self._cfg
        return [
            CuratorStageSpec(
                FixedStrideExtractorStage(
                    clip_len_s=cfg.clip_len_s,
                    clip_stride_s=cfg.clip_stride_s,
                    min_clip_length_s=cfg.min_clip_length_s,
                    limit_clips=cfg.limit_clips,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
                num_workers_per_node=1,
            ),
        ]


class TranscodePhase(CurationPhase):
    """Transcode raw clips to H.264."""

    def __init__(self, config: TranscodeConfig) -> None:
        """Initialise the transcode phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "transcode"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"split"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"transcoded"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the clip transcoding stage."""
        cfg = self._cfg
        return [
            ClipTranscodingStage(
                num_cpus_per_worker=cfg.num_cpus_per_worker,
                encoder=cfg.encoder,
                encoder_threads=cfg.encoder_threads,
                encode_batch_size=cfg.encode_batch_size,
                use_hwaccel=cfg.use_hwaccel,
                use_input_bit_rate=cfg.use_input_bit_rate,
                num_clips_per_chunk=cfg.num_clips_per_chunk,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            ),
        ]


class MotionFilterPhase(CurationPhase):
    """Decode motion vectors and optionally filter low-motion clips."""

    def __init__(self, config: MotionFilterConfig) -> None:
        """Initialise the motion filter phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "motion_filter"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"motion_scored"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the motion decode and filter stages."""
        cfg = self._cfg
        return [
            MotionVectorDecodeStage(
                num_cpus_per_worker=cfg.decode_cpus_per_worker,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
                target_fps=cfg.decode_target_fps,
                target_duration_ratio=cfg.decode_target_duration_ratio,
            ),
            MotionFilterStage(
                score_only=cfg.score_only,
                global_mean_threshold=cfg.global_mean_threshold,
                per_patch_min_256_threshold=cfg.per_patch_min_256_threshold,
                num_gpus_per_worker=cfg.score_gpus_per_worker,
                batch_size=cfg.score_batch_size,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            ),
        ]


class FrameExtractionPhase(CurationPhase):
    """Extract per-clip frame arrays for use by aesthetics and embedding stages."""

    def __init__(self, config: FrameExtractionConfig) -> None:
        """Initialise the frame extraction phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "frame_extraction"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"frames_extracted"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the clip frame extraction stage."""
        cfg = self._cfg
        return [
            ClipFrameExtractionStage(
                extraction_policies=(FrameExtractionPolicy.sequence,),
                target_fps=cfg.target_fps,
                target_res=(cfg.target_res, cfg.target_res),
                num_cpus_per_worker=cfg.cpus_per_worker,
                log_stats=cfg.perf_profile,
            ),
        ]


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


class EmbeddingPhase(CurationPhase):
    """Generate clip embeddings using InternVideo2 or Cosmos-Embed1."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialise the embedding phase with the given configuration."""
        self._cfg = config

    def _build_embedding_stage(self) -> CuratorStage:
        """Construct the embedding stage matching the configured algorithm."""
        cfg = self._cfg
        if cfg.algorithm == "internvideo2":
            return InternVideo2EmbeddingStage(
                num_gpus_per_worker=cfg.gpus_per_worker,
                batch_size=cfg.batch_size,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        if cfg.algorithm.startswith("cosmos-embed1-"):
            assert cfg.cosmos_embed1_variant is not None
            return CosmosEmbed1EmbeddingStage(
                cfg.cosmos_embed1_variant,
                num_gpus_per_worker=cfg.gpus_per_worker,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        msg = f"Unknown embedding algorithm: {cfg.algorithm!r}"
        raise NotImplementedError(msg)

    @property
    def model_version(self) -> str:
        """Return the embedding model version string for output metadata."""
        model = self._build_embedding_stage().model
        if model is not None:
            model_id = model.model_id_names[0]
            return str(get_all_models_by_id().get(model_id, {}).get("version", "unspecified"))
        return "unspecified"

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "embedding"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"frames_extracted"})

    @property
    def populates(self) -> frozenset[str]:
        """Return the field tokens populated by this phase."""
        return frozenset({"embedded"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the frame creation and embedding stages."""
        cfg = self._cfg
        frame_stage: CuratorStage
        if cfg.algorithm == "internvideo2":
            frame_stage = InternVideo2FrameCreationStage(
                target_fps=2.0,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        else:
            assert cfg.cosmos_embed1_variant is not None
            frame_stage = CosmosEmbed1FrameCreationStage(
                cfg.cosmos_embed1_variant,
                target_fps=2.0,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        return [frame_stage, self._build_embedding_stage()]


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


class OutputPhase(CurationPhase):
    """Write clips, embeddings, captions, and metadata to object storage."""

    def __init__(self, config: OutputConfig) -> None:
        """Initialise the output phase with the given configuration."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return "output"

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return frozenset({"transcoded"})

    @property
    def populates(self) -> frozenset[str]:
        """Return an empty set (output phase produces no new field tokens)."""
        return frozenset()

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the clip writer stage."""
        cfg = self._cfg
        return [
            CuratorStageSpec(
                ClipWriterStage(
                    output_path=cfg.output_path,
                    input_path=cfg.input_path,
                    output_s3_profile_name=cfg.output_s3_profile_name,
                    upload_clips=cfg.upload_clips,
                    upload_clip_info_in_chunks=cfg.upload_clip_info_in_chunks,
                    upload_clip_info_in_lance=cfg.upload_clip_info_in_lance,
                    upload_cds_parquet=cfg.upload_cds_parquet,
                    dry_run=cfg.dry_run,
                    generate_embeddings=cfg.generate_embeddings,
                    embedding_algorithm=cfg.embedding_algorithm,
                    embedding_model_version=cfg.embedding_model_version,
                    generate_previews=cfg.generate_previews,
                    caption_models=cfg.caption_models,
                    enhanced_caption_models=cfg.enhanced_caption_models,
                    generate_cosmos_predict_dataset=cfg.generate_cosmos_predict_dataset,
                    verbose=cfg.verbose,
                    log_stats=cfg.perf_profile,
                ),
                num_workers_per_node=cfg.num_workers_per_node,
                num_run_attempts_python=cfg.num_run_attempts,
            ),
        ]
