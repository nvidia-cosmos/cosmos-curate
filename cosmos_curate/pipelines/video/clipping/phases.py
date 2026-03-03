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
"""CurationPhase implementations for splitting, transcoding, and frame extraction stages."""

from typing import Literal

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curate.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curate.pipelines.video.clipping.frame_extraction_stages import VideoFrameExtractionStage
from cosmos_curate.pipelines.video.clipping.transnetv2_extraction_stages import TransNetV2ClipExtractionStage
from cosmos_curate.pipelines.video.utils.decoder_utils import FrameExtractionPolicy


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
class FrameExtractionConfig:
    """Configuration for shared per-clip frame extraction (used by aesthetics and embedding)."""

    target_fps: list[float | int]
    target_res: int = -1
    cpus_per_worker: float = 3.0
    perf_profile: bool = False


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
