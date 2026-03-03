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
"""CurationPhase implementation for motion vector decoding and filtering."""

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.filtering.motion.motion_filter_stages import (
    MotionFilterStage,
    MotionVectorDecodeStage,
)


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
