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
"""CurationPhase implementation for clip embedding generation."""

from typing import Literal, cast

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.models.all_models import get_all_models_by_id
from cosmos_curate.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curate.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2EmbeddingStage,
    InternVideo2FrameCreationStage,
)
from cosmos_curate.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage

_COSMOS_EMBED1_VARIANTS: frozenset[str] = frozenset({"224p", "336p", "448p"})


@attrs.define(frozen=True)
class OpenAIEmbeddingConfig:
    """Configuration specific to the OpenAI-compatible API embedding path."""

    model_name: str = "auto"
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_requests: int = 8


@attrs.define(frozen=True)
class EmbeddingConfig:
    """Configuration for clip embedding generation."""

    algorithm: str = "internvideo2"
    target_fps: float = 2.0
    gpus_per_worker: float = 0.25
    batch_size: int = 8
    verbose: bool = False
    perf_profile: bool = False
    openai_config: OpenAIEmbeddingConfig | None = None
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


class EmbeddingPhase(CurationPhase):
    """Generate clip embeddings using InternVideo2, Cosmos-Embed1, or an OpenAI-compatible API."""

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
        if cfg.algorithm == "openai":
            if cfg.openai_config is None:
                msg = "openai_config required for algorithm='openai'"
                raise ValueError(msg)
            return OpenAIEmbeddingStage(
                model_name=cfg.openai_config.model_name,
                target_fps=cfg.target_fps,
                max_retries=cfg.openai_config.max_retries,
                retry_delay_seconds=cfg.openai_config.retry_delay_seconds,
                max_concurrent_requests=cfg.openai_config.max_concurrent_requests,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        msg = f"Unknown embedding algorithm: {cfg.algorithm!r}"
        raise NotImplementedError(msg)

    @property
    def model_version(self) -> str:
        """Return the embedding model version string for output metadata."""
        if self._cfg.algorithm == "openai":
            # No local model registry entry for API-based embedding; use the model name.
            return self._cfg.openai_config.model_name if self._cfg.openai_config else "unspecified"
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

        # OpenAI embedding reads pre-extracted frames directly; no model-specific frame creation stage.
        if cfg.algorithm == "openai":
            return [self._build_embedding_stage()]

        frame_stage: CuratorStage
        if cfg.algorithm == "internvideo2":
            frame_stage = InternVideo2FrameCreationStage(
                target_fps=cfg.target_fps,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        else:
            assert cfg.cosmos_embed1_variant is not None
            frame_stage = CosmosEmbed1FrameCreationStage(
                cfg.cosmos_embed1_variant,
                target_fps=cfg.target_fps,
                verbose=cfg.verbose,
                log_stats=cfg.perf_profile,
            )
        return [frame_stage, self._build_embedding_stage()]
