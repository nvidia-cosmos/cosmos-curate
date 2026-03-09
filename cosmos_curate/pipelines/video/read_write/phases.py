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
"""CurationPhase implementations for download (includes inline remux) and output stages."""

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.read_write.download_stages import VideoDownloader
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage


@attrs.define(frozen=True)
class IngestConfig:
    """Configuration for the ingest phase (download, includes inline remux)."""

    input_path: str
    num_workers_per_node: int = 4
    num_run_attempts: int = 5
    input_s3_profile_name: str = "default"
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


class IngestPhase(CurationPhase):
    """Download input videos (includes inline remux for mpegts containers)."""

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
        """Construct and return the download stage (includes inline remux)."""
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
