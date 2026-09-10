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
"""Stage builders for download (includes inline remux) and output stages."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.read_write.download_stages import VideoDownloader
from cosmos_curator.pipelines.video.read_write.mcap_writer_stage import (
    DEFAULT_CAPTURE_TIMEZONE,
    McapWriterStage,
)
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.tracking.serialization import Sam3OutputFormat


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
    # Safer default for non-captioning callers; split pipeline opts in explicitly when captions run.
    caption_quality_stats_enabled: bool = False
    caption_quality_flags_enabled: bool = True
    generate_cosmos_predict_dataset: bool = False
    sam3_output_format: Sam3OutputFormat = "native"
    # Also appends an McapWriterStage after the clip writer, which then retains the
    # clip annotations (captions/embeddings) the MCAP writer consumes; clip mp4s are
    # read back from the written clips/ output rather than retained. Keeping both on
    # one knob guarantees a retaining ClipWriterStage always has its consumer.
    generate_mcap: bool = False
    mcap_num_workers_per_node: int = 2
    # Zone the capture time embedded in a source video's path is expressed in; the MCAP
    # writer needs it to turn that local reading into an absolute (epoch) log_time.
    mcap_capture_timezone: str = DEFAULT_CAPTURE_TIMEZONE
    num_workers_per_node: int = 8
    num_run_attempts: int = 5
    verbose: bool = False
    perf_profile: bool = False


def build_ingest_stages(config: IngestConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the download stage (includes inline remux)."""
    return [
        CuratorStageSpec(
            VideoDownloader(
                input_path=config.input_path,
                input_s3_profile_name=config.input_s3_profile_name,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=config.num_workers_per_node,
            num_run_attempts_python=config.num_run_attempts,
        ),
    ]


def build_output_stages(config: OutputConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the clip writer stage (plus the MCAP writer when enabled)."""
    if config.generate_mcap and not config.upload_clips:
        msg = "generate_mcap requires upload_clips: the MCAP writer reads clip mp4s back from the clips/ output"
        raise ValueError(msg)
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            ClipWriterStage(
                output_path=config.output_path,
                input_path=config.input_path,
                output_s3_profile_name=config.output_s3_profile_name,
                upload_clips=config.upload_clips,
                upload_clip_info_in_chunks=config.upload_clip_info_in_chunks,
                upload_clip_info_in_lance=config.upload_clip_info_in_lance,
                upload_cds_parquet=config.upload_cds_parquet,
                dry_run=config.dry_run,
                generate_embeddings=config.generate_embeddings,
                embedding_algorithm=config.embedding_algorithm,
                embedding_model_version=config.embedding_model_version,
                generate_previews=config.generate_previews,
                caption_models=config.caption_models,
                enhanced_caption_models=config.enhanced_caption_models,
                caption_quality_stats_enabled=config.caption_quality_stats_enabled,
                caption_quality_flags_enabled=config.caption_quality_flags_enabled,
                generate_cosmos_predict_dataset=config.generate_cosmos_predict_dataset,
                sam3_output_format=config.sam3_output_format,
                retain_clip_data=config.generate_mcap,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=config.num_workers_per_node,
            num_run_attempts_python=config.num_run_attempts,
        ),
    ]
    if config.generate_mcap:
        stages.append(
            CuratorStageSpec(
                McapWriterStage(
                    output_path=config.output_path,
                    input_path=config.input_path,
                    output_s3_profile_name=config.output_s3_profile_name,
                    embedding_algorithm=config.embedding_algorithm,
                    embedding_model_version=config.embedding_model_version,
                    caption_models=config.caption_models,
                    capture_timezone=config.mcap_capture_timezone,
                    dry_run=config.dry_run,
                    verbose=config.verbose,
                    log_stats=config.perf_profile,
                ),
                # Defaults below the clip writer: each worker holds one clip mp4 while remuxing.
                num_workers_per_node=config.mcap_num_workers_per_node,
                num_run_attempts_python=config.num_run_attempts,
            ),
        )
    return stages
