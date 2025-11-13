# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Ray pipelines.

Which:
  - Download videos
  - Split videos into segments
  - Transcode raw videos into clips
  - Generate an embedding for the clip
  - Caption the clip
"""

from __future__ import annotations

import argparse
import pathlib
import time

from loguru import logger

from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.core.utils.config import args_utils
from cosmos_curate.core.utils.storage.storage_utils import (
    create_path,
    is_path_nested,
    verify_path,
)
from cosmos_curate.models.all_models import get_all_models_by_id
from cosmos_curate.pipelines.pipeline_args import (
    add_common_args,
)
from cosmos_curate.pipelines.video.captioning.api_caption_stage import ApiCaptionStage, ApiPrepStage
from cosmos_curate.pipelines.video.captioning.captioning_stages import (
    EnhanceCaptionStage,
    T5StageForSplit,
)
from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import (
    VllmCaptionStage,
    VllmPrepStage,
)
from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curate.pipelines.video.clipping.clip_frame_extraction_stages import (
    ClipFrameExtractionStage,
)
from cosmos_curate.pipelines.video.clipping.frame_extraction_stages import (
    VideoFrameExtractionStage,
)
from cosmos_curate.pipelines.video.clipping.transnetv2_extraction_stages import (
    TransNetV2ClipExtractionStage,
)
from cosmos_curate.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curate.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2EmbeddingStage,
    InternVideo2FrameCreationStage,
)
from cosmos_curate.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import (
    AestheticFilterStage,
)
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
from cosmos_curate.pipelines.video.read_write.summary_writers import (
    write_split_summary,
)
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, VllmConfig, WindowConfig
from cosmos_curate.pipelines.video.utils.decoder_utils import FrameExtractionPolicy
from cosmos_curate.pipelines.video.utils.video_pipe_input import (
    extract_split_tasks,
)


def build_input_data(
    args: argparse.Namespace,
) -> tuple[list[SplitPipeTask], list[str], int]:
    """Build input data for the pipeline.

    This function validates input arguments, extracts input data, and returns a list of tasks and relative paths.

    Args:
        args: Command line arguments.

    Returns:
        A tuple containing:
        - A list of SplitPipeTask objects.
        - A list of relative paths to the input videos.
        - The number of processed videos.

    """
    # validate input arguments
    verify_path(args.input_video_path)
    verify_path(args.output_clip_path, level=1)
    create_path(args.output_clip_path)
    if is_path_nested(args.input_video_path, args.output_clip_path):
        error_msg = "Do not make input and output paths nested"
        raise ValueError(error_msg)

    # extract input data
    input_videos, input_videos_relative, num_processed = extract_split_tasks(
        input_path=args.input_video_path,
        input_video_list_json_path=args.input_video_list_json_path,
        output_path=args.output_clip_path,
        output_video_path=ClipWriterStage.get_output_path_processed_videos(args.output_clip_path),
        output_clip_chunk_path=ClipWriterStage.get_output_path_processed_clip_chunks(args.output_clip_path),
        input_s3_profile_name=args.input_s3_profile_name,
        input_video_list_s3_profile_name=args.input_video_list_s3_profile_name,
        output_s3_profile_name=args.output_s3_profile_name,
        limit=args.limit,
        verbose=args.verbose,
    )

    logger.info(f"About to process {len(input_videos)} raw videos ...")
    if args.verbose:
        logger.debug("\n".join(str(x.input_video) for x in input_videos))

    input_tasks = [SplitPipeTask(video) for video in input_videos]

    return input_tasks, input_videos_relative, num_processed


def write_summary(
    args: argparse.Namespace,
    input_videos: list[str],
    output_tasks: list[SplitPipeTask],
    pipeline_run_time: float,
) -> float:
    """Write a summary of the pipeline run.

    This function writes a summary of the pipeline run, including the total video length and performance metrics.

    Args:
        args: Command line arguments.
        input_videos: List of input video paths.
        output_tasks: List of output tasks.
        pipeline_run_time: Total runtime of the pipeline in minutes.

    Returns:
        Total video length in hours.

    """
    write_split_summary(
        args.input_video_path,
        input_videos,
        args.output_clip_path,
        args.output_s3_profile_name,
        output_tasks,
        args.embedding_algorithm,
        args.limit,
        perf_profile=args.perf_profile,
        pipeline_run_time=pipeline_run_time,
        write_all_caption_json=args.write_all_caption_json,
    )

    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    total_video_length = 0.0
    for task in output_tasks:
        video = task.video
        if video.clip_chunk_index == 0:
            total_video_length += video.metadata.duration / 3600 if video.metadata.duration else 0

    return total_video_length


def get_embedding_stages(args: argparse.Namespace) -> list[CuratorStage | CuratorStageSpec]:
    """Get the embedding stages for the pipeline.

    Args:
        args: Command line arguments.

    Returns:
        A list of embedding stages.

    """
    stages: list[CuratorStage | CuratorStageSpec] = []
    if args.embedding_algorithm == "internvideo2":
        stages.extend(
            [
                InternVideo2FrameCreationStage(target_fps=2.0, verbose=args.verbose, log_stats=args.perf_profile),
                InternVideo2EmbeddingStage(
                    num_gpus_per_worker=args.embedding_gpus_per_worker,
                    batch_size=args.embedding_batch_size,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]
        )
    elif args.embedding_algorithm.startswith("cosmos-embed1-"):
        variant = args.embedding_algorithm.split("-")[-1]
        stages.extend(
            [
                CosmosEmbed1FrameCreationStage(
                    variant,
                    target_fps=2.0,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
                CosmosEmbed1EmbeddingStage(
                    variant,
                    num_gpus_per_worker=args.embedding_gpus_per_worker,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]
        )
    else:
        error_msg = f"{args.embedding_algorithm} embedding algorithm not implemented."
        raise NotImplementedError(error_msg)
    return stages


def split(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    zero_start = time.time()
    input_tasks, input_videos_relative, _ = build_input_data(args)

    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            VideoDownloader(
                input_path=args.input_video_path,
                input_s3_profile_name=args.input_s3_profile_name,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=args.num_download_workers_per_node,
            num_run_attempts_python=5,
        ),
        RemuxStage(
            verbose=args.verbose,
            log_stats=args.perf_profile,
        ),
    ]

    if args.splitting_algorithm == "fixed-stride":
        stages.append(
            CuratorStageSpec(
                FixedStrideExtractorStage(
                    clip_len_s=args.fixed_stride_split_duration,
                    clip_stride_s=args.fixed_stride_split_duration,
                    min_clip_length_s=args.fixed_stride_min_clip_length_s,
                    limit_clips=args.limit_clips,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
                num_workers_per_node=1,
            ),
        )
    elif args.splitting_algorithm == "transnetv2":
        # TransNetV2 is a neural-network based shot-detection algorithm
        # that takes strided windows of ~100 frames and detects whether
        # a given frame is a scene transition or not.
        # See https://arxiv.org/abs/2008.04838 for more details.
        stages.extend(
            [
                CuratorStageSpec(
                    VideoFrameExtractionStage(
                        decoder_mode=args.transnetv2_frame_decoder_mode,
                        num_cpus_per_worker=args.transnetv2_frame_decode_cpus_per_worker,
                        raise_on_pynvc_error_without_cpu_fallback=args.transnetv2_frame_decode_raise_on_pynvc_error,
                        verbose=args.verbose,
                        log_stats=args.perf_profile,
                    ),
                ),
                CuratorStageSpec(
                    TransNetV2ClipExtractionStage(
                        threshold=args.transnetv2_threshold,
                        min_length_s=args.transnetv2_min_length_s,
                        min_length_frames=args.transnetv2_min_length_frames,
                        max_length_s=args.transnetv2_max_length_s,
                        max_length_mode=args.transnetv2_max_length_mode,
                        crop_s=args.transnetv2_crop_s,
                        num_gpus_per_worker=args.transnetv2_gpus_per_worker,
                        limit_clips=args.limit_clips,
                        verbose=args.verbose,
                        log_stats=args.perf_profile,
                    ),
                    over_provision_factor=2.0,
                ),
                # TransNetV2ClipExtraction stage is generally so fast that it will be
                # scaled down to 1 worker total even in a multi-node run.
                # But because it operates on videos, which can vary from 30MB to 10GB,
                # the single worker can stuck on a long video and not produce tasks
                # for downstream stages for a while and cause starvation.
                # One better solution is to support specifying a minimum number of
                # workers, but I did not find an easy way to implement that.
                # But, if we are not running captioning, this will scale up.
            ],
        )
    else:
        error_msg = f"{args.splitting_algorithm} algorithm type not implemented."
        raise NotImplementedError(error_msg)

    stages.append(
        ClipTranscodingStage(
            num_cpus_per_worker=args.transcode_cpus_per_worker,
            encoder=args.transcode_encoder,
            encoder_threads=args.transcode_encoder_threads,
            encode_batch_size=args.transcode_ffmpeg_batch_size,
            use_hwaccel=args.transcode_use_hwaccel,
            use_input_bit_rate=args.transcode_use_input_video_bit_rate,
            num_clips_per_chunk=args.clip_re_chunk_size,
            verbose=args.verbose,
            log_stats=args.perf_profile,
        ),
    )

    if args.motion_filter != "disable":
        stages += [
            MotionVectorDecodeStage(
                num_cpus_per_worker=args.motion_decode_cpus_per_worker,
                verbose=args.verbose,
                log_stats=args.perf_profile,
                target_fps=args.motion_decode_target_fps,
                target_duration_ratio=args.motion_decode_target_duration_ratio,
            ),
            MotionFilterStage(
                score_only=args.motion_filter == "score-only",
                global_mean_threshold=args.motion_global_mean_threshold,
                per_patch_min_256_threshold=args.motion_per_patch_min_256_threshold,
                num_gpus_per_worker=args.motion_score_gpus_per_worker,
                batch_size=args.motion_score_batch_size,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
        ]

    has_embeddings = args.generate_embeddings
    has_aesthetics = args.aesthetic_threshold is not None
    target_fps: list[float | int] = (
        [1, 2] if has_aesthetics and has_embeddings else [1] if has_aesthetics else [2] if has_embeddings else []
    )
    if target_fps:
        stages += [
            ClipFrameExtractionStage(
                extraction_policies=(FrameExtractionPolicy.sequence,),
                target_fps=target_fps,
                target_res=(
                    args.clip_extraction_target_res,
                    args.clip_extraction_target_res,
                ),
                num_cpus_per_worker=args.clip_extraction_cpus_per_worker,
                log_stats=args.perf_profile,
            ),
        ]

    if args.aesthetic_threshold is not None:
        stages += [
            AestheticFilterStage(
                score_threshold=args.aesthetic_threshold,
                reduction=args.aesthetic_reduction,
                num_gpus_per_worker=args.aesthetic_gpus_per_worker,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
        ]

    if args.qwen_filter != "disable":
        stages += [
            QwenInputPreparationStageFiltering(
                model_variant=args.qwen_filter_model_variant,
                filter_categories=args.qwen_filter_categories,
                prompt_variant=args.qwen_filter_prompt_variant,
                sampling_fps=args.captioning_sampling_fps,
                window_size=args.captioning_window_size,
                remainder_threshold=args.captioning_remainder_threshold,
                preprocess_dtype=args.qwen_preprocess_dtype,
                model_does_preprocess=args.qwen_model_does_preprocess,
                generate_previews=args.generate_previews,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            CuratorStageSpec(
                QwenFilteringStage(
                    model_variant=args.qwen_filter_model_variant,
                    filter_variant=args.qwen_filter_prompt_variant,
                    rejection_threshold=args.qwen_filter_rejection_threshold,
                    user_prompt=args.qwen_filter_categories,
                    batch_size=args.qwen_filter_batch_size,
                    fp8_enable=args.qwen_filter_fp8_enable,
                    max_output_tokens=args.qwen_filter_max_output_tokens,
                    disable_mmcache=not args.qwen_use_vllm_mmcache,
                    score_only=args.qwen_filter == "score-only",
                    use_async_engine=args.qwen_use_async_engine,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ),
        ]

    embedding_model_version: str = "unspecified"
    if args.generate_embeddings:
        stages += get_embedding_stages(args)
        embedding_stage: CuratorStage = stages[-1].stage if isinstance(stages[-1], CuratorStageSpec) else stages[-1]  # type: ignore[assignment]
        assert embedding_stage.model is not None, "Embedding stage model should be set"
        embedding_model_id = embedding_stage.model.model_id_names[0]
        embedding_model_version = get_all_models_by_id().get(embedding_model_id, {}).get("version", "unspecified")  # type: ignore[assignment]
        logger.debug(f"Embedding model id={embedding_model_id} version={embedding_model_version}")

    caption_algo = args.captioning_algorithm.lower()
    keep_mp4 = args.generate_previews or (args.generate_cosmos_predict_dataset != "disable") or caption_algo == "gemini"

    if args.generate_captions:
        if caption_algo not in {"cosmos_r1", "gemini", "nemotron", "phi4", "qwen"}:
            msg = f"Unsupported captioning algorithm: {caption_algo}"
            raise RuntimeError(msg)

        vllm_config = VllmConfig(
            model_variant=args.captioning_algorithm,
            prompt_variant=args.captioning_prompt_variant,
            prompt_text=args.captioning_prompt_text,
            max_output_tokens=args.captioning_max_output_tokens,
            num_cpus_for_prepare=args.vllm_prepare_num_cpus_per_worker,
            max_retries=args.vllm_max_retries,
            copy_weights_to=pathlib.Path(args.copy_weights_to) if args.copy_weights_to else None,
        )

        window_config = WindowConfig(
            window_size=args.captioning_window_size,
            remainder_threshold=args.captioning_remainder_threshold,
            sampling_fps=args.captioning_sampling_fps,
            preprocess_dtype=args.qwen_preprocess_dtype,
            use_input_bit_rate=args.transcode_use_input_video_bit_rate,
        )

        if caption_algo == "qwen":
            vllm_config.batch_size = args.qwen_batch_size
            vllm_config.fp8 = args.qwen_use_fp8_weights
            vllm_config.disable_mmcache = not args.qwen_use_vllm_mmcache
            vllm_config.num_gpus = args.qwen_num_gpus_per_worker
            vllm_config.stage2_caption = args.qwen_stage2_caption
            vllm_config.stage2_prompt_text = args.qwen_stage2_prompt_text
            window_config.preprocess_dtype = args.qwen_preprocess_dtype
            window_config.model_does_preprocess = args.qwen_model_does_preprocess
        elif caption_algo == "phi4":
            vllm_config.stage2_caption = args.phi4_stage2_caption
            vllm_config.stage2_prompt_text = args.phi4_stage2_prompt_text
        elif caption_algo == "cosmos_r1":
            vllm_config.batch_size = args.qwen_batch_size
            vllm_config.fp8 = args.qwen_use_fp8_weights
            vllm_config.disable_mmcache = not args.qwen_use_vllm_mmcache
            vllm_config.num_gpus = args.qwen_num_gpus_per_worker
            vllm_config.stage2_caption = args.qwen_stage2_caption
            vllm_config.stage2_prompt_text = args.qwen_stage2_prompt_text
            window_config.preprocess_dtype = "float16"
            window_config.model_does_preprocess = False
        elif caption_algo == "gemini":
            window_config.model_does_preprocess = args.qwen_model_does_preprocess
        elif caption_algo == "nemotron":
            vllm_config.stage2_caption = args.nemotron_stage2_caption

        if caption_algo == "gemini":
            stages += [
                ApiPrepStage(
                    window_config=window_config,
                    model_variant=args.captioning_algorithm,
                    num_cpus_for_prepare=args.vllm_prepare_num_cpus_per_worker,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]
        else:
            assert vllm_config is not None
            stages += [
                VllmPrepStage(
                    vllm_config=vllm_config,
                    window_config=window_config,
                    keep_mp4=keep_mp4,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]

        # preview
        if args.generate_previews:
            stages += [
                PreviewStage(
                    target_fps=args.preview_target_fps,
                    target_height=args.preview_target_height,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]

        # captioning
        if caption_algo in {"cosmos_r1", "nemotron", "phi4", "qwen"}:
            stages += [
                CuratorStageSpec(
                    VllmCaptionStage(
                        vllm_config=vllm_config,
                        verbose=args.verbose,
                        keep_mp4=keep_mp4,
                        log_stats=args.perf_profile,
                        inflight_batching=args.vllm_use_inflight_batching,
                    ),
                    num_setup_attempts_python=2,
                ),
            ]
        elif caption_algo == "gemini":
            stages += [
                ApiCaptionStage(
                    model_variant=args.captioning_algorithm,
                    model_name=args.gemini_model_name,
                    prompt_variant=args.captioning_prompt_variant,
                    prompt_text=args.captioning_prompt_text,
                    max_output_tokens=args.captioning_max_output_tokens,
                    max_caption_retries=args.gemini_caption_retries,
                    retry_delay_seconds=args.gemini_retry_delay_seconds,
                    max_video_size_bytes=int(args.gemini_max_inline_mb * 1024 * 1024),
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                )
            ]

        # enhance caption
        if args.enhance_captions:
            stages += [
                EnhanceCaptionStage(
                    model_variant=args.enhance_captions_lm_variant,
                    batch_size=args.enhance_captions_batch_size,
                    azure_deployment=args.enhance_captions_azure_openai_deployment,
                    fp8_enable=args.qwen_lm_use_fp8_weights,
                    max_output_tokens=args.enhance_captions_max_output_tokens,
                    prompt_variant=args.enhance_captions_prompt_variant,
                    prompt_text=args.enhance_captions_prompt_text,
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ]

    if args.generate_cosmos_predict_dataset != "disable":
        # run T5 embedding on captions
        stages += [
            CuratorStageSpec(
                T5StageForSplit(
                    caption_fields=[args.captioning_algorithm],
                    verbose=args.verbose,
                    log_stats=args.perf_profile,
                ),
            ),
        ]

    stages.append(
        CuratorStageSpec(
            ClipWriterStage(
                output_path=args.output_clip_path,
                input_path=args.input_video_path,
                output_s3_profile_name=args.output_s3_profile_name,
                upload_clips=args.upload_clips,
                upload_clip_info_in_chunks=args.upload_clip_info_in_chunks,
                upload_cvds_parquet=args.upload_cvds_parquet,
                dry_run=args.dry_run,
                generate_embeddings=args.generate_embeddings,
                embedding_algorithm=args.embedding_algorithm,
                embedding_model_version=embedding_model_version,
                generate_previews=args.generate_previews,
                caption_models=[args.captioning_algorithm],
                enhanced_caption_models=[args.enhance_captions_lm_variant],
                generate_cosmos_predict_dataset=args.generate_cosmos_predict_dataset,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=args.num_clip_writer_workers_per_node,
            num_run_attempts_python=5,
        ),
    )

    pipeline_start = time.time()
    output_tasks: list[SplitPipeTask] = run_pipeline(
        input_tasks,
        stages,
        args.model_weights_path,
    )

    summary_start = time.time()

    pipeline_run_time = (summary_start - pipeline_start) / 60
    input_build_time = (pipeline_start - zero_start) / 60

    total_video_length = write_summary(args, input_videos_relative, output_tasks, pipeline_run_time)

    summary_run_time = (time.time() - summary_start) / 60

    logger.info(
        f"Split-Transcode-Filter-Annotate pipeline: {input_build_time=:.2f} / "
        f"{pipeline_run_time=:.2f} / {summary_run_time=:.2f} mins processing "
        f"time for {total_video_length=:.3f} hours of raw videos",
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:  # noqa: PLR0915
    """Set up the parser for the split pipeline.

    This function adds arguments to the parser for the split pipeline.

    Args:
        parser: The parser to add arguments to.

    """
    parser.add_argument(
        "--input-video-path",
        type=str,
        required=False,
        default=None,
        help=("S3 or local path which has input raw videos. Not required if --input-presigned-s3-url is provided."),
    )
    parser.add_argument(
        "--input-video-list-json-path",
        type=str,
        default=None,
        help="S3 or local path to a json with a list of specific videos under --input-video-path.",
    )
    parser.add_argument(
        "--input-video-list-s3-profile-name",
        type=str,
        default="default",
        help="S3 profile name to use for input_video_list_json_path.",
    )
    parser.add_argument(
        "--output-clip-path",
        type=str,
        required=False,
        default=None,
        help=(
            "S3 or local path to store output clips. "
            "If omitted and --output-presigned-s3-url is provided, a temporary directory will be used."
        ),
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="limit number of clips from each input video to process.",
    )
    parser.add_argument(
        "--no-generate-embeddings",
        dest="generate_embeddings",
        action="store_false",
        default=True,
        help="Whether to generate embeddings for clips.",
    )
    parser.add_argument(
        "--embedding-algorithm",
        type=str,
        default="internvideo2",
        choices=["cosmos-embed1-224p", "cosmos-embed1-336p", "cosmos-embed1-448p", "internvideo2"],
        help="Embedding algorithm to use.",
    )
    parser.add_argument(
        "--generate-previews",
        dest="generate_previews",
        action="store_true",
        default=False,
        help="Whether to generate previews for clip windows.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Whether to generate captions for clip windows.",
    )
    parser.add_argument(
        "--no-upload-clips",
        dest="upload_clips",
        action="store_false",
        default=True,
        help="Whether to upload clips to output path.",
    )
    parser.add_argument(
        "--upload-clip-info-in-chunks",
        dest="upload_clip_info_in_chunks",
        action="store_true",
        default=False,
        help=(
            "Whether to group clip metadata in chunks as jsonl and "
            "skip writing per-clip embedding pickles, i.e. grouped clip embeddings as parquet only."
        ),
    )
    parser.add_argument(
        "--upload-cvds-parquet",
        dest="upload_cvds_parquet",
        action="store_true",
        default=False,
        help="Whether to upload parquet files for CVDS.",
    )
    parser.add_argument(
        "--generate-cosmos-predict-dataset",
        choices=["disable", "predict2"],
        default="disable",
        help="Whether and how to generate Cosmos-PredictX post-training dataset.",
    )
    parser.add_argument(
        "--no-write-all-caption-json",
        dest="write_all_caption_json",
        action="store_false",
        default=True,
        help="Whether to write all captions to a single JSON file in the output path.",
    )
    parser.add_argument(
        "--splitting-algorithm",
        type=str,
        default="transnetv2",
        choices=["fixed-stride", "transnetv2"],
        help="Splitting algorithm to use on full videos.",
    )
    parser.add_argument(
        "--fixed-stride-split-duration",
        type=int,
        default=10,
        help="Duration of clips (in seconds) generated from the fixed stride splitting stage.",
    )
    parser.add_argument(
        "--fixed-stride-min-clip-length-s",
        type=float,
        default=2,
        help="Minimum length of clips (in seconds) for fixed stride splitting stage.",
    )
    parser.add_argument(
        "--transnetv2-frame-decoder-mode",
        choices=["ffmpeg_cpu", "ffmpeg_gpu", "pynvc"],
        default="ffmpeg_cpu",
        help="Choose between ffmpeg on CPU or GPU or PyNvVideoCodec for video decode.",
    )
    parser.add_argument(
        "--transnetv2-frame-decode-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPU threads per worker for video frame decoding when using ffmpeg_cpu mode.",
    )
    parser.add_argument(
        "--transnetv2-frame-decode-raise-on-pynvc-error",
        dest="transnetv2_frame_decode_raise_on_pynvc_error",
        action="store_true",
        default=False,
        help="Disable CPU ffmpeg fallback from PyNvVideoCodec and raise exception (for testing).",
    )
    parser.add_argument(
        "--transnetv2-threshold",
        type=float,
        default=0.4,
        help=(
            "TransNetV2 probability threshold above which a frame is classified as a shot transition. "
            "Default is 0.4, which prioritizes recall over precision."
        ),
    )
    parser.add_argument(
        "--transnetv2-min-length-s",
        type=float,
        default=2.0,
        help=(
            "Minimum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will remove any scenes below this length."
        ),
    )
    parser.add_argument(
        "--transnetv2-min-length-frames",
        type=int,
        default=48,
        help=(
            "Minimum length of clips (in frames) for TransNetV2 splitting stage. "
            "If specified, will remove any scenes below this length."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-s",
        type=float,
        default=60.0,
        help=(
            "Maximum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will deal with the scene by the `max_length_mode` specified."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-mode",
        type=str,
        default="stride",
        choices=["truncate", "stride"],
        help=(
            "Maximum length mode for TransNetV2 splitting stage. "
            "If `truncate`, will truncate the scene to `max_length_s`. "
            "If `stride`, will generate a number of max_length_s scenes until the end of the scene. "
            "If the end scene is less than `min_length_s`, it will drop the last scene."
        ),
    )
    parser.add_argument(
        "--transnetv2-crop-s",
        type=float,
        default=0.5,
        help=(
            "Crop size for TransNetV2 splitting stage. If specified, will crop each scene at start and end. "
            "E.g. 0.25 will crop ~250ms from start, and ~250ms from end frame (reducing all clips by ~0.5 seconds). "
            "If cropped scenes result in zero-length scenes, these will be filtered."
        ),
    )
    parser.add_argument(
        "--transnetv2-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker for TransNetV2 splitting stage.",
    )
    parser.add_argument(
        "--transcode-cpus-per-worker",
        type=float,
        default=5.0,
        help="Number of CPU threads per worker. The stage uses a batched ffmpeg "
        "commandline with batch_size (--transcode-ffmpeg-batch-size) of ~64 and per-batch thread count of 1.",
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "h264_nvenc"],
        help="Codec for transcoding clips; None to skip transocding.",
    )
    parser.add_argument(
        "--transcode-encoder-threads",
        type=int,
        default=1,
        help="Number of threads per ffmpeg encoding sub-command for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-ffmpeg-batch-size",
        type=int,
        default=16,
        help="FFMPEG batchsize for transcoding clips. Each clip/sub-command in "
        "the batch uses --transcode-encoder-threads number of CPU threads",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Whether to use cuda acceleration for decoding in transcoding stage.",
    )
    parser.add_argument(
        "--transcode-use-input-video-bit-rate",
        action="store_true",
        default=False,
        help="Whether to use input video's bit rate for encoding clips.",
    )
    parser.add_argument(
        "--clip-re-chunk-size",
        type=int,
        default=32,
        help="Number of clips per chunk after transcoding stage.",
    )
    parser.add_argument(
        "--motion-filter",
        choices=["disable", "enable", "score-only"],
        default="disable",
        help=(
            "Control motion filtering behavior:\n"
            "  - disable: No filtering or scoring.\n"
            "  - enable: Automatically filter clips based on motion thresholds.\n"
            "      (controlled by --motion-global-mean-threshold and --motion-per-patch-min-256-threshold).\n"
            "  - score-only: Calculate motion scores without filtering clips."
        ),
    )
    parser.add_argument(
        "--motion-global-mean-threshold",
        type=float,
        default=0.00098,
        help=(
            "Threshold for global average motion magnitude. "
            "Clips with global motion below this value may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-per-patch-min-256-threshold",
        type=float,
        default=0.000001,
        help=(
            "Threshold for minimal average motion magnitude in any 256x256-pixel patch. "
            "Clips containing patches below this threshold may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-decode-target-fps",
        type=float,
        default=2.0,
        help="Target frames per second to sample for motion vector decoding.",
    )
    parser.add_argument(
        "--motion-decode-target-duration-ratio",
        type=float,
        default=0.5,
        help="Target ratio of video duration to sample for motion vector decoding (0.5 = 50%%).",
    )
    parser.add_argument(
        "--motion-decode-cpus-per-worker",
        type=float,
        default=2.0,
        help="Number of CPUs per worker allocated to motion vector decoding.",
    )
    parser.add_argument(
        "--motion-score-batch-size",
        type=int,
        default=64,
        help="Batch size for motion score computation.",
    )
    parser.add_argument(
        "--motion-score-gpus-per-worker",
        type=float,
        default=0.5,
        help="Number of GPUs per worker allocated to motion score computation. Set to 0 to use CPU instead of GPU.",
    )
    parser.add_argument(
        "--clip-extraction-target-res",
        type=int,
        default=-1,
        help="Target resolution for clip extraction as (height, width). A value of -1 implies disables resize",
    )
    parser.add_argument(
        "--clip-extraction-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPUs per worker allocated to clip frame extraction.",
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=None,
        help="If specified (e.g. 3.5), filter out clips with an aesthetic score below this threshold.",
    )
    parser.add_argument(
        "--aesthetic-reduction",
        choices=[
            "mean",
            "min",
        ],
        default="min",
        help="Method to reduce the frame-level aesthetic scores.",
    )
    parser.add_argument(
        "--aesthetic-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker allocated to aesthetic filter.",
    )
    parser.add_argument(
        "--qwen-filter",
        choices=["enable", "disable", "score-only"],
        default="disable",
        help=(
            "Whether to enable Qwen-based content filtering for video clips.\n"
            "  - enable: Automatically filter clips based on Qwen-based content filtering.\n"
            "  - disable: Disable Qwen-based content filtering.\n"
            "  - score-only: Calculate Qwen-based content filtering results without filtering clips."
        ),
    )
    parser.add_argument(
        "--qwen-filter-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
        ],
        help="Prompt variant for Qwen filtering stage.",
    )
    parser.add_argument(
        "--qwen-filter-categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to filter out. If not provided, default categories will be filtered.",
    )
    parser.add_argument(
        "--qwen-filter-rejection-threshold",
        type=float,
        default=0.5,
        help="Threshold for Qwen filtering stage. If not provided, the default threshold of .5 will be used.",
    )
    parser.add_argument(
        "--qwen-filter-batch-size",
        type=int,
        default=16,
        help="Batch size for Qwen filtering stage.",
    )
    parser.add_argument(
        "--qwen-filter-model-variant",
        type=str,
        default="qwen",
        help="Model variant to use for Qwen filtering.",
    )
    parser.add_argument(
        "--qwen-filter-fp8-enable",
        action="store_true",
        default=False,
        help="Whether to use FP8 weights for Qwen filtering model.",
    )
    parser.add_argument(
        "--qwen-filter-max-output-tokens",
        type=int,
        default=512,
        help="Max number of output tokens for Qwen filtering model.",
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker for InternVideo2 or Cosmos-Embed1 embedding stage.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size for InternVideo2 embedding stage.",
    )
    parser.add_argument(
        "--captioning-algorithm",
        type=str,
        default="qwen",
        choices=["cosmos_r1", "gemini", "nemotron", "phi4", "qwen"],
        help="Captioning algorithm to use in annotation pipeline.",
    )
    parser.add_argument(
        "--captioning-window-size",
        type=int,
        default=256,
        help="Window size for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-remainder-threshold",
        type=int,
        default=128,
        help="Remainder threshold for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-text",
        type=str,
        default=None,
        help="Prompt text for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-sampling-fps",
        type=float,
        default=2.0,
        help="Controls number of frames sampled per second from input clip for captioning model",
    )
    parser.add_argument(
        "--captioning-max-output-tokens",
        type=int,
        default=512,
        help="Max number of output tokens requested from captioning model",
    )
    parser.add_argument(
        "--gemini-model-name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name used when --captioning-algorithm is 'gemini'.",
    )
    parser.add_argument(
        "--gemini-caption-retries",
        type=int,
        default=3,
        help="Max number of retries for Gemini caption requests.",
    )
    parser.add_argument(
        "--gemini-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between retries for Gemini caption requests.",
    )
    parser.add_argument(
        "--gemini-max-inline-mb",
        type=float,
        default=20.0,
        help="Maximum inline video size accepted by Gemini when captioning (in megabytes).",
    )
    parser.add_argument(
        "--qwen-preprocess-dtype",
        type=str,
        default="float16",
        choices=[
            "float32",
            "float16",
            "bfloat16",
            "uint8",
        ],
        help="Precision for tensor preprocess operations in QwenInputPreparationStage.",
    )
    parser.add_argument(
        "--qwen-model-does-preprocess",
        dest="qwen_model_does_preprocess",
        action="store_true",
        default=False,
        help="If set, Qwen will handle preprocessing (resize, rescale, normalize) instead of our code.",
    )
    parser.add_argument(
        "--qwen-stage2-caption",
        dest="qwen_stage2_caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into QwenVL to refine them",
    )
    parser.add_argument(
        "--qwen-stage2-prompt-text",
        type=str,
        default=None,
        help="Specify the input prompt used to generate stage2 Qwen captions",
    )
    parser.add_argument(
        "--qwen-batch-size",
        type=int,
        default=8,
        help="Batch size for Qwen captioning stage.",
    )
    parser.add_argument(
        "--qwen-use-vllm-mmcache",
        action="store_true",
        default=False,
        help="vLLM MultiModal Cache Usage, default disabled for better performance and GPU Utilization",
    )
    parser.add_argument(
        "--qwen-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for Qwen VL model or not.",
    )
    parser.add_argument(
        "--qwen-use-async-engine",
        action="store_true",
        default=False,
        help="Whether to use async engine for Qwen VL model or not.",
    )
    parser.add_argument(
        "--qwen-num-gpus-per-worker",
        type=float,
        default=1.0,
        help="Number of GPUs per worker for Qwen captioning stage.",
    )
    parser.add_argument(
        "--phi4-stage2-caption",
        dest="phi4_stage2_caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into Phi4 to refine them",
    )
    parser.add_argument(
        "--phi4-stage2-prompt-text",
        type=str,
        default=None,
        help="Specify the input prompt used to generate stage2 Phi4 captions",
    )
    parser.add_argument(
        "--vllm-prepare-num-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPUs per worker for VllmPrepStage.",
    )
    parser.add_argument(
        "--vllm-use-inflight-batching",
        type=int,
        default=1,
        help="Whether to use inflight batching for vLLM captioning stage.",
    )
    parser.add_argument(
        "--vllm-max-retries",
        type=int,
        default=3,
        help="Number of times to retry vLLM captioning failures",
    )
    parser.add_argument(
        "--copy-weights-to",
        type=str,
        default=None,
        help="Optional directory to copy model weights to before loading. "
        "Useful for copying weights to faster storage, like local NVME on compute nodes, "
        "and can reduce model load time. Common location is /raid/scratch/models.",
    )
    parser.add_argument(
        "--enhance-captions",
        dest="enhance_captions",
        action="store_true",
        default=False,
        help="Whether to further enhance captions with a language model",
    )
    parser.add_argument(
        "--enhance-captions-lm-variant",
        type=str,
        default="qwen_lm",
        choices=["qwen_lm", "gpt_oss_20b", "azure_openai"],
        help="Select language model for enhance captions stage.",
    )
    parser.add_argument(
        "--enhance-captions-azure-openai-deployment",
        type=str,
        default="gpt-5-chat-20250807",
        help="Azure OpenAI deployment name (only used when --enhance-captions-lm-variant is 'azure_openai').",
    )
    parser.add_argument(
        "--enhance-captions-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for enhanced captioning algorithm.",
    )
    parser.add_argument(
        "--enhance-captions-prompt-text",
        type=str,
        default=None,
        help="Prompt text for further enhancing captions using EnhanceCaptionStage.",
    )
    parser.add_argument(
        "--enhance-captions-max-output-tokens",
        type=int,
        default=2048,
        help="Max number of output tokens requested from the enhance captions model.",
    )
    parser.add_argument(
        "--enhance-captions-batch-size",
        type=int,
        default=32,
        help="Batch size for enhance captioning stage.",
    )
    parser.add_argument(
        "--qwen-lm-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for Qwen-LM model or not.",
    )
    parser.add_argument(
        "--preview-target-fps",
        type=int,
        default=1,
        help="Target FPS for preview generation.",
    )
    parser.add_argument(
        "--preview-target-height",
        type=int,
        default=240,
        help="Target height for preview generation.",
    )
    parser.add_argument(
        "--num-download-workers-per-node",
        type=int,
        default=4,
        help="Number of workers to use for downloading videos.",
    )
    parser.add_argument(
        "--num-clip-writer-workers-per-node",
        type=int,
        default=8,
        help="Number of workers to use for writing clips.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set only write minimum metadata",
    )
    parser.add_argument(
        "--input-presigned-s3-url",
        type=str,
        default=None,
        help="Presigned S3 URL pointing to a zip archive that contains the input videos.",
    )
    parser.add_argument(
        "--output-presigned-s3-url",
        type=str,
        default=None,
        help="Presigned S3 URL where the zipped output clips will be uploaded.",
    )
    parser.add_argument(
        "--nemotron-stage2-caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into Nemotron to refine them",
    )
    # add common args applicable to all pipelines
    add_common_args(parser)


def nvcf_run_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    cli_run_split(args)


def cli_run_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    split(args)


def add_split_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the split command to the parser.

    This function adds a subparser for the split command to the main parser.
    It sets up the parser with the appropriate arguments and default values.

    Args:
        subparsers: The subparsers action to add the parser to.

    """
    parser = subparsers.add_parser(
        "split",
        help="Split videos into clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=cli_run_split)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
