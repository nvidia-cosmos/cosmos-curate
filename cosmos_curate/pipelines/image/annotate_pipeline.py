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

"""Image annotate pipeline: load images and write to output (load → write)."""

import argparse
import time
from typing import Any

from loguru import logger

from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.core.utils.infra.performance_utils import dump_and_write_perf_stats
from cosmos_curate.core.utils.misc.retry_utils import do_with_retries
from cosmos_curate.core.utils.storage.storage_utils import (
    create_path,
    get_full_path,
    get_storage_client,
    is_path_nested,
    verify_path,
)
from cosmos_curate.core.utils.storage.writer_utils import write_json
from cosmos_curate.pipelines.image.captioning.captioning_builders import (
    IMAGE_CAPTION_ALGOS,
    ImageCaptioningConfig,
    build_image_captioning_stages,
)
from cosmos_curate.pipelines.image.read_write.image_writer_stage import get_image_output_id
from cosmos_curate.pipelines.image.read_write.read_write_builders import (
    ImageIngestConfig,
    ImageOutputConfig,
    build_image_ingest_stages,
    build_image_output_stages,
)
from cosmos_curate.pipelines.image.utils.data_model import ImagePipeTask
from cosmos_curate.pipelines.image.utils.image_pipe_input import extract_image_tasks
from cosmos_curate.pipelines.pipeline_args import add_common_args


def write_summary(
    args: argparse.Namespace,
    num_tasks: int,
    output_tasks: list[ImagePipeTask],
    pipeline_run_time_min: float,
) -> None:
    """Write summary.json and optionally per-stage performance stats to terminal."""
    num_with_caption = sum(1 for t in output_tasks if t.image.has_caption())
    # Use prep-stage defaults when not set via CLI so summary reflects actual values used
    _default_min = 128 * 28 * 28
    _default_max = 768 * 28 * 28
    resize_min_pixels = getattr(args, "caption_prep_min_pixels", None)
    resize_max_pixels = getattr(args, "caption_prep_max_pixels", None)
    if resize_min_pixels is None:
        resize_min_pixels = _default_min
    if resize_max_pixels is None:
        resize_max_pixels = _default_max
    captioned_images = [get_image_output_id(t.session_id) for t in output_tasks if t.image.has_caption()]

    summary_data: dict[str, Any] = {
        "num_input_images": num_tasks,
        "num_output_tasks": len(output_tasks),
        "pipeline_run_time": round(pipeline_run_time_min, 4),
        "num_images_with_caption": num_with_caption,
        "resize_min_pixels": resize_min_pixels,
        "resize_max_pixels": resize_max_pixels,
        "captioned_images": captioned_images,
    }

    client_output = get_storage_client(
        args.output_path,
        profile_name=args.output_s3_profile_name,
        can_overwrite=True,
    )

    def func_write_summary() -> None:
        summary_dest = get_full_path(args.output_path, "summary.json")
        write_json(
            summary_data,
            summary_dest,
            "summary",
            "all images",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )
        logger.info(f"Wrote summary to {summary_dest}")

    do_with_retries(func_write_summary)

    if args.perf_profile and output_tasks:
        dump_and_write_perf_stats(
            [t.stage_perf for t in output_tasks],
            args.output_path,
            args.output_s3_profile_name,
        )


def build_input_data(args: argparse.Namespace) -> tuple[list[ImagePipeTask], int]:
    """Build input tasks for the image pipeline.

    Validates paths, creates output directory, and discovers image files.

    Args:
        args: Parsed CLI namespace (must have input_image_path, output_path, limit, etc.).

    Returns:
        (list of ImagePipeTask, number of tasks).

    """
    verify_path(args.input_image_path)
    verify_path(args.output_path, level=1)
    create_path(args.output_path)
    if is_path_nested(args.input_image_path, args.output_path):
        msg = "Do not make input and output paths nested"
        raise ValueError(msg)

    tasks = extract_image_tasks(
        args.input_image_path,
        args.input_s3_profile_name,
        limit=args.limit,
        output_path_and_profile=(args.output_path, args.output_s3_profile_name),
        verbose=args.verbose,
    )
    n = len(tasks)
    logger.info(f"About to process {n} image(s) ...")
    return tasks, n


def _assemble_stages(args: argparse.Namespace) -> list[CuratorStage | CuratorStageSpec]:
    """Build the image stage list via the current builder architecture."""
    stages: list[CuratorStage | CuratorStageSpec] = []
    stages.extend(
        build_image_ingest_stages(
            ImageIngestConfig(
                input_path=args.input_image_path,
                input_s3_profile_name=args.input_s3_profile_name,
                num_workers_per_node=args.num_ingest_workers_per_node,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )
    if getattr(args, "generate_captions", True):
        stages.extend(
            build_image_captioning_stages(
                ImageCaptioningConfig(
                    caption_algo=args.captioning_algorithm,
                    num_gpus=args.caption_num_gpus,
                    num_prep_workers_per_node=args.num_caption_prep_workers_per_node,
                    batch_size=args.caption_batch_size,
                    max_output_tokens=args.caption_max_output_tokens,
                    prompt_variant=args.caption_prompt_variant,
                    prompt_text=args.caption_prompt_text or None,
                    stage2_caption=False,
                    stage2_prompt_text=None,
                    caption_prep_min_pixels=getattr(args, "caption_prep_min_pixels", None),
                    caption_prep_max_pixels=getattr(args, "caption_prep_max_pixels", None),
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )
    stages.extend(
        build_image_output_stages(
            ImageOutputConfig(
                output_path=args.output_path,
                output_s3_profile_name=args.output_s3_profile_name,
                num_workers_per_node=args.num_output_workers_per_node,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )
    return stages


def annotate(args: argparse.Namespace) -> None:
    """Run the image annotate pipeline (load → write)."""
    zero_start = time.time()
    input_tasks, num_tasks = build_input_data(args)
    if num_tasks == 0:
        logger.warning("No images to process; exiting.")
        return

    stages = _assemble_stages(args)
    pipeline_start = time.time()
    output_tasks: list[ImagePipeTask] = run_pipeline(
        input_tasks,
        stages,
        args.model_weights_path,
        args=args,
    )
    pipeline_run_time_min = (time.time() - pipeline_start) / 60
    total_elapsed = (time.time() - zero_start) / 60
    write_summary(args, num_tasks, output_tasks, pipeline_run_time_min=pipeline_run_time_min)
    logger.info(
        f"Image annotate pipeline: {pipeline_run_time_min:.2f} min processing, {total_elapsed:.2f} min total "
        f"for {num_tasks} image(s), {len(output_tasks)} task(s) returned."
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    """Add image annotate arguments to the parser."""
    parser.add_argument(
        "--input-image-path",
        type=str,
        required=True,
        help="Local or S3 path to a directory of input images.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local or S3 path for output (images/ and metas/ created under this).",
    )
    parser.add_argument(
        "--num-ingest-workers-per-node",
        type=int,
        default=4,
        help="Number of image load workers per node.",
    )
    parser.add_argument(
        "--num-output-workers-per-node",
        type=int,
        default=8,
        help="Number of image writer workers per node.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Skip captioning (load and write only).",
    )
    parser.add_argument(
        "--captioning-algorithm",
        type=str,
        default="qwen",
        choices=sorted(IMAGE_CAPTION_ALGOS),
        help="Captioning algorithm for images (vLLM image-capable models).",
    )
    parser.add_argument(
        "--caption-num-gpus",
        type=int,
        default=1,
        help="GPUs per node for caption stage.",
    )
    parser.add_argument(
        "--num-caption-prep-workers-per-node",
        type=int,
        default=2,
        help="Workers per node for caption prep stage.",
    )
    parser.add_argument(
        "--caption-prep-min-pixels",
        type=int,
        default=None,
        metavar="N",
        help="Min total pixels for prep resize (default: video-style 128*28*28).",
    )
    parser.add_argument(
        "--caption-prep-max-pixels",
        type=int,
        default=None,
        metavar="N",
        help="Max total pixels for prep resize (default: video-style 768*28*28).",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM caption stage.",
    )
    parser.add_argument(
        "--caption-max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for caption generation.",
    )
    parser.add_argument(
        "--caption-prompt-variant",
        type=str,
        default="image",
        help="Prompt variant for captioning (e.g. 'image', 'default').",
    )
    parser.add_argument(
        "--caption-prompt-text",
        type=str,
        default=None,
        help="Custom prompt text for captioning (overrides prompt variant).",
    )
    add_common_args(parser)


def add_annotate_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the annotate subcommand on the given subparsers."""
    parser = subparsers.add_parser(
        "annotate",
        help="Load images and write to output (images/ + metas/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=annotate)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
