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
"""Ray pipeline.

Which:
  - Downloads videos along with trajectory information
  - Captions the clips with the trajectory information
  - Writes the captions
"""

import argparse
import time

from loguru import logger

from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.core.utils import args_utils
from cosmos_curate.core.utils.storage_utils import (
    create_path,
    get_files_relative,
    get_full_path,
    get_storage_client,
    path_exists,
    verify_path,
)
from cosmos_curate.pipelines.pipeline_args import add_common_args
from cosmos_curate.pipelines.video.captioning.captioning_stages import (
    QwenCaptionStage,
    QwenInputPreparationStage,
)
from cosmos_curate.pipelines.video.captioning.trajectory_caption_stages import ReadClipArchive
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curate.pipelines.video.read_write.summary_writers import write_split_summary
from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask, Video, VideoMetadata


def _get_archives(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    output_archive_path: str,
    input_s3_profile_name: str,
    output_s3_profile_name: str,
    limit: int = 0,
    *,
    verbose: bool = False,
) -> list[Video]:
    """Find the list of archives we need to run the job over."""
    client_input = get_storage_client(target_path=input_path, profile_name=input_s3_profile_name)
    client_output = get_storage_client(target_path=output_path, profile_name=output_s3_profile_name)
    # check output path
    if output_path is not None and path_exists(get_full_path(output_path, "summary.json"), client_output):
        logger.warning(f"Output path {output_path} already concluded with a summary.json file.")
    # input
    raw_archives = get_files_relative(input_path, client_input)
    raw_archives = [x for x in raw_archives if x.endswith(".tar")]
    if verbose:
        logger.info(f"Found {len(raw_archives)} input clip archives in {input_path}:")
        for archive in raw_archives:
            logger.info(archive)
    # filtered out already processed archives
    if output_archive_path is not None:
        processed_archives = get_files_relative(output_archive_path, client_output)
        raw_archives = [x for x in raw_archives if f"{x}.json" not in processed_archives]
        if verbose:
            logger.info(f"Found {len(processed_archives)} processed archives in {output_archive_path}:")
            for archive in processed_archives:
                logger.info(archive)

    # apply limit
    if limit > 0:
        raw_archives = raw_archives[:limit]

    # we're slightly abusing the data structure by stuffing the .tar path into the "video"
    metadata = VideoMetadata(duration=0)
    return [Video(get_full_path(input_path, x), metadata=metadata) for x in raw_archives]


def trajectory_caption(args: argparse.Namespace) -> None:
    """Run the trajectory caption pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, retrieves archives, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    start_time = time.time()
    # validate input arguments
    verify_path(args.input_archive_path)
    # create tasks
    archives = _get_archives(
        args.input_archive_path,
        args.output_caption_path,
        ClipWriterStage.get_output_path_processed_videos(args.output_caption_path),
        args.input_s3_profile_name,
        args.output_s3_profile_name,
        args.limit,
        verbose=args.verbose,
    )
    tasks = [SplitPipeTask(archive) for archive in archives]
    logger.info(f"About to process {len(tasks)} raw clip archives ...")
    if args.verbose:
        logger.info("\n".join(str(x.video.input_video) for x in tasks))

    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(
            ReadClipArchive(
                input_path=args.input_archive_path,
                input_s3_profile_name=args.input_s3_profile_name,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=4,
            num_run_attempts_python=5,
        ),
    ]

    if args.generate_captions:
        stages += [
            QwenInputPreparationStage(
                prompt_variant="alpamayo",
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            QwenCaptionStage(
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
        ]

    verify_path(args.output_caption_path, level=1)
    create_path(args.output_caption_path)
    stages.append(
        CuratorStageSpec(
            ClipWriterStage(
                output_path=args.output_caption_path,
                input_path=args.input_archive_path,
                output_s3_profile_name=args.output_s3_profile_name,
                upload_clips=True,
                dry_run=False,
                generate_embeddings=False,
                generate_previews=False,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            ),
            num_workers_per_node=8,
            num_run_attempts_python=5,
        ),
    )

    output_tasks: list[SplitPipeTask] = run_pipeline(tasks, stages)
    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    write_split_summary(
        args.input_archive_path,
        args.output_caption_path,
        args.input_s3_profile_name,
        args.output_s3_profile_name,
        output_tasks,
        args.limit,
        perf_profile=args.perf_profile,
    )

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Trajectory Caption pipeline completed in {elapsed_time:.2f} minutes")


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-archive-path",
        type=str,
        required=True,
        help="S3 or local path which has input archives of clips and trajectory information",
    )
    parser.add_argument(
        "--output-caption-path",
        type=str,
        required=True,
        help="S3 or local path to store output captions.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Whether to generate captions for clip windows.",
    )

    # add common args applicable to all pipelines
    add_common_args(parser)


def run_trajectory_caption(args: argparse.Namespace) -> None:
    """Run the trajectory caption pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, retrieves archives, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    trajectory_caption(args)


def add_trajectory_caption_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the trajectory caption command to the parser.

    This function adds a subparser for the trajectory caption command to the main parser.
    It sets up the parser with the appropriate arguments and default values.

    Args:
        subparsers: The subparsers action to add the parser to.

    Returns:
        The parser with the trajectory caption command added.

    """
    parser = subparsers.add_parser(
        "trajectory-caption",
        help="Caption clips with trajectory information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=run_trajectory_caption)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
