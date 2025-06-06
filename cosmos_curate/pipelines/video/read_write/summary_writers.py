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
"""Write metadata for clips to DB."""

import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import attrs
from loguru import logger

from cosmos_curate.core.utils import (
    grouping,
    storage_client,
    storage_utils,
)
from cosmos_curate.core.utils.misc.retry_utils import do_with_retries
from cosmos_curate.core.utils.runtime.performance_utils import (
    dump_and_write_perf_stats,
)
from cosmos_curate.core.utils.storage_utils import (
    get_files_relative,
    get_full_path,
    path_exists,
    read_json_file,
)
from cosmos_curate.core.utils.writer_utils import (
    write_csv,
    write_json,
)
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curate.pipelines.video.utils.data_model import (
    ShardPipeTask,
    SplitPipeTask,
)


@attrs.define
class ProcessedVideoMetadata:
    """Container for processed video metadata and clip chunk information.

    This class stores video metadata and a list of clip chunks that have been processed
    through the video processing pipeline.
    """

    video_metadata: dict[str, Any] | None = None
    clip_chunks: list[dict[str, Any]] = attrs.Factory(list)


def _worker_read_video_metadata(  # noqa: C901
    output_path: str,
    output_s3_profile_name: str,
    input_videos_relative: list[str],
    limit: int,
    *,
    verbose: bool = False,
) -> dict[str, ProcessedVideoMetadata]:
    all_video_data: dict[str, ProcessedVideoMetadata] = {}
    client = storage_utils.get_storage_client(output_path, profile_name=output_s3_profile_name)
    for input_video in input_videos_relative:
        video_metadata_path = get_full_path(
            ClipWriterStage.get_output_path_processed_videos(output_path),
            f"{input_video}.json",
        )

        def func_to_call(
            video_metadata_path: storage_client.StoragePrefix | pathlib.Path = video_metadata_path,
        ) -> dict[str, Any]:
            if verbose:
                logger.info(f"Reading video metadata from {video_metadata_path} ...")
            return read_json_file(video_metadata_path, client)

        all_video_data[input_video] = ProcessedVideoMetadata()
        if not path_exists(video_metadata_path, client):
            if limit == 0:
                logger.error(f"video process-record {input_video} not found ???")
        else:
            data = do_with_retries(func_to_call)
            all_video_data[input_video].video_metadata = data

            num_clip_chunks = data.get("num_clip_chunks", 0)
            if num_clip_chunks == 0:
                logger.warning(f"video process-record {input_video} has no clip")
                continue
            for idx in range(num_clip_chunks):
                clip_chunk_path = get_full_path(
                    ClipWriterStage.get_output_path_processed_clip_chunks(output_path),
                    f"{input_video}_{idx}.json",
                )

                def func_to_call2(
                    clip_chunk_path: storage_client.StoragePrefix | pathlib.Path = clip_chunk_path,
                ) -> dict[str, Any]:
                    if verbose:
                        logger.info(f"Reading clip chunk from {clip_chunk_path} ...")
                    return read_json_file(clip_chunk_path, client)

                if not path_exists(clip_chunk_path, client):
                    logger.error(f"clip chunk record {clip_chunk_path} not found ???")
                    continue

                all_video_data[input_video].clip_chunks.append(do_with_retries(func_to_call2))
    return all_video_data


def _write_split_result_summary(  # noqa: PLR0913
    input_path: str,
    input_videos_relative: list[str],
    output_path: str,
    output_s3_profile_name: str,
    limit: int,
    pipeline_run_time: float = 0.0,
) -> None:
    client_output = storage_utils.get_storage_client(
        output_path,
        profile_name=output_s3_profile_name,
        can_overwrite=True,
    )

    processed_videos = get_files_relative(ClipWriterStage.get_output_path_processed_videos(output_path), client_output)
    clip_stats_keys = [
        "num_clips_filtered_by_motion",
        "num_clips_filtered_by_aesthetic",
        "num_clips_passed",
        "num_clips_transcoded",
        "num_clips_with_embeddings",
        "num_clips_with_caption",
        "num_clips_with_webp",
    ]
    summary_data: dict[str, Any] = {
        "num_input_videos": len(input_videos_relative),
        "num_processed_videos": len(processed_videos),
        "total_video_duration": 0,
        "total_clip_duration": 0,
        "max_clip_duration": 0,
        "pipeline_run_time": pipeline_run_time,
    }
    for key in clip_stats_keys:
        summary_data[f"total_{key}"] = 0

    all_window_captions_data: dict[str, dict[str, str]] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures.extend(
            [
                executor.submit(
                    _worker_read_video_metadata,
                    output_path,
                    output_s3_profile_name,
                    input_video_chunk,
                    limit,
                )
                for input_video_chunk in grouping.split_into_n_chunks(input_videos_relative, 32)
            ],
        )
        for future in futures:
            for input_video, data in future.result().items():
                summary_data[input_video] = {
                    "source_video": str(get_full_path(input_path, input_video)),
                }
                if data.video_metadata is None:
                    summary_data[input_video]["processed"] = False
                    continue

                summary_data[input_video]["num_total_clips"] = data.video_metadata.get("num_total_clips", "N/A")
                summary_data[input_video]["clips"] = []
                summary_data[input_video]["filtered_clips"] = []
                for key in clip_stats_keys:
                    summary_data[input_video][key] = 0
                all_window_captions_data[input_video] = {}
                for clip_chunk in data.clip_chunks:
                    for key in clip_stats_keys:
                        summary_data[input_video][key] += clip_chunk.get(key, 0)
                        summary_data[f"total_{key}"] += clip_chunk.get(key, 0)
                    summary_data[input_video]["clips"].extend(clip_chunk.get("clips", []))
                    summary_data[input_video]["filtered_clips"].extend(clip_chunk.get("filtered_clips", []))
                    all_window_captions_data[input_video].update(clip_chunk.get("all_windows", {}))
                    summary_data["total_clip_duration"] += clip_chunk.get("total_clip_duration", 0)
                    summary_data["max_clip_duration"] = max(
                        summary_data["max_clip_duration"],
                        clip_chunk.get("max_clip_duration", 0),
                    )
                summary_data["total_video_duration"] += data.video_metadata.get("duration", 0)

    def func_write_summary() -> None:
        summary_dest = get_full_path(output_path, "summary.json")
        write_json(
            summary_data,
            summary_dest,
            "summary",
            "all videos",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )

    do_with_retries(func_write_summary)

    def func_write_captions() -> None:
        all_window_captions_dest = get_full_path(output_path, "v0", "all_window_captions.json")
        write_json(
            all_window_captions_data,
            all_window_captions_dest,
            "all window captions",
            "all videos",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )

    do_with_retries(func_write_captions)


def write_split_summary(  # noqa: PLR0913
    input_path: str,
    input_videos_relative: list[str],
    output_path: str,
    output_s3_profile_name: str,
    output_tasks: list[SplitPipeTask],
    limit: int,
    *,
    perf_profile: bool = False,
    pipeline_run_time: float = 0.0,
) -> None:
    """Write summary of split pipeline results including job stats and performance metrics.

    Args:
        input_path: Path to input videos.
        input_videos_relative: List of relative paths to input videos.
        output_path: Path to write output files.
        output_s3_profile_name: S3 profile name for output.
        output_tasks: List of completed split pipeline tasks.
        limit: Maximum number of videos to process.
        perf_profile: Whether to write performance statistics.
        pipeline_run_time: Total runtime of the pipeline in minutes.

    """
    # dump and write job summary
    _write_split_result_summary(
        input_path,
        input_videos_relative,
        output_path,
        output_s3_profile_name,
        limit,
        pipeline_run_time,
    )
    # dump and write performance stats
    if perf_profile:
        dump_and_write_perf_stats(
            [task.stage_perf for task in output_tasks],
            output_path,
            output_s3_profile_name,
        )


def write_shard_summary(  # noqa: PLR0913
    output_path: str,
    raw_output_path: str,
    output_s3_profile_name: str,
    all_bins: list[storage_client.StoragePrefix | pathlib.Path],
    max_tars_per_part: int,
    output_tasks: list[ShardPipeTask],
    *,
    perf_profile: bool = False,
) -> None:
    """Write summary of shard pipeline results including job stats and performance metrics.

    Args:
        output_path: Path to write output files.
        raw_output_path: Path to write raw output files.
        output_s3_profile_name: S3 profile name for output.
        all_bins: List of storage prefixes for all bins.
        max_tars_per_part: Maximum number of tars per part.
        output_tasks: List of completed shard pipeline tasks.
        perf_profile: Whether to write performance statistics.

    """
    # dump and write job summary
    client = storage_utils.get_storage_client(output_path, profile_name=output_s3_profile_name)
    # get per-bin total key count
    key_counts = {}
    for task in output_tasks:
        if task.bin_path not in key_counts:
            key_counts[task.bin_path] = 0
        key_counts[task.bin_path] += task.key_count
    # write wdinfo.json for each bin
    all_wdinfo = []
    for lbin in all_bins:
        bin_prefix = lbin.prefix if isinstance(lbin, storage_client.StoragePrefix) else str(lbin)
        dest = get_full_path(lbin, "wdinfo.json")
        video_path = str(get_full_path(lbin, "video"))
        data = {
            "data_keys": ["video", "t5_xxl", "metas"],
            "chunk_size": max_tars_per_part,
            "data_list": [str(x) for x in get_files_relative(video_path, client)],
            "root": bin_prefix,
            "total_key_count": key_counts[str(lbin)],
        }

        def write_wdinfo(
            data: dict = data,  # type: ignore[type-arg]
            dest: storage_client.StoragePrefix | pathlib.Path = dest,
            lbin: storage_client.StoragePrefix | pathlib.Path = lbin,
        ) -> None:
            write_json(data, dest, "wdinfo", str(lbin), verbose=True, client=client)

        do_with_retries(write_wdinfo)
        all_wdinfo.append([str(dest)])

    # write a top-level csv
    def write_wdinfo_list() -> None:
        summary_dest = get_full_path(output_path, "wdinfo_list.csv")
        write_csv(
            summary_dest,
            "wdinfo list",
            "all bins",
            all_wdinfo,
            verbose=True,
            client=client,
        )

    do_with_retries(write_wdinfo_list)
    # dump and write performance stats
    if perf_profile:
        dump_and_write_perf_stats(
            [task.stage_perf for task in output_tasks],
            raw_output_path,
            output_s3_profile_name,
        )
