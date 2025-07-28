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

"""Video Pipe Input."""

import pathlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger
from six import BytesIO

from cosmos_curate.core.utils.misc import filter_predicates
from cosmos_curate.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curate.core.utils.storage.storage_utils import (
    get_files_relative,
    get_full_path,
    get_storage_client,
    path_exists,
    read_bytes,
    read_json_file,
)
from cosmos_curate.pipelines.video.utils.data_model import ClipSample, Video


def _check_output_path(output_path: str, client: StorageClient | None) -> None:
    if output_path is not None and path_exists(get_full_path(output_path, "summary.json"), client):
        logger.warning(f"Output path {output_path} already concluded with a summary.json file")


def _find_processed_video_jsons(
    output_video_path: str,
    client: StorageClient | None,
    *,
    verbose: bool,
) -> list[str]:
    processed_video_jsons = get_files_relative(output_video_path, client)
    logger.info(f"Found {len(processed_video_jsons)} processed videos in {output_video_path}")
    if verbose:
        for video in processed_video_jsons:
            logger.debug(video)
    return processed_video_jsons


def _worker_verify_processed_video(
    processed_video_json: str,
    output_video_path: str,
    output_clip_chunk_path: str,
    client: StorageClient | None,
) -> str | None:
    # read the processed video json
    processed_video_json_path = get_full_path(output_video_path, processed_video_json)
    try:
        data = read_json_file(processed_video_json_path, client)
        num_clip_chunks = int(data["num_clip_chunks"])
        for idx in range(num_clip_chunks):
            processed_clip_chunk_path = processed_video_json.removesuffix(".json") + f"_{idx}.json"
            clip_chunk_path = get_full_path(output_clip_chunk_path, processed_clip_chunk_path)
            if not path_exists(clip_chunk_path, client):
                logger.debug(f"Semi-processed video {processed_video_json} missing chunk-{idx}")
                return None
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read processed video json {processed_video_json_path}: {e}")
        return None
    else:
        return processed_video_json


def _read_video_list_json(
    input_path: str,
    input_video_list_json_path: str,
    input_video_list_s3_profile_name: str,
) -> list[str]:
    input_videos = []
    client = get_storage_client(input_video_list_json_path, profile_name=input_video_list_s3_profile_name)
    try:
        data = read_json_file(input_video_list_json_path, client)
        listed_input_videos = [str(x) for x in data]
    except Exception as e:
        logger.exception(f"Failed to read input video list from {input_video_list_json_path}: {e}")
        raise

    for video_path in listed_input_videos:
        _input_path = input_path.rstrip("/") + "/"
        if not video_path.startswith(_input_path):
            error_msg = f"Input video {video_path} is not in {_input_path}"
            logger.exception(error_msg)
            raise ValueError(error_msg)
        input_videos.append(video_path[len(_input_path) :])

    return input_videos


def extract_split_tasks(  # noqa: PLR0913
    input_path: str,
    input_video_list_json_path: str | None,
    output_path: str,
    output_video_path: str,
    output_clip_chunk_path: str,
    input_s3_profile_name: str,
    input_video_list_s3_profile_name: str,
    output_s3_profile_name: str,
    limit: int = 0,
    *,
    verbose: bool = False,
    filter_files_func: Callable[[str], bool] | None = None,
) -> tuple[list[Video], list[str], int]:
    """Extract list of input video paths from the input S3 or local path."""
    client_input = get_storage_client(input_path, profile_name=input_s3_profile_name)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)

    _check_output_path(output_path, client_output)

    # find already processed videos
    processed_video_jsons = _find_processed_video_jsons(output_video_path, client_output, verbose=verbose)
    # verify if all chunks are processed
    fully_processed_video_jsons = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _worker_verify_processed_video,
                processed_video_json,
                output_video_path,
                output_clip_chunk_path,
                client_output,
            )
            for processed_video_json in processed_video_jsons
        ]
        for future in futures:
            processed_video_json = future.result()
            if processed_video_json is not None:
                fully_processed_video_jsons.append(processed_video_json)
    # build the final set of processed videos
    processed_videos = {x.removesuffix(".json") for x in fully_processed_video_jsons}
    logger.info(f"Found {len(processed_videos)} fully processed videos in {output_video_path}")

    # input
    if input_video_list_json_path is not None:
        input_videos = _read_video_list_json(input_path, input_video_list_json_path, input_video_list_s3_profile_name)
    else:
        _limit = 0 if limit == 0 else len(processed_videos) + limit
        input_videos = get_files_relative(input_path, client_input, _limit)

    # apply filter func
    _filter_files_func = filter_predicates.accept if filter_files_func is None else filter_files_func
    all_videos = [video for video in input_videos if _filter_files_func(video)]
    logger.info(f"Found {len(all_videos)} input videos in {input_path}")
    if verbose:
        for video in all_videos:
            logger.debug(video)

    # remove already processed videos
    raw_videos = [x for x in all_videos if x not in processed_videos]
    # apply limit
    if limit > 0:
        raw_videos = raw_videos[:limit]
    # prepare the final list of videos
    return (
        [Video(get_full_path(input_path, x)) for x in raw_videos],
        all_videos,
        len(processed_videos),
    )


def extract_shard_tasks(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    input_s3_profile_name: str,
    output_s3_profile_name: str,
    version: str,
    *,
    verbose: bool = False,
) -> list[ClipSample]:
    """Extract list of clip paths from the input S3 or local path."""
    clip_samples = []
    client_input = get_storage_client(input_path, profile_name=input_s3_profile_name)
    client_output = get_storage_client(output_path, profile_name=output_s3_profile_name)
    # TODO: support fail-restart
    # verify output path is empty
    objects_in_output = get_files_relative(output_path, client_output)
    if len(objects_in_output) > 0:
        error_msg = f"Expect output path {output_path} to be empty"
        raise ValueError(error_msg)
    # extract clip metadata paths
    metadata_path = get_full_path(input_path, "metas", version)
    items = get_files_relative(str(metadata_path), client_input)
    logger.info(f"Reading {len(items)} clip metadata from {metadata_path} ...")
    for item in items:
        if not item:
            continue
        clip_metadata_path = get_full_path(metadata_path, item)
        try:
            clip_metadata = read_json_file(clip_metadata_path, client_input)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read clip metadata from {clip_metadata_path}")
            logger.error(e)
            continue
        if clip_metadata["valid"]:
            clip_samples.append(
                ClipSample(
                    uuid=str(clip_metadata["span_uuid"]),
                    width=clip_metadata["width"],
                    height=clip_metadata["height"],
                    num_frames=clip_metadata["num_frames"],
                    num_bytes=clip_metadata["num_bytes"],
                    clip_location=get_full_path(clip_metadata["clip_location"]),
                    clip_metadata=clip_metadata,
                ),
            )
        else:
            logger.warning(f"Clip {clip_metadata['span_uuid']} is invalid, skipping ...")
            if verbose:
                logger.warning(clip_metadata)
    return clip_samples


def is_parquet_file(filename: str) -> bool:
    """Check if a file is a parquet file."""
    return filename.lower().endswith(".parquet")


def _load_parquet_ids(
    client: StorageClient | None,
    dedup_path: str | StoragePrefix | pathlib.Path,
    filter_threshold: float | None = None,
) -> tuple[set[str], int, int]:
    """Load IDs from parquet files, optionally applying a filter threshold."""
    ids_to_remove_set = set()
    total_records = 0
    parquet_files_count = 0

    for item in get_files_relative(str(dedup_path), client):
        if not is_parquet_file(item):
            continue

        parquet_file = get_full_path(dedup_path, item)
        bytes_data = read_bytes(parquet_file, client)
        dedup_df = pd.read_parquet(BytesIO(bytes_data))
        total_records += len(dedup_df)

        if filter_threshold is not None:
            dedup_df = dedup_df.loc[dedup_df["cosine_sim_score"] >= filter_threshold]

        ids_to_remove_set.update(dedup_df["id"].tolist())
        parquet_files_count += 1

    return ids_to_remove_set, total_records, parquet_files_count


def filter_shard_tasks_by_semantic_dedup(
    clip_samples: list[ClipSample],
    input_semantic_dedup_path: str,
    input_semantic_dedup_s3_profile_name: str,
    semantic_dedup_epsilon: float,
    *,
    verbose: bool = False,
) -> list[ClipSample]:
    """Filter out clip samples based on semantic dedup results."""
    client = get_storage_client(
        input_semantic_dedup_path,
        profile_name=input_semantic_dedup_s3_profile_name,
    )

    # Attempt to read filtered results first
    formatted_epsilon = f"{semantic_dedup_epsilon:.5f}".rstrip("0").rstrip(".")
    dedup_filtered_path = get_full_path(
        input_semantic_dedup_path,
        "extraction",
        f"unique_ids_{formatted_epsilon}.parquet",
    )
    ids_to_remove_set, total_records, parquet_files_count = _load_parquet_ids(client, dedup_filtered_path)

    if parquet_files_count > 0:
        if verbose:
            logger.info(f"Using filtered results from {dedup_filtered_path}")
            logger.info(f"Loaded {total_records} filtered clips from {parquet_files_count} files.")
    else:
        # Fall back to raw results and filter
        dedup_raw_path = get_full_path(input_semantic_dedup_path, "extraction", "semdedup_pruning_tables")
        if verbose:
            logger.info(f"Filtered parquet not found; falling back to raw results from {dedup_raw_path}")

        similarity_threshold = 1 - semantic_dedup_epsilon
        ids_to_remove_set, total_records, parquet_files_count = _load_parquet_ids(
            client,
            dedup_raw_path,
            filter_threshold=similarity_threshold,
        )

        if parquet_files_count == 0:
            logger.warning(f"No valid parquet files found in {dedup_raw_path}")
            return clip_samples

        if verbose:
            logger.info(f"Processed {parquet_files_count} parquet files with {total_records} records.")
            logger.info(f"Removing {len(ids_to_remove_set)} clips after filtering (epsilon={semantic_dedup_epsilon}).")

    filtered_clip_samples = [sample for sample in clip_samples if sample.uuid not in ids_to_remove_set]

    if verbose:
        logger.info(
            f"Filtered from {len(clip_samples)} to {len(filtered_clip_samples)} clip samples "
            f"after applying semantic dedup from {input_semantic_dedup_path}.",
        )

    return filtered_clip_samples
