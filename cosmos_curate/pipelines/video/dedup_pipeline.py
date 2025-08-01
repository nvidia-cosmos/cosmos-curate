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
"""Deduplication pipeline for video clips."""

import argparse
import os
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from cosmos_curate.core.utils.config import args_utils
from cosmos_curate.core.utils.model.conda_utils import is_running_in_env
from cosmos_curate.core.utils.storage.azure_client import get_azure_client_config, is_azure_path
from cosmos_curate.core.utils.storage.s3_client import get_s3_client_config, is_s3path

if is_running_in_env("text-curator") or TYPE_CHECKING:
    import dask
    import dask_cudf  # type: ignore[import-untyped]
    from nemo_curator.datasets import DocumentDataset  # type: ignore[import-not-found]
    from nemo_curator.modules import (  # type: ignore[import-not-found]
        ClusteringModel,
        SemanticClusterLevelDedup,
        SemDedupConfig,
    )
    from nemo_curator.utils.distributed_utils import get_client  # type: ignore[import-not-found]


def setup_clustering_semantic(
    id_column: str,
    embedding_column: str,  # noqa: ARG001
    semantic_config: "SemDedupConfig",
    output_dir: str,
    storage_options: dict[str, Any] | None = None,
) -> tuple["ClusteringModel", "SemanticClusterLevelDedup"]:
    """Set up clustering and semantic deduplication models.

    Args:
        id_column: Name of the ID column.
        embedding_column: Name of the embedding column.
        semantic_config: Configuration for semantic deduplication.
        output_dir: Directory to store output files.
        storage_options: Optional storage configuration.

    Returns:
        Tuple of (clustering model, semantic deduplication model).

    """
    cluster = ClusteringModel(
        # column args
        id_column=id_column,
        embedding_column=semantic_config.embedding_column,
        # configureable args
        max_iter=semantic_config.max_iter,
        n_clusters=semantic_config.n_clusters,
        random_state=semantic_config.random_state,
        # i/o args
        storage_options=storage_options,
        clustering_output_dir=str(os.path.join(output_dir, semantic_config.clustering_save_loc)),  # noqa: PTH118
        clustering_input_partition_size=semantic_config.clustering_input_partition_size,
        logger="./",
        profile_dir=None,
    )

    semantic = SemanticClusterLevelDedup(
        # configureable args
        sim_metric=semantic_config.sim_metric,
        which_to_keep=semantic_config.which_to_keep,
        n_clusters=semantic_config.n_clusters,
        # i/o args
        # TODO: get rid of so many directory variables
        emb_by_clust_dir=str(os.path.join(output_dir, semantic_config.clustering_save_loc, "embs_by_nearest_center")),  # noqa: PTH118
        output_dir=str(os.path.join(output_dir, "extraction")),  # noqa: PTH118
        storage_options=storage_options,
        # column args
        id_column=id_column,
        embedding_column=semantic_config.embedding_column,
        # no-op args
        logger="./",
        profile_dir=None,
    )
    return cluster, semantic


def start_dask_client(args: argparse.Namespace) -> Any:  # noqa: ANN401
    """Start a Dask client.

    Args:
        args: Command line arguments.

    Returns:
        Dask client.

    """
    dask_client = get_client(scheduler_file=args.scheduler_file) if args.scheduler_file else get_client("gpu")

    logger.info(f"Dask client started with {dask_client.dashboard_link}")
    return dask_client


def setup_dask_and_run_semantic_dedup(args: argparse.Namespace) -> None:
    """Set up Dask and run semantic deduplication.

    Args:
        args: Command line arguments.

    Raises:
        OSError: If the environment is not text-curator.

    """
    if not is_running_in_env("text-curator"):
        msg = "Deduplication pipeline is only supported in text-curator environment."
        raise OSError(msg)

    # Start dask
    dask_client = start_dask_client(args)
    # Run dedup
    try:
        _run_semantic_dedup(args)
    except Exception as e:
        logger.error(f"Error running dedup: {e}")
        raise
    finally:
        # Stop dask
        time.sleep(5)
        dask_client.close()
        time.sleep(5)
        dask_client.shutdown()
        logger.info("Dask client closed and shutdown")


def _run_semantic_dedup(args: argparse.Namespace) -> None:
    if is_s3path(args.input_embeddings_path):
        s3_client_config = get_s3_client_config()
        storage_options = {
            "key": s3_client_config.aws_access_key_id,
            "secret": s3_client_config.aws_secret_access_key,
            "token": s3_client_config.aws_session_token,
            "client_kwargs": {
                "endpoint_url": s3_client_config.endpoint_url,
                "region_name": s3_client_config.region,
            },
        }
    elif is_azure_path(args.input_embeddings_path):
        azure_client_config = get_azure_client_config()
        storage_options = {
            "connection_string": azure_client_config.connection_string,
            "account_name": azure_client_config.account_name,
            "account_key": azure_client_config.account_key,
        }
    else:
        storage_options = None

    output_dir = args.output_path
    id_column = "id"
    embedding_column = "embedding"

    semantic_config = SemDedupConfig(
        # i/o args
        cache_dir=output_dir,
        clustering_save_loc="clustering_results",
        embedding_column=embedding_column,
        random_state=args.random_state,
        # configureable args
        max_iter=args.max_iter,
        n_clusters=args.n_clusters,
        which_to_keep=args.which_to_keep,
        sim_metric=args.sim_metric,
        eps_to_extract=args.eps_to_extract,
        # no-op args
        clustering_input_partition_size=None,
        profile_dir=None,
    )
    # Load embeddings
    dataset = DocumentDataset(
        dask_cudf.read_parquet(
            path=args.input_embeddings_path,
            blocksize=args.blocksize,
            storage_options=storage_options,
        ),
    )

    cluster, semantic = setup_clustering_semantic(
        id_column,
        embedding_column,
        semantic_config,
        output_dir,
        storage_options,
    )
    start_time = _start_time = time.perf_counter()
    # Step 1) Run clustering
    with dask.config.set({"optimization.fuse.active": False}):
        cluster(dataset)
    clustering_time = time.perf_counter() - start_time
    logger.info(f"Clustering took {clustering_time:.2f} seconds")

    # Step 2) Compute semantic matches
    _start_time = time.perf_counter()
    semantic.compute_semantic_match_dfs()
    semantic_matching_time = time.perf_counter() - _start_time
    logger.info(f"Semantic matching took {semantic_matching_time:.2f} seconds")

    # Step 3) Extract dedup data based on eps_to_extract
    _start_time = time.perf_counter()
    output = semantic.extract_dedup_data(eps_to_extract=semantic_config.eps_to_extract)
    extract_dedup_data_time = time.perf_counter() - _start_time
    logger.info(f"Extract dedup data took {extract_dedup_data_time:.2f} seconds")

    logger.info(f"Total time taken: {(time.perf_counter() - start_time):.2f} seconds")
    # end of pipeline
    logger.success(f"Num clips that can be removed with eps={semantic_config.eps_to_extract}={len(output.df)}")


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scheduler-file",
        type=str,
        required=False,
        help="Path to dask scheduler file",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of clusters for K-means clustering",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for clustering",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for reproducibility",
    )
    parser.add_argument(
        "--eps-to-extract",
        type=float,
        default=0.01,
        help="Epsilon value to extract deduplicated data. Must be in eps_thresholds.",
    )
    parser.add_argument(
        "--sim-metric",
        type=str,
        default="cosine",
        help="Metric to use to order within a cluster with respect to the centroid. Options are 'cosine', 'l2'.",
    )
    parser.add_argument(
        "--which-to-keep",
        type=str,
        default="hard",
        help="Determins the order in which to keep items within a cluster. "
        "Options are 'hard', 'random', or 'easy'."
        " hard:   retains edge-case or outlier items farthest from the centroid by"
        "         sorting points by decreasing distance from the centroid.."
        " easy:   retains representative items closest to the centroid by"
        "         sorting points by increasing distance from the centroid."
        " random: retains items randomly.",
    )
    parser.add_argument(
        "--blocksize",
        type=str,
        default="128MiB ",
        help="Size of each partition to load from input embeddings location",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output location",
    )
    parser.add_argument(
        "--input-embeddings-path",
        type=str,
        required=True,
        help="Path to input embeddings location (typically ends with iv2_embd_parquet or ce1_embd_parquet)",
    )


def nvcf_run_semdedup(args: argparse.Namespace) -> None:
    """Run semantic deduplication.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    cli_run_dedup(args)


def cli_run_dedup(args: argparse.Namespace) -> None:
    """Run semantic deduplication.

    Args:
        args: Command line arguments.

    """
    setup_dask_and_run_semantic_dedup(args)


def add_dedup_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add a deduplication command to the parser.

    Args:
        subparsers: Subparsers action.

    Returns:
        Argument parser.

    """
    parser = subparsers.add_parser(
        "dedup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Deduplicate clips.",
    )
    parser.set_defaults(func=cli_run_dedup)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    args = parser.parse_args()
    cli_run_dedup(args)
