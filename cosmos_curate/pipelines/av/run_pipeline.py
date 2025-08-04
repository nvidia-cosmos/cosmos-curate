# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""AV data curation pipelines."""

import argparse

from cosmos_curate.core.utils.config.operation_context import check_if_running_in_pixi_env


def cli() -> None:
    """CLI for AV data curation pipelines."""
    # Lazy-import pipeline commands after PIXI env check
    from cosmos_curate.pipelines.av.av_video_captioning_pipeline import add_caption_command
    from cosmos_curate.pipelines.av.av_video_ingesting_pipeline import add_ingest_command
    from cosmos_curate.pipelines.av.av_video_sharding_pipeline import add_shard_command
    from cosmos_curate.pipelines.av.av_video_splitting_pipeline import add_split_command

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="AV data curation pipelines",
    )
    subparsers = parser.add_subparsers(dest="command")
    add_ingest_command(subparsers)
    add_split_command(subparsers)
    add_caption_command(subparsers)
    add_shard_command(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    check_if_running_in_pixi_env()
    cli()
