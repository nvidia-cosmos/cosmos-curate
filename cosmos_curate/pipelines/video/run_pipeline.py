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

"""Run Pipeline CLI."""

import argparse
import importlib

from cosmos_curate.core.utils.config.operation_context import check_if_running_in_pixi_env


def cli() -> None:
    """Run the video curation pipeline CLI.

    Parses command line arguments and executes the selected pipeline command.
    """
    # Lazy-import pipeline commands after PIXI env check
    dedup_module = importlib.import_module("cosmos_curate.pipelines.video.dedup_pipeline")
    shard_module = importlib.import_module("cosmos_curate.pipelines.video.sharding_pipeline")
    split_module = importlib.import_module("cosmos_curate.pipelines.video.splitting_pipeline")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Video curation pipelines",
    )
    subparsers = parser.add_subparsers(dest="command")
    shard_module.add_shard_command(subparsers)
    split_module.add_split_command(subparsers)
    dedup_module.add_dedup_command(subparsers)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    check_if_running_in_pixi_env()
    cli()
