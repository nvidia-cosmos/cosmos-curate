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
"""Common arguments for different pipelines."""

import argparse

from cosmos_curate.core.utils.environment import MODEL_WEIGHTS_PREFIX


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common command line arguments to the parser.

    Args:
        parser: The argument parser to add arguments to.

    """
    parser.add_argument(
        "--input-s3-profile-name",
        type=str,
        default="default",
        help="S3 profile name to use for input S3 path.",
    )
    parser.add_argument(
        "--output-s3-profile-name",
        type=str,
        default="default",
        help="S3 profile name to use for output S3 path.",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="BATCH",
        choices=["BATCH", "STREAMING"],
        help="Execution mode of Cosmos-Curator pipeline; STREAMING can be enabled when there more GPUs than models",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of input videos to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print verbose logs.",
    )
    parser.add_argument(
        "--perf-profile",
        action="store_true",
        default=False,
        help="Whether to enable performance profiling.",
    )
    parser.add_argument(
        "--model-weights-path",
        type=str,
        default=MODEL_WEIGHTS_PREFIX,
        help=(
            "Local path or S3 prefix for model weights. "
            "Used to download model weights to local cache if they are not already present. "
            "If a unix path is provided, it must be accessible from all nodes."
        ),
    )
