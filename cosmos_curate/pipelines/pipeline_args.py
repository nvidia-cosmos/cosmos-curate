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
"""Common arguments for different pipelines."""

import argparse

from cosmos_curate.core.utils.environment import MODEL_WEIGHTS_PREFIX


def add_profiling_args(parser: argparse.ArgumentParser) -> None:
    """Add profiling / instrumentation CLI flags to the parser.

    Args:
        parser: The argument parser to add profiling flags to.

    """
    parser.add_argument(
        "--perf-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lightweight basic performance profiling (use --no-perf-profile to disable).",
    )
    parser.add_argument(
        "--profile-tracing",
        action="store_true",
        default=False,
        help=(
            "Enable distributed tracing (OpenTelemetry) via Ray's tracing hook. "
            "Captures cross-actor spans (task scheduling, actor creation, method "
            "invocations) as NDJSON files in <output-path>/profile/traces/. "
            "Implies --perf-profile. "
            "Note: should be set to True by default once Xenna adds proper "
            "tracing support."
        ),
    )
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        default=False,
        help=(
            "Enable CPU profiling (pyinstrument) for every pipeline stage. "
            "Saves per-task HTML flame-tree reports to <output-path>/profile/cpu/. "
            "Implies --perf-profile."
        ),
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help=(
            "Enable memory profiling (memray) for every pipeline stage. "
            "Saves per-task .bin captures and HTML flamegraphs to "
            "<output-path>/profile/memory/. Implies --perf-profile."
        ),
    )
    parser.add_argument(
        "--profile-gpu",
        action="store_true",
        default=False,
        help=(
            "Enable GPU profiling (torch.profiler) for every pipeline stage. "
            "Captures CUDA kernel launches, operator breakdown, and GPU "
            "memory allocations.  Saves per-task Chrome Trace JSON to "
            "<output-path>/profile/gpu/.  Silently disabled on CPU-only workers. "
            "Implies --perf-profile."
        ),
    )
    parser.add_argument(
        "--profile-cpu-exclude",
        type=str,
        default="_root",
        help=(
            "Comma-separated list of scope names to exclude from CPU profiling. "
            "Scope names are stage class names (e.g. VideoDownloader, RemuxStage) "
            "or '_root' for the driver process."
        ),
    )
    parser.add_argument(
        "--profile-memory-exclude",
        type=str,
        default="_root",
        help=(
            "Comma-separated list of scope names to exclude from memory profiling. "
            "Scope names are stage class names (e.g. VideoDownloader, RemuxStage) "
            "or '_root' for the driver process. "
            "Note: memray may conflict with pyinstrument on long-lived driver processes; "
            "pass '_root' to avoid this."
        ),
    )
    parser.add_argument(
        "--profile-gpu-exclude",
        type=str,
        default="_root",
        help=(
            "Comma-separated list of scope names to exclude from GPU profiling. "
            "Scope names are stage class names (e.g. VideoDownloader, RemuxStage) "
            "or '_root' for the driver process. "
            "Default: '_root' (driver process typically has no CUDA context)."
        ),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common command line arguments to the parser.

    Includes S3, execution-mode, limit, verbose, model-weights, and
    profiling flags (via :func:`add_profiling_args`).

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
        "--model-weights-path",
        type=str,
        default=MODEL_WEIGHTS_PREFIX,
        help=(
            "Local path or S3 prefix for model weights. "
            "Used to download model weights to local cache if they are not already present. "
            "If a unix path is provided, it must be accessible from all nodes."
        ),
    )
    add_profiling_args(parser)
