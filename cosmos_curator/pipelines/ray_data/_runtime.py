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
"""Shared runtime helpers for Ray Data-native pipelines."""

from typing import TYPE_CHECKING, cast

import ray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Substring patterns matched against ``"ClassName: message"`` (via
# ``ray._common.retry.format_exception``) to decide whether a map function
# error is transient and worth retrying. Class-name substrings avoid
# dragging in botocore just to reference exception types.
#
# ``botocore.exceptions.ClientError`` is deliberately excluded because it
# is the base class for *all* AWS HTTP responses -- including deterministic
# 4xx errors (NoSuchKey, AccessDenied, InvalidBucketName) that should fail
# fast, not consume ``_MAP_MAX_RETRIES`` attempts. Botocore already retries
# transient 5xx / throttling at its own HTTP client layer; what still
# bubbles up to us is caught via the transport-level patterns below
# (DNS / URL, socket read stalls, mid-stream drops on large GETs).
_MAP_RETRY_PATTERNS: list[str] = [
    "OSError",
    "ConnectionError",
    "TimeoutError",
    "EndpointConnectionError",
    "ReadTimeoutError",
    "IncompleteRead",
]
_MAP_MAX_RETRIES = 3


def configure_ray_data_progress(*, progress: bool) -> None:
    """Configure Ray Data progress output before creating datasets."""
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = progress
    ctx.enable_operator_progress_bars = progress
    ctx.enable_rich_progress_bars = progress
    ctx.print_on_execution_start = progress
    ctx.use_ray_tqdm = False


def configure_ray_data_stability() -> None:
    """Enable Ray 2.56 stability knobs for map-heavy Ray Data pipelines.

    - ``default_map_logical_memory_enabled`` (PR ray-project/ray#63814) gives
      map operators a default logical memory footprint so the scheduler
      back-pressures before triggering object-store spills or worker OOM
      kills in IO / transcode / write stages.
    - ``retried_map_errors`` + ``max_map_retries`` (PR ray-project/ray#63023)
      retry map functions when the raised exception looks like a transient
      network / S3 / OS error, up to ``_MAP_MAX_RETRIES`` attempts.

    Attributes are guarded with ``hasattr`` so importing this module on a
    stale (pre-2.56) Ray wheel still works.
    """
    ctx = ray.data.DataContext.get_current()
    if hasattr(ctx, "default_map_logical_memory_enabled"):
        ctx.default_map_logical_memory_enabled = True  # type: ignore[attr-defined]
    # retried_map_errors and max_map_retries both shipped in ray-project/ray#63023;
    # gate them together so a partial-implementation Ray build can't end up with
    # the error list set but no retry cap (or vice versa).
    if hasattr(ctx, "retried_map_errors") and hasattr(ctx, "max_map_retries"):
        ctx.retried_map_errors = list(_MAP_RETRY_PATTERNS)  # type: ignore[attr-defined]
        ctx.max_map_retries = _MAP_MAX_RETRIES  # type: ignore[attr-defined]


def ensure_ray_initialized() -> None:
    """Initialize Ray locally when the caller has not already connected to a cluster."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def live_ray_node_count() -> int:
    """Return the number of live Ray nodes, floored at one for local callers."""
    nodes = cast("Sequence[Mapping[str, object]]", ray.nodes())  # type: ignore[no-untyped-call]
    return max(1, sum(1 for node in nodes if bool(node.get("Alive"))))


def capped_slots_for_items(*, num_items: int, num_nodes: int, max_per_node: int) -> int:
    """Cap concurrent work by both input item count and cluster node count."""
    if num_items <= 0:
        return 0
    cluster_cap = max_per_node * max(1, num_nodes)
    return min(num_items, cluster_cap)
