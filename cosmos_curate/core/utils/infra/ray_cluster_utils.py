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

"""Ray utilities."""

import loguru
import ray
from loguru import logger


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    # Initialize a default logger
    return logger


def init_or_connect_to_cluster() -> None:
    """Initialize a new local Ray cluster or connects to an existing one."""
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
    )


def get_node_idx_and_name() -> tuple[int, str]:
    """Get the node index and name."""
    node_id = ray.get_runtime_context().get_node_id()
    nodes = ray.nodes()
    for idx in range(len(nodes)):
        node = nodes[idx]
        if node["NodeID"] == node_id:
            return idx, node["NodeManagerHostname"]
    return -1, ""


def get_node_name() -> str:
    """Get the node name."""
    _, node_name = get_node_idx_and_name()
    return node_name
