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

import socket

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


def get_live_nodes(*, dump_info: bool = True) -> list[dict[str, str]]:
    """Get the list of alive nodes in the Ray cluster."""
    try:
        all_nodes = ray.nodes()
    except Exception as e:
        logger.error(f"Failed to get nodes: {e}")
        msg = "Failed to get nodes from Ray cluster"
        raise RuntimeError(msg) from e

    live_nodes = []
    for node in all_nodes:
        if node["Alive"]:
            if dump_info:
                logger.info(f"Found node {node['NodeID']} on {node['NodeManagerHostname']}")
            node_info = {
                "NodeID": node["NodeID"],
                "NodeName": node["NodeManagerHostname"],
            }
            live_nodes.append(node_info)
        elif dump_info:
            logger.warning(f"Found node {node['NodeID']} on {node['NodeManagerHostname']} is NOT alive")
    return live_nodes


def get_node_idx_and_name() -> tuple[int, str]:
    """Get the node index and name."""
    node_id = ray.get_runtime_context().get_node_id()
    nodes = get_live_nodes(dump_info=False)
    for idx, node in enumerate(nodes):
        if node["NodeID"] == node_id:
            return idx, node["NodeName"]
    return -1, socket.gethostname()


def get_node_name() -> str:
    """Get the node name."""
    _, node_name = get_node_idx_and_name()
    return node_name
