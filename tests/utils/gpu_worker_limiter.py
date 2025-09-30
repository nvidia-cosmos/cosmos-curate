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
"""Pytest plugin to limit workers to available GPU count."""

import importlib

from _pytest.config import Config, Parser
from loguru import logger


def _get_available_gpus() -> int:
    """Get the number of available GPUs using pynvml (no CUDA initialization)."""
    try:
        pynvml = importlib.import_module("pynvml")
    except ImportError:
        logger.warning("pynvml not installed; GPU worker limiting disabled")
        return 0

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=exc).exception("Failed to query GPU count via pynvml")
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            logger.debug("nvmlShutdown failed, continuing")

    return device_count


def pytest_load_initial_conftests(early_config: Config, parser: Parser, args: list[str]) -> None:  # noqa: ARG001
    """Limit the number of workers to the number of available GPUs.

    Args:
        args: The command line arguments.
        early_config: The early config object.
        parser: The parser object.

    """
    logger.info("Loading initial conftests")
    num_gpus = _get_available_gpus()
    if num_gpus > 0:
        # Look for -n argument and its value
        for i, arg in enumerate(args):
            if arg in ("-n", "--numprocesses") and i + 1 < len(args):
                try:
                    requested_workers = int(args[i + 1])
                    if requested_workers > num_gpus:
                        logger.info(f"Available GPUs: {num_gpus}")
                        logger.info(f"Requested workers: {requested_workers}")
                        logger.info(f"Limited workers from {requested_workers} to {num_gpus} (available GPUs)")
                        args[i + 1] = str(num_gpus)
                except (ValueError, IndexError):
                    pass
