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
"""Pytest configuration and fixtures for curator tests.

This module provides custom pytest functionality, including environment-specific
test filtering for running tests only in their respective conda environments.
"""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """When the user runs `pytest -m env`, filter tests by environment.

    `@pytest.mark.env("<name>")` argument must matche the current
    CONDA_DEFAULT_ENV.  Everything else is deselected.
    """
    # Was the user's command-line marker expression exactly `env`?
    if (config.option.markexpr or "").strip() != "env":
        # User didn't ask for the env filter; do nothing special.
        return

    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not current_env:
        # No active conda env ⇒ deselect all tests with the marker.
        marked_tests = [item for item in items if item.get_closest_marker("env") is not None]
        if marked_tests:
            config.hook.pytest_deselected(items=marked_tests)
            items[:] = [item for item in items if item.get_closest_marker("env") is None]
        return

    selected, deselected = [], []
    for item in items:
        m = item.get_closest_marker("env")
        if m is None:
            # Test is not marked with @pytest.mark.env → drop it
            deselected.append(item)
            continue

        # Accept the test only if its first positional arg equals the env
        if m.args and m.args[0] == current_env:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
