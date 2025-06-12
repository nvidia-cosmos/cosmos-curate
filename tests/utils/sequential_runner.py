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
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to run a sequence of pipeline stages in tests."""

from typing import TypeVar

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, PipelineTask

T = TypeVar("T", bound=PipelineTask)


def run_pipeline(
    tasks: list[T],
    stages: list[CuratorStage],
) -> list[T]:
    """Run a sequence of stages sequentially on the provided tasks and return the final tasks."""
    for stage in stages:
        stage.stage_setup()

    for stage in stages:
        tasks = stage.process_data(tasks)
        stage.destroy()
    return tasks
