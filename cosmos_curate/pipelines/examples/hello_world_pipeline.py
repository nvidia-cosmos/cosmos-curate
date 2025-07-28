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

"""Example Hello World."""

import os
import time

import attrs
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curate.core.utils.model.conda_utils import get_conda_env_name
from cosmos_curate.models.gpt2 import GPT2


# pipeline task object that is being passed between stages
@attrs.define
class HelloWorldTask(PipelineTask):
    """A pipeline task that processes text prompts and stores generated output.

    This task is used in the hello world example pipeline to demonstrate basic text processing
    and generation capabilities.
    """

    prompt: str
    output: str | None = None


def _get_stage_processing_log_str(stage: CuratorStage, task: HelloWorldTask) -> str:
    return (
        f"processing task prompt='{task.prompt}' in "
        f"stage={stage.__class__.__name__} pid={os.getpid()} "
        f"env={get_conda_env_name()}"
    )


class _LowerCaseStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[HelloWorldTask]) -> list[HelloWorldTask] | None:  # type: ignore[override]
        # convert the prompt to lowercase
        for task in tasks:
            logger.debug(_get_stage_processing_log_str(self, task))
            task.prompt = task.prompt.lower()
        return tasks


class _PrintStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=0.5, gpus=0.0)

    def process_data(self, tasks: list[HelloWorldTask]) -> list[HelloWorldTask] | None:  # type: ignore[override]
        for task in tasks:
            logger.debug(_get_stage_processing_log_str(self, task))
            # print the prompt
            print(task.prompt)  # noqa: T201
        return tasks


class _GPT2Stage(CuratorStage):
    def __init__(self) -> None:
        self._model = GPT2()

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.8, entire_gpu=False)

    @property
    def model(self) -> ModelInterface | None:
        return self._model

    def stage_setup(self) -> None:
        self._model.setup()

    def process_data(self, tasks: list[HelloWorldTask]) -> list[HelloWorldTask] | None:  # type: ignore[override]
        for task in tasks:
            logger.debug(_get_stage_processing_log_str(self, task))
            # generate text based on the prompt and print out result
            task.output = self._model.generate(task.prompt)
            print(" ".join(task.output.split()))  # noqa: T201
            time.sleep(1)
        return tasks


def main() -> None:
    """Run the hello world pipeline with example prompts."""
    # build a list of input tasks
    prompts = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]
    tasks: list[PipelineTask] = [HelloWorldTask(prompt=x) for x in prompts]
    logger.info(f"Number of input tasks: {len(tasks)}")

    # construct a list of pipeline stages
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(_LowerCaseStage(), num_workers_per_node=2),
        _PrintStage(),
        _GPT2Stage(),
    ]

    # run the pipeline
    run_pipeline(tasks, stages)

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
