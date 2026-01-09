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

"""Example Hello World using Ray Data."""

import time

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import ray
from loguru import logger
from ray.data import ActorPoolStrategy
from ray.runtime_env import RuntimeEnv

from cosmos_curate.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curate.models.gpt2 import GPT2
from cosmos_curate.pipelines.examples.hello_world_pipeline import EXAMPLE_PROMPTS, get_processing_log_str


def _ensure_ray_initialized() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def _lowercase_batch(batch: pa.Table) -> pa.Table:
    for prompt in batch["prompt"].to_pylist():
        logger.debug(get_processing_log_str("LowerCaseStage", prompt))
    lowered = pc.utf8_lower(batch["prompt"])
    column_index = batch.schema.get_field_index("prompt")
    return batch.set_column(column_index, "prompt", lowered)


def _print_row(row: dict[str, str]) -> dict[str, str]:
    prompt = row["prompt"]
    logger.debug(get_processing_log_str("PrintStage", prompt))
    print(prompt)  # noqa: T201
    return row


class _GPT2BatchPredictor:
    def __init__(self) -> None:
        self._model = GPT2()
        self._model.setup()

    def __call__(self, batch: pa.Table) -> pa.Table:
        outputs: list[str] = []
        for prompt in batch["prompt"].to_pylist():
            logger.debug(get_processing_log_str(self.__class__.__name__, prompt))
            output = self._model.generate(prompt)
            outputs.append(output)
            print(" ".join(output.split()))  # noqa: T201
            time.sleep(1)
        return batch.append_column("output", pa.array(outputs))

    @classmethod
    def runtime_env(cls) -> RuntimeEnv:
        return PixiRuntimeEnv(GPT2().conda_env_name)


def main() -> None:
    """Run the hello world Ray Data pipeline with example prompts."""
    _ensure_ray_initialized()

    rows = [{"prompt": prompt} for prompt in EXAMPLE_PROMPTS]
    logger.info(f"Number of input tasks: {len(rows)}")

    dataset = ray.data.from_items(rows)

    # Ray Data builds a lazy DAG of transforms: batch lowercasing in Arrow,
    # per-row logging, then batched GPT-2 inference via a single GPU actor.
    dataset = dataset.map_batches(_lowercase_batch, batch_format="pyarrow")
    dataset = dataset.map(_print_row)
    dataset = dataset.map_batches(
        _GPT2BatchPredictor,
        batch_format="pyarrow",
        batch_size=2,
        compute=ActorPoolStrategy(size=1),
        num_gpus=0.8,
        runtime_env=_GPT2BatchPredictor.runtime_env(),
    )

    dataset.materialize()
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
