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
"""Entry function to run a pipeline."""

from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import ray
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask
from cosmos_curate.core.utils.config.operation_context import get_tmp_dir, is_running_on_slurm, is_running_on_the_cloud
from cosmos_curate.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curate.core.utils.infra import hardware_info, ray_cluster_utils
from cosmos_curate.core.utils.model.model_utils import download_model_weights_on_all_nodes
from cosmos_xenna.pipelines.private.pipelines import run_pipeline as xenna_run_pipeline
from cosmos_xenna.pipelines.private.specs import (
    ExecutionMode,
    PipelineConfig,
    PipelineSpec,
    StreamingSpecificSpec,
)
from cosmos_xenna.utils.verbosity import VerbosityLevel


class PipelineExecutionError(Exception):
    """Exception raised when pipeline execution fails.

    This exception is used to wrap and provide context for various types of failures
    that can occur during pipeline execution, such as:
    - Invalid execution mode
    - Stage execution failures
    - Resource allocation failures
    - Model download failures

    Attributes:
        message: A human-readable error message describing the failure.
        original_error: The original exception that caused the pipeline failure, if any.

    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize PipelineExecutionError.

        Args:
            message: A human-readable error message describing the failure.
            original_error: The original exception that caused the pipeline failure, if any.

        """
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            A string containing both the error message and the original error if present.

        """
        if self.original_error:
            return f"{self.message} (Original error: {self.original_error!s})"
        return self.message


def _worker_download_models(model_names: list[str], model_weights_prefix: str) -> int:
    """Worker function to prepare the pipeline by downloading models.

    Args:
        model_names: List of model names to download.
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        int: Number of GPUs available in the Ray cluster.

    """
    ray_cluster_utils.init_or_connect_to_cluster()
    num_gpus: int = ray.cluster_resources().get("GPU", 0)
    num_nodes = len(ray_cluster_utils.get_live_nodes())
    logger.info(f"The ray cluster has {num_gpus} GPUs on {num_nodes} nodes.")

    # schedule a model-downloader task on each node
    download_model_weights_on_all_nodes(model_names, model_weights_prefix, num_nodes)
    # dump disk usage info
    hardware_info.print_disk_path_info(get_tmp_dir())

    ray_cluster_utils.shutdown_cluster()
    return num_gpus


def download_models(model_names: list[str], model_weights_prefix: str = MODEL_WEIGHTS_PREFIX) -> int:
    """Download model weights on all nodes in the Ray cluster.

    Args:
        model_names: List of model names to download.
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        int: Number of GPUs available in the Ray cluster.

    """
    # start a separate process to keep main process clean
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker_download_models, list(model_names), model_weights_prefix)
        return future.result()


def _prepare_to_run_pipeline(
    stages: list[CuratorStageSpec], model_weights_prefix: str = MODEL_WEIGHTS_PREFIX
) -> ExecutionMode:
    """Run a few steps to prepare the pipeline for execution.

    - downloading required models
    - determining execution mode

    Args:
        stages: List of pipeline stages to process.
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        ExecutionMode: The execution mode for the pipeline.

    """
    model_names: set[str] = set()
    num_gpus_requested: float = 0
    logger.debug(f"Number of pipeline stages: {len(stages)}")
    for stage in stages:
        # mypy type-narrowing
        if stage.stage.model is not None:  # type: ignore[attr-defined]
            model_names.update(stage.stage.model.model_id_names)  # type: ignore[attr-defined]
        num_gpus_requested += stage.stage.required_resources.gpus
        logger.debug(stage.display_str())

    num_gpus_available = download_models(list(model_names), model_weights_prefix)
    logger.info(
        f"Required GPUs to run STREAMING mode vs. available GPUs: {num_gpus_requested} vs. {num_gpus_available}"
    )
    return ExecutionMode["STREAMING"] if num_gpus_requested <= num_gpus_available else ExecutionMode["BATCH"]


def _build_pipeline_stage_specs(stages: list[CuratorStage | CuratorStageSpec]) -> list[CuratorStageSpec]:
    """Build a list of pipeline stage specs from the given stages."""
    # Unify the stage spec and stage
    stage_specs: list[CuratorStageSpec] = []
    for stage in stages:
        if isinstance(stage, CuratorStage):
            stage_specs.append(CuratorStageSpec(stage))
        elif isinstance(stage, CuratorStageSpec):
            stage_specs.append(stage)
        else:
            err_msg = f"Invalid stage type: {type(stage)}. Expected CuratorStage or CuratorStageSpec."  # type: ignore[unreachable]
            raise PipelineExecutionError(err_msg, original_error=None)

    # Fill in some default pipeline stage spec values
    for stage_spec in stage_specs:
        # set default values for worker max lifetime
        if stage_spec.worker_max_lifetime_m is None:
            if stage_spec.stage.required_resources.gpus > 0.0:
                # GPU stage can be more expensive to restart
                stage_spec.worker_max_lifetime_m = 120
            elif stage_spec.stage.required_resources.cpus < 1.0:
                # likely an IO stage, do not restart
                stage_spec.worker_max_lifetime_m = 0
            else:
                stage_spec.worker_max_lifetime_m = 60
        # set default values for worker restart interval
        if stage_spec.worker_restart_interval_m is None:
            if stage_spec.stage.required_resources.gpus > 0.0:
                # again GPU stage can be more expensive to restart
                stage_spec.worker_restart_interval_m = 5
            else:
                stage_spec.worker_restart_interval_m = 1

    return stage_specs


T = TypeVar("T", bound=PipelineTask)


def run_pipeline(
    input_tasks: list[T],
    stages: list[CuratorStage | CuratorStageSpec],
    model_weights_prefix: str = MODEL_WEIGHTS_PREFIX,
) -> list[T]:
    """Run the pipeline with the given pipeline spec.

    Args:
        input_tasks: A list of pipeline tasks to process.
        stages: A list of stages.
        execution_mode: "STREAMING" or "BATCH".
        model_weights_prefix: Prefix for model weights in local or cloud storage.

    Returns:
        A list of pipeline payloads.

    Raises:
        PipelineExecutionError: If pipeline execution fails.

    """
    err_msg: str = ""

    # Build a list of StageSpecs and fill in default config values.
    stage_specs = _build_pipeline_stage_specs(stages)

    # Run a few steps to prepare the pipeline for execution.
    try:
        execution_mode = _prepare_to_run_pipeline(stage_specs, model_weights_prefix)
    except Exception as e:
        err_msg = "Failed in run_pipeline preparation step"
        raise PipelineExecutionError(err_msg, original_error=e) from e

    # Construct the pipeline configuration
    pipeline_config = PipelineConfig(
        execution_mode=execution_mode,
        enable_work_stealing=False,
        return_last_stage_outputs=True,
        actor_pool_verbosity_level=VerbosityLevel.NONE,
        monitoring_verbosity_level=VerbosityLevel.NONE
        if is_running_on_the_cloud() and not is_running_on_slurm()
        else VerbosityLevel.INFO,
        mode_specific=StreamingSpecificSpec(
            max_queued_multiplier=1.5,
            max_queued_lower_bound=16,
            autoscaler_verbosity_level=VerbosityLevel.NONE,
            executor_verbosity_level=VerbosityLevel.NONE,
        ),
    )

    # Run the pipeline!
    try:
        logger.info(f"Running pipeline in {execution_mode.name} mode with {len(input_tasks)} input tasks")
        pipeline_spec = PipelineSpec(input_data=input_tasks, stages=stage_specs, config=pipeline_config)
        output_tasks = xenna_run_pipeline(pipeline_spec)
    except Exception as e:
        err_msg = f"Pipeline execution failed: {e!s}"
        logger.error("Pipeline execution failed: ")
        raise PipelineExecutionError(err_msg, original_error=e) from e
    else:
        if output_tasks is None:
            logger.warning("Pipeline execution returned None")
            return []
        return output_tasks
    finally:
        ray_shutdown_delay = 5
        logger.info(f"Disconnecting from Ray cluster in {ray_shutdown_delay} seconds")
        ray_cluster_utils.shutdown_cluster(flush_seconds=ray_shutdown_delay)
