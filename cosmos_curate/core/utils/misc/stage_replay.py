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
"""Stage replay.

Implementation notes:

To improve testability, protocol classes are defined for dependency injection.

This leads to a longer implementation, but it makes the code more testable.

The code is organized into the following sections:

- Protocols for dependency injection
- Default implementations of the protocols
- Helper functions
- Public API
"""

import argparse
import pickle
import random
import secrets
from pathlib import Path
from typing import Protocol, TypeVar, cast

import attrs
import ray
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask
from cosmos_curate.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curate.pipelines.video.utils.data_model import Video
from cosmos_xenna.pipelines.private.resources import NodeInfo, WorkerMetadata

BaseStage = TypeVar("BaseStage", bound="CuratorStage")

MAX_STAGE_REPLAY_ARGS = 2


# ============================================================================
# Protocols for Dependency Injection (for testability)
# ============================================================================


class TaskSerializer(Protocol):
    """Protocol for task serialization and deserialization."""

    def save(self, path: Path, tasks: list[PipelineTask]) -> None:
        """Save tasks to a file.

        Args:
            path: Path to save the tasks to.
            tasks: Tasks to save.

        """
        ...  # pragma: no cover

    def load(self, path: Path) -> list[PipelineTask]:
        """Load tasks from a file.

        Args:
            path: Path to load the tasks from.

        Returns:
            Loaded tasks.

        """
        ...  # pragma: no cover

    def find_task_files(self, directory: Path, pattern: str, limit: int = 0) -> list[Path]:
        """Find task files in a directory.

        Args:
            directory: Directory to search in.
            pattern: Glob pattern to match files.
            limit: Maximum number of files to return.

        Returns:
            Sorted list of matching file paths.

        """
        ...  # pragma: no cover


class StageExecutor(Protocol):
    """Protocol for executing stages on task batches."""

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage on task batches.

        Args:
            stage: The stage to execute.
            task_batches: Batches of tasks to process.
            node_info: Node information for stage setup.
            worker_metadata: Worker metadata for stage setup.

        Returns:
            Processed task batches.

        """
        ...  # pragma: no cover


# ============================================================================
# Helper class for wrapping stages
# ============================================================================
class StageRunner:
    """Run a stage."""

    def __init__(self, stage: CuratorStage) -> None:
        """Initialize the stage runner.

        Args:
            stage: The stage to run.

        """
        self.stage = stage

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:
        """Set up the stage on the node.

        Args:
            node_info: The node info.
            worker_metadata: The worker metadata.

        """
        self.stage.setup_on_node(node_info, worker_metadata)

    def stage_setup(self) -> None:
        """Set up the stage."""
        self.stage.stage_setup()

    def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask] | None:
        """Process the data.

        Args:
            tasks: The tasks to process.

        Returns:
            Result of processing the tasks.

        """
        return self.stage.process_data(tasks)

    def destroy(self) -> None:
        """Destroy the stage runner."""
        self.stage.destroy()


# ============================================================================
# Default Implementations
# ============================================================================


class PickleTaskSerializer:
    """Default pickle-based task serializer."""

    def save(self, path: Path, tasks: list[PipelineTask]) -> None:
        """Save tasks to a pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(tasks, f)

    def load(self, path: Path) -> list[PipelineTask]:
        """Load tasks from a pickle file."""
        with path.open("rb") as f:
            logger.info(f"Loading tasks from {path}")
            return cast("list[PipelineTask]", pickle.load(f))  # noqa: S301

    def find_task_files(self, directory: Path, pattern: str, limit: int = 0) -> list[Path]:
        """Find task files matching a pattern."""
        files = sorted(directory.glob(pattern))
        if limit > 0:
            files = files[:limit]
        return files


class RayStageExecutor:
    """Ray-based stage executor for distributed processing."""

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage using Ray actors."""
        conda_env_name = stage.conda_env_name if stage.conda_env_name is not None else "default"
        runtime_env = PixiRuntimeEnv(conda_env_name)

        logger.info(f"Starting actor for stage {stage.__class__.__name__}")
        stage_runner = (
            ray.remote(StageRunner)
            .options(  # type: ignore[no-untyped-call]
                runtime_env=runtime_env,
                num_cpus=stage.required_resources.cpus,
                num_gpus=stage.required_resources.gpus,
            )
            .remote(stage)
        )

        ray.get(stage_runner.setup_on_node.remote(node_info, worker_metadata))
        ray.get(stage_runner.stage_setup.remote())

        logger.info(f"Processing {len(task_batches)} task batches for stage {stage.__class__.__name__}")
        out_task_batches = []
        for task_batch in task_batches:
            result = ray.get(stage_runner.process_data.remote(task_batch))
            out_task_batch = result if result is not None else []
            out_task_batches.append(out_task_batch)
        logger.info(f"Processed {len(out_task_batches)} task batches for stage {stage.__class__.__name__}")

        ray.get(stage_runner.destroy.remote())
        ray.kill(stage_runner)

        return out_task_batches


class DirectStageExecutor:
    """Direct stage executor without Ray.

    Executes stages directly in the current process without using Ray actors.
    Useful for unit testing and debugging.

    """

    def execute_stage(
        self,
        stage: CuratorStage,
        task_batches: list[list[PipelineTask]],
        node_info: NodeInfo,
        worker_metadata: WorkerMetadata,
    ) -> list[list[PipelineTask]]:
        """Execute a stage directly without Ray."""
        logger.info(f"Executing stage {stage.__class__.__name__} directly (no Ray)")
        stage.setup_on_node(node_info, worker_metadata)
        stage.stage_setup()

        result = []
        for batch in task_batches:
            output = stage.process_data(batch)
            result.append(output if output is not None else [])

        stage.destroy()
        return result


def _clamp_sample_rate(value: float) -> float:
    """Clamp sample rate between 0.0 and 1.0."""
    return min(max(value, 0.0), 1.0)


@attrs.define
class StageSaveConfig:
    """Configuration for saving tasks from the pipeline.

    Args:
        path: Path to save tasks to.
        stages: List of stage names to save tasks from.
        sample_rate: Sample rate for saving tasks. Range is [0.0, 1.0].

    """

    path: Path
    stages: list[str]
    sample_rate: float = attrs.field(converter=_clamp_sample_rate)


# ============================================================================
# Helper Functions
# ============================================================================


def _load_task_batches(
    path: Path,
    limit: int,
    serializer: TaskSerializer | None = None,
) -> list[list[PipelineTask]]:
    """Load tasks from the tasks directory.

    Args:
        path: The path to the tasks directory.
        limit: The maximum number of task files to load.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.

    Returns:
        A list of task objects.

    """
    _serializer = serializer if serializer is not None else PickleTaskSerializer()
    files = _serializer.find_task_files(path, "*.pkl", limit)
    return [_serializer.load(file) for file in files]


def _get_name_from_tasks(class_name: str, tasks: list[PipelineTask]) -> str:
    """Get the name of the tasks.

    Args:
        class_name: The name of the class to save the tasks for.
        tasks: The tasks to get the name from.

    Returns:
        The name of the tasks.

    """
    if len(tasks) == 0:
        msg = "No tasks to get the name from"
        raise ValueError(msg)

    first_task = tasks[0]

    video = getattr(first_task, "video", None)
    if video is not None and isinstance(video, Video):
        input_video = video.input_video
        if isinstance(input_video, Path):
            task_suffix = input_video.name
        else:
            msg = "_get_name_from_tasks only supports local video files"
            raise TypeError(msg)
    else:
        task_suffix = secrets.token_hex(8)

    return f"{class_name}/{task_suffix}"


def _get_output_path(base_output_path: Path, name: str, extension: str) -> Path:
    """Get a unique output path for a task.

    For tasks that have a video attribute (type Video), the task name will be
    derived from the video's file name. However, a single video may span
    multiple tasks, which leads to a conflict if the same name is used for
    tasks.

    To resolve this, a unique index is appended to the task name. This is done
    by incrementing the index until a unique path is found. This index roughly
    corresponds to the chunk index of a video.

    Args:
        base_output_path: The base output path.
        name: The name of the task.
        extension: The extension of the task.

    Returns:
        The output path for the task.

    Raises:
        RuntimeError: If a unique output path cannot be found.

    """
    for i in range(1000):
        output_path = base_output_path / f"{name}_{i:03d}.{extension}"
        if not output_path.exists():
            return output_path
    msg = f"Failed to find a unique output path for {name}"
    raise RuntimeError(msg)


def _save_tasks(
    class_name: str,
    config: StageSaveConfig,
    tasks: list[PipelineTask],
    serializer: TaskSerializer | None = None,
) -> None:
    """Save tasks to a pickle file.

    Args:
        class_name: The name of the class to save the tasks for.
        config: Configuration for saving stages for replay.
        tasks: The tasks to save.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.

    """
    if serializer is None:
        serializer = PickleTaskSerializer()

    name = _get_name_from_tasks(class_name, tasks)
    output_path = _get_output_path(config.path, name, "task.pkl")
    serializer.save(output_path, tasks)
    logger.info(f"Saved tasks to {output_path}")


def _make_stage_save_class[T: CuratorStage](
    stage_cls: type[T],
    config: StageSaveConfig,
    serializer: TaskSerializer | None = None,
) -> type[T]:
    """Make a task-saving stage class.

    Create a new stage class that wraps the base stage class and save tasks
    passed to process_data to a file.

    Tasks are only saved if a random number between 0 and 1 is less than the
    sample rate.

    Args:
        stage_cls: The base stage class to wrap.
        config: Configuration for saving stages for replay.
        serializer: Task serializer. Defaults to PickleTaskSerializer.

    Returns:
        The task-saving stage class.

    """
    _serializer: TaskSerializer = serializer if serializer is not None else PickleTaskSerializer()

    if config.sample_rate <= 0.0:
        return stage_cls

    base_name = stage_cls.__name__

    class TaskSavingStage(stage_cls):  # type: ignore[valid-type, misc]
        _config = config
        _serializer_inst = _serializer

        def process_data(self, tasks: list[PipelineTask]) -> list[PipelineTask]:
            if tasks and random.random() <= self._config.sample_rate:  # noqa: S311
                _save_tasks(base_name, self._config, tasks, self._serializer_inst)

            return super().process_data(tasks)  # type: ignore[no-any-return]

    TaskSavingStage.__name__ = f"{base_name}WithTaskSaving"
    TaskSavingStage.__qualname__ = TaskSavingStage.__name__
    return TaskSavingStage


# ============================================================================
# Public API
# ============================================================================


def add_stage_replay_args(parser: argparse.ArgumentParser) -> None:
    """Add stage replay arguments to the parser.

    Args:
        parser: The parser to add the arguments to.

    """
    parser.add_argument(
        "--stage-save",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Comma-separated list of stage names to save input tasks from (e.g., 'Stage1,Stage2').",
    )
    parser.add_argument(
        "--stage-save-sample-rate",
        type=float,
        default=0.0,
        help="Fraction of tasks to save for each stage (0.0 = none, 1.0 = all)",
    )
    parser.add_argument(
        "--stage-replay",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[],
        help="Comma-separated list of stage names to replay using saved tasks. If one stage is provided, it will be "
        "in isolation. two stages are first_stage,last_stage. "
        "Saved tasks are loaded from --output-clip-path / tasks / stage_name / *.pkl",
    )


def validate_stage_replay_args(args: argparse.Namespace) -> None:
    """Validate the stage replay arguments.

    Args:
        args: The arguments to validate.

    """
    if len(args.stage_save) == 0 and len(args.stage_replay) == 0:
        return

    if len(args.stage_save) > 0 and len(args.stage_replay) > 0:
        msg = "Cannot save tasks and replay stages at the same time"
        raise ValueError(msg)

    if len(args.stage_replay) > MAX_STAGE_REPLAY_ARGS:
        msg = "--stage-replay should only have one stage, or two stages: start, end."
        raise ValueError(msg)


def get_stages_to_replay(
    stages: list[CuratorStage | CuratorStageSpec], start_stage_name: str, end_stage_name: str
) -> list[CuratorStage]:
    """Get the replay stages from the stages list.

    Args:
        stages: The list of stages.
        start_stage_name: The name of the start stage.
        end_stage_name: The name of the end stage.

    Returns:
        A list of stages to replay.

    Raises:
        ValueError: If the end stages precedes the start stage in the pipeline, or if
            either stage is not found.

    """
    replay_stages: list[CuratorStage] = []
    started = False
    found_end = False

    for stage in stages:
        _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
        name = _stage.__class__.__name__

        # Check if this is the start stage
        if name == start_stage_name:
            started = True

        # Add stage if we've started collecting
        if started:
            replay_stages.append(_stage)

        # Check if this is the end stage
        if name == end_stage_name:
            if not started:
                msg = f"Stage {end_stage_name} is the first stage found, but it should be the last stage"
                raise ValueError(msg)
            found_end = True
            break

    if len(replay_stages) == 0:
        msg = f"No stages found to replay, stages {start_stage_name}, {end_stage_name} not present"
        raise ValueError(msg)

    if not found_end:
        msg = f"End stage {end_stage_name} not found in pipeline"
        raise ValueError(msg)

    return replay_stages


def run_stage_replay(  # noqa: PLR0913
    stages: list[CuratorStage],
    path: Path,
    limit: int,
    *,
    executor: StageExecutor | None = None,
    serializer: TaskSerializer | None = None,
    init_ray: bool = True,
) -> list[PipelineTask]:
    """Replay stages with task pickles.

    Args:
        stages: list of stages to replay.
        path: The path to the tasks directory that holds the task pickles.
        limit: The maximum number of tasks to load.
        executor: Stage executor to use. Defaults to RayStageExecutor.
        serializer: Task serializer to use. Defaults to PickleTaskSerializer.
        init_ray: Whether to initialize Ray. Set to False if Ray is already initialized.

    Returns:
        A list of task objects.

    """
    if len(stages) == 0:
        msg = "No stages to replay"
        raise ValueError(msg)

    _executor = executor if executor is not None else RayStageExecutor()
    _serializer = serializer if serializer is not None else PickleTaskSerializer()

    start_stage = stages[0]
    end_stage = stages[-1]
    logger.info(
        f"Running isolated stages {start_stage.__class__.__name__} -> {end_stage.__class__.__name__} "
        f"loading input tasks from {path}, {limit=}"
    )

    node_info, worker_metadata = NodeInfo(node_id="localhost"), WorkerMetadata.make_dummy()

    if init_ray and not ray.is_initialized():
        ray.init()

    task_batches = _load_task_batches(path, limit, _serializer)

    if len(task_batches) == 0:
        msg = f"No input tasks found in {path}"
        raise ValueError(msg)

    for stage in stages:
        task_batches = _executor.execute_stage(stage, task_batches, node_info, worker_metadata)

    return [task for batch in task_batches for task in batch]


def should_save_stage(stage: CuratorStage | CuratorStageSpec, config: StageSaveConfig) -> bool:
    """Check if the stage should be saved.

    Args:
        stage: The stage to check.
        config: Configuration for saving stages for replay.

    Returns:
        True if the stage should be saved, False otherwise.

    """
    _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
    return _stage.__class__.__name__ in config.stages


def stage_save_wrapper(
    stage: CuratorStage | CuratorStageSpec,
    config: StageSaveConfig,
) -> CuratorStage | CuratorStageSpec:
    """Wrap the process_data method of a stage so that it saves tasks.

    This function modifies the stage's class in place, so that the stage's
    state is preserved.

    The new class is a subclass of the stage's original class, and that class
    overrides process_data method to save tasks.

    Args:
        stage: The stage to wrap.
        config: Configuration for saving stages for replay.

    Returns:
        The stage or stage spec with the process_data method wrapped.

    """
    _stage = cast("CuratorStage", stage.stage) if isinstance(stage, CuratorStageSpec) else stage
    name = _stage.__class__.__name__

    logger.info(f"Wrapping process_data for stage {name} with path {config.path} and sample_rate {config.sample_rate}")

    # Swap the instance's class in place, keeping all attributes as-is.
    _stage.__class__ = _make_stage_save_class(_stage.__class__, config)

    if isinstance(stage, CuratorStage):
        return _stage

    stage.stage = _stage
    return stage
