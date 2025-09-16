"""Example image inference pipeline."""

import attrs
import numpy as np
import torch

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curate.models.image_model import ImageInferenceModel


# pipeline task object that is being passed between stages
@attrs.define
class ImageInferenceTask(PipelineTask):
    input_parquet_path: str
    metadata: dict[str, str | float] | None = None
    image_bytes: list[bytes] | None = None
    image_tensors: list[np.ndarray] | None = None
    outputs: list[np.ndarray] | None = None
    labels: list[str] | None = None


class DownloadStage(CuratorStage):
    def __init__(self, cameras: list[str]) -> None:
        self._cameras = cameras
        self._client = BlobStorageClient()

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=0.5, gpus=0.0)

    def process_data(self, tasks: list[ImageInferenceTask]) -> list[ImageInferenceTask] | None:
        # fill up metadata and download raw bytes
        for task in tasks:
            task.metadata = {"id": some_id, "timestamp": some_timestamp}
            task.image_bytes = self._client.download_bytes(self.input_parquet_path)
        # passing to next stage
        return tasks


class PreprocessStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=2.0, gpus=0.0)

    def process_data(self, tasks: list[ImageInferenceTask]) -> list[ImageInferenceTask] | None:
        # convert raw bytes to tensors
        for task in tasks:
            task.image_tensors = [preprocess_image_cpu(x) for x in task.image_bytes]
        return tasks


class InferenceStage(CuratorStage):
    def __init__(self) -> None:
        self._model = ImageInferenceModel()

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.5)

    @property
    def model(self) -> ModelInterface | None:
        return self._model

    def stage_setup(self):
        self._model.setup()

    def process_data(self, tasks: list[ImageInferenceTask]) -> list[ImageInferenceTask] | None:
        # run inference on GPU
        for task in tasks:
            task.outputs = self._model.generate(torch.from_numpy(np.stack(task.image_tensors)).to("cuda"))
        return tasks


class PostprocessStage(CuratorStage):
    def __init__(self) -> None:
        pass

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[ImageInferenceTask]) -> list[ImageInferenceTask] | None:
        # postprocess the output labels
        for task in tasks:
            task.labels = [postprocess_label_cpu(x) for x in task.outputs]
        return tasks


class UploadStage(CuratorStage):
    def __init__(self) -> None:
        self._client = BlobStorageClient()

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=0.5, gpus=0.0)

    def process_data(self, tasks: list[ImageInferenceTask]) -> list[ImageInferenceTask] | None:
        # upload the labels
        for task in tasks:
            self._client.upload_labels(task.labels, task.metadata)
        return tasks


def main(input_paths) -> None:
    # build a list of input tasks
    input_tasks = [ImageInferenceTask(input_parquet_path=path) for path in input_paths]

    # define the pipeline stages
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(DownloadStage(cameras=["camera_1", "camera_2"]), num_workers_per_node=8),
        PreprocessStage(),
        InferenceStage(),
        PostprocessStage(),
        CuratorStageSpec(UploadStage(), num_workers_per_node=4),
    ]

    run_pipeline(input_tasks, stages)


if __name__ == "__main__":
    example_input_paths = ["path/to/parquet1", "path/to/parquet2"]
    main(example_input_paths)
