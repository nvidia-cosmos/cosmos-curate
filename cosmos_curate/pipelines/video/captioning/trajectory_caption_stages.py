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

"""Trajectory Caption stage."""

import uuid

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage
from cosmos_curate.core.utils import storage_client, storage_utils
from cosmos_curate.core.utils.dataset_utils import webdataset_utils
from cosmos_curate.core.utils.runtime.performance_utils import StageTimer
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask


class ReadClipArchive(CuratorStage):
    """Stage for reading video clips from an archive.

    This class processes video clips through a series of steps including reading,
    extracting metadata, and storing the results in the task.
    """

    def __init__(
        self,
        input_path: str,
        input_s3_profile_name: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the clip archive reader.

        Args:
            input_path: Path to input archive.
            input_s3_profile_name: S3 profile name for input.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._input_path = input_path
        self._input_s3_profile_name = input_s3_profile_name
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: storage_client.StorageClient | None = None

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._client = storage_utils.get_storage_client(
            target_path=self._input_path,
            profile_name=self._input_s3_profile_name,
        )

    @nvtx.annotate("ReadClipArchive")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Read clips from the archive.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            assert self._client is not None
            bytes_ = self._client.download_object_as_bytes(
                video.input_video,  # type: ignore[arg-type]
                chunk_size_bytes=storage_client.DOWNLOAD_CHUNK_SIZE_BYTES,
            )
            samples = webdataset_utils.read_raw_samples_from_archive(bytes_, "tar")
            clip_id = str(video.input_video).split("/")[-1].split(".")[0]
            egomotion = {}
            for sample in samples:
                if sample.key == f"{clip_id}.egomotion" and "npz" in sample.data:
                    egomotion["egomotion.npz"] = sample.data["npz"]
                if sample.key == f"{clip_id}.live_egomotion" and "npz" in sample.data:
                    egomotion["live_egomotion.npz"] = sample.data["npz"]
                if sample.key == f"{clip_id}.rig" and "json" in sample.data:
                    egomotion["rig.json"] = sample.data["json"]
                if sample.key == f"{clip_id}.camera_front_wide_120fov" and "json" in sample.data:
                    egomotion["camera_front_wide_120fov.json"] = sample.data["json"]
            for sample in samples:
                if "mp4" in sample.data:
                    view_id = sample.key.split(".")[-1]
                    is_trajectory_base_view = view_id == "camera_front_wide_120fov"
                    video.clips.append(
                        Clip(
                            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{video.input_video}_{view_id}"),
                            source_video=str(video.input_video),
                            span=(0, 0),
                            buffer=sample.data["mp4"],
                            egomotion=egomotion if is_trajectory_base_view else None,  # type: ignore[arg-type]
                        ),
                    )

            if self._verbose:
                logger.info(f"Read {video.input_video}: #clips={len(video.clips)}")

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
