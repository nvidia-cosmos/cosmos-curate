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
"""Cosmos-Embed1 embedding stage."""

import io
from typing import Literal

import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models.cosmos_embed1 import CosmosEmbed1
from cosmos_curate.pipelines.video.utils.data_model import (
    SplitPipeTask,
)
from cosmos_curate.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    extract_frames,
)


class CosmosEmbed1FrameCreationStage(CuratorStage):
    """Stage for creating Cosmos-Embed1 input frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    model initialization, and input frame creation.
    """

    def __init__(
        self,
        variant: Literal["224p", "336p", "448p"] = "336p",
        *,
        target_fps: float = 2.0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the Cosmos-Embed1 frame creation stage.

        Args:
            variant: Variant of Cosmos-Embed1 model to use.
            target_fps: Target frames per second for frame extraction.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._variant = variant
        self._target_fps = target_fps
        self._extraction_policy = FrameExtractionPolicy
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self._target_fps,
        ).to_str()
        # utils_only set to true to skip initializing the actual model
        self._model = CosmosEmbed1(variant=variant, utils_only=True)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def model(self) -> ModelInterface:
        """Get the Cosmos-Embed1 model.

        Returns:
            The Cosmos-Embed1 model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=1.0)

    @nvtx.annotate("CosmosEmbed1FrameCreationStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to create Cosmos-Embed1 input frames.

        Args:
            tasks: Tasks containing video to process.

        Returns:
            Processed task with Cosmos-Embed1 input frames.

        """
        max_fps: int = 20

        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                if clip.encoded_data is None:
                    clip.errors["encoded_data"] = "empty"
                    continue
                if self._frame_extraction_signature not in clip.extracted_frames:
                    clip.errors[f"frames-{self._frame_extraction_signature}"] = "missing"
                    logger.error(f"Clip {clip.uuid} has buffer but no extracted frames for ???")
                    continue
                with self._timer.time_process():
                    frames = clip.extracted_frames[self._frame_extraction_signature]
                    # check if we need re-extract
                    target_num_frames = self._model.get_target_num_frames()
                    regen_fps = self._target_fps
                    while frames.shape[0] < target_num_frames:
                        regen_fps *= 2
                        if regen_fps > max_fps:
                            logger.error(f"Clip {clip.uuid} is too short to extract enough frames.")
                            break
                        if self._verbose:
                            logger.warning(
                                f"Clip {clip.uuid} has <{target_num_frames} frames. "
                                f"Re-extracting with higher target_fps={regen_fps}. "
                                f"Current # frames={frames.shape[0]}.",
                            )
                        with io.BytesIO(clip.encoded_data) as fp:
                            frames = extract_frames(
                                fp,
                                extraction_policy=FrameExtractionPolicy.sequence,
                                sample_rate_fps=regen_fps,
                            )
                    # create input frames for Cosmos-Embed1 model
                    clip.cosmos_embed1_frames = self._model.formulate_input_frames(list(frames))
                # done with extracted_frames
                clip.extracted_frames.clear()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class CosmosEmbed1EmbeddingStage(CuratorStage):
    """Stage for generating embeddings from Cosmos-Embed1 input frames.

    This class processes video clips through a series of steps including embedding generation,
    text verification, and memory management.
    """

    def __init__(
        self,
        variant: Literal["224p", "336p", "448p"] = "336p",
        num_gpus_per_worker: float = 0.25,
        *,
        verbose: bool = False,
        log_stats: bool = False,
        texts_to_verify: list[str] | None = None,
    ) -> None:
        """Initialize the Cosmos-Embed1 embedding stage.

        Args:
            variant: Variant of Cosmos-Embed1 model to use.
            num_gpus_per_worker: Number of GPUs per worker.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            texts_to_verify: Optional list of texts to verify against embeddings.

        """
        self._timer = StageTimer(self)
        self._variant = variant
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._texts_to_verify = texts_to_verify
        self._model = CosmosEmbed1(variant=variant, utils_only=False)
        self._process_count = 0

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Get the Cosmos-Embed1 model.

        Returns:
            The Cosmos-Embed1 model.

        """
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    @nvtx.annotate("CosmosEmbed1EmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to generate embeddings.

        Args:
            tasks: Tasks containing video to process.

        Returns:
            Processed task with generated embeddings.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            with self._timer.time_process(len(video.clips)):
                for clip in video.clips:
                    if clip.cosmos_embed1_frames is None:
                        clip.errors["cosmos_embed1_frames"] = "empty"
                        continue
                    embedding = self._model.encode_video_frames(clip.cosmos_embed1_frames)
                    if embedding.numel() == 0:
                        logger.error(f"Unable to compute cosmos-embed1 embedding for clip={clip.uuid}")
                        clip.errors["cosmos_embed1_embedding"] = "failed"
                    else:
                        clip.cosmos_embed1_embedding = embedding.cpu().numpy()
                    if self._texts_to_verify:
                        text_embeddings = [self._model.get_text_embedding(x) for x in self._texts_to_verify]
                        probs, idxs = self._model.evaluate(embedding, text_embeddings)
                        clip.cosmos_embed1_text_match = (
                            self._texts_to_verify[idxs[0]],
                            probs[0],
                        )
                    # done with cosmos_embed1_frames
                    clip.cosmos_embed1_frames = None
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # free memory periodically
        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks
