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

"""Model Cosmos-Embed1."""

from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils import conda_utils, model_utils

_COSMOS_EMBED1_VARIANTS_INFO: Final = {
    "224p": "nvidia/Cosmos-Embed1-224p",
    "336p": "nvidia/Cosmos-Embed1-336p",
    "448p": "nvidia/Cosmos-Embed1-448p",
}

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false
if conda_utils.is_running_in_env("unified"):
    from transformers import AutoModel, AutoProcessor


class CosmosEmbed1(ModelInterface):
    """Cosmos-Embed1 embedding model."""

    def __init__(self, *, variant: Literal["224p", "336p", "448p"] = "336p", utils_only: bool = False) -> None:
        """Initialize Cosmos-Embed1 model.

        Args:
            variant: Choose from "224p", "336p", "448p".
            utils_only: Whether to only initialize utility functions.

        """
        super().__init__()
        self.variant = variant
        self._weights_name = _COSMOS_EMBED1_VARIANTS_INFO[variant]
        self._weights_dir = str(model_utils.get_local_dir_for_weights_name(self._weights_name))
        self._utils_only = utils_only
        self._model: AutoModel | None = None

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [self._weights_name]

    @nvtx.annotate("Setup Cosmos-Embed1 model")  # type: ignore[misc]
    def setup(self) -> None:
        """Set up the Cosmos-Embed1 model.

        This method initializes the model and its configuration for processing video and text data.
        """
        logger.info("Setting up Cosmos-Embed1 model")
        if not Path(self._weights_dir).exists():
            exception = f"Weights directory {self._weights_dir} not found!"
            raise FileNotFoundError(exception)
        if not self._utils_only:
            self._model = AutoModel.from_pretrained(  # type: ignore[no-untyped-call]
                self._weights_dir,
                trust_remote_code=True,
                local_files_only=True,
            ).to("cuda", dtype=torch.bfloat16)
            assert self._model is not None
            self._model.eval()  # type: ignore[attr-defined]
        self._processor = AutoProcessor.from_pretrained(
            self._weights_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    def get_target_num_frames(self) -> int:
        """Get the target number of frames for the model.

        Returns:
            The target number of frames.

        """
        return cast("int", self._processor.num_video_frames)

    def formulate_input_frames(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32] | None:
        """Formulate input frames for the model.

        Args:
            frames: List of video frames.

        Returns:
            The formulated input frames.

        """
        fn = self.get_target_num_frames()
        if len(frames) < fn:
            logger.error(f"Frame count {len(frames)} is smaller than minimal requirement {fn}")
            return None
        step = len(frames) // fn
        video_batch = np.expand_dims(np.stack(frames[::step][:fn]), 0)
        video_batch = np.transpose(video_batch, (0, 1, 4, 2, 3))
        return cast(
            "npt.NDArray[np.float32]",
            self._processor(videos=video_batch, return_tensors="pt")["videos"].numpy(),
        )

    def encode_video_frames(self, frames: npt.NDArray[np.float32]) -> torch.Tensor:
        """Encode video frames for the model.

        Args:
            frames: The input video frames.

        Returns:
            The encoded video frames.

        """
        assert self._model is not None
        if frames.size == 0:
            return torch.empty((0, self._model.config.embed_dim), dtype=torch.float16)  # type: ignore[attr-defined]

        with torch.no_grad():
            videos = torch.from_numpy(frames).to("cuda", dtype=torch.bfloat16)
            output = self._model.get_video_embeddings(videos=videos)  # type: ignore[attr-defined]
            return cast("torch.Tensor", output.visual_proj.to("cpu", dtype=torch.float16))

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get the text embedding for the given text.

        Args:
            text: The input text.

        Returns:
            The text embedding.

        """
        assert self._model is not None
        batch = self._processor(text=[text], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            output = self._model.get_text_embeddings(**batch)  # type: ignore[attr-defined]
            return cast("torch.Tensor", output.text_proj.to("cpu", dtype=torch.float16))

    def evaluate(self, video_embd: torch.Tensor, text_embds: list[torch.Tensor]) -> tuple[list[float], list[int]]:
        """Evaluate the model.

        Args:
            video_embd: The video embedding.
            text_embds: The text embeddings.

        Returns:
            The predicted probabilities and indices.

        """
        count = len(text_embds)
        text_embds_tensor = torch.cat(text_embds, 0)
        assert self._model is not None
        label_probs = (100.0 * video_embd @ text_embds_tensor.T).softmax(dim=-1)
        probs, idxs = label_probs.float().cpu().topk(count, dim=-1)
        return probs.cpu().numpy()[0].tolist(), idxs.cpu().long().numpy()[0].tolist()
