# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Builders for the embeddings component tests.

Ray and the mecka action artifacts live in the parent ``conftest.py``, because
the recipe tests need them too.
"""

import pathlib
from collections.abc import Callable

import av
import numpy as np
import pytest

from cosmos_curator.next.embeddings.action.embedder import (
    DualWristMotionDescriptorExtractor,
    DualWristMotionReadConfig,
)
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM
from cosmos_curator.next.embeddings.schemas import ACTION_DIM

_RGB_CHANNELS = 3


@pytest.fixture
def extractor() -> DualWristMotionDescriptorExtractor:
    """Return a descriptor extractor with default read settings."""
    return DualWristMotionDescriptorExtractor(DualWristMotionReadConfig())


@pytest.fixture
def synthetic_pca() -> PcaArtifact:
    """Return a 600->97 basis built directly, with no fit.

    The projector only reads ``mean`` and ``components``, so a random basis
    exercises the projection without the cost or the row floor of a real fit.
    """
    rng = np.random.default_rng(0)
    return PcaArtifact(
        mean=np.zeros(DESCRIPTOR_DIM),
        components=rng.standard_normal((ACTION_DIM, DESCRIPTOR_DIM)),
    )


@pytest.fixture
def make_clip() -> Callable[..., None]:
    """Return a builder for a tiny synthetic mp4 that is discriminating on CPU.

    Two independent axes let a caller assert that decoding picked the right frame
    and preserved channel order, without any model weights:

    * per-frame brightness ramp -- frame ``i`` is brighter than frame ``i - 1``,
      so the first *displayable* frame is the darkest (``mean < 30``). Returning a
      later frame, or the first *decoded* frame of a reordered stream, is
      detectable.
    * a fixed red bias over a flat blue channel -- the red channel mean exceeds
      the blue channel mean by a wide margin, so an RGB/BGR channel swap is
      detectable. A uniform-grey clip is channel-symmetric and cannot catch it.

    ``max_b_frames`` > 0 emits a B-frame stream (decode order != display order)
    via ``libx264``; the default uses ``mpeg4`` with no reordering.
    """

    def build(
        path: pathlib.Path,
        *,
        frames: int = 3,
        width: int = 32,
        height: int = 32,
        max_b_frames: int = 0,
    ) -> None:
        # B-frames need an encoder that reorders; mpeg4 (no B-frames) is the default.
        codec = "libx264" if max_b_frames else "mpeg4"
        # Left-to-right red gradient (mean ~45) applied to the red channel only,
        # so R's mean sits well above B's flat mean even after lossy chroma.
        red_bias = np.linspace(10.0, 80.0, width, dtype=np.float32)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream(codec, rate=10)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.max_b_frames = max_b_frames
            for index in range(frames):
                brightness = float((index * 60) % 256)
                image = np.zeros((height, width, _RGB_CHANNELS), dtype=np.uint8)
                image[..., 0] = np.clip(brightness + red_bias, 0, 255).astype(np.uint8)
                image[..., 1] = np.uint8(brightness)
                image[..., 2] = np.uint8(brightness)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    return build
