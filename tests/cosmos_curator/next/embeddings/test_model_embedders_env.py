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

"""Model-backed embedder smoke tests (env-gated; require staged weights).

The only tests that run a real model, so they are what pins the properties the
CPU suite must stub: the emitted width, L2 normalization, and the group's exact
field names. Everything about batching, null masking, and ordering is covered on
CPU in ``test_model_embedders.py``.

Marked ``env("default")`` so they run only under the GPU/weights task, not the
CPU suite. Each skips (rather than errors) when its weights are not staged, so a
GPU box without the model still reports a clear reason.
"""

import pathlib
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.next.embeddings.image.embedder import HfVisionImageEmbedder
from cosmos_curator.next.embeddings.model_specs import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL
from cosmos_curator.next.embeddings.schemas import (
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
)
from cosmos_curator.next.embeddings.text.embedder import SentenceTransformerTextEmbedder

_RGB_CHANNELS = 3


def _skip_unless_staged(weights_name: str) -> None:
    weights = get_local_dir_for_weights_name(weights_name)
    if not weights.exists():
        pytest.skip(f"weights for {weights_name} not staged at {weights}")


@pytest.mark.env("default")
def test_bge_text_embedder_produces_normalized_384d() -> None:
    """BGE emits two L2-normalized 384-d vectors per clip as the text group's fields."""
    _skip_unless_staged(DEFAULT_TEXT_MODEL.weights_name)
    embedder = SentenceTransformerTextEmbedder(spec=DEFAULT_TEXT_MODEL, encode_batch_size=8)
    batch = pa.table(
        {
            "task_name": ["pick up the block", "pick up the block"],
            "subtask_name": ["grasp the block", "release the block"],
        }
    )
    out = embedder(batch)
    assert out.schema.names == list(TEXT_COLUMN_GROUP.field_names)
    assert out.num_rows == 2
    vector = np.asarray(out.column("embedding_text_subtask")[0].as_py(), dtype=np.float32)
    assert vector.shape == (TEXT_DIM,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-4)


@pytest.mark.env("default")
def test_dinov2_image_embedder_produces_normalized_384d(make_clip: Callable[..., None], tmp_path: pathlib.Path) -> None:
    """DINOv2 emits one L2-normalized 384-d vector per readable clip as the image group's fields."""
    _skip_unless_staged(DEFAULT_IMAGE_MODEL.weights_name)
    clip = tmp_path / "clip.mp4"
    make_clip(clip, width=64, height=64)
    embedder = HfVisionImageEmbedder(spec=DEFAULT_IMAGE_MODEL)
    out = embedder(pa.table({"clip_uri": [str(clip)]}))
    assert out.schema.names == list(IMAGE_COLUMN_GROUP.field_names)
    assert out.num_rows == 1
    vector = np.asarray(out.column("embedding_image")[0].as_py(), dtype=np.float32)
    assert vector.shape == (IMAGE_DIM,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-4)


@pytest.mark.env("default")
def test_dinov2_keeps_unreadable_clip_as_a_null_row(tmp_path: pathlib.Path) -> None:
    """A missing clip yields an all-NULL group row, not a dropped row.

    The row must survive so the fill's positional ``clip_id`` attachment stays
    aligned and the clip is retried next run instead of being silently excluded.
    """
    _skip_unless_staged(DEFAULT_IMAGE_MODEL.weights_name)
    embedder = HfVisionImageEmbedder(spec=DEFAULT_IMAGE_MODEL)
    out = embedder(pa.table({"clip_uri": [str(tmp_path / "nope.mp4")]}))
    assert out.num_rows == 1
    assert out.column("embedding_image").is_valid().to_pylist() == [False]
    assert out.column("embedding_image_model_id").to_pylist() == [None]
