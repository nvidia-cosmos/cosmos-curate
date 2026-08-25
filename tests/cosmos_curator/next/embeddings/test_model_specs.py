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

"""Contract tests for the frozen model specs and the vision pooling reduction.

CPU-only: no staged weights or GPU. The registry checks read all_models.json;
the pooling and dim-guard checks drive the vision embedder's ``_pool`` / ``_embed``
against synthetic torch tensors via a bare instance (``object.__new__``), so no
checkpoint is loaded and the model-specific reductions are pinned directly.
"""

import json
import pathlib
from typing import Any

import numpy as np
import pytest
import torch

import cosmos_curator
from cosmos_curator.next.embeddings.image.embedder import HfVisionImageEmbedder
from cosmos_curator.next.embeddings.model_specs import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    TextModelSpec,
    VisionModelSpec,
    VisionPooling,
)

_SPECS = (DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL)


def _registry() -> dict[str, Any]:
    """Load ``all_models.json`` from the installed package."""
    path = pathlib.Path(cosmos_curator.__file__).parent / "configs" / "all_models.json"
    return json.loads(path.read_text("utf-8"))


class _Outputs:
    """Stand-in for a transformers vision model output (only the fields ``_pool`` reads)."""

    def __init__(
        self, *, pooler_output: torch.Tensor | None = None, last_hidden_state: torch.Tensor | None = None
    ) -> None:
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state


def _bare_vision_embedder(spec: VisionModelSpec) -> HfVisionImageEmbedder:
    """Build an embedder with only the attributes ``_pool`` / ``_embed`` need (no weights)."""
    embedder = object.__new__(HfVisionImageEmbedder)
    embedder._spec = spec
    embedder._torch = torch
    embedder._device = "cpu"
    return embedder


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.weights_name)
def test_spec_weights_name_and_model_id_match_registry(spec: VisionModelSpec | TextModelSpec) -> None:
    """Each spec's weights_name resolves in all_models.json and its model_id matches that entry."""
    registry = _registry()
    assert spec.weights_name in registry, f"{spec.weights_name} not in all_models.json"
    assert registry[spec.weights_name]["model_id"] == spec.model_id


def test_pooler_output_none_raises_naming_spec_and_suggesting_cls_token() -> None:
    """POOLER_OUTPUT against a backbone that returns None names the spec and suggests CLS_TOKEN."""
    embedder = _bare_vision_embedder(DEFAULT_IMAGE_MODEL)  # default pooling is POOLER_OUTPUT
    outputs = _Outputs(pooler_output=None, last_hidden_state=torch.zeros(2, 5, DEFAULT_IMAGE_MODEL.dim))
    with pytest.raises(ValueError, match=DEFAULT_IMAGE_MODEL.weights_name) as excinfo:
        embedder._pool(outputs)
    assert "CLS_TOKEN" in str(excinfo.value)


def test_cls_token_pooling_takes_the_first_token() -> None:
    """CLS_TOKEN pooling returns position 0 of the token sequence."""
    spec = VisionModelSpec("dinov2_small", "facebook/dinov2-small", 4, VisionPooling.CLS_TOKEN)
    embedder = _bare_vision_embedder(spec)
    last_hidden_state = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    pooled = embedder._pool(_Outputs(last_hidden_state=last_hidden_state))
    torch.testing.assert_close(pooled, last_hidden_state[:, 0])


def test_mean_tokens_pooling_averages_over_the_token_axis() -> None:
    """MEAN_TOKENS pooling averages the token axis (dim 1)."""
    spec = VisionModelSpec("dinov2_small", "facebook/dinov2-small", 4, VisionPooling.MEAN_TOKENS)
    embedder = _bare_vision_embedder(spec)
    last_hidden_state = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    pooled = embedder._pool(_Outputs(last_hidden_state=last_hidden_state))
    torch.testing.assert_close(pooled, last_hidden_state.mean(dim=1))


def test_dim_disagreement_fails_on_first_batch_naming_spec() -> None:
    """A spec whose dim disagrees with the backbone's real width is rejected, naming the spec.

    Drives ``_embed`` with a fake processor + model so the width check runs on the
    first batch without loading any checkpoint.
    """
    real_width = 4
    spec = VisionModelSpec("dinov2_small", "facebook/dinov2-small", 999, VisionPooling.CLS_TOKEN)
    embedder = _bare_vision_embedder(spec)

    class _FakeInputs(dict[str, Any]):
        def to(self, _device: object) -> "_FakeInputs":
            return self

    class _FakeProcessor:
        def __call__(self, **_kwargs: object) -> _FakeInputs:
            return _FakeInputs()

    class _FakeModel:
        def __call__(self, **_kwargs: object) -> _Outputs:
            return _Outputs(last_hidden_state=torch.zeros(2, 3, real_width))

    embedder._processor = _FakeProcessor()
    embedder._model = _FakeModel()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match=spec.weights_name):
        embedder._embed(frames)
