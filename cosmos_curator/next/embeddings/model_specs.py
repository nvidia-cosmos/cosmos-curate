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

"""Frozen model specifications for the vision and text embedders.

A checkpoint is data, not code: the only model-specific facts a generic embedder
needs are which staged weights to load, which HuggingFace id to record as
provenance, the output width to assert, and - for a raw vision backbone - how to
reduce its output to one vector. Bundling those into a frozen spec lets one
embedder class host any model that satisfies the same library contract
(``AutoModel`` for vision, ``SentenceTransformer`` for text). A checkpoint swap is
then an explicit code change plus a table rebuild, never a silent runtime
override - the ``model_id`` recorded per row is only trustworthy if nothing can
point an embedder at other weights while keeping the id.

This module is intentionally dependency-light (only ``enum`` + ``attrs``: no
torch, no Ray, no transformers) so the pure embeddings package can re-export it
without breaking the import-layer purity guard; see
tests/cosmos_curator/next/embeddings/test_import_layers.py.
"""

import enum

import attrs


class VisionPooling(enum.StrEnum):
    """How to reduce a vision backbone's token output to one vector per image.

    ``pooler_output`` exists on every HF vision backbone but means different
    things per architecture (DINOv2 is the CLS token with no learned layer,
    SigLIP is attention-pooled, a bare ViT has a randomly-initialised pooler when
    the checkpoint ships none), so the reduction is a per-model choice with no
    safe default across architectures.
    """

    POOLER_OUTPUT = "pooler_output"
    CLS_TOKEN = "cls_token"  # noqa: S105 - an enum value, not a credential (the name ends in TOKEN)
    MEAN_TOKENS = "mean_tokens"


@attrs.frozen
class VisionModelSpec:
    """The model-specific surface of an ``AutoModel`` vision embedder.

    Attributes:
        weights_name: ``all_models.json`` key; resolves to the staged local dir.
        model_id: HuggingFace id, recorded per row as embedding provenance.
        dim: Expected output width, asserted against the first batch's real width.
        pooling: How ``_pool`` reduces the backbone output to one vector.

    """

    weights_name: str
    model_id: str
    dim: int
    pooling: VisionPooling = VisionPooling.POOLER_OUTPUT


@attrs.frozen
class TextModelSpec:
    """The model-specific surface of a ``SentenceTransformer`` text embedder.

    No pooling field: sentence-transformers bakes pooling into the checkpoint's
    module config, so unlike a raw vision backbone there is no per-call choice.

    Attributes:
        weights_name: ``all_models.json`` key; resolves to the staged local dir.
        model_id: HuggingFace id, recorded per row as embedding provenance.
        dim: Expected output width, asserted against the first batch's real width.

    """

    weights_name: str
    model_id: str
    dim: int


# A staged checkpoint the driver's weight-staging path needs by name. Both spec
# shapes expose ``weights_name`` / ``model_id`` / ``dim``; the union lets a GPU
# modality carry either without the recipe layer matching on which.
type ModelSpec = VisionModelSpec | TextModelSpec


# The default image / text checkpoints the embedding recipe ships with today,
# named by the modality they serve rather than the model so a checkpoint swap is a
# one-line edit here with no rename churn at the call sites. Adding a model (here
# and in all_models.json) is the whole cost of hosting a new one; test_model_specs.py
# asserts each weights_name resolves in the registry and its model_id matches.
DEFAULT_IMAGE_MODEL = VisionModelSpec("dinov2_small", "facebook/dinov2-small", 384)
DEFAULT_TEXT_MODEL = TextModelSpec("bge_small_en_v1_5", "BAAI/bge-small-en-v1.5", 384)
