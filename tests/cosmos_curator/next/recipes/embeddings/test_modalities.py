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

"""Coverage for the driver-side translation of config into a modality's fill spec.

A modality's tuning knobs are validated on the driver but consumed inside a Ray
worker, so a value that stops at the config object is silently ignored by the run
that was tuned. These tests assert the translation itself - a set value reaches the
worker's constructor arguments or its actor shape - and deliberately never assert a
default, which is a tuning decision that should be free to change.

``ModalityFill`` types the embedder as a bare ``type`` and its arguments as
``dict[str, Any]``, so the ``embedder_cls(**embedder_kwargs)`` the worker performs
is unchecked by mypy. One test per modality therefore binds the shipped kwargs
against the named constructor's real signature, which is the only place that
mismatch can be caught before a Ray worker raises it.
"""

import inspect
from collections.abc import Callable

import numpy as np
import pytest

from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM
from cosmos_curator.next.embeddings.schemas import ACTION_DIM
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig
from cosmos_curator.next.recipes.embeddings.modalities import (
    ModalityFill,
    build_action_fill,
    build_image_fill,
    build_text_fill,
)

from .conftest import EmbeddingsConfigFactory


def _synthetic_pca() -> PcaArtifact:
    """Return a zero basis of the right shape; the binding test never projects with it."""
    return PcaArtifact(
        mean=np.zeros(DESCRIPTOR_DIM, dtype=np.float64),
        components=np.zeros((ACTION_DIM, DESCRIPTOR_DIM), dtype=np.float64),
    )


_FILL_BUILDERS: dict[str, Callable[[EmbeddingPipelineConfig], ModalityFill]] = {
    "action": lambda config: build_action_fill(config, _synthetic_pca()),
    "image": build_image_fill,
    "text": build_text_fill,
}


def test_image_fill_carries_the_configured_read_width_to_the_worker(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured read concurrency arrives in the embedder's constructor arguments.

    The knob only has an effect inside the worker, so a value that never leaves the
    config would leave every run reading serially while reporting the tuned width.
    """
    config = make_embeddings_config(image={"read_concurrency": 7})

    assert build_image_fill(config).embedder_kwargs["read_concurrency"] == 7


def test_action_fill_carries_the_configured_read_width_to_the_worker(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured read concurrency arrives in the action reader's constructor config.

    The width is consumed inside the worker's extract call, so a value that stops
    at the recipe config would leave every run fetching artifacts one at a time
    while reporting the tuned width.
    """
    config = make_embeddings_config(action={"read_concurrency": 7})

    fill = build_action_fill(config, _synthetic_pca())

    assert fill.embedder_kwargs["config"].read_concurrency == 7


def test_image_fill_carries_the_configured_cpu_reservation_to_the_actor_shape(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured CPU reservation arrives in the worker's resource request.

    Only a value that reaches ``WorkerResources`` is passed to Ray. Left in the
    config the actor would be scheduled with no CPU reservation at all, which is
    the oversubscription the field exists to prevent.
    """
    config = make_embeddings_config(image={"num_cpus": 3.0})

    assert build_image_fill(config).resources.num_cpus == 3.0


@pytest.mark.parametrize("modality", sorted(_FILL_BUILDERS))
def test_fill_kwargs_satisfy_the_embedder_constructor(
    modality: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Every kwarg a fill ships binds to a real parameter of the embedder it names.

    The worker calls ``embedder_cls(**embedder_kwargs)`` through untyped fields, so
    a rename on either side is invisible until a Ray actor raises ``TypeError``
    after loading its model. Binding the signature covers every kwarg the fill
    carries, including ones added later, rather than one name at a time.
    """
    fill = _FILL_BUILDERS[modality](make_embeddings_config())

    inspect.signature(fill.embedder_cls).bind(**fill.embedder_kwargs)
