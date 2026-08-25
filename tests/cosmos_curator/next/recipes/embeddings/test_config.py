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

"""Validation tests for the direct-``clips.lance`` embedding-recipe config and CLI translation."""

import argparse

import pytest
from pydantic import ValidationError

from cosmos_curator.next.embeddings.schemas import ACTION_DIM
from cosmos_curator.next.recipes.embeddings.config import _MIN_PCA_SAMPLE_SIZE, EmbeddingPipelineConfig, Modality
from cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline import _build_config

_CLIPS_URI = "s3://bucket/run/clips.lance"


def test_minimal_config_only_needs_clips_uri_and_defaults_all_modalities() -> None:
    """The smallest valid config only needs ``clips_lance_uri`` and enables all modalities."""
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI)
    assert cfg.modalities == (Modality.TEXT, Modality.IMAGE, Modality.ACTION)
    assert cfg.max_fragments is None


def test_modalities_are_deduplicated_order_preserved() -> None:
    """Duplicate modalities collapse, keeping first-seen order."""
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, modalities=["image", "text", "image"])
    assert cfg.modalities == (Modality.IMAGE, Modality.TEXT)


def test_empty_modalities_rejected() -> None:
    """An empty modality set is rejected - there would be nothing to run."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, modalities=[])


def test_invalid_modality_rejected() -> None:
    """A modality outside text/image/action is rejected by the StrEnum."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, modalities=["audio"])


def test_max_fragments_is_accepted() -> None:
    """``max_fragments`` (the smoke-test fragment cap) assembles cleanly."""
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, max_fragments=2)
    assert cfg.max_fragments == 2


def test_max_fragments_below_one_is_rejected() -> None:
    """A cap of zero fragments would visit nothing, so it is rejected at assembly."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, max_fragments=0)


def test_pca_sample_floor_is_derived_from_action_dim() -> None:
    """The sample floor tracks the action width so the two cannot drift."""
    assert _MIN_PCA_SAMPLE_SIZE == ACTION_DIM + 1


def test_pca_sample_size_below_floor_rejected() -> None:
    """pca_sample_size below the derived floor is rejected at config assembly."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, action={"pca_sample_size": _MIN_PCA_SAMPLE_SIZE - 1})


def test_padded_clips_uri_is_stored_trimmed() -> None:
    """Surrounding whitespace is stripped, so the stored URI differs from the configured text.

    Every downstream comparison - the artifact root derived from the clips URI, a
    log line, an operator diffing two runs - sees the trimmed value, so the
    stripping is part of the config's contract rather than an input convenience.
    """
    cfg = EmbeddingPipelineConfig(clips_lance_uri=f"  {_CLIPS_URI}\t")
    assert cfg.clips_lance_uri == _CLIPS_URI


def test_blank_clips_lance_uri_rejected() -> None:
    """A whitespace-only clips URI is rejected (min_length allows it)."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri="   ")


def test_blank_storage_profile_rejected() -> None:
    """A whitespace-only storage profile is rejected (min_length allows it)."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, storage_profile="  ")


def test_blank_model_weights_path_rejected() -> None:
    """A whitespace-only weights path is rejected (min_length allows it)."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, model_weights_path="   ")


def test_unknown_key_rejected() -> None:
    """The model forbids extra keys (extra='forbid'); a removed field name is now unknown."""
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, source_lance_uri=_CLIPS_URI)


def test_per_modality_resources_are_overridable_and_bounded() -> None:
    """Each modality's resource block accepts an override and rejects a negative value.

    Asserts the mechanism the config module owns rather than the numbers, which are
    tuning knobs: re-profiling a modality onto a different GPU fraction is a correct
    edit that should not redden this file.
    """
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, text={"num_gpus": 0.5})
    assert cfg.text.num_gpus == 0.5
    with pytest.raises(ValidationError):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, text={"num_gpus": -1.0})


def test_image_read_concurrency_above_batch_size_rejected() -> None:
    """A read width wider than the scan batch is rejected instead of silently clamped.

    The reader submits ``min(read_concurrency, len(pending))`` workers from one
    scan batch, so an over-wide value never takes effect. Explicit values are
    passed on both sides so re-tuning either default cannot redden this test.
    """
    # match= is load-bearing: with extra="forbid" a bare ValidationError also passes
    # when the field is simply gone, so the pattern is what proves the cross-field
    # rule fired rather than the key becoming unknown.
    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, image={"batch_size": 16, "read_concurrency": 17})


def test_image_read_concurrency_equal_to_batch_size_accepted() -> None:
    """The bound is inclusive: a read width exactly filling the scan batch is valid."""
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, image={"batch_size": 16, "read_concurrency": 16})
    assert cfg.image.read_concurrency == cfg.image.batch_size


@pytest.mark.parametrize("width", [0, -1])
def test_action_read_concurrency_below_one_rejected(width: int) -> None:
    """A read width below one worker is rejected: the leg would fetch nothing.

    match= pins the bound that fired: with extra="forbid" a bare ValidationError
    also passes when the field is simply gone.
    """
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, action={"read_concurrency": width})


def test_action_read_concurrency_above_batch_size_rejected() -> None:
    """A read width wider than the scan batch is rejected instead of silently clamped.

    The extractor draws its artifacts from one scan batch, so an over-wide value
    never takes effect. Explicit values are passed on both sides so re-tuning
    either default cannot redden this test.
    """
    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, action={"batch_size": 8, "read_concurrency": 9})


def test_action_read_concurrency_equal_to_batch_size_accepted() -> None:
    """The bound is inclusive: a read width exactly filling the scan batch is valid."""
    cfg = EmbeddingPipelineConfig(clips_lance_uri=_CLIPS_URI, action={"batch_size": 8, "read_concurrency": 8})
    assert cfg.action.read_concurrency == cfg.action.batch_size


def test_build_config_translates_cli_args() -> None:
    """CLI args map onto a validated config (modalities de-duplicated; ``max_fragments`` carried)."""
    args = argparse.Namespace(
        clips_lance_uri=_CLIPS_URI,
        modalities=["action", "action", "text"],
        storage_profile="default",
        model_weights_path=None,
        max_fragments=2,
    )
    config = _build_config(args)
    assert config.clips_lance_uri == _CLIPS_URI
    assert config.modalities == (Modality.ACTION, Modality.TEXT)
    assert config.max_fragments == 2
