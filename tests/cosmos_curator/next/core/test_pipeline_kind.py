# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusable pipeline-kind lookup and validation."""

from unittest.mock import Mock

import pytest

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelineKindRegistry


def _kind(name: str) -> PipelineKind:
    return PipelineKind(
        name=name,
        template_yaml=Mock(),
        template_payload=Mock(),
        validate=Mock(),
        render=Mock(),
        schema_json=Mock(),
        list_presets=Mock(),
        prepare_run=Mock(),
    )


def test_registry_lookup_and_iteration_are_stable() -> None:
    """Registry behavior is independent of concrete pipeline implementations."""
    alpha = _kind("alpha")
    zeta = _kind("zeta")
    registry = PipelineKindRegistry((zeta, alpha))

    assert registry.names() == ("alpha", "zeta")
    assert tuple(registry) == (alpha, zeta)
    assert registry.get("zeta") is zeta


def test_registry_rejects_duplicate_names() -> None:
    """Duplicate discriminators must fail at the composition root."""
    with pytest.raises(ValueError, match="Duplicate pipeline kind: duplicate"):
        PipelineKindRegistry((_kind("duplicate"), _kind("duplicate")))


def test_registry_rejects_empty_names() -> None:
    """An empty discriminator cannot be represented in a config."""
    with pytest.raises(ValueError, match="Pipeline kind names must not be empty"):
        PipelineKindRegistry((_kind(""),))


def test_registry_error_lists_valid_names() -> None:
    """Unknown-name errors expose the complete sorted discriminator set."""
    registry = PipelineKindRegistry((_kind("alpha"), _kind("zeta")))

    with pytest.raises(ValueError, match=r"Unknown pipeline kind 'missing'\. Valid pipeline kinds: alpha, zeta"):
        registry.get("missing")
