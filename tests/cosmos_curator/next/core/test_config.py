# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Curator Next config resolution."""

import pytest

from cosmos_curator.next.core.config import apply_dotted_overrides


def test_dotted_overrides_parse_yaml_and_distinguish_empty_from_null() -> None:
    """Bare empty assignments remain distinct from explicit YAML nulls."""
    data: dict[str, object] = {}

    apply_dotted_overrides(
        data,
        [
            "execution.progress=true",
            "execution.workers=4",
            "empty=",
            "null=null",
        ],
    )

    assert data == {
        "execution": {"progress": True, "workers": 4},
        "empty": "",
        "null": None,
    }


@pytest.mark.parametrize("override", ["missing-value", "=value", "section..value=1"])
def test_dotted_overrides_reject_malformed_assignments(override: str) -> None:
    """Every assignment requires a non-empty dotted path and equals sign."""
    with pytest.raises(ValueError, match="--set override"):
        apply_dotted_overrides({}, [override])


def test_dotted_overrides_reject_paths_through_scalars() -> None:
    """An override cannot silently replace an existing scalar parent."""
    with pytest.raises(TypeError, match="non-object"):
        apply_dotted_overrides({"execution": False}, ["execution.progress=true"])
