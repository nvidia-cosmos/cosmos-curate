# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that Curator Next owns its Ray Data runtime configuration."""

import subprocess
import sys


def test_recipes_do_not_reach_into_the_deprecated_ray_data_package() -> None:
    """``next`` must stay importable once ``pipelines.ray_data`` is deleted."""
    probe = (
        "import sys;"
        "import cosmos_curator.next.recipes.video_split.pipeline;"
        "print([m for m in sys.modules if m.startswith('cosmos_curator.pipelines')])"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"
