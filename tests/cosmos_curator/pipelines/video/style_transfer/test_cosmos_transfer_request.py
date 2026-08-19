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

"""CPU tests for the in-process Cosmos3 transfer request builder + helpers.

These pin the vLLM-Omni ``extra_args`` transfer contract (hint object, on-the-fly
vs pre-computed control, transfer options) plus the Super multi-GPU clamp — no GPU
or vLLM-Omni engine required.
"""

from pathlib import Path

import pytest

from cosmos_curator.models.style_transfer import (
    Cosmos3OmniTransferModel,
    StyleTransferParams,
    build_transfer_extra_args,
    clamp_num_gpus_for_variant,
    style_transfer_variants,
)


def _params(control: str = "edge") -> StyleTransferParams:
    return StyleTransferParams(
        prompt="a watercolor painting",
        negative_prompt="blurry, low quality",
        control=control,
        control_guidance=1.5,
        guidance=3.0,
        seed=2026,
        resolution="720",
        fps=24,
        num_frames=121,
        num_video_frames_per_chunk=93,
        num_conditional_frames=1,
        edge_preset="high",
        blur_preset="low",
    )


def test_variants_are_nano_and_super() -> None:
    """The first implementation ships the nano + super Cosmos3 transfer variants."""
    assert set(style_transfer_variants()) == {"cosmos3_nano", "cosmos3_super"}


def test_vllm_omni_model_uses_style_transfer_environment() -> None:
    """The transfer backend runs where its isolated vLLM-Omni dependency is installed."""
    assert Cosmos3OmniTransferModel().conda_env_name == "style-transfer"


def test_edge_on_the_fly_request() -> None:
    """Edge control with no pre-computed path -> hint carries only the edge preset."""
    extra = build_transfer_extra_args(_params("edge"), control_paths=None)
    assert extra["edge"] == {"preset_edge_threshold": "high"}
    assert "control_path" not in extra["edge"]
    assert extra["control_guidance"] == 1.5
    assert extra["num_video_frames_per_chunk"] == 93
    assert extra["num_conditional_frames"] == 1
    assert extra["max_frames"] == 121
    # Resolution bucket must be present (read at both preprocess and in the worker).
    assert extra["resolution"] == "720"
    # Only the active control hint key is emitted (transfer.py keys on hint presence).
    assert "blur" not in extra


def test_blur_on_the_fly_request() -> None:
    """Blur control emits the blur strength preset under the 'blur' hint."""
    extra = build_transfer_extra_args(_params("blur"), control_paths=None)
    assert extra["blur"] == {"preset_blur_strength": "low"}
    assert "edge" not in extra


def test_precomputed_control_path_request(tmp_path: Path) -> None:
    """A pre-computed control video is forwarded as control_path on the hint."""
    control_edge = tmp_path / "control_edge.mp4"
    extra = build_transfer_extra_args(_params("edge"), control_paths={"edge": control_edge})
    assert extra["edge"]["control_path"] == str(control_edge)
    assert extra["edge"]["preset_edge_threshold"] == "high"


def test_precomputed_path_for_other_control_is_ignored(tmp_path: Path) -> None:
    """A control_path keyed to a different control does not leak into the active hint."""
    control_paths = {"blur": tmp_path / "control_blur.mp4"}
    extra = build_transfer_extra_args(_params("edge"), control_paths=control_paths)
    assert "control_path" not in extra["edge"]


@pytest.mark.parametrize(
    ("variant", "requested", "expected"),
    [
        ("cosmos3_nano", 1, 1),
        ("cosmos3_nano", 2, 2),
        ("cosmos3_super", 1, 4),
        ("cosmos3_super", 4, 4),
        ("cosmos3_super", 8, 8),
    ],
)
def test_clamp_num_gpus_for_variant(variant: str, requested: int, expected: int) -> None:
    """Super clamps up to its 4-GPU minimum; nano is left as-is (min 1)."""
    assert clamp_num_gpus_for_variant(variant, requested) == expected
