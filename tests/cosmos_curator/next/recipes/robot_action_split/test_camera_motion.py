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

"""Unit tests for the ported camera-trajectory-to-natural-language module."""

from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from cosmos_curator.next.recipes.robot_action_split import camera_motion
from cosmos_curator.next.recipes.robot_action_split.camera_motion import (
    compute_camera_motion_annotation,
    describe_camera_motion,
)

_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def test_describe_camera_motion_still_for_short_trajectory() -> None:
    """A single-frame trajectory is too short to describe and reports as still."""
    positions = np.zeros((1, 3), dtype=np.float32)
    rotations = np.array([_IDENTITY_QUAT], dtype=np.float32)
    assert describe_camera_motion(positions, rotations) == "The camera remains relatively still."


def test_describe_camera_motion_still_for_stationary_camera() -> None:
    """A camera with no position or orientation change describes as still."""
    n = 30
    positions = np.zeros((n, 3), dtype=np.float32)
    rotations = np.tile(_IDENTITY_QUAT, (n, 1)).astype(np.float32)
    description = describe_camera_motion(positions, rotations, fmt="raw", fps=15.0)
    assert "still" in description


def test_describe_camera_motion_detects_dolly() -> None:
    """A straight-line translation along the forward axis is described as a dolly."""
    n = 60
    # Well above CAM_TRANSLATION_THRESH over the clip.
    positions = np.array([[0.0, i * 0.02, 0.0] for i in range(n)], dtype=np.float32)
    rotations = np.tile(_IDENTITY_QUAT, (n, 1)).astype(np.float32)
    description = describe_camera_motion(positions, rotations, fmt="raw", fps=15.0)
    assert "dollies" in description.lower()


def test_describe_camera_motion_detects_pan_without_gimbal_lock() -> None:
    """A steady yaw sweep should register as a pan, including when pitched steeply down.

    Regression guard for the gimbal-lock failure mode called out in i4's
    design doc: Euler-angle decomposition mis-attributed yaw as roll when the
    camera pointed steeply downward. This uses vector-based rotation matrices
    throughout, so it should not reproduce that failure.
    """
    n = 60
    positions = np.zeros((n, 3), dtype=np.float32)
    # Compose a steep downward pitch with a steady yaw sweep across the clip.
    yaw_degrees = np.linspace(0.0, 90.0, n)
    rotations = np.array(
        [R.from_euler("zx", [yaw, -80.0], degrees=True).as_quat() for yaw in yaw_degrees],
        dtype=np.float32,
    )
    description = describe_camera_motion(positions, rotations, fmt="raw", fps=15.0)
    assert "pan" in description.lower()
    assert "roll" not in description.lower()


def test_describe_camera_motion_holds_direction_across_zero_norm_smoothed_vector() -> None:
    """A single exact-zero-norm smoothed camera-forward vector must not fabricate motion.

    This can happen when the Gaussian smoothing window averages near-opposing
    directions (e.g. a rapid ~180 deg spin) to a near-zero vector at one frame.
    Two wrong fixes are checked against here, not just "doesn't crash":

    * Unguarded re-normalization produces NaN; since tilt is an accumulated
      `np.sum()` over every frame's delta, a single NaN frame poisons the whole
      sum (`nan > threshold` is False), silently dropping a real, sustained
      tilt spanning the entire clip.
    * Substituting a zero *vector* for the degenerate frame is not neutral
      either: `arcsin(0)` reads as a real heading of 0 at that frame, which
      then differences against its steady-tilting neighbors into a phantom
      swing away from and back to the true heading — a fabricated pan/tilt in
      each direction, not the fix.

    The correct behavior (holding the last valid direction across the
    degenerate frame, so its delta is ~0) should make the output identical to
    a clean run with no degenerate frame at all: same tilt magnitude, and no
    phantom pan. A realistic trajectory only gets close to zero (not exact),
    so this directly forces the zero-norm case via a patched
    ``gaussian_filter1d`` (called, in order, for positions, then camera-right,
    then camera-forward) to exercise the guard deterministically.
    """
    n = 60
    positions = np.zeros((n, 3), dtype=np.float32)
    pitch_degrees = np.linspace(0.0, 60.0, n)
    rotations = np.array(
        [R.from_euler("x", pitch, degrees=True).as_quat() for pitch in pitch_degrees],
        dtype=np.float32,
    )

    baseline = describe_camera_motion(positions, rotations, fmt="raw", fps=15.0)
    assert "tilt" in baseline.lower()  # sanity: the clean trajectory does have real tilt to lose

    real_filter = camera_motion.gaussian_filter1d
    calls = {"n": 0}

    def fake_filter(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
        calls["n"] += 1
        result = real_filter(arr, sigma=sigma, axis=axis)
        if calls["n"] in (2, 3):  # camera-right, then camera-forward smoothing
            result = result.copy()
            result[30] = 0.0  # force one frame's smoothed vector to exact zero
        return result

    with patch.object(camera_motion, "gaussian_filter1d", side_effect=fake_filter):
        description = describe_camera_motion(positions, rotations, fmt="raw", fps=15.0)

    assert "nan" not in description.lower()
    assert "pan" not in description.lower()  # no fabricated pan from the degenerate frame
    assert description == baseline  # degenerate frame is fully invisible: same tilt, nothing extra


def test_compute_camera_motion_annotation_none_for_missing_trajectory() -> None:
    """Missing or empty camera_position/camera_rotation yields no annotation."""
    assert compute_camera_motion_annotation(None, None, fps=15.0) is None
    assert compute_camera_motion_annotation([], [], fps=15.0) is None


def test_compute_camera_motion_annotation_none_for_too_few_frames() -> None:
    """A single-frame trajectory (below describe_camera_motion's minimum) yields no annotation."""
    assert (
        compute_camera_motion_annotation(
            [[0.0, 0.0, 0.0]],
            [_IDENTITY_QUAT],
            fps=15.0,
        )
        is None
    )


def test_compute_camera_motion_annotation_none_for_malformed_shapes() -> None:
    """Mismatched camera_position/camera_rotation frame counts yield no annotation."""
    assert (
        compute_camera_motion_annotation(
            [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [_IDENTITY_QUAT],
            fps=15.0,
        )
        is None
    )


def test_compute_camera_motion_annotation_returns_string_for_valid_trajectory() -> None:
    """A well-formed multi-frame trajectory yields a non-empty annotation string."""
    n = 60
    camera_position = [[0.0, i * 0.02, 0.0] for i in range(n)]
    camera_rotation = [_IDENTITY_QUAT for _ in range(n)]
    annotation = compute_camera_motion_annotation(camera_position, camera_rotation, fps=15.0)
    assert isinstance(annotation, str)
    assert annotation
