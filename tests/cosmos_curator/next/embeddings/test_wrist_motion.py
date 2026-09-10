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

"""Reference-parity and invariance tests for the dual-wrist motion descriptor.

The descriptor is the highest-risk math in the leg and has no working reference
in this repo, so these tests pin the transform order and the invariances it is
supposed to buy: head-motion (camera composition) and global rigid transform
(frame-0 anchoring).
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from cosmos_curator.next.embeddings.action.wrist_motion import (
    _MIN_VALID_FRAMES,
    DESCRIPTOR_DIM,
    WRIST_FRAME_ALIGN_MECKA,
    DescriptorRejection,
    _poses_from_pos_quat,
    _unwrap_rotvecs,
    dual_wrist_motion_descriptor,
)

_QUAT_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw


def _poses(positions: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) poses from positions and axis-angle rotation vectors."""
    mats = Rotation.from_rotvec(rotvecs).as_matrix()
    poses = np.tile(np.eye(4), (positions.shape[0], 1, 1))
    poses[:, :3, :3] = mats
    poses[:, :3, 3] = positions
    return poses


def _decompose(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split (N, 4, 4) poses into (positions, xyzw quaternions)."""
    pos = poses[:, :3, 3]
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()  # xyzw
    return pos, quat


def _hand_arrays(wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embed a wrist track at joint 0 of a 21-joint skeleton (others zero-filled)."""
    n = wrist_pos.shape[0]
    hand_cam = np.zeros((n, 63), dtype=np.float32)
    hand_cam[:, 0:3] = wrist_pos
    hand_rot = np.zeros((n, 84), dtype=np.float32)
    hand_rot[:, 0:4] = wrist_quat
    return hand_cam, hand_rot


def _payload(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> dict[str, Any]:
    """Assemble a decoded action payload from per-arm wrist tracks + camera pose."""
    lh, lr = _hand_arrays(*left)
    rh, rr = _hand_arrays(*right)
    return {
        "hand_left_cam": lh,
        "hand_left_cam_rotation": lr,
        "hand_right_cam": rh,
        "hand_right_cam_rotation": rr,
        "camera_position": cam_pos.astype(np.float32),
        "camera_rotation": cam_quat.astype(np.float32),
    }


def _moving_wrist(n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a smoothly moving, non-degenerate wrist track (positions + xyzw quats)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    positions = np.stack([t, 0.5 * np.sin(t * 3.0), 0.2 * t], axis=1) + 0.01 * rng.standard_normal((n, 3))
    rotvecs = np.stack([0.3 * t, 0.1 * t, 0.2 * np.cos(t)], axis=1)
    quats = Rotation.from_rotvec(rotvecs).as_quat()
    return positions, quats


def _moving_poses(n: int, *, seed: int) -> np.ndarray:
    """Build a smoothly moving, non-degenerate (N, 4, 4) pose sequence."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    positions = np.stack([t, 0.5 * np.sin(t * 3.0), 0.2 * t], axis=1) + 0.01 * rng.standard_normal((n, 3))
    rotvecs = np.stack([0.3 * t, 0.1 * t, 0.2 * np.cos(t)], axis=1) + 0.01 * rng.standard_normal((n, 3))
    return _poses(positions, rotvecs)


def _static_camera(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a stationary camera at the origin (identity orientation)."""
    return np.zeros((n, 3)), np.tile(_QUAT_IDENTITY, (n, 1))


def _rotation_sweep_payload(n: int, total_deg: float, axis: np.ndarray) -> dict[str, Any]:
    """Payload whose both wrists rotate by ``total_deg`` about ``axis`` over n frames.

    Position is static and the camera identity, so the descriptor's arc length is
    driven purely by the rotation channel - which is what isolates the rotvec
    branch-cut behaviour at 180 degrees.
    """
    unit_axis = axis / np.linalg.norm(axis)
    angles = np.deg2rad(total_deg) * np.linspace(0.0, 1.0, n)
    quats = Rotation.from_rotvec(angles[:, None] * unit_axis[None, :]).as_quat()
    arm = (np.zeros((n, 3)), quats)
    return _payload(arm, arm, *_static_camera(n))


def test_descriptor_dim_is_600() -> None:
    """2 arms x 50 stations x (3 pos + 3 rot) = 600."""
    assert DESCRIPTOR_DIM == 600


def test_wrist_frame_align_mecka_exact_values() -> None:
    """Pin the alignment matrix - it is replicated as a literal, not imported."""
    expected = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    assert np.array_equal(WRIST_FRAME_ALIGN_MECKA, expected)


def test_wrist_frame_align_mecka_is_read_only() -> None:
    """The shared alignment matrix is read-only, so an importer cannot corrupt every vector."""
    assert WRIST_FRAME_ALIGN_MECKA.flags.writeable is False


def test_pose_reads_quaternion_as_xyzw() -> None:
    """A 90-degree xyzw rotation about +Z maps +X to +Y (wxyz would not)."""
    quat_z90 = np.array([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])  # xyzw
    pose = _poses_from_pos_quat(np.zeros((1, 3)), quat_z90)
    rotated = pose[0, :3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_valid_clip_returns_600d_float32() -> None:
    """A well-formed clip yields a (600,) float32 descriptor."""
    n = 16
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    descriptor = dual_wrist_motion_descriptor(payload, align_mecka=True)
    assert descriptor is not None
    assert descriptor.shape == (600,)
    assert descriptor.dtype == np.float32


def test_deterministic() -> None:
    """Identical input yields a bit-identical descriptor."""
    n = 16
    payload = _payload(_moving_wrist(n, seed=3), _moving_wrist(n, seed=4), *_static_camera(n))
    a = dual_wrist_motion_descriptor(payload, align_mecka=True)
    b = dual_wrist_motion_descriptor(payload, align_mecka=True)
    assert a is not None
    np.testing.assert_array_equal(a, b)


def test_head_motion_invariance() -> None:
    """The same hand world-trajectory under two camera motions yields one descriptor.

    Constructs ``wrist_cam = inv(camera) @ W`` so ``camera @ wrist_cam == W`` for
    any camera; the composed (head-removed) trajectory is identical, so the
    descriptor must be too. This fails on a camera-frame descriptor.
    """
    n = 16
    world_left = _moving_poses(n, seed=10)
    world_right = _moving_poses(n, seed=11)

    def descriptor_under(camera: np.ndarray) -> np.ndarray:
        cam_inv = np.linalg.inv(camera)
        left = _decompose(cam_inv @ world_left)
        right = _decompose(cam_inv @ world_right)
        cam_pos, cam_quat = _decompose(camera)
        out = dual_wrist_motion_descriptor(_payload(left, right, cam_pos, cam_quat), align_mecka=False)
        assert out is not None
        return out

    camera_a = _poses(np.zeros((n, 3)), np.zeros((n, 3)))
    camera_b = _moving_poses(n, seed=12)  # a wildly different head motion
    # The identity holds to float accumulation only; the measured residual is
    # ~6e-8, so 1e-6 keeps 10x headroom for a different BLAS while still failing a
    # partial regression (a mis-ordered composition lands near 1e-5).
    np.testing.assert_allclose(descriptor_under(camera_a), descriptor_under(camera_b), atol=1e-6)


def test_global_rigid_transform_invariance() -> None:
    """A random rigid transform of the whole composed trajectory leaves it unchanged.

    Left-multiplying the camera pose by ``G`` sends ``wrist_static -> G @
    wrist_static``; frame-0 anchoring cancels ``G`` exactly, so the vendor's
    world-frame convention is irrelevant to the descriptor.
    """
    n = 16
    left = _moving_wrist(n, seed=20)
    right = _moving_wrist(n, seed=21)
    cam = _moving_poses(n, seed=22)
    cam_pos, cam_quat = _decompose(cam)
    base = dual_wrist_motion_descriptor(_payload(left, right, cam_pos, cam_quat), align_mecka=False)

    rng = np.random.default_rng(99)
    g = np.eye(4)
    g[:3, :3] = Rotation.from_rotvec(rng.standard_normal(3)).as_matrix()
    g[:3, 3] = rng.standard_normal(3)
    tcam_pos, tcam_quat = _decompose(g @ cam)
    transformed = dual_wrist_motion_descriptor(_payload(left, right, tcam_pos, tcam_quat), align_mecka=False)

    assert base is not None
    assert transformed is not None
    # Measured residual ~9e-8; 1e-6 leaves 10x headroom. See the head-motion test
    # for why the looser 1e-4 would hide a partial regression.
    np.testing.assert_allclose(base, transformed, atol=1e-6)


def test_quaternion_sign_invariance() -> None:
    """Negating every quaternion (q == -q under double cover) is a no-op."""
    n = 16
    lp, lq = _moving_wrist(n, seed=30)
    rp, rq = _moving_wrist(n, seed=31)
    cam_pos, cam_quat = _static_camera(n)
    base = dual_wrist_motion_descriptor(_payload((lp, lq), (rp, rq), cam_pos, cam_quat), align_mecka=True)
    negated = dual_wrist_motion_descriptor(_payload((lp, -lq), (rp, -rq), cam_pos, -cam_quat), align_mecka=True)
    assert base is not None
    assert negated is not None
    # Exact, not approximate: the sign is normalized before any arithmetic, so the
    # two paths run identical float operations. A tolerance here would hide a
    # canonicalization applied too late.
    np.testing.assert_array_equal(base, negated)


def test_rotation_descriptor_is_continuous_across_180_degrees() -> None:
    """A 2-degree change in total sweep produces the same descriptor delta either side of 180.

    Regression for the rotvec branch cut: ``as_rotvec`` jumps by ~2*pi as the
    accumulated rotation crosses pi, so without unwrapping a 179 vs 181 sweep lands
    roughly 180x farther apart than the 120 vs 122 control. The unwrap restores the
    straddling delta to the control magnitude.
    """
    axis = np.array([0.3, 0.5, 0.8])
    n = 16

    def delta(a_deg: float, b_deg: float) -> float:
        da = dual_wrist_motion_descriptor(_rotation_sweep_payload(n, a_deg, axis), align_mecka=False)
        db = dual_wrist_motion_descriptor(_rotation_sweep_payload(n, b_deg, axis), align_mecka=False)
        assert da is not None
        assert db is not None
        return float(np.abs(da - db).max())

    straddle = delta(179.0, 181.0)
    control = delta(120.0, 122.0)
    # With the unwrap the two deltas match closely; without it the ratio is ~180x,
    # so a 3x bound passes the fix and fails loudly on a regression.
    assert straddle < 3.0 * control


def test_unwrap_preserves_rotation() -> None:
    """Each unwrapped rotvec maps back to the same rotation matrix as the original.

    This is what proves the fix is information-preserving rather than merely
    smoothing: it selects the antipodal representative, which denotes the identical
    rotation, so no geometry is altered.
    """
    axis = np.array([0.0, 0.0, 1.0])
    # A continuous sweep across 180 degrees. as_rotvec (the real code path)
    # canonicalizes to [0, pi] and introduces the branch cut the unwrap repairs.
    true_mats = Rotation.from_rotvec(np.deg2rad(np.linspace(150.0, 210.0, 12))[:, None] * axis[None, :]).as_matrix()
    canonical = np.asarray(Rotation.from_matrix(true_mats).as_rotvec(), dtype=np.float64)
    unwrapped = _unwrap_rotvecs(canonical)
    np.testing.assert_allclose(Rotation.from_rotvec(unwrapped).as_matrix(), true_mats, atol=1e-9)
    # The canonical form did wrap at the cut, and the unwrap repaired it.
    assert not np.allclose(canonical, unwrapped)


def test_unwrap_is_continuous_across_multiple_turns() -> None:
    """A multi-turn sweep stays step-wise continuous and preserves each rotation.

    ``as_rotvec`` folds angles past ``2*pi`` back into ``[0, pi]``; integrating
    relative deltas keeps the unwrapped track monotonic in angle before resampling.
    """
    axis = np.array([0.0, 0.0, 1.0])
    n = 32
    angles = np.linspace(0.0, 4.0 * np.pi, n)
    true_mats = Rotation.from_rotvec(angles[:, None] * axis[None, :]).as_matrix()
    canonical = np.asarray(Rotation.from_matrix(true_mats).as_rotvec(), dtype=np.float64)
    unwrapped = _unwrap_rotvecs(canonical)
    np.testing.assert_allclose(Rotation.from_rotvec(unwrapped).as_matrix(), true_mats, atol=1e-9)
    step_norms = np.linalg.norm(np.diff(unwrapped, axis=0), axis=1)
    assert float(step_norms.max()) < np.pi


def test_mecka_alignment_changes_the_descriptor() -> None:
    """Applying WRIST_FRAME_ALIGN_MECKA is not a silent identity.

    Verifies more than "not an identity": a constant *left*-multiplication would
    cancel exactly under frame-0 anchoring (``inv(A*W0) @ (A*Wi) == inv(W0)@Wi``),
    so the mecka matrix is only observable because it acts on the *right*, as a
    wrist-local frame change.
    """
    n = 16
    payload = _payload(_moving_wrist(n, seed=40), _moving_wrist(n, seed=41), *_static_camera(n))
    aligned = dual_wrist_motion_descriptor(payload, align_mecka=True)
    unaligned = dual_wrist_motion_descriptor(payload, align_mecka=False)
    assert aligned is not None
    assert unaligned is not None
    assert not np.allclose(aligned, unaligned)


def test_motionless_track_is_not_scaled_up() -> None:
    """A static hand (no motion) does not become a full-scale noise descriptor."""
    n = 16
    static_pos = np.tile([0.1, 0.2, 0.3], (n, 1))
    static_quat = np.tile(_QUAT_IDENTITY, (n, 1))
    payload = _payload((static_pos, static_quat), (static_pos, static_quat), *_static_camera(n))
    descriptor = dual_wrist_motion_descriptor(payload, align_mecka=True)
    assert descriptor is not None
    # Exactly zero, not merely small: a static track has zero arc length, so the
    # radius guard suppresses normalization rather than dividing by a tiny number.
    # A tolerance would pass even if the guard were removed and the result were
    # unit-scale noise several orders of magnitude below 1e-3.
    #
    # The exactness relies on n being a power of two: mean() of n identical floats
    # is bit-exact only when n*v and the division by n round trivially. A non-power
    # of-two frame count could leave centered ~1e-16 and flip this to failing.
    np.testing.assert_array_equal(descriptor, np.zeros_like(descriptor))


def test_missing_arm_is_rejected() -> None:
    """A clip missing one arm's arrays is rejected as MISSING_ARM (no partial vector)."""
    n = 16
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    del payload["hand_right_cam"]
    assert dual_wrist_motion_descriptor(payload, align_mecka=True) is DescriptorRejection.MISSING_ARM


def test_missing_camera_is_rejected() -> None:
    """A clip without camera pose is rejected as MISSING_CAMERA (cannot compose the camera frame)."""
    n = 16
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    del payload["camera_position"]
    assert dual_wrist_motion_descriptor(payload, align_mecka=True) is DescriptorRejection.MISSING_CAMERA


def test_camera_frame_count_mismatch_is_rejected() -> None:
    """A wrist track longer than the camera track is rejected as LENGTH_MISMATCH.

    LENGTH_MISMATCH is reserved for this operator-actionable case (re-export the
    artifact); a within-camera pos/quat count mismatch is MALFORMED_CAMERA instead.
    """
    n = 16
    left = _moving_wrist(n, seed=1)
    right = _moving_wrist(n, seed=2)
    cam_pos, cam_quat = _static_camera(n - 1)  # one short
    result = dual_wrist_motion_descriptor(_payload(left, right, cam_pos, cam_quat), align_mecka=True)
    assert result is DescriptorRejection.LENGTH_MISMATCH


def test_too_few_valid_frames_is_rejected() -> None:
    """A single-frame track is rejected as TOO_FEW_VALID_FRAMES (below the floor, no segment)."""
    lp, lq = _moving_wrist(1, seed=1)
    rp, rq = _moving_wrist(1, seed=2)
    cam_pos, cam_quat = _static_camera(1)
    result = dual_wrist_motion_descriptor(_payload((lp, lq), (rp, rq), cam_pos, cam_quat), align_mecka=True)
    assert result is DescriptorRejection.TOO_FEW_VALID_FRAMES


def test_zero_norm_quaternions_are_dropped() -> None:
    """No-hand (~0-norm) quaternion frames are dropped before pose construction.

    Sized so the surviving-frame count sits clear of the validity floor rather than
    on it: a fixture at the boundary tests the boundary by accident, and reddens
    whenever the floor is retuned. The floor and fraction rejections have their own
    tests below.
    """
    n = 16
    lp, lq = _moving_wrist(n, seed=1)
    lq = lq.copy()
    lq[9:] = 0.0  # 9 of 16 frames survive -- above the floor and the 0.5 fraction
    rp, rq = _moving_wrist(n, seed=2)
    cam_pos, cam_quat = _static_camera(n)
    out = dual_wrist_motion_descriptor(_payload((lp, lq), (rp, rq), cam_pos, cam_quat), align_mecka=True)
    assert out is not None


def test_track_at_frame_floor_is_accepted() -> None:
    """A clean track with exactly _MIN_VALID_FRAMES frames is kept (compared with >=)."""
    n = _MIN_VALID_FRAMES
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    assert dual_wrist_motion_descriptor(payload, align_mecka=True) is not None


def test_track_below_frame_floor_is_rejected() -> None:
    """A clean track one frame below the floor is TOO_FEW_VALID_FRAMES: too few points for a shape."""
    n = _MIN_VALID_FRAMES - 1
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    assert dual_wrist_motion_descriptor(payload, align_mecka=True) is DescriptorRejection.TOO_FEW_VALID_FRAMES


def test_track_passing_floor_but_failing_fraction_is_rejected() -> None:
    """Enough absolute survivors, too small a surviving fraction: the gaps make the path fiction.

    9 survivors clears the absolute floor of 8 but is below 0.5 * 20, so the
    fraction gate - which the absolute floor cannot express - rejects it.
    """
    n = 20
    lp, lq = _moving_wrist(n, seed=1)
    lq = lq.copy()
    lq[9:] = 0.0  # 9 survive: 9 >= 8 (floor) but 9 < 10 (0.5 * 20 fraction)
    rp, rq = _moving_wrist(n, seed=2)
    cam_pos, cam_quat = _static_camera(n)
    result = dual_wrist_motion_descriptor(_payload((lp, lq), (rp, rq), cam_pos, cam_quat), align_mecka=True)
    assert result is DescriptorRejection.TOO_FEW_VALID_FRAMES


def test_malformed_hand_width_is_rejected() -> None:
    """A hand array whose width is not a multiple of the 21-joint layout is MALFORMED_ARM."""
    n = 16
    payload = _payload(_moving_wrist(n, seed=1), _moving_wrist(n, seed=2), *_static_camera(n))
    payload["hand_left_cam"] = np.zeros((n, 61), dtype=np.float32)  # 61 is not 21*3
    assert dual_wrist_motion_descriptor(payload, align_mecka=True) is DescriptorRejection.MALFORMED_ARM


def test_non_finite_positions_are_dropped() -> None:
    """A NaN wrist frame is dropped, not zero-filled: masking equals physical removal.

    Asserting only finiteness would also pass an implementation that replaced the
    NaN with 0.0 -- a different, wrong descriptor. Comparing against the same track
    with the bad frame physically removed pins that the frame is genuinely dropped.
    """
    n = 16
    lp, lq = _moving_wrist(n, seed=1)
    rp, rq = _moving_wrist(n, seed=2)
    cam_pos, cam_quat = _static_camera(n)
    removed = dual_wrist_motion_descriptor(
        _payload((lp[1:], lq[1:]), (rp[1:], rq[1:]), cam_pos[1:], cam_quat[1:]),
        align_mecka=True,
    )
    lp_nan, rp_nan = lp.copy(), rp.copy()
    lp_nan[0, 0] = np.nan
    rp_nan[0, 0] = np.nan  # drop frame 0 of both arms so the surviving sets match
    masked = dual_wrist_motion_descriptor(
        _payload((lp_nan, lq), (rp_nan, rq), cam_pos, cam_quat),
        align_mecka=True,
    )
    assert removed is not None
    assert masked is not None
    np.testing.assert_allclose(masked, removed, atol=1e-6)
