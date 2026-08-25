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

"""Dual-wrist motion descriptor (pure numpy/scipy geometry, no Ray/GPU/Lance).

Turns a clip's per-frame hand + camera pose arrays into a fixed 600-d motion
descriptor. Per arm, the wrist (joint 0) track is composed out of the moving
camera frame, anchored to frame 0, then split into a centered/unit-ball position
channel and an unwrapped-rotvec rotation channel, and arc-length resampled to 50
stations. Two arms give ``2 x 50 x (3 pos + 3 rot) = 600``. The full step-by-step
derivation lives in the design doc, section 7
(``docs/curator/design/curator-next-embeddings.md``); keeping it there avoids a
line-by-line copy that any edit to the chain would silently falsify.

Quaternions are **xyzw** (scalar-last), consumed with no reordering because that
is ``scipy.spatial.transform.Rotation``'s default component order. PCA reduction
lives in ``pca`` and the Ray wiring in ``embedder``.
"""

import enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]


class DescriptorRejection(enum.StrEnum):
    """Why one clip produced no wrist-motion descriptor.

    Returned in place of the ``(600,)`` vector so a drop is attributable to a
    specific cause instead of an undifferentiated ``None``. A ``StrEnum`` member
    is neither ``None`` nor falsy, so a caller MUST branch on the type (see
    ``DescriptorResult``) rather than an ``is None`` / truthiness test.
    """

    MISSING_CAMERA = "missing_camera"
    MALFORMED_CAMERA = "malformed_camera"
    MISSING_ARM = "missing_arm"
    MALFORMED_ARM = "malformed_arm"
    LENGTH_MISMATCH = "length_mismatch"
    TOO_FEW_VALID_FRAMES = "too_few_valid_frames"


# A descriptor build either yields the (600,) vector or a typed reason it did not.
type DescriptorResult = npt.NDArray[np.float32] | DescriptorRejection

# Descriptor geometry (fixed). 2 arms x 50 samples x (3 pos + 3 rot) = 600.
_N_SAMPLES = 50
_HAND_JOINTS = 21  # dataset-specific: mecka egocentric hand skeleton is 21 joints.
_WRIST = 0  # wrist is joint index 0 (the hand skeleton's origin).
_POS_DIMS = 3
_QUAT_DIMS = 4
_ROT_DIMS = 3
_MIN_QUAT_NORM = 0.05  # no-hand frames carry ~0-norm quaternions; drop them.
# A wrist track needs enough surviving frames to describe a path, not merely to
# admit an arc length: with only a couple of survivors the resampler interpolates
# a full-amplitude straight line indistinguishable from real motion. Compared
# with >=, so a track with exactly this many survivors is kept.
_MIN_VALID_FRAMES = 8
# Reject a track whose survivors are a small slice of the clip, where masking has
# spliced a discontinuous path back together. Evaluated per arm against the
# PRE-MASK arm length, and compared with >= so exactly half is kept.
_MIN_VALID_FRACTION = 0.5
# Smallest wrist track extent (meters) treated as real motion. A track whose max
# centered radius is below this is motionless within sensor precision; dividing
# by such a radius would amplify jitter to fill the unit ball and fabricate a
# full-scale noise descriptor for a hand that never moved.
_MIN_TRACK_RADIUS_M = 1e-4
# Smallest total arc length treated as motion in the resampler. Unlike the radius
# above, this is NOT in meters: the resampler's input is hstack([pos, rot]) with
# pos already divided by its own max radius (dimensionless, max 1) and rot in
# radians, so the threshold lives in that mixed space. Because positions are
# pre-normalized, rotation dominates station spacing by roughly pi-to-2; that
# weighting is intentional -- wrist orientation carries more gesture identity
# than translation extent. Kept equal to the radius value so the split from a
# single shared constant is behavior-neutral.
_MIN_TRACK_ARC_LENGTH = 1e-4
# Below this rotation angle (radians) an incremental rotvec step is treated as
# identity: the previous unwrapped vector is repeated rather than accumulating
# numerical noise at a stationary frame.
_ROTVEC_UNWRAP_MIN_ANGLE = 1e-8

DESCRIPTOR_DIM = 2 * _N_SAMPLES * (_POS_DIMS + _ROT_DIMS)  # 600

# Fingerprint of the descriptor's SEMANTICS, not its width. A persisted PCA basis
# records the version it was fit under, so a later run whose code carries a
# different version refuses to reuse that basis (see action/pca.py) rather than
# projecting new-semantics vectors onto an old, dimensionally-valid one. v3 adds
# rotvec unwrapping along each track, which changes the rotation channels (and,
# via arc-length resampling, the position channels too) versus the prior aligned
# chain -- the width stays 600 but the semantics do not, which is exactly the
# case this fingerprint exists to catch.
DESCRIPTOR_VERSION = "dual-wrist-v3"

# 90-degree CCW rotation about local Z, post-multiplied onto the wrist pose so
# the wrist-local frame becomes X = thumb->pinky, Y = palm normal (outward),
# Z = wrist->fingertips. These are the mecka dataset's own wrist-frame values,
# replicated here as a literal because the dataset config cannot be imported
# across packages. Its exact values are pinned by a test - a silent edit would
# change every action vector. Made read-only immediately below so an importer
# cannot mutate the shared array process-wide (the test pins the source literal,
# not the runtime object).
WRIST_FRAME_ALIGN_MECKA: npt.NDArray[np.float64] = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
WRIST_FRAME_ALIGN_MECKA.setflags(write=False)

# (position-key, rotation-key) per arm; both arms are required for the 600-d shape.
_ARM_KEYS: tuple[tuple[str, str], ...] = (
    ("hand_left_cam", "hand_left_cam_rotation"),
    ("hand_right_cam", "hand_right_cam_rotation"),
)
_CAM_POS_KEY = "camera_position"
_CAM_ROT_KEY = "camera_rotation"


def _poses_from_pos_quat(pos: npt.NDArray[np.float64], quat_xyzw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Build ``(N, 4, 4)`` homogeneous poses from translation and xyzw quaternion.

    Quaternions are xyzw (scipy's default component order), consumed with no
    reordering.
    """
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, _QUAT_DIMS)
    mats = np.asarray(Rotation.from_quat(quat).as_matrix(), dtype=np.float64)
    poses = np.tile(np.eye(4, dtype=np.float64), (mats.shape[0], 1, 1))
    poses[:, :3, :3] = mats
    poses[:, :3, 3] = np.asarray(pos, dtype=np.float64).reshape(-1, _POS_DIMS)
    return poses


def _rigid_inverse(pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the exact inverse of a single ``(4, 4)`` rigid transform.

    Uses the closed form ``(R.T, -R.T @ t)`` - exact for a rotation+translation,
    avoiding any conditioning question a general ``inv`` would raise.
    """
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def _center_unit_ball(pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Center a position track and scale it into the unit ball (offset/scale invariant).

    A track whose extent is below ``_MIN_TRACK_RADIUS_M`` is motionless within
    sensor precision, so it is returned centered rather than divided by its tiny
    radius (which would amplify jitter into a full-scale noise descriptor).
    """
    centered: npt.NDArray[np.float64] = pos - pos.mean(axis=0)
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius < _MIN_TRACK_RADIUS_M:
        return centered
    scaled: npt.NDArray[np.float64] = centered / radius
    return scaled


def _arc_length_resample(seq: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """Resample an ``(N, C)`` track to ``n`` stations uniform in arc length.

    Parametrizing by cumulative arc length (not frame index) makes the descriptor
    speed/pause invariant; endpoints are preserved and a track below
    ``_MIN_TRACK_ARC_LENGTH`` collapses to its first sample. Channels interpolate
    linearly (a similarity summary, not a replay; SLERP would need a version bump).
    """
    segment_lengths = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total < _MIN_TRACK_ARC_LENGTH:
        return np.repeat(seq[:1], n, axis=0)
    stations = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(stations, cumulative, seq[:, col]) for col in range(seq.shape[1])])


def _wrist_pos_quat(
    pos_raw: npt.ArrayLike, rot_raw: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | DescriptorRejection:
    """Extract the wrist (joint 0) position/quaternion track for one arm.

    Reshapes the flat per-frame arrays to the 21-joint skeleton and selects
    joint 0. Returns ``MALFORMED_ARM`` for a non-numeric payload, a width that is
    not a whole number of 21-joint frames, or mismatched position/rotation frame
    counts (all three are corrupt-artifact conditions, not the wrist-vs-camera
    length mismatch, which ``LENGTH_MISMATCH`` reserves). Validity (finite /
    non-zero quaternion) is applied later, jointly with the camera track.
    """
    try:
        pos_all = np.asarray(pos_raw, dtype=np.float64)
        rot_all = np.asarray(rot_raw, dtype=np.float64)
        pos = pos_all.reshape(-1, _HAND_JOINTS, _POS_DIMS)[:, _WRIST, :]
        quat = rot_all.reshape(-1, _HAND_JOINTS, _QUAT_DIMS)[:, _WRIST, :]
    except (TypeError, ValueError):
        return DescriptorRejection.MALFORMED_ARM
    if pos.shape[0] != quat.shape[0]:
        return DescriptorRejection.MALFORMED_ARM
    return pos, quat


def _camera_pos_quat(
    cam_pos_raw: npt.ArrayLike | None, cam_rot_raw: npt.ArrayLike | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | DescriptorRejection:
    """Extract the ego-camera ``(N, 3)`` position and ``(N, 4)`` xyzw quaternion track.

    A missing key returns ``MISSING_CAMERA`` (rejected explicitly, rather than
    relying on ``np.asarray(None)`` producing a 0-d NaN that raises on reshape); a
    non-numeric payload or mismatched position/rotation counts return
    ``MALFORMED_CAMERA``.
    """
    if cam_pos_raw is None or cam_rot_raw is None:
        return DescriptorRejection.MISSING_CAMERA
    try:
        cpos = np.asarray(cam_pos_raw, dtype=np.float64).reshape(-1, _POS_DIMS)
        cquat = np.asarray(cam_rot_raw, dtype=np.float64).reshape(-1, _QUAT_DIMS)
    except (TypeError, ValueError):
        return DescriptorRejection.MALFORMED_CAMERA
    if cpos.shape[0] != cquat.shape[0]:
        return DescriptorRejection.MALFORMED_CAMERA
    return cpos, cquat


def _finite_unit_quat_mask(pos: npt.NDArray[np.float64], quat: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return a per-frame mask of finite positions and finite, non-degenerate quaternions."""
    # Drop near-zero-norm quaternions BEFORE any pose is built: scipy's from_quat
    # silently normalizes, so a no-hand (~0-norm) quaternion would otherwise become
    # a full-magnitude rotation of sensor noise. A non-finite position would also
    # survive centering (radius -> inf/NaN) and poison the descriptor.
    mask = (
        np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > _MIN_QUAT_NORM)
    )
    return np.asarray(mask, dtype=np.bool_)


def _unwrap_rotvecs(rotvecs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Integrate relative rotation deltas into a continuous rotvec track.

    ``as_rotvec`` returns ``theta*n`` with ``theta`` in ``[0, pi]``, so a track
    whose relative rotation crosses ``pi`` flips sign and jumps by ~``2*pi``, and
    motion past one full turn can fold back near zero::

        theta:   178 deg      179.9   |   180.1      182
        rotvec:  +3.107       +3.140  |  -3.140     -3.107
                                 branch cut

    Each step adds ``(R_{i-1}^{-1} R_i).as_rotvec()`` to the previous unwrapped
    vector. That incremental rotvec is the true relative rotation, so the track
    stays continuous across branch cuts and multiple turns before arc-length
    resampling. Steps below ``_ROTVEC_UNWRAP_MIN_ANGLE`` repeat the previous
    vector to avoid amplifying numerical noise at identity.
    """
    if rotvecs.shape[0] <= 1:
        return rotvecs.copy()
    out = rotvecs.copy()
    rotations = Rotation.from_rotvec(out)
    for i in range(1, out.shape[0]):
        delta = np.asarray((rotations[i - 1].inv() * rotations[i]).as_rotvec(), dtype=np.float64)
        if float(np.linalg.norm(delta)) <= _ROTVEC_UNWRAP_MIN_ANGLE:
            out[i] = out[i - 1]
            continue
        out[i] = out[i - 1] + delta
    return out


def _arm_feature(
    pos_raw: npt.ArrayLike,
    rot_raw: npt.ArrayLike,
    camera: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    *,
    align_mecka: bool,
) -> npt.NDArray[np.float64] | DescriptorRejection:
    """Build one arm's ``(50, 6)`` feature, or a ``DescriptorRejection`` if unusable.

    Applies the descriptor chain (design doc section 7): pose -> optional mecka
    alignment -> compose out the moving camera frame -> frame-0 anchor ->
    centered/scaled position + rotvec -> arc-length resample. Rejection reasons:
    ``MALFORMED_ARM`` (bad hand arrays, from ``_wrist_pos_quat``),
    ``LENGTH_MISMATCH`` (wrist and camera tracks differ in length), or
    ``TOO_FEW_VALID_FRAMES`` (the absolute floor or the surviving-fraction gate).
    """
    wrist = _wrist_pos_quat(pos_raw, rot_raw)
    if isinstance(wrist, DescriptorRejection):
        return wrist
    wpos, wquat = wrist
    cpos, cquat = camera
    # Camera pose multiplies the wrist pose frame-by-frame, so the two tracks
    # must have the same length before a shared mask is applied.
    if wpos.shape[0] != cpos.shape[0]:
        return DescriptorRejection.LENGTH_MISMATCH
    pre_mask_frames = wpos.shape[0]

    mask = _finite_unit_quat_mask(wpos, wquat) & _finite_unit_quat_mask(cpos, cquat)
    wpos, wquat, cpos, cquat = wpos[mask], wquat[mask], cpos[mask], cquat[mask]
    # Two independent gates. The absolute floor rejects a track with too few
    # points to have a shape; the fraction gate rejects a track that has enough
    # points but whose masked-out gaps mean the "path" between survivors is
    # fiction. Neither is expressible as the other.
    survivors = wpos.shape[0]
    if survivors < _MIN_VALID_FRAMES:
        return DescriptorRejection.TOO_FEW_VALID_FRAMES
    if survivors < _MIN_VALID_FRACTION * pre_mask_frames:
        return DescriptorRejection.TOO_FEW_VALID_FRAMES

    wrist_cam = _poses_from_pos_quat(wpos, wquat)
    cam_pose = _poses_from_pos_quat(cpos, cquat)
    if align_mecka:
        wrist_cam = wrist_cam @ WRIST_FRAME_ALIGN_MECKA
    wrist_static = cam_pose @ wrist_cam
    # Frame-0 anchor: left-multiply every pose by the inverse of the first, so
    # the track is expressed relative to the clip's opening wrist pose. This
    # cancels any global rigid transform of the world frame (including the
    # vendor's absolute start pose), which is the invariance the descriptor needs
    # - "what did the wrist do", not "where in the world did it start".
    wrist_rel = _rigid_inverse(wrist_static[0]) @ wrist_static

    pos = _center_unit_ball(wrist_rel[:, :3, 3])
    # Unwrap before hstack so the arc-length parametrization sees continuous
    # rotvecs; a 2*pi branch-cut jump would otherwise dominate the segment sum
    # (see _unwrap_rotvecs) and corrupt the shared station placement.
    rot = _unwrap_rotvecs(np.asarray(Rotation.from_matrix(wrist_rel[:, :3, :3]).as_rotvec(), dtype=np.float64))
    arm = np.hstack([pos, rot])  # (N, 6)
    return _arc_length_resample(arm, _N_SAMPLES)  # (50, 6)


def dual_wrist_motion_descriptor(payload: dict[str, Any], *, align_mecka: bool) -> DescriptorResult:
    """Build the 600-d dual-wrist motion descriptor for one clip, or a rejection.

    Args:
        payload: A decoded action payload providing the four hand arrays
            (``hand_left_cam`` / ``hand_right_cam`` and their ``*_rotation``)
            plus the ego-camera pose (``camera_position`` / ``camera_rotation``).
        align_mecka: Apply ``WRIST_FRAME_ALIGN_MECKA``, the wrist-frame correction
            the mecka layout needs. The caller decides; the embedding leg passes
            True because every artifact it consumes uses that layout.

    Returns:
        A ``(DESCRIPTOR_DIM,)`` float32 descriptor, or a
        :class:`DescriptorRejection` naming why the clip was rejected. Both arms
        are required; a clip is rejected rather than yielding a partial vector.

    """
    camera = _camera_pos_quat(payload.get(_CAM_POS_KEY), payload.get(_CAM_ROT_KEY))
    if isinstance(camera, DescriptorRejection):
        return camera

    parts: list[npt.NDArray[np.float64]] = []
    for pos_key, rot_key in _ARM_KEYS:
        pos_raw = payload.get(pos_key)
        rot_raw = payload.get(rot_key)
        if pos_raw is None or rot_raw is None:
            return DescriptorRejection.MISSING_ARM
        arm = _arm_feature(pos_raw, rot_raw, camera, align_mecka=align_mecka)
        if isinstance(arm, DescriptorRejection):
            return arm
        parts.append(arm)
    return np.concatenate(parts).reshape(-1).astype(np.float32)  # (600,)
