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

"""Camera-trajectory-to-natural-language annotation for egocentric robot clips.

Ported from imaginaire4's egocentric captioning pipeline
(``pipelines/sila/video/captioning/egocentric_camera_conversion/trajectory_to_tags.py``).
Converts a per-frame camera trajectory (world-space position + quaternion
orientation) into a natural-language camera-motion description using
cinematographic terminology (pan, tilt, dolly, truck, pedestal). Combines
graduated thresholding with continuous frame-by-frame delta accumulation, and
uses vector math (rather than Euler angles) to avoid gimbal lock on
head-mounted cameras.

:func:`compute_camera_motion_annotation` is the entry point for
``robot_action_split``: it adapts the ``camera_position`` / ``camera_rotation``
lists already decoded in memory by ``processing.py``'s ``_slice_action`` into
the array shapes :func:`describe_camera_motion` expects, and returns ``None``
for a trajectory too short to describe (mirrors upstream's row-skip behavior
instead of raising).
"""

from typing import Any

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R  # noqa: N817

CAM_ROTATION_THRESH = 10.0  # Degrees minimum to mention rotation (tuned to keep only dominant motions)
CAM_TRANSLATION_THRESH = 0.15  # Metres minimum to mention translation (tuned to keep only dominant motions)
POS_NOISE_GATE = 0.01  # Per-frame position-shift noise gate (meters) at DEFAULT_FPS — scaled inversely with fps at call time so the equivalent wall-clock velocity threshold stays constant (0.01m/frame @ 15fps = 0.15 m/s).  # noqa: E501
ROT_NOISE_GATE = 0.5  # Per-frame rotation-shift noise gate (degrees) at DEFAULT_FPS — scaled inversely with fps at call time so the equivalent wall-clock angular velocity threshold stays constant (0.5°/frame @ 15fps = 7.5 °/s).  # noqa: E501
SMOOTHING_SIGMA = 3  # Gaussian smoothing window — expressed in frames at DEFAULT_FPS. Scaled by fps / DEFAULT_FPS at call time so the effective wall-time window (~0.2s) stays consistent across source framerates.  # noqa: E501

# Steadiness thresholds (mean residual after gaussian smoothing)
STEADINESS_MINIMAL = 0.002  # below = static/no shaking
STEADINESS_UNSTEADY = 0.0028  # above = unsteady

DEFAULT_FPS = 15.0  # Fallback frame rate when the caller does not supply one.

# Speed thresholds
ROT_SPEED_SLOW = 10.0  # deg/s
ROT_SPEED_FAST = 30.0  # deg/s
TRANS_SPEED_SLOW = 0.05  # m/s
TRANS_SPEED_FAST = 0.15  # m/s

_MIN_TRAJECTORY_FRAMES = 2
_UNIT_VECTOR_NORM_EPS = 1e-6


def _angle_descriptor(deg: float) -> str:
    """Graduated intensity for accumulated angles (tuned to ego-centric data distribution)."""
    if deg < 10:  # noqa: PLR2004
        return "slightly"
    if deg < 30:  # noqa: PLR2004
        return ""
    return "significantly"


def _dist_descriptor(m: float) -> str:
    """Graduated intensity for accumulated distances (tuned to ego-centric data distribution)."""
    if m < 0.15:  # noqa: PLR2004
        return "slightly"
    if m < 0.40:  # noqa: PLR2004
        return ""
    return "significantly"


def _rot_speed_descriptor(deg_per_sec: float) -> str:
    if deg_per_sec < ROT_SPEED_SLOW:
        return "slowly"
    if deg_per_sec < ROT_SPEED_FAST:
        return ""
    return "quickly"


def _trans_speed_descriptor(m_per_sec: float) -> str:
    if m_per_sec < TRANS_SPEED_SLOW:
        return "slowly"
    if m_per_sec < TRANS_SPEED_FAST:
        return ""
    return "quickly"


def _compute_speed(val: float, frame_range: tuple[int, int] | None, fps: float) -> float:
    """Compute average speed (val per second) from value and frame range."""
    if frame_range is None or fps <= 0:
        return 0.0
    duration = (frame_range[1] - frame_range[0]) / fps
    if duration <= 0:
        return 0.0
    return val / duration


def _format_time_range(frame_range: tuple[int, int], fps: float) -> str:
    if fps <= 0:
        return f"[frames {frame_range[0]}-{frame_range[1]}]"
    t_start = frame_range[0] / fps
    t_end = frame_range[1] / fps
    return f"[{t_start:.1f}s-{t_end:.1f}s, frames {frame_range[0]}-{frame_range[1]}]"


def _format_trans(
    val: float,
    direction: str,
    fmt: str,
    frame_range: tuple[int, int] | None,
    fps: float,
) -> str:
    """Format a translation string. ``direction`` uses cinematographic terms."""
    if fmt == "qualitative":
        return direction
    if fmt == "raw":
        s = f"{direction} (~{val:.1f}m)"
        if frame_range:
            s += f" {_format_time_range(frame_range, fps)}"
        return s
    # graduated
    i = _dist_descriptor(val)
    speed = _compute_speed(val, frame_range, fps)
    sp = _trans_speed_descriptor(speed)
    modifiers = ", ".join(p for p in [sp, i] if p)
    mod_str = f" ({modifiers})" if modifiers else ""
    s = f"{direction} (~{val:.1f}m{mod_str})"
    if frame_range:
        s += f" {_format_time_range(frame_range, fps)}"
    return s


def _format_rot(  # noqa: PLR0913 - one parameter per rendered value; splitting would fragment the format call
    val: float,
    action: str,
    direction: str,
    fmt: str,
    frame_range: tuple[int, int] | None,
    fps: float,
) -> str:
    """Format a rotation string. ``action``+``direction`` use cinematographic terms."""
    if fmt == "qualitative":
        return f"{action} {direction}"
    if fmt == "raw":
        s = f"{action} {direction} (~{round(val)}°)"
        if frame_range:
            s += f" {_format_time_range(frame_range, fps)}"
        return s
    # graduated
    i = _angle_descriptor(val)
    speed = _compute_speed(val, frame_range, fps)
    sp = _rot_speed_descriptor(speed)
    modifiers = ", ".join(p for p in [sp, i] if p)
    mod_str = f" ({modifiers})" if modifiers else ""
    s = f"{action} {direction} (~{round(val)}°{mod_str})"
    if frame_range:
        s += f" {_format_time_range(frame_range, fps)}"
    return s


def _hold_last_valid_unit_vector(vectors: np.ndarray[Any, Any], eps: float) -> np.ndarray[Any, Any]:
    """Normalize row vectors to unit length, holding direction across degenerate frames.

    A frame is degenerate when its magnitude is at or below ``eps`` — e.g. the
    Gaussian smoothing window averaged near-opposing directions to a near-zero
    vector. Substituting the zero vector there is not neutral: since yaw/tilt
    are ``arcsin``/``arctan2`` of these components and then differenced, a
    forced-zero frame reads as a real angle of 0, producing a phantom jump away
    from and back to the true heading (two fabricated motions, not none).
    Holding the last valid direction instead makes the delta at that frame ~0,
    correctly encoding "no new information" rather than a fictional swing.
    Leading degenerate frames hold the first valid direction found; an
    all-degenerate input returns zero vectors throughout (nothing valid to
    hold).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = (norms > eps)[:, 0]
    unit = np.zeros_like(vectors)
    unit[valid] = vectors[valid] / norms[valid]
    if not valid.all() and valid.any():
        fill_idx = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))
        fill_idx[fill_idx < 0] = int(np.argmax(valid))
        unit = unit[fill_idx]
    return unit


def _find_active_range(deltas: np.ndarray[Any, Any], *, positive: bool) -> tuple[int, int] | None:
    """Find the first and last frame where the motion is active (nonzero contribution)."""
    active = deltas > 0 if positive else deltas < 0
    indices = np.where(active)[0]
    if len(indices) == 0:
        return None
    return (int(indices[0]), int(indices[-1]) + 1)


def describe_camera_motion(  # noqa: C901, PLR0912, PLR0915
    positions: np.ndarray[Any, Any],
    rotations_q: np.ndarray[Any, Any],
    fmt: str = "graduated",
    *,
    return_full: bool = False,
    fps: float = DEFAULT_FPS,
) -> str | tuple[str, str, str]:
    """Convert a camera trajectory to a natural language description using robust vector tracking.

    ``fps`` is the source frame rate of ``positions`` / ``rotations_q`` and is
    used to convert frame indices to seconds (timestamps in the ``raw`` /
    ``graduated`` output) and to compute speed-in-seconds for the graduated
    speed descriptors. Pass the row's real framerate — the default of 15 fps
    is only a historical fallback.
    """
    if len(positions) < _MIN_TRAJECTORY_FRAMES:
        return "The camera remains relatively still."

    graduated = fmt in ("graduated", "raw")
    fmt_str = fmt

    # 1. Extract 3D Rotation Matrices
    rot_matrices = R.from_quat(rotations_q).as_matrix()  # (N, 3, 3)

    # Mecka convention: -Y is Camera Forward, +X is Camera Right, Z is world up
    cam_right = rot_matrices[:, :, 0]
    cam_forward = -rot_matrices[:, :, 1]

    # 2. Smooth Positions and Vectors
    # SMOOTHING_SIGMA is defined in frames at DEFAULT_FPS; rescale so the
    # wall-time smoothing window stays consistent across source framerates.
    # For fps <= 0 we fall back to the unscaled SMOOTHING_SIGMA.
    if fps > 0:
        fps_scale = fps / DEFAULT_FPS
        # Inverse scaling: more samples per second = smaller per-frame delta
        # for the same physical motion, so the per-frame noise gate shrinks.
        smoothing_sigma = SMOOTHING_SIGMA * fps_scale
        pos_noise_gate = POS_NOISE_GATE / fps_scale
        rot_noise_gate = ROT_NOISE_GATE / fps_scale
    else:
        smoothing_sigma = SMOOTHING_SIGMA
        pos_noise_gate = POS_NOISE_GATE
        rot_noise_gate = ROT_NOISE_GATE
    smooth_pos = gaussian_filter1d(positions, sigma=smoothing_sigma, axis=0)
    smooth_right = gaussian_filter1d(cam_right, sigma=smoothing_sigma, axis=0)
    smooth_fwd = gaussian_filter1d(cam_forward, sigma=smoothing_sigma, axis=0)

    # Re-normalize vectors after smoothing, holding the last valid direction
    # across any degenerate (near-zero-norm) frame instead of substituting a
    # zero vector — see _hold_last_valid_unit_vector for why a zero substitute
    # is not neutral here.
    smooth_right = _hold_last_valid_unit_vector(smooth_right, _UNIT_VECTOR_NORM_EPS)
    smooth_fwd = _hold_last_valid_unit_vector(smooth_fwd, _UNIT_VECTOR_NORM_EPS)

    # 3. Create a stable "Body Heading" by flattening Camera Right to the XY ground plane
    body_right = smooth_right.copy()
    body_right[:, 2] = 0
    norms = np.linalg.norm(body_right, axis=1, keepdims=True)
    body_right = np.divide(body_right, norms, out=np.zeros_like(body_right), where=norms > _UNIT_VECTOR_NORM_EPS)

    # Body Forward: Right x Up = -Y (forward in this coordinate system)
    world_up = np.array([0, 0, 1])
    body_forward = np.cross(body_right, world_up)

    # 4. Calculate continuous deltas
    dp = np.diff(smooth_pos, axis=0)

    dp[np.abs(dp) < pos_noise_gate] = 0

    # Use the initial body heading for translation projection so that
    # head turns don't flip forward/backward mid-clip
    initial_body_right = body_right[0]
    initial_body_forward = body_forward[0]
    dolly_deltas = np.sum(dp * initial_body_forward, axis=1)
    truck_deltas = np.sum(dp * initial_body_right, axis=1)
    ped_deltas = dp[:, 2]

    truck_right = np.sum(np.maximum(truck_deltas, 0))
    truck_left = np.abs(np.sum(np.minimum(truck_deltas, 0)))

    dolly_fwd = np.sum(np.maximum(dolly_deltas, 0))
    dolly_bwd = np.abs(np.sum(np.minimum(dolly_deltas, 0)))

    ped_up = np.sum(np.maximum(ped_deltas, 0))
    ped_down = np.abs(np.sum(np.minimum(ped_deltas, 0)))

    # 5. Rotation Accumulation (Pan and Tilt)
    yaw_angles = np.arctan2(body_forward[:, 1], body_forward[:, 0])
    delta_yaw = np.diff(yaw_angles)
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    delta_yaw_deg = np.degrees(delta_yaw)

    delta_yaw_deg[np.abs(delta_yaw_deg) < rot_noise_gate] = 0

    pan_left = np.sum(np.maximum(delta_yaw_deg, 0))
    pan_right = np.abs(np.sum(np.minimum(delta_yaw_deg, 0)))

    tilt_angles = np.arcsin(np.clip(smooth_fwd[:, 2], -1.0, 1.0))
    delta_tilt = np.diff(tilt_angles)
    delta_tilt_deg = np.degrees(delta_tilt)

    delta_tilt_deg[np.abs(delta_tilt_deg) < rot_noise_gate] = 0

    tilt_up = np.sum(np.maximum(delta_tilt_deg, 0))
    tilt_down = np.abs(np.sum(np.minimum(delta_tilt_deg, 0)))

    # 6. Compute frame ranges for each motion
    fr_pan_right = _find_active_range(delta_yaw_deg, positive=False) if graduated else None
    fr_pan_left = _find_active_range(delta_yaw_deg, positive=True) if graduated else None
    fr_tilt_up = _find_active_range(delta_tilt_deg, positive=True) if graduated else None
    fr_tilt_down = _find_active_range(delta_tilt_deg, positive=False) if graduated else None
    fr_truck_right = _find_active_range(truck_deltas, positive=True) if graduated else None
    fr_truck_left = _find_active_range(truck_deltas, positive=False) if graduated else None
    fr_ped_up = _find_active_range(ped_deltas, positive=True) if graduated else None
    fr_ped_down = _find_active_range(ped_deltas, positive=False) if graduated else None
    fr_dolly_fwd = _find_active_range(dolly_deltas, positive=True) if graduated else None
    fr_dolly_bwd = _find_active_range(dolly_deltas, positive=False) if graduated else None

    # 7. Build candidate list: (start_frame, text, is_significant)
    #    is_significant is tracked via the raw magnitude values
    _rot_sig = 30.0  # matches _angle_descriptor "significantly" threshold
    _trans_sig = 0.40  # matches _dist_descriptor "significantly" threshold

    candidates: list[tuple[int, str, bool]] = []

    if pan_right > CAM_ROTATION_THRESH:
        candidates.append(
            (
                fr_pan_right[0] if fr_pan_right else 0,
                _format_rot(pan_right, "pans", "right", fmt_str, fr_pan_right, fps),
                pan_right >= _rot_sig,
            )
        )
    if pan_left > CAM_ROTATION_THRESH:
        candidates.append(
            (
                fr_pan_left[0] if fr_pan_left else 0,
                _format_rot(pan_left, "pans", "left", fmt_str, fr_pan_left, fps),
                pan_left >= _rot_sig,
            )
        )

    if tilt_up > CAM_ROTATION_THRESH:
        candidates.append(
            (
                fr_tilt_up[0] if fr_tilt_up else 0,
                _format_rot(tilt_up, "tilts", "up", fmt_str, fr_tilt_up, fps),
                tilt_up >= _rot_sig,
            )
        )
    if tilt_down > CAM_ROTATION_THRESH:
        candidates.append(
            (
                fr_tilt_down[0] if fr_tilt_down else 0,
                _format_rot(tilt_down, "tilts", "down", fmt_str, fr_tilt_down, fps),
                tilt_down >= _rot_sig,
            )
        )

    if truck_right > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_truck_right[0] if fr_truck_right else 0,
                _format_trans(truck_right, "trucks right", fmt_str, fr_truck_right, fps),
                truck_right >= _trans_sig,
            )
        )
    if truck_left > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_truck_left[0] if fr_truck_left else 0,
                _format_trans(truck_left, "trucks left", fmt_str, fr_truck_left, fps),
                truck_left >= _trans_sig,
            )
        )

    if ped_up > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_ped_up[0] if fr_ped_up else 0,
                _format_trans(ped_up, "pedestals up", fmt_str, fr_ped_up, fps),
                ped_up >= _trans_sig,
            )
        )
    if ped_down > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_ped_down[0] if fr_ped_down else 0,
                _format_trans(ped_down, "pedestals down", fmt_str, fr_ped_down, fps),
                ped_down >= _trans_sig,
            )
        )

    if dolly_fwd > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_dolly_fwd[0] if fr_dolly_fwd else 0,
                _format_trans(dolly_fwd, "dollies in", fmt_str, fr_dolly_fwd, fps),
                dolly_fwd >= _trans_sig,
            )
        )
    if dolly_bwd > CAM_TRANSLATION_THRESH:
        candidates.append(
            (
                fr_dolly_bwd[0] if fr_dolly_bwd else 0,
                _format_trans(dolly_bwd, "dollies out", fmt_str, fr_dolly_bwd, fps),
                dolly_bwd >= _trans_sig,
            )
        )

    # Full (unfiltered) version — all motions above threshold
    all_candidates = [(f, text) for f, text, _ in candidates]
    all_candidates.sort(key=lambda x: x[0])

    # Filtered version — if any significant motions exist, only keep those
    has_significant = any(sig for _, _, sig in candidates)
    filtered = [(f, text, sig) for f, text, sig in candidates if sig] if has_significant else list(candidates)

    filtered.sort(key=lambda x: x[0])
    candidates.sort(key=lambda x: x[0])

    # 8. Steadiness from residual after removing smooth trend
    residual = positions - smooth_pos
    residual_mag = np.linalg.norm(residual, axis=1)
    jitter = float(np.mean(residual_mag))

    if jitter < STEADINESS_MINIMAL:
        steadiness = "steady"
    elif jitter < STEADINESS_UNSTEADY:
        steadiness = "slightly unsteady"
    else:
        steadiness = "unsteady"

    def _build_desc(part_list: list[str]) -> str:
        if not part_list:
            if steadiness == "unsteady":
                return "The camera remains relatively still but is unsteady."
            if steadiness == "slightly unsteady":
                return "The camera remains relatively still but is slightly unsteady."
            return "The camera remains steady and still."

        pl = list(part_list)
        if len(pl) > 1:
            pl[-1] = "and " + pl[-1]
        join_str = ", " if len(pl) > 2 else " "  # noqa: PLR2004
        desc = "The camera " + join_str.join(pl) + "."

        if steadiness == "unsteady":
            desc += " The motion is unsteady."
        elif steadiness == "slightly unsteady":
            desc += " The motion is slightly unsteady."
        else:
            desc += " The motion is steady."
        return desc

    # Build all versions (grouped/multi-format rendering is not needed for the
    # robot-action-split use case, which only consumes fmt="raw"; the upstream
    # "grouped" variant that merges overlapping-frame-range motions was dropped
    # here as unused).
    all_parts = [text for _, text, _ in candidates]
    dominant_parts = [text for _, text, _ in filtered]
    full_desc = _build_desc(all_parts)
    dominant_desc = _build_desc(dominant_parts)

    if return_full:
        return dominant_desc, full_desc, dominant_desc
    return full_desc


def compute_camera_motion_annotation(  # noqa: PLR0911 - each return is a distinct validation gate
    camera_position: list[Any] | None,
    camera_rotation: list[Any] | None,
    fps: float,
    *,
    clip_id: str = "<unknown>",
) -> str | None:
    """Adapt ``_slice_action``'s decoded trajectory lists into a motion annotation.

    Args:
        camera_position: Per-frame ``[x, y, z]`` world-space positions, as
            produced by ``processing._slice_action``'s ``camera_position`` key
            (``None`` when the row has no camera trajectory).
        camera_rotation: Per-frame ``[x, y, z, w]`` quaternion orientations,
            as produced by the same ``camera_rotation`` key.
        fps: The clip's native frame rate (``SpanWorkItem.native_fps`` /
            outcome row's ``native_fps``).
        clip_id: Identifies the row in warning logs when the trajectory is
            present but malformed — distinguishes a systematic problem (wrong
            quaternion convention, corrupt export, ...) from a dataset that
            genuinely carries no camera trajectory, which returns ``None``
            here silently since that's the expected, common case.

    Returns:
        A ``fmt="raw"`` camera-motion description (matches imaginaire4's
        prompt-input usage — a VLM later rewrites this into the final caption
        text), or ``None`` when there's no usable trajectory (missing,
        malformed, or fewer than two frames).

    """
    if not camera_position or not camera_rotation:
        return None
    try:
        positions = np.asarray(camera_position, dtype=np.float32)
        rotations = np.asarray(camera_rotation, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        logger.warning(f"clip {clip_id}: camera trajectory is not a valid numeric array ({exc}); skipping")
        return None

    if positions.ndim != 2 or positions.shape[1] != 3:  # noqa: PLR2004
        logger.warning(f"clip {clip_id}: camera_position has shape {positions.shape}, expected (N, 3); skipping")
        return None
    if rotations.ndim != 2 or rotations.shape[1] != 4:  # noqa: PLR2004
        logger.warning(f"clip {clip_id}: camera_rotation has shape {rotations.shape}, expected (N, 4); skipping")
        return None
    if positions.shape[0] != rotations.shape[0]:
        logger.warning(
            f"clip {clip_id}: camera_position has {positions.shape[0]} frames but camera_rotation has "
            f"{rotations.shape[0]}; skipping"
        )
        return None
    if positions.shape[0] < _MIN_TRAJECTORY_FRAMES:
        return None

    try:
        description = describe_camera_motion(positions, rotations, fmt="raw", fps=fps)
    except ValueError as exc:
        # describe_camera_motion's own inputs are already shape-validated above; a
        # ValueError here is scipy's Rotation rejecting the quaternions themselves
        # (e.g. all-zero-norm), which points at a genuinely malformed export rather
        # than an expectedly-absent trajectory, so it's worth surfacing.
        logger.warning(f"clip {clip_id}: describe_camera_motion rejected the trajectory ({exc}); skipping")
        return None
    return description if isinstance(description, str) else None
