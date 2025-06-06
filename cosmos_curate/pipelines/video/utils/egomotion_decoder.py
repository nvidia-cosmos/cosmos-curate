# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Egomotion decoder."""

import io
import itertools
import json
import random
import re
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as spt  # type: ignore[import-untyped]
import torch
from loguru import logger
from scipy.interpolate import interp1d  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation

_VALID_DECODE_STRATEGY_PATTERN = r"^(random|uniform|at)_(-?\d+)_frame$"


class EgoMotionData(TypedDict):
    """Encompasses all required information to interpolate ego motion data."""

    tmin: int
    tmax: int
    tparam: npt.NDArray[np.float32]
    xyzs: npt.NDArray[np.float32]
    quats: npt.NDArray[np.float32]


def parse_egopose_data(egopose_info: dict) -> dict:
    """Minimal parsing here.

    Args:
        egopose_info: a dict containing the raw egopose data.

    Returns:
        info (dict): a dict of numpy arrays with dtype object

    """
    N, C = egopose_info["labels_data"].shape
    (K,) = egopose_info["labels_keys"].shape
    assert K == C, f"{K} {C}"

    info = {egopose_info["labels_keys"][ki]: egopose_info["labels_data"][:, ki] for ki in range(K)}

    # make sure sorted by time
    assert np.all(info["timestamp"][1:] - info["timestamp"][:-1] > 0), info["timestamp"]

    return info


def adjust_orientation(
    vals: npt.NDArray[np.float32] | torch.Tensor,
) -> npt.NDArray[np.float32] | torch.Tensor:
    """Adjust the orientation of the quaternions.

    Adjusts the orientation of the quaternions so that the dot product
    between vals[i] and vals[i+1] is non-negative.

    Args:
        vals (np.array or torch.tensor): (N, C)

    Returns:
        vals (np.array or torch.tensor): (N, C) adjusted quaternions

    """
    N, C = vals.shape
    if isinstance(vals, torch.Tensor):
        signs_ts = torch.ones(N, dtype=vals.dtype, device=vals.device)
        signs_ts[1:] = torch.where((vals[:-1] * vals[1:]).sum(dim=1) >= 0, 1.0, -1.0)
        signs_ts = torch.cumprod(signs_ts, dim=0)

        return vals * signs_ts.reshape((N, 1))

    signs_np = np.ones(N, dtype=vals.dtype)
    signs_np[1:] = np.where((vals[:-1] * vals[1:]).sum(axis=1) >= 0, 1.0, -1.0)
    signs_np = np.cumprod(signs_np)

    return vals * signs_np.reshape((N, 1))


def preprocess_egopose(poses: dict) -> EgoMotionData:
    """Convert the poses for interpolation.

    The dtype of all the inputs to the interpolator is float32.
    TODO: instead of a linear interpolator for quaternions,
    it'd be better to do slerp.

    Args:
        poses (dict): a dict containing the raw egopose data.

    Returns:
        A dictionary containing the following
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose

    """
    # bounds of the interpolator as timestamps (ints)
    tmin = poses["timestamp"][0]
    tmax = poses["timestamp"][-1]

    # convert timestamps to float32 only after subtracting off tmin and
    # converting from microseconds to seconds
    tparam = (1e-6 * (poses["timestamp"] - tmin)).astype(np.float32)

    # prep x,y,z
    # convert to float64, subtract off mean, convert to float32
    xyzs = np.stack(
        (
            poses["x"].astype(np.float64),
            poses["y"].astype(np.float64),
            poses["z"].astype(np.float64),
        ),
        1,
    )
    xyzs = xyzs - xyzs.mean(axis=0, keepdims=True)
    xyzs = xyzs.astype(np.float32)

    # prep quaternions
    # parse directly as float32
    quats = np.stack(
        (
            poses["qw"].astype(np.float32),
            poses["qx"].astype(np.float32),
            poses["qy"].astype(np.float32),
            poses["qz"].astype(np.float32),
        ),
        1,
    )

    # prep quaternions for interpolation https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    # make sure normalized
    quat_norm = np.linalg.norm(quats, axis=1)
    EPS = 1e-3
    if not np.all(np.abs(quat_norm - 1.0) < EPS):
        error_msg = f"Raw pose quaternions are too far from normalized; {quat_norm=}"
        raise ValueError(error_msg)
    # adjust signs so that sequential dot product is always positive
    quats = adjust_orientation(quats / quat_norm[:, None])

    return EgoMotionData(
        tmin=tmin,
        tmax=tmax,
        tparam=tparam,
        xyzs=xyzs,
        quats=quats,
    )


def get_sensor_to_sensor_flu(sensor: str) -> np.ndarray:
    """Compute a Rotation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting Rotation matrix.

    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)  # type: ignore[attr-defined]

    return np.asarray(rot, dtype=np.float32)


def parse_rig_sensors_from_dict(rig: dict) -> dict[str, dict]:
    """Parse the provided rig dict into a dictionary indexed by sensor name.

    Args:
        rig (Dict): Complete rig file as a dictionary.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.

    """
    # Parse the properties from the rig file
    sensors = rig["rig"]["sensors"]

    return {sensor["name"]: sensor for sensor in sensors}


def euler_2_so3(euler_angles: np.ndarray, *, degrees: bool = True, seq: str = "xyz") -> np.ndarray:
    """Convert the euler angles representation to the so3 Rotation matrix.

    Args:
        euler_angles:(np.array) euler angles [n,3]
        degrees: (bool) True if angle is given in degrees else False
        seq: (str): sequence in which the euler angles are given

    Returns:
        (np array): Rotations given so3 matrix representation [n,3,3]

    """
    return Rotation.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)


def sensor_to_rig(sensor: dict) -> np.ndarray | None:
    """Obtain sensor to rig transformation matrix."""
    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

    if "nominalSensor2Rig_FLU" not in sensor:
        # Some sensors (like CAN sensors) don't have an associated sensorToRig
        return None

    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if "correction_rig_T" in sensor:
        correction_T = sensor["correction_rig_T"]

    if "correction_sensor_R_FLU" in sensor:
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"], str(sensor["correction_sensor_R_FLU"])
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = euler_2_so3(nominal_R)
    correction_R = euler_2_so3(correction_R)

    R = nominal_R @ correction_R
    T = np.array(nominal_T, dtype=np.float32) + np.array(correction_T, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T

    return transform @ sensor_to_FLU


def load_egopose(sample: dict, *, live: bool = False, min_fps: int = 5) -> EgoMotionData | None:
    """Load egopose from raw tar files.

    If we have fewer than 20 seconds, return None

    Args:
        sample: A dictionary containing the raw data to be decoded.
        live: if `True`, the "live" (estimated online) egomotion will be loaded
            from the sample, otherwise the ground truth (optimized offline)
            egomotion will be loaded.
        min_fps: The minimum FPS of the egomotion data.

    Returns:
        A dictionary containing the following keys:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose

    """
    pose_info = np.load(
        io.BytesIO(sample["live_egomotion.npz" if live else "egomotion.npz"]),
        allow_pickle=True,
    )
    egopose_parsed = parse_egopose_data(pose_info)

    TMIN = 20.0  # seconds
    egopose_span = 1e-6 * (egopose_parsed["timestamp"][-1] - egopose_parsed["timestamp"][0])
    if egopose_span < TMIN:
        logger.warning(f"Insufficient egomotion data for this clip: {egopose_span=}")
        return None

    # Check the FPS of egomotion data
    delta = 1e-6 * (egopose_parsed["timestamp"][1:] - egopose_parsed["timestamp"][:-1])
    max_delta = 1.0 / min_fps
    if not np.all(delta < max_delta):
        logger.warning(f"Egomotion data does not meet frequency requirement: {max(delta)=}")
        return None

    ego_lerp_inp = preprocess_egopose(egopose_parsed)

    coordinate_frame = egopose_parsed["coordinate_frame"][0].replace("_", ":")
    if coordinate_frame != "rig":
        # The logged egomotion is tracking a sensor's coordinate frame (e.g., the pose
        # of the lidar) that is not the rig frame (origin at the rear axle center projected
        # to ground, oriented front-left-up with respect to the vehicle's body).
        # Here we use the "rig.json" to convert the sensor's pose to the rig's pose.
        sensors = parse_rig_sensors_from_dict(json.loads(sample["rig.json"]))
        if coordinate_frame not in sensors:
            error_msg = f"Egomotion {coordinate_frame=} not found in rig.json."
            raise ValueError(error_msg)
        s2r = sensor_to_rig(sensors[coordinate_frame])
        sensor_to_world_rots = spt.Rotation.from_quat(ego_lerp_inp["quats"], scalar_first=True)
        # We derive rig_to_world by composing sensor_to_world and rig_to_sensor;
        # with 4x4 rigid transformation matrices this would be:
        # `rig_to_world = sensor_to_world @ np.linalg.inv(sensor_to_rig)`
        rig_to_world_rots = sensor_to_world_rots * spt.Rotation.from_matrix(s2r[:3, :3].T)
        ego_lerp_inp["xyzs"] = ego_lerp_inp["xyzs"] - rig_to_world_rots.apply(s2r[:3, 3])
        ego_lerp_inp["quats"] = adjust_orientation(rig_to_world_rots.as_quat(scalar_first=True))

    return EgoMotionData(
        tmin=ego_lerp_inp["tmin"],
        tmax=ego_lerp_inp["tmax"],
        tparam=ego_lerp_inp["tparam"].tolist(),
        xyzs=ego_lerp_inp["xyzs"].tolist(),
        quats=ego_lerp_inp["quats"].tolist(),
    )


class EgoPoseInterp:
    """Interpolates egopose data."""

    def __init__(
        self,
        tmin: int,
        tmax: int,
        tparam: npt.NDArray[np.float32],
        xyzs: npt.NDArray[np.float32],
        quats: npt.NDArray[np.float32],
    ) -> None:
        """Initialize the interpolator.

        Args:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose

        """
        self.tmin = tmin
        self.tmax = tmax

        self.interp = interp1d(
            tparam,
            np.concatenate((xyzs, quats), 1),
            kind="linear",
            axis=0,
            copy=False,
            bounds_error=True,
            assume_sorted=True,
        )

    def convert_tstamp(self, tstamp: int | npt.NDArray[np.int64]) -> int:
        """Convert the absolute timestamp (microsecond) to relative (s)."""
        return 1e-6 * (tstamp - self.tmin)

    def __call__(
        self,
        t: npt.NDArray[np.float32],
        *,
        is_microsecond: bool = False,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Interpolate pose for t in seconds or microsecond."""
        EPS = 1e-5
        if is_microsecond:
            t = self.convert_tstamp(t)

        out = self.interp(t)
        xyzs = out[..., :3]
        quats = out[..., 3:]

        # normalize quats
        norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        assert np.all(norm > EPS), norm
        quats = quats / norm

        return xyzs, quats


class Quaternion:
    """Utility class for quaternion operations and conversions.

    This class provides methods for converting between quaternions and rotation matrices,
    as well as operations like inversion, multiplication, and interpolation.
    """

    def q_to_r(self, q: torch.Tensor) -> torch.Tensor:  # [...,4]
        """Convert quaternions to rotation matrices.

        Args:
            q: Quaternions to convert.

        Returns:
            Rotation matrices.

        """
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa, qb, qc, qd = q.unbind(dim=-1)
        return torch.stack(
            [
                torch.stack([1 - 2 * (qc**2 + qd**2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
                torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb**2 + qd**2), 2 * (qc * qd - qa * qb)], dim=-1),
                torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb**2 + qc**2)], dim=-1),
            ],
            dim=-2,
        )

    def r_to_q(self, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:  # [...,3,3]
        """Convert rotation matrices to quaternions.

        Args:
            r: Rotation matrices to convert.
            eps: Small value to prevent division by zero.

        Returns:
            Quaternions

        """
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        row0, row1, row2 = r.unbind(dim=-2)
        R00, R01, R02 = row0.unbind(dim=-1)
        R10, R11, R12 = row1.unbind(dim=-1)
        R20, R21, R22 = row2.unbind(dim=-1)
        t = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
        r = (1 + t + eps).sqrt()
        qa = 0.5 * r
        qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
        qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
        qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
        return torch.stack([qa, qb, qc, qd], dim=-1)

    def invert(self, q: torch.Tensor) -> torch.Tensor:  # [...,4]
        """Invert a quaternion.

        Args:
            q: Quaternions to invert.

        Returns:
            Inverted quaternions.

        """
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        return torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm**2

    def product(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:  # [...,4]
        """Multiply two quaternions.

        Args:
            q1: First quaternions.
            q2: Second quaternions.

        Returns:
            Product of the two quaternions.

        """
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        # hamil_prod
        return torch.stack(
            [
                q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d,
                q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c,
                q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b,
                q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a,
            ],
            dim=-1,
        )

    def apply(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Apply the quaternion rotation to a point.

        Args:
            q: Quaternions to apply.
            p: Points to apply.

        Returns:
            Rotated points.

        """
        out = self.product(
            self.product(q, torch.cat((torch.zeros(p.shape[:-1] + (1,)), p), -1)),
            self.invert(q),
        )
        return out[..., 1:]

    def interpolate(self, q1: torch.Tensor, q2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Interpolate between two quaternions.

        Args:
            q1: First quaternions.
            q2: Second quaternions.
            alpha: Interpolation factor.

        Returns:
            Interpolated quaternions.

        """
        # `return [...,4],[...,4],[...,1]`
        # https://en.wikipedia.org/wiki/Slerp
        cos_angle = (q1 * q2).sum(dim=-1, keepdim=True)  # [...,1]
        flip = cos_angle < 0
        q1 = q1 * (~flip) - q1 * flip  # [...,4]
        theta = cos_angle.abs().acos()  # [...,1]
        return (((1 - alpha) * theta).sin() * q1 + (alpha * theta).sin() * q2) / theta.sin()  # [...,4]


def _check_valid_tstamp_0(  # noqa: PLR0913
    tstamp_0: int,
    prediction_start_offset_range: list[float],
    ego_history_tvals: np.ndarray,
    ego_future_tvals: np.ndarray,
    history_ego_lerp_inp: dict,
    future_ego_lerp_inp: dict,
) -> bool:
    """Check if the tstamp_0 is valid for the given prediction_start_offset_range."""
    min_random_offset = int(prediction_start_offset_range[0] * 1e6)
    max_random_offset = int(prediction_start_offset_range[1] * 1e6)
    return (
        tstamp_0 + min_random_offset + int(ego_history_tvals[0] * 1e6) >= history_ego_lerp_inp["tmin"]
        and tstamp_0 + max_random_offset + int(ego_history_tvals[-1] * 1e6) <= history_ego_lerp_inp["tmax"]
        and tstamp_0 + min_random_offset + int(ego_future_tvals[0] * 1e6) >= future_ego_lerp_inp["tmin"]
        and tstamp_0 + max_random_offset + int(ego_future_tvals[-1] * 1e6) <= future_ego_lerp_inp["tmax"]
    )


def interpolate_egopose(  # noqa: C901, PLR0912, PLR0913, PLR0915
    ego_lerp_inp: dict | None,
    live_ego_lerp_inp: dict | None,
    prediction_start_offset_range: tuple[float, float],
    ego_history_tvals: list[float],
    ego_future_tvals: list[float],
    base_timestamps: list[int],
    decode_strategy: str,
    num_route_points: int = 32,
) -> dict:
    """Interpolates the egopose data starting at the certain timestamps.

    Taking the raw egopose data (ego_lerp_inp), we first decide starting time
    indices (`ego_t0_frame_idx`) and then interpolate the egopose at the timestamps
    base_timestamps[ego_t0_frame_idx] + trajectory_tvals.

    Args:
        ego_lerp_inp (dict): A dictionary containing the complete egopose data (gt).
        live_ego_lerp_inp (dict): A dictionary containing the complete live egopose data (live).
        prediction_start_offset_range (tuple): range of possible relative
            time offsets from last input image frame to prediction start time.
        ego_history_tvals (list): the notional relative timestamps (i.e. t0 = 0.0s) of the
            ego history data in seconds.
        ego_future_tvals (list): the notional relative timestamps (i.e. t0 = 0.0s) of the
            ego future data (i.e., ground truth for prediction) in seconds.
        base_timestamps (list): the timestamps of the base frame
            in microseconds (assume sorted).
        decode_strategy: the strategy defines at which frames to decode the egopose.
            valid strategies are:
                - `random_N_frame`: randomly sample N frames from the available frames.
                - `uniform_N_frame`: sample N uniformly spaced frames from the available frames.
                - `at_N_frame`: sample the N-th frame from the available frames.
        num_route_points: The number of points from egopose data to select as a temporary (TODO)
            stand-in for route information.

    Returns:
        A dictionary containing the following
            ego_available (bool): whether the egopose data is available
            ego_t0 (torch.tensor): (num_sample,)
                the absolute start time of the egopose data in microseconds
            ego_t0_relative (torch.tensor): (num_sample,) the relative start time of the egopose
                data in seconds it is normalized to the first timestamp of the base_timestamps
            ego_t0_frame_idx (torch.tensor): (num_sample,) the frame index of the base frame
            prediction_start_offset (torch.tensor): (num_sample,)
                the prediction start time offset
            ego_history_tvals (torch.tensor): (Th,)
                time in seconds corresponding to each position
            ego_history_xyz (torch.tensor): (num_sample,Th,3)
                the ego history (live) x,y,z positions
            ego_history_rot (torch.tensor): (num_sample,Th,3,3)
                the ego history (live) orientations as 3x3 matrices
            ego_future_tvals (torch.tensor): (Tf,)
                time in seconds corresponding to each position
            ego_future_xyz (torch.tensor): (num_sample,Tf,3)
                the ego future (gt) x,y,z positions
            ego_future_rot (torch.tensor): (num_sample,Tf,3,3)
                the ego future (gt) orientations as 3x3 matrices
            route_xy (torch.tensor): (num_sample, num_route_points, 3)
                the route x,y positions (from ego gt)

    """
    if ego_lerp_inp is None:
        error_msg = "Invalid ego pose data for this clip."
        raise ValueError(error_msg)

    if not all(x <= y for x, y in itertools.pairwise(base_timestamps)):
        error_msg = "base_timestamps is not sorted."
        raise ValueError(error_msg)

    ego_lerp = EgoPoseInterp(
        tmin=ego_lerp_inp["tmin"],
        tmax=ego_lerp_inp["tmax"],
        tparam=ego_lerp_inp["tparam"],
        xyzs=ego_lerp_inp["xyzs"],
        quats=ego_lerp_inp["quats"],
    )

    if live_ego_lerp_inp is None:
        logger.warning("Using ego_lerp_inp in place of live_ego_lerp_inp (= None).")
        live_ego_lerp_inp = ego_lerp_inp
        live_ego_lerp = ego_lerp
    else:
        live_ego_lerp = EgoPoseInterp(
            tmin=live_ego_lerp_inp["tmin"],
            tmax=live_ego_lerp_inp["tmax"],
            tparam=live_ego_lerp_inp["tparam"],
            xyzs=live_ego_lerp_inp["xyzs"],
            quats=live_ego_lerp_inp["quats"],
        )

    ego_history_tvals = np.array(ego_history_tvals, dtype=np.float32)
    ego_future_tvals = np.array(ego_future_tvals, dtype=np.float32)

    match = re.match(_VALID_DECODE_STRATEGY_PATTERN, decode_strategy)
    if match is None:
        error_msg = f"Decode strategy {decode_strategy} not implemented."
        raise NotImplementedError(error_msg)
    strategy_type = match.group(1)
    rng = np.random.Generator(np.random.PCG64(seed=93))
    if strategy_type in {"random", "uniform"}:
        # We work in timestamps (microseconds) to ensure temporal consistency between ego_lerp
        # and live_ego_lerp (which have different notions of time t relative to their respective
        # first timestamps).
        valid_frame_indices = [
            ori
            for ori, _tstamp_0 in enumerate(base_timestamps)
            if _check_valid_tstamp_0(
                tstamp_0=_tstamp_0,
                prediction_start_offset_range=prediction_start_offset_range,
                ego_history_tvals=ego_history_tvals,
                ego_future_tvals=ego_future_tvals,
                history_ego_lerp_inp=live_ego_lerp_inp,
                future_ego_lerp_inp=ego_lerp_inp,
            )
        ]
        if len(valid_frame_indices) == 0:
            error_msg = (
                "Insufficient ego pose data to fit history + future "
                f"with maximum start offset {prediction_start_offset_range[1]}."
            )
            raise ValueError(error_msg)
        num_frames = int(match.group(2))
        if num_frames > len(valid_frame_indices):
            error_msg = f"Requested {num_frames} frames, but only {len(valid_frame_indices)} frames are available."
            raise ValueError(error_msg)
        prediction_start_offset = rng.uniform(*prediction_start_offset_range, size=num_frames)
        if strategy_type == "random":
            # sample randomly from timestamps for which history + future steps (shifted by
            # prediction_start_offset) fit within the available ego data.
            ego_t0_frame_idx = sorted(random.sample(valid_frame_indices, k=num_frames))
        else:
            # sample uniformly from timestamps for which history + future steps (shifted by
            # prediction_start_offset) fit within the available ego data.
            # NOTE: we sample the last frame first to ensure that the last frame is included.
            _step = len(valid_frame_indices) / num_frames
            ego_t0_frame_idx = [valid_frame_indices[-(int(_step * i) + 1)] for i in range(num_frames)][::-1]
    elif strategy_type == "at":
        frame_idx = int(match.group(2))
        ego_t0_frame_idx = [len(base_timestamps) + frame_idx] if frame_idx < 0 else [frame_idx]
        prediction_start_offset = rng.uniform(*prediction_start_offset_range, size=1)

    tstamp_0 = [
        base_timestamps[_idx] + int(_start_offset * 1e6)
        for _idx, _start_offset in zip(ego_t0_frame_idx, prediction_start_offset, strict=False)
    ]
    # convert prediction-relative tvals to data sample tvals
    # `shape: (num_frames, num_history_steps)`
    history_live_ego_lerp_tvals = (
        np.array(
            [live_ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )[:, None]
        + ego_history_tvals[None, :]
    )
    # This check should only fail at "at" strategy
    if not (
        live_ego_lerp_inp["tparam"][0] <= history_live_ego_lerp_tvals[:, 0].min()
        and history_live_ego_lerp_tvals[:, -1].max() <= live_ego_lerp_inp["tparam"][-1]
    ):
        error_msg = (
            f"data: {live_ego_lerp_inp['tparam'][0]=} to {live_ego_lerp_inp['tparam'][-1]=}, "
            f"while asking {history_live_ego_lerp_tvals[:, 0].min()=} to "
            f"{history_live_ego_lerp_tvals[:, -1].max()=}"
        )
        raise ValueError(error_msg)

    # `shape: (num_frames, num_future_steps)`
    future_future_ego_lerp_tvals = (
        np.array(
            [ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )[:, None]
        + ego_future_tvals[None, :]
    )
    # This check should only fail at "at" strategy
    if not (
        ego_lerp_inp["tparam"][0] <= future_future_ego_lerp_tvals[:, 0].min()
        and future_future_ego_lerp_tvals[:, -1].max() <= ego_lerp_inp["tparam"][-1]
    ):
        error_msg = (
            f"data: {ego_lerp_inp['tparam'][0]=} to {ego_lerp_inp['tparam'][-1]=}, "
            f"while asking {future_future_ego_lerp_tvals[:, 0].min()=} to "
            f"{future_future_ego_lerp_tvals[:, -1].max()=}"
        )
        raise ValueError(error_msg)

    # evaluate live pose at the history timesteps
    ego_history_xyz, ego_history_quat = live_ego_lerp(history_live_ego_lerp_tvals)
    # `(num_frames, num_history_steps, 3)`
    ego_history_xyz = torch.tensor(ego_history_xyz, dtype=torch.float32)
    # `(num_frames, num_history_steps, 4)`
    ego_history_quat = torch.tensor(ego_history_quat, dtype=torch.float32)

    # transform coordinates to the ego's body frame (according to live) at the start of prediction
    # `(num_frames, 3), (num_frames, 3)`
    quaternion = Quaternion()

    live_xyz0, live_quat0 = live_ego_lerp(
        np.array(
            [live_ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        ),
    )
    live_xyz0 = torch.tensor(live_xyz0, dtype=torch.float32)
    live_inv_quat0 = quaternion.invert(torch.tensor(live_quat0, dtype=torch.float32))

    ego_history_xyz = quaternion.apply(live_inv_quat0.unsqueeze(1), ego_history_xyz - live_xyz0.unsqueeze(1))
    ego_history_quat = quaternion.product(live_inv_quat0.unsqueeze(1), ego_history_quat)

    # TODO: remove duplicated code
    # evaluate gt pose at the future timesteps
    ego_future_xyz, ego_future_quat = ego_lerp(future_future_ego_lerp_tvals)
    # `(num_frames, num_future_steps, 3)`
    ego_future_xyz = torch.tensor(ego_future_xyz, dtype=torch.float32)
    # `(num_frames, num_future_steps, 3)`
    ego_future_quat = torch.tensor(ego_future_quat, dtype=torch.float32)

    # transform coordinates to the ego's body frame (according to gt) at the start of prediction
    # `(num_frames, 3), (num_frames, 3)`
    xyz0, quat0 = ego_lerp(
        np.array(
            [ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        ),
    )
    xyz0 = torch.tensor(xyz0, dtype=torch.float32)
    inv_quat0 = quaternion.invert(torch.tensor(quat0, dtype=torch.float32))
    ego_future_xyz = quaternion.apply(inv_quat0.unsqueeze(1), ego_future_xyz - xyz0.unsqueeze(1))
    ego_future_quat = quaternion.product(inv_quat0.unsqueeze(1), ego_future_quat)

    # infer ego route from gt pose and transform to ego's body frame at the start of prediction
    # TODO: update route decoding when we have access to an alternative source
    # `(num_route_points, 3)`
    route_points = torch.tensor(ego_lerp_inp["xyzs"])[
        np.linspace(0, len(ego_lerp_inp["xyzs"]) - 1, num_route_points, dtype=int)
    ]
    route_xy = quaternion.apply(inv_quat0.unsqueeze(1), route_points.unsqueeze(0) - xyz0.unsqueeze(1))[..., :2]

    time_base = base_timestamps[0]
    # make sure the relative_timestamp is slightly larger than original timestamp
    # because later we will sort the timestamps of ego and images
    # and we want to make sure the ego appears after the image
    ego_t0_relative = [(t0 - time_base) * 1e-6 + 1e-5 for t0 in tstamp_0]

    return {
        "ego_available": torch.tensor(True, dtype=torch.bool),  # noqa: FBT003
        "ego_t0": torch.tensor(tstamp_0),
        "ego_t0_relative": torch.tensor(ego_t0_relative),
        "ego_t0_frame_idx": torch.tensor(ego_t0_frame_idx),
        "prediction_start_offset": torch.from_numpy(prediction_start_offset).float(),
        "ego_history_tvals": torch.from_numpy(ego_history_tvals).float(),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": quaternion.q_to_r(ego_history_quat),
        "ego_future_tvals": torch.from_numpy(ego_future_tvals).float(),
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": quaternion.q_to_r(ego_future_quat),
        "route_xy": route_xy,
    }


@dataclass(frozen=True)
class EgoMotionDecoderConfig:
    """Configuration for the egomotion decoder.

    Attributes:
        num_history_steps: number of history steps to load, i.e., with relative timestamps
            `(-(num_history_steps - 1) * time_step, ..., -2 * time_step, -time_step, 0)`
            to the prediction start time.
        num_future_step: number of future steps to load, i.e., with relative timestamps
            `(time_step, 2 * time_step, ..., num_future_steps * time_step)`
            to the prediction start time.
        time_step: time step (in seconds) between successive egomotion poses.
        prediction_start_offset_range: min and max of possible relative time offsets from base image
            frame to prediction start time.
        force_base_frame_index: if not `None`, loaded trajectory is based from the specified
            frame index in image_frames. (TODO: Deprecate this in favor of `decode_strategy`)
        num_route_points: The number of points from egopose data to select as a temporary (TODO)
            stand-in for route information.
        decode_strategy: the strategy defines at which frames to decode the egopose.
            valid strategies are:
                - `random_N_frame`: randomly sample N frames from the available frames.
                - `uniform_N_frame`: sample N uniformly spaced frames from the available frames.
                - `at_N_frame`: sample the N-th frame from the available frames.

    """

    num_history_steps: int = 15
    num_future_steps: int = 64
    time_step: float = 0.1
    prediction_start_offset_range: tuple[float, float] = field(default_factory=lambda: (0.0, 1.5))
    num_route_points: int = 32
    decode_strategy: str = "random_1_frame"

    def __post_init__(self) -> None:
        """Make sure the config is valid."""
        if not re.match(_VALID_DECODE_STRATEGY_PATTERN, self.decode_strategy):
            error_msg = f"Invalid decode strategy: {self.decode_strategy}"
            raise ValueError(error_msg)


def decode_egomotion(
    data: dict,
    base_timestamps: list[int],
    config: EgoMotionDecoderConfig,
) -> dict:
    """Decode egomotion from the data.

    Args:
        data (dict): The raw data to be decoded.
            it is assumed to contain the "egomotion.npz" (gt) and
            "live_egomotion.npz" (live) fields
        base_timestamps (list[int]): time stamps for each image frame
            in microseconds. This is used to decide the 0-th timestamp of
            the trajectory to load.
        config: EgoMotionDecoderConfig

    Returns:
        decoded_data (dict): containing the following fields:
            ego_available (bool): whether the egopose data is available
            ego_t0 (torch.tensor): (num_sample,)
                the absolute start time of the egopose data in microseconds
            ego_t0_relative (torch.tensor): (num_sample,) the relative start time of the egopose
                data in seconds it is normalized to the first timestamp of the base_timestamps
            ego_t0_frame_idx (torch.tensor): (num_sample,) the frame index of the base frame
            prediction_start_offset (torch.tensor): (num_sample,)
                the prediction start time offset
            ego_history_tvals (torch.tensor): (Th,)
                time in seconds corresponding to each position
            ego_history_xyz (torch.tensor): (num_sample,Th,3)
                the ego history (live) x,y,z positions
            ego_history_rot (torch.tensor): (num_sample,Th,3,3)
                the ego history (live) orientations as 3x3 matrices
            ego_future_tvals (torch.tensor): (Tf,)
                time in seconds corresponding to each position
            ego_future_xyz (torch.tensor): (num_sample,Tf,3)
                the ego future (gt) x,y,z positions
            ego_future_rot (torch.tensor): (num_sample,Tf,3,3)
                the ego future (gt) orientations as 3x3 matrices
            route_xy (torch.tensor): (num_sample, config.num_route_points, 3)
                the route x,y positions (from ego gt)

    """
    ego_pose = load_egopose(data, live=False)
    live_ego_pose = load_egopose(data, live=True)
    ego_history_tvals = [config.time_step * t for t in range(-config.num_history_steps + 1, 1)]
    ego_future_tvals = [config.time_step * t for t in range(1, config.num_future_steps + 1)]
    return interpolate_egopose(
        ego_lerp_inp=ego_pose,
        live_ego_lerp_inp=live_ego_pose,
        prediction_start_offset_range=config.prediction_start_offset_range,
        ego_history_tvals=ego_history_tvals,
        ego_future_tvals=ego_future_tvals,
        base_timestamps=base_timestamps,
        decode_strategy=config.decode_strategy,
        num_route_points=config.num_route_points,
    )


def frame_to_microseconds(frame_idx: int, fps: float = 30) -> int:
    """Convert a frame index to microseconds.

    Args:
        frame_idx: The frame index to convert.
        fps: The frame rate of the video.

    Returns:
        The frame index in microseconds.

    """
    return int(frame_idx * 1_000_000 / fps)


def decode(egomotion: dict[str, bytes], start: int, end: int) -> str:
    """Decode the egomotion data.

    Args:
        egomotion: The egomotion data.
        start: The start frame index.
        end: The end frame index.

    Returns:
        The decoded egomotion data.

    """
    ed_config = EgoMotionDecoderConfig(decode_strategy="uniform_2_frame")

    metadata = json.loads(egomotion["camera_front_wide_120fov.json"])
    frame_map = {frame["frame_num"]: frame["timestamp"] for frame in metadata}

    num_frames_per_view = 32  # 4 fps
    frame_idxs = list(range(start, end, (end - start) // num_frames_per_view))
    timestamps = [frame_map[frame_idx] for frame_idx in frame_idxs]
    ego_data = decode_egomotion(egomotion, timestamps, ed_config)

    # flatten xyz and Rotation into a 12-dim vector
    xyz = ego_data["ego_future_xyz"]
    rot = ego_data["ego_future_rot"]
    # traj is of shape (B, N, 12) where N is the number of future ego poses
    # default N = 64, 6.4 seconds, each one sampled at 0.1 sec interval.
    traj = torch.cat([xyz, rot.flatten(2, 3)], dim=-1)
    xyzs = traj[:, :, :3]

    trajectory = [f"({xyzs[0, j, 0]:.2f}, {xyzs[0, j, 1]:.2f})" for j in range(0, 48, 48 // 16)]
    return ", ".join(trajectory)
