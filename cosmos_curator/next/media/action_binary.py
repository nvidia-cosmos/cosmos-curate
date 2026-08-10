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

"""ACT2 binary codec for per-frame robot action data.

Binary layout::

    bytes[0:4]    magic b"ACT2"
    bytes[4:8]    little-endian uint32 JSON length
    bytes[8:1024] UTF-8 JSON header padded to 1024 bytes total (including prefix)
    bytes[1024:]  fixed-stride frame records then per-clip tail
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

MAGIC: bytes = b"ACT2"
_VERSION: int = 2
_HEADER_SIZE: int = 1024
_HEADER_PREFIX_SIZE: int = 8
_DTYPE = np.dtype("<f4")


@dataclass(frozen=True)
class ActionFieldSpec:
    """One named array within a frame or clip record."""

    name: str
    shape: tuple[int, ...]

    @property
    def value_count(self) -> int:
        """Total number of float32 values in this field."""
        return prod(self.shape)


@dataclass(frozen=True)
class ActionBinarySpec:
    """Reusable fixed-stride action layout for one dataset family."""

    per_frame_fields: tuple[ActionFieldSpec, ...]
    per_clip_fields: tuple[ActionFieldSpec, ...] = ()
    dtype: np.dtype[Any] = _DTYPE

    @property
    def per_frame_value_count(self) -> int:
        """Total number of values in one frame record."""
        return sum(field.value_count for field in self.per_frame_fields)

    @property
    def frame_record_size_bytes(self) -> int:
        """Byte size of one frame record."""
        return self.per_frame_value_count * self.dtype.itemsize

    @property
    def per_clip_value_count(self) -> int:
        """Total number of values in the per-clip tail."""
        return sum(field.value_count for field in self.per_clip_fields)

    @property
    def per_clip_size_bytes(self) -> int:
        """Byte size of the per-clip tail."""
        return self.per_clip_value_count * self.dtype.itemsize


ACTION_BINARY_SPECS: dict[str, ActionBinarySpec] = {
    "libero": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
        ),
    ),
    "mecka": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "droid_lerobot": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(7,)),
        ),
    ),
    "robomind_franka": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(8,)),
            ActionFieldSpec(name="state_joint_position", shape=(8,)),
            ActionFieldSpec(name="state_end_effector", shape=(6,)),
        ),
    ),
    "robomind_franka_dual": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(16,)),
            ActionFieldSpec(name="state_joint_position", shape=(16,)),
            ActionFieldSpec(name="state_end_effector", shape=(12,)),
        ),
    ),
    "robomind_ur": ActionBinarySpec(
        per_frame_fields=(ActionFieldSpec(name="action", shape=(7,)),),
    ),
}

ACTION_BINARY_SPEC_BY_DATASET: dict[str, str] = {
    "libero_10": "libero",
    "libero_90": "libero",
    "feb_08_500hr_lerobot_no_bframes": "mecka",
    "mar_6_ego_dexterous_lerobot_no_bframes": "mecka",
    "feb_08_500hr_lerobot_updated": "mecka",
    "feb_15_1500hr_lerobot_no_bframes": "mecka",
    "feb_23_1000hr_lerobot_no_bframes": "mecka",
    "mar_02_1000hr_lerobot": "mecka",
    "mar_09_4000hr_lerobot": "mecka",
    "mar_16_7000hr_lerobot": "mecka",
    "mar_30_9000hr_lerobot": "mecka",
    "apr_03_lerobot": "mecka",
    "apr_06_10000hr_lerobot": "mecka",
    "droid_lerobot": "droid_lerobot",
    "droid_plus_lerobot_640x360_20260412": "droid_lerobot",
    "droid_plus_lerobot_640x360_20260412_success": "droid_lerobot",
    "robomind-franka": "robomind_franka",
    "robomind_franka": "robomind_franka",
    "robomind-franka-dual": "robomind_franka_dual",
    "robomind_franka_dual": "robomind_franka_dual",
    "robomind-ur": "robomind_ur",
    "robomind_ur": "robomind_ur",
}


def get_action_binary_spec(source_dataset: str) -> ActionBinarySpec:
    """Return the fixed-stride spec for *source_dataset*.

    Raises ValueError for unregistered dataset names.
    """
    try:
        spec_name = ACTION_BINARY_SPEC_BY_DATASET[source_dataset]
        return ACTION_BINARY_SPECS[spec_name]
    except KeyError as exc:
        msg = f"No action-binary spec registered for source dataset {source_dataset!r}."
        raise ValueError(msg) from exc


def _build_header(
    spec_name: str,
    spec: ActionBinarySpec,
    source_dataset: str,
    num_frames: int,
) -> bytes:
    """Build the 1024-byte ACT2 binary header."""
    per_frame_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_frame_fields:
        per_frame_fields.append({"name": field.name, "shape": list(field.shape), "byte_offset": byte_offset})
        byte_offset += field.value_count * spec.dtype.itemsize

    per_clip_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_clip_fields:
        per_clip_fields.append({"name": field.name, "shape": list(field.shape), "byte_offset": byte_offset})
        byte_offset += field.value_count * spec.dtype.itemsize

    per_clip_offset = _HEADER_SIZE + num_frames * spec.frame_record_size_bytes
    header_obj = {
        "dtype": spec.dtype.str,
        "format_version": _VERSION,
        "frame_data_offset": _HEADER_SIZE,
        "frame_record_size_bytes": spec.frame_record_size_bytes,
        "layout": "frame_major_with_per_clip_tail",
        "num_frames": num_frames,
        "per_clip_fields": per_clip_fields,
        "per_clip_offset": per_clip_offset,
        "per_clip_size_bytes": spec.per_clip_size_bytes,
        "per_frame_fields": per_frame_fields,
        "source_dataset": source_dataset,
        "spec_name": spec_name,
    }
    header_bytes = json.dumps(header_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header_bytes) > _HEADER_SIZE - _HEADER_PREFIX_SIZE:
        msg = f"ACT2 header is too large: {len(header_bytes)} bytes"
        raise ValueError(msg)
    prefix = MAGIC + len(header_bytes).to_bytes(4, "little", signed=False)
    return prefix + header_bytes + bytes(_HEADER_SIZE - len(prefix) - len(header_bytes))


def encode_action_bin(action_data: Mapping[str, Any], source_dataset: str) -> bytes:
    """Return an ACT2 payload for *action_data* using the spec for *source_dataset*.

    Validates that the provided field names match the registered spec exactly,
    then concatenates the fixed-size header, frame-major body, and per-clip tail.
    """
    try:
        spec_name = ACTION_BINARY_SPEC_BY_DATASET[source_dataset]
        spec = ACTION_BINARY_SPECS[spec_name]
    except KeyError as exc:
        msg = f"No action-binary spec registered for source dataset {source_dataset!r}."
        raise ValueError(msg) from exc

    expected_names = {field.name for field in (*spec.per_frame_fields, *spec.per_clip_fields)}
    if set(action_data) != expected_names:
        msg = (
            f"Action fields for {source_dataset!r} must be exactly {sorted(expected_names)}, got {sorted(action_data)}."
        )
        raise ValueError(msg)

    # Build frame-major body.
    columns: list[np.ndarray[Any, Any]] = []
    num_frames: int | None = None
    for field in spec.per_frame_fields:
        array = np.asarray(action_data[field.name], dtype=spec.dtype)
        if array.ndim != len(field.shape) + 1 or array.shape[1:] != field.shape:
            msg = f"Action field {field.name!r} must have shape (frames, {field.shape}), got {array.shape}."
            raise ValueError(msg)
        if num_frames is None:
            num_frames = array.shape[0]
        elif array.shape[0] != num_frames:
            msg = f"Action field {field.name!r} has {array.shape[0]} frames; expected {num_frames}."
            raise ValueError(msg)
        columns.append(array.reshape(array.shape[0], field.value_count))

    if num_frames is None:
        num_frames = 0
    frame_data = np.concatenate(columns, axis=1) if columns else np.empty((0, 0), dtype=spec.dtype)

    # Build per-clip tail.
    per_clip_columns: list[np.ndarray[Any, Any]] = []
    for field in spec.per_clip_fields:
        array = np.asarray(action_data[field.name], dtype=spec.dtype)
        if array.shape != field.shape:
            msg = f"Per-clip field {field.name!r} must have shape {field.shape}, got {array.shape}."
            raise ValueError(msg)
        per_clip_columns.append(array.reshape(field.value_count))
    per_clip_data = np.concatenate(per_clip_columns) if per_clip_columns else np.empty((0,), dtype=spec.dtype)

    return (
        _build_header(spec_name, spec, source_dataset, num_frames)
        + np.ascontiguousarray(frame_data, dtype=spec.dtype).tobytes()
        + np.ascontiguousarray(per_clip_data, dtype=spec.dtype).tobytes()
    )
