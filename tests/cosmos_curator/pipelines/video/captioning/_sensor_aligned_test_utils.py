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
"""Configurable Sensor Library fixture and alignment helpers for captioning tests."""

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.data.preintegrated_imu_data import (
    ImuIntegrationInvalidReason,
    PreintegratedImuData,
)
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.sensors.group import Sensor, SensorGroup
from cosmos_curator.core.sensors.sensors.imu_sensor import ImuSensor
from cosmos_curator.core.sensors.sensors.preintegrated_imu_sensor import PreintegratedImuSensor
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    make_sampling_grid,
    protobuf_descriptor_set_from_proto,
    protobuf_message_class,
    write_protobuf_mcap,
)

FRONT_CAMERA_SENSOR_ID = "camera.front"
REAR_CAMERA_SENSOR_ID = "camera.rear"
IMU_SENSOR_ID = "imu"
CHECKED_IN_ALIGNED_ROWS = 40
CHECKED_IN_FRONT_FRAME_SHAPE = (CHECKED_IN_ALIGNED_ROWS, 3, 480, 854)
CHECKED_IN_FRONT_FRAMES_SHA256 = "96a639a18766baabf6ba92ba12cde4e33ff9ce378866d4ca1d8ed93ccee5a18e"

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VIDEO_FIXTURE_DIR = _REPO_ROOT / "tests" / "cosmos_curator" / "pipelines" / "video" / "data"
_REFERENCE_IMU_SCHEMA_NAME = "cosmos_curator.sensors.imu.v1.ImuSample"
_REFERENCE_IMU_PROTO_PATH = _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "schemas" / "imu.proto"
_REFERENCE_IMU_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "imu_protobuf_mapping.yaml"
)
_TEN_SECONDS_NS = 10_000_000_000
_CAMERA_SAMPLE_PERIOD_NS = 250_000_000
_IMU_SAMPLE_PERIOD_NS = 10_000_000


# Input contracts separate variable sources from derived alignment evidence.
class CameraPairMode(enum.Enum):
    """Evidence contract for camera-source provenance and timeline relationships.

    ``INDEPENDENT_PAIR`` still requires two genuinely synchronized camera
    sources. It permits independently derived selections; it is not an escape
    hatch for a failing identical-timeline assertion.
    """

    MIRRORED_SOURCE = "mirrored_source"
    IDENTICAL_TIMELINE_PAIR = "identical_timeline_pair"
    INDEPENDENT_PAIR = "independent_pair"


@dataclass(frozen=True)
class SensorEpisodeInput:
    """Variable local inputs for one sensor-aligned episode."""

    front_source: str | Path
    rear_source: str | Path
    start_ns: int = 0
    duration_ns: int = _TEN_SECONDS_NS
    sampling_period_ns: int = _CAMERA_SAMPLE_PERIOD_NS
    pair_mode: CameraPairMode = CameraPairMode.IDENTICAL_TIMELINE_PAIR


CHECKED_IN_CODEC_INPUT = SensorEpisodeInput(
    front_source=_VIDEO_FIXTURE_DIR / "test_clip_10s_bframes.mp4",
    rear_source=_VIDEO_FIXTURE_DIR / "test_clip_10s.mp4",
)


@dataclass(frozen=True)
class SensorSessionDescriptor:
    """Serializable source locators and timing parameters for one sensor session."""

    episode_input: SensorEpisodeInput
    imu_mcap_path: str
    imu_proto_path: str
    imu_mapping_path: str
    imu_schema_name: str
    front_camera_sensor_id: str
    rear_camera_sensor_id: str
    imu_sensor_id: str
    exclusive_end_ns: int
    imu_sample_period_ns: int

    @property
    def start_ns(self) -> int:
        """Return the requested episode start timestamp."""
        return self.episode_input.start_ns

    @property
    def camera_sample_period_ns(self) -> int:
        """Return the requested aligned-camera sampling period."""
        return self.episode_input.sampling_period_ns


@dataclass(frozen=True)
class AlignmentContract:
    """Structural expectations derived without reproducing sensor sampling."""

    pair_mode: CameraPairMode
    aligned_rows: int
    front_frame_shape_tchw: tuple[int, int, int, int]
    rear_frame_shape_nhwc: tuple[int, int, int, int]
    front_has_bframes: bool
    rear_has_bframes: bool


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Measured alignment and selection properties retained for the GPU artifact."""

    max_front_alignment_error_ns: int
    max_rear_alignment_error_ns: int
    max_cross_camera_skew_ns: int
    valid_imu_interval_count: int
    camera_frames_selected: int


@dataclass(frozen=True)
class ImuQualityDiagnostics:
    """Uniform synthetic-IMU integration evidence measured over valid intervals."""

    timestamp_domain_mode: str
    valid_interval_count: int
    integration_duration_ns: int
    sample_count_total_min: int
    sample_count_total_max: int
    sample_count_used_min: int
    sample_count_used_max: int
    sample_count_rejected_min: int
    sample_count_rejected_max: int
    max_inter_sample_gap_ns: int
    delta_velocity_m_s: tuple[float, float, float]
    delta_position_m: tuple[float, float, float]


@dataclass(frozen=True)
class SensorAlignmentResult:
    """In-process aligned result and exact front-camera model input frames."""

    aligned_frame: AlignedFrame
    contract: AlignmentContract
    diagnostics: AlignmentDiagnostics
    imu_quality: ImuQualityDiagnostics
    front_frames_tchw: npt.NDArray[np.uint8]
    front_frames_sha256: str


# Source utilities normalize local provenance, including symlink aliases.
def _resolve_local_camera_source(source: str | Path) -> Path:
    """Resolve one local camera path while rejecting remote URI inputs."""
    if "://" in str(source):
        msg = f"camera source must be a local path, got {source!r}"
        raise ValueError(msg)
    return Path(source).expanduser().resolve()


def _camera_sources_match(front_source: str | Path, rear_source: str | Path) -> bool:
    """Compare normalized local camera paths, including symlink aliases."""
    front_path = _resolve_local_camera_source(front_source)
    rear_path = _resolve_local_camera_source(rear_source)
    try:
        return front_path.samefile(rear_path)
    except FileNotFoundError:
        return front_path == rear_path


def _validate_episode_input(episode_input: SensorEpisodeInput) -> None:
    """Reject inputs that cannot form one exact, regularly spaced episode grid."""
    if episode_input.duration_ns <= 0:
        msg = f"duration_ns must be positive, got {episode_input.duration_ns}"
        raise ValueError(msg)
    if episode_input.sampling_period_ns <= 0:
        msg = f"sampling_period_ns must be positive, got {episode_input.sampling_period_ns}"
        raise ValueError(msg)
    if episode_input.duration_ns % episode_input.sampling_period_ns != 0:
        msg = (
            "duration_ns must be exactly divisible by sampling_period_ns: "
            f"duration={episode_input.duration_ns}, period={episode_input.sampling_period_ns}"
        )
        raise ValueError(msg)
    if episode_input.duration_ns < 2 * episode_input.sampling_period_ns:
        msg = (
            "duration_ns must produce at least two aligned rows for IMU preintegration: "
            f"duration={episode_input.duration_ns}, period={episode_input.sampling_period_ns}"
        )
        raise ValueError(msg)
    if episode_input.duration_ns % _IMU_SAMPLE_PERIOD_NS != 0:
        msg = (
            "duration_ns must be exactly divisible by the synthetic IMU sampling period: "
            f"duration={episode_input.duration_ns}, imu_period={_IMU_SAMPLE_PERIOD_NS}"
        )
        raise ValueError(msg)
    sources_match = _camera_sources_match(episode_input.front_source, episode_input.rear_source)
    if episode_input.pair_mode is CameraPairMode.MIRRORED_SOURCE and not sources_match:
        msg = "MIRRORED_SOURCE requires front_source and rear_source to resolve to the same input"
        raise ValueError(msg)
    if episode_input.pair_mode is not CameraPairMode.MIRRORED_SOURCE and sources_match:
        msg = f"{episode_input.pair_mode.name} requires two distinct camera sources"
        raise ValueError(msg)


# IMU fixture utilities write and validate a real protobuf-backed MCAP sequence.
def _expected_imu_sample_count(descriptor: SensorSessionDescriptor) -> int:
    """Return the inclusive-end synthetic IMU sample count."""
    duration_ns = descriptor.exclusive_end_ns - descriptor.start_ns
    assert duration_ns % descriptor.imu_sample_period_ns == 0, (
        "episode duration must be divisible by the IMU period: "
        f"duration={duration_ns}, period={descriptor.imu_sample_period_ns}"
    )
    return duration_ns // descriptor.imu_sample_period_ns + 1


def _write_synthetic_imu_mcap(descriptor: SensorSessionDescriptor) -> None:
    """Write valid constant-motion IMU samples through the reference protobuf contract."""
    descriptor_set = protobuf_descriptor_set_from_proto(Path(descriptor.imu_proto_path))
    message_cls = protobuf_message_class(descriptor_set, descriptor.imu_schema_name)
    timestamps_ns = np.arange(
        descriptor.start_ns,
        descriptor.exclusive_end_ns + descriptor.imu_sample_period_ns,
        descriptor.imu_sample_period_ns,
        dtype=np.int64,
    )
    samples: list[McapSample] = []
    for sequence, timestamp_ns in enumerate(timestamps_ns):
        sample = message_cls()
        sample.sensor_timestamp_ns = int(timestamp_ns)  # type: ignore[attr-defined]
        sample.host_timestamp_ns = int(timestamp_ns)  # type: ignore[attr-defined]
        sample.linear_acceleration_x_m_s2 = 2.0  # type: ignore[attr-defined]
        sample.angular_velocity_x_valid = True  # type: ignore[attr-defined]
        sample.angular_velocity_y_valid = True  # type: ignore[attr-defined]
        sample.angular_velocity_z_valid = True  # type: ignore[attr-defined]
        sample.linear_acceleration_x_valid = True  # type: ignore[attr-defined]
        sample.linear_acceleration_y_valid = True  # type: ignore[attr-defined]
        sample.linear_acceleration_z_valid = True  # type: ignore[attr-defined]
        sample.sequence_counter = sequence  # type: ignore[attr-defined]
        samples.append(McapSample(int(timestamp_ns), sample.SerializeToString()))

    expected_samples = _expected_imu_sample_count(descriptor)
    assert len(samples) == expected_samples, (
        f"synthetic IMU sample count: expected {expected_samples}, got {len(samples)}"
    )
    write_protobuf_mcap(
        Path(descriptor.imu_mcap_path),
        samples,
        topic="/imu",
        schema_name=descriptor.imu_schema_name,
        schema_data=descriptor_set.SerializeToString(),
        library="sensor-aligned captioning test",
    )


def prepare_sensor_session(
    tmp_path: Path,
    episode_input: SensorEpisodeInput = CHECKED_IN_CODEC_INPUT,
) -> SensorSessionDescriptor:
    """Create the real MCAP fixture and bind it to configurable camera inputs."""
    _validate_episode_input(episode_input)
    descriptor = SensorSessionDescriptor(
        episode_input=episode_input,
        imu_mcap_path=str(tmp_path / "imu.mcap"),
        imu_proto_path=str(_REFERENCE_IMU_PROTO_PATH),
        imu_mapping_path=str(_REFERENCE_IMU_MAPPING_PATH),
        imu_schema_name=_REFERENCE_IMU_SCHEMA_NAME,
        front_camera_sensor_id=FRONT_CAMERA_SENSOR_ID,
        rear_camera_sensor_id=REAR_CAMERA_SENSOR_ID,
        imu_sensor_id=IMU_SENSOR_ID,
        exclusive_end_ns=episode_input.start_ns + episode_input.duration_ns,
        imu_sample_period_ns=_IMU_SAMPLE_PERIOD_NS,
    )
    _write_synthetic_imu_mcap(descriptor)
    return descriptor


# Frame identity covers the exact contiguous tensor handed to captioning.
def compute_frame_sha256(frames: npt.NDArray[np.uint8]) -> str:
    """Hash frame bytes in their C-order tensor layout."""
    return hashlib.sha256(frames.tobytes(order="C")).hexdigest()


def _derive_alignment_contract(
    descriptor: SensorSessionDescriptor,
    front_sensor: CameraSensor,
    rear_sensor: CameraSensor,
    grid_ns: npt.NDArray[np.int64],
) -> AlignmentContract:
    """Derive input structure while leaving sampling behavior to the Sensor Library."""
    assert front_sensor.start_ns <= descriptor.start_ns, (
        f"front camera starts after the episode: camera={front_sensor.start_ns}, episode={descriptor.start_ns}"
    )
    assert rear_sensor.start_ns <= descriptor.start_ns, (
        f"rear camera starts after the episode: camera={rear_sensor.start_ns}, episode={descriptor.start_ns}"
    )
    assert front_sensor.end_ns >= int(grid_ns[-1]), (
        f"front camera ends before the final grid row: camera={front_sensor.end_ns}, grid={int(grid_ns[-1])}"
    )
    assert rear_sensor.end_ns >= int(grid_ns[-1]), (
        f"rear camera ends before the final grid row: camera={rear_sensor.end_ns}, grid={int(grid_ns[-1])}"
    )

    pair_mode = descriptor.episode_input.pair_mode
    if pair_mode in {CameraPairMode.MIRRORED_SOURCE, CameraPairMode.IDENTICAL_TIMELINE_PAIR}:
        np.testing.assert_array_equal(front_sensor.timestamps_ns, rear_sensor.timestamps_ns)

    aligned_rows = len(grid_ns)
    return AlignmentContract(
        pair_mode=pair_mode,
        aligned_rows=aligned_rows,
        front_frame_shape_tchw=(
            aligned_rows,
            3,
            front_sensor.video_metadata.height,
            front_sensor.video_metadata.width,
        ),
        rear_frame_shape_nhwc=(
            aligned_rows,
            rear_sensor.video_metadata.height,
            rear_sensor.video_metadata.width,
            3,
        ),
        front_has_bframes=front_sensor.has_bframes,
        rear_has_bframes=rear_sensor.has_bframes,
    )


def _summarize_imu_quality(imu_data: PreintegratedImuData) -> ImuQualityDiagnostics:
    """Require uniform valid intervals and retain their compact measured evidence."""
    np.testing.assert_array_equal(imu_data.align_timestamps_ns, imu_data.sensor_timestamps_ns)
    valid = imu_data.integration_valid
    valid_interval_count = int(np.count_nonzero(valid))
    assert valid_interval_count > 0, "synthetic IMU must produce at least one valid integration interval"

    sample_count_total = imu_data.sample_count_total[valid]
    sample_count_used = imu_data.sample_count_used[valid]
    sample_count_rejected = imu_data.sample_count_rejected[valid]
    np.testing.assert_array_equal(sample_count_used + sample_count_rejected, sample_count_total)

    def uniform_integer(values: npt.NDArray[np.integer], name: str) -> int:
        unique_values = np.unique(values)
        assert len(unique_values) == 1, f"{name} must be uniform over valid intervals, got {unique_values.tolist()}"
        return int(unique_values[0])

    delta_velocity = imu_data.delta_velocity_m_s[valid]
    delta_position = imu_data.delta_position_m[valid]
    assert np.all(np.isfinite(delta_velocity)), "valid IMU delta velocity values must be finite"
    assert np.all(np.isfinite(delta_position)), "valid IMU delta position values must be finite"
    np.testing.assert_allclose(
        delta_velocity,
        np.broadcast_to(delta_velocity[0], delta_velocity.shape),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        delta_position,
        np.broadcast_to(delta_position[0], delta_position.shape),
        rtol=0.0,
        atol=1e-12,
    )

    return ImuQualityDiagnostics(
        timestamp_domain_mode="shared_synthetic_timeline",
        valid_interval_count=valid_interval_count,
        integration_duration_ns=uniform_integer(imu_data.integration_duration_ns[valid], "integration duration"),
        sample_count_total_min=int(np.min(sample_count_total)),
        sample_count_total_max=int(np.max(sample_count_total)),
        sample_count_used_min=int(np.min(sample_count_used)),
        sample_count_used_max=int(np.max(sample_count_used)),
        sample_count_rejected_min=int(np.min(sample_count_rejected)),
        sample_count_rejected_max=int(np.max(sample_count_rejected)),
        max_inter_sample_gap_ns=int(np.max(imu_data.max_inter_sample_gap_ns[valid])),
        delta_velocity_m_s=(
            float(delta_velocity[0, 0]),
            float(delta_velocity[0, 1]),
            float(delta_velocity[0, 2]),
        ),
        delta_position_m=(
            float(delta_position[0, 0]),
            float(delta_position[0, 1]),
            float(delta_position[0, 2]),
        ),
    )


def align_sensor_session(descriptor: SensorSessionDescriptor) -> SensorAlignmentResult:
    """Align configurable cameras and synthetic preintegrated IMU into one window."""
    # Build one half-open episode grid whose final boundary is the exclusive end.
    boundary_timestamps_ns = np.arange(
        descriptor.start_ns,
        descriptor.exclusive_end_ns + descriptor.camera_sample_period_ns,
        descriptor.camera_sample_period_ns,
        dtype=np.int64,
    )
    assert boundary_timestamps_ns[-1] == descriptor.exclusive_end_ns, (
        "alignment boundary must land exactly on the session end: "
        f"expected {descriptor.exclusive_end_ns}, got {boundary_timestamps_ns[-1]}"
    )
    grid = make_sampling_grid(
        boundary_timestamps_ns,
        stride_ns=descriptor.exclusive_end_ns - descriptor.start_ns,
        duration_ns=descriptor.exclusive_end_ns - descriptor.start_ns,
    )
    expected_rows = descriptor.episode_input.duration_ns // descriptor.camera_sample_period_ns
    assert len(grid.timestamps_ns) == expected_rows, (
        f"alignment row count: expected {expected_rows}, got {len(grid.timestamps_ns)}"
    )

    # Decode and sample the local cameras with the synthetic protobuf-backed IMU.
    front_sensor = CameraSensor(_resolve_local_camera_source(descriptor.episode_input.front_source))
    rear_sensor = CameraSensor(_resolve_local_camera_source(descriptor.episode_input.rear_source))
    contract = _derive_alignment_contract(descriptor, front_sensor, rear_sensor, grid.timestamps_ns)

    raw_imu_sensor = ImuSensor(
        Path(descriptor.imu_mcap_path),
        schema_name=descriptor.imu_schema_name,
        protobuf_mapping=Path(descriptor.imu_mapping_path),
    )
    assert raw_imu_sensor.start_ns == descriptor.start_ns, (
        f"IMU start timestamp: expected {descriptor.start_ns}, got {raw_imu_sensor.start_ns}"
    )
    assert raw_imu_sensor.end_ns == descriptor.exclusive_end_ns, (
        f"IMU end timestamp: expected {descriptor.exclusive_end_ns}, got {raw_imu_sensor.end_ns}"
    )
    expected_imu_samples = _expected_imu_sample_count(descriptor)
    assert len(raw_imu_sensor.timestamps_ns) == expected_imu_samples, (
        f"decoded IMU sample count: expected {expected_imu_samples}, got {len(raw_imu_sensor.timestamps_ns)}"
    )
    imu_sensor = PreintegratedImuSensor(raw_imu_sensor)
    sensors = {
        descriptor.front_camera_sensor_id: cast("Sensor", front_sensor),
        descriptor.rear_camera_sensor_id: cast("Sensor", rear_sensor),
        descriptor.imu_sensor_id: cast("Sensor", imu_sensor),
    }
    sensor_group = SensorGroup(sensors)
    spec = SamplingSpec(grid=grid)
    policies = {
        descriptor.front_camera_sensor_id: NearestTimestampPolicy(),
        descriptor.rear_camera_sensor_id: NearestTimestampPolicy(),
        descriptor.imu_sensor_id: NoSamplingPolicy(),
    }
    (aligned_frame,) = sensor_group.sample(spec, policies=policies)

    # Measure the aligned output before applying any acceptance assertions.
    expected_sensor_ids = {
        descriptor.front_camera_sensor_id,
        descriptor.rear_camera_sensor_id,
        descriptor.imu_sensor_id,
    }
    assert set(aligned_frame.sensor_data) == expected_sensor_ids, (
        f"aligned sensor ids: expected {expected_sensor_ids}, got {set(aligned_frame.sensor_data)}"
    )
    front_data = aligned_frame[descriptor.front_camera_sensor_id]
    rear_data = aligned_frame[descriptor.rear_camera_sensor_id]
    imu_data = aligned_frame[descriptor.imu_sensor_id]
    assert isinstance(front_data, CameraData), (
        f"front payload type: expected CameraData, got {type(front_data).__name__}"
    )
    assert isinstance(rear_data, CameraData), f"rear payload type: expected CameraData, got {type(rear_data).__name__}"
    assert isinstance(imu_data, PreintegratedImuData), (
        f"IMU payload type: expected PreintegratedImuData, got {type(imu_data).__name__}"
    )

    diagnostics = AlignmentDiagnostics(
        max_front_alignment_error_ns=int(
            np.max(np.abs(front_data.sensor_timestamps_ns - aligned_frame.align_timestamps_ns))
        ),
        max_rear_alignment_error_ns=int(
            np.max(np.abs(rear_data.sensor_timestamps_ns - aligned_frame.align_timestamps_ns))
        ),
        max_cross_camera_skew_ns=int(np.max(np.abs(front_data.sensor_timestamps_ns - rear_data.sensor_timestamps_ns))),
        valid_imu_interval_count=int(np.count_nonzero(imu_data.integration_valid)),
        camera_frames_selected=len(front_data.frames) + len(rear_data.frames),
    )
    imu_quality = _summarize_imu_quality(imu_data)
    front_frames_tchw = np.ascontiguousarray(front_data.frames.transpose(0, 3, 1, 2))
    assert front_frames_tchw.dtype == np.uint8, f"front frame dtype: expected uint8, got {front_frames_tchw.dtype}"
    assert front_frames_tchw.flags.c_contiguous, "front TCHW frame tensor must be C-contiguous"

    return SensorAlignmentResult(
        aligned_frame=aligned_frame,
        contract=contract,
        diagnostics=diagnostics,
        imu_quality=imu_quality,
        front_frames_tchw=front_frames_tchw,
        front_frames_sha256=compute_frame_sha256(front_frames_tchw),
    )


# Generic assertions cover every case; the checked-in fixture adds codec-specific guarantees.
def assert_sensor_alignment(result: SensorAlignmentResult) -> None:
    """Validate structural alignment invariants without duplicating the sampling algorithm."""
    contract = result.contract
    diagnostics = result.diagnostics
    aligned_rows = contract.aligned_rows
    assert len(result.aligned_frame.align_timestamps_ns) == aligned_rows, (
        f"aligned frame row count: expected {aligned_rows}, got {len(result.aligned_frame.align_timestamps_ns)}"
    )

    front_data = result.aligned_frame[FRONT_CAMERA_SENSOR_ID]
    rear_data = result.aligned_frame[REAR_CAMERA_SENSOR_ID]
    assert isinstance(front_data, CameraData), (
        f"front payload type: expected CameraData, got {type(front_data).__name__}"
    )
    assert isinstance(rear_data, CameraData), f"rear payload type: expected CameraData, got {type(rear_data).__name__}"
    assert len(front_data.sensor_timestamps_ns) == aligned_rows, (
        f"front-camera row count: expected {aligned_rows}, got {len(front_data.sensor_timestamps_ns)}"
    )
    assert len(rear_data.sensor_timestamps_ns) == aligned_rows, (
        f"rear-camera row count: expected {aligned_rows}, got {len(rear_data.sensor_timestamps_ns)}"
    )
    assert result.front_frames_tchw.shape == contract.front_frame_shape_tchw, (
        f"front frame shape: expected {contract.front_frame_shape_tchw}, got {result.front_frames_tchw.shape}"
    )
    assert rear_data.frames.shape == contract.rear_frame_shape_nhwc, (
        f"rear frame shape: expected {contract.rear_frame_shape_nhwc}, got {rear_data.frames.shape}"
    )
    if contract.pair_mode in {CameraPairMode.MIRRORED_SOURCE, CameraPairMode.IDENTICAL_TIMELINE_PAIR}:
        assert diagnostics.max_cross_camera_skew_ns == 0, (
            f"equal-timeline pair must have zero skew, got {diagnostics.max_cross_camera_skew_ns} ns"
        )

    expected_valid_intervals = aligned_rows - 1
    assert diagnostics.valid_imu_interval_count == expected_valid_intervals, (
        f"valid IMU intervals: expected {expected_valid_intervals}, got {diagnostics.valid_imu_interval_count}"
    )
    assert result.imu_quality.valid_interval_count == diagnostics.valid_imu_interval_count
    expected_camera_frames = aligned_rows * 2
    assert diagnostics.camera_frames_selected == expected_camera_frames, (
        f"selected camera frames: expected {expected_camera_frames}, got {diagnostics.camera_frames_selected}"
    )

    imu_data = result.aligned_frame[IMU_SENSOR_ID]
    assert isinstance(imu_data, PreintegratedImuData), (
        f"IMU payload type: expected PreintegratedImuData, got {type(imu_data).__name__}"
    )
    expected_validity = [False, *([True] * expected_valid_intervals)]
    actual_validity = imu_data.integration_valid.tolist()
    assert actual_validity == expected_validity, (
        f"IMU integration validity: expected {expected_validity}, got {actual_validity}"
    )
    first_invalid_reason = int(imu_data.integration_invalid_reason[0])
    assert first_invalid_reason == ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value, (
        "first IMU invalid reason: "
        f"expected {ImuIntegrationInvalidReason.FIRST_ALIGNMENT.value}, got {first_invalid_reason}"
    )
    remaining_invalid_reasons = imu_data.integration_invalid_reason[1:].tolist()
    assert np.all(imu_data.integration_invalid_reason[1:] == ImuIntegrationInvalidReason.NONE.value), (
        f"remaining IMU invalid reasons: expected all 0, got {remaining_invalid_reasons}"
    )


def assert_checked_in_codec_alignment(result: SensorAlignmentResult) -> None:
    """Apply the unconditional stronger contract for the collected local fixture case."""
    contract = result.contract
    diagnostics = result.diagnostics
    assert contract.pair_mode is CameraPairMode.IDENTICAL_TIMELINE_PAIR
    assert contract.aligned_rows == CHECKED_IN_ALIGNED_ROWS
    assert result.front_frames_tchw.shape == CHECKED_IN_FRONT_FRAME_SHAPE
    assert result.front_frames_sha256 == CHECKED_IN_FRONT_FRAMES_SHA256, (
        f"checked-in front frame SHA-256: expected {CHECKED_IN_FRONT_FRAMES_SHA256}, got {result.front_frames_sha256}"
    )
    assert contract.front_has_bframes is True, (
        f"front checked-in fixture must signal B-frames, got {contract.front_has_bframes}"
    )
    assert contract.rear_has_bframes is False, (
        f"rear checked-in fixture must not signal B-frames, got {contract.rear_has_bframes}"
    )
    front_data = result.aligned_frame[FRONT_CAMERA_SENSOR_ID]
    rear_data = result.aligned_frame[REAR_CAMERA_SENSOR_ID]
    assert isinstance(front_data, CameraData)
    assert isinstance(rear_data, CameraData)
    np.testing.assert_array_equal(front_data.sensor_timestamps_ns, result.aligned_frame.align_timestamps_ns)
    np.testing.assert_array_equal(rear_data.sensor_timestamps_ns, result.aligned_frame.align_timestamps_ns)
    assert diagnostics.max_front_alignment_error_ns == 0, (
        f"checked-in front alignment error: expected 0 ns, got {diagnostics.max_front_alignment_error_ns} ns"
    )
    assert diagnostics.max_rear_alignment_error_ns == 0, (
        f"checked-in rear alignment error: expected 0 ns, got {diagnostics.max_rear_alignment_error_ns} ns"
    )
    assert diagnostics.max_cross_camera_skew_ns == 0, (
        f"checked-in cross-camera skew: expected 0 ns, got {diagnostics.max_cross_camera_skew_ns} ns"
    )
    assert diagnostics.valid_imu_interval_count == CHECKED_IN_ALIGNED_ROWS - 1, (
        "checked-in valid IMU intervals: "
        f"expected {CHECKED_IN_ALIGNED_ROWS - 1}, got {diagnostics.valid_imu_interval_count}"
    )
    imu_quality = result.imu_quality
    assert imu_quality.timestamp_domain_mode == "shared_synthetic_timeline"
    assert imu_quality.valid_interval_count == CHECKED_IN_ALIGNED_ROWS - 1
    assert imu_quality.integration_duration_ns == _CAMERA_SAMPLE_PERIOD_NS
    assert imu_quality.sample_count_total_min == 26
    assert imu_quality.sample_count_total_max == 26
    assert imu_quality.sample_count_used_min == 26
    assert imu_quality.sample_count_used_max == 26
    assert imu_quality.sample_count_rejected_min == 0
    assert imu_quality.sample_count_rejected_max == 0
    assert imu_quality.max_inter_sample_gap_ns == _IMU_SAMPLE_PERIOD_NS
    np.testing.assert_allclose(imu_quality.delta_velocity_m_s, [0.5, 0.0, 0.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(imu_quality.delta_position_m, [0.0625, 0.0, 0.0], rtol=0.0, atol=1e-12)
