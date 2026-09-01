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
"""Tests for Foxglove CompressedVideo McapCameraSensor."""

import base64
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import numpy.typing as npt
import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from google.protobuf import descriptor_pb2
from mcap.records import Channel, Message, Schema
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors import mcap_camera_sensor
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from cosmos_curator.core.sensors.sensors.mcap_camera_sensor import (
    McapCameraSensor,
    _compressed_video_payload,
    _contains_keyframe,
    _ForwardCompressedVideoDecoder,
)
from tests.cosmos_curator.core.sensors.test_utils import make_sampling_grid

_TOPIC = "/camera/image-raw"
_SCHEMA_NAME = "foxglove.CompressedVideo"
_NS_PER_FRAME_30FPS = 33_333_333
_TEST_CLIP = Path("tests/cosmos_curator/pipelines/video/data/test_clip_10s.mp4")
_MCAP_FIXTURE_DIR = Path("tests/cosmos_curator/core/sensors/data")
_TEST_CLIP_MCAP = _MCAP_FIXTURE_DIR / "test_clip_10s.mcap"
_BFRAME_TEST_CLIP_MCAP = _MCAP_FIXTURE_DIR / "test_clip_10s_bframes.mcap"


def _policy() -> NearestTimestampPolicy:
    """Build the policy supported by McapCameraSensor."""
    return NearestTimestampPolicy()


def _file_like_source(path: Path) -> io.BytesIO:
    """Open an MCAP fixture as the primary DataSource shape used by callers."""
    return io.BytesIO(path.read_bytes())


def _make_metadata(*, width: int = 2, height: int = 2) -> VideoMetadata:
    """Build minimal VideoMetadata for McapCameraSensor tests."""
    return VideoMetadata(
        codec_name="h264",
        has_bframes=False,
        codec_profile="",
        container_format="mcap",
        height=height,
        width=width,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="rgb24",
        bit_rate_bps=0,
    )


def _decoded_frame(value: int = 0) -> npt.NDArray[np.uint8]:
    """Build a decoded RGB frame for private helper tests."""
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _compressed_video_descriptor_set() -> bytes:
    """Return schema data for a Foxglove CompressedVideo protobuf MCAP channel."""
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    CompressedVideo.DESCRIPTOR.file.CopyToProto(descriptor_set.file.add())
    return descriptor_set.SerializeToString()


def _compressed_video_schema(*, encoding: str = "jsonschema") -> Schema:
    """Build a minimal MCAP schema for Foxglove CompressedVideo tests."""
    data = b'{"title":"foxglove.CompressedVideo"}' if encoding == "jsonschema" else _compressed_video_descriptor_set()
    return Schema(id=1, name=_SCHEMA_NAME, encoding=encoding, data=data)


def _compressed_video_channel(*, message_encoding: str = "json") -> Channel:
    """Build a minimal MCAP channel for Foxglove CompressedVideo tests."""
    return Channel(id=1, topic=_TOPIC, message_encoding=message_encoding, metadata={}, schema_id=1)


def _mcap_message(data: bytes, *, log_time_ns: int = 0) -> Message:
    """Build a minimal MCAP message for parser tests."""
    return Message(channel_id=1, log_time=log_time_ns, publish_time=log_time_ns, sequence=1, data=data)


def _json_payload(format_name: str, data: bytes, *, timestamp_ns: int | None = 123) -> bytes:
    """Serialize one Foxglove CompressedVideo JSON payload."""
    payload: dict[str, object] = {
        "format": format_name,
        "data": base64.b64encode(data).decode("ascii"),
    }
    if timestamp_ns is not None:
        payload["timestamp"] = {"sec": timestamp_ns // 1_000_000_000, "nsec": timestamp_ns % 1_000_000_000}
    return json.dumps(payload).encode("utf-8")


def _protobuf_payload(format_name: str, data: bytes, *, timestamp_ns: int | None = 123) -> bytes:
    """Serialize one Foxglove CompressedVideo protobuf payload."""
    message = CompressedVideo(format=format_name, data=data)
    if timestamp_ns is not None:
        message.timestamp.seconds = timestamp_ns // 1_000_000_000
        message.timestamp.nanos = timestamp_ns % 1_000_000_000
    return message.SerializeToString()


def _annex_b_packets_from_mp4(
    path: Path,
    *,
    codec_name: str = "h264",
    packet_limit: int | None = 8,
) -> list[bytes]:
    """Demux MP4 packets and convert them to Annex B access units for tests."""
    packets: list[bytes] = []
    bitstream_filter_name = "h264_mp4toannexb" if codec_name == "h264" else "hevc_mp4toannexb"
    with av.open(path) as container:
        stream = container.streams.video[0]
        filter_context = av.bitstream.BitStreamFilterContext(bitstream_filter_name, stream)
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            for filtered in filter_context.filter(packet):
                payload = bytes(filtered)
                if payload:
                    packets.append(payload)
                    if packet_limit is not None and len(packets) >= packet_limit:
                        return packets
        for filtered in filter_context.filter(None):
            payload = bytes(filtered)
            if payload:
                packets.append(payload)
                if packet_limit is not None and len(packets) >= packet_limit:
                    break
    return packets


def _write_compressed_video_mcap(
    path: Path,
    packets: list[bytes],
    *,
    message_encoding: str = "json",
    format_name: str = "h264",
    payload_timestamp_offset_ns: int = 0,
) -> None:
    """Write one generated Foxglove CompressedVideo MCAP fixture."""
    schema_encoding = "jsonschema" if message_encoding == "json" else "protobuf"
    schema_data = (
        b'{"title":"foxglove.CompressedVideo"}' if message_encoding == "json" else _compressed_video_descriptor_set()
    )
    with path.open("wb") as out_file:
        writer = Writer(out_file, compression=CompressionType.ZSTD)
        writer.start(library="cosmos_curator sensor test")
        schema_id = writer.register_schema(name=_SCHEMA_NAME, encoding=schema_encoding, data=schema_data)
        channel_id = writer.register_channel(schema_id=schema_id, topic=_TOPIC, message_encoding=message_encoding)
        for sequence, packet in enumerate(packets):
            log_time_ns = sequence * _NS_PER_FRAME_30FPS
            payload_timestamp_ns = log_time_ns + payload_timestamp_offset_ns
            payload = (
                _json_payload(format_name, packet, timestamp_ns=payload_timestamp_ns)
                if message_encoding == "json"
                else _protobuf_payload(format_name, packet, timestamp_ns=payload_timestamp_ns)
            )
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time_ns,
                data=payload,
                publish_time=log_time_ns,
                sequence=sequence,
            )
        writer.finish()  # type: ignore[no-untyped-call]


def _write_synthetic_hevc_mp4(path: Path) -> bool:
    """Write a tiny HEVC MP4 when the local FFmpeg build can encode one."""
    try:
        with av.open(path, "w") as container:
            stream = container.add_stream("hevc", rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for value in (0, 32, 64, 96):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), value, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except (av.FFmpegError, OSError, PermissionError, ValueError):
        return False
    return True


def _sampling_spec() -> SamplingSpec:
    """Build two adjacent non-overlapping windows over the generated fixture."""
    return SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array(
                [0, _NS_PER_FRAME_30FPS, 2 * _NS_PER_FRAME_30FPS, 3 * _NS_PER_FRAME_30FPS, 4 * _NS_PER_FRAME_30FPS],
                dtype=np.int64,
            ),
            stride_ns=2 * _NS_PER_FRAME_30FPS,
            duration_ns=2 * _NS_PER_FRAME_30FPS,
        )
    )


def test_json_compressed_video_payload_is_normalized() -> None:
    """JSON CompressedVideo messages should parse base64 payloads and preserve log_time."""
    log_time_ns, foxglove_timestamp_ns, format_name, data = _compressed_video_payload(
        _compressed_video_schema(),
        _compressed_video_channel(message_encoding="json"),
        _mcap_message(_json_payload("H264", b"\x00\x00\x01\x65payload"), log_time_ns=42),
        topic=_TOPIC,
    )

    assert log_time_ns == 42
    assert foxglove_timestamp_ns == 123
    assert format_name == "h264"
    assert data == b"\x00\x00\x01\x65payload"


def test_protobuf_compressed_video_payload_is_normalized() -> None:
    """Protobuf CompressedVideo messages should parse with the Foxglove schema class."""
    log_time_ns, foxglove_timestamp_ns, format_name, data = _compressed_video_payload(
        _compressed_video_schema(encoding="protobuf"),
        _compressed_video_channel(message_encoding="protobuf"),
        _mcap_message(_protobuf_payload("h265", b"\x00\x00\x01&payload"), log_time_ns=43),
        topic=_TOPIC,
    )

    assert log_time_ns == 43
    assert foxglove_timestamp_ns == 123
    assert format_name == "h265"
    assert data == b"\x00\x00\x01&payload"


@pytest.mark.parametrize(
    ("message_encoding", "payload", "match"),
    [
        ("rgb8", b"raw", "message_encoding"),
        ("json", b"[]", "must be an object"),
        ("json", _json_payload("vp9", b"data"), "unsupported"),
        ("json", _json_payload("", b"data"), "empty format"),
        ("json", _json_payload("h264", b""), "empty data"),
    ],
)
def test_compressed_video_payload_rejects_unsupported_contracts(
    message_encoding: str,
    payload: bytes,
    match: str,
) -> None:
    """Unsupported encodings, formats, and malformed payloads should fail clearly."""
    with pytest.raises(ValueError, match=match):
        _compressed_video_payload(
            _compressed_video_schema(),
            _compressed_video_channel(message_encoding=message_encoding),
            _mcap_message(payload),
            topic=_TOPIC,
        )


def test_annex_b_keyframe_detection_covers_h264_and_h265() -> None:
    """Keyframe validation should understand the supported Annex B NAL formats."""
    assert _contains_keyframe(b"\x00\x00\x00\x01\x65h264-idr", "h264")
    assert not _contains_keyframe(b"\x00\x00\x00\x01\x41h264-p", "h264")
    assert _contains_keyframe(b"\x00\x00\x00\x01&h265-irap", "h265")
    assert not _contains_keyframe(b"\x00\x00\x00\x01\x02h265-trail", "h265")


def test_annex_b_nal_units_preserve_leading_zero_header() -> None:
    """Malformed NAL header bytes should not be silently stripped."""
    assert not _contains_keyframe(b"\x00\x00\x01\x00\x65not-idr", "h264")


def test_compressed_video_mcap_fixture_decodes_one_frame_per_message() -> None:
    """Codec-path proof: no-B-frame CompressedVideo MCAP fixture decodes sequentially."""
    sensor = McapCameraSensor(_file_like_source(_TEST_CLIP_MCAP), topic=_TOPIC)
    batches = list(sensor.sample(_sampling_spec(), policy=_policy()))

    assert len(batches) == 2
    assert [len(batch.frames) for batch in batches] == [2, 2]
    np.testing.assert_array_equal(batches[0].sensor_timestamps_ns, np.array([0, _NS_PER_FRAME_30FPS], dtype=np.int64))
    np.testing.assert_array_equal(
        batches[1].sensor_timestamps_ns,
        np.array([2 * _NS_PER_FRAME_30FPS, 3 * _NS_PER_FRAME_30FPS], dtype=np.int64),
    )
    assert batches[0].frames.dtype == np.uint8
    assert batches[0].frames.shape[1:] == (sensor.video_metadata.height, sensor.video_metadata.width, 3)
    assert sensor.video_metadata.codec_name == "h264"
    assert sensor.video_metadata.container_format == "mcap"
    assert sensor.video_metadata.pix_fmt != "rgb24"
    assert batches[0].metadata.avg_frame_rate == Fraction(1_000_000_000, _NS_PER_FRAME_30FPS)
    assert sensor.video_metadata.avg_frame_rate == Fraction(1_000_000_000, _NS_PER_FRAME_30FPS)


def test_mcap_camera_sensor_samples_from_single_seekable_stream() -> None:
    """Caller-owned streams should support sampling without requiring fresh file handles."""
    sensor = McapCameraSensor(_file_like_source(_TEST_CLIP_MCAP), topic=_TOPIC)

    batches = list(sensor.sample(_sampling_spec(), policy=_policy()))

    assert [len(batch.frames) for batch in batches] == [2, 2]


def test_mcap_camera_sensor_uses_log_time_when_payload_timestamp_differs(tmp_path: Path) -> None:
    """v1 samples by MCAP log_time, not by the CompressedVideo.timestamp payload field."""
    packets = _annex_b_packets_from_mp4(_TEST_CLIP, packet_limit=4)
    path = tmp_path / "payload-timestamp-differs.mcap"
    _write_compressed_video_mcap(path, packets, payload_timestamp_offset_ns=10 * _NS_PER_FRAME_30FPS)

    [first_batch, second_batch] = list(
        McapCameraSensor(_file_like_source(path), topic=_TOPIC).sample(_sampling_spec(), policy=_policy())
    )

    np.testing.assert_array_equal(first_batch.sensor_timestamps_ns, np.array([0, _NS_PER_FRAME_30FPS], dtype=np.int64))
    np.testing.assert_array_equal(first_batch.pts_stream, first_batch.sensor_timestamps_ns)
    np.testing.assert_array_equal(
        second_batch.sensor_timestamps_ns,
        np.array([2 * _NS_PER_FRAME_30FPS, 3 * _NS_PER_FRAME_30FPS], dtype=np.int64),
    )
    np.testing.assert_array_equal(second_batch.pts_stream, second_batch.sensor_timestamps_ns)


def test_generated_hevc_compressed_video_mcap_samples_end_to_end(tmp_path: Path) -> None:
    """HEVC support should exercise the full MCAP sample path when encoding is available."""
    hevc_mp4 = tmp_path / "hevc.mp4"
    if not _write_synthetic_hevc_mp4(hevc_mp4):
        pytest.skip("local PyAV/FFmpeg build cannot encode HEVC test fixtures")
    packets = _annex_b_packets_from_mp4(hevc_mp4, codec_name="h265", packet_limit=4)
    path = tmp_path / "compressed-video-hevc.mcap"
    _write_compressed_video_mcap(path, packets, message_encoding="json", format_name="h265")

    [batch, *_] = list(
        McapCameraSensor(_file_like_source(path), topic=_TOPIC).sample(_sampling_spec(), policy=_policy())
    )

    assert len(batch.frames) == 2
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([0, _NS_PER_FRAME_30FPS], dtype=np.int64))
    assert batch.metadata.codec_name == "hevc"


def test_generated_protobuf_compressed_video_mcap_samples_with_same_contract(tmp_path: Path) -> None:
    """The sensor should accept protobuf-encoded Foxglove CompressedVideo channels."""
    packets = _annex_b_packets_from_mp4(_TEST_CLIP, packet_limit=4)
    path = tmp_path / "compressed-video-protobuf.mcap"
    _write_compressed_video_mcap(path, packets, message_encoding="protobuf")

    [batch, *_] = list(
        McapCameraSensor(_file_like_source(path), topic=_TOPIC).sample(_sampling_spec(), policy=_policy())
    )

    assert len(batch.frames) == 2
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([0, _NS_PER_FRAME_30FPS], dtype=np.int64))
    np.testing.assert_array_equal(batch.pts_stream, batch.sensor_timestamps_ns)


def test_sampling_decodes_from_topic_start_for_late_first_window(tmp_path: Path) -> None:
    """A first window after topic start should decode earlier packets for GOP state."""
    packets = _annex_b_packets_from_mp4(_TEST_CLIP, packet_limit=6)
    path = tmp_path / "late-window.mcap"
    _write_compressed_video_mcap(path, packets, message_encoding="json")
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array(
                [3 * _NS_PER_FRAME_30FPS, 4 * _NS_PER_FRAME_30FPS, 5 * _NS_PER_FRAME_30FPS],
                dtype=np.int64,
            ),
            stride_ns=2 * _NS_PER_FRAME_30FPS,
            duration_ns=2 * _NS_PER_FRAME_30FPS,
        )
    )

    [batch] = list(McapCameraSensor(_file_like_source(path), topic=_TOPIC).sample(spec, policy=_policy()))

    assert len(batch.frames) == 2
    np.testing.assert_array_equal(
        batch.sensor_timestamps_ns,
        np.array([3 * _NS_PER_FRAME_30FPS, 4 * _NS_PER_FRAME_30FPS], dtype=np.int64),
    )


def test_sampling_rejects_overlapping_windows() -> None:
    """Overlapping SamplingGrid windows are outside the forward-only contract."""
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([0, 1, 2, 3], dtype=np.int64),
            stride_ns=1,
            duration_ns=2,
        )
    )

    with pytest.raises(ValueError, match="non-overlapping"):
        list(McapCameraSensor(b"not-used").sample(spec, policy=_policy()))


def test_sampling_rejects_non_monotonic_windows() -> None:
    """Non-monotonic window streams should fail before decode starts."""
    windows = (
        SamplingWindow(start_ns=10, exclusive_end_ns=20, timestamps_ns=np.array([10], dtype=np.int64)),
        SamplingWindow(start_ns=5, exclusive_end_ns=10, timestamps_ns=np.array([5], dtype=np.int64)),
    )
    spec = SimpleNamespace(grid=windows)

    with pytest.raises(ValueError, match="monotonically increasing"):
        list(McapCameraSensor(b"not-used").sample(spec, policy=_policy()))  # type: ignore[arg-type]


def test_empty_window_yields_empty_camera_data(tmp_path: Path) -> None:
    """A window with no messages should still yield an empty CameraData with decoded shape."""
    packets = _annex_b_packets_from_mp4(_TEST_CLIP, packet_limit=2)
    path = tmp_path / "empty-window.mcap"
    _write_compressed_video_mcap(path, packets, message_encoding="json")
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100 * _NS_PER_FRAME_30FPS, 101 * _NS_PER_FRAME_30FPS], dtype=np.int64),
            stride_ns=_NS_PER_FRAME_30FPS,
            duration_ns=_NS_PER_FRAME_30FPS,
        )
    )

    [batch] = list(McapCameraSensor(_file_like_source(path), topic=_TOPIC).sample(spec, policy=_policy()))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.pts_stream.shape == (0,)
    assert batch.frames.shape == (0, batch.metadata.height, batch.metadata.width, 3)
    assert batch.metadata.avg_frame_rate == Fraction(1_000_000_000, _NS_PER_FRAME_30FPS)


def test_a_window_with_no_decoded_frames_raises_empty_batch(tmp_path: Path) -> None:
    """A real sensor with nothing to decode in a window is what ``empty_batch`` is for.

    The window carries reference timestamps but the recording has no frames
    there, so the sensor yields zero rows and ``SensorGroup`` fails the window
    rather than handing back a frame missing a modality. This is the forward-only
    decode path, which serves each window only from its own decoded frames.
    """
    packets = _annex_b_packets_from_mp4(_TEST_CLIP, packet_limit=2)
    path = tmp_path / "no-frames-in-window.mcap"
    _write_compressed_video_mcap(path, packets, message_encoding="json")
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100 * _NS_PER_FRAME_30FPS, 101 * _NS_PER_FRAME_30FPS], dtype=np.int64),
            stride_ns=_NS_PER_FRAME_30FPS,
            duration_ns=_NS_PER_FRAME_30FPS,
        )
    )
    sensor = McapCameraSensor(_file_like_source(path), topic=_TOPIC)
    group = SensorGroup({"front": sensor})

    with pytest.raises(AlignmentError) as caught:
        list(group.sample(spec, policies={"front": _policy()}))

    assert caught.value.reason is AlignmentFailureReason.EMPTY_BATCH
    assert caught.value.sensor_id == "front"


def test_first_delta_packet_fails_before_decode() -> None:
    """A stream that starts on a delta frame should fail with a keyframe error."""
    log_time_ns, _foxglove_timestamp_ns, format_name, data = _compressed_video_payload(
        _compressed_video_schema(),
        _compressed_video_channel(message_encoding="json"),
        _mcap_message(_json_payload("h264", b"\x00\x00\x01\x41delta"), log_time_ns=0),
        topic=_TOPIC,
    )

    with pytest.raises(ValueError, match="must contain a keyframe"):
        _ForwardCompressedVideoDecoder("h264").decode(format_name, log_time_ns, data)


def test_b_frame_mcap_fixture_contract_is_rejected() -> None:
    """B-frame CompressedVideo MCAP fixture should violate the one-frame-per-message contract."""
    sensor = McapCameraSensor(_file_like_source(_BFRAME_TEST_CLIP_MCAP), topic=_TOPIC)

    with pytest.raises(ValueError, match=r"one decoded display frame|delayed frames|B-frames"):
        list(sensor.sample(_sampling_spec(), policy=_policy()))


def test_mcap_camera_sensor_timestamps_ns_caches_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timestamps_ns should load the timeline once and reuse it for start/end."""
    timeline = np.array([100, 200, 300], dtype=np.int64)

    class FakeAccessor:
        def __init__(self, source: object, topic: str) -> None:
            del source, topic
            self.timeline_calls = 0
            self.start_end_calls = 0

        @property
        def timestamps_ns(self) -> npt.NDArray[np.int64]:
            self.timeline_calls += 1
            return timeline

        @property
        def start_ns(self) -> int:
            return int(self.timestamps_ns[0])

        @property
        def end_ns(self) -> int:
            return int(self.timestamps_ns[-1])

    fake = FakeAccessor(b"not-used", _TOPIC)

    def fake_accessor(source: object, topic: str) -> FakeAccessor:
        del source, topic
        return fake

    monkeypatch.setattr(mcap_camera_sensor, "McapTopicAccessor", fake_accessor)

    sensor = McapCameraSensor(b"not-used")

    np.testing.assert_array_equal(sensor.timestamps_ns, timeline)
    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert fake.timeline_calls == 3


def test_mcap_camera_sensor_start_end_ns_can_load_without_full_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_ns/end_ns should not require loading the full timeline."""

    class FakeAccessor:
        def __init__(self, source: object, topic: str) -> None:
            del source, topic
            self.timeline_calls = 0
            self.start_end_calls = 0

        @property
        def timestamps_ns(self) -> npt.NDArray[np.int64]:
            self.timeline_calls += 1
            return np.array([100, 200, 300], dtype=np.int64)

        @property
        def start_ns(self) -> int:
            self.start_end_calls += 1
            return 100

        @property
        def end_ns(self) -> int:
            return 300

    fake = FakeAccessor(b"not-used", _TOPIC)

    def fake_accessor(source: object, topic: str) -> FakeAccessor:
        del source, topic
        return fake

    monkeypatch.setattr(mcap_camera_sensor, "McapTopicAccessor", fake_accessor)

    sensor = McapCameraSensor(b"not-used")

    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert fake.timeline_calls == 0
    assert fake.start_end_calls == 1


def test_mcap_camera_sensor_sample_returns_no_batches_when_grid_yields_nothing() -> None:
    """sample() should cleanly return when the provided grid yields no windows."""
    spec = SimpleNamespace(grid=())

    assert list(McapCameraSensor(b"not-used").sample(spec, policy=_policy())) == []  # type: ignore[arg-type]


def test_mcap_camera_sensor_reports_supported_policy() -> None:
    """McapCameraSensor should participate in explicit policy routing."""
    sensor = McapCameraSensor(b"not-used")

    assert sensor.supports_sampling_policy(NearestTimestampPolicy())
    assert not sensor.supports_sampling_policy(NoSamplingPolicy())


def test_mcap_camera_sensor_get_empty_camera_data_is_cached() -> None:
    """_get_empty_camera_data should cache and reuse the empty CameraData batch."""
    sensor = McapCameraSensor(b"not-used")
    sensor._video_metadata = _make_metadata(width=2, height=3)

    empty0 = sensor._get_empty_camera_data()
    empty1 = sensor._get_empty_camera_data()

    assert empty0 is empty1
    assert empty0.frames.shape == (0, 3, 2, 3)
    assert empty0.metadata.width == 2
    assert empty0.metadata.height == 3


def test_mcap_camera_sensor_sample_window_stacks_only_selected_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sparse sampling should not stack every decoded frame in the source window."""

    def fake_sample_window_indices(
        timestamps_ns: npt.NDArray[np.int64],
        window: SamplingWindow,
        *,
        policy: object,
        dedup: bool,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del timestamps_ns, window, policy, dedup
        return np.array([2], dtype=np.int64), np.array([1], dtype=np.int64)

    monkeypatch.setattr(mcap_camera_sensor, "sample_window_indices", fake_sample_window_indices)
    sensor = McapCameraSensor(b"not-used")
    metadata = _make_metadata(width=2, height=2)

    batch = sensor._sample_window(
        SamplingWindow(start_ns=100, exclusive_end_ns=400, timestamps_ns=np.array([300], dtype=np.int64)),
        [100, 200, 300],
        [_decoded_frame(1), _decoded_frame(2), _decoded_frame(3)],
        policy=_policy(),
        metadata=metadata,
    )

    assert batch.frames.shape == (1, 2, 2, 3)
    assert int(batch.frames[0, 0, 0, 0]) == 3


def test_mcap_camera_sensor_sample_finishes_decoder_when_generator_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Early generator close should still flush the active decoder."""
    finish_calls = 0
    messages = [
        (
            _compressed_video_schema(),
            _compressed_video_channel(message_encoding="json"),
            _mcap_message(_json_payload("h264", b"\x00\x00\x01\x65keyframe"), log_time_ns=0),
        )
    ]

    class FakeDecoder:
        def __init__(self, format_name: str) -> None:
            self.format_name = format_name

        def decode(self, format_name: str, log_time_ns: int, data: bytes) -> npt.NDArray[np.uint8]:
            del format_name, log_time_ns, data
            return _decoded_frame()

        def video_metadata(
            self,
            frame: npt.NDArray[np.uint8],
            *,
            timeline_ns: npt.NDArray[np.int64],
        ) -> VideoMetadata:
            del frame, timeline_ns
            return _make_metadata()

        def finish(self) -> None:
            nonlocal finish_calls
            finish_calls += 1

    class FakeAccessor:
        def __init__(self, source: object, topic: str) -> None:
            del source, topic

        @contextmanager
        def open_reader(self) -> Iterator[object]:
            yield object()

        def iter_messages(
            self,
            reader: object,
            start_ns: int,
            end_ns_exclusive: int,
            *,
            log_time_order: bool = True,
        ) -> Iterator[tuple[Schema | None, Channel, Message]]:
            del reader, start_ns, end_ns_exclusive, log_time_order
            yield from messages

    monkeypatch.setattr(mcap_camera_sensor, "McapTopicAccessor", FakeAccessor)
    monkeypatch.setattr(mcap_camera_sensor, "_ForwardCompressedVideoDecoder", FakeDecoder)
    sensor = McapCameraSensor(b"not-used", topic=_TOPIC)
    sensor._video_metadata = _make_metadata(width=2, height=2)
    spec = SimpleNamespace(
        grid=iter(
            [
                SamplingWindow(start_ns=0, exclusive_end_ns=1, timestamps_ns=np.array([0], dtype=np.int64)),
                SamplingWindow(start_ns=1, exclusive_end_ns=2, timestamps_ns=np.array([1], dtype=np.int64)),
            ]
        )
    )

    generator = sensor.sample(spec, policy=_policy())  # type: ignore[arg-type]
    next(generator)
    generator.close()

    assert finish_calls == 1


def test_mcap_camera_sensor_stream_timestamps_not_implemented() -> None:
    """stream_timestamps is camera-only; McapCameraSensor raises NotImplementedError."""
    sensor = McapCameraSensor(b"not-used")

    with pytest.raises(NotImplementedError, match="CameraSensor"):
        list(sensor.stream_timestamps())
