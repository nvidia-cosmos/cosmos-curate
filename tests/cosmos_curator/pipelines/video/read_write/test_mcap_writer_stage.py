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
"""Tests for the MCAP writer stage and fragment consolidation."""

import base64
import functools
import io
import json
import uuid
from pathlib import Path

import av
import numpy as np
import numpy.testing as npt
import pytest
from mcap.reader import make_reader
from mcap.records import Channel, Message, Metadata, Schema

from cosmos_curator.pipelines.video.read_write import mcap_schemas, mcap_writer_stage
from cosmos_curator.pipelines.video.read_write.mcap_writer_stage import (
    McapWriterStage,
    consolidate_mcap_fragments,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VideoMetadata,
    Window,
)
from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND

NUM_FRAMES = 10
FPS = 30
CLIP_DURATION_S = NUM_FRAMES / FPS

# A clip long enough for the muxer to interleave audio and video the way a real capture
# does: the audio blocks a container emits between two video packets do not line up with
# the running sample clock the writer timestamps them from, so writing in demux order
# yields out-of-order log times. Ten frames are too few to hit that.
LONG_NUM_FRAMES = 60
LONG_NUM_AUDIO_FRAMES = 94
LONG_CLIP_DURATION_S = LONG_NUM_FRAMES / FPS


@functools.lru_cache
def _make_mp4(*, with_audio: bool = False, num_frames: int = NUM_FRAMES, num_audio_frames: int = 16) -> bytes:
    """Encode a tiny synthetic h264 mp4 (optionally with an aac audio track) in memory."""
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=FPS)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    audio_stream = None
    if with_audio:
        audio_stream = container.add_stream("aac", rate=48000)
        audio_stream.layout = "mono"

    for i in range(num_frames):
        array = np.full((stream.height, stream.width, 3), (i * 10) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    if audio_stream is not None:
        for i in range(num_audio_frames):
            audio_frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
            for plane in audio_frame.planes:
                plane.update(b"\x00" * plane.buffer_size)
            audio_frame.sample_rate = 48000
            audio_frame.pts = i * 1024
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
        for packet in audio_stream.encode(None):
            container.mux(packet)

    container.close()
    return buffer.getvalue()


def _make_long_av_mp4() -> bytes:
    return _make_mp4(
        with_audio=True,
        num_frames=LONG_NUM_FRAMES,
        num_audio_frames=LONG_NUM_AUDIO_FRAMES,
    )


def _make_clip(
    tmp_path: Path,
    video_path: Path,
    *,
    mp4_bytes: bytes | None,
    start_s: float = 0.0,
    with_annotations: bool = True,
    num_frames: int = NUM_FRAMES,
) -> Clip:
    """Build a post-ClipWriterStage clip, writing its mp4 to the clips/ output the stage reads."""
    duration_s = num_frames / FPS
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(start_s, start_s + duration_s),
        windows=[Window(start_frame=0, end_frame=num_frames - 1, caption={"qwen": "a test scene"})]
        if with_annotations
        else [],
    )
    if mp4_bytes is not None:
        clip_file = tmp_path / "output" / "clips" / f"{clip.uuid}.mp4"
        clip_file.parent.mkdir(parents=True, exist_ok=True)
        clip_file.write_bytes(mp4_bytes)
    clip.pts_ns = (np.arange(num_frames) * (NS_PER_SECOND // FPS)).astype(np.int64)
    clip.start_ns = round(start_s * NS_PER_SECOND)
    clip.end_ns = clip.start_ns + round(duration_s * NS_PER_SECOND)
    if with_annotations:
        clip.intern_video_2_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        clip.sam3_frames = [
            {"frame_idx": 0, "timestamp_s": 0.0, "detections": [{"prompt": "car", "object_id": 1}]},
            {"frame_idx": 5, "timestamp_s": 5 / FPS, "detections": []},
        ]
    return clip


def _make_video(
    video_path: Path,
    clips: list[Clip],
    *,
    clip_chunk_index: int = 0,
    num_clip_chunks: int = 1,
) -> Video:
    return Video(
        input_video=video_path,
        metadata=VideoMetadata(
            height=16,
            width=16,
            framerate=float(FPS),
            num_frames=NUM_FRAMES,
            duration=CLIP_DURATION_S,
            video_codec="h264",
            pixel_format="yuv420p",
        ),
        clips=clips,
        filtered_clips=[],
        num_total_clips=len(clips),
        num_clip_chunks=num_clip_chunks,
        clip_chunk_index=clip_chunk_index,
    )


def _process(tmp_path: Path, *videos: Video, capture_timezone: str = "UTC") -> None:
    """Run a fresh stage over one task holding *videos* (first video is primary)."""
    stage = McapWriterStage(
        output_path=str(tmp_path / "output"),
        input_path=str(tmp_path / "input"),
        output_s3_profile_name="default",
        embedding_algorithm="internvideo2",
        embedding_model_version="v1",
        caption_models=["qwen"],
        capture_timezone=capture_timezone,
    )
    stage.stage_setup()
    stage.process_data([SplitPipeTask(session_id="test-session", videos=list(videos))])


def _read_mcap(path: Path) -> tuple[list[tuple[Schema | None, Channel, Message]], list[Metadata]]:
    with path.open("rb") as fh:
        reader = make_reader(fh)
        messages = list(reader.iter_messages())
        metadata_records = list(reader.iter_metadata())
    return messages, metadata_records


def _assert_log_time_sorted(path: Path) -> int:
    """Assert the file's messages are in non-decreasing log_time order, as `mcap doctor` does.

    Read with ``log_time_order=False``: the reader's default re-sorts via the message
    index and would hide exactly the defect this checks.
    """
    with path.open("rb") as fh:
        log_times = [message.log_time for _, _, message in make_reader(fh).iter_messages(log_time_order=False)]
    out_of_order = [
        (index, log_times[index - 1], log_times[index])
        for index in range(1, len(log_times))
        if log_times[index] < log_times[index - 1]
    ]
    assert not out_of_order, f"{len(out_of_order)} of {len(log_times)} messages out of order: {out_of_order[:5]}"
    return len(log_times)


def _session_metadata(path: Path) -> dict[str, str]:
    records = [record for record in _read_mcap(path)[1] if record.name == mcap_schemas.SESSION_METADATA_RECORD_NAME]
    assert len(records) == 1
    return dict(records[0].metadata)


def _messages_by_topic(
    messages: list[tuple[Schema | None, Channel, Message]],
) -> dict[str, list[tuple[Schema | None, Message]]]:
    by_topic: dict[str, list[tuple[Schema | None, Message]]] = {}
    for schema, channel, message in messages:
        by_topic.setdefault(channel.topic, []).append((schema, message))
    return by_topic


def _fragment_path(tmp_path: Path, video_path: Path, chunk_index: int) -> Path:
    relative = video_path.relative_to(tmp_path / "input")
    return tmp_path / "output" / "mcap_fragments" / relative / f"{chunk_index}.mcap"


def test_fragment_writes_expected_channels(tmp_path: Path) -> None:
    """Chunk-0 fragment carries media, annotations, embedding, and one-shot session records."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4())
    video = _make_video(video_path, [clip])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    fragment = _fragment_path(tmp_path, video_path, 0)
    assert fragment.is_file()
    messages, metadata_records = _read_mcap(fragment)
    by_topic = _messages_by_topic(messages)

    assert set(by_topic) == {
        mcap_schemas.TOPIC_IMAGE_RAW,
        mcap_schemas.TOPIC_SCENE_ANNOTATION,
        mcap_schemas.TOPIC_CAMERA_INFO,
        mcap_schemas.TOPIC_TF_STATIC,
        mcap_schemas.TOPIC_CLIP_EMBEDDING,
    }

    # Video frames: one CompressedVideo message per frame, Annex-B bitstream, h264.
    frames = by_topic[mcap_schemas.TOPIC_IMAGE_RAW]
    assert len(frames) == NUM_FRAMES
    for schema, _ in frames:
        assert schema is not None
        assert schema.name == mcap_schemas.COMPRESSED_VIDEO_SCHEMA_NAME
        assert schema.encoding == mcap_schemas.JSONSCHEMA_ENCODING
    first_payload = json.loads(frames[0][1].data)
    assert first_payload["format"] == "h264"
    assert first_payload["frame_id"] == "camera"
    bitstream = base64.b64decode(first_payload["data"])
    assert bitstream.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01"))
    assert min(message.log_time for _, message in frames) == clip.start_ns

    # Annotations: one caption window plus two SAM3 frame payloads, sharing the topic.
    annotations = by_topic[mcap_schemas.TOPIC_SCENE_ANNOTATION]
    assert len(annotations) == 3
    annotation_data = [json.loads(message.data)["data"] for _, message in annotations]
    caption_index = annotation_data.index("a test scene")
    assert annotations[caption_index][1].log_time == clip.start_ns
    detection_payloads = [json.loads(data) for i, data in enumerate(annotation_data) if i != caption_index]
    assert detection_payloads[0]["detections"] == [{"prompt": "car", "object_id": 1}]
    assert detection_payloads[1]["detections"] == []

    # Embedding round-trip.
    embeddings = by_topic[mcap_schemas.TOPIC_CLIP_EMBEDDING]
    assert len(embeddings) == 1
    embedding_payload = json.loads(embeddings[0][1].data)
    assert embedding_payload["model_name"] == "internvideo2"
    assert embedding_payload["model_version"] == "v1"
    decoded = np.frombuffer(base64.b64decode(embedding_payload["data"]), dtype="<f4")
    npt.assert_allclose(decoded, [0.1, 0.2, 0.3])

    # Session metadata record.
    assert len(metadata_records) == 1
    session_metadata = metadata_records[0]
    assert session_metadata.name == mcap_schemas.SESSION_METADATA_RECORD_NAME
    assert session_metadata.metadata["source-video"] == video_path.as_posix()
    assert session_metadata.metadata["num-clip-chunks"] == "1"

    # Retained annotations dropped after the stage.
    assert clip.intern_video_2_embedding is None
    assert clip.windows[0].caption == {}


def test_fragment_audio_channel(tmp_path: Path) -> None:
    """A clip with an audio track adds pcm-s16 RawAudio messages; one without does not."""
    video_path = tmp_path / "input" / "video.mp4"
    video = _make_video(video_path, [_make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(with_audio=True))])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    messages, _ = _read_mcap(_fragment_path(tmp_path, video_path, 0))
    audio = _messages_by_topic(messages)[mcap_schemas.TOPIC_AUDIO]
    assert audio
    payload = json.loads(audio[0][1].data)
    assert payload["format"] == "pcm-s16"
    assert payload["sample_rate"] == 48000
    assert payload["number_of_channels"] == 1
    assert base64.b64decode(payload["data"])


def test_chunk1_fragment_omits_session_records(tmp_path: Path) -> None:
    """Only chunk 0 writes session metadata, camera-info, and tf-static."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)
    video = _make_video(video_path, [clip], clip_chunk_index=1, num_clip_chunks=2)

    _process(tmp_path, video)

    messages, metadata_records = _read_mcap(_fragment_path(tmp_path, video_path, 1))
    by_topic = _messages_by_topic(messages)
    assert not metadata_records
    assert mcap_schemas.TOPIC_CAMERA_INFO not in by_topic
    assert mcap_schemas.TOPIC_TF_STATIC not in by_topic
    assert len(by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == NUM_FRAMES
    # Frames are logged at source-timeline offsets: chunk 1 starts one clip in.
    assert min(m.log_time for _, m in by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == clip.start_ns


def test_zero_clip_chunks(tmp_path: Path) -> None:
    """A zero-clip chunk 0 writes a metadata-only fragment; other empty chunks write nothing."""
    video_path = tmp_path / "input" / "video.mp4"

    empty_chunk1 = _make_video(video_path, [], clip_chunk_index=1, num_clip_chunks=2)
    _process(tmp_path, empty_chunk1)
    assert not _fragment_path(tmp_path, video_path, 1).exists()

    empty_chunk0 = _make_video(video_path, [], clip_chunk_index=0, num_clip_chunks=2)
    _process(tmp_path, empty_chunk0)
    fragment = _fragment_path(tmp_path, video_path, 0)
    assert fragment.is_file()
    messages, metadata_records = _read_mcap(fragment)
    assert len(metadata_records) == 1
    by_topic = _messages_by_topic(messages)
    assert set(by_topic) == {mcap_schemas.TOPIC_CAMERA_INFO, mcap_schemas.TOPIC_TF_STATIC}


def test_span_fallback_when_pts_missing(tmp_path: Path) -> None:
    """Clips without decoded timestamps fall back to span seconds for the base offset."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(), start_s=1.0)
    clip.pts_ns = None
    clip.start_ns = None
    clip.end_ns = None
    video = _make_video(video_path, [clip])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    messages, _ = _read_mcap(_fragment_path(tmp_path, video_path, 0))
    frames = _messages_by_topic(messages)[mcap_schemas.TOPIC_IMAGE_RAW]
    assert len(frames) == NUM_FRAMES
    assert min(m.log_time for _, m in frames) == NS_PER_SECOND


def test_secondary_video_keeps_sam3_annotations(tmp_path: Path) -> None:
    """Secondary cameras keep their per-camera SAM3 detections; absent data yields no channel."""
    primary_path = tmp_path / "input" / "cam0.mp4"
    secondary_path = tmp_path / "input" / "cam1.mp4"
    primary = _make_video(primary_path, [_make_clip(tmp_path, primary_path, mp4_bytes=_make_mp4())])
    # Secondary cameras carry SAM3 detections but no captions/embeddings.
    secondary_clip = _make_clip(tmp_path, secondary_path, mp4_bytes=_make_mp4(), with_annotations=False)
    secondary_clip.sam3_frames = [
        {"frame_idx": 2, "timestamp_s": 2 / FPS, "detections": [{"prompt": "truck", "object_id": 7}]},
    ]
    secondary = _make_video(secondary_path, [secondary_clip])

    _process(tmp_path, primary, secondary)

    messages, _ = _read_mcap(_fragment_path(tmp_path, secondary_path, 0))
    by_topic = _messages_by_topic(messages)
    annotations = by_topic[mcap_schemas.TOPIC_SCENE_ANNOTATION]
    assert len(annotations) == 1
    detections = json.loads(json.loads(annotations[0][1].data)["data"])["detections"]
    assert detections == [{"prompt": "truck", "object_id": 7}]
    # No embeddings/captions exist on the secondary, so that channel never appears.
    assert mcap_schemas.TOPIC_CLIP_EMBEDDING not in by_topic
    assert len(by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == NUM_FRAMES


def test_consolidate_merges_fragments(tmp_path: Path) -> None:
    """Two chunk fragments merge into one MCAP named after the input video's relative path."""
    video_path = tmp_path / "input" / "video.mp4"

    chunk0_clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4())
    chunk0 = _make_video(video_path, [chunk0_clip], clip_chunk_index=0, num_clip_chunks=2)
    _process(tmp_path, chunk0)

    chunk1_clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)
    chunk1 = _make_video(video_path, [chunk1_clip], clip_chunk_index=1, num_clip_chunks=2)
    _process(tmp_path, chunk1)

    fragment0 = _fragment_path(tmp_path, video_path, 0)
    fragment1 = _fragment_path(tmp_path, video_path, 1)
    fragment_messages = len(_read_mcap(fragment0)[0]) + len(_read_mcap(fragment1)[0])

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    final_path = tmp_path / "output" / "mcap" / "video.mp4.mcap"
    assert final_path.is_file()
    messages, metadata_records = _read_mcap(final_path)
    assert len(messages) == fragment_messages
    assert len(metadata_records) == 1

    # Exactly one channel per topic and one schema per name after id remapping.
    with final_path.open("rb") as fh:
        summary = make_reader(fh).get_summary()
    assert summary is not None
    assert len({c.topic for c in summary.channels.values()}) == len(summary.channels)
    assert len({s.name for s in summary.schemas.values()}) == len(summary.schemas)
    assert mcap_schemas.TOPIC_IMAGE_RAW in {c.topic for c in summary.channels.values()}

    # Merged timeline covers both chunks and fragments are removed.
    frames = [m for _, c, m in messages if c.topic == mcap_schemas.TOPIC_IMAGE_RAW]
    frame_times = [m.log_time for m in frames]
    assert min(frame_times) == chunk0_clip.start_ns
    assert max(frame_times) >= chunk1_clip.start_ns
    # Sequences are renumbered across the merge instead of restarting per chunk.
    assert sorted(m.sequence for m in frames) == list(range(len(frames)))
    assert not fragment0.exists()
    assert not fragment1.exists()


def test_consolidate_single_fragment(tmp_path: Path) -> None:
    """A single-fragment video is validated and rewritten to the final MCAP, fragment removed."""
    video_path = tmp_path / "input" / "video.mp4"
    _process(tmp_path, _make_video(video_path, [_make_clip(tmp_path, video_path, mp4_bytes=_make_mp4())]))

    fragment = _fragment_path(tmp_path, video_path, 0)
    fragment_messages, fragment_metadata = _read_mcap(fragment)

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    final_path = tmp_path / "output" / "mcap" / "video.mp4.mcap"
    messages, metadata_records = _read_mcap(final_path)
    assert len(messages) == len(fragment_messages)
    assert len(metadata_records) == len(fragment_metadata)
    assert not fragment.exists()


def test_consolidate_skips_incomplete_fragment_sets(tmp_path: Path) -> None:
    """Fragment sets missing chunks are preserved untouched instead of merged and deleted."""
    # Chunk 0 announces two chunks in its session metadata, but chunk 1 is missing.
    tail_missing_path = tmp_path / "input" / "tail_missing.mp4"
    chunk0 = _make_video(
        tail_missing_path,
        [_make_clip(tmp_path, tail_missing_path, mp4_bytes=_make_mp4())],
        clip_chunk_index=0,
        num_clip_chunks=2,
    )
    _process(tmp_path, chunk0)

    # A set without chunk 0 (leftover of an interrupted run).
    head_missing_path = tmp_path / "input" / "head_missing.mp4"
    chunk1 = _make_video(
        head_missing_path,
        [_make_clip(tmp_path, head_missing_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)],
        clip_chunk_index=1,
        num_clip_chunks=2,
    )
    _process(tmp_path, chunk1)

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    assert not (tmp_path / "output" / "mcap" / "tail_missing.mp4.mcap").exists()
    assert not (tmp_path / "output" / "mcap" / "head_missing.mp4.mcap").exists()
    assert _fragment_path(tmp_path, tail_missing_path, 0).is_file()
    assert _fragment_path(tmp_path, head_missing_path, 1).is_file()


def test_consolidate_raises_on_corrupt_fragment(tmp_path: Path) -> None:
    """A truncated/corrupt fragment fails consolidation loudly and is preserved for inspection."""
    corrupt = tmp_path / "output" / "mcap_fragments" / "video.mp4" / "0.mcap"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a valid mcap file")

    with pytest.raises(RuntimeError, match="MCAP consolidation failed"):
        consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    assert not (tmp_path / "output" / "mcap" / "video.mp4.mcap").exists()
    assert corrupt.is_file()


def test_write_failure_propagates_with_payloads_intact(tmp_path: Path) -> None:
    """A fragment-write failure raises (enabling Xenna run attempts) and keeps annotations."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=b"not-an-mp4")
    video = _make_video(video_path, [clip])

    with pytest.raises(av.FFmpegError):
        _process(tmp_path, video)

    # Annotations survive the failure so a retry can re-produce the fragment
    # (the clip mp4 itself is re-read from the clips/ output on retry).
    assert clip.windows[0].caption == {"qwen": "a test scene"}
    assert clip.intern_video_2_embedding is not None


def test_missing_clip_mp4_skips_media(tmp_path: Path) -> None:
    """A clip whose mp4 was never written still contributes annotations, without failing."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=None)
    video = _make_video(video_path, [clip])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    messages, _ = _read_mcap(_fragment_path(tmp_path, video_path, 0))
    by_topic = _messages_by_topic(messages)
    assert mcap_schemas.TOPIC_IMAGE_RAW not in by_topic
    assert len(by_topic[mcap_schemas.TOPIC_SCENE_ANNOTATION]) == 3
    assert len(by_topic[mcap_schemas.TOPIC_CLIP_EMBEDDING]) == 1


def test_consolidate_no_fragments_is_noop(tmp_path: Path) -> None:
    """Consolidation returns quietly when there is nothing to merge."""
    consolidate_mcap_fragments(str(tmp_path / "missing-output"), "default")
    assert not (tmp_path / "missing-output").exists()


def test_fragment_is_log_time_sorted(tmp_path: Path) -> None:
    """A fragment is written in non-decreasing log_time order, annotations included.

    Regression: annotations and the embedding used to be appended after a clip's whole
    media stream while carrying timestamps back at the clip start, so every fragment with
    a caption was out of order.
    """
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4())
    _process(tmp_path, _make_video(video_path, [clip]))

    fragment = _fragment_path(tmp_path, video_path, 0)
    assert _assert_log_time_sorted(fragment) == NUM_FRAMES + 3 + 1 + 2  # frames, annotations, embedding, one-shots


def test_fragment_with_audio_and_video_is_log_time_sorted(tmp_path: Path) -> None:
    """A clip carrying both media channels comes out ordered across them."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_long_av_mp4(), num_frames=LONG_NUM_FRAMES)
    _process(tmp_path, _make_video(video_path, [clip]))

    fragment = _fragment_path(tmp_path, video_path, 0)
    _assert_log_time_sorted(fragment)

    # Both media channels really are present, so the assertion above spans them.
    by_topic = _messages_by_topic(_read_mcap(fragment)[0])
    assert len(by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == LONG_NUM_FRAMES
    assert len(by_topic[mcap_schemas.TOPIC_AUDIO]) > LONG_NUM_FRAMES


def test_fragment_orders_media_produced_out_of_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media messages are ordered by log_time however the demuxer hands them over.

    In production the skew comes from the container's own audio/video interleave, which the
    running audio sample clock does not track: a real 600 s capture had 11,621 video and
    audio messages written before earlier ones. A synthetic mp4 small enough for a unit test
    happens to mux in order, so the producer is stubbed to hand back the same messages
    shuffled -- what the writer must not preserve.
    """
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(with_audio=True))

    real_media_messages = mcap_writer_stage._clip_media_messages

    def shuffled(clip: Clip, base_ns: int, media: bytes) -> list[tuple[int, str, bytes]]:
        messages = real_media_messages(clip, base_ns, media)
        # Interleave the two halves so audio and video land far from their log times.
        midpoint = len(messages) // 2
        return [message for pair in zip(messages[midpoint:], messages[:midpoint], strict=False) for message in pair]

    monkeypatch.setattr(mcap_writer_stage, "_clip_media_messages", shuffled)
    _process(tmp_path, _make_video(video_path, [clip]))

    _assert_log_time_sorted(_fragment_path(tmp_path, video_path, 0))


def test_multi_clip_fragment_is_log_time_sorted(tmp_path: Path) -> None:
    """Several clips in one chunk stay ordered relative to each other."""
    video_path = tmp_path / "input" / "video.mp4"
    clips = [
        _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(with_audio=True), start_s=index * 2 * CLIP_DURATION_S)
        for index in range(3)
    ]
    _process(tmp_path, _make_video(video_path, clips))

    _assert_log_time_sorted(_fragment_path(tmp_path, video_path, 0))


def test_consolidated_file_is_log_time_sorted(tmp_path: Path) -> None:
    """The merged file is ordered even when chunk index order disagrees with time order."""
    video_path = tmp_path / "input" / "video.mp4"

    # Chunk 0 holds the *later* clip and chunk 1 the earlier one: concatenating fragments
    # in chunk order would produce an unsorted file, and overlapping chunk index entries.
    late_clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(with_audio=True), start_s=4 * CLIP_DURATION_S)
    _process(tmp_path, _make_video(video_path, [late_clip], clip_chunk_index=0, num_clip_chunks=2))
    early_clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(with_audio=True), start_s=0.0)
    _process(tmp_path, _make_video(video_path, [early_clip], clip_chunk_index=1, num_clip_chunks=2))

    fragment_messages = sum(len(_read_mcap(_fragment_path(tmp_path, video_path, index))[0]) for index in (0, 1))

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    final_path = tmp_path / "output" / "mcap" / "video.mp4.mcap"
    assert _assert_log_time_sorted(final_path) == fragment_messages


def test_capture_start_from_path_sets_absolute_log_times(tmp_path: Path) -> None:
    """A capture folder in the source path anchors the timeline to real UTC."""
    video_path = tmp_path / "input" / "2026-08-18-09-00" / "3.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(), start_s=0.5)
    _process(tmp_path, _make_video(video_path, [clip]), capture_timezone="America/Los_Angeles")

    # 2026-08-18 09:00 Pacific == 16:00Z, matching the reference 375mcap recordings.
    epoch_ns = 1787068800 * NS_PER_SECOND
    fragment = _fragment_path(tmp_path, video_path, 0)
    _assert_log_time_sorted(fragment)

    messages, _ = _read_mcap(fragment)
    by_topic = _messages_by_topic(messages)
    # The one-shot records sit at the video start; clip media at the clip's own offset.
    assert by_topic[mcap_schemas.TOPIC_TF_STATIC][0][1].log_time == epoch_ns
    frames = by_topic[mcap_schemas.TOPIC_IMAGE_RAW]
    assert min(message.log_time for _, message in frames) == epoch_ns + clip.start_ns

    metadata = _session_metadata(fragment)
    assert metadata["start-time-unix-ns"] == str(epoch_ns)
    assert metadata["start-time-source"] == "path:2026-08-18-09-00"
    assert metadata["start-time-timezone"] == "America/Los_Angeles"


def test_capture_start_falls_back_to_zero_based(tmp_path: Path) -> None:
    """A path naming no capture time keeps the 0-based timeline and says so in metadata."""
    video_path = tmp_path / "input" / "3PANEL.mp4"
    clip = _make_clip(tmp_path, video_path, mp4_bytes=_make_mp4(), start_s=0.5)
    _process(tmp_path, _make_video(video_path, [clip]), capture_timezone="America/Los_Angeles")

    fragment = _fragment_path(tmp_path, video_path, 0)
    metadata = _session_metadata(fragment)
    assert metadata["start-time-unix-ns"] == "0"
    assert metadata["start-time-source"] == "none"

    frames = _messages_by_topic(_read_mcap(fragment)[0])[mcap_schemas.TOPIC_IMAGE_RAW]
    assert min(message.log_time for _, message in frames) == clip.start_ns
