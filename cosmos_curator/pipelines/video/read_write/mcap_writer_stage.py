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
"""MCAP writer stage and post-pipeline consolidation for the split pipeline.

``McapWriterStage`` runs right after ``ClipWriterStage`` and writes one MCAP
*fragment* per (video, clip-chunk) to
``<output>/mcap_fragments/<relative_input_path>/<chunk_index>.mcap``. Because
``ClipTranscodingStage`` re-chunks tasks, no single stage invocation sees all
clips of one input video; ``consolidate_mcap_fragments`` runs on the driver
after the pipeline finishes and merges each video's fragments into one final
``<output>/mcap/<relative_input_path>.mcap``.

Messages carry the source video's capture start (parsed from its path, see
``mcap_time``) plus their offset on the source timeline, so gaps between clips
remain gaps in the MCAP. A video whose path names no capture time falls back to
a 0-based timeline.

Every file this module writes -- fragments and the merged result alike -- is
written in non-decreasing ``log_time`` order, which is what ``mcap doctor``
checks and what indexed readers need to seek without decompressing overlapping
chunks. Per-clip messages are therefore sorted before they are handed to the
writer, and consolidation merges fragments by ``log_time`` rather than
concatenating them. Note that video frames are not reordered relative to their
presentation times: Foxglove does not support B-frames in
``foxglove.CompressedVideo`` at all, so decode order and presentation order must
coincide, and this stage warns when a clip violates that.
"""

import functools
import heapq
import io
import json
import operator
import pathlib
import shutil
import tempfile
import uuid
import zoneinfo
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from importlib import metadata as importlib_metadata
from typing import IO, Any

import av
import smart_open  # type: ignore[import-untyped]
from av.bitstream import BitStreamFilterContext
from loguru import logger
from mcap.reader import McapReader, make_reader
from mcap.records import Channel, Message, Schema
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.sensors.utils.video import pts_to_ns
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.core.utils.storage.storage_utils import (
    StorageWriter,
    get_files_relative,
    get_full_path,
    read_bytes,
)
from cosmos_curator.pipelines.video.read_write import mcap_schemas, mcap_time
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import (
    ClipWriterStage,
    drop_clip_intermediate_data,
    select_clip_embedding,
    window_ns_bounds,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window
from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND, seconds_to_ns

MCAP_LIBRARY = "cosmos-curator split-pipeline mcap-writer"
DEFAULT_FRAME_ID = "camera"
DEFAULT_CAPTURE_TIMEZONE = "UTC"

# PyAV codec name -> foxglove.CompressedVideo ``format`` value. h264/hevc additionally
# need an AVCC -> Annex-B bitstream filter; av1/vp9 packets are already in the
# low-overhead form Foxglove expects.
_FOXGLOVE_VIDEO_FORMATS = {"h264": "h264", "hevc": "h265", "av1": "av1", "vp9": "vp9"}
_ANNEXB_BSF_NAMES = {"h264": "h264_mp4toannexb", "hevc": "hevc_mp4toannexb"}

# One pending MCAP message: (log_time_ns, topic, payload). Produced rather than written
# directly so a clip's messages can be ordered before they reach the writer.
_Msg = tuple[int, str, bytes]

_log_time_of = operator.itemgetter(0)


@functools.cache
def _curator_version() -> str:
    try:
        return importlib_metadata.version("cosmos_curator")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


@contextmanager
def _open_mcap_writer(out_file: IO[bytes]) -> Iterator[Writer]:
    """Yield a started zstd-chunked MCAP writer, finishing (sealing) it only on success.

    On error the file is deliberately left without a valid MCAP footer, so a
    partial write can never be mistaken for a complete file downstream.
    """
    writer = Writer(out_file, compression=CompressionType.ZSTD)
    writer.start(library=MCAP_LIBRARY)
    yield writer
    writer.finish()  # type: ignore[no-untyped-call]


class _McapChannels:
    """Lazily register channels on first message and track per-channel sequences.

    Topic -> schema is the fixed 1:1 map in ``mcap_schemas.TOPIC_SCHEMAS``.
    """

    def __init__(self, writer: Writer) -> None:
        self._writer = writer
        self._channel_ids: dict[str, int] = {}
        self._sequences: dict[str, int] = {}

    def add_message(self, topic: str, log_time_ns: int, payload: bytes) -> None:
        if topic not in self._channel_ids:
            schema = mcap_schemas.TOPIC_SCHEMAS[topic]
            schema_id = self._writer.register_schema(
                name=schema["title"],
                encoding=mcap_schemas.JSONSCHEMA_ENCODING,
                data=json.dumps(schema).encode("utf-8"),
            )
            self._channel_ids[topic] = self._writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding=mcap_schemas.JSON_MESSAGE_ENCODING,
            )
        sequence = self._sequences.get(topic, 0)
        self._sequences[topic] = sequence + 1
        self._writer.add_message(
            channel_id=self._channel_ids[topic],
            log_time=log_time_ns,
            data=payload,
            publish_time=log_time_ns,
            sequence=sequence,
        )


class McapWriterStage(CuratorStage):
    """Write one MCAP fragment per (video, clip-chunk) with clip media and annotations.

    Runs after ``ClipWriterStage`` (constructed with ``retain_clip_data=True`` so the
    small annotations — captions and embeddings — survive that stage's cleanup;
    ``build_output_stages`` wires this pairing). Clip mp4 bytes are deliberately NOT
    retained: this stage reads each clip back from the just-written ``clips/`` output
    (a page-cache hit locally), so the feature adds no clip payloads to the Ray
    object store and each worker holds at most one clip in memory at a time.
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        input_path: str,
        output_s3_profile_name: str,
        *,
        embedding_algorithm: str,
        embedding_model_version: str,
        caption_models: list[str],
        capture_timezone: str = DEFAULT_CAPTURE_TIMEZONE,
        dry_run: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Construct the MCAP fragment writer stage."""
        self._timer = StageTimer(self)
        self._output_path = output_path
        self._input_path = input_path.rstrip("/") + "/"
        self._output_s3_profile_name = output_s3_profile_name
        self._embedding_algorithm = embedding_algorithm
        self._embedding_model_version = embedding_model_version
        self._caption_models = caption_models
        # Held as a name rather than a tzinfo so the stage stays trivially serializable
        # for Xenna's remote actors; resolved once per worker in stage_setup.
        self._capture_timezone_name = capture_timezone
        self._dry_run = dry_run
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=0.5)

    def stage_setup(self) -> None:
        """Initialize the fragment storage writer, the clip read client, and the capture zone."""
        self._fragments_writer = StorageWriter(
            ClipWriterStage.get_output_path_mcap_fragments(self._output_path),
            profile_name=self._output_s3_profile_name,
        )
        self._storage_client = storage_utils.get_storage_client(
            self._output_path,
            profile_name=self._output_s3_profile_name,
        )
        self._capture_timezone = zoneinfo.ZoneInfo(self._capture_timezone_name)
        # One video yields several chunks, each parsed from the same path.
        self._capture_start_cache: dict[str, tuple[int, str] | None] = {}

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Write one MCAP fragment per video chunk, then drop retained clip payloads.

        Exceptions propagate (like ``ClipWriterStage``) so Xenna's run attempts can
        retry the task; the retained payloads are dropped only once every video's
        fragment is durably written, keeping retries reproducible.
        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process(len(video.clips)):
                    self._write_video_fragment(video)
            for video in task.videos:
                for clip in video.clips:
                    drop_clip_intermediate_data(clip)
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks

    def _relative_video_path(self, video: Video) -> str:
        input_video_path = video.input_path
        assert input_video_path.startswith(self._input_path)
        return input_video_path[len(self._input_path) :]

    def _capture_start(self, video: Video) -> tuple[int, str]:
        """Return the video's ``(epoch_base_ns, source_label)`` for the MCAP timeline.

        Falls back to a 0-based timeline (and warns once per video) when the source
        path names no capture time.
        """
        input_video_path = video.input_path
        if input_video_path not in self._capture_start_cache:
            parsed = mcap_time.parse_capture_start_ns(input_video_path, self._capture_timezone)
            if parsed is None:
                logger.warning(
                    f"No capture time in path {input_video_path}; MCAP log times will be "
                    "0-based source-video offsets rather than absolute"
                )
            self._capture_start_cache[input_video_path] = parsed
        parsed = self._capture_start_cache[input_video_path]
        if parsed is None:
            return 0, mcap_time.CAPTURE_START_SOURCE_NONE
        epoch_ns, matched = parsed
        return epoch_ns, f"path:{matched}"

    def _write_video_fragment(self, video: Video) -> None:
        # A fully filtered / zero-clip chunk 0 still writes a metadata-only fragment so
        # every processed video yields a final MCAP; other empty chunks write nothing.
        if not video.clips and video.clip_chunk_index != 0:
            return
        if self._dry_run:
            logger.info(f"Dry-run: skipping MCAP fragment for {video.input_path} chunk {video.clip_chunk_index}")
            return
        sub_path = f"{self._relative_video_path(video)}/{video.clip_chunk_index}.mcap"
        with self._fragments_writer.open_writer(sub_path, mode="wb") as out_file:
            self._write_chunk_mcap(out_file, video)
        if self._verbose:
            logger.info(f"Wrote MCAP fragment {sub_path} for {video.input_path}")

    def _write_chunk_mcap(self, out_file: IO[bytes], video: Video) -> None:
        """Write one video chunk's clips, annotations, and embeddings as a complete MCAP file.

        Annotations and embeddings are written for every camera video: fragment paths
        are per-camera so nothing can collide, secondary cameras simply lack captions
        and embeddings (lazy channel registration omits those channels), and SAM3
        detections genuinely exist per camera.
        """
        epoch_base_ns, start_source = self._capture_start(video)
        with _open_mcap_writer(out_file) as writer:
            channels = _McapChannels(writer)
            if video.clip_chunk_index == 0:
                self._write_session_start(writer, channels, video, epoch_base_ns, start_source)
            # Clips already arrive in ascending source-timeline order (``chunk_video``
            # slices ``video.clips`` contiguously), and they do not overlap, so ordering
            # each clip's own messages is enough to order the whole fragment.
            for clip in sorted(video.clips, key=_clip_base_ns):
                base_ns = epoch_base_ns + _clip_base_ns(clip)
                messages: list[_Msg] = []
                media = self._read_clip_mp4(clip, video.relative_path)
                if media is not None:
                    messages.extend(_clip_media_messages(clip, base_ns, media))
                messages.extend(self._clip_annotation_messages(clip, base_ns))
                messages.extend(self._clip_embedding_messages(clip, base_ns))
                # A stable sort keeps production order for equal log times, so a keyframe
                # carrying SPS/PPS stays ahead of anything sharing its timestamp.
                messages.sort(key=_log_time_of)
                for log_time_ns, topic, payload in messages:
                    channels.add_message(topic, log_time_ns, payload)

    def _read_clip_mp4(self, clip: Clip, relative_path: str) -> bytes | None:
        """Read one clip's mp4 back from the ``clips/`` output written by ClipWriterStage."""
        clip_uri = ClipWriterStage.get_clip_mp4_uri(self._output_path, clip.uuid, relative_path)
        try:
            return read_bytes(clip_uri, self._storage_client)
        except FileNotFoundError:
            # Mirrors the pre-existing "clip has no data" path: the writer already
            # logged why the mp4 is missing; the MCAP keeps annotations only.
            logger.warning(f"Clip {clip.uuid} from {clip.source_video} has no written mp4; skipping MCAP media")
            return None

    def _session_metadata(self, video: Video, epoch_base_ns: int, start_source: str) -> dict[str, str]:
        meta = video.metadata
        values: dict[str, Any] = {
            "source-video": video.input_path,
            "video-uuid": str(ClipWriterStage.get_video_uuid(video.input_path)),
            "width": meta.width,
            "height": meta.height,
            "framerate": meta.framerate,
            "num-frames": meta.num_frames,
            "duration-s": meta.duration,
            "video-codec": meta.video_codec,
            "pixel-format": meta.pixel_format,
            "audio-codec": meta.audio_codec,
            "num-total-clips": video.num_total_clips,
            "num-clip-chunks": video.num_clip_chunks,
            "embedding-algorithm": self._embedding_algorithm,
            "embedding-model-version": self._embedding_model_version,
            "curator-version": _curator_version(),
            # Where the MCAP timeline's zero sits, and how it was established, so a
            # reader can tell an absolute recording from a 0-based fallback.
            "start-time-unix-ns": epoch_base_ns,
            "start-time-source": start_source,
            "start-time-timezone": self._capture_timezone_name,
        }
        return {key: str(value) for key, value in values.items() if value is not None}

    def _write_session_start(
        self,
        writer: Writer,
        channels: _McapChannels,
        video: Video,
        epoch_base_ns: int,
        start_source: str,
    ) -> None:
        """Write the one-shot records: session metadata, camera calibration, static transform.

        Emitted at the source video's start (its capture time, or 0 when the path names
        none), from chunk 0 only so the merged file carries them exactly once. Being the
        earliest instant on the timeline, they also sort ahead of every clip message.
        """
        writer.add_metadata(
            mcap_schemas.SESSION_METADATA_RECORD_NAME,
            self._session_metadata(video, epoch_base_ns, start_source),
        )
        if video.metadata.width and video.metadata.height:
            channels.add_message(
                mcap_schemas.TOPIC_CAMERA_INFO,
                epoch_base_ns,
                mcap_schemas.camera_calibration_message(
                    epoch_base_ns, DEFAULT_FRAME_ID, video.metadata.width, video.metadata.height
                ),
            )
        channels.add_message(
            mcap_schemas.TOPIC_TF_STATIC,
            epoch_base_ns,
            mcap_schemas.frame_transforms_message(epoch_base_ns, DEFAULT_FRAME_ID),
        )

    def _clip_annotation_messages(self, clip: Clip, base_ns: int) -> list[_Msg]:
        """Build window captions and SAM3 per-frame detections for ``/scene-annotation``.

        Mirrors the reference recordings, where plain-text scene descriptions and
        JSON-encoded detection payloads share one topic.
        """
        messages: list[_Msg] = []
        for window in clip.windows:
            caption = _select_window_caption(window, self._caption_models)
            if caption is None:
                continue
            window_start_ns, _ = window_ns_bounds(clip, window)
            log_time = base_ns + (window_start_ns if window_start_ns is not None else 0)
            messages.append(
                (
                    log_time,
                    mcap_schemas.TOPIC_SCENE_ANNOTATION,
                    mcap_schemas.scene_annotation_message(log_time, caption),
                )
            )
        for entry in clip.sam3_frames or []:
            timestamp_s = entry.get("timestamp_s")
            offset_ns = seconds_to_ns(float(timestamp_s)) if timestamp_s is not None else 0
            log_time = base_ns + offset_ns
            payload = json.dumps({"frame_idx": entry.get("frame_idx"), "detections": entry.get("detections", [])})
            messages.append(
                (
                    log_time,
                    mcap_schemas.TOPIC_SCENE_ANNOTATION,
                    mcap_schemas.scene_annotation_message(log_time, payload),
                )
            )
        return messages

    def _clip_embedding_messages(self, clip: Clip, base_ns: int) -> list[_Msg]:
        embedding = select_clip_embedding(clip, self._embedding_algorithm)
        if embedding is None:
            return []
        return [
            (
                base_ns,
                mcap_schemas.TOPIC_CLIP_EMBEDDING,
                mcap_schemas.clip_embedding_message(
                    base_ns, self._embedding_algorithm, self._embedding_model_version, embedding
                ),
            )
        ]


def _clip_base_ns(clip: Clip) -> int:
    """Source-timeline start of the clip in ns (span-seconds fallback for errored clips)."""
    if clip.start_ns is not None:
        return clip.start_ns
    return seconds_to_ns(clip.span[0])


def _select_window_caption(window: Window, caption_models: list[str]) -> str | None:
    for model in caption_models:
        if model in window.caption:
            return window.caption[model]
    return None


def _clip_media_messages(clip: Clip, base_ns: int, media: bytes) -> list[_Msg]:
    """Demux one clip mp4 and build its video-packet and decoded-audio messages."""
    messages: list[_Msg] = []
    with av.open(io.BytesIO(media), mode="r") as container:
        maybe_encoders = (
            _ClipVideoEncoder.create(container, clip, base_ns),
            _ClipAudioEncoder.create(container, base_ns),
        )
        encoders = {e.stream_index: e for e in maybe_encoders if e is not None}
        if not encoders:
            return messages
        for packet in container.demux():
            if packet.dts is None:  # demux flush sentinel
                continue
            if (encoder := encoders.get(packet.stream_index)) is not None:
                messages.extend(encoder.write_packet(packet))
        for encoder in encoders.values():
            messages.extend(encoder.flush())
    return messages


class _ClipVideoEncoder:
    """Build ``foxglove.CompressedVideo`` messages from a clip's compressed video packets.

    h264/hevc packets are converted from AVCC (length-prefixed NALs, mp4) to
    Annex-B (start codes, SPS/PPS inline) via the ``*_mp4toannexb`` bitstream
    filter, as required by the CompressedVideo spec.
    """

    def __init__(
        self,
        clip_uuid: uuid.UUID,
        base_ns: int,
        *,
        video_format: str,
        stream_index: int,
        bsf: BitStreamFilterContext | None,
    ) -> None:
        self._clip_uuid = clip_uuid
        self._base_ns = base_ns
        self._video_format = video_format
        self.stream_index = stream_index
        self._bsf = bsf
        # Seeded from the first demuxed packet (the IDR frame): its PTS is the clip's zero point.
        self._first_pts_ns: int | None = None
        self._warned_missing_pts = False
        self._reordered_packets = 0

    @classmethod
    def create(
        cls,
        container: "av.container.InputContainer",
        clip: Clip,
        base_ns: int,
    ) -> "_ClipVideoEncoder | None":
        if not container.streams.video:
            logger.warning(f"Clip {clip.uuid} from {clip.source_video} has no video stream")
            return None
        stream = container.streams.video[0]
        codec_name = stream.codec_context.name
        video_format = _FOXGLOVE_VIDEO_FORMATS.get(codec_name)
        if video_format is None:
            logger.warning(
                f"Clip {clip.uuid} from {clip.source_video} uses codec {codec_name!r}, which "
                "foxglove.CompressedVideo does not support; skipping video frames"
            )
            return None
        bsf_name = _ANNEXB_BSF_NAMES.get(codec_name)
        bsf = BitStreamFilterContext(bsf_name, stream) if bsf_name is not None else None
        return cls(
            clip.uuid,
            base_ns,
            video_format=video_format,
            stream_index=stream.index,
            bsf=bsf,
        )

    def write_packet(self, packet: "av.Packet[Any]") -> list[_Msg]:
        if packet.pts is not None and packet.dts is not None and packet.pts != packet.dts:
            self._reordered_packets += 1
        out_packets = self._bsf.filter(packet) if self._bsf is not None else [packet]
        return [message for out_packet in out_packets if (message := self._packet_message(out_packet)) is not None]

    def flush(self) -> list[_Msg]:
        messages: list[_Msg] = []
        if self._bsf is not None:
            messages = [
                message
                for out_packet in self._bsf.filter(None)
                if (message := self._packet_message(out_packet)) is not None
            ]
        if self._reordered_packets:
            logger.warning(
                f"Clip {self._clip_uuid} has {self._reordered_packets} reordered (B-frame) video "
                "packets; Foxglove does not support B-frames in foxglove.CompressedVideo, so this "
                "clip's frames will not decode there. Re-encode without B-frames (ffmpeg -bf 0)."
            )
        return messages

    def _packet_message(self, packet: "av.Packet[Any]") -> _Msg | None:
        if packet.pts is None or packet.time_base is None:
            if not self._warned_missing_pts:
                self._warned_missing_pts = True
                logger.warning(f"Clip {self._clip_uuid} has video packets without PTS; skipping those frames")
            return None
        packet_pts_ns = pts_to_ns(packet.pts, packet.time_base)
        if self._first_pts_ns is None:
            self._first_pts_ns = packet_pts_ns
        log_time = self._base_ns + (packet_pts_ns - self._first_pts_ns)
        return (
            log_time,
            mcap_schemas.TOPIC_IMAGE_RAW,
            mcap_schemas.compressed_video_message(log_time, DEFAULT_FRAME_ID, packet, self._video_format),
        )


class _ClipAudioEncoder:
    """Decode a clip's audio track into pcm-s16 ``foxglove.RawAudio`` messages."""

    def __init__(self, stream: "av.audio.stream.AudioStream", base_ns: int) -> None:
        self._stream = stream
        self.stream_index = stream.index
        self._base_ns = base_ns
        self._resampler: av.AudioResampler | None = None
        self._elapsed_samples = 0

    @classmethod
    def create(
        cls,
        container: "av.container.InputContainer",
        base_ns: int,
    ) -> "_ClipAudioEncoder | None":
        if not container.streams.audio:
            return None
        return cls(container.streams.audio[0], base_ns)

    def write_packet(self, packet: "av.Packet[Any]") -> list[_Msg]:
        messages: list[_Msg] = []
        for frame in self._stream.codec_context.decode(packet):
            messages.extend(self._frame_messages(frame))
        return messages

    def flush(self) -> list[_Msg]:
        messages: list[_Msg] = []
        for frame in self._stream.codec_context.decode(None):
            messages.extend(self._frame_messages(frame))
        if self._resampler is not None:
            messages.extend(
                message
                for resampled in self._resampler.resample(None)
                if (message := self._resampled_message(resampled)) is not None
            )
        return messages

    def _frame_messages(self, frame: "av.AudioFrame") -> list[_Msg]:
        if self._resampler is None:
            self._resampler = av.AudioResampler(format="s16", layout=frame.layout.name, rate=frame.sample_rate)
        return [
            message
            for resampled in self._resampler.resample(frame)
            if (message := self._resampled_message(resampled)) is not None
        ]

    def _resampled_message(self, frame: "av.AudioFrame") -> _Msg | None:
        data = frame.to_ndarray().tobytes()
        if not data:
            return None
        number_of_channels = len(frame.layout.channels)
        # Resampled PCM output is contiguous, so a running sample count is the single
        # source of truth; decoder PTS gaps or absences cannot rewind or overlap blocks.
        # Counting samples rather than accumulating per-block nanoseconds keeps the clock
        # exact instead of truncating a fraction of a nanosecond per block.
        log_time = self._base_ns + self._elapsed_samples * NS_PER_SECOND // frame.sample_rate
        self._elapsed_samples += frame.samples
        return (
            log_time,
            mcap_schemas.TOPIC_AUDIO,
            mcap_schemas.raw_audio_message(log_time, data, frame.sample_rate, number_of_channels),
        )


def consolidate_mcap_fragments(output_path: str, output_s3_profile_name: str) -> None:
    """Merge per-chunk MCAP fragments into one MCAP per input video and delete the fragments."""
    fragments_root = ClipWriterStage.get_output_path_mcap_fragments(output_path)
    client = storage_utils.get_storage_client(
        fragments_root,
        profile_name=output_s3_profile_name,
        can_overwrite=True,
        can_delete=True,
    )

    # Fragments live at <relative_input_path>/<chunk_index>.mcap; group per video.
    # No existence pre-check: a missing root simply lists as empty (a head_object on
    # a bare remote prefix would 404 even when fragments exist under it).
    fragments_by_video: dict[str, list[tuple[int, str]]] = {}
    for fname in get_files_relative(fragments_root, client):
        relative_path, _, chunk_name = fname.rpartition("/")
        if not fname.endswith(".mcap") or not relative_path:
            continue
        try:
            chunk_index = int(chunk_name.removesuffix(".mcap"))
        except ValueError:
            logger.warning(f"Unexpected MCAP fragment name {fname!r}; skipping")
            continue
        fragments_by_video.setdefault(relative_path, []).append((chunk_index, fname))
    if not fragments_by_video:
        return

    final_writer = StorageWriter(
        ClipWriterStage.get_output_path_mcap(output_path),
        profile_name=output_s3_profile_name,
    )

    consolidated = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _consolidate_video_fragments,
                relative_path,
                sorted(entries),
                fragments_root=fragments_root,
                client=client,
                final_writer=final_writer,
            ): relative_path
            for relative_path, entries in fragments_by_video.items()
        }
        for future, relative_path in futures.items():
            try:
                consolidated += 1 if future.result() else 0
            except Exception:  # noqa: BLE001 - collected and re-raised as one error below
                logger.exception(f"Failed to consolidate MCAP fragments for {relative_path}")
                failed.append(relative_path)
    logger.info(f"Consolidated MCAP fragments for {consolidated}/{len(fragments_by_video)} videos")
    if failed:
        msg = f"MCAP consolidation failed for {len(failed)} of {len(fragments_by_video)} videos: {failed[:10]}"
        raise RuntimeError(msg)


def _expected_chunk_count(chunk0_reader: McapReader) -> int | None:
    """Read the expected chunk count from the chunk-0 fragment's session metadata, if present."""
    for record in chunk0_reader.iter_metadata():
        if record.name == mcap_schemas.SESSION_METADATA_RECORD_NAME:
            value = record.metadata.get("num-clip-chunks")
            return int(value) if value is not None else None
    return None


def _consolidate_video_fragments(
    relative_path: str,
    entries: list[tuple[int, str]],
    *,
    fragments_root: str,
    client: "storage_client.StorageClient | None",
    final_writer: StorageWriter,
) -> bool:
    """Merge one video's fragments into its final MCAP; returns False for an incomplete set.

    Incomplete sets (missing chunk 0, gaps, or fewer chunks than the count recorded in
    chunk 0's session metadata — e.g. leftovers of an interrupted run) are skipped with
    their fragments preserved, so a future complete run can still consolidate them.
    Corrupt/truncated fragments raise before the final file is created.
    """
    chunk_indices = [chunk_index for chunk_index, _ in entries]
    fragment_uris = [get_full_path(fragments_root, fname) for _, fname in entries]

    with ExitStack() as stack:
        # Open (and for remote, download) every fragment once, validating each via the
        # MCAP reader up front so a truncated leftover can never ship as a final file.
        readers = [make_reader(stack.enter_context(_open_fragment(uri, client))) for uri in fragment_uris]  # type: ignore[no-untyped-call]
        for reader in readers:
            reader.get_summary()

        expected_chunks = _expected_chunk_count(readers[0]) if chunk_indices[0] == 0 else None
        complete = (
            chunk_indices == list(range(expected_chunks))
            if expected_chunks is not None
            else chunk_indices == list(range(len(chunk_indices)))
        )
        if not complete:
            logger.warning(
                f"Video {relative_path} has an incomplete MCAP fragment set (chunks {chunk_indices}, "
                f"expected {expected_chunks}); leaving fragments in place"
            )
            return False

        with final_writer.open_writer(f"{relative_path}.mcap", mode="wb") as out_file:
            _merge_fragments(out_file, list(zip(fragment_uris, readers, strict=True)))

    for fragment_uri in fragment_uris:
        if isinstance(fragment_uri, pathlib.Path):
            fragment_uri.unlink(missing_ok=True)
        elif client is not None:
            try:
                client.delete_object(fragment_uri)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to delete MCAP fragment {fragment_uri}: {exc}")
    return True


def _download_fragment_to(
    uri: "storage_client.StoragePrefix",
    client: "storage_client.StorageClient | None",
    out_file: IO[bytes],
) -> None:
    """Stream a remote fragment into *out_file* with constant memory, retrying transient errors."""
    client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}

    def _download() -> None:
        out_file.seek(0)
        out_file.truncate()
        with smart_open.open(str(uri), "rb", **client_params) as src:
            shutil.copyfileobj(src, out_file)

    # Same retry policy as storage_utils.read_bytes.
    do_with_retries(_download, max_attempts=5, backoff_factor=4.0, max_wait_time_s=256.0)


@contextmanager
def _open_fragment(
    uri: "storage_client.StoragePrefix | pathlib.Path",
    client: "storage_client.StorageClient | None",
) -> Iterator[IO[bytes]]:
    """Yield a seekable binary handle: the file itself locally, a temp-file download for remote."""
    if isinstance(uri, pathlib.Path):
        with uri.open("rb") as fh:
            yield fh
    else:
        with tempfile.TemporaryFile(suffix=".mcap") as tmp_file:
            _download_fragment_to(uri, client, tmp_file)
            tmp_file.seek(0)
            yield tmp_file


def _fragment_messages(
    fragment_uri: "storage_client.StoragePrefix | pathlib.Path",
    reader: McapReader,
) -> Iterator[tuple["storage_client.StoragePrefix | pathlib.Path", Schema | None, Channel, Message]]:
    """Stream one fragment's messages in file order, tagged with the fragment they came from."""
    for schema, channel, message in reader.iter_messages(log_time_order=False):
        yield fragment_uri, schema, channel, message


def _merge_fragments(
    out_file: IO[bytes],
    fragments: list[tuple["storage_client.StoragePrefix | pathlib.Path", McapReader]],
) -> None:
    """Rewrite the fragments' records into one MCAP, remapping schema/channel ids.

    Each fragment is already written in ``log_time`` order, so the fragments are
    k-way merged on ``log_time`` rather than concatenated: the merged file is then
    ordered whatever the relationship between chunk index and time, and an indexed
    reader never has to decompress overlapping chunks. ``heapq.merge`` is stable in
    argument order, so equal log times keep chunk order, and it pulls lazily -- only
    one message per fragment is ever held.
    """
    with _open_mcap_writer(out_file) as writer:
        schema_ids: dict[tuple[str, str, bytes], int] = {}
        channel_ids: dict[tuple[str, str, str], int] = {}
        # Fragments restart sequences at 0, so renumber per channel across the merge.
        sequences: dict[int, int] = {}
        metadata_seen: set[str] = set()
        for _, reader in fragments:
            for metadata_record in reader.iter_metadata():
                if metadata_record.name in metadata_seen:
                    continue
                metadata_seen.add(metadata_record.name)
                writer.add_metadata(metadata_record.name, dict(metadata_record.metadata))
        merged = heapq.merge(
            *(_fragment_messages(fragment_uri, reader) for fragment_uri, reader in fragments),
            key=lambda entry: entry[3].log_time,
        )
        for fragment_uri, schema, channel, message in merged:
            if schema is None:
                logger.warning(f"Fragment {fragment_uri} has a channel without schema; skipping its messages")
                continue
            schema_key = (schema.name, schema.encoding, schema.data)
            if schema_key not in schema_ids:
                schema_ids[schema_key] = writer.register_schema(
                    name=schema.name,
                    encoding=schema.encoding,
                    data=schema.data,
                )
            channel_key = (channel.topic, schema.name, channel.message_encoding)
            if channel_key not in channel_ids:
                channel_ids[channel_key] = writer.register_channel(
                    schema_id=schema_ids[schema_key],
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    metadata=dict(channel.metadata),
                )
            channel_id = channel_ids[channel_key]
            sequence = sequences.get(channel_id, 0)
            sequences[channel_id] = sequence + 1
            writer.add_message(
                channel_id=channel_id,
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
                sequence=sequence,
            )
