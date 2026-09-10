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

"""First-frame decode tests: happy path over a synthetic clip, drop paths, batch reads.

The batch entry point (``read_many``) is covered here rather than at the embedder,
because the reader owns the fan-out: the width, the pre-resolution of every backend
the batch names, and the mapping from a frame back to the row it came from are all
its contract. The embedder's own tests cover only what it does with that mapping.
"""

import io
import pathlib
import threading
from collections.abc import Callable
from typing import Any, NoReturn

import av
import botocore.exceptions
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.next.embeddings.image import frame_reader
from cosmos_curator.next.embeddings.image.frame_reader import ClipFrameReader, read_first_frame

_RGB_CHANNELS = 3
_HWC_NDIM = 3
# Ceiling on a stubbed read waiting for the row after it, so a regression that
# serializes a concurrent batch fails the assertion instead of hanging the suite.
_READ_BARRIER_TIMEOUT_S = 10.0
# The make_clip fixture ramps brightness per frame; the first displayable frame
# is the darkest, so a returned mean under this bound proves frame selection.
_FIRST_FRAME_MAX_MEAN = 30.0
# The fixture biases red over a flat blue channel; a wide margin survives lossy
# chroma while still catching an RGB/BGR channel swap.
_MIN_RED_BLUE_GAP = 20.0


def _libx264_or_skip() -> None:
    try:
        av.codec.Codec("libx264", "w")
    except Exception as exc:  # noqa: BLE001 - encoder availability is environment-dependent
        pytest.skip(f"libx264 encoder not available: {exc}")


def _decode_all_rgb(path: pathlib.Path) -> list[np.ndarray]:
    """Decode every frame of a clip to an (H, W, 3) uint8 RGB array, in display order."""
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


class _ForwardOnlyStream(io.BytesIO):
    """A transport stream that can be read but not seeked, as a raw HTTP body is.

    The sensor library refuses a non-seekable source, so a clip can only be decoded
    from a copy of these bytes. That makes the stream a probe for where the decoder's
    bytes come from: it yields a frame under a buffering transport and nothing under
    one that hands its own stream to the decoder.
    """

    def seekable(self) -> bool:
        """Report the stream as forward-only."""
        return False

    def seek(self, pos: int, whence: int = 0, /) -> int:
        """Refuse to seek, as an object-store body would."""
        del pos, whence
        msg = "this stream is forward-only"
        raise OSError(msg)


class _FailsMidTransferStream(io.BytesIO):
    """A stream that opens cleanly and then fails inside ``read``.

    Isolates the transfer from the open: the object was located, so the fault has to
    be classified on its own rather than inherited from a failed lookup.
    """

    def __init__(self, fail: Callable[[], NoReturn]) -> None:
        """Bind the failure raised when the transfer is attempted."""
        super().__init__(b"")
        self._fail = fail

    def read(self, size: int | None = -1, /) -> bytes:
        """Raise the bound failure instead of returning bytes."""
        del size
        self._fail()


def _raise_missing_object() -> NoReturn:
    """Raise the ``OSError`` shape ``smart_open`` uses for an absent S3 key."""
    client_error = botocore.exceptions.ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    msg = "unable to read the s3 object"
    raise OSError(msg) from client_error


def _raise_connectivity_fault() -> NoReturn:
    """Raise a mid-transfer connectivity fault - wrong for every clip, not just this one."""
    msg = "connection reset by peer"
    raise OSError(msg)


def _reader_over(monkeypatch: pytest.MonkeyPatch, stream_factory: Callable[[], io.BytesIO]) -> ClipFrameReader:
    """Return a reader whose transport opens ``stream_factory()`` for any URI.

    Param resolution is stubbed out as well, so a remote URI can be used without the
    environment holding that backend's credentials; what these tests exercise is the
    reader's behaviour once a stream exists.
    """

    def fake_open(_uri: str, _mode: str, **_kwargs: object) -> io.BytesIO:
        return stream_factory()

    monkeypatch.setattr(frame_reader, "get_smart_open_params", lambda *_a, **_k: {})
    monkeypatch.setattr(frame_reader.smart_open, "open", fake_open)
    return ClipFrameReader()


def _marker_frame(marker: int) -> npt.NDArray[np.uint8]:
    """Return a small RGB frame filled with ``marker``, so a frame names the URI it came from."""
    return np.full((4, 4, _RGB_CHANNELS), marker, dtype=np.uint8)


def _stub_single_reads(
    monkeypatch: pytest.MonkeyPatch,
    read: Callable[[str], npt.NDArray[np.uint8] | None],
) -> None:
    """Replace the per-URI read with ``read`` and resolve transport params to ``{}``.

    Lets a batch test state exactly what each URI yields - a frame, a drop, or an
    exception - without a stream, a codec, or a storage backend, which is what
    keeps these tests about the fan-out rather than about the decode.
    """
    monkeypatch.setattr(frame_reader, "get_smart_open_params", lambda *_a, **_k: {})
    monkeypatch.setattr(ClipFrameReader, "read", lambda _self, uri: read(uri))


def test_reads_first_displayable_frame(make_clip: Callable[..., None], tmp_path: pathlib.Path) -> None:
    """A valid clip decodes to a single (H, W, 3) uint8 RGB frame, and it is the first one.

    The mean bound is the discriminating part: without it, an implementation that
    returned any later frame -- or the first *decoded* frame of a reordered
    stream -- would pass on shape / dtype alone.
    """
    path = tmp_path / "clip.mp4"
    make_clip(path)
    with path.open("rb") as handle:
        frame = read_first_frame(handle)
    assert frame is not None
    assert frame.ndim == _HWC_NDIM
    assert frame.shape == (32, 32, _RGB_CHANNELS)
    assert frame.dtype == np.uint8
    assert frame.mean() < _FIRST_FRAME_MAX_MEAN


def test_reads_first_displayable_frame_from_b_frame_stream(
    make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """On a stream that contains B-frames, the returned frame is still display-order frame 0.

    The IDR that opens the stream is first in both decode and display order, so
    this does not isolate a decode-vs-display swap at position 0; it exercises the
    decode path over a reordered (B-frame) stream end to end and confirms frame
    selection returns the darkest (frame-0) frame rather than a later, brighter one.
    """
    _libx264_or_skip()
    path = tmp_path / "bframes.mp4"
    make_clip(path, frames=6, max_b_frames=2)
    with path.open("rb") as handle:
        frame = read_first_frame(handle)
    assert frame is not None
    assert frame.mean() < _FIRST_FRAME_MAX_MEAN


def test_make_clip_is_frame_and_channel_discriminating(make_clip: Callable[..., None], tmp_path: pathlib.Path) -> None:
    """The synthetic clip actually varies by frame and by channel.

    Pins the fixture's discriminating power directly, so the (non-run) env smoke
    tests that rely on it cannot silently regress to a uniform-grey clip that
    hides a frame-selection or channel-order bug.
    """
    path = tmp_path / "clip.mp4"
    make_clip(path, frames=4)
    frames = _decode_all_rgb(path)
    assert frames[0].mean() != frames[-1].mean()
    red_mean = float(np.mean([frame[..., 0].mean() for frame in frames]))
    blue_mean = float(np.mean([frame[..., 2].mean() for frame in frames]))
    assert red_mean - blue_mean > _MIN_RED_BLUE_GAP


def test_garbage_stream_returns_none() -> None:
    """A non-video byte stream is dropped (None), not raised."""
    assert read_first_frame(io.BytesIO(b"this is not a video")) is None


def test_empty_stream_returns_none() -> None:
    """An empty stream has no decodable frame."""
    assert read_first_frame(io.BytesIO(b"")) is None


def test_clip_reader_drops_when_transport_open_fails(tmp_path: pathlib.Path) -> None:
    """A clip whose transport open fails (missing file) is dropped (None), not raised."""
    missing = tmp_path / "missing.mp4"
    assert ClipFrameReader().read(str(missing)) is None


def test_clip_reader_propagates_systemic_open_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A systemic (non-missing-object) open failure fails the leg rather than dropping one row.

    An auth / backend / connectivity fault is wrong for every clip, so nulling one
    row and continuing would silently commit an all-NULL image column. Only an
    EXPECTED missing object is a per-row drop; everything else must surface.
    """

    def _boom(_uri: str, _mode: str, **_kwargs: object) -> object:
        msg = "connection reset by peer"
        raise OSError(msg)

    monkeypatch.setattr(frame_reader.smart_open, "open", _boom)
    with pytest.raises(OSError, match="connection reset"):
        ClipFrameReader().read(str(tmp_path / "clip.mp4"))


def test_clip_reader_drops_wrapped_missing_s3_object(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """An S3 NoSuchKey (which smart_open wraps in an OSError) is dropped, not raised.

    Pins the remote counterpart of the missing-file drop: a genuinely absent
    object surfaces as an ``OSError`` whose cause is a boto3 ``NoSuchKey``
    ``ClientError``, which ``is_missing_object_error`` classifies as a per-row drop.
    """
    client_error = botocore.exceptions.ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def _boom(_uri: str, _mode: str, **_kwargs: object) -> object:
        msg = "unable to open the s3 object"
        raise OSError(msg) from client_error

    monkeypatch.setattr(frame_reader.smart_open, "open", _boom)
    assert ClipFrameReader().read(str(tmp_path / "clip.mp4")) is None


def test_clip_reader_decodes_a_forward_only_transport_stream(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """A clip arriving on a non-seekable stream still decodes to a frame.

    The observable proof that the object is fetched into memory before decoding: an
    object store answers a seek with a fresh ranged request, and the decoder's index
    build seeks repeatedly, so the transport reads the bytes once and hands the
    decoder a buffer. A transport that passed its own stream through would be refused
    by the sensor library and drop this clip.
    """
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    payload = clip.read_bytes()
    reader = _reader_over(monkeypatch, lambda: _ForwardOnlyStream(payload))

    frame = reader.read("s3://bucket/clip.mp4")

    assert frame is not None
    assert frame.shape == (32, 32, _RGB_CHANNELS)


def test_clip_reader_drops_a_clip_that_vanishes_mid_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object that is found and then unreadable as missing is dropped, not raised.

    The transfer is classified by the same rule as the open, so a genuinely absent
    object costs its own row wherever in the transport it surfaces.
    """
    reader = _reader_over(monkeypatch, lambda: _FailsMidTransferStream(_raise_missing_object))

    assert reader.read("s3://bucket/clip.mp4") is None


def test_clip_reader_propagates_a_systemic_mid_transfer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connectivity fault during the transfer fails the leg rather than nulling one row.

    This is the classification the transfer gained by moving into the transport
    layer: inside the decode helper its broad handler would read a transient network
    fault as a corrupt clip and silently NULL the row, so a whole run could commit an
    empty column while exiting successfully.
    """
    reader = _reader_over(monkeypatch, lambda: _FailsMidTransferStream(_raise_connectivity_fault))

    with pytest.raises(OSError, match="connection reset"):
        reader.read("s3://bucket/clip.mp4")


def test_clip_reader_propagates_decode_programming_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A programming error raised inside read_first_frame surfaces, not swallowed as a drop.

    The transport open succeeds, so the only exception source is the decode call.
    read_first_frame owns its own per-clip drop path (it returns None for corrupt
    input), so anything it *raises* is a real bug that must fail the leg rather
    than be logged-and-dropped by ClipFrameReader.read.
    """
    readable = tmp_path / "clip.bin"
    readable.write_bytes(b"payload")

    def _boom(_stream: io.BufferedIOBase, *, source_label: str = "") -> np.ndarray:
        msg = f"malformed sampling grid for {source_label}"
        raise RuntimeError(msg)

    monkeypatch.setattr(frame_reader, "read_first_frame", _boom)
    with pytest.raises(RuntimeError, match="malformed sampling grid"):
        ClipFrameReader().read(str(readable))


def test_clip_reader_drops_a_clip_larger_than_the_cap(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """A clip whose object exceeds the cap is dropped like any other unreadable clip.

    Without the cap one mis-populated ``clip_uri`` naming an uncut chunk exhausts
    the actor's heap, and an OOM-killed actor costs the whole fragment - not the
    single row every other read failure costs. The cap is patched down rather than
    fed a real oversized object, so the test pins the mechanism, not the figure.
    """
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    payload = clip.read_bytes()
    monkeypatch.setattr(ClipFrameReader, "MAX_CLIP_BYTES", len(payload) - 1)
    reader = _reader_over(monkeypatch, lambda: io.BytesIO(payload))

    assert reader.read("s3://bucket/clip.mp4") is None


def test_clip_reader_accepts_a_clip_exactly_at_the_cap(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """The cap is inclusive: an object exactly at the limit still decodes.

    The read asks for one byte past the cap to detect an overrun, so an
    off-by-one there would drop every clip of exactly the permitted size.
    """
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    payload = clip.read_bytes()
    monkeypatch.setattr(ClipFrameReader, "MAX_CLIP_BYTES", len(payload))
    reader = _reader_over(monkeypatch, lambda: io.BytesIO(payload))

    assert reader.read("s3://bucket/clip.mp4") is not None


def test_clip_reader_drop_log_escapes_a_record_separator_in_the_uri(
    loguru_records: list[dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """A record separator in a dropped clip's URI is escaped in the warning, not passed or dropped.

    ``clip_uri`` is untrusted table data that this warning interpolates directly,
    so the image leg routes it through the same redaction the action leg uses. The
    escape is asserted PRESENT: a redaction that deleted the separator would also
    keep it out of the record, while naming a clip that exists in no store.
    """
    forged = str(tmp_path / "a\nWARNING forged line.mp4")

    assert ClipFrameReader().read(forged) is None

    (drop,) = [record["message"] for record in loguru_records if "image open failed" in record["message"]]
    assert r"a\u000aWARNING forged line.mp4" in drop


def test_drop_log_names_source_and_carries_traceback(loguru_records: list[dict[str, Any]]) -> None:
    """A dropped clip is logged with its source label and a traceback, not anonymously.

    At batch scale an unattributable warning cannot be triaged; the label lets an
    operator find the clip and the traceback lets them see why it failed.
    """
    label = "s3://bucket/prefix/broken-clip.mp4"
    assert read_first_frame(io.BytesIO(b"not a video"), source_label=label) is None
    drops = [record for record in loguru_records if "decode failed" in record["message"]]
    assert drops
    assert all(label in record["message"] for record in drops)
    assert all(record["exception"] is not None for record in drops)


@pytest.mark.parametrize("width", [0, -1])
def test_clip_reader_rejects_a_read_width_below_one(width: int) -> None:
    """A width below one worker is rejected at construction: such a reader would read nothing."""
    with pytest.raises(ValueError, match="must be >= 1"):
        ClipFrameReader(read_concurrency=width)


def test_read_many_keys_each_frame_by_its_input_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every frame comes back under the position of the URI it was read from.

    The key IS the caller's row, which is what leaves no sequence for a consumer
    to line up against the URIs it passed in.
    """
    _stub_single_reads(monkeypatch, lambda uri: _marker_frame(int(uri)))

    frames = ClipFrameReader().read_many(["7", "8", "9"])

    assert {position: int(frame[0, 0, 0]) for position, frame in frames.items()} == {0: 7, 1: 8, 2: 9}


def test_read_many_skips_a_row_that_names_no_media_without_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NULL or empty URI yields no entry and costs no read at all."""
    attempted: list[str] = []

    def read(uri: str) -> npt.NDArray[np.uint8]:
        attempted.append(uri)
        return _marker_frame(1)

    _stub_single_reads(monkeypatch, read)

    frames = ClipFrameReader().read_many([None, "", "s3://bucket/clip.mp4"])

    assert attempted == ["s3://bucket/clip.mp4"]
    assert list(frames) == [2]


def test_read_many_drops_one_clip_without_disturbing_the_positions_around_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable clip costs its own entry only; the rows either side keep their positions.

    A per-URI drop stays a per-URI drop in the batch form: it must neither fail the
    batch nor shift the survivors into a contiguous run at the front.
    """
    _stub_single_reads(monkeypatch, lambda uri: None if uri == "2" else _marker_frame(int(uri)))

    frames = ClipFrameReader(read_concurrency=3).read_many(["1", "2", "3"])

    assert {position: int(frame[0, 0, 0]) for position, frame in frames.items()} == {0: 1, 2: 3}


def test_read_many_keys_by_input_position_when_reads_complete_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every frame keeps its own position even though the reads finish in reverse order.

    The single highest-risk property of a concurrent read path. The caller attaches
    ``clip_id`` positionally, so results collected in completion order would pair
    every embedding with the wrong clip - a corruption that raises nothing, fails
    no schema check, and is invisible in the row counts. Each stubbed read waits
    for its successor, and the completion order is asserted, so the test cannot
    pass by having quietly run the reads one at a time.
    """
    uris = ["1", "2", "3", "4"]
    finished = [threading.Event() for _ in uris]
    completion_order: list[int] = []

    def read(uri: str) -> npt.NDArray[np.uint8]:
        position = int(uri) - 1
        successor = position + 1
        if successor < len(finished):
            finished[successor].wait(timeout=_READ_BARRIER_TIMEOUT_S)
        completion_order.append(position)
        finished[position].set()
        return _marker_frame(int(uri))

    _stub_single_reads(monkeypatch, read)

    frames = ClipFrameReader(read_concurrency=len(uris)).read_many(uris)

    assert completion_order == list(reversed(range(len(uris))))
    assert {position: int(frame[0, 0, 0]) for position, frame in frames.items()} == {0: 1, 1: 2, 2: 3, 3: 4}


def test_read_many_propagates_a_systemic_read_failure_from_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker's systemic transport fault fails the batch instead of omitting its position.

    Every submitted read is awaited, so the exception cannot die with the thread
    that raised it. Losing it would leave that row indistinguishable from a
    genuinely unreadable clip, so a whole run could commit an all-NULL column and
    exit successfully.
    """

    def read(uri: str) -> npt.NDArray[np.uint8]:
        if uri == "2":
            msg = "connection reset by peer"
            raise OSError(msg)
        return _marker_frame(int(uri))

    _stub_single_reads(monkeypatch, read)

    with pytest.raises(OSError, match="connection reset"):
        ClipFrameReader(read_concurrency=4).read_many(["1", "2", "3", "4"])


def test_read_many_reads_on_the_calling_thread_at_concurrency_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Width one reads inline, with no pool interposed.

    Pins the fast path as a real path rather than an alias for a one-worker pool:
    a caller that has not asked for concurrency gets a plain loop, so a serial
    deployment carries no thread hand-off it did not opt into.
    """
    threads: list[threading.Thread] = []

    def read(uri: str) -> npt.NDArray[np.uint8]:
        threads.append(threading.current_thread())
        return _marker_frame(int(uri))

    _stub_single_reads(monkeypatch, read)

    ClipFrameReader().read_many(["1", "2", "3"])

    assert threads == [threading.current_thread()] * 3


def test_read_many_reads_inline_when_only_one_row_names_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch with one readable URI reads on the calling thread even with width to spare.

    A pool cannot overlap one read with anything, so spawning it would add a thread
    hand-off for no parallelism. Pins the bound on the batch's own contents, which
    a partially-filled trailing scan batch reaches routinely.
    """
    threads: list[threading.Thread] = []

    def read(uri: str) -> npt.NDArray[np.uint8]:
        threads.append(threading.current_thread())
        return _marker_frame(int(uri))

    _stub_single_reads(monkeypatch, read)

    ClipFrameReader(read_concurrency=4).read_many([None, "1"])

    assert threads == [threading.current_thread()]


def test_read_many_resolves_every_backend_before_the_first_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport params for every URI in the batch resolve before any read starts.

    Resolution is what a bad storage profile fails on, so it has to happen on the
    calling thread: resolved up front, the misconfiguration raises once and
    deterministically instead of racing in every worker, and no worker is the first
    to populate the shared cache.
    """
    calls: list[tuple[str, str]] = []

    def resolve(uri: str, *, profile_name: str) -> dict[str, Any]:
        del profile_name
        calls.append(("resolve", uri))
        return {}

    def read(uri: str) -> npt.NDArray[np.uint8]:
        calls.append(("read", uri))
        return _marker_frame(1)

    monkeypatch.setattr(frame_reader, "get_smart_open_params", resolve)
    monkeypatch.setattr(ClipFrameReader, "read", lambda _self, uri: read(uri))

    ClipFrameReader(read_concurrency=4).read_many(["s3://bucket-a/1.mp4", "s3://bucket-b/2.mp4"])

    assert [kind for kind, _ in calls] == ["resolve", "resolve", "read", "read"]
