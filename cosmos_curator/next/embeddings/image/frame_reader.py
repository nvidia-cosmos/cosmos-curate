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

"""First-displayable-frame acquisition for the image leg (via the sensor library).

The image embedding of a clip is computed from a single representative RGB
frame: the first *displayable* frame. Using the sensor library's
``CameraSensor`` (rather than a hand-rolled PyAV loop) gets correct
presentation-order handling for B-frame streams for free - decode order is not
display order, so the first decoded frame is not necessarily the first frame a
viewer sees. ``CameraSensor.start_ns`` is the canonical first-frame timestamp,
and a one-timestamp ``SamplingGrid`` at that instant pulls exactly that frame.

Three layers cooperate::

    ClipFrameReader.read_many(uris) --> {input position: frame}
        |                               reads overlap up to read_concurrency
        v
    ClipFrameReader.read(uri)
        |
        v
    smart_open.open(uri) --> one sequential read (capped) --> in-memory buffer
        |
        v
    read_first_frame(buffer) --> (H, W, 3) uint8 RGB frame

``read_first_frame`` decodes an already-open stream; ``ClipFrameReader`` owns the
transport around it - resolving ``smart_open`` params lazily per backend, fetching
the object's bytes, and splitting a systemic misconfiguration (which must fail the
whole leg) from a single unreadable clip (which is dropped). A stream with no
decodable frame returns ``None`` (the caller drops the row) rather than raising, so
one corrupt clip cannot fail an entire batch.

The reader also owns the concurrency of its own I/O: ``read_many`` fans one
caller's URIs out over a bounded thread pool and returns each frame under the
position it was read from, so a consumer neither sizes a pool for a resource it
does not own nor re-derives which row a frame belongs to.

The transfer is one sequential read into memory rather than a seekable remote
stream handed to the decoder, because on an object store each ``seek`` becomes a
fresh ranged GET and a decoder's index build is seek-heavy. The rule is in
``docs/curator/guides/sensor-library-cloud-storage.md``; the measured wall times
behind it are in ``docs/curator/design/sensor-library-cloud-storage.md``.
"""

import contextlib
import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Literal

import attrs
import numpy as np
import numpy.typing as npt
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.utils.storage.storage_utils import (
    backend_key,
    get_smart_open_params,
    is_missing_object_error,
)
from cosmos_curator.next.embeddings.uri_redaction import redact_for_log

_RGB_CHANNELS = 3
_HWC_NDIM = 3  # a decoded frame is a 3-axis (height, width, channel) array


def read_first_frame(
    stream: io.BufferedIOBase,
    *,
    source_label: str = "",
    stream_idx: int = 0,
) -> npt.NDArray[np.uint8] | None:
    """Decode the first displayable RGB frame from an open binary video stream.

    Args:
        stream: An open, readable, seekable binary stream positioned at the video's
            start (e.g. a local file handle or an in-memory buffer).
        source_label: Identifier (e.g. clip URI) included in the drop warning so a
            failed clip is attributable at batch scale.
        stream_idx: Video stream index within the container (default 0).

    Returns:
        An ``(H, W, 3)`` ``uint8`` RGB frame, or ``None`` when the stream has no
        decodable / displayable frame or does not decode to 3-channel RGB.

    """
    try:
        sensor = CameraSensor(source=stream, stream_idx=stream_idx)
        start_ns = sensor.start_ns
    except Exception:  # noqa: BLE001 - a corrupt clip must drop, not fail the batch
        logger.opt(exception=True).warning(f"first-frame decode failed for {redact_for_log(source_label)}; dropping")
        return None

    # Pure arithmetic on start_ns: a bug here is a programming error, so the grid
    # is built OUTSIDE the decode try/except and is allowed to surface loudly
    # rather than being swallowed as a per-clip data problem.
    grid = SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=start_ns + 1,
        timestamps_ns=np.array([start_ns], dtype=np.int64),
        stride_ns=1,
        duration_ns=1,
    )

    try:
        # closing() throws GeneratorExit into the suspended generator, so the
        # sensor's decoder context manager (owning the PyAV container) unwinds
        # deterministically here rather than at refcount finalization. The
        # single-timestamp grid sits exactly on start_ns, so the nearest-
        # timestamp match is exact; no max-delta gate is needed.
        with contextlib.closing(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy())) as batches:
            camera_data = next(batches, None)
    except Exception:  # noqa: BLE001 - a corrupt clip must drop, not fail the batch
        logger.opt(exception=True).warning(f"first-frame decode failed for {redact_for_log(source_label)}; dropping")
        return None

    if camera_data is None or len(camera_data.frames) == 0:
        return None
    # np.array (copy) yields a writable, self-owned array: it drops the read-only
    # view into the sensor's frame buffer (torchvision warns on read-only input)
    # and does not retain the parent (N, H, W, 3) buffer past this call.
    frame = np.array(camera_data.frames[0])
    # Belt-and-braces on the CameraData (N, H, W, 3) uint8 invariant: a future
    # library change that returned another shape drops the clip rather than
    # feeding a malformed frame into the processor.
    if frame.ndim != _HWC_NDIM or frame.shape[2] != _RGB_CHANNELS:
        return None
    return frame.astype(np.uint8, copy=False)


@attrs.frozen
class ClipFrameReader:
    """Fetch a clip URI into memory and return its first displayable RGB frame, or ``None``.

    Owns the transport around ``read_first_frame`` - and the concurrency of that
    transport - which is what leaves the embedder body model-only: hand it a
    ``clip_uri`` or a whole batch of them and get frames back, a per-clip failure
    surfacing as ``None`` or as an absent position.

    ``smart_open`` params resolve lazily per backend on first use, never at
    construction, so the reader stays cloudpickle-safe when shipped to a Ray
    actor. Resolution sits OUTSIDE the per-clip drop handler on purpose: a
    systemic misconfiguration (wrong ``storage_profile`` / endpoint, or an
    unbridgeable backend) must fail the whole leg rather than be
    logged-and-dropped per clip, leaving every vector NULL. ``read_many``
    resolves every backend its batch names before it fans out, so that failure
    raises once from the calling thread instead of racing in every worker, and the
    cache is never first populated concurrently - it is then the only mutable
    state and the workers only read it.

    Attributes:
        MAX_CLIP_BYTES: Ceiling on one clip's buffered bytes. A larger object is
            dropped like any other unreadable clip.
        storage_profile: Storage profile used to bridge remote reads; the params
            it resolves to are ``{}`` for local paths.
        read_concurrency: Clips ``read_many`` fetches at once. It widens the
            download wait only - the decode those fetches feed is GIL-serialized,
            so added width shortens nothing else, and each concurrent read holds
            one clip's bytes plus its decoded frame
            (``docs/curator/design/curator-next-embeddings.md`` section 4.2).
            Defaults to serial reads, so a directly constructed reader behaves
            like a plain loop.

    """

    # Sized far above one cut clip so it never trips on real media; it exists for
    # the mis-populated ``clip_uri`` that names an uncut chunk instead. Without it
    # that one row exhausts the actor's heap, and an OOM-killed actor costs the
    # whole fragment rather than the single row every other read failure costs.
    MAX_CLIP_BYTES: ClassVar[int] = 256 * 1024 * 1024

    storage_profile: str = "default"
    read_concurrency: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    # Lazily-resolved smart_open transport params, cached per backend key. A dict
    # is the mutable holder a frozen class needs for lazy caching; values can
    # legitimately be ``{}`` for local paths, so the cache cannot be a single
    # optional dict keyed only by "resolved vs not".
    _params: dict[str, dict[str, Any]] = attrs.field(factory=dict, init=False, eq=False, repr=False)

    def read(self, uri: str) -> npt.NDArray[np.uint8] | None:
        """Fetch ``uri`` into memory and decode its first displayable frame.

        Returns:
            An ``(H, W, 3)`` ``uint8`` RGB frame, or ``None`` when the clip is
            missing / corrupt / undecodable / over ``MAX_CLIP_BYTES`` (the caller
            drops that row).

        Raises:
            OSError: If the open or the transfer fails with anything
                ``is_missing_object_error`` does not classify as an expected
                missing object.
            Exception: If transport-param resolution fails, or if
                ``read_first_frame`` raises a programming error it deliberately
                does not swallow (e.g. a malformed sampling grid). A corrupt /
                undecodable clip is handled inside ``read_first_frame``, so only
                the transport is wrapped here.

        """
        params = self._transport_params(uri)
        # Both transport phases drop ONLY an expected missing object;
        # read_first_frame owns its own per-clip decode drop and lets programming
        # errors surface. The phase label distinguishes a naming or permission
        # fault ("open") from a connectivity one ("fetch").
        try:
            stream_cm = smart_open.open(uri, "rb", **params)
        except OSError as exc:
            return self._drop_or_raise(uri, "open", exc)
        try:
            with stream_cm as stream:
                # One byte past the cap is enough to detect an overrun without
                # holding the whole object. BytesIO shares the ``bytes`` object
                # rather than copying it, so the fetched object is resident once,
                # not twice. A short read would truncate the clip rather than
                # detect an overrun, which relies on both transports honouring
                # BufferedIOBase.read(size): a local handle and smart_open's S3
                # reader both loop to the requested size or EOF.
                data = stream.read(self.MAX_CLIP_BYTES + 1)
        except OSError as exc:
            return self._drop_or_raise(uri, "fetch", exc)
        if len(data) > self.MAX_CLIP_BYTES:
            logger.warning(
                f"image clip {redact_for_log(uri)} exceeds {self.MAX_CLIP_BYTES} bytes; dropping. An object "
                f"this large is not one cut clip, so the URI names the wrong media"
            )
            return None
        return read_first_frame(io.BytesIO(data), source_label=uri)

    def read_many(self, uris: Sequence[str | None]) -> dict[int, npt.NDArray[np.uint8]]:
        """Read one frame per URI, keyed by that URI's position in ``uris``.

        Args:
            uris: One URI per caller row, in row order. A ``None`` or empty entry
                is a row that names no media and is skipped without a read.

        Returns:
            ``{position: frame}`` holding an entry for every URI that yielded a
            frame. Keying by position rather than returning a sequence is what
            makes a misalignment unrepresentable: a frame carries the row it was
            read from, and a row that named no media or whose clip was dropped is
            simply absent rather than a hole the caller has to line up.

        Raises:
            Exception: Whatever transport-param resolution or ``read`` raises.
                Every submitted read is awaited, so a systemic fault fails the
                batch rather than leaving its row silently absent among the
                genuine per-clip drops.

        """
        pending = [(position, uri) for position, uri in enumerate(uris) if uri]
        if not pending:
            return {}
        # Resolve every backend in the batch on THIS thread before any worker runs,
        # so a systemic misconfiguration raises once rather than racing in N
        # workers. After the first URI per backend this is a dict hit.
        for _, uri in pending:
            self._transport_params(uri)
        workers = min(self.read_concurrency, len(pending))
        if workers == 1:
            return {position: frame for position, uri in pending if (frame := self.read(uri)) is not None}
        # Per-call executor: it needs no shutdown hook in an actor that has none.
        # Threads (not processes) because the per-clip cost is network wait, which
        # a process pool would pay for by shipping decoded frames across an IPC
        # boundary. The decode does not overlap at all - it is GIL-serialized - so
        # added threads shorten only the waiting (design doc section 4.2).
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clip-read") as pool:
            submitted = [(position, pool.submit(self.read, uri)) for position, uri in pending]
            # Awaited in submission order, so the exception that surfaces is the
            # earliest failing ROW rather than whichever thread lost the race.
            return {position: frame for position, future in submitted if (frame := future.result()) is not None}

    @staticmethod
    def _drop_or_raise(uri: str, phase: Literal["open", "fetch"], exc: OSError) -> npt.NDArray[np.uint8] | None:
        """Return the drop result (``None``) for an expected missing object, or re-raise ``exc``.

        Owns the whole transport-failure drop so a caller cannot log it and then
        fall through. The over-cap drop is separate: it is not an ``OSError``.

        Raises:
            OSError: ``exc``, when it is not an expected missing object.

        """
        if not is_missing_object_error(exc):
            raise exc
        logger.opt(exception=True).warning(f"image {phase} failed for {redact_for_log(uri)}; dropping")
        return None

    def _transport_params(self, uri: str) -> dict[str, Any]:
        """Resolve and cache ``smart_open`` transport params per backend for ``uri``."""
        key = backend_key(uri)
        cached = self._params.get(key)
        if cached is not None:
            return cached
        resolved = dict(get_smart_open_params(uri, profile_name=self.storage_profile))
        self._params[key] = resolved
        return resolved
