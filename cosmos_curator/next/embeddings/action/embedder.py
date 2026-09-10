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

r"""CPU wrist-motion action embedder: read artifacts, derive descriptors, project by PCA.

The action leg is two compute phases separated by the choice of a PCA basis:

::

    source batch --extract--> DESCRIPTOR_ROW --project--> action group batch
                  (read +      (uri +                     (vector + provenance)
                   derive)      descriptor)
                                   ^
                                   | the basis is chosen BEFORE compute starts:
                                   | loaded by fingerprint, or fit by the driver
                                   | from a bounded URI sample

``DualWristMotionDescriptorExtractor`` reads the artifact and builds the raw
``DESCRIPTOR_DIM`` descriptor. ``DualWristMotionProjector`` reduces it to the
``ACTION_DIM`` group; it never touches storage. ``DualWristMotionEmbedder``
composes the two into the single ``__call__`` the storage layer drives, so each
artifact is read once and no intermediate descriptor is persisted or shuffled.

The extractor is also used on its own by the driver when it must FIT a basis: the
driver picks a bounded sample of ``action_data_uri`` values and extracts only
those, so the fit never scans the corpus.

Every phase is pure compute over positions: no clip-identity column is read or
emitted, and both are cardinality- and order-preserving, which is what lets the
caller attach ``clip_id`` positionally. ``action_data_uri`` is a payload column,
not a row key: it is read, carried through the descriptor row, and named
(redacted) in rejection warnings. A clip with no ``action_data_uri``, or
whose artifact is unreadable / undecodable / geometrically rejected, keeps its row
but carries a NULL descriptor, which projects to an all-NULL (pending /
retryable) action group rather than being dropped.

State lives on the object and is built lazily: the storage client is created on
first read, so nothing live crosses a pickle boundary when the object is shipped
to a worker, and it is then reused for every batch that worker handles.

One action artifact is shared by every view of a span, so within one extract call
the descriptor for a given ``action_data_uri`` is computed once and fanned out to
each view's row: the call resolves its distinct URIs first and derives only those.
Those derivations overlap behind a bounded thread pool, because the per-artifact
cost is dominated by the network fetch rather than by the geometry it feeds.
"""

import typing
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import backend_key, get_storage_client, read_bytes
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import (
    DESCRIPTOR_DIM,
    DESCRIPTOR_VERSION,
    DescriptorRejection,
    dual_wrist_motion_descriptor,
)
from cosmos_curator.next.embeddings.schemas import (
    ACTION_DIM,
    action_columns_batch,
    descriptor_batch,
)
from cosmos_curator.next.embeddings.uri_redaction import redact_diagnostic_for_log, redact_for_log
from cosmos_curator.next.media.action_binary import decode_action_artifact

type _DescriptorOutcome = npt.NDArray[np.float32] | DescriptorRejection | None
"""What one artifact yielded: its descriptor, why it was rejected, or nothing.

``None`` is a read or decode failure, which the reader has already logged; the
two other variants are the geometry's own result. Naming the union lets the
worker return a rejection instead of recording it, so the ledger can be written
on one thread in a deterministic order.
"""


@attrs.frozen
class DualWristMotionReadConfig:
    """Constructor config for the wrist-motion action-leg readers.

    Attributes:
        storage_profile: Storage profile used to read action artifacts.
        read_concurrency: Artifacts fetched at once within one extract call. It
            widens the network wait only. Defaults to serial reads, so a directly
            constructed extractor behaves like a plain loop; the tuned production
            value lives on the recipe's config.

    Action input is the Mecka action-artifact contract: production artifacts are
    self-describing ACT2 ``.bin`` payloads. Any other payload (a legacy pickle, a
    non-dexterous export) simply fails ``decode_action_artifact`` and drops that
    row to a NULL descriptor - there is no format field and no dataset registry.

    """

    storage_profile: str = "default"
    read_concurrency: int = attrs.field(default=1, validator=attrs.validators.ge(1))


class _ActionPayloadReader:
    """Reader that decodes an action artifact by URL.

    One storage client is built lazily per backend (scheme + bucket) and reused for
    the owning object's lifetime; nothing live is captured before the first read, so
    the owner is cloudpickle-safe. A read/decode failure leaves that row's vector
    NULL (logged as one redacted warning) rather than failing the leg, so the clip
    stays pending and is retried on the next run.
    """

    def __init__(self, config: DualWristMotionReadConfig) -> None:
        """Capture the read config; defer client construction to the first read."""
        self._config = config
        self._clients: dict[str, StorageClient | None] = {}

    def _ensure_client(self, url: str) -> StorageClient | None:
        """Resolve (once per backend) the storage client for ``url``."""
        # Resolved OUTSIDE the per-row drop handler: a storage_profile with no
        # matching credentials is wrong for every row, so it must fail the leg
        # loudly rather than be logged-and-dropped per artifact, silently emptying
        # the table. Only profile resolution is guaranteed here; a bad endpoint or
        # missing object still surfaces later as a per-row drop.
        key = backend_key(url)
        if key not in self._clients:
            self._clients[key] = get_storage_client(url, profile_name=self._config.storage_profile)
        return self._clients[key]

    def warm_client(self, url: str) -> None:
        """Resolve and cache ``url``'s backend storage client without reading the object.

        Lets a caller that reads a batch concurrently resolve every backend it will
        touch from one thread first. Two things follow: a systemic misconfiguration
        raises once, deterministically, rather than racing in every worker at once,
        and the per-backend cache is never first populated concurrently - which is
        what makes concurrent ``read`` on one instance safe, since the cache is then
        the only mutable state and the workers only read it.
        """
        self._ensure_client(url)

    def read(self, url: str) -> dict[str, Any] | None:
        """Read and decode one ACT2 action artifact, returning its arrays or ``None``.

        Reads the bytes and decodes the self-describing ACT2 ``.bin`` header; any
        read or decode failure (missing object, non-ACT2 bytes such as a legacy
        pickle, truncated payload) leaves that row ``None`` rather than failing
        the leg, warning once with the artifact named and the cause summarized.
        """
        client = self._ensure_client(url)
        try:
            data = read_bytes(url, client)
            artifact = decode_action_artifact(data)
        except (OSError, ValueError, TypeError) as exc:
            # The exception is RENDERED and sanitized rather than attached. Its
            # own text quotes the URI back - an OSError reports the filename it
            # failed on, presigned query included - and a traceback cannot be
            # sanitized from here at all, because loguru renders each frame's
            # argument values and would publish the raw url from several frames.
            # Type plus message keeps what identifies the failure: the S3 error
            # code for a missing object, the byte count for a short payload,
            # neither of which the exception class alone reports.
            diagnostic = redact_diagnostic_for_log(f"{type(exc).__name__}: {exc}", url)
            logger.warning(
                f"action artifact read failed for {redact_for_log(url)}: {diagnostic}; row kept as pending NULL"
            )
            return None
        return artifact.arrays


@attrs.define
class _RejectionLedger:
    """Per-batch tally of geometric rejections, naming a bounded sample of artifacts.

    Attributes:
        counts: Rejected artifacts per reason. One artifact can back several view
            rows, so this counts artifacts, not the rows left pending.
        examples: Redacted artifact URIs per reason, in first-seen order, capped at
            ``MAX_EXAMPLES_PER_REASON``.

    """

    # A few examples separate a one-off bad artifact from a systemic pattern (a
    # shared prefix, one bad shard) while keeping a fully-rejecting batch to a
    # handful of URIs per reason, rather than one log line per rejected artifact.
    MAX_EXAMPLES_PER_REASON: ClassVar[int] = 3

    # init=False so the two maps cannot be constructed holding different reason
    # keys; ``record`` populates them together.
    counts: Counter[DescriptorRejection] = attrs.field(init=False, factory=Counter)
    examples: dict[DescriptorRejection, list[str]] = attrs.field(init=False, factory=dict)

    def record(self, reason: DescriptorRejection, uri: str) -> None:
        """Count one rejected artifact, retaining its redacted URI while under the cap."""
        self.counts[reason] += 1
        retained = self.examples.setdefault(reason, [])
        if len(retained) < self.MAX_EXAMPLES_PER_REASON:
            retained.append(redact_for_log(uri))

    def summary(self) -> dict[str, dict[str, int | list[str]]]:
        """Return a caller-owned per-reason view of the count and the retained URIs."""
        return {
            reason.value: {"count": count, "examples": list(self.examples.get(reason, ()))}
            for reason, count in self.counts.items()
        }


class DualWristMotionDescriptorExtractor:
    """Reader deriving raw ``DESCRIPTOR_DIM`` wrist-motion descriptors from ACT2 artifacts.

    Maps a batch carrying ``action_data_uri`` to one ``DESCRIPTOR_ROW`` per input
    row. It serves two callers: it is the first half of
    ``DualWristMotionEmbedder``, and it runs on its own when the driver extracts a
    bounded URI sample to fit the PCA basis. The entry point is ``__call__`` (not a
    named method) because that second caller schedules the class itself as a Ray
    Data actor - a bound method cannot be given an actor pool - and because it
    matches how the text and image legs are invoked.

    Per artifact the leg is dominated by the network fetch, so the reads within one
    call are overlapped up to the config's ``read_concurrency``. That widens only
    the waiting: the wrist geometry each read feeds is GIL-held numpy, so one
    actor's throughput reaches a ceiling that no read width moves past. Dividing
    that ceiling takes more processes, which here means more actors - a property of
    the table's fragment geometry rather than a knob on this class.

    The storage clients it memoizes live for the object's lifetime, so one instance
    per actor amortizes client setup across every batch that actor sees.

    Attributes:
        SOURCE_COLUMNS: Columns the leg's scan must project. It includes
            ``clip_id`` even though this class never reads it: the caller that
            owns storage joins the computed group back on ``clip_id``, so the scan
            has to carry it. Held on the class (not module-level) so the name
            matches the other legs' ``SOURCE_COLUMNS`` convention and is pinned by
            test to be a subset of ``EMBED_SOURCE_COLUMNS``.

    """

    SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = ("clip_id", "action_data_uri")

    def __init__(self, config: DualWristMotionReadConfig) -> None:
        """Build the (lazy) payload reader from the read config."""
        self._reader = _ActionPayloadReader(config)
        self._read_concurrency = config.read_concurrency

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Emit one ``DESCRIPTOR_ROW`` per input row (cardinality- and order-preserving).

        A clip with no ``action_data_uri``, or whose artifact fails to read /
        decode / is geometrically rejected, carries a NULL descriptor (``valid``
        False) rather than being dropped. Applicability is purely non-empty
        ``action_data_uri``: no ``source_dataset`` and no dexterous registry are
        consulted.

        The distinct URIs are resolved first, so each span's artifact is read and
        derived once and then fanned out to every view sharing it.

        ::

            action_uris (row order; NULL and "" stay distinct)
                |
                +-- distinct, first-seen order --> warm clients   [calling thread]
                |                              --> read + derive  [worker threads]
                |                                      |
                |                                      v
                |                          ndarray | rejection | None
                |                                      |
                |            record rejections in distinct order  [calling thread]
                v
            scatter each descriptor back by INPUT ROW INDEX
        """
        rows = batch.num_rows
        # NULL and empty-string URIs are kept DISTINCT: both mean "no artifact" for
        # this extract (both skip below), but the NULL is preserved into the
        # DESCRIPTOR_ROW so action_data_uri stays the true URI identity that the PCA
        # sampling / de-dup key on, rather than being collapsed to a fake "".
        action_uris: list[str | None] = batch.column("action_data_uri").to_pylist()
        matrix = np.zeros((rows, DESCRIPTOR_DIM), dtype=np.float32)
        valid = np.zeros(rows, dtype=np.bool_)
        # One artifact backs every view row of a span, so collapsing to the distinct
        # set is what keeps the reads at one per artifact; first-seen order is what
        # the ledger's "first-seen" example order below is measured against.
        distinct_uris = list(dict.fromkeys(uri for uri in action_uris if uri))
        outcomes = self._outcomes_for_uris(distinct_uris)
        descriptors = self._record_and_collect(distinct_uris, outcomes)
        for index, uri in enumerate(action_uris):
            descriptor = descriptors.get(uri) if uri else None
            if descriptor is None:
                continue
            matrix[index] = descriptor
            valid[index] = True
        return descriptor_batch(action_uris, matrix, DESCRIPTOR_DIM, valid)

    def _outcomes_for_uris(self, uris: Sequence[str]) -> dict[str, _DescriptorOutcome]:
        """Derive one outcome per distinct URI, overlapping only the artifact reads.

        Returns:
            Every URI in ``uris`` mapped to its descriptor, its typed rejection, or
            ``None`` for a read/decode failure the reader has already logged.

        Raises:
            Exception: Whatever resolving a storage client or a worker raises. The
                first failure in submission order propagates and the reads still
                queued behind it are cancelled, so a systemic fault fails the leg
                promptly instead of leaving its rows silently NULL beside the
                genuine per-row drops.

        """
        if not uris:
            return {}
        # Resolve every backend on THIS thread before any worker starts. A profile
        # with no matching credentials is wrong for every row, so it has to fail the
        # leg loudly; resolving it concurrently would both race the client cache and
        # turn one systemic fault into N per-artifact drops. After the first URI per
        # backend this is a dict hit.
        for uri in uris:
            self._reader.warm_client(uri)
        workers = min(self._read_concurrency, len(uris))
        if workers == 1:
            return {uri: self._descriptor_for_uri(uri) for uri in uris}
        # Threads, not processes: an artifact read is network wait, which releases
        # the GIL, whereas a process pool would ship every payload across an IPC
        # boundary. Read, decode and geometry all run in the worker so each payload
        # is freed as soon as its descriptor exists - the geometry is GIL-held
        # either way, so hoisting it back to the caller would buy nothing. Per-call
        # executor: an actor with no destructor needs no shutdown hook.
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="action-read")
        try:
            submitted = [(uri, pool.submit(self._descriptor_for_uri, uri)) for uri in uris]
            # Awaited in submission order, so the exception that surfaces is the
            # earliest failing artifact rather than whichever thread lost the race.
            return {uri: future.result() for uri, future in submitted}
        finally:
            # Shut down explicitly, because the context manager would pass
            # cancel_futures=False and so run every queued read anyway. On the
            # raising path each of those would spend its own full read_bytes retry
            # budget before the first exception could propagate, surfacing one
            # systemic fault a wave at a time. A no-op on the happy path, where
            # every read has already been awaited.
            pool.shutdown(cancel_futures=True)

    def _record_and_collect(
        self, uris: Sequence[str], outcomes: dict[str, _DescriptorOutcome]
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Split the outcomes into surviving descriptors, recording rejections in ``uris`` order.

        The single site that consumes the outcome union, and it runs on the CALLING
        thread walking ``uris``: ``_RejectionLedger`` is not thread-safe, and its
        examples are documented as first-seen, which arrival order would scramble
        even if the mutation were safe. Recording here rather than inside the worker
        is what keeps the ledger identical to the serial path's, at any read width.
        The ``assert_never`` makes a future fourth variant of the union a mypy error
        rather than a silently unrecorded drop.
        """
        ledger = _RejectionLedger()
        descriptors: dict[str, npt.NDArray[np.float32]] = {}
        for uri in uris:
            outcome = outcomes[uri]
            match outcome:
                case None:
                    # Read or decode failure; the reader already logged one
                    # redacted warning naming the artifact and its cause.
                    pass
                case DescriptorRejection() as reason:
                    ledger.record(reason, uri)
                case np.ndarray() as descriptor:
                    descriptors[uri] = descriptor
                case _:
                    typing.assert_never(outcome)
        self._log_rejections(ledger)
        return descriptors

    def _descriptor_for_uri(self, uri: str) -> _DescriptorOutcome:
        """Read one artifact and derive its descriptor, or return why it produced none.

        Runs whole in a worker thread, so it returns its rejection rather than
        recording it; the caller records on the calling thread in a deterministic
        order. A read failure returns ``None`` (already logged by the reader as one
        redacted warning, not a traceback - see ``_ActionPayloadReader.read``). The
        Mecka contract assumes every artifact is a mecka export, so
        the descriptor is always built with mecka wrist-frame alignment; the
        ego-camera track is extracted first, so a non-mecka payload rejects to
        ``MISSING_CAMERA`` and only an artifact that carries a camera but no hands
        reaches ``MISSING_ARM``.
        """
        payload = self._reader.read(uri)
        if payload is None:
            return None
        return dual_wrist_motion_descriptor(payload, align_mecka=True)

    @staticmethod
    def _log_rejections(ledger: _RejectionLedger) -> None:
        """Emit one per-batch summary of geometric rejections, keyed by reason.

        Reports artifacts rather than clips: the per-batch memo derives each
        artifact once, so a rejection is a property of the artifact and one of them
        can leave several view rows pending. No clip key is available here anyway -
        the driver's PCA-fit pass builds batches from URIs alone. Unreadable
        artifacts are logged individually by the reader (one redacted warning each,
        no traceback), so they are not re-summarized here.
        """
        if not ledger.counts:
            return
        # Log-safe by construction: the ledger stores URIs already escaped by
        # redact_for_log, so no artifact path can split this record into a forged
        # second log line.
        logger.warning(
            "action extract rejected artifacts this batch (their rows kept as pending NULL): "
            f"descriptor_rejections={ledger.summary()}"
        )


class DualWristMotionProjector:
    """Reduce ``DESCRIPTOR_ROW`` batches to the ``ACTION_DIM`` action group batch.

    Performs no storage I/O - the descriptors were already read and derived by the
    extractor - so the PCA fit sample and this projection never re-read an
    artifact. The ``PcaArtifact`` is captured at construction (read-only numpy
    data, so the object is picklable) and every worker shares one basis, keeping
    all action vectors comparable across runs. Each valid row is stamped with the
    basis ``fingerprint`` so a group accidentally filled from two bases is rejected
    by the single-producer check the driver runs before every fill.

    Cardinality- and order-preserving: a NULL descriptor (a per-row extract
    failure) projects to an all-NULL action group at the same position, so every
    input row yields exactly one output row.
    """

    def __init__(self, pca: PcaArtifact) -> None:
        """Capture the shared PCA basis."""
        self._pca = pca

    def project(self, batch: pa.Table) -> pa.Table:
        """Project each ``DESCRIPTOR_ROW`` in ``batch`` to an action group batch.

        Only the non-null descriptors are projected; their reduced vectors are
        scattered back into their input row positions and invalid rows stay NULL.
        The empty batch needs no special case: ``decode_descriptors`` yields a
        ``(0, DESCRIPTOR_DIM)`` matrix, which projects to ``(0, ACTION_DIM)``.
        """
        rows = batch.num_rows
        valid, present = decode_descriptors(batch)
        reduced = np.zeros((rows, ACTION_DIM), dtype=np.float32)
        if present.shape[0]:
            # Scatter the survivors' PCA coordinates back to their input rows; the
            # zero rows left for invalid inputs are masked out by ``valid``.
            reduced[valid] = self._pca.project(present).astype(np.float32)
        return action_columns_batch(rows, reduced, DESCRIPTOR_VERSION, self._pca.fingerprint, valid)


class DualWristMotionEmbedder:
    """The action leg's single compute call: read the artifact, derive, project.

    Composes ``DualWristMotionDescriptorExtractor`` and
    ``DualWristMotionProjector`` so the storage layer drives one uniform
    ``__call__(batch) -> group columns`` for every modality. Fusing the two phases
    in one call is what keeps descriptors transient: each artifact is read once and
    the raw ``DESCRIPTOR_DIM`` intermediate never leaves this object.

    The basis must already be chosen when this is constructed - it is resolved by
    the driver, either loaded by the fingerprint the group's rows already carry or
    fit from a bounded URI sample.

    Attributes:
        SOURCE_COLUMNS: Columns the leg's scan must project, shared with the
            extractor so the two cannot disagree.

    """

    SOURCE_COLUMNS: ClassVar[tuple[str, ...]] = DualWristMotionDescriptorExtractor.SOURCE_COLUMNS

    def __init__(self, config: DualWristMotionReadConfig, pca: PcaArtifact) -> None:
        """Build the extractor (lazy reader) and bind the resolved PCA basis."""
        self._extractor = DualWristMotionDescriptorExtractor(config)
        self._projector = DualWristMotionProjector(pca)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return the action group columns for ``batch``, one row per input row."""
        return self._projector.project(self._extractor(batch))


def decode_descriptors(batch: pa.Table) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """Decode a ``DESCRIPTOR_ROW`` batch into ``(valid_mask, present_matrix)``.

    Returns the per-row validity mask (True where the descriptor is non-NULL) and
    the compact ``(num_valid, DESCRIPTOR_DIM)`` matrix of only the present
    descriptors, in row order. A NULL row (a per-row extract failure) contributes
    no matrix row, so the caller scatters the present rows back to their positions
    via the mask.

    Round-trip-safe on purpose: it reads only the non-null lists via
    ``drop_null()`` and never assumes a NULL list entry retains a fixed offset
    stride (Arrow may compact NULL entries to zero length on an IPC round-trip, so
    a whole-buffer reshape over all rows would misalign after a Ray materialize).
    """
    column = batch.column("descriptor").combine_chunks()
    valid = column.is_valid().to_numpy(zero_copy_only=False)
    num_valid = int(valid.sum())
    if num_valid == 0:
        return valid, np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32)
    present = column.drop_null()
    matrix = np.asarray(present.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        num_valid, DESCRIPTOR_DIM
    )
    return valid, matrix
