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

"""Distributed per-fragment fill of one embedding column group, plus its single commit.

The unit of work is a LANCE FRAGMENT. Each Ray actor is handed fragment ids, and for
each one it re-opens the table at the run's pinned version, scans just that
fragment's pending rows, computes, and writes the group's columns back in place::

    from_items(fragment_ids).map_batches(_FragmentWorker, batch_size=1)
        |   per fragment, inside one actor:
        |     READ       dataset(uri, version).get_fragment(id)
        |                fragment.scanner(columns=[clip_id, *inputs], filter=row_filter)
        |                nothing pending -> emit zero rows, fragment left untouched
        |     TRANSFORM  embedder(batch) -> ONLY the group's columns, same rows,
        |                in the same order
        |     WRITE      attach clip_id positionally, cast to the stored types
        |                fragment.update_columns(tbl, left_on=clip_id, right_on=clip_id)
        |                emit one JSON string row
        |     ON FAILURE log a WARNING and emit an ERROR payload instead; the
        |                fragment is skipped, not fatal (see below). A violated
        |                fill contract emits a FATAL payload instead of an ERROR one
        v
    driver: collect O(touched fragments) results, raise on a FATAL payload,
            otherwise partition written from skipped; nothing written while
            something was skipped is an outage and also raises
        v
    Transaction(Update(written_fragments, fields_modified))   one commit per modality

Why this shape:

- **No shuffle.** The scan is fragment-local, so compute and write are fused inside
  one actor. There is no ``groupby`` barrier forcing all compute to finish before any
  write can start, and no intermediate vector is ever materialized in Ray's object
  store.
- **Driver state is O(fragments), not O(rows).** ``_RESULT_SCHEMA`` is the enforced
  guarantee: a worker returns one JSON string per touched fragment, and a single
  string column STRUCTURALLY cannot carry column data back to the driver.
- **No row is ever replaced, so no tombstone is created.** ``update_columns`` writes a
  NEW file holding only the group's columns for the fragment's existing rows, in their
  existing offset order, and returns updated ``FragmentMetadata`` plus the field ids it
  rebound. ``LanceOperation.Update`` then re-points those field ids at the new file for
  the touched fragments only. Row addresses, row count, and every other column's file
  are untouched, so no deletion vector is written. The superseded column file stays on
  disk referenced only by older versions.
- **A fragment with nothing pending is not rewritten at all.** The worker probes the
  scanner for a first batch before calling ``update_columns``; with no pending row it
  emits zero result rows, so an unchanged rerun of a fully embedded table commits
  nothing.
- **A fragment that fails is skipped, not fatal.** One flaky fragment must not
  discard the hours of GPU time every other fragment already spent, so the worker
  catches the failure, warns, and reports it; the run commits everything that did
  succeed. No retry machinery is needed because a skipped fragment was never
  written: any rows it still owed stay pending, so the ordinary pending predicate
  re-selects them on the next run. The exception is ``FillContractError``, which
  stops the run instead; see its docstring for the two things it covers.
- **Unless NOTHING was written, which is not a partial loss.** The skip trades one
  fragment's work for the rest of the run's, and stops being a trade when there is
  no rest: the group is untouched and every count is zero, exactly as for a run
  that owed no work. ``FillContractError`` is that boundary too, and the skip count
  is what distinguishes the two - hence it is carried on the result, not only logged.
- **The inputs every fragment shares are checked once, on the driver.** The
  projection and the row filter do not vary by fragment, so a broken one is not a
  fragment fault at all: it would fail identically on every fragment, and a run
  that skipped them all would report success over an untouched group.
  ``_check_fill_preconditions`` resolves the scan plan before any worker starts,
  which is what leaves the per-fragment handler free to be broad - by the time it
  runs, the faults still reachable are genuinely local ones.
- **A contract violation travels as a payload, not as an exception.** Ray Data
  replaces whatever a UDF raises with a ``UserCodeException`` wrapper and does not
  carry the original object across the task boundary, so an exception raised in a
  worker reaches the driver as a ``RayTaskError`` that is no longer an instance of
  its own cause. The worker therefore emits a FATAL payload and the DRIVER raises
  ``FillContractError`` - on the first such payload it sees, so the remaining
  fragments are abandoned rather than run for a result that would be discarded.

``update_columns`` joins on the ordinary persisted ``clip_id`` column with left-outer
semantics: a fragment row absent from the update table keeps its previous group value,
which is what makes writing only the pending subset of a fragment safe.

The producer does not guarantee ``clip_id`` uniqueness, and Lance resolves a duplicate
key by taking one matching row's value, so this path ASSUMES rows sharing a ``clip_id``
agree: same source columns, and therefore the same computed group value and the same
applicability. Under that assumption every duplicate ends up with the value it would
have computed for itself. Rows that share a key but disagree on their source columns
are malformed input, not a case this path reconciles - it neither detects nor
de-duplicates them, because doing so would cost a shuffle on every run to guard
against data the producer is not supposed to emit.

The table is assumed COMPLETE AND QUIESCENT for the duration of a run: no producer
append, no second embedding writer. Under that contract the pinned version is not a
concurrency mechanism - it is there because every worker must scan one schema, and
because Lance needs a base version to build a valid ``Update``.

A commit that loses a race is nonetheless REBASED onto the newer manifest, up to
``LanceDataset.commit``'s own default retry count, and that is desirable here rather
than merely tolerated: this transaction names specific fragments and specific field
ids, so a producer append of NEW fragments cannot conflict with it. The replayed
``Update`` still lands, the appended rows keep their NULL group values, and the next
run fills them through the ordinary pending path. Only a genuine conflict - a second
embedding writer on the same group, which the quiescence contract forbids - exhausts
the rebases, and that surfaces as a ``ValueError`` rather than a merged result.
"""

import json
from collections.abc import Iterator
from typing import Any

import attrs
import lance
import pyarrow as pa
import ray
from lance.fragment import FragmentMetadata, LanceFragment
from loguru import logger
from ray.data import ActorPoolStrategy

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.schemas import KEY_COLUMN, EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.columns import pending_filter
from cosmos_curator.next.recipes.embeddings.modalities import ModalityFill, ModalityResult

# The single column a worker is allowed to send the driver: one JSON payload per
# touched fragment, carrying its identity, its serialized updated metadata, the
# rebound field ids and two counters. A lone string column is the mechanism that
# keeps driver memory O(touched fragments) rather than O(rows) - it cannot be
# widened into a full-corpus vector gather without changing its type.
_RESULT_COLUMN = "result_json"
_RESULT_SCHEMA: pa.Schema = pa.schema([pa.field(_RESULT_COLUMN, pa.large_string())])

# Keys that DISCRIMINATE the two FAILURE payload shapes a worker can emit; a
# fragment that was written carries the metadata keys instead. They are separate
# keys rather than one flag because the driver must react to them in opposite ways:
# ``_ERROR_KEY`` marks a fragment that was skipped and the run carries on without
# it (``_collect`` partitions on its presence), while ``_FATAL_KEY`` marks a
# violated fill contract that stops the run (``_run_workers`` raises on it).
_ERROR_KEY = "error"
_FATAL_KEY = "fatal"

# The column ``from_items`` carries into the worker: one fragment id per work item.
_FRAGMENT_ID_COLUMN = "fragment_id"

# Ray Data work-item batch size. Pinned to one fragment per ``__call__`` so a worker's
# memory is bounded by one fragment's in-flight batch. Distinct from
# ``WorkerResources.scan_batch_size``, which is how many ROWS of that fragment reach
# the embedder at a time. It does NOT set the run's concurrency: that ceiling belongs
# to the block count below.
_FRAGMENTS_PER_TASK = 1

# Ray Data's own block ceiling for ``from_items``, and therefore this fill's
# concurrency ceiling: ``min(len(fragment_ids), _MAX_FROM_ITEMS_BLOCKS)``. Above it one
# block carries about ceil(N / 200) fragment ids that a single batch stream walks
# serially, so a one-fragment table runs on one worker however large the actor pool is.
# Measured on ray 2.55.1 and NOT re-verified on the pinned 2.56.0 - it is a Ray Data
# autodetect default, so Ray is precisely the axis that could move it. Every run logs
# the ceiling it derived; the lever that lifts it, and why this recipe declines to pull
# it, are in ``docs/curator/design/curator-next-embeddings.md`` section 7.
_MAX_FROM_ITEMS_BLOCKS = 200


class FillContractError(ValueError):
    """A failure the run must stop for, rather than skip past. Two kinds qualify.

    Scoped to what the FILL detects, so the two are a violated invariant - a wrong
    row count or schema from the embedder, a fragment absent at the pinned version,
    none of which a retry fixes - and a total outage: nothing written while at least
    one fragment failed. Their responses are opposite, re-run the outage versus fix
    the inputs behind the violation, and that guidance lives in each raise's
    MESSAGE, not in the type (``docs/curator/design/curator-next-embeddings.md``
    section 7).

    A ``ValueError`` subclass so it joins the recipe's other pre-commit refusals as
    one thing a caller handles: the run stopped before anything was published. It
    always reaches the CALLER from the driver - a worker raises it internally, but
    as a signal its own handler turns into a FATAL payload, because the raise would
    otherwise arrive at the driver as a Ray wrapper of another type.
    """


@attrs.define
class _FillCounters:
    """Mutable row/fill tallies accumulated while a fragment's update stream is consumed.

    ``update_columns`` pulls the update batches itself, so the counts are only final
    once it returns; a small mutable holder lets the generator that produces those
    batches report what it saw without buffering the batches to count them first.

    Attributes:
        rows: Pending rows the fragment scan yielded (this fragment's ``selected``).
        filled: Of those, the rows whose primary vector came out non-NULL.
        contract_error: Message of a fill-contract violation the generator raised,
            recorded because the raise itself does not survive the reader boundary.

    """

    rows: int = 0
    filled: int = 0
    contract_error: str | None = None


class _FragmentWorker:
    """Ray Data callable class filling one fragment's column group per invocation.

    Constructed once per actor from picklable arguments only (no dataset handle, no
    loaded model): it builds the embedder in ``__init__`` so the weights load once per
    actor rather than once per fragment, and opens the pinned dataset lazily on first
    use so nothing live crosses the pickle boundary.

    Every fragment it touches is read at the run's pinned ``read_version``, so every
    worker sees one schema and one row set for the whole run.
    """

    def __init__(  # noqa: PLR0913 -- every argument is required worker state; grouping them would just hide the same set
        self,
        *,
        uri: str,
        group: EmbeddingColumnGroup,
        read_version: int,
        storage_options: dict[str, str] | None,
        source_columns: tuple[str, ...],
        row_filter: str | None,
        scan_batch_size: int,
        embedder_cls: type,
        embedder_kwargs: dict[str, Any],
    ) -> None:
        """Store the fill parameters and build this actor's embedder once."""
        self._uri = uri
        self._group = group
        self._read_version = read_version
        self._storage_options = storage_options
        self._source_columns = source_columns
        self._row_filter = row_filter
        self._scan_batch_size = scan_batch_size
        self._embedder = embedder_cls(**embedder_kwargs)
        self._dataset: lance.LanceDataset | None = None
        self._update_schema: pa.Schema | None = None

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Fill every fragment named in ``batch``, returning one result row per touched fragment.

        A fragment with no pending rows contributes no result row, so the driver's
        collector sees only fragments that were either written or skipped, and needs
        no sentinel filtering.
        """
        payloads = [self._fill_fragment(int(fragment_id)) for fragment_id in batch.column(_FRAGMENT_ID_COLUMN)]
        touched = [payload for payload in payloads if payload is not None]
        return pa.table({_RESULT_COLUMN: pa.array(touched, type=pa.large_string())}, schema=_RESULT_SCHEMA)

    def _fill_fragment(self, fragment_id: int) -> str | None:
        """Fill one fragment, reporting a failure as a payload; ``None`` if nothing was pending.

        Two failure shapes, because the driver must react to them in opposite ways.
        An ordinary fault becomes a SKIP, which is safe rather than lossy because a
        failed fragment was never written: any rows it still owed stay pending, so
        the ordinary pending predicate re-selects them on the next run, and one flaky
        fragment no longer discards every other fragment's completed work. A violated
        fill contract becomes a FATAL payload the driver raises on, because no retry
        can fix it.

        Neither leaves the worker as an exception, so the two must be distinguishable
        by payload: Ray Data does not carry the original object across the task
        boundary, so a raise here would reach the driver as an untyped wrapper. Ray
        also sees no task failure, so its own retry never fires - an intra-run retry
        becomes an inter-run one.
        """
        try:
            return self._write_fragment(fragment_id)
        except FillContractError as e:
            logger.error(
                f"embedding group {self._group.name!r}: fragment {fragment_id} of {self._uri} violated a fill "
                f"contract ({e}); the run stops without committing"
            )
            return json.dumps({"fragment_id": fragment_id, _FATAL_KEY: str(e)})
        except Exception as e:  # noqa: BLE001 - any fault must cost one fragment, not the modality
            logger.warning(
                f"embedding group {self._group.name!r}: fragment {fragment_id} of {self._uri} failed and is "
                f"SKIPPED ({e!r}); it was not written, so any rows it still owed stay pending and the next "
                f"run's pending filter re-selects them"
            )
            return json.dumps({"fragment_id": fragment_id, _ERROR_KEY: repr(e)})

    def _write_fragment(self, fragment_id: int) -> str | None:
        """Compute and write one fragment's group, or return ``None`` if nothing was pending.

        The scanner is probed for a first batch BEFORE ``update_columns`` is called:
        with no pending row the fragment must be left byte-for-byte untouched, but
        ``update_columns`` would already have written an (empty) column file by the
        time an exhausted reader revealed there was no work.

        Raises:
            FillContractError: If the fragment id is absent at the pinned version, so
                the fragment list the run derived and the table disagree, or if the
                update stream reported a violated fill contract.

        """
        dataset = self._open()
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"fragment {fragment_id} is not present in {self._uri} at version {self._read_version}"
            raise FillContractError(msg)
        counters = _FillCounters()
        batches = self._update_batches(fragment, counters)
        first = next(batches, None)
        if first is None:
            return None
        schema = self._stored_update_schema()

        def remaining() -> Iterator[pa.RecordBatch]:
            yield first
            yield from batches

        reader = pa.RecordBatchReader.from_batches(schema, remaining())
        try:
            metadata, modified_field_ids = fragment.update_columns(reader, left_on=KEY_COLUMN, right_on=KEY_COLUMN)
        except Exception as e:
            # Only the first batch is embedded on this stack; Lance pulls the rest
            # through the Arrow C data interface, which carries an error MESSAGE but
            # not the Python exception object - a contract violation raised there
            # re-emerges as a bare RuntimeError. Re-raising it from the recorded
            # message is what keeps the fatal-vs-skip decision a property of the
            # violation rather than of which batch it happened on.
            if counters.contract_error is not None:
                raise FillContractError(counters.contract_error) from e
            raise
        # The worker-to-driver SUCCESS payload (the skip path emits the error shape
        # instead). Its keys ARE the collector's contract, so a change here has to
        # land in ``_collect`` in the same edit. The metadata is
        # embedded as the exact string ``FragmentMetadata.from_json`` expects, rather
        # than as a nested object, so the round-trip cannot lose a field to re-encoding.
        return json.dumps(
            {
                "fragment_id": fragment_id,
                "metadata_json": json.dumps(metadata.to_json()),
                "modified_field_ids": [int(field_id) for field_id in modified_field_ids],
                "rows": counters.rows,
                "filled": counters.filled,
            }
        )

    def _update_batches(self, fragment: LanceFragment, counters: _FillCounters) -> Iterator[pa.RecordBatch]:
        """Yield ``clip_id`` + group batches for one fragment's pending rows, tallying as it goes.

        Streams rather than buffering: each scanned batch is embedded and handed
        straight to ``update_columns``, so peak memory is one batch of vectors, not one
        fragment's worth.

        A contract violation is recorded on ``counters`` on its way out, because every
        batch but the first is pulled from beneath a reader boundary that keeps the
        message and discards the type; the caller re-raises it from there.
        """
        scanner = fragment.scanner(
            columns=list(self._source_columns),
            filter=self._row_filter,
            batch_size=self._scan_batch_size,
        )
        schema = self._stored_update_schema()
        for record_batch in scanner.to_batches():
            if not record_batch.num_rows:
                continue
            source = pa.Table.from_batches([record_batch])
            try:
                update = self._embed(source, schema)
            except FillContractError as e:
                counters.contract_error = str(e)
                raise
            counters.rows += source.num_rows
            counters.filled += update.num_rows - update.column(self._group.primary_vector).null_count
            yield from update.to_batches()

    def _embed(self, source: pa.Table, schema: pa.Schema) -> pa.Table:
        """Embed one scanned batch and attach ``clip_id``, casting to the stored types.

        The key is attached POSITIONALLY, which is sound only because the embedder is
        cardinality- and order-preserving; the row-count check makes a violation of
        that contract fail loudly here instead of silently pairing vectors with the
        wrong clips.

        Raises:
            FillContractError: If the embedder did not return exactly one row per
                input row, or returned columns that do not cast to the stored
                schema. Deliberately NOT failures a fragment is skipped for: both
                mean the embedder is broken, so every fragment would fail the same
                way and the run would report an empty group as a success.

        """
        computed = self._embedder(source)
        if computed.num_rows != source.num_rows:
            msg = (
                f"embedder for group {self._group.name!r} returned {computed.num_rows} row(s) for a "
                f"{source.num_rows}-row batch; the contract is one output row per input row, in input order"
            )
            raise FillContractError(msg)
        keyed = computed.add_column(0, KEY_COLUMN, source.column(KEY_COLUMN))
        try:
            return keyed.cast(schema)
        except (ValueError, pa.ArrowTypeError, pa.ArrowNotImplementedError) as e:
            # All three arms were measured on pyarrow 24: a missing, renamed, extra
            # or REORDERED column raises a plain ValueError, an uncastable dtype
            # raises ArrowNotImplementedError, and a wrong vector width raises
            # ArrowTypeError. The common base pa.ArrowException would be shorter but
            # also catches ArrowMemoryError, and an allocation failure is transient -
            # promoting it here would discard every other fragment's work.
            msg = (
                f"embedder for group {self._group.name!r} returned columns that do not match the stored "
                f"schema: {e}. Expected {schema.to_string(show_field_metadata=False)}, "
                f"got {keyed.schema.to_string(show_field_metadata=False)}"
            )
            raise FillContractError(msg) from e

    def _stored_update_schema(self) -> pa.Schema:
        """Return ``clip_id`` + the group's fields, typed exactly as the table stores them.

        Taken from the dataset schema rather than rebuilt, so the update table is cast
        to the stored types (``clip_id`` may be ``large_string``) and ``update_columns``
        never has to coerce.
        """
        if self._update_schema is None:
            schema = self._open().schema
            self._update_schema = pa.schema([schema.field(name) for name in (KEY_COLUMN, *self._group.field_names)])
        return self._update_schema

    def _open(self) -> lance.LanceDataset:
        """Open (once per actor) the table pinned at the run's read version."""
        if self._dataset is None:
            self._dataset = lance.dataset(
                self._uri,
                version=self._read_version,
                storage_options=self._storage_options,
            )
        return self._dataset


def fill_embedding_group(
    dataset: lance.LanceDataset,
    fill: ModalityFill,
    *,
    storage_options: dict[str, str] | None,
    max_fragments: int | None = None,
) -> ModalityResult:
    """Fill one modality's pending rows fragment by fragment and publish them in one commit.

    A fragment that fails is SKIPPED rather than failing the modality: it is left
    unwritten, so any rows it still owed stay pending and the next run's pending
    predicate re-selects them. The skip count reaches the caller on the result, and
    a run that skipped a fragment without writing ANY of them is refused outright,
    because its counts are indistinguishable from an honestly empty run's.

    Args:
        dataset: The clips table, open at the version the fill reads and commits against.
        fill: The modality's embedder, source columns, group and actor shape.
        storage_options: Lance credentials for the workers and for the commit.
        max_fragments: Visit at most this many fragments (``None`` visits all of them).

    Returns:
        The run's accounting over the fragments that were written;
        ``committed_version`` is ``None`` when nothing was committed.

    Raises:
        FillContractError: If the run's shared scan inputs do not resolve against
            the table, a worker reported a violated fill invariant, or no fragment
            was written while at least one failed.
        ValueError: If two workers report the same fragment, or the commit fails.

    """
    version = int(dataset.version)
    # Listing fragments is a manifest read, not a scan. Slicing by ``max_fragments``
    # bounds a first run against a large table; ``[:None]`` is the identity slice.
    fragment_ids = [fragment.fragment_id for fragment in dataset.get_fragments()][:max_fragments]
    row_filter = pending_filter(fill.group, fill.applicability_filter)
    _check_fill_preconditions(dataset, fill, row_filter=row_filter)
    if not fragment_ids:
        logger.info(f"embedding group {fill.group.name!r}: table has no fragments to fill")
        return ModalityResult.nothing_to_do(fill.modality)

    # The number that actually caps this run, reported rather than inferred: a
    # fragment-poor table starves an actor pool of any size, and nothing else in the
    # log distinguishes that from a cluster too small to place the actors.
    logger.info(
        f"embedding group {fill.group.name!r}: visiting {len(fragment_ids)} fragment(s) of {dataset.uri} at "
        f"v{version}; at most {min(len(fragment_ids), _MAX_FROM_ITEMS_BLOCKS)} run concurrently, whatever the "
        f"cluster offers"
    )
    payloads = _run_workers(
        dataset, fill, fragment_ids=fragment_ids, row_filter=row_filter, storage_options=storage_options
    )
    collected = _collect(payloads, fill.group)
    # Total outage (design doc section 7). Checked before the log lines below so a
    # refused run is reported once, as the refusal, rather than as a reassuring
    # warning followed by a failure. The message claims no lost rows deliberately:
    # a fragment can fault before its scan could report whether it held any.
    if not collected.fragments and collected.skipped_fragments:
        msg = (
            f"embedding group {fill.group.name!r}: no fragment was written while {collected.skipped_fragments} "
            f"of the {len(fragment_ids)} fragment(s) visited failed, so nothing was committed and the group is "
            f"untouched. Whether those fragments held pending rows is not knowable from here. Re-run first - a "
            f"transient storage fault fails fragments alike and there are no in-run retries - then read the "
            f"per-fragment WARNING lines if it persists."
        )
        raise FillContractError(msg)
    if not collected.fragments:
        logger.info(
            f"embedding group {fill.group.name!r}: nothing to commit across {len(fragment_ids)} fragment(s); "
            f"none of them had a pending row"
        )
        return ModalityResult.nothing_to_do(fill.modality)
    if collected.skipped_fragments:
        logger.warning(
            f"embedding group {fill.group.name!r}: {collected.skipped_fragments} of {len(fragment_ids)} "
            f"fragment(s) failed and were SKIPPED (see the per-fragment warnings for the causes); they were "
            f"not written, so any rows they still owed stay pending and the next run re-selects them"
        )

    committed_version = _commit(
        dataset, fill.group, list(collected.fragments.values()), collected.field_ids, storage_options=storage_options
    )
    logger.info(
        f"embedding group {fill.group.name!r}: filled {collected.filled}/{collected.selected} row(s) across "
        f"{len(collected.fragments)} fragment(s), {collected.skipped_fragments} skipped; "
        f"v{version} -> v{committed_version}"
    )
    return ModalityResult(
        modality=fill.modality,
        selected=collected.selected,
        filled=collected.filled,
        skipped_fragments=collected.skipped_fragments,
        committed_version=committed_version,
    )


def _check_fill_preconditions(dataset: lance.LanceDataset, fill: ModalityFill, *, row_filter: str) -> None:
    """Verify the scan inputs every fragment shares, before any worker is started.

    The projection and the row filter are properties of the RUN, not of a
    fragment, so a broken one fails on every fragment alike. Left to the workers
    it arrives as a fault per fragment, and because a faulting fragment is
    skipped, the modality would report a successful run that embedded nothing.

    ``explain_plan`` is what makes checking this affordable: it resolves and
    optimizes the scan - which is where an unknown column or an unparsable
    predicate is rejected - without reading a row. A re-run with nothing pending
    therefore still reads nothing, whereas probing for a first matching row would
    scan the whole table to discover the table is already complete.

    Raises:
        FillContractError: If ``clip_id`` is absent from the projection, or the
            projection and filter do not resolve against the table.

    """
    if KEY_COLUMN not in fill.source_columns:
        msg = (
            f"embedding group {fill.group.name!r}: source columns {list(fill.source_columns)} omit "
            f"{KEY_COLUMN!r}; the embedder never reads it, but the fill attaches it positionally and "
            f"update_columns joins on it, so it must be projected"
        )
        raise FillContractError(msg)
    try:
        dataset.scanner(columns=list(fill.source_columns), filter=row_filter).explain_plan()
    except ValueError as e:
        # Lance raises ValueError for an unparsable predicate, a predicate naming
        # an absent column, and a projection naming an absent column alike. An IO
        # fault would be an OSError and is deliberately NOT caught here: it is not
        # a statement about the run's inputs, so it must not be relabelled as a
        # violated contract.
        msg = (
            f"embedding group {fill.group.name!r}: the per-fragment scan does not resolve against "
            f"{dataset.uri} at v{int(dataset.version)} (columns={list(fill.source_columns)}, "
            f"filter={row_filter!r}): {e}"
        )
        raise FillContractError(msg) from e


def _run_workers(
    dataset: lance.LanceDataset,
    fill: ModalityFill,
    *,
    fragment_ids: list[int],
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Run the actor pool over the given fragment ids and return the decoded payloads.

    The pool floors at one actor and pins no maximum, so it is bounded by two things
    it does not measure: the free resource slices Ray can place an actor on, and the
    work-item count (see ``_MAX_FROM_ITEMS_BLOCKS``).
    ``scheduling_strategy="DEFAULT"`` packs fractional-GPU actors rather than
    spreading them.

    Raises:
        FillContractError: If a worker reported a violated fill invariant. Raised on
            the driver because a worker cannot: Ray Data replaces whatever a UDF
            raises with a wrapper that is no longer an instance of its own cause.
            Raised while the results are still streaming in, so the fragments not yet
            started are abandoned instead of computing a result nothing will commit.

    """
    resources = fill.resources
    items = [{_FRAGMENT_ID_COLUMN: fragment_id} for fragment_id in fragment_ids]
    results = ray.data.from_items(items).map_batches(
        _FragmentWorker,
        fn_constructor_kwargs={
            # Picklable values only: the worker re-opens the table at the pinned
            # version itself rather than receiving this driver-side handle.
            "uri": dataset.uri,
            "group": fill.group,
            "read_version": int(dataset.version),
            "storage_options": storage_options,
            "source_columns": fill.source_columns,
            "row_filter": row_filter,
            "scan_batch_size": resources.scan_batch_size,
            "embedder_cls": fill.embedder_cls,
            "embedder_kwargs": fill.embedder_kwargs,
        },
        batch_format="pyarrow",
        batch_size=_FRAGMENTS_PER_TASK,
        compute=ActorPoolStrategy(min_size=1),
        # Pinned for every modality, GPU and CPU alike: the worker needs the
        # interpreter that holds both the model stack and pylance, and an unpinned
        # worker would silently inherit whichever env Ray happened to start.
        runtime_env=ray_data_gpu_runtime_env(resources.env_name),
        scheduling_strategy="DEFAULT",
        # Ray declares all three as optional and treats an explicit ``None`` exactly
        # as an omitted kwarg, so an unset resource is passed straight through.
        num_gpus=resources.num_gpus,
        num_cpus=resources.num_cpus,
        memory=resources.memory_bytes,
    )
    payloads: list[dict[str, Any]] = []
    for row in results.iter_rows():
        payload = json.loads(str(row[_RESULT_COLUMN]))
        detail = payload.get(_FATAL_KEY)
        if detail is not None:
            msg = (
                f"embedding group {fill.group.name!r}: fragment {payload['fragment_id']} violated a fill "
                f"contract, so the run stops without committing: {detail}"
            )
            raise FillContractError(msg)
        payloads.append(payload)
    return payloads


@attrs.frozen
class _CollectedFill:
    """The worker payloads reduced to what the commit and the summary need.

    Attributes:
        fragments: Updated metadata of every WRITTEN fragment, keyed by fragment id.
        field_ids: Union of the field ids those writes rebound.
        selected: Pending rows the written fragments scanned.
        filled: Of those, the rows whose primary vector came out non-NULL.
        skipped_fragments: Fragments that failed and were left unwritten. Their rows
            are not counted in ``selected``: nothing was written for them, so from
            the table's point of view the run never visited them.

    """

    fragments: dict[int, FragmentMetadata]
    field_ids: list[int]
    selected: int
    filled: int
    skipped_fragments: int


def _collect(payloads: list[dict[str, Any]], group: EmbeddingColumnGroup) -> _CollectedFill:
    """Partition the worker payloads into the commit inputs, the counters, and the skips.

    Sees only the written and skipped shapes: a FATAL payload never reaches here,
    because ``_run_workers`` raises on it rather than returning it.

    Raises:
        ValueError: If two written payloads claim the same fragment. Each fragment is
            a single work item, so a duplicate means a retried task's write was also
            counted; committing two metadata versions of one fragment would make the
            commit's outcome depend on ordering.

    """
    written = [payload for payload in payloads if _ERROR_KEY not in payload]
    fragments = {
        int(payload["fragment_id"]): FragmentMetadata.from_json(str(payload["metadata_json"])) for payload in written
    }
    if len(fragments) != len(written):
        msg = (
            f"embedding group {group.name!r}: {len(written)} fill result(s) named only {len(fragments)} distinct "
            f"fragment(s); refusing to commit two metadata versions of one fragment"
        )
        raise ValueError(msg)
    field_ids = {int(field_id) for payload in written for field_id in payload["modified_field_ids"]}
    return _CollectedFill(
        fragments=fragments,
        field_ids=sorted(field_ids),
        selected=sum(int(payload["rows"]) for payload in written),
        filled=sum(int(payload["filled"]) for payload in written),
        skipped_fragments=len(payloads) - len(written),
    )


def _commit(
    dataset: lance.LanceDataset,
    group: EmbeddingColumnGroup,
    fragments: list[FragmentMetadata],
    field_ids: list[int],
    *,
    storage_options: dict[str, str] | None,
) -> int:
    """Commit the touched fragments' rebound field ids as one ``Update`` transaction.

    ``LanceOperation.Update`` accepts a PARTIAL fragment list, so only the fragments
    that actually changed are named and every other fragment keeps its existing
    metadata. ``fields_modified`` scopes the rebinding to this group's field ids, so
    every column outside the group stays bound to the file it already had.
    ``transaction_properties`` tags the commit so it is attributable in the table's
    version history.

    Raises:
        ValueError: If the commit fails, so the CLI's ``except ValueError`` handler
            reports a cause instead of a traceback.

    """
    read_version = int(dataset.version)
    transaction = lance.Transaction(
        read_version=read_version,
        operation=lance.LanceOperation.Update(
            updated_fragments=fragments,
            fields_modified=field_ids,
        ),
        transaction_properties={
            "kind": "curator-next-embedding",
            "group": group.name,
        },
    )
    try:
        committed = lance.LanceDataset.commit(dataset.uri, transaction, storage_options=storage_options)
    except (OSError, RuntimeError) as e:
        # Lance routes only IO faults to OSError; its commit-conflict and
        # write-contention variants all fall through to RuntimeError, so both must be
        # caught to name a cause rather than surface a bare traceback.
        msg = f"embedding group {group.name!r}: commit against v{read_version} of {dataset.uri} failed ({e})"
        raise ValueError(msg) from e
    return int(committed.version)
