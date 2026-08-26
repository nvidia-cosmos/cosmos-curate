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

"""Durable Lance store for data-integrity measurements and evaluations.

Writes what a run learned so it never has to be learned again: the facts go into
per-metric datasets, the verdicts into ``evaluation.lance``, and every stream --
including one that could not be opened -- into ``stream.lance``. Tightening a
threshold later re-judges the stored facts (see :mod:`.reevaluate`) instead of
re-reading a single byte of source data.

Datasets are **append-only**. A re-run adds rows rather than replacing them, so the
history stays auditable; readers that want current state ask for the newest row per
key, which every ``read_*`` function here does by default. The keys are
``stream_id`` for streams and metrics, and ``(stream_id, metric_name, policy_id)``
for evaluations, ordered by ``created_at`` then ``run_id`` -- the tie-break matters
because every row one invocation writes shares a timestamp, so timestamps alone
give no total order.

A run spans several datasets, so it needs a commit point: ``run.lance`` gets its row
only once every other write has landed, and readers ignore rows from runs that never
got that far. A write killed halfway therefore leaves rows that are present but not
believed, instead of a half-finished run passing itself off as the newest state.

The full schema rationale lives in
``docs/curator/design/data-integrity-store-schema.md``.

Imported lazily by the CLIs, only when ``--store-path`` is given, so a plain check
never pays for ``lance`` / ``pyarrow``.
"""

import datetime
import hashlib
import json
import pathlib
import uuid
from collections.abc import Mapping

import attrs
import lance  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.instruments import (
    INSTRUMENTS,
    InstrumentSpec,
    Thresholds,
    instrument,
    instrument_versions,
)
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    CheckStatus,
    OverallStatus,
    StreamResult,
)
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    CloudObjectStat,
    get_cloud_object_stat,
    get_cloud_text,
    get_lance_storage_options,
    is_cloud_uri,
    is_s3_uri,
    put_cloud_text,
)
from cosmos_curator.next.recipes.data_integrity import store_schema

#: Newest-first ordering applied before de-duplication. ``run_id`` is a UUID so its
#: descending order is arbitrary -- but it is deterministic, which is the point: two
#: readers resolving the same data must land on the same row.
_ORDER_COLUMNS = ("created_at", "run_id")


@attrs.define(frozen=True)
class ContentIdentity:
    """What the bytes behind a source looked like when we measured them.

    The staleness signal neither ``instrument_version`` nor the dataset schema can
    see: unchanged code, replaced data. See :class:`CloudObjectStat` for why the
    ETag is a change token rather than a checksum.
    """

    etag: str | None = None
    size_bytes: int | None = None
    last_modified: datetime.datetime | None = None


def new_run_id() -> str:
    """Mint the id shared by every row one invocation writes."""
    return uuid.uuid4().hex


def thresholds_json(thresholds: Thresholds) -> str:
    """Serialise a policy so a stored verdict explains itself without a lookup."""
    return json.dumps(attrs.asdict(thresholds), sort_keys=True)


def policy_id(thresholds: Thresholds) -> str:
    """Derive a stable id for one policy: identical thresholds collide on purpose.

    Re-running under the same policy should land on the same ``policy_id`` so the
    rows supersede each other, while a genuinely different policy opens a new
    generation of verdicts alongside the old ones.
    """
    digest = hashlib.sha256(thresholds_json(thresholds).encode()).hexdigest()
    return f"p_{digest[:16]}"


def join(root: str, name: str) -> str:
    """Join a store-relative name onto the root, for local paths and URIs alike."""
    return f"{root.rstrip('/')}/{name}"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def content_identity(
    source: str,
    *,
    stat: CloudObjectStat | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> ContentIdentity:
    """Resolve the content identity of one source, best-effort.

    Cloud sources go through the same single ``HEAD`` that the progress display
    issues; local files through one ``stat``. Anything unreadable yields an empty
    identity rather than raising -- a store write must not fail over provenance.

    ``stat`` short-circuits the cloud lookup with a response the caller already has,
    which is how the session CLI keeps the total at one ``HEAD`` per stream. An empty
    stat is not such a response -- it is what a failed lookup returns -- so it falls
    through to a fresh attempt rather than being recorded as fact.
    """
    if is_cloud_uri(source):
        resolved = (
            stat
            if stat is not None and not stat.is_empty
            else get_cloud_object_stat(
                source,
                s3_profile_name=s3_profile_name,
                azure_profile_name=azure_profile_name,
                endpoint_url=endpoint_url,
            )
        )
        return ContentIdentity(etag=resolved.etag, size_bytes=resolved.size_bytes, last_modified=resolved.last_modified)
    try:
        info = pathlib.Path(source).stat()
    except OSError:
        return ContentIdentity()
    return ContentIdentity(
        size_bytes=info.st_size,
        last_modified=datetime.datetime.fromtimestamp(info.st_mtime, tz=datetime.UTC),
    )


def stream_key(stream: StreamResult) -> str:
    """Derive the dedup key of one stream, selector included.

    One function so the ``stream.lance`` row and the metric / evaluation rows that
    reference it cannot disagree about what the key is.
    """
    return identity.stream_id(
        stream.source,
        selector_type=stream.selector_type,
        selector_value=stream.selector_value,
    )


def stream_row(  # noqa: PLR0913 -- one argument per fact the row records
    stream: StreamResult,
    *,
    stream_id: str,
    run_id: str,
    created_at: datetime.datetime,
    session_path: str | None,
    content: ContentIdentity,
) -> dict[str, object]:
    """Build the ``stream.lance`` row for one stream, errored or not."""
    return {
        "stream_id": stream_id,
        "run_id": run_id,
        "created_at": created_at,
        "session_id": identity.session_id(session_path),
        "session_path": session_path,
        "locator_namespace": identity.locator_namespace(stream.source),
        "source": stream.source,
        "relative_key": identity.relative_key(session_path, stream.source),
        "selector_type": stream.selector_type,
        "selector_value": stream.selector_value,
        "content_etag": content.etag,
        "content_size_bytes": content.size_bytes,
        "content_last_modified": content.last_modified,
        "codec_name": stream.codec_name,
        "has_bframes": stream.has_bframes,
        "num_samples": stream.num_samples,
        "start_ns": stream.start_ns,
        "end_ns": stream.end_ns,
        "expected_hz": stream.expected_hz,
        "expected_hz_source": (stream.expected_hz_source.value if stream.expected_hz_source is not None else None),
        "error": stream.error,
    }


def measurement_row(  # noqa: PLR0913 -- a metric row is its identity prefix, all independent
    spec: InstrumentSpec,
    result: CheckResult,
    *,
    stream_id: str,
    source: str,
    run_id: str,
    created_at: datetime.datetime,
    session_path: str | None,
) -> dict[str, object]:
    """Build one metric dataset's row, including the "never ran" shape.

    ``raw_measurement is None`` means the metric was never constructed (no usable
    expected rate), which is stored as ``is_defined = null`` with every metric
    column null -- deliberately a row rather than an absence, so "we considered this
    metric and it could not run" stays distinguishable from "this metric was not
    part of the run".
    """
    measurement = result.raw_measurement
    row: dict[str, object] = {
        "stream_id": stream_id,
        "run_id": run_id,
        "created_at": created_at,
        "session_path": session_path,
        "source": source,
        "instrument_version": spec.version,
        "is_defined": None if measurement is None else measurement.is_defined,
    }
    if measurement is None:
        row.update(dict.fromkeys(spec.field_names))
    else:
        row.update(spec.to_row(measurement))
    return row


def build_evaluation_row(  # noqa: PLR0913 -- a verdict row is many independent provenance fields
    spec: InstrumentSpec,
    result: CheckResult,
    *,
    stream_id: str,
    source: str,
    run_id: str,
    measurement_run_id: str,
    created_at: datetime.datetime,
    session_path: str | None,
    thresholds: Thresholds,
    instrument_version: int | None = None,
) -> dict[str, object]:
    """Build one ``evaluation.lance`` row.

    ``margin`` and ``threshold`` are null together on a SKIPPED row, which is what
    makes it structurally impossible to store a verdict for something that was never
    judged.

    ``instrument_version`` records which measuring code produced the *input*, so
    re-evaluation must pass the version off the stored measurement rather than let it
    default to the registry's current one -- a re-judge does not re-measure, and
    claiming otherwise would hide that the facts came from older code.
    """
    judged = result.status is not CheckStatus.SKIPPED
    margin = result.evaluation.get("margin") if result.evaluation is not None else None
    return {
        "stream_id": stream_id,
        "run_id": run_id,
        "measurement_run_id": measurement_run_id,
        "policy_id": policy_id(thresholds),
        "thresholds_json": thresholds_json(thresholds),
        "created_at": created_at,
        "session_path": session_path,
        "source": source,
        "metric_name": spec.name,
        "check_status": result.status.value,
        "margin": float(margin) if isinstance(margin, (int, float)) else None,
        "threshold": float(spec.threshold(thresholds)) if judged else None,
        "reason": result.reason,
        "instrument_version": spec.version if instrument_version is None else instrument_version,
    }


def _ensure_local_parent(uri: str) -> None:
    """Create the parent directory of a local dataset path; a no-op for cloud URIs."""
    if not is_cloud_uri(uri):
        pathlib.Path(uri).parent.mkdir(parents=True, exist_ok=True)


def _dataset_exists(uri: str, storage_options: dict[str, str] | None) -> bool:
    try:
        lance.dataset(uri, storage_options=storage_options)
    except (ValueError, FileNotFoundError):
        return False
    return True


def append_rows(
    rows: list[dict[str, object]], uri: str, schema: pa.Schema, storage_options: dict[str, str] | None
) -> None:
    """Append ``rows`` to a Lance dataset, creating it on first write.

    The mode is chosen rather than always passing ``append`` because Lance logs a
    warning when asked to append to a dataset that does not exist yet -- true of
    every first run, which is not something to warn about.
    """
    if not rows:
        return
    _ensure_local_parent(uri)
    table = pa.Table.from_pylist(rows, schema=schema)
    mode = "append" if _dataset_exists(uri, storage_options) else "create"
    lance.write_dataset(table, uri, mode=mode, storage_options=storage_options)


def commit_run(  # noqa: PLR0913 -- a commit records the run it closes, which is many facts
    root: str,
    *,
    run_id: str,
    created_at: datetime.datetime,
    tool: str,
    session_path: str | None,
    thresholds: Thresholds,
    num_streams: int | None,
    storage_options: dict[str, str] | None,
) -> None:
    """Mark ``run_id`` finished by appending its row to ``run.lance``.

    The store's commit point, and the reason it has one: a write touches several
    datasets in sequence, so a process that dies partway through leaves a newer
    ``run_id`` that "latest per key" would otherwise hand back as current state.
    Every ``read_*`` here ignores rows whose run never reached this function, which
    turns a torn write into rows that are merely present rather than believed.

    Must therefore be the last write of a run.
    """
    row = {
        "run_id": run_id,
        "created_at": created_at,
        "committed_at": _utc_now(),
        "tool": tool,
        "session_path": session_path,
        "policy_id": policy_id(thresholds),
        "num_streams": num_streams,
        "store_schema_version": store_schema.STORE_SCHEMA_VERSION,
    }
    append_rows([row], join(root, store_schema.RUN_DATASET), store_schema.RUN_SCHEMA, storage_options)


def completed_runs(
    root: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
    storage_options: dict[str, str] | None = None,
) -> frozenset[str]:
    """Read the ids of every run that finished writing.

    A store with no ``run.lance`` has no finished runs, so it reads as empty rather
    than as "assume everything is fine": the case that produces it is a first run
    that died mid-write, which is exactly the state this is here to catch.
    """
    if storage_options is None:
        storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    uri = join(root, store_schema.RUN_DATASET)
    if not _dataset_exists(uri, storage_options):
        return frozenset()
    table = lance.dataset(uri, storage_options=storage_options).to_table(columns=["run_id"])
    return frozenset(str(value) for value in table.column("run_id").to_pylist())


def write_run(  # noqa: PLR0913 -- a run's provenance is genuinely many independent facts
    root: str,
    streams: list[StreamResult],
    *,
    session_path: str | None,
    thresholds: Thresholds,
    tool: str,
    run_id: str | None = None,
    created_at: datetime.datetime | None = None,
    cloud_stats: Mapping[str, CloudObjectStat] | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> str:
    """Append one run's streams, measurements and evaluations to the store at ``root``.

    The datasets are written first and the run is committed to ``run.lance`` last, so
    a write that dies partway through leaves rows that no reader will believe (see
    :func:`commit_run`). ``manifest.json`` follows the commit: it is a convenience for
    a human reading the directory, not part of the contract.

    Args:
        root: local directory or ``s3://`` prefix holding the store. Created on
            first write.
        streams: the run's per-stream results, errored ones included -- an errored
            stream is the case the store exists to preserve.
        session_path: the session these streams were discovered under, or ``None``
            for a single-stream run. The only structural difference between what the
            two CLIs write.
        thresholds: the policy the verdicts were produced under.
        tool: which CLI wrote this run, recorded in the manifest.
        run_id: id shared by every row written here; minted when omitted.
        created_at: timestamp shared by every row; now (UTC) when omitted.
        cloud_stats: ``HEAD`` responses the caller already has, by source. The
            session CLI collects these for its progress display, so passing them
            keeps the run at one ``HEAD`` per stream; anything absent is resolved
            here.
        s3_profile_name: AWS profile for an ``s3://`` store and for resolving
            content identity.
        azure_profile_name: Azure profile used only for content identity; an
            ``az://`` *store* is rejected (see ``get_lance_storage_options``).
        endpoint_url: S3 endpoint override for S3-compatible stores.

    Returns:
        The ``run_id`` every row was written under.

    """
    run_id = run_id or new_run_id()
    created_at = created_at or _utc_now()
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    known = dict(cloud_stats or {})

    stream_rows: list[dict[str, object]] = []
    measurement_rows: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in INSTRUMENTS}
    evaluation_rows: list[dict[str, object]] = []

    for stream in streams:
        stat = content_identity(
            stream.source,
            stat=known.get(stream.source),
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        )
        key = stream_key(stream)
        stream_rows.append(
            stream_row(
                stream,
                stream_id=key,
                run_id=run_id,
                created_at=created_at,
                session_path=session_path,
                content=stat,
            )
        )
        # An errored stream was never measured, so it has no metric or verdict rows
        # anywhere -- only the stream row above records that it was attempted.
        if stream.error is not None:
            continue
        for result in stream.metrics:
            spec = instrument(result.name)
            measurement_rows[spec.name].append(
                measurement_row(
                    spec,
                    result,
                    stream_id=key,
                    source=stream.source,
                    run_id=run_id,
                    created_at=created_at,
                    session_path=session_path,
                )
            )
            evaluation_rows.append(
                build_evaluation_row(
                    spec,
                    result,
                    stream_id=key,
                    source=stream.source,
                    run_id=run_id,
                    measurement_run_id=run_id,
                    created_at=created_at,
                    session_path=session_path,
                    thresholds=thresholds,
                )
            )

    append_rows(stream_rows, join(root, store_schema.STREAM_DATASET), store_schema.STREAM_SCHEMA, storage_options)
    for spec in INSTRUMENTS:
        append_rows(
            measurement_rows[spec.name],
            join(root, store_schema.metric_dataset_path(spec.name)),
            store_schema.MEASUREMENT_SCHEMAS[spec.name],
            storage_options,
        )
    append_rows(
        evaluation_rows,
        join(root, store_schema.EVALUATION_DATASET),
        store_schema.EVALUATION_SCHEMA,
        storage_options,
    )

    # Last, and deliberately so: until this lands, nothing written above is visible to
    # a reader. See commit_run.
    commit_run(
        root,
        run_id=run_id,
        created_at=created_at,
        tool=tool,
        session_path=session_path,
        thresholds=thresholds,
        num_streams=len(streams),
        storage_options=storage_options,
    )

    write_manifest(
        root,
        run_id=run_id,
        created_at=created_at,
        session_path=session_path,
        thresholds=thresholds,
        tool=tool,
        num_streams=len(streams),
        s3_profile_name=s3_profile_name,
        endpoint_url=endpoint_url,
    )
    return run_id


def write_manifest(  # noqa: PLR0913 -- a manifest records many provenance fields
    root: str,
    *,
    run_id: str,
    created_at: datetime.datetime,
    session_path: str | None,
    thresholds: Thresholds,
    tool: str,
    num_streams: int,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, object]:
    """Write ``manifest.json`` describing the run that just finished, and return it.

    Unlike the datasets, the manifest is a single overwritten document: it describes
    the *most recent* run, and per-run provenance lives in the rows themselves
    (``run_id`` on every one of them).
    """
    manifest: dict[str, object] = {
        "store_schema_version": store_schema.STORE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "tool": tool,
        "session_path": session_path,
        "num_streams": num_streams,
        "policy_id": policy_id(thresholds),
        "thresholds_json": thresholds_json(thresholds),
        "instruments": instrument_versions(),
        "datasets": {
            "stream": store_schema.STREAM_DATASET,
            "evaluation": store_schema.EVALUATION_DATASET,
            "run": store_schema.RUN_DATASET,
            "measurements": {spec.name: store_schema.metric_dataset_path(spec.name) for spec in INSTRUMENTS},
        },
    }
    uri = join(root, store_schema.MANIFEST_NAME)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if is_s3_uri(uri):
        put_cloud_text(uri, payload, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    else:
        path = pathlib.Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    return manifest


def read_manifest(
    root: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, object]:
    """Read ``manifest.json`` from the store at ``root``."""
    uri = join(root, store_schema.MANIFEST_NAME)
    if is_s3_uri(uri):
        payload = get_cloud_text(uri, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    else:
        payload = pathlib.Path(uri).read_text()
    loaded: dict[str, object] = json.loads(payload)
    return loaded


def _latest_per_key(rows: list[dict[str, object]], key: tuple[str, ...]) -> list[dict[str, object]]:
    """Keep the newest row per ``key``, newest decided by ``created_at`` then ``run_id``.

    Output order follows first appearance of each key in the newest-first sort, so
    the result is deterministic for a given set of rows.
    """
    # Sorted on the raw values, not their reprs: each position holds one type across
    # all rows (a datetime, then a str), so the tuple comparison is well-defined.
    ordered = sorted(rows, key=lambda row: tuple(row[name] for name in _ORDER_COLUMNS), reverse=True)  # type: ignore[arg-type,return-value]
    seen: dict[tuple[object, ...], dict[str, object]] = {}
    for row in ordered:
        seen.setdefault(tuple(row[name] for name in key), row)
    return list(seen.values())


def _read(  # noqa: PLR0913 -- a read is its dataset, its key, and how to resolve it
    root: str,
    uri: str,
    *,
    key: tuple[str, ...],
    latest: bool,
    include_incomplete: bool,
    completed: frozenset[str] | None,
    storage_options: dict[str, str] | None,
) -> list[dict[str, object]]:
    """Read one dataset as row dicts, optionally resolved to the newest row per key.

    A dataset that does not exist reads as no rows: a metric that has never been
    written is indistinguishable from one written with nothing to say, and neither is
    an error for a reader.

    Rows from uncommitted runs are dropped *before* the newest-per-key pass, not
    after. The other order would let a torn write hide the good row underneath it:
    the partial row would win the sort, then be filtered out, and the key would come
    back empty.
    """
    if not _dataset_exists(uri, storage_options):
        return []
    rows: list[dict[str, object]] = lance.dataset(uri, storage_options=storage_options).to_table().to_pylist()
    if not include_incomplete:
        if completed is None:
            completed = completed_runs(root, storage_options=storage_options)
        rows = [row for row in rows if str(row["run_id"]) in completed]
    return _latest_per_key(rows, key) if latest else rows


def read_streams(  # noqa: PLR0913 -- read options plus credentials, all independent
    root: str,
    *,
    latest: bool = True,
    include_incomplete: bool = False,
    completed: frozenset[str] | None = None,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> list[dict[str, object]]:
    """Read ``stream.lance``, newest row per ``stream_id`` unless ``latest`` is False.

    Args:
        root: the store root.
        latest: resolve to the newest row per key; ``False`` returns the full history.
        include_incomplete: also return rows from runs that never committed. Off by
            default -- those rows describe a write that died partway through (see
            :func:`commit_run`) -- and available for looking into one.
        completed: the set from :func:`completed_runs`, to save re-reading the ledger
            when several reads happen back to back.
        s3_profile_name: AWS profile for an ``s3://`` store.
        endpoint_url: S3 endpoint override.

    """
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    return _read(
        root,
        join(root, store_schema.STREAM_DATASET),
        key=("stream_id",),
        latest=latest,
        include_incomplete=include_incomplete,
        completed=completed,
        storage_options=storage_options,
    )


def read_measurements(  # noqa: PLR0913 -- read options plus credentials, all independent
    root: str,
    metric_name: str,
    *,
    latest: bool = True,
    include_incomplete: bool = False,
    completed: frozenset[str] | None = None,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> list[dict[str, object]]:
    """Read one metric's dataset, newest row per ``stream_id`` unless ``latest`` is False.

    See :func:`read_streams` for the shared arguments.

    Raises:
        KeyError: if ``metric_name`` is not a registered metric.

    """
    instrument(metric_name)
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    return _read(
        root,
        join(root, store_schema.metric_dataset_path(metric_name)),
        key=("stream_id",),
        latest=latest,
        include_incomplete=include_incomplete,
        completed=completed,
        storage_options=storage_options,
    )


def read_runs(
    root: str,
    *,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> list[dict[str, object]]:
    """Read ``run.lance``: one row per finished run, oldest first.

    No newest-per-key resolution, because a run is written once and never superseded.
    This is the store's history, which ``manifest.json`` cannot be -- that file is
    overwritten and only ever describes the latest run.
    """
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    uri = join(root, store_schema.RUN_DATASET)
    if not _dataset_exists(uri, storage_options):
        return []
    rows: list[dict[str, object]] = lance.dataset(uri, storage_options=storage_options).to_table().to_pylist()
    return sorted(rows, key=lambda row: (row["committed_at"], row["run_id"]))  # type: ignore[arg-type,return-value]


def read_evaluations(  # noqa: PLR0913 -- read options plus credentials, all independent
    root: str,
    *,
    latest: bool = True,
    include_incomplete: bool = False,
    completed: frozenset[str] | None = None,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> list[dict[str, object]]:
    """Read ``evaluation.lance``, newest row per ``(stream_id, metric_name, policy_id)``.

    Keying on ``policy_id`` is what keeps every generation of verdicts readable: a
    re-judge under a tightened policy adds rows beside the originals rather than
    superseding them.

    See :func:`read_streams` for the shared arguments.
    """
    storage_options = get_lance_storage_options(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    return _read(
        root,
        join(root, store_schema.EVALUATION_DATASET),
        key=("stream_id", "metric_name", "policy_id"),
        latest=latest,
        include_incomplete=include_incomplete,
        completed=completed,
        storage_options=storage_options,
    )


def stale_reasons(
    metric_name: str,
    measurement_row: Mapping[str, object],
    *,
    stream_row: Mapping[str, object] | None = None,
    content: ContentIdentity | None = None,
) -> list[str]:
    """Why a stored measurement no longer reflects what we'd measure today.

    Two of the three staleness signals are checkable here: ``instrument_version``
    against the registry, and the recorded content identity against the source as it
    is now. The third -- the dataset schema -- is enforced by Lance at write time and
    so cannot silently disagree.

    The ETag comparison is skipped when either side lacks one (a local file has no
    ETag), falling back to size, which is weaker but still catches a replacement that
    changed length.

    Args:
        metric_name: which metric's row this is.
        measurement_row: a row from :func:`read_measurements`.
        stream_row: the matching row from :func:`read_streams`, which is where the
            recorded content identity lives. Required for the content check.
        content: the source's identity *now*, from :func:`content_identity`. Pass
            ``None`` to skip the content check.

    Returns:
        Zero or more reasons; empty means the measurement still stands.

    """
    reasons: list[str] = []
    spec = instrument(metric_name)
    stored_version = measurement_row.get("instrument_version")
    if stored_version != spec.version:
        reasons.append(f"instrument_version {stored_version} != {spec.version}")
    if content is None or stream_row is None:
        return reasons

    recorded_etag = stream_row.get("content_etag")
    recorded_size = stream_row.get("content_size_bytes")
    if recorded_etag is not None and content.etag is not None:
        if recorded_etag != content.etag:
            reasons.append("content_etag changed")
    elif recorded_size is not None and content.size_bytes != recorded_size:
        reasons.append("content_size_bytes changed")
    return reasons


def session_rollup(
    root: str,
    *,
    policy: str | None = None,
    s3_profile_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, object]:
    """Roll the stored rows back up into a session verdict.

    Reproduces the live report's precedence exactly (see
    :meth:`~cosmos_curator.core.sensors.data_integrity.results.SessionReport.status`):
    ``ERROR`` outranks ``FAIL`` because an errored stream was never measured, so no
    re-evaluation can complete it, and an empty store is ``ERROR`` for the same
    reason. ``SKIPPED`` never fails a stream.

    A stream with no verdict under the requested ``policy`` is ``ERROR`` too, on the
    same grounds: nothing was judged, so there is nothing to call a pass.

    Args:
        root: the store root.
        policy: restrict verdicts to one ``policy_id``; ``None`` uses every stored
            verdict, which is what you want for a store holding a single policy.
            A policy no stream was judged under rolls up to ``ERROR``, not ``PASS``.
        s3_profile_name: AWS profile for an ``s3://`` store.
        endpoint_url: S3 endpoint override.

    """
    # One ledger read shared by both passes, so the two cannot disagree about which
    # runs are committed even if one lands mid-write.
    finished = completed_runs(root, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    streams = read_streams(root, completed=finished, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    evaluations = read_evaluations(root, completed=finished, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
    if policy is not None:
        evaluations = [row for row in evaluations if row["policy_id"] == policy]

    failed: set[object] = {row["stream_id"] for row in evaluations if row["check_status"] == CheckStatus.FAIL.value}
    # Every metric a run considered writes a verdict row, SKIPPED ones included, so a
    # stream missing from this set was not judged under the requested policy at all.
    judged: set[object] = {row["stream_id"] for row in evaluations}
    per_stream: list[dict[str, object]] = []
    for row in sorted(streams, key=lambda item: str(item["source"])):
        if row["error"] is not None:
            status = OverallStatus.ERROR
        elif row["stream_id"] in failed:
            status = OverallStatus.FAIL
        elif row["stream_id"] not in judged:
            # No verdict is not a pass. Reached when the policy asked for does not
            # exist, or when only some streams have been re-judged under it: the
            # session is incomplete, which is the same claim ERROR already makes.
            status = OverallStatus.ERROR
        else:
            status = OverallStatus.PASS
        per_stream.append({"stream_id": row["stream_id"], "source": row["source"], "status": status.value})

    counts = {status.value: 0 for status in OverallStatus}
    for entry in per_stream:
        counts[str(entry["status"])] += 1
    if not per_stream or counts[OverallStatus.ERROR.value]:
        overall = OverallStatus.ERROR
    elif counts[OverallStatus.FAIL.value]:
        overall = OverallStatus.FAIL
    else:
        overall = OverallStatus.PASS

    return {
        "status": overall.value,
        "num_streams": len(per_stream),
        "stream_status_counts": counts,
        "streams": per_stream,
    }
