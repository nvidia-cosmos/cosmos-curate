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

"""What the Ray Data stage runs: measure one whole session.

Deliberately Ray-free -- :func:`check_session` takes and returns plain dicts, so it
is testable by calling it, and :mod:`.pipeline` is the only module that knows about
Ray.

The session is the unit because a session is what a check is about. Today every metric
judges one stream on its own, but the cross-sensor checks this store is heading for --
does the camera timeline agree with the IMU's -- have no per-stream task to run in, and
a whole session in one worker is what they need.

Nothing here raises. Every failure is classified by what it says, and the split is
between the two things a failure can be evidence *of*:

* **The data is bad.** An unreadable or undecodable stream becomes that stream's
  ``error`` row, which is what keeps one corrupt video from failing a run over
  thousands of them.
* **We could not reach the data.** An unlistable session and an unreachable stream are
  both recorded as counts on the session's record rather than as rows, because neither
  learned anything to write down (see
  :class:`~.session_runner.InfrastructureError`). The driver commits what *was*
  measured and then fails the run, so the answer is durable and the gap is still
  reported.

Transient transport failures are retried inside
:func:`~.session_runner.run_one_stream` before either verdict is reached.
"""

import datetime
import pickle

import pyarrow as pa  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.sensors.data_integrity.instruments import instrument
from cosmos_curator.core.sensors.data_integrity.results import CheckStatus, StreamResult
from cosmos_curator.next.recipes.data_integrity import store, store_schema
from cosmos_curator.next.recipes.data_integrity.config import ResolvedDataIntegrityConfig
from cosmos_curator.next.recipes.data_integrity.discovery import discover_streams
from cosmos_curator.next.recipes.data_integrity.session_runner import InfrastructureError, run_one_stream

STREAM_DATASET = store_schema.STREAM_DATASET
EVALUATION_DATASET = store_schema.EVALUATION_DATASET

#: Every dataset one stream can contribute to, keyed by the store-relative name the
#: ``rows`` payload uses. Defined here rather than on the driver because the payload's
#: keys and the schemas they are written under are one contract: a worker that starts
#: emitting a new dataset should not also need the driver to learn about it.
DATASET_SCHEMAS: dict[str, pa.Schema] = {
    STREAM_DATASET: store_schema.STREAM_SCHEMA,
    EVALUATION_DATASET: store_schema.EVALUATION_SCHEMA,
    **{store_schema.metric_dataset_path(name): schema for name, schema in store_schema.MEASUREMENT_SCHEMAS.items()},
}


def discover_session(session_path: str, *, config: ResolvedDataIntegrityConfig) -> list[str]:
    """List the stream sources of one session, in discovery order.

    Args:
        session_path: the session to list.
        config: the resolved config, for the per-session stream cap and credentials.

    Returns:
        The session's stream sources. A session holding no videos yields an empty
        list rather than an error: which sessions are empty is a finding about the
        input, and the caller logs it.

    """
    return discover_streams(
        session_path,
        limit=config.input.limit or 0,
        s3_profile_name=config.execution.s3_profile_name,
        azure_profile_name=config.execution.azure_profile_name,
        endpoint_url=config.execution.endpoint_url,
    )


def build_rows(
    result: StreamResult,
    *,
    config: ResolvedDataIntegrityConfig,
    run_id: str,
    created_at: datetime.datetime,
    session_path: str,
) -> dict[str, list[dict[str, object]]]:
    """Turn one stream's result into the store rows it produces, by dataset name.

    Built in the worker rather than on the driver for two reasons. The provenance
    ``HEAD`` belongs where the stream is already being opened -- ``write_run`` only
    avoids issuing those serially because the session CLI hands it the responses it
    already gathered, and a Ray driver has no such collection. And rows that arrive
    already stamped with their own session leave the driver a plain appender with
    nothing to regroup.

    Args:
        result: the measured stream, errored or not.
        config: the resolved config, for credentials and the threshold policy.
        run_id: the id every row of this invocation shares.
        created_at: the timestamp every row of this invocation shares.
        session_path: the session this stream was discovered under.

    Returns:
        Store-relative dataset name to rows. An errored stream yields only its
        ``stream.lance`` row -- it was never measured, so there is nothing to judge.

    """
    thresholds = config.checks.thresholds.to_thresholds()
    content = store.content_identity(
        result.source,
        s3_profile_name=config.execution.s3_profile_name,
        azure_profile_name=config.execution.azure_profile_name,
        endpoint_url=config.execution.endpoint_url,
    )
    key = store.stream_key(result)
    rows: dict[str, list[dict[str, object]]] = {
        STREAM_DATASET: [
            store.stream_row(
                result,
                stream_id=key,
                run_id=run_id,
                created_at=created_at,
                session_path=session_path,
                content=content,
            )
        ]
    }
    if result.error is not None:
        return rows

    evaluations: list[dict[str, object]] = []
    for check in result.metrics:
        spec = instrument(check.name)
        rows.setdefault(store_schema.metric_dataset_path(spec.name), []).append(
            store.measurement_row(
                spec,
                check,
                stream_id=key,
                source=result.source,
                run_id=run_id,
                created_at=created_at,
                session_path=session_path,
            )
        )
        evaluations.append(
            store.build_evaluation_row(
                spec,
                check,
                stream_id=key,
                source=result.source,
                run_id=run_id,
                measurement_run_id=run_id,
                created_at=created_at,
                session_path=session_path,
                thresholds=thresholds,
            )
        )
    rows[EVALUATION_DATASET] = evaluations
    return rows


def check_session(
    record: dict[str, str],
    *,
    config: ResolvedDataIntegrityConfig,
    run_id: str,
    created_at: datetime.datetime,
) -> dict[str, object]:
    """List one session and measure every stream in it, in one task.

    Never raises: a session always produces a record, whether it could not be listed
    at all or held streams that could not be reached or read. What separates those
    outcomes is which columns the record comes back with.

    Args:
        record: ``{"session_path": ...}``, as the driver seeded it.
        config: the resolved config.
        run_id: the id every row of this invocation shares.
        created_at: the timestamp every row of this invocation shares.

    Returns:
        The session's flat columns -- its path, how many streams it held, how many
        were unreadable, how many were unreachable, how many metric verdicts failed,
        and the listing error if there was one -- plus ``rows``, the pickled
        dataset-name-to-rows mapping for the whole session. ``rows`` is opaque
        because the store keeps one Lance dataset per metric, each with its own
        schema, so the rows do not share an Arrow type and cannot travel as typed
        columns. Nothing reads it but the driver's append step; the flat columns
        carry what logging, the run summary and the exit status need.

    """
    session_path = record["session_path"]
    try:
        sources = discover_session(session_path, config=config)
    # Wide on purpose. A listing fails as a stale path, a missing bucket, a denied
    # prefix or an expired token, and here they all mean one thing: this session went
    # unmeasured. Raising instead would abort the whole dataset -- Ray's
    # max_errored_blocks is 0 by default, and a block holds several sessions -- and
    # since the run commits last, that discards every session already measured.
    except Exception as exc:  # noqa: BLE001 - see above
        logger.opt(exception=True).debug("data-integrity listing failed for {}", session_path)
        logger.warning("session {} could not be listed: {}", session_path, exc)
        return _session_record(session_path, streams=0, listing_error=str(exc), rows={})

    rows: dict[str, list[dict[str, object]]] = {}
    unreadable = 0
    failed_metrics = 0
    unreachable = 0
    for source in sources:
        try:
            result = run_one_stream(
                source,
                expected_hz=config.checks.expected_hz,
                thresholds=config.checks.thresholds.to_thresholds(),
                batch_size=config.checks.batch_size,
                s3_profile_name=config.execution.s3_profile_name,
                azure_profile_name=config.execution.azure_profile_name,
                endpoint_url=config.execution.endpoint_url,
                max_attempts=config.execution.stream_attempts,
                raise_infrastructure_errors=True,
            )
        except InfrastructureError as exc:
            # Counted, not recorded: see InfrastructureError for why a row here would
            # be worse than none.
            logger.warning("{}", exc)
            unreachable += 1
            continue
        if result.error is not None:
            unreadable += 1
        failed_metrics += sum(1 for check in result.metrics if check.status is CheckStatus.FAIL)
        for dataset, dataset_rows in build_rows(
            result,
            config=config,
            run_id=run_id,
            created_at=created_at,
            session_path=session_path,
        ).items():
            rows.setdefault(dataset, []).extend(dataset_rows)

    return _session_record(
        session_path,
        streams=len(sources),
        unreadable=unreadable,
        failed_metrics=failed_metrics,
        unreachable=unreachable,
        rows=rows,
    )


def _session_record(  # noqa: PLR0913 -- one argument per column, all independent
    session_path: str,
    *,
    streams: int,
    unreadable: int = 0,
    failed_metrics: int = 0,
    unreachable: int = 0,
    listing_error: str = "",
    rows: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Build the record :func:`check_session` returns, with every column always present.

    One constructor because every column has to hold one Arrow type across every block:
    the driver's ``iter_batches`` concatenates blocks, and a column that is null in one
    and a string in another has no common type to concatenate under. Hence
    ``listing_error`` defaulting to ``""`` rather than to ``None``.
    """
    return {
        "session_path": session_path,
        "streams": streams,
        "unreadable": unreadable,
        "failed_metrics": failed_metrics,
        "unreachable": unreachable,
        "listing_error": listing_error,
        "rows": pickle.dumps(rows),
    }
