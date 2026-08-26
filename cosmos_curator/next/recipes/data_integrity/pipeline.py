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

r"""Ray Data entry point for the Curator Next ``data-integrity`` recipe.

The only module in the recipe that imports Ray, so
``python -m cosmos_curator.next.recipes.data_integrity.cli --help`` stays Ray-free.

One stage, one commit. Each task takes one session, lists it and measures every
stream in it, and every Lance write stays on the driver -- concurrent ``append_rows``
from many workers would race on create-or-append, which is a problem worth not
having.

Usage::

    python -m cosmos_curator.next.recipes.data_integrity.pipeline config.yaml
"""

import argparse
import datetime
import pickle
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import ray
from loguru import logger
from ray.data import TaskPoolStrategy

if TYPE_CHECKING:
    import pyarrow as pa

from cosmos_curator.core.sensors.scripts._cli_cloud import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.data_integrity import processing, store
from cosmos_curator.next.recipes.data_integrity.config import (
    TOOL_NAME,
    ResolvedDataIntegrityConfig,
    resolve_config,
)
from cosmos_curator.next.recipes.data_integrity.sessions import expand_sessions


def run_config(config: ResolvedDataIntegrityConfig) -> dict[str, object]:
    """Measure every stream of every configured session and commit one run.

    Args:
        config: the resolved config.

    Returns:
        The run summary: ``run_id``, the session and stream counts, how many streams
        were unreadable, how many metric verdicts failed, and the store root.

    Raises:
        ValueError: if the input expands to no sessions, or if those sessions hold no
            streams at all. Both mean the run would measure nothing, which is a
            mistake in the config rather than a passing run. The second is only
            knowable once every session has been listed, so it is raised after the
            appends and before the commit -- leaving rows that no reader believes,
            which is the same outcome as any other run that does not reach its commit.

    """
    sessions = expand_sessions(config.input, execution=config.execution)
    logger.info("data-integrity: {} session(s) to check", len(sessions))

    run_id = store.new_run_id()
    created_at = datetime.datetime.now(datetime.UTC)
    root = config.output.store_root
    thresholds = config.checks.thresholds.to_thresholds()
    storage_options = get_lance_storage_options(
        root,
        s3_profile_name=config.execution.s3_profile_name,
        endpoint_url=config.execution.endpoint_url,
    )

    ensure_ray_initialized()
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()

    num_streams = 0
    unreadable = 0
    failed_metrics = 0
    for batch in _measured_batches(sessions, config, run_id=run_id, created_at=created_at):
        pending: dict[str, list[dict[str, object]]] = {}
        for record in batch:
            num_streams += int(record["streams"])
            unreadable += int(record["unreadable"])
            failed_metrics += int(record["failed_metrics"])
            logger.info("session {}: {} stream(s)", record["session_path"], record["streams"])
            # Not untrusted input: the payload was pickled by this run's own workers,
            # a few lines of Ray plumbing away. See processing.check_session for why the
            # rows cannot travel as typed Arrow columns.
            for dataset, rows in pickle.loads(record["rows"]).items():  # noqa: S301
                pending.setdefault(dataset, []).extend(rows)
        for dataset, rows in pending.items():
            store.append_rows(rows, store.join(root, dataset), processing.DATASET_SCHEMAS[dataset], storage_options)
        logger.info(
            "appended {} session(s), {} stream(s) so far; {} unreadable",
            len(batch),
            num_streams,
            unreadable,
        )

    if not num_streams:
        msg = f"{len(sessions)} session(s) hold no video streams; check the session paths and input.limit"
        raise ValueError(msg)

    # Last, and deliberately so: nothing appended above is visible to a reader until
    # this row lands. session_path is null because this run describes many sessions --
    # the per-stream rows keep their own.
    store.commit_run(
        root,
        run_id=run_id,
        created_at=created_at,
        tool=TOOL_NAME,
        session_path=None,
        thresholds=thresholds,
        num_streams=num_streams,
        storage_options=storage_options,
    )
    store.write_manifest(
        root,
        run_id=run_id,
        created_at=created_at,
        session_path=None,
        thresholds=thresholds,
        tool=TOOL_NAME,
        num_streams=num_streams,
        s3_profile_name=config.execution.s3_profile_name,
        endpoint_url=config.execution.endpoint_url,
    )

    summary: dict[str, object] = {
        "run_id": run_id,
        "sessions": len(sessions),
        "streams": num_streams,
        "unreadable": unreadable,
        "failed_metrics": failed_metrics,
        "store_root": root,
    }
    logger.info(
        "committed run {} over {} stream(s) in {} session(s): {} unreadable, {} failed metric(s)",
        run_id,
        num_streams,
        len(sessions),
        unreadable,
        failed_metrics,
    )
    return summary


def _measured_batches(
    sessions: tuple[str, ...],
    config: ResolvedDataIntegrityConfig,
    *,
    run_id: str,
    created_at: datetime.datetime,
) -> Iterator[list[dict[str, Any]]]:
    """Measure every session, yielding finished ones to the driver a batch at a time.

    Batching bounds what the driver holds: row payloads cross it a batch at a time
    rather than a run at a time. Each task takes Ray's default single CPU, since the
    engine decodes one stream at a time and a session's streams are measured in
    sequence, and ``session_concurrency`` caps how many sessions are in flight.

    One task per session rather than per stream, because a whole session in one worker is
    what a cross-sensor check needs and what a per-stream task cannot offer. Load stays
    even without a finer record: a rig carries the same sensor suite between sessions, so
    stream counts are similar, and Ray hands the next session to whichever worker frees up.
    """
    # Wrapped in single-key dicts because from_items on bare strings names the column
    # "item"; from_items also spreads the sessions across blocks on its own.
    measured: ray.data.Dataset = ray.data.from_items([{"session_path": path} for path in sessions]).map(
        cast("Any", processing.check_session),
        fn_kwargs={"config": config, "run_id": run_id, "created_at": created_at},
        compute=TaskPoolStrategy(size=config.execution.session_concurrency),
    )
    for batch in measured.iter_batches(
        batch_size=config.execution.append_batch_size,
        batch_format="pyarrow",
    ):
        # ``batch_format="pyarrow"`` yields tables; the annotation Ray gives
        # ``iter_batches`` is the union of every format it can produce.
        yield cast("pa.Table", batch).to_pylist()


def main(argv: Sequence[str] | None = None) -> int:
    """Run from a JSON/YAML config path.

    Mirrors ``pipeline_runtime``'s exit contract rather than ``di-session``'s: a
    completed run exits 0 whatever its findings, and only an operational failure --
    which surfaces here as an exception -- is nonzero.
    """
    parser = argparse.ArgumentParser(description="Curator Next data-integrity pipeline")
    parser.add_argument("config", help="Path to a JSON/YAML data-integrity config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Small resolved-config override in dotted PATH=VALUE form.",
    )
    args = parser.parse_args(argv)
    run_config(resolve_config(args.config, overrides=args.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
