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

r"""Ray Data entry point for the Curator Next ``video-split`` recipe.

Usage::

    python -m cosmos_curator.next.recipes.video_split.pipeline config.yaml
"""

import argparse
import logging
from collections.abc import Sequence
from typing import Any, cast

import pyarrow as pa
import ray

from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.media.ffmpeg import assert_video_encoder_available
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config
from cosmos_curator.next.recipes.video_split.discovery import resolve_input_selection
from cosmos_curator.next.recipes.video_split.identities import make_source_id
from cosmos_curator.next.recipes.video_split.lance_sink import publish_snapshots, write_clip_fragments
from cosmos_curator.next.recipes.video_split.processing import process_source
from cosmos_curator.next.recipes.video_split.records import SOURCE_OUTCOME_SCHEMA, source_outcome_table

logger = logging.getLogger(__name__)

# One row per publish batch, carrying that batch's fragment metadata and the
# complete source outcomes that landed in it. Both are bounded by source or
# batch count, not clip count.
PUBLISH_BATCH_SCHEMA = pa.schema(
    [
        pa.field("fragments", pa.list_(pa.string()), nullable=False),
        pa.field("source_outcomes", pa.list_(pa.struct(list(SOURCE_OUTCOME_SCHEMA))), nullable=False),
    ]
)


def run_config(config: ResolvedVideoSplitConfig) -> dict[str, object]:
    """Execute source-granular work and publish both complete snapshots."""
    source_uris = resolve_input_selection(
        config.input,
        storage_profile=config.execution.storage_profile,
    )
    logger.info("Realized %d source video(s)", len(source_uris))
    if not source_uris:
        # Publishing here would overwrite both datasets with empty snapshots, so
        # a mistyped root or a prefix that has not landed yet would silently
        # destroy the previous run's output.
        msg = (
            f"Input selection realized 0 source videos; refusing to overwrite "
            f"{config.output.clips_lance_uri} and {config.output.sources_lance_uri} with empty snapshots"
        )
        raise ValueError(msg)

    assert_video_encoder_available(config.transcode.video_encoder)
    ensure_ray_initialized()
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()

    terminal = source_result_dataset(source_uris, config)
    clip_fragments, source_outcomes = _publish_clip_fragments(terminal, config)
    source_rows = _order_source_outcomes(source_outcomes, source_uris)

    snapshots = publish_snapshots(
        clip_fragments,
        source_rows,
        output=config.output,
        storage_profile=config.execution.storage_profile,
    )
    summary = _summary(config, source_rows, snapshots.clips_version, snapshots.sources_version)
    logger.info(
        "Published %d/%d clips from %d source(s) to %s at version %d",
        summary["clips_published"],
        summary["clips_planned"],
        summary["sources"],
        config.output.clips_lance_uri,
        snapshots.clips_version,
    )
    return summary


def source_result_dataset(source_uris: tuple[str, ...], config: ResolvedVideoSplitConfig) -> ray.data.Dataset:
    """Process each source as one recoverable unit and emit its terminal rows."""
    sources: ray.data.Dataset = ray.data.from_items([{"source_uri": uri} for uri in source_uris])
    return sources.flat_map(
        cast("Any", process_source),
        fn_kwargs={"config": config},
        num_cpus=config.execution.transcode_cpus,
    )


def publish_batch(work_records: pa.Table, *, uri: str, storage_profile: str) -> pa.Table:
    """Write one batch's clip fragments and return its complete source outcomes.

    Both outputs are derived from the same rows, so doing them in one pass keeps
    the terminal records streaming. Splitting them would mean materializing every
    terminal record in the object store just to scan it twice.
    """
    return pa.Table.from_pylist(
        [
            {
                "fragments": write_clip_fragments(work_records, uri=uri, storage_profile=storage_profile),
                "source_outcomes": source_outcome_table(work_records).to_pylist(),
            }
        ],
        schema=PUBLISH_BATCH_SCHEMA,
    )


def _publish_clip_fragments(
    terminal: ray.data.Dataset,
    config: ResolvedVideoSplitConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Collect worker-written fragment metadata and complete source outcomes."""
    # Keep publication batching downstream of source processing. Using ``batch_size``
    # directly on ``publish_batch`` lets Ray fuse the two map operators and
    # bundle up to ``clips_per_publish_batch`` rows before it starts the fused
    # task. With the default million-row publication batch, that would collapse
    # a typical run into one task and serialize all FFmpeg work. A strict streaming
    # repartition preserves the source-processing boundary while still producing
    # the intended Lance fragment sizes without an all-to-all shuffle.
    publication_batches = terminal.repartition(
        target_num_rows_per_block=config.execution.clips_per_publish_batch,
        strict=True,
    )
    batches = publication_batches.map_batches(
        cast("Any", publish_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={
            "uri": config.output.clips_lance_uri,
            "storage_profile": config.execution.storage_profile,
        },
    ).take_all()
    fragments = [str(fragment) for batch in batches for fragment in batch["fragments"]]
    source_outcomes = [outcome for batch in batches for outcome in batch["source_outcomes"]]
    return fragments, source_outcomes


def _order_source_outcomes(
    source_outcomes: list[dict[str, Any]],
    source_uris: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Validate one worker-owned outcome per source and restore selection order."""
    by_uri: dict[str, dict[str, Any]] = {}
    for outcome in source_outcomes:
        source_uri = str(outcome["source_uri"])
        if source_uri in by_uri:
            msg = f"Execution produced multiple source outcomes for {source_uri}"
            raise RuntimeError(msg)
        if outcome["source_id"] != make_source_id(source_uri):
            msg = f"Execution produced an inconsistent source_id for {source_uri}"
            raise RuntimeError(msg)
        by_uri[source_uri] = outcome

    selected = set(source_uris)
    missing = sorted(selected - by_uri.keys())
    if missing:
        msg = f"Execution did not produce source outcomes for: {', '.join(missing)}"
        raise RuntimeError(msg)
    unexpected = sorted(by_uri.keys() - selected)
    if unexpected:
        msg = f"Execution produced outcomes for unselected sources: {', '.join(unexpected)}"
        raise RuntimeError(msg)
    return [by_uri[uri] for uri in source_uris]


def _summary(
    config: ResolvedVideoSplitConfig,
    source_rows: list[dict[str, Any]],
    clips_version: int,
    sources_version: int,
) -> dict[str, object]:
    sources_succeeded = sum(row["status"] == "success" for row in source_rows)
    return {
        "sources": len(source_rows),
        "sources_succeeded": sources_succeeded,
        "sources_failed": len(source_rows) - sources_succeeded,
        "clips_planned": sum(int(row["planned_clip_count"]) for row in source_rows),
        "clips_published": sum(int(row["published_clip_count"]) for row in source_rows),
        "clips_failed": sum(int(row["failed_clip_count"]) for row in source_rows),
        "clips_lance_uri": config.output.clips_lance_uri,
        "clips_lance_version": clips_version,
        "sources_lance_uri": config.output.sources_lance_uri,
        "sources_lance_version": sources_version,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run from a JSON/YAML config path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Curator Next video-split pipeline")
    parser.add_argument("config", help="Path to a JSON/YAML video-split config.")
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
