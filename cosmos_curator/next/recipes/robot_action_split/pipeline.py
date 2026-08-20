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

r"""Entry point for ``robot-action-split`` pipeline runs.

Minimal v1: discover spans from local or S3 dataset roots, cut clips with
smart cut (stub -> full re-encode for now), write output locally or to S3,
and publish successful clips to a Lance dataset.

Usage::

    python -m cosmos_curator.next.recipes.robot_action_split.pipeline \
        --config path/to/config.yaml

Example config (local paths, handful of videos)::

    schema_version: 1
    kind: robot-action-split
    input:
      uris: [/data/robot_dataset/]
      source_dataset: my_dataset
      limit: 5
    output:
      media_root: /tmp/robot_clips/
      lance_uri: /tmp/robot_clips/lance/clips.lance
    execution:
      discovery_workers: 2
"""

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig, load_config
from cosmos_curator.next.recipes.robot_action_split.discovery import ChunkSpanBatch, discover_spans
from cosmos_curator.next.recipes.robot_action_split.lance_sink import write_outcomes_to_lance
from cosmos_curator.next.recipes.robot_action_split.processing import process_batch


def _run_sequential(
    batches: list[ChunkSpanBatch],
    config: ResolvedRobotActionSplitConfig,
) -> list[dict[str, Any]]:
    """Process batches one at a time, reusing each downloaded chunk across its batches."""
    all_outcomes: list[dict[str, Any]] = []
    total = len(batches)
    batch_num = 0
    # Group consecutive batches by source chunk so the video file is downloaded
    # once per chunk rather than once per batch.  max_segments_per_batch splits
    # one chunk's spans across many batches; without grouping each call to
    # process_batch would re-download the same (potentially large) source file.
    for _chunk_uri, chunk_batches_iter in itertools.groupby(batches, key=lambda b: b.chunk_mp4_uri):
        with tempfile.TemporaryDirectory(dir=config.execution.tmp_dir) as chunk_tmp:
            staged_path = str(Path(chunk_tmp) / "chunk.mp4")
            for batch in chunk_batches_iter:
                batch_num += 1
                logger.info(f"  Batch {batch_num}/{total}: {batch.chunk_mp4_uri} ({len(batch.items)} span(s))")
                outcomes = process_batch(batch, config=config, staged_chunk_path=staged_path)
                all_outcomes.extend(outcomes)
                n_ok = sum(1 for o in outcomes if o["status"] == "success")
                n_fail = len(outcomes) - n_ok
                logger.info(f"    -> {n_ok} succeeded, {n_fail} failed")
                for o in outcomes:
                    if o["status"] != "success":
                        logger.warning(f"      FAIL [{o.get('error_stage')}] {o.get('error_message', '')[:300]}")
    return all_outcomes


def _run_ray_data(
    batches: list[ChunkSpanBatch],
    config: ResolvedRobotActionSplitConfig,
) -> list[dict[str, Any]]:
    """Process batches in parallel using Ray Data flat_map across Ray workers.

    Each task downloads its own copy of the source chunk — the staged-path
    optimisation used by the sequential path does not apply here because tasks
    are distributed across workers that do not share a local filesystem.  The
    trade-off is acceptable: Ray Data parallelises many chunks simultaneously,
    so overall throughput improves even though individual chunks are re-fetched
    per task.
    """
    import ray  # noqa: PLC0415
    import ray.data  # noqa: PLC0415

    if not ray.is_initialized():
        # In a managed Slurm-Ray run the driver sets RAY_ADDRESS to the head's
        # address before launching this process.  For standalone sbatch jobs and
        # local runs (development, CI) RAY_ADDRESS is unset and ray.init() starts
        # a single-node cluster on the current machine.
        ray_address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=ray_address)
        logger.info(f"Ray initialised: {ray.cluster_resources()}")  # type: ignore[no-untyped-call]

    def _udf(batch_row: dict[str, Any]) -> list[dict[str, Any]]:
        """Ray Data UDF: process one ChunkSpanBatch; config is captured in the closure."""
        batch: ChunkSpanBatch = batch_row["batch"]
        return process_batch(batch, config=config)

    rows = [{"batch": b} for b in batches]
    ds = ray.data.from_items(rows, override_num_blocks=len(rows))

    # concurrency=None lets Ray choose based on available CPU resources.
    # Each task uses cut_cpus CPUs as declared in the execution config.
    all_outcomes: list[dict[str, Any]] = ds.flat_map(
        _udf,
        num_cpus=config.execution.cut_cpus,
    ).take_all()

    succeeded = sum(1 for o in all_outcomes if o["status"] == "success")
    failed = len(all_outcomes) - succeeded
    logger.info(f"Ray Data complete: {succeeded} succeeded, {failed} failed across {len(batches)} batch(es)")
    for o in all_outcomes:
        if o["status"] != "success":
            logger.warning(f"  FAIL [{o.get('error_stage')}] {o.get('error_message', '')[:300]}")

    return all_outcomes


def run(
    config_path: str,
    *,
    config: ResolvedRobotActionSplitConfig | None = None,
) -> dict[str, object]:
    """Discover spans, cut clips, write outputs, and publish to Lance."""
    resolved = config if config is not None else load_config(config_path)
    logger.info(f"Loaded config: kind={resolved.kind} source_dataset={resolved.input.source_dataset}")

    # Phase 1: span discovery (pre-Ray, CPU-bound parquet reads).
    logger.info("Discovering spans...")
    batches = discover_spans(resolved)
    total_items = sum(len(b.items) for b in batches)
    logger.info(f"Discovery complete: {len(batches)} batch(es), {total_items} span(s)")

    if not batches:
        logger.warning("No spans found; nothing to do.")
        return {"total": 0, "succeeded": 0, "failed": 0, "outcomes": []}

    # Phase 2: cut + action bin.
    logger.info("Cutting clips...")
    if resolved.execution.ray_data:
        all_outcomes = _run_ray_data(batches, resolved)
    else:
        all_outcomes = _run_sequential(batches, resolved)

    # Summary.
    succeeded = [o for o in all_outcomes if o["status"] == "success"]
    failed = [o for o in all_outcomes if o["status"] != "success"]
    logger.info(f"Done: {len(succeeded)} clips written, {len(failed)} failed")
    if failed:
        for o in failed:
            logger.warning(f"  FAILED {o['clip_id']}: [{o.get('error_stage')}] {o.get('error_message')}")

    # Phase 3: Lance publication.
    if succeeded:
        logger.info(f"Publishing {len(succeeded)} clip(s) to Lance: {resolved.output.lance_uri}")
        storage_options = get_lance_storage_options(
            resolved.output.lance_uri,
            profile_name=resolved.execution.storage_profile,
        )
        lance_version = write_outcomes_to_lance(
            all_outcomes,
            lance_uri=resolved.output.lance_uri,
            storage_options=storage_options,
        )
        logger.info(f"Lance dataset version after write: {lance_version}")
    else:
        logger.warning("No successful clips; skipping Lance write.")

    # Write a run summary (local only; skipped for S3 media_root to keep the
    # driver stateless — the Lance dataset is the durable record).
    media_root = resolved.output.media_root
    if not media_root.startswith("s3://"):
        summary_path = Path(media_root.rstrip("/")) / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "total": len(all_outcomes),
                    "succeeded": len(succeeded),
                    "failed": len(failed),
                    "outcomes": all_outcomes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Summary written to {summary_path}")

    return {
        "total": len(all_outcomes),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "outcomes": all_outcomes,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Curator Next robot-action-split pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
