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
import json
import logging
import tempfile
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, cast

import pyarrow as pa
import ray
from loguru import logger

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.media.ffmpeg import assert_video_encoder_available
from cosmos_curator.next.recipes.video_split.config import ResolvedVideoSplitConfig, resolve_config
from cosmos_curator.next.recipes.video_split.discovery import resolve_input_selection
from cosmos_curator.next.recipes.video_split.lance_sink import commit_clip_snapshot, write_clip_fragments
from cosmos_curator.next.recipes.video_split.processing import (
    download_and_plan_source,
    transcode_source,
    upload_clip,
)
from cosmos_curator.next.recipes.video_split.records import (
    clip_table,
    error_table,
    validate_terminal_record_types,
)
from cosmos_curator.next.recipes.video_split.storage import RETRYABLE_STORAGE_ERRORS, upload_file

PUBLISH_RESULT_SCHEMA = pa.schema(
    [
        pa.field("result_type", pa.string(), nullable=False),
        pa.field("payload", pa.large_string(), nullable=False),
        pa.field("clip_count", pa.int64(), nullable=False),
    ]
)

_IO_STAGE_CPUS = 0.25
_IO_TASK_RESOURCES = {environment.CURATOR_IO_RESOURCE_NAME: 1.0}
_BACKOFF_FACTOR = 2.0
_MAX_BACKOFF_S = 30.0


def run_config(config: ResolvedVideoSplitConfig) -> dict[str, object]:
    """Run the streaming happy path and publish clips plus sparse errors."""
    started_at = monotonic()
    selection = resolve_input_selection(
        config.input,
        storage_profile=config.execution.storage_profile,
    )
    source_uris = selection.canonical_uris
    logger.info("Realized {} source video(s)", len(source_uris))
    if not source_uris:
        msg = (
            "Input selection realized 0 source videos; refusing to overwrite "
            f"{config.output.clips_lance_uri} and {config.output.errors_uri}"
        )
        raise ValueError(msg)

    assert_video_encoder_available(config.transcode.video_encoder)
    ensure_ray_initialized(local_resources=curator_io_resources())
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()

    terminal = clip_result_dataset(selection.scheduled_uris, config)
    clips_version, clips_published, errors = _publish_results(terminal, config)
    summary: dict[str, object] = {
        "sources": len(source_uris),
        "clips_published": clips_published,
        "errors": errors,
        "clips_lance_uri": config.output.clips_lance_uri,
        "clips_lance_version": clips_version,
        "errors_uri": config.output.errors_uri,
    }
    logger.info(
        "Published {} clips from {} source(s) with {} error(s) to {} at version {}",
        clips_published,
        len(source_uris),
        errors,
        config.output.clips_lance_uri,
        clips_version,
    )
    logger.info("Video split finished: elapsed={}", _format_elapsed(monotonic() - started_at))
    return summary


def clip_result_dataset(source_uris: tuple[str, ...], config: ResolvedVideoSplitConfig) -> ray.data.Dataset:
    """Build the download, streaming transcode, and independent upload path."""
    source_items = [{"source_uri": uri} for uri in source_uris]
    sources: ray.data.Dataset = ray.data.from_items(source_items, override_num_blocks=len(source_items))
    downloaded = sources.map(
        cast("Any", download_and_plan_source),
        fn_kwargs={"config": config},
        num_cpus=_IO_STAGE_CPUS,
        resources=_IO_TASK_RESOURCES,
    )
    transcoded = downloaded.flat_map(
        cast("Any", transcode_source),
        fn_kwargs={"config": config},
        num_cpus=config.execution.transcode_cpus,
    )
    return transcoded.map(
        cast("Any", upload_clip),
        fn_kwargs={"config": config},
        num_cpus=_IO_STAGE_CPUS,
        resources=_IO_TASK_RESOURCES,
    )


def publish_batch(work_records: pa.Table, *, uri: str, storage_profile: str) -> pa.Table:
    """Write clip fragments and return streaming driver-control records."""
    validate_terminal_record_types(work_records)
    clips = clip_table(work_records)
    errors = error_table(work_records)
    rows = [
        {
            "result_type": "stats",
            "payload": "",
            "clip_count": clips.num_rows,
        }
    ]
    rows.extend(
        {"result_type": "fragment", "payload": fragment, "clip_count": 0}
        for fragment in write_clip_fragments(clips, uri=uri, storage_profile=storage_profile)
    )
    rows.extend(
        {
            "result_type": "error",
            "payload": json.dumps(error, ensure_ascii=False, sort_keys=True),
            "clip_count": 0,
        }
        for error in errors.to_pylist()
    )
    return pa.Table.from_pylist(rows, schema=PUBLISH_RESULT_SCHEMA)


def _publish_results(terminal: ray.data.Dataset, config: ResolvedVideoSplitConfig) -> tuple[int, int, int]:
    """Commit the clip snapshot, then replace the complete JSON error report."""
    # This boundary keeps source transcodes as independent Ray tasks while
    # coalescing their metadata into useful Lance fragments. The conservative
    # default also bounds a pathological batch made entirely of error messages.
    publication_batches = terminal.repartition(
        target_num_rows_per_block=config.execution.clips_per_publish_batch,
        strict=True,
    )
    publication_results = publication_batches.map_batches(
        cast("Any", publish_batch),
        batch_format="pyarrow",
        batch_size=None,
        fn_kwargs={
            "uri": config.output.clips_lance_uri,
            "storage_profile": config.execution.storage_profile,
        },
    )

    fragments: list[str] = []
    clips_published = 0
    error_count = 0
    with tempfile.TemporaryDirectory(prefix="curator_next_video_split_publish_") as tmp_dir:
        errors_path = Path(tmp_dir) / "errors.json"
        with errors_path.open("w", encoding="utf-8") as report:
            report.write("[")
            for result in publication_results.iter_rows():
                result_type = str(result["result_type"])
                if result_type == "stats":
                    clips_published += int(result["clip_count"])
                elif result_type == "fragment":
                    fragments.append(str(result["payload"]))
                elif result_type == "error":
                    report.write("\n" if error_count == 0 else ",\n")
                    report.write(str(result["payload"]))
                    error_count += 1
                else:
                    msg = f"Publication returned unexpected result type {result_type!r}"
                    raise ValueError(msg)
            report.write("\n]\n" if error_count else "]\n")

        clips_version = commit_clip_snapshot(
            fragments,
            uri=config.output.clips_lance_uri,
            storage_profile=config.execution.storage_profile,
        )
        _retry_report_upload(errors_path, config=config)

    return clips_version, clips_published, error_count


def _retry_report_upload(report_path: Path, *, config: ResolvedVideoSplitConfig) -> None:
    """Upload the complete report after the canonical clip commit succeeds."""
    do_with_retries(
        partial(
            upload_file,
            str(report_path),
            config.output.errors_uri,
            storage_profile=config.execution.storage_profile,
        ),
        RETRYABLE_STORAGE_ERRORS,
        max_attempts=config.execution.storage_attempts,
        backoff_factor=_BACKOFF_FACTOR,
        max_wait_time_s=_MAX_BACKOFF_S,
        name="error-report-write",
    )


def _format_elapsed(elapsed_seconds: float) -> str:
    """Format a monotonic duration with explicit, compact time units."""
    total_tenths = max(0, round(elapsed_seconds * 10))
    hours, remaining_tenths = divmod(total_tenths, 36_000)
    minutes, remaining_tenths = divmod(remaining_tenths, 600)
    seconds, tenths = divmod(remaining_tenths, 10)
    seconds_text = f"{seconds}.{tenths}s" if tenths else f"{seconds}s"
    if hours:
        return f"{hours}h {minutes}m {seconds_text}"
    if minutes:
        return f"{minutes}m {seconds_text}"
    return seconds_text


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
