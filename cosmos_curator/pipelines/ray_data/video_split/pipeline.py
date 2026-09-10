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

r"""Ray Data video splitting pipeline.

Downloads videos, splits them into TransNetV2 or fixed-stride clips,
transcodes each clip with FFmpeg, and writes the results to local or remote
storage.

Usage::

    python -m cosmos_curator.pipelines.ray_data.video_split.pipeline \
        video_split.yaml
"""

import argparse
import logging
from collections.abc import Sequence
from typing import Any, cast

import attrs
import ray
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from cosmos_curator.core.interfaces.pipeline_interface import download_models
from cosmos_curator.core.utils.ffmpeg_utils import assert_ffmpeg_supports_h264
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.core.utils.storage.storage_utils import get_files_relative, get_full_path, get_storage_client
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.ray_data._runtime import (
    capped_slots_for_items,
    configure_ray_data_progress,
    configure_ray_data_stability,
    live_ray_node_count,
)
from cosmos_curator.pipelines.ray_data.constants import DEFAULT_IO_SLOTS_PER_NODE
from cosmos_curator.pipelines.ray_data.video_split._clip_transcoder import make_transcode_fn
from cosmos_curator.pipelines.ray_data.video_split._clip_writer import make_write_fn
from cosmos_curator.pipelines.ray_data.video_split._fixed_stride_splitter import make_split_fn
from cosmos_curator.pipelines.ray_data.video_split._summary_writer import write_summary
from cosmos_curator.pipelines.ray_data.video_split._transnetv2_splitter import (
    TransNetV2Splitter,
    transnetv2_model_ids,
    validate_transnetv2_length_bounds,
)
from cosmos_curator.pipelines.ray_data.video_split._video_reader import read_video
from cosmos_curator.pipelines.ray_data.video_split._vllm_caption import (
    _max_caption_workers,
    caption_window_rows,
    make_default_vllm_config,
    qwen_model_id,
    qwen_model_source,
    write_captioned_metadata_and_summary,
)
from cosmos_curator.pipelines.ray_data.video_split.config import (
    ResolvedVideoSplitConfig,
    resolve_video_split_config,
)
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig

logger = logging.getLogger(__name__)


def _configure_ray_data_progress(*, progress: bool) -> None:
    """Configure Ray Data progress output before creating datasets."""
    configure_ray_data_progress(progress=progress)


def _download_slots_for_video_count(*, num_videos: int, num_nodes: int) -> int:
    """Cap read/split task concurrency by both cluster size and input count."""
    return capped_slots_for_items(
        num_items=num_videos,
        num_nodes=num_nodes,
        max_per_node=DEFAULT_IO_SLOTS_PER_NODE,
    )


def _limit_or_unlimited(limit: int | None) -> int:
    """Return the legacy splitter/storage sentinel for an optional positive limit."""
    return limit or 0


def _discover_videos(input_video_path: str, limit: int | None = None) -> list[str]:
    """List video files under *input_video_path* and return full paths.

    Works for both local directories and remote storage (S3/Azure).
    """
    client = get_storage_client(input_video_path)
    relative_paths = get_files_relative(input_video_path, client, _limit_or_unlimited(limit))
    return [str(get_full_path(input_video_path, rp)) for rp in relative_paths]


def _required_model_ids(
    config: ResolvedVideoSplitConfig,
    *,
    generate_captions: bool,
) -> list[str]:
    """Return model IDs required for the selected Ray Data pipeline."""
    model_ids: list[str] = []
    if config.split.method == "transnetv2":
        model_ids.extend(transnetv2_model_ids())
    if generate_captions:
        model_ids.append(qwen_model_id())
    return list(dict.fromkeys(model_ids))


def _validate_transnetv2_cluster_resources(
    config: ResolvedVideoSplitConfig,
    *,
    total_visible_gpus: int,
) -> None:
    """Fail fast when the selected TransNetV2 stage cannot be scheduled."""
    if config.split.method != "transnetv2":
        return
    decode_cpus = config.split.transnetv2.frame_decode_cpus_per_worker
    gpus_per_worker = config.split.transnetv2.gpus_per_worker
    if float(total_visible_gpus) < gpus_per_worker:
        msg = (
            "TransNetV2 splitting requires visible GPUs. "
            "Use split.method=fixed_stride or lower split.transnetv2.gpus_per_worker."
        )
        raise ValueError(msg)

    ray_nodes = cast("list[dict[str, Any]]", ray.nodes())  # type: ignore[no-untyped-call]
    for node in ray_nodes:
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        if not isinstance(resources, dict):
            continue
        node_cpus = float(resources.get("CPU", 0.0))
        node_gpus = float(resources.get("GPU", 0.0))
        if node_cpus >= decode_cpus and node_gpus >= gpus_per_worker:
            return

    msg = (
        "TransNetV2 splitting requires at least one live Ray node with "
        f"{decode_cpus} CPU(s) and {gpus_per_worker:g} GPU(s) available for one worker. "
        "Lower split.transnetv2.frame_decode_cpus_per_worker or split.transnetv2.gpus_per_worker, "
        "or use split.method=fixed_stride."
    )
    raise ValueError(msg)


def _caption_vllm_config(config: ResolvedVideoSplitConfig) -> VllmConfig:
    """Return the caption vLLM config selected by the resolved config."""
    return attrs.evolve(
        make_default_vllm_config(),
        batch_size=config.caption.batch_size,
        preprocess_mode=PreprocessMode(config.caption.preprocess_mode),
    )


def _caption_workers_from_downloaded_gpus(total_visible_gpus: int, vllm_config: VllmConfig | None) -> int:
    """Return the caption actor ceiling from the GPU count discovered during model download."""
    worker_config = vllm_config or make_default_vllm_config()
    return _max_caption_workers(total_visible_gpus, worker_config.num_gpus)


def _apply_split_stage(
    ds: ray.data.Dataset,
    config: ResolvedVideoSplitConfig,
    *,
    download_slots: int,
) -> ray.data.Dataset:
    """Apply the selected span-generation stage to the video dataset."""
    if config.split.method == "fixed_stride":
        fixed_stride = config.split.fixed_stride
        return ds.map(
            make_split_fn(
                clip_len_s=fixed_stride.duration_s,
                clip_stride_s=fixed_stride.stride_s,
                min_clip_length_s=fixed_stride.min_clip_length_s,
                limit_clips=_limit_or_unlimited(config.split.limit_clips),
            ),
            num_cpus=0.25,
            compute=TaskPoolStrategy(size=download_slots),
        )

    transnetv2_config = config.split.transnetv2
    transnetv2_kwargs: dict[str, Any] = {
        "threshold": transnetv2_config.threshold,
        "min_length_s": transnetv2_config.min_length_s,
        "min_length_frames": transnetv2_config.min_length_frames,
        "max_length_s": transnetv2_config.max_length_s,
        "max_length_mode": transnetv2_config.max_length_mode,
        "crop_s": transnetv2_config.crop_s,
        "num_decode_cpus_per_worker": transnetv2_config.frame_decode_cpus_per_worker,
        "limit_clips": _limit_or_unlimited(config.split.limit_clips),
    }
    transnetv2_fn = cast("Any", TransNetV2Splitter)
    return ds.map(
        transnetv2_fn,
        fn_constructor_kwargs=transnetv2_kwargs,
        num_cpus=transnetv2_config.frame_decode_cpus_per_worker,
        num_gpus=transnetv2_config.gpus_per_worker,
        compute=ActorPoolStrategy(min_size=1, max_size=download_slots, initial_size=1),
        runtime_env=ray_data_gpu_runtime_env("default"),
        scheduling_strategy="DEFAULT",
    )


def run_config(config: ResolvedVideoSplitConfig) -> int:
    """Build and execute the splitting pipeline from a resolved typed config.

    Args:
        config: Resolved pipeline config.

    Returns:
        Number of clips written.

    """
    if config.split.method == "transnetv2":
        validate_transnetv2_length_bounds(config.split.transnetv2.min_length_s, config.split.transnetv2.max_length_s)

    assert_ffmpeg_supports_h264()

    generate_captions = config.caption.enabled
    model_ids = _required_model_ids(config, generate_captions=generate_captions)
    num_gpus_available = 0
    if model_ids:
        num_gpus_available = int(download_models(model_ids, config.execution.model_weights_path))

    caption_model_source: str | None = None
    if generate_captions:
        caption_model_source = qwen_model_source()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    _configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability()

    video_paths = _discover_videos(config.input.video_path, limit=config.input.limit)
    if not video_paths:
        logger.warning("No videos found in %s", config.input.video_path)
        return 0
    logger.info("Found %d input video(s)", len(video_paths))

    _validate_transnetv2_cluster_resources(config, total_visible_gpus=num_gpus_available)

    # Seed dataset — one row per video.
    ds: ray.data.Dataset = ray.data.from_items([{"video_path": vp} for vp in video_paths])

    # Fractional CPU (matching Xenna's 0.25 precedent for IO-bound download)
    # with an explicit TaskPoolStrategy size cap scaled to the smaller of
    # cluster size and input rows.
    # Task compute (not actor pool) preserves fusion with split.
    download_slots = _download_slots_for_video_count(
        num_videos=len(video_paths),
        num_nodes=live_ray_node_count(),
    )

    # Stage 1: Download + extract metadata (1:1).
    ds = ds.map(
        read_video,
        num_cpus=0.25,
        compute=TaskPoolStrategy(size=download_slots),
    )

    # Stage 2: Compute clip spans (1:1, no fan-out).
    ds = _apply_split_stage(ds, config, download_slots=download_slots)

    # Stage 3: Transcode + fan-out (1:N — one video in, N clips out).
    ds = ds.flat_map(
        make_transcode_fn(
            encoder=config.transcode.encoder,
            encoder_threads=config.transcode.encoder_threads,
            ffmpeg_batch_size=config.transcode.ffmpeg_batch_size,
            use_input_bit_rate=config.transcode.use_input_video_bit_rate,
        ),
        num_cpus=config.transcode.cpus_per_worker,
    )

    # Stage 4: Write clips to output (1:1). IO-bound upload; fractional CPU
    # matches Xenna's ClipWriterStage at cpus=0.25.
    if generate_captions:
        ds = ds.map(
            make_write_fn(config.output.clip_path, write_metadata=False, keep_clip_bytes=True),
            num_cpus=0.25,
        )

        if caption_model_source is None:
            msg = "caption_model_source must be set when generate_captions is True"
            raise RuntimeError(msg)
        vllm_config = _caption_vllm_config(config)
        ds = caption_window_rows(
            ds,
            model_source=caption_model_source,
            caption_workers=_caption_workers_from_downloaded_gpus(num_gpus_available, vllm_config),
            vllm_config=vllm_config,
        )

        num_clips = write_captioned_metadata_and_summary(
            ds,
            input_video_path=config.input.video_path,
            output_path=config.output.clip_path,
            num_input_videos=len(video_paths),
        )
        logger.info("Wrote %d clip(s) with captions to %s", num_clips, config.output.clip_path)
        return num_clips

    ds = ds.map(make_write_fn(config.output.clip_path), num_cpus=0.25)

    # Stage 5: Aggregate per-video and write summary.json.
    num_clips = write_summary(
        ds,
        input_video_path=config.input.video_path,
        output_path=config.output.clip_path,
        num_input_videos=len(video_paths),
    )
    logger.info("Wrote %d clip(s) to %s", num_clips, config.output.clip_path)
    return num_clips


def run(config: ResolvedVideoSplitConfig) -> int:
    """Build and execute the splitting pipeline from a resolved config."""
    return run_config(config)


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    """Add the config-only module CLI arguments."""
    parser.add_argument("config", help="Path to a JSON/YAML video_split config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Small resolved-config override in dotted PATH=VALUE form.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Ray Data splitting pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Ray Data video splitting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _setup_parser(parser)
    args = parser.parse_args(argv)
    resolution = resolve_video_split_config(args.config, overrides=args.overrides)
    run_config(resolution.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
