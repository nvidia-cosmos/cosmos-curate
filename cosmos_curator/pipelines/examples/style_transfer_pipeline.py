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

r"""Example pipeline: Cosmos3 video style transfer.

A standalone pipeline for testing the style-transfer stage end-to-end. Reads
videos from an input directory, applies Cosmos3 Generator Transfer (in-process
via vLLM-Omni), and writes restyled videos to an output directory.

The driver runs in the ``default`` Pixi environment, while the transfer actor runs
in the dedicated ``style-transfer`` environment that carries vLLM-Omni. The model
weights are downloaded on first use into the curator model cache; no separate
framework checkout is required.

Usage::

    cosmos-curator local launch --curator-path . -- pixi run --as-is -e default python -m \
        cosmos_curator.pipelines.examples.style_transfer_pipeline \
        --input-dir /data/test_videos \
        --output-dir /data/style_output \
        --style-transfer-model cosmos3_nano \
        --style-transfer-prompt "a watercolor painting of the scene" \
        --style-transfer-control edge
"""

import argparse
from pathlib import Path
from uuid import uuid4

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.models.style_transfer import style_transfer_variants
from cosmos_curator.pipelines.video.style_transfer.style_transfer_builders import (
    STYLE_TRANSFER_CONTROL_PRESETS,
    STYLE_TRANSFER_CONTROLS,
    STYLE_TRANSFER_RESOLUTIONS,
    StyleTransferConfig,
    build_style_transfer_stages,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _discover_videos(input_dir: str) -> list[Path]:
    """Find all video files in the input directory.

    Args:
        input_dir: Path to directory containing video files.

    Returns:
        Sorted list of video file paths.

    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)
    videos = [p for p in input_path.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS]
    return sorted(videos)


class _VideoReadStage(CuratorStage):
    """Read video files from disk into clip encoded_data."""

    @property
    def conda_env_name(self) -> str | None:
        return None

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        for task in tasks:
            video = task.video
            video_path = Path(str(video.input_video))
            if not video_path.is_file():
                logger.warning(f"Video file not found: {video_path}")
                continue
            raw_bytes = video_path.read_bytes()
            video.encoded_data = bytes_to_numpy(raw_bytes)  # type: ignore[assignment]
            clip = video.clips[0]
            clip.encoded_data = bytes_to_numpy(raw_bytes)  # type: ignore[assignment]
            logger.info(f"Read {len(raw_bytes)} bytes from {video_path}")
        return tasks


class _VideoWriteStage(CuratorStage):
    """Write restyled clip data back to disk."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    @property
    def conda_env_name(self) -> str | None:
        return None

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            video = task.video
            clip = video.clips[0]
            restyled = clip.style_transfer_video.resolve()
            if restyled is None:
                logger.warning(f"No style-transfer output for {video.input_video}")
                continue
            src_name = Path(str(video.input_video)).stem
            out_path = out_dir / f"{src_name}_{clip.uuid}_style.mp4"
            out_path.write_bytes(bytes(restyled))
            logger.info(f"Wrote restyled video to {out_path} ({restyled.nbytes} bytes)")
        return tasks


def _build_tasks(video_paths: list[Path]) -> list[PipelineTask]:
    """Create pipeline tasks wrapping each video file as a single-clip SplitPipeTask."""
    tasks: list[PipelineTask] = []
    for vp in video_paths:
        clip = Clip(
            uuid=uuid4(),
            source_video=str(vp),
            span=(0.0, 0.0),
        )
        video = Video(
            input_video=vp,
            clips=[clip],
        )
        task = SplitPipeTask(
            session_id=str(vp),
            video=video,
        )
        tasks.append(task)
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3 Style Transfer Example Pipeline")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input video files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write restyled videos.")

    st = parser.add_argument_group("style-transfer")
    st.add_argument(
        "--style-transfer-model",
        type=str,
        default="cosmos3_nano",
        choices=style_transfer_variants(),
        help="Style-transfer model variant.",
    )
    st.add_argument("--style-transfer-prompt", type=str, required=True, help="Target style prompt.")
    st.add_argument("--style-transfer-negative-prompt", type=str, default="", help="Negative prompt.")
    st.add_argument(
        "--style-transfer-control",
        type=str,
        default="edge",
        choices=list(STYLE_TRANSFER_CONTROLS),
        help="Spatial control signal.",
    )
    st.add_argument("--style-transfer-control-guidance", type=float, default=1.5, help="Global control CFG.")
    st.add_argument(
        "--style-transfer-edge-preset",
        type=str,
        default="medium",
        choices=list(STYLE_TRANSFER_CONTROL_PRESETS),
        help="Canny edge threshold preset.",
    )
    st.add_argument(
        "--style-transfer-blur-preset",
        type=str,
        default="medium",
        choices=list(STYLE_TRANSFER_CONTROL_PRESETS),
        help="Blur strength preset.",
    )
    st.add_argument(
        "--style-transfer-precompute-control",
        action="store_true",
        default=False,
        help="Extract the control on the host instead of via vLLM-Omni.",
    )
    st.add_argument(
        "--style-transfer-enable-guardrails",
        action="store_true",
        default=False,
        help="Enable Cosmos3 safety guardrails (off by default; requires the cosmos-guardrail package).",
    )
    st.add_argument("--style-transfer-guidance", type=float, default=3.0, help="Prompt CFG scale.")
    st.add_argument("--style-transfer-seed", type=int, default=2026, help="Random seed.")
    st.add_argument(
        "--style-transfer-resolution",
        type=str,
        default="720",
        choices=list(STYLE_TRANSFER_RESOLUTIONS),
        help="Cosmos3 output resolution bucket; source aspect is scaled to it.",
    )
    st.add_argument("--style-transfer-fps", type=int, default=None, help="Output FPS (None = preserve source).")
    st.add_argument("--style-transfer-chunk-frames", type=int, default=93, help="Frames per generation chunk.")
    st.add_argument("--style-transfer-conditional-frames", type=int, default=1, help="Overlap frames between chunks.")
    st.add_argument("--style-transfer-num-gpus", type=int, default=1, help="GPUs for the transfer engine.")
    st.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging.")

    return parser.parse_args()


def main() -> None:
    """Run the style-transfer example pipeline."""
    args = _parse_args()

    video_paths = _discover_videos(args.input_dir)
    logger.info(f"Found {len(video_paths)} videos in {args.input_dir}")
    if not video_paths:
        logger.warning("No video files found. Exiting.")
        return

    tasks = _build_tasks(video_paths)

    st_config = StyleTransferConfig(
        model_variant=args.style_transfer_model,
        prompt=args.style_transfer_prompt,
        negative_prompt=args.style_transfer_negative_prompt,
        control=args.style_transfer_control,
        control_guidance=args.style_transfer_control_guidance,
        edge_preset=args.style_transfer_edge_preset,
        blur_preset=args.style_transfer_blur_preset,
        precompute_control=args.style_transfer_precompute_control,
        guardrails=args.style_transfer_enable_guardrails,
        guidance=args.style_transfer_guidance,
        seed=args.style_transfer_seed,
        resolution=args.style_transfer_resolution,
        fps=args.style_transfer_fps,
        num_video_frames_per_chunk=args.style_transfer_chunk_frames,
        num_conditional_frames=args.style_transfer_conditional_frames,
        num_gpus=args.style_transfer_num_gpus,
        verbose=args.verbose,
    )

    stages: list[CuratorStage | CuratorStageSpec] = [
        _VideoReadStage(),
        *build_style_transfer_stages(st_config),
        _VideoWriteStage(args.output_dir),
    ]

    run_pipeline(tasks, stages)
    logger.info("Style-transfer pipeline completed")


if __name__ == "__main__":
    main()
