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

"""Cosmos3 video style-transfer pipeline stage (Curator adapter).

For each clip, this stage runs Cosmos3 Generator Transfer (in-process via
vLLM-Omni) to produce a restyled clip that follows a spatial control signal
derived from the source while matching the text prompt. It mirrors
``SuperResolutionStage``: dedicated GPU environment, per-clip error isolation.

Unlike SR (which replaces ``clip.encoded_data`` in place), style transfer writes
the restyled clip to a separate ``clip.style_transfer_video`` field so downstream
stages keep operating on the original clip and the writer emits a sidecar
``style_transfer/<uuid>.mp4``.

The clip is decoded once via the shared ``decode_clip_at_fps`` sensor helper (the
same one the SAM3 tracking stages use); those frames are the conditioning video and
also feed host control extraction.

Two control-input modes (see ``StyleTransferConfig.precompute_control``):

- on-the-fly mode (default): hand the decoded frames to vLLM-Omni as the video
  input; it derives edge/blur internally.
- pre-computed mode: extract the edge/blur control on the host via
  ``control_signals`` and pass it as a ``control_path`` (unit-testable, and the
  only path for controls vLLM-Omni can't derive). The decoded frames are still
  supplied as the video conditioning input.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.models.style_transfer import (
    Cosmos3OmniTransferModel,
    StyleTransferParams,
    clamp_num_gpus_for_variant,
)
from cosmos_curator.pipelines.video.style_transfer.control_signals import (
    HOST_COMPUTABLE_CONTROLS,
    extract_control_frames,
)
from cosmos_curator.pipelines.video.style_transfer.style_transfer_builders import StyleTransferConfig
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask


class StyleTransferStage(CuratorStage):
    """Pipeline stage that applies Cosmos3 Generator Transfer to clips.

    For each clip: resolve the encoded bytes, derive (or hand off) the spatial
    control signal, run the in-process transfer, and store the restyled mp4 bytes
    on ``clip.style_transfer_video``.
    """

    def __init__(self, config: StyleTransferConfig) -> None:
        """Initialize the style-transfer stage.

        Args:
            config: Style-transfer configuration.

        """
        self._config = config
        self._model = Cosmos3OmniTransferModel(
            variant=config.model_variant,
            num_gpus=config.num_gpus,
            guardrails=config.guardrails,
        )

    @property
    def resources(self) -> CuratorStageResource:
        """Return the resource requirements (variant-aware GPU count)."""
        num_gpus = clamp_num_gpus_for_variant(self._config.model_variant, self._config.num_gpus)
        return CuratorStageResource(cpus=1.0, gpus=float(num_gpus))

    @property
    def model(self) -> ModelInterface | None:
        """Return the Cosmos transfer model interface."""
        return self._model

    def stage_setup(self) -> None:
        """Build the in-process vLLM-Omni engine on the worker actor."""
        self._model.setup()

    def destroy(self) -> None:
        """Gracefully shut down the in-process vLLM-Omni engine on stage stop."""
        self._model.shutdown()

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Apply style transfer to the primary camera's clips in each task.

        Args:
            tasks: Pipeline tasks containing video clips to restyle.

        Returns:
            The tasks with ``clip.style_transfer_video`` populated on the primary
            video's clips.

        """
        for task in tasks:
            # Restyle the primary camera only: the metadata writer persists just the
            # primary video (video_index == 0), so restyling secondary cameras would
            # burn diffusion compute that never gets written. No-op for single-camera
            # tasks (one video), only matters in multicam mode.
            video = task.video
            for clip in video.clips:
                try:
                    self._transfer_clip(clip)
                except Exception:  # noqa: BLE001 -- isolate one clip's failure from the batch
                    logger.exception(f"Style transfer failed for clip {clip.uuid} in {video.input_video}")
                    clip.errors["style_transfer"] = "style transfer inference failed"
        return tasks

    def _transfer_clip(self, clip: Clip) -> None:
        """Run Cosmos3 transfer on a single clip, storing restyled bytes on the clip."""
        clip_bytes = clip.encoded_data.resolve()
        if clip_bytes is None:
            logger.warning(f"Clip {clip.uuid}: no encoded_data, skipping style transfer")
            return

        metadata = clip.extract_metadata()
        if metadata is None:
            logger.warning(f"Clip {clip.uuid}: no metadata, skipping style transfer")
            return

        # Clamp the target fps to the source fps. decode_clip_at_fps snaps a sampling
        # grid to the nearest real frame, so requesting more than the source has just
        # duplicates frames (and max_frames would then truncate) -- never upsample.
        source_fps = float(metadata["framerate"])
        requested_fps = self._config.fps if self._config.fps is not None else source_fps
        if self._config.fps is not None and self._config.fps > source_fps:
            logger.warning(
                f"Clip {clip.uuid}: requested style-transfer fps {self._config.fps} exceeds source fps "
                f"{source_fps:.3f}; clamping to source to avoid duplicated frames."
            )
        fps = max(1, round(min(requested_fps, source_fps)))
        width = int(metadata["width"])
        height = int(metadata["height"])

        # Decode the clip once with the shared sensor decoder (the same helper the
        # SAM3 tracking stages use); the frames feed both the conditioning video and
        # the host control extraction, so we never decode twice.
        from cosmos_curator.pipelines.video.tracking.sensor_decode import decode_clip_at_fps  # noqa: PLC0415

        decoded = decode_clip_at_fps(clip_bytes.tobytes(), float(fps))
        if not decoded.frames_rgb:
            logger.warning(f"Clip {clip.uuid}: decoded 0 frames, skipping style transfer")
            return
        vision_frames = np.stack(decoded.frames_rgb, axis=0)

        # `num_frames` -> vLLM-Omni `max_frames`, the cap on how many conditioning
        # frames are processed. Set it to the exact number of decoded frames so vLLM-Omni
        # covers the whole clip via internal chunking at any fps; the per-chunk NATTEN
        # window is bounded separately by `num_video_frames_per_chunk`.
        params = StyleTransferParams(
            prompt=self._config.prompt,
            negative_prompt=self._config.negative_prompt,
            control=self._config.control,
            control_guidance=self._config.control_guidance,
            guidance=self._config.guidance,
            seed=self._config.seed,
            resolution=self._config.resolution,
            fps=int(fps),
            num_frames=len(vision_frames),
            num_video_frames_per_chunk=self._config.num_video_frames_per_chunk,
            num_conditional_frames=self._config.num_conditional_frames,
            edge_preset=self._config.edge_preset,
            blur_preset=self._config.blur_preset,
        )

        with tempfile.TemporaryDirectory(prefix="style_transfer_") as tmp:
            work_dir = Path(tmp)
            # The source frames are always the video conditioning input; pre-computed
            # mode additionally derives a host-extracted control video from them.
            control_paths: dict[str, Path] | None = None
            if self._config.precompute_control:
                control_paths = {
                    self._config.control: self._encode_control_video(
                        decoded.frames_rgb, work_dir, width, height, float(fps)
                    )
                }

            result = self._model.generate(
                vision_frames=vision_frames,
                control_paths=control_paths,
                params=params,
                work_dir=work_dir,
            )

        clip.style_transfer_video = bytes_to_numpy(result.mp4_bytes)  # type: ignore[assignment]
        clip.style_transfer_metadata = self._build_provenance(
            clip, params, metadata, num_generated_frames=result.num_frames, output_size_bytes=len(result.mp4_bytes)
        )

        if self._config.verbose:
            logger.info(
                f"Clip {clip.uuid}: style transfer ({self._config.control}, {self._config.model_variant}) "
                f"{len(vision_frames)} src frames (chunk {self._config.num_video_frames_per_chunk}) -> "
                f"{result.num_frames} frames, {len(result.mp4_bytes)} bytes"
            )

    def _build_provenance(
        self,
        clip: Clip,
        params: StyleTransferParams,
        metadata: dict[str, Any],
        *,
        num_generated_frames: int,
        output_size_bytes: int,
    ) -> dict[str, Any]:
        """Assemble the restyle provenance dict serialized as ``style_transfer/<uuid>.json``.

        Records the generation params, source-clip linkage, and output size so a
        restyle can be reproduced/compared later. ``output.num_frames`` is the number
        of frames the backend actually generated. The active control preset is
        recorded for edge/blur controls (None for controls that use no preset).
        """
        cfg = self._config
        # Only edge/blur controls carry a preset; record None for any other control
        # (e.g. future depth/seg) rather than mislabeling it with an unused preset.
        preset_by_control = {"edge": cfg.edge_preset, "blur": cfg.blur_preset}
        active_preset = preset_by_control.get(cfg.control)
        return {
            "schema_version": 1,
            "model": {"variant": cfg.model_variant, "guardrails": cfg.guardrails},
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "control": {
                "type": cfg.control,
                "guidance": params.control_guidance,
                "preset": active_preset,
                "precomputed": cfg.precompute_control,
            },
            "sampling": {
                "guidance": params.guidance,
                "seed": params.seed,
                "resolution": params.resolution,
                "fps": params.fps,
                "num_video_frames_per_chunk": params.num_video_frames_per_chunk,
                "num_conditional_frames": params.num_conditional_frames,
            },
            "source": {
                "clip_uuid": str(clip.uuid),
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
                "fps": float(metadata["framerate"]),
                "num_frames": int(metadata["num_frames"]),
            },
            "output": {"num_frames": num_generated_frames, "size_bytes": output_size_bytes},
        }

    def _encode_control_video(
        self,
        frames_rgb: list[npt.NDArray[np.uint8]],
        work_dir: Path,
        width: int,
        height: int,
        fps: float,
    ) -> Path:
        """Extract the edge/blur control from decoded frames and encode it to an mp4.

        Reuses the already-decoded source frames (no second decode) and the shared
        SAM3 ``encode_frames_to_mp4`` writer.

        Raises:
            ValueError: If the configured control is not host-computable.
            RuntimeError: If control frames could not be encoded.

        """
        import cv2  # noqa: PLC0415

        from cosmos_curator.pipelines.video.tracking.track_funcs import encode_frames_to_mp4  # noqa: PLC0415

        if self._config.control not in HOST_COMPUTABLE_CONTROLS:
            msg = (
                f"precompute_control is set but control '{self._config.control}' is not host-computable; "
                f"choose one of {HOST_COMPUTABLE_CONTROLS}."
            )
            raise ValueError(msg)

        preset = self._config.edge_preset if self._config.control == "edge" else self._config.blur_preset
        control_rgb = extract_control_frames(frames_rgb, self._config.control, preset=preset)
        control_bgr = [np.asarray(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), dtype=np.uint8) for f in control_rgb]

        control_bytes = encode_frames_to_mp4(control_bgr, fps, width, height)
        if control_bytes is None:
            msg = "failed to encode control video"
            raise RuntimeError(msg)

        control_path = work_dir / f"control_{self._config.control}.mp4"
        control_path.write_bytes(control_bytes)
        return control_path
