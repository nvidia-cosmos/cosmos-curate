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
"""GPU E2E for captioning the exact frames selected by the Sensor Library."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.core.utils.model.model_utils import get_local_dir_for_weights_name
from cosmos_curator.models.vllm_model_ids import get_vllm_model_id
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode, resolve_preprocess_mode
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, VllmSamplingConfig
from cosmos_curator.pipelines.video.utils.vllm_defaults import resolve_vllm_sampling_config
from tests.cosmos_curator.pipelines.video.captioning._sensor_aligned_test_utils import (
    CHECKED_IN_ALIGNED_ROWS,
    CHECKED_IN_CODEC_INPUT,
    FRONT_CAMERA_SENSOR_ID,
    IMU_SENSOR_ID,
    REAR_CAMERA_SENSOR_ID,
    SensorAlignmentResult,
    SensorEpisodeInput,
    SensorSessionDescriptor,
    align_sensor_session,
    assert_checked_in_codec_alignment,
    assert_sensor_alignment,
    compute_frame_sha256,
    prepare_sensor_session,
)

_DEFAULT_MODEL_VARIANT = "qwen"
_FREE_FORM_CAPTION_MAX_TOKENS = 256
_FREE_FORM_CAPTION_PROMPT_VERSION = "free-form-caption-v1"
# Keep the collected prompt bounded so a 256-token budget can require a clean stop.
_FREE_FORM_CAPTION_PROMPT = """Describe this video in two or three concise, factual sentences.
Mention the visible scene, important motion, and changes over time.
"""
_CHECKED_IN_FREE_FORM_PROMPT_SHA256 = "61cf8cfe6a077bd60e8928e88a631b307a1de28a70e1228863a115be98caf59e"


@dataclass(frozen=True)
class _CaptionGeneration:
    """Raw and plugin-decoded synchronous generation evidence."""

    raw_text: str
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    model_prompt_token_ids_sha256: str | None
    output_tokens: int
    output_processing_error: Exception | None


def _resolve_max_tokens(max_tokens: int | None) -> int:
    """Return a valid free-form output budget, allowing larger model-specific probes."""
    resolved_max_tokens = _FREE_FORM_CAPTION_MAX_TOKENS if max_tokens is None else max_tokens
    if resolved_max_tokens <= 0:
        msg = f"max_tokens must be positive, got {resolved_max_tokens}"
        raise ValueError(msg)
    return resolved_max_tokens


def _prompt_text_sha256(prompt: str) -> str:
    """Hash the exact prompt bytes used for generation."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _model_prompt_token_ids_sha256(prompt_token_ids: Sequence[int]) -> str:
    """Hash the canonical JSON encoding of the model-specific prompt token sequence."""
    payload = json.dumps(list(prompt_token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_prompt(prompt: str, prompt_id: str) -> None:
    """Reject empty prompts and bind the reserved collected-smoke prompt identity."""
    if not prompt.strip():
        msg = "prompt must not be empty"
        raise ValueError(msg)
    if not prompt_id.strip():
        msg = "prompt_id must not be empty"
        raise ValueError(msg)
    is_checked_in_prompt = _prompt_text_sha256(prompt) == _CHECKED_IN_FREE_FORM_PROMPT_SHA256
    has_checked_in_id = prompt_id == _FREE_FORM_CAPTION_PROMPT_VERSION
    if is_checked_in_prompt != has_checked_in_id:
        msg = f"{_FREE_FORM_CAPTION_PROMPT_VERSION!r} is reserved for the checked-in prompt text"
        raise ValueError(msg)


def _require_staged_model(model_variant: str) -> str:
    """Return the model ID or skip before preparing expensive sensor inputs."""
    model_id = get_vllm_model_id(model_variant)
    weights_dir = get_local_dir_for_weights_name(model_id)
    try:
        weights_available = weights_dir.is_dir() and any(weights_dir.iterdir())
    except PermissionError:
        weights_available = False
    if not weights_available:
        pytest.skip(f"{model_variant} weights are not staged at {weights_dir}")
    return model_id


# GPU-only imports keep non-default collection free of vLLM and Torch dependencies.
if pixi_utils.is_running_in_env("default"):
    import attrs
    import torch
    from transformers import AutoProcessor
    from vllm.outputs import RequestOutput

    from cosmos_curator.models.vllm_interface import make_model_inputs, process_vllm_output, vllm_generate
    from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import CaptionSingleOptions, VllmCaptionStage
    from cosmos_curator.pipelines.video.utils.data_model import VllmCaptionRequest

    def _pin_processor_temporal_selection(llm_input: dict[str, Any]) -> dict[str, Any]:
        """Disable temporal sampling on the processor request actually sent to vLLM."""
        processor_kwargs = llm_input.setdefault("mm_processor_kwargs", {})
        if not isinstance(processor_kwargs, dict):
            msg = f"mm_processor_kwargs must be a dictionary, got {type(processor_kwargs).__name__}"
            raise TypeError(msg)
        processor_kwargs["do_sample_frames"] = False
        return processor_kwargs

    def _model_input_video_payload(llm_input: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        """Recover one registered plugin's effective video payload and metadata."""
        multi_modal_data = llm_input.get("multi_modal_data")
        if not isinstance(multi_modal_data, dict) or "video" not in multi_modal_data:
            msg = "registered model input must contain multi_modal_data['video']"
            raise TypeError(msg)
        video_payload = multi_modal_data["video"]
        if isinstance(video_payload, list):
            if len(video_payload) != 1:
                msg = f"registered model input must contain one video payload, got {len(video_payload)}"
                raise ValueError(msg)
            video_payload = video_payload[0]
        if not isinstance(video_payload, tuple) or len(video_payload) != 2:
            msg = f"unsupported registered model video payload: {type(video_payload).__name__}"
            raise TypeError(msg)
        frames, metadata = video_payload
        if not isinstance(metadata, dict):
            msg = f"registered model video metadata must be a dictionary, got {type(metadata).__name__}"
            raise TypeError(msg)
        return frames, metadata

    def _video_payload_tchw(frames: object) -> npt.NDArray[np.uint8]:
        """Normalize registered plugin TCHW or THWC payloads for exact hashing."""
        if isinstance(frames, torch.Tensor):
            frames_array = frames.detach().cpu().numpy()
        elif isinstance(frames, np.ndarray):
            frames_array = frames
        else:
            msg = f"unsupported registered model video tensor: {type(frames).__name__}"
            raise TypeError(msg)
        if frames_array.dtype != np.uint8:
            msg = f"registered model video tensor must remain uint8, got {frames_array.dtype}"
            raise TypeError(msg)
        if frames_array.ndim != 4:
            msg = f"registered model video tensor must be four-dimensional, got {frames_array.shape}"
            raise ValueError(msg)
        if frames_array.shape[1] == 3:
            frames_tchw = frames_array
        elif frames_array.shape[-1] == 3:
            frames_tchw = frames_array.transpose(0, 3, 1, 2)
        else:
            msg = f"registered model video tensor must have three channels, got {frames_array.shape}"
            raise ValueError(msg)
        return cast("npt.NDArray[np.uint8]", np.ascontiguousarray(frames_tchw))

    def _assert_video_processor_supports_temporal_override(processor: AutoProcessor) -> None:
        """Require the model's video processor to declare the no-resampling option."""
        video_processor = getattr(processor, "video_processor", None)
        assert video_processor is not None, "registered model processor must expose a video_processor"
        valid_kwargs = getattr(video_processor, "valid_kwargs", None)
        declared_kwargs = getattr(valid_kwargs, "__annotations__", {})
        assert "do_sample_frames" in declared_kwargs, "video processor must declare do_sample_frames"

    def _effective_processor_video_metadata(
        processor_kwargs: dict[str, Any],
        payload_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Use plugin-built processor metadata when supplied, otherwise payload metadata."""
        processor_metadata = processor_kwargs.get("video_metadata")
        if processor_metadata is None:
            return payload_metadata
        if not isinstance(processor_metadata, list) or len(processor_metadata) != 1:
            msg = "processor video_metadata must contain exactly one item"
            raise TypeError(msg)
        metadata = processor_metadata[0]
        if not isinstance(metadata, dict):
            msg = f"processor video metadata must be a dictionary, got {type(metadata).__name__}"
            raise TypeError(msg)
        return metadata

    def _processor_temporal_evidence(
        processor: AutoProcessor,
        processor_kwargs: dict[str, Any],
        processor_metadata: dict[str, Any],
        *,
        frame_count: int,
        sampling_fps: float,
    ) -> dict[str, Any]:
        """Validate and normalize the effective uniform aligned-view timeline."""
        expected_indices = list(range(frame_count))
        expected_duration = frame_count / sampling_fps
        assert processor_metadata["total_num_frames"] == frame_count
        assert processor_metadata["fps"] == sampling_fps
        assert processor_metadata["duration"] == expected_duration
        assert processor_metadata["frames_indices"] == expected_indices
        assert processor_kwargs["do_sample_frames"] is False

        video_processor = getattr(processor, "video_processor", None)
        temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
        temporal_group_count = None
        if isinstance(temporal_patch_size, int) and temporal_patch_size > 0:
            temporal_group_count = (frame_count + temporal_patch_size - 1) // temporal_patch_size

        return {
            "timeline_domain": "aligned_sampled_view",
            "fps": float(processor_metadata["fps"]),
            "duration": float(processor_metadata["duration"]),
            "total_num_frames": int(processor_metadata["total_num_frames"]),
            "frames_indices": processor_metadata["frames_indices"],
            "do_sample_frames": processor_kwargs["do_sample_frames"],
            "temporal_patch_size": temporal_patch_size if isinstance(temporal_patch_size, int) else None,
            "temporal_group_count": temporal_group_count,
        }

    def _collect_generation_evidence(
        outputs: list[RequestOutput],
        request: VllmCaptionRequest,
        vllm_config: VllmConfig,
    ) -> _CaptionGeneration:
        """Retain raw output while deferring any output-processing failure."""
        if not outputs or not outputs[0].outputs:
            msg = "vLLM engine returned no outputs for caption_single_frames."
            raise RuntimeError(msg)
        raw_output = outputs[0].outputs[0]
        raw_text = "" if raw_output.text is None else str(raw_output.text)
        finish_reason = raw_output.finish_reason
        prompt_token_ids = outputs[0].prompt_token_ids
        prompt_tokens = None if prompt_token_ids is None else len(prompt_token_ids)
        model_prompt_token_ids_sha256 = (
            None if prompt_token_ids is None else _model_prompt_token_ids_sha256(prompt_token_ids)
        )
        output_tokens = len(raw_output.token_ids) if raw_output.token_ids else 0
        text = ""
        output_processing_error: Exception | None = None
        try:
            finished = process_vllm_output(outputs, {request.request_id: request}, vllm_config)
        except Exception as error:  # noqa: BLE001 - persist evidence before re-raising output-processing failures
            output_processing_error = error
        else:
            if finished == [request]:
                text = "" if request.caption is None else request.caption
            else:
                output_processing_error = RuntimeError(f"Expected one finished caption request, got {len(finished)}")
        return _CaptionGeneration(
            raw_text=raw_text,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            model_prompt_token_ids_sha256=model_prompt_token_ids_sha256,
            output_tokens=output_tokens,
            output_processing_error=output_processing_error,
        )

    class _FrameInputVllmCaptionStage(VllmCaptionStage):
        """Prototype-only synchronous caption seam for preselected TCHW frames."""

        frame_input_sha256: str | None = None
        frame_input_processor_kwargs: dict[str, Any] | None = None
        frame_input_temporal_metadata: dict[str, Any] | None = None

        def caption_single_frames(
            self,
            prompt: str,
            frames_tchw: npt.NDArray[np.uint8],
            *,
            expected_sha256: str,
            sampling_fps: float,
        ) -> _CaptionGeneration:
            """Caption one preselected frame tensor without decoding or temporal sampling."""
            if self._llm is None:
                msg = "vLLM model not initialised; call stage_setup before caption_single_frames."
                raise RuntimeError(msg)
            if self._processor is None:
                msg = "Processor not initialised; call stage_setup before caption_single_frames."
                raise RuntimeError(msg)
            if self._caption_single_sampling_params is None:
                msg = "caption_single sampling params not built; call stage_setup first."
                raise RuntimeError(msg)

            # Verify exact tensor identity before model-owned preprocessing begins.
            assert frames_tchw.dtype == np.uint8
            assert frames_tchw.ndim == 4
            assert frames_tchw.shape[0] > 0
            assert frames_tchw.shape[1] == 3
            assert frames_tchw.flags.c_contiguous
            assert compute_frame_sha256(frames_tchw) == expected_sha256

            # Preserve every Sensor Library-selected frame in its original order.
            frame_count = int(frames_tchw.shape[0])
            video_metadata: dict[str, Any] = {
                "total_num_frames": frame_count,
                "fps": sampling_fps,
                "duration": frame_count / sampling_fps,
                "video_backend": "opencv_dynamic",
                "frames_indices": list(range(frame_count)),
                "do_sample_frames": False,
            }
            # Delegate all pixel transforms to the registered model processor.
            caption_single_config = attrs.evolve(self._vllm_config, video_max_pixels_per_frame=None)
            assert caption_single_config.preprocess_mode == PreprocessMode.MODEL
            assert caption_single_config.video_max_pixels_per_frame is None
            video_tensor = torch.from_numpy(frames_tchw)
            llm_inputs = make_model_inputs(
                videos=[video_tensor],
                metadata=[video_metadata],
                config=caption_single_config,
                processor=self._processor,
                prompt=prompt,
                debug_render_context=None,
            )
            if not llm_inputs:
                msg = "make_model_inputs produced no inputs for caption_single_frames."
                raise RuntimeError(msg)

            # Pin the actual processor request against any temporal resampling default.
            assert len(llm_inputs) == 1
            model_input_payload, model_input_metadata = _model_input_video_payload(llm_inputs[0])
            model_input_frames = _video_payload_tchw(model_input_payload)
            self.frame_input_sha256 = compute_frame_sha256(model_input_frames)
            assert self.frame_input_sha256 == expected_sha256
            assert model_input_frames.shape[0] == frame_count
            processor_kwargs = _pin_processor_temporal_selection(llm_inputs[0])
            _assert_video_processor_supports_temporal_override(self._processor)
            processor_metadata = _effective_processor_video_metadata(processor_kwargs, model_input_metadata)
            self.frame_input_temporal_metadata = _processor_temporal_evidence(
                self._processor,
                processor_kwargs,
                processor_metadata,
                frame_count=frame_count,
                sampling_fps=sampling_fps,
            )
            self.frame_input_processor_kwargs = processor_kwargs

            # Reuse production generation and plugin decoding for the selected model.
            request = VllmCaptionRequest(
                request_id="sensor-aligned-frame-input",
                inputs=llm_inputs[0],
            )
            outputs = vllm_generate(
                self._llm,
                self._caption_single_sampling_params,
                [request],
                batch_size=1,
            )
            return _collect_generation_evidence(outputs, request, self._vllm_config)


# Artifact utilities retain pre-inference evidence and enforce the response contract.
def _alignment_preflight_artifact(
    episode_input: SensorEpisodeInput,
    descriptor: SensorSessionDescriptor,
    aligned_result: SensorAlignmentResult,
) -> dict[str, Any]:
    """Build the measured and input-derived evidence recorded before inference."""
    diagnostics = aligned_result.diagnostics
    contract = aligned_result.contract
    imu_quality = aligned_result.imu_quality
    return {
        "tag": "sensor-aligned-captioning-e2e",
        "input": {
            "front_source": str(episode_input.front_source),
            "rear_source": str(episode_input.rear_source),
            "camera_pair_mode": episode_input.pair_mode.value,
            "start_ns": episode_input.start_ns,
            "duration_ns": episode_input.duration_ns,
            "sampling_period_ns": episode_input.sampling_period_ns,
        },
        "episode_duration_s": (descriptor.exclusive_end_ns - descriptor.start_ns) / 1_000_000_000,
        "aligned_rows": len(aligned_result.aligned_frame.align_timestamps_ns),
        "camera_frames_selected": diagnostics.camera_frames_selected,
        "vlm_input_frames": len(aligned_result.front_frames_tchw),
        "vlm_input_source": FRONT_CAMERA_SENSOR_ID,
        "sensor_roles": {
            "model_input": [FRONT_CAMERA_SENSOR_ID],
            "alignment_only": [REAR_CAMERA_SENSOR_ID, IMU_SENSOR_ID],
        },
        "max_front_alignment_error_ns": diagnostics.max_front_alignment_error_ns,
        "max_rear_alignment_error_ns": diagnostics.max_rear_alignment_error_ns,
        "max_cross_camera_skew_ns": diagnostics.max_cross_camera_skew_ns,
        "valid_imu_intervals": diagnostics.valid_imu_interval_count,
        "imu_quality": {
            "timestamp_domain_mode": imu_quality.timestamp_domain_mode,
            "valid_interval_count": imu_quality.valid_interval_count,
            "integration_duration_ns": imu_quality.integration_duration_ns,
            "sample_count_total": {
                "min": imu_quality.sample_count_total_min,
                "max": imu_quality.sample_count_total_max,
            },
            "sample_count_used": {
                "min": imu_quality.sample_count_used_min,
                "max": imu_quality.sample_count_used_max,
            },
            "sample_count_rejected": {
                "min": imu_quality.sample_count_rejected_min,
                "max": imu_quality.sample_count_rejected_max,
            },
            "max_inter_sample_gap_ns": imu_quality.max_inter_sample_gap_ns,
            "per_valid_interval_delta_velocity_m_s": list(imu_quality.delta_velocity_m_s),
            "per_valid_interval_delta_position_m": list(imu_quality.delta_position_m),
        },
        "alignment_contract": {
            "front_frame_shape_tchw": list(contract.front_frame_shape_tchw),
            "rear_frame_shape_nhwc": list(contract.rear_frame_shape_nhwc),
            "front_has_bframes": contract.front_has_bframes,
            "rear_has_bframes": contract.rear_has_bframes,
        },
        "frame_identity": {
            "dtype": str(aligned_result.front_frames_tchw.dtype),
            "shape": list(aligned_result.front_frames_tchw.shape),
            "sha256": aligned_result.front_frames_sha256,
        },
        "captioning": None,
        "status": "alignment_preflight",
    }


def _validate_generation(
    generation: _CaptionGeneration,
    max_tokens: int,
) -> None:
    """Require cleanly terminated, non-empty plugin-decoded output."""
    if generation.output_processing_error is not None:
        raise generation.output_processing_error
    assert generation.finish_reason == "stop", (
        f"vLLM generation must finish with 'stop', got {generation.finish_reason!r} "
        f"after {generation.output_tokens}/{max_tokens} output tokens"
    )
    assert generation.output_tokens > 0, "vLLM generation must report at least one output token"
    decoded_text = generation.text.strip()
    assert decoded_text, f"vLLM engine returned an empty caption (finish_reason={generation.finish_reason!r})"


def run_sensor_aligned_episode_captioning(  # noqa: PLR0913 - explicit probe inputs keep local runs reproducible
    episode_input: SensorEpisodeInput,
    artifact_dir: Path,
    *,
    assert_case_alignment: Callable[[SensorAlignmentResult], None],
    model_variant: str = _DEFAULT_MODEL_VARIANT,
    max_tokens: int | None = None,
    prompt: str = _FREE_FORM_CAPTION_PROMPT,
    prompt_id: str = _FREE_FORM_CAPTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run the reusable prompted-captioning path for one local camera-input case."""
    if not pixi_utils.is_running_in_env("default"):
        msg = "sensor-aligned captioning requires the 'default' Pixi environment"
        raise RuntimeError(msg)

    _validate_prompt(prompt, prompt_id)
    prompt_text_sha256 = _prompt_text_sha256(prompt)
    resolved_max_tokens = _resolve_max_tokens(max_tokens)
    model_id = _require_staged_model(model_variant)
    pipeline_default_preprocess_mode = resolve_preprocess_mode(model_variant)

    # Stage 1: produce one aligned episode and its exact front-camera tensor.
    descriptor = prepare_sensor_session(artifact_dir, episode_input)
    aligned_result = align_sensor_session(descriptor)

    # Stage 2: persist and reopen alignment evidence before model setup can fail.
    pre_inference_sha256 = compute_frame_sha256(aligned_result.front_frames_tchw)
    artifact_path = artifact_dir / "sensor_aligned_episode_captioning.json"
    artifact = _alignment_preflight_artifact(episode_input, descriptor, aligned_result)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact

    assert pre_inference_sha256 == aligned_result.front_frames_sha256
    assert_sensor_alignment(aligned_result)
    assert_case_alignment(aligned_result)

    # Stage 3: configure the frame-input adapter for one caller-supplied prompt.
    sampling_config = resolve_vllm_sampling_config(
        model_variant,
        VllmSamplingConfig(min_tokens=0),
    )
    # Use model-owned preprocessing across sensor-aligned probes; retain the model's pipeline default for comparison.
    vllm_config = VllmConfig(
        model_variant=model_variant,
        sampling_config=sampling_config,
        preprocess_mode=PreprocessMode.MODEL,
        batch_size=1,
    )
    stage = _FrameInputVllmCaptionStage(
        vllm_config=vllm_config,
        caption_single_options=CaptionSingleOptions(
            temperature=0.0,
            max_tokens=resolved_max_tokens,
        ),
    )
    sampling_fps = 1_000_000_000 / episode_input.sampling_period_ns

    # Stage 4: run one synchronous generation and always tear down vLLM children.
    try:
        stage.stage_setup()
        generation = stage.caption_single_frames(
            prompt,
            aligned_result.front_frames_tchw,
            expected_sha256=aligned_result.front_frames_sha256,
            sampling_fps=sampling_fps,
        )
    finally:
        stage.destroy()

    # Stage 5: persist raw output and plugin-decode outcome before validating the caption.
    assert stage.frame_input_temporal_metadata is not None
    captioning_artifact = {
        "backend": "vllm_sync",
        "model": model_variant,
        "model_id": model_id,
        "prompt_id": prompt_id,
        "prompt_text_sha256": prompt_text_sha256,
        "model_prompt_token_ids_sha256": generation.model_prompt_token_ids_sha256,
        "preprocessing": {
            "owner": vllm_config.preprocess_mode.value,
            "pipeline_default_owner": pipeline_default_preprocess_mode.value,
            "test_override": vllm_config.preprocess_mode != pipeline_default_preprocess_mode,
            "input_layout": "TCHW",
            "input_dtype": "uint8",
            "video_max_pixels_per_frame": None,
            "processor_do_sample_frames": (
                None
                if stage.frame_input_processor_kwargs is None
                else stage.frame_input_processor_kwargs.get("do_sample_frames")
            ),
        },
        "temporal_input": stage.frame_input_temporal_metadata,
        "sampling": {
            "temperature": 0.0,
            "presence_penalty": sampling_config.presence_penalty,
            "max_tokens": resolved_max_tokens,
            "min_tokens": sampling_config.min_tokens,
        },
        "raw_generated_text": generation.raw_text,
        "generated_text": generation.text,
        "output_processing_error": (
            None
            if generation.output_processing_error is None
            else {
                "type": type(generation.output_processing_error).__name__,
                "message": str(generation.output_processing_error),
            }
        ),
        "finish_reason": generation.finish_reason,
        "prompt_tokens": generation.prompt_tokens,
        "output_tokens": generation.output_tokens,
    }
    artifact["captioning"] = captioning_artifact
    artifact["status"] = "caption_generated"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact

    # Stage 6: validate the generated text, then recheck the frame seam.
    _validate_generation(generation, resolved_max_tokens)
    assert stage.frame_input_sha256 == aligned_result.front_frames_sha256

    # Stage 7: finalize and reopen the serialized passing evidence.
    artifact["status"] = "pass"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    reopened_artifact = cast("dict[str, Any]", json.loads(artifact_path.read_text(encoding="utf-8")))

    assert reopened_artifact == artifact
    assert reopened_artifact["frame_identity"]["sha256"] == stage.frame_input_sha256
    assert reopened_artifact["captioning"]["finish_reason"] == "stop"
    assert reopened_artifact["captioning"]["generated_text"]
    assert reopened_artifact["captioning"]["preprocessing"]["owner"] == PreprocessMode.MODEL.value
    return reopened_artifact


@pytest.mark.env("default")
def test_sensor_aligned_episode_captioning(tmp_path: Path) -> None:
    """Caption exact aligned Sintel frames with model preprocessing deliberately enabled for raw Qwen input."""
    artifact = run_sensor_aligned_episode_captioning(
        CHECKED_IN_CODEC_INPUT,
        tmp_path,
        assert_case_alignment=assert_checked_in_codec_alignment,
    )

    assert artifact["status"] == "pass"
    assert artifact["aligned_rows"] == CHECKED_IN_ALIGNED_ROWS
    assert artifact["sensor_roles"] == {
        "model_input": [FRONT_CAMERA_SENSOR_ID],
        "alignment_only": [REAR_CAMERA_SENSOR_ID, IMU_SENSOR_ID],
    }
    imu_quality = artifact["imu_quality"]
    assert imu_quality["timestamp_domain_mode"] == "shared_synthetic_timeline"
    assert imu_quality["valid_interval_count"] == CHECKED_IN_ALIGNED_ROWS - 1
    assert imu_quality["integration_duration_ns"] == 250_000_000
    assert imu_quality["sample_count_total"] == {"min": 26, "max": 26}
    assert imu_quality["sample_count_used"] == {"min": 26, "max": 26}
    assert imu_quality["sample_count_rejected"] == {"min": 0, "max": 0}
    assert imu_quality["max_inter_sample_gap_ns"] == 10_000_000
    np.testing.assert_allclose(
        imu_quality["per_valid_interval_delta_velocity_m_s"],
        [0.5, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        imu_quality["per_valid_interval_delta_position_m"],
        [0.0625, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    assert artifact["captioning"]["model"] == _DEFAULT_MODEL_VARIANT
    assert artifact["captioning"]["model_id"] == get_vllm_model_id(_DEFAULT_MODEL_VARIANT)
    assert artifact["captioning"]["prompt_id"] == _FREE_FORM_CAPTION_PROMPT_VERSION
    assert _prompt_text_sha256(_FREE_FORM_CAPTION_PROMPT) == _CHECKED_IN_FREE_FORM_PROMPT_SHA256
    assert artifact["captioning"]["prompt_text_sha256"] == _CHECKED_IN_FREE_FORM_PROMPT_SHA256
    assert artifact["captioning"]["raw_generated_text"]
    assert artifact["captioning"]["generated_text"]
    assert artifact["captioning"]["output_processing_error"] is None
    temporal_input = artifact["captioning"]["temporal_input"]
    assert temporal_input["timeline_domain"] == "aligned_sampled_view"
    assert temporal_input["fps"] == 4.0
    assert temporal_input["duration"] == 10.0
    assert temporal_input["total_num_frames"] == CHECKED_IN_ALIGNED_ROWS
    assert temporal_input["frames_indices"] == list(range(CHECKED_IN_ALIGNED_ROWS))
    assert temporal_input["do_sample_frames"] is False
    preprocessing = artifact["captioning"]["preprocessing"]
    assert preprocessing["owner"] == PreprocessMode.MODEL.value
    assert preprocessing["input_layout"] == "TCHW"
    assert preprocessing["input_dtype"] == "uint8"
    assert preprocessing["video_max_pixels_per_frame"] is None
    assert preprocessing["processor_do_sample_frames"] is False
