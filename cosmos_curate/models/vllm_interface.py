# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cosmos-Curate vLLM interface.

═══════════════════════════════════════════════════════════════════════════════
READING THIS CODE? START HERE - Flow Trace Guide
═══════════════════════════════════════════════════════════════════════════════

Problem this solves:
- Without this interface, every CuratorStage that uses vLLM would need to reimplement:
  * Model loading with correct quantization/tensor parallelism settings
  * Two-stage captioning workflow (initial caption → refinement request → final caption)
  * Batching logic with proper request tracking
  * Model-specific input formatting (Qwen token IDs vs Phi-4 prompts vs ...)

This was leading to code duplication across the input preparation and captioning stages
for each model.

Design decision: Plugin architecture rather than if/elif chain because:
- Models have fundamentally different input formats
- Makes adding new models contained to a single file
- Allows model-specific refinement logic
- Supporting 5+ models, expanding to image/audio captioning
- Plugins can be removed from the code base, e.g. if the user of the code
  cannot have a specific model or related code in their code base

FLOW TRACE - How a video gets captioned (follow this to understand the code):

1. Entry: Pipeline calls vllm_caption() with model_inputs
   └─> Dispatches to _caption_inflight_batching() (typical) or _caption_no_inflight_batching() (fallback)

2. Request Creation: Wraps each input in VllmCaptionRequest
   └─> Includes unique request_id, inputs dict, and optional stage2_prompt

3. Continuous Processing: engine.step() processes requests as they arrive
   └─> Submits requests when capacity available
   └─> Allows interleaved stage 1 and stage 2 processing for better throughput

4. Output Decoding: process_vllm_output() extracts caption text
   └─> Calls plugin.decode() - e.g., cosmos_curate/models/vllm_qwen.py

5. Stage 2 (if needed): Creates refinement request, adds back to queue
   └─> Uses plugin.make_refined_llm_request()
   └─> Example: cosmos_curate/models/vllm_qwen.py

6. Return: List of caption strings

Note: No-inflight batching path (_caption_no_inflight_batching, line ~277) is a
fallback for simpler debugging and testing. Production code uses inflight batching.

Plugin Implementations (model-specific code):
- VllmQwen7B:           cosmos_curate/models/vllm_qwen.py
- VllmPhi4:             cosmos_curate/models/vllm_phi.py
- VllmCosmosReason1VL:  cosmos_curate/models/vllm_cosmos_reason1_vl.py
- Plugin Interface:     cosmos_curate/models/vllm_plugin.py (7 abstract methods)
- Registry:             _VLLM_PLUGINS dict

Public API (what pipeline stages call):
- vllm_model()       - Create model instance
- auto_processor()   - Get model processor
- sampling_params()  - Create sampling config
- make_model_inputs() - Convert frames to model-specific format
- vllm_caption()     - Main captioning function (entry point)
- vllm_generate()    - Lower-level batch generation (no inflight batching)

DEBUG TIP: Set breakpoint in vllm_caption() and step through for complete flow
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import secrets
from collections import deque
from typing import (  # noqa: UP035, remove noqa when we drop support for python 3.10
    TYPE_CHECKING,
    Any,
    Deque,
    Iterable,
    TypeVar,
    cast,
)

from vllm import LLM, PoolingOutput, PoolingRequestOutput, RequestOutput, SamplingParams
from vllm.sampling_params import RequestOutputKind

from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.models.vllm_cosmos_reason1_vl import VllmCosmosReason1VL
from cosmos_curate.models.vllm_nemotron import VllmNemotronNano12Bv2VL
from cosmos_curate.models.vllm_phi import VllmPhi4
from cosmos_curate.models.vllm_qwen import VllmQwen7B
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest, WindowConfig

if TYPE_CHECKING:
    import torch
    from transformers import AutoProcessor

    from cosmos_curate.models.vllm_plugin import VllmPlugin
    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig


# Add new vLLM plugins to _VLLM_PLUGINS
_VLLM_PLUGINS = {
    VllmCosmosReason1VL.model_variant(): VllmCosmosReason1VL,
    VllmNemotronNano12Bv2VL.model_variant(): VllmNemotronNano12Bv2VL,
    VllmPhi4.model_variant(): VllmPhi4,
    VllmQwen7B.model_variant(): VllmQwen7B,
}

T = TypeVar("T")


def _get_vllm_plugin(variant: str) -> VllmPlugin:
    """Get the vLLM plugin for the model variant.

    Args:
        variant: The variant of the model.

    Returns:
        The vLLM plugin.

    Raises:
        ValueError: If the model variant is not supported.

    """
    plugin = _VLLM_PLUGINS.get(variant)
    if plugin is None:
        msg = f"vLLM model variant {variant} not supported"
        raise ValueError(msg)
    return cast("VllmPlugin", plugin)


def vllm_model(config: VllmConfig) -> LLM:
    """Create a vLLM model instance.

    Args:
       config: Configuration for the vLLM model.

    Returns:
        A vLLM model instance.

    """
    return _get_vllm_plugin(config.model_variant).model(config)


def sampling_params(config: VllmConfig) -> SamplingParams:
    """Create a sampling parameters object for the vLLM model.

    Args:
        config: Configuration for the vLLM model.

    Returns:
        A sampling parameters object.

    """
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_output_tokens,
        stop_token_ids=[],
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def auto_processor(config: VllmConfig) -> AutoProcessor:
    """Get the auto process for the model.

    Args:
        config: The configuration of the model.

    Returns:
        The auto processor for the model.

    """
    return _get_vllm_plugin(config.model_variant).processor()


def make_metadata(frames: Iterable[torch.Tensor], window_config: WindowConfig) -> list[dict[str, Any]]:
    """Make metadata for a iterable of frames.

    Args:
        frames: The frames to make metadata for.
        window_config: The window configuration to use for the metadata.

    Returns:
        The metadata for the frames.

    """
    # Verify that all tensors are 4D
    NUM_EXPECTED_DIMS = 4
    for i, f in enumerate(frames):
        if f.ndim != NUM_EXPECTED_DIMS:
            msg = (
                f"Expected all frames to have 4 dimensions (batch of videos of shape [num_frames, C, H, W]), "
                f"but frames[{i}] has shape {getattr(f, 'shape', None)}"
            )
            raise ValueError(msg)

    def _make_metadata(frames: torch.Tensor) -> dict[str, Any]:
        fps = window_config.sampling_fps
        num_frames = frames.shape[0]

        return {
            "fps": fps,
            "duration": num_frames / fps,
            "width": frames.shape[3],
            "height": frames.shape[2],
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        }

    return [_make_metadata(frames) for frames in frames]


def make_model_inputs(
    videos: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    config: VllmConfig,
    processor: AutoProcessor,
    prompt: str,
) -> list[dict[str, Any]]:
    """Make model inputs for a list of videos.

    Args:
        videos: list of decoded videos
        metadata: The metadata for each video
        config: The configuration for the vLLM model.
        processor: The processor to use for the vLLM model.
        prompt: The prompt to use for the vLLM model.

    Returns:
        A list of LLM inputs for each video

    """
    vllm_plugin = _get_vllm_plugin(config.model_variant)
    return [
        vllm_plugin.make_llm_input(prompt, frames, md, processor) for frames, md in zip(videos, metadata, strict=True)
    ]


def vllm_generate(
    llm: LLM,
    sampling_params: SamplingParams,
    requests: list[VllmCaptionRequest],
    batch_size: int,
) -> list[RequestOutput]:
    """Generate captions for the data using the vLLM model.

    Args:
        llm: The vLLM model.
        sampling_params: The sampling parameters.
        requests: The captioning requests for the llm
        batch_size: The batch size.

    Returns:
        A list of captions.

    """
    inputs = [r.inputs for r in requests]
    all_outputs: list[RequestOutput] = []

    for batch_data in grouping.split_by_chunk_size(inputs, batch_size):
        # llm.generate can take a list of dicts, but does not advertize this in its type hints
        outputs = llm.generate(batch_data, sampling_params=sampling_params, use_tqdm=False)  # type: ignore[arg-type]
        all_outputs.extend(outputs)

    # Change request ids from integer strings to the vllm_interface unique request ids.
    # Zip is safe because the requests and outputs are in the same order
    for out, req in zip(all_outputs, requests, strict=True):
        out.request_id = req.request_id

    return all_outputs


def process_vllm_output(
    engine_output: list[RequestOutput | PoolingRequestOutput[PoolingOutput]],
    in_flight_requests: dict[str, VllmCaptionRequest],
    vllm_config: VllmConfig,
) -> list[VllmCaptionRequest]:
    """Process vLLM engine output, updating the in-flight requests with the decoded text.

    The output comes from vLLM engine.step().

    Args:
        engine_output: The output from the engine.
        in_flight_requests: The in-flight requests, keyed by request_id.
        vllm_config: The configuration for the VLLM model.

    Returns:
        A list of finished requests.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)
    finished: list[VllmCaptionRequest] = []

    for out in engine_output:
        if not isinstance(out, RequestOutput):
            msg = f"Expected RequestOutput, got {type(out)}. If you are using a pooling model, this is not supported."
            raise TypeError(msg)

        if out.finished:
            request = in_flight_requests[out.request_id]
            request.caption = vllm_plugin.decode(out)
            finished.append(request)

    return finished


def _caption_no_inflight_batching(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    stage2_prompts: list[str | None],
) -> list[str]:
    """Caption the videos without inflight batching.

    Assumption:
       len(model_inputs) == len(stage2_prompts)

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the vLLM model.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Assumed to be the same length as model_inputs.

    Returns:
        Captions for each video.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)

    requests = [
        VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=model_input,
            stage2_prompt=stage2_prompt,
        )
        for model_input, stage2_prompt in zip(model_inputs, stage2_prompts, strict=True)
    ]

    def _process_requests(requests: list[VllmCaptionRequest]) -> list[VllmCaptionRequest]:
        in_flight_requests: dict[str, VllmCaptionRequest] = {r.request_id: r for r in requests}
        outputs = vllm_generate(llm, sampling_params, requests, vllm_config.batch_size)
        finished_requests = process_vllm_output(outputs, in_flight_requests, vllm_config)  # type: ignore[arg-type]

        # Sanity check
        if len(finished_requests) != len(requests):
            msg = f"Expected {len(requests)} finished requests, got {len(finished_requests)}, this is a bug"
            raise RuntimeError(msg)

        return finished_requests

    # stage 1 captioning
    finished_s1 = _process_requests(requests)
    finished = [r for r in finished_s1 if r.stage2_prompt is None]
    needs_stage2 = [r for r in finished_s1 if r.stage2_prompt is not None]

    # stage 2 captioning
    refine_requests = [vllm_plugin.make_refined_llm_request(r, processor, r.stage2_prompt) for r in needs_stage2]

    finished += _process_requests(refine_requests)

    return [request.caption or "Unknown caption" for request in finished]


def _caption_inflight_batching(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    stage2_prompts: list[str | None],
) -> list[str]:
    """Caption the videos using inflight batching.

    Assumption:
       len(model_inputs) == len(stage2_prompts)

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Assumed to be the same length as model_inputs.

    Returns:
        Captions for each video.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.model_variant)
    request_q: Deque[VllmCaptionRequest] = deque()  # noqa: UP006, remove noqa when python 3.10 support is dropped
    in_flight_requests: dict[str, VllmCaptionRequest] = {}
    captions: list[str] = []

    for model_input, stage2_prompt in zip(model_inputs, stage2_prompts, strict=True):
        request_q.append(
            VllmCaptionRequest(
                request_id=secrets.token_hex(8),
                inputs=model_input,
                stage2_prompt=stage2_prompt,
            )
        )

    total_requests = len(request_q)
    engine = llm.llm_engine

    while len(captions) < total_requests:
        if request_q and (max_inflight_requests == 0 or len(in_flight_requests) < max_inflight_requests):
            request = request_q.popleft()
            # engine.add_request can accept a dictionary, but does not advertise this in its type hints
            engine.add_request(request.request_id, request.inputs, sampling_params)  # type: ignore[arg-type]
            in_flight_requests[request.request_id] = request

        engine_output = engine.step()

        # Finished requests are requests that have been completed by the vLLM engine and have a caption.
        # These requests may still need stage2 refinement.
        finished = process_vllm_output(engine_output, in_flight_requests, vllm_config)

        for request in finished:
            del in_flight_requests[request.request_id]

        captions += [r.caption or "Unknown caption" for r in finished if r.stage2_prompt is None]
        needs_stage2 = [r for r in finished if r.stage2_prompt is not None]

        for request in needs_stage2:
            refined_request = vllm_plugin.make_refined_llm_request(request, processor, request.stage2_prompt)
            request_q.append(refined_request)

    return captions


def vllm_caption(  # noqa: PLR0913
    model_inputs: list[dict[str, Any]],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    *,
    inflight_batching: bool,
    stage2_prompts: list[str | None] | None = None,
) -> list[str]:
    """Caption the videos using the vLLM model.

    This is the main entry point for video captioning. It handles:
    1. Creating VllmCaptionRequest objects (each with unique ID)
    2. Batching and generating captions via vLLM
    3. Two-stage captioning if stage2_prompts provided
    4. Returning final caption strings

    Flow: This function → _caption_[no_]inflight_batching() → vllm_generate()
          → process_vllm_output() → plugin.decode() → captions

    Args:
        model_inputs: The model inputs for each video.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        inflight_batching: Whether to use inflight batching.
        stage2_prompts: A list of second-stage prompts to use for the
           captioning. If None, no second-stage captioning will be performed.
           Must be the same length as model_inputs.

    Returns:
        Captions for each video.

    Raises:
        ValueError: If max_inflight_requests is negative.
        ValueError: If stage2_prompts is not None and not the same length as model_inputs.

    """
    if max_inflight_requests < 0:
        msg = f"{max_inflight_requests=} must be >= 0"
        raise ValueError(msg)

    if stage2_prompts is None:
        stage2_prompts = [None] * len(model_inputs)

    if len(stage2_prompts) != len(model_inputs):
        msg = f"{len(stage2_prompts)=} != {len(model_inputs)=}, must be same length"
        raise ValueError(msg)

    if inflight_batching:
        return _caption_inflight_batching(
            model_inputs, llm, processor, sampling_params, vllm_config, max_inflight_requests, stage2_prompts
        )

    return _caption_no_inflight_batching(
        model_inputs,
        llm,
        processor,
        sampling_params,
        vllm_config,
        stage2_prompts,
    )
