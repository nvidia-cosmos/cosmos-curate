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

Example usage:

```python
model = vllm_model(config)
sampling_params = sampling_params(config)
processor = auto_processor(config)
prep_windows_for_vllm(windows, frames, config, processor, prompt)
llm_inputs, caption_mappings = gather_vllm_inputs(videos, model_variant)
vllm_outputs = vllm_generate(model, sampling_params, llm_inputs, batch_size, use_tqdm=use_tqdm)
captions = decode_vllm_outputs(vllm_outputs, model_variant)
scatter_vllm_captions(model_variant, videos, caption_mappings, captions)
free_vllm_inputs(video, model_variant)
```
"""

from __future__ import annotations

import secrets
from collections import deque
from typing import (  # noqa: UP035, remove noqa when we drop support for python 3.10
    TYPE_CHECKING,
    Any,
    Deque,
    TypeVar,
    cast,
)

from loguru import logger
from vllm import LLM, PoolingOutput, PoolingRequestOutput, RequestOutput, SamplingParams
from vllm.sampling_params import RequestOutputKind

from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.models.vllm_cosmos_reason1_vl import VllmCosmosReason1VL
from cosmos_curate.models.vllm_phi import VllmPhi4
from cosmos_curate.models.vllm_qwen import VllmQwen7B
from cosmos_curate.pipelines.video.utils.data_model import Video, VllmCaptionRequest

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch
    from transformers import AutoProcessor

    from cosmos_curate.models.vllm_plugin import VllmPlugin
    from cosmos_curate.pipelines.video.utils.data_model import (
        VllmConfig,
        Window,
    )


# Add new vLLM plugins to _VLLM_PLUGINS
_VLLM_PLUGINS = {
    VllmPhi4.model_variant(): VllmPhi4,
    VllmQwen7B.model_variant(): VllmQwen7B,
    VllmCosmosReason1VL.model_variant(): VllmCosmosReason1VL,
}

T = TypeVar("T")

_CAPTION_STAGE2_MAX_ITERATIONS = 2


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
    return _get_vllm_plugin(config.variant).model(config)


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
    return _get_vllm_plugin(config.variant).processor()


def prep_windows_for_vllm(
    windows: list[Window],
    frames: list[torch.Tensor],
    config: VllmConfig,
    processor: AutoProcessor,
    prompt: str,
) -> None:
    """Prepare windows for the vLLM model.

    Takes a list of windows and frames and prepares them for the vLLM model.
    The windows are modified in-place by adding LLM inputs.

    Args:
        windows: The windows to encode.
        frames: The frames to encode.
        config: The configuration for the vLLM model.
        processor: The processor to use for the vLLM model.
        prompt: The prompt to use for the vLLM model.

    Raises:
        ValueError: If the windows and frames are not the same length.

    """
    vllm_plugin = _get_vllm_plugin(config.variant)

    if len(windows) != len(frames):
        msg = f"The number of windows ({len(windows)}) and frames ({len(frames)}) must be the same"
        raise ValueError(msg)

    for window, frame in zip(windows, frames, strict=True):
        llm_input = vllm_plugin.make_llm_input(prompt, frame, processor)
        window.model_input[vllm_plugin.model_variant()] = llm_input


def gather_vllm_requests(
    videos: list[Video],
    vllm_config: VllmConfig,
) -> Generator[VllmCaptionRequest, None, None]:
    """Gather vLLM requests from the videos.

    Args:
        videos: The videos to gather requests from.
        vllm_config: The configuration for the VLLM model.

    Returns:
        A generator of vLLM requests.

    Raises:
        RuntimeError: If a window has no prepared inputs.

    """
    vllm_plugin = _get_vllm_plugin(vllm_config.variant)

    for video_idx, video in enumerate(videos):
        for clip_idx, clip in enumerate(video.clips):
            if not clip.windows:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["clip_windowing"] = "empty"
                continue

            for window_idx, window in enumerate(clip.windows):
                llm_input = window.model_input.get(vllm_plugin.model_variant())

                if not llm_input:
                    logger.error(f"Clip {clip.uuid} window {window_idx} has no prepared inputs.")
                    clip.errors[f"window-{window_idx}"] = "empty"
                    continue

                yield VllmCaptionRequest(
                    request_id=secrets.token_hex(8),
                    inputs=llm_input,
                    video_idx=video_idx,
                    clip_idx=clip_idx,
                    window_idx=window_idx,
                )


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
        use_tqdm: Whether to use tqdm.

    Returns:
        A list of captions.

    """
    inputs = [r.inputs for r in requests]
    all_outputs: list[RequestOutput] = []

    for batch_data in grouping.split_by_chunk_size(inputs, batch_size):
        # llm.generate can take a list of dicts, but does not advertize this in its type hints
        outputs = llm.generate(batch_data, sampling_params=sampling_params, use_tqdm=False)  # type: ignore[arg-type]
        all_outputs.extend(outputs)

    # Change request ids from integer strings to the vllm_interface unique request ids
    # zip is safe because the requests and outputs are in the same order
    for out, req in zip(all_outputs, requests, strict=True):
        out.request_id = req.request_id

    return all_outputs


def scatter_vllm_captions(
    model_variant: str,
    videos: list[Video],
    requests: list[VllmCaptionRequest],
    *,
    verbose: bool = False,
) -> None:
    """Scatter finished requests and their captions to the videos.

    The finished requests come from process_vllm_output(). Requests are
    expected to be finished, and an exception will be raised if this is not
    the case.

    Args:
        model_variant: The variant of the model.
        videos: The videos to assign captions to.
        requests: The requests to scatter.
        verbose: whether to print verbose logs.

    Raises:
        RuntimeError: If a request is not finished.

    """
    for request in requests:
        if not request.finished:
            # If this happens, this means there's a bug, and it should be fixed.
            msg = f"Request {request.request_id} is not finished"
            raise RuntimeError(msg)

        caption = request.caption if request.caption is not None else "No caption"
        videos[request.video_idx].clips[request.clip_idx].windows[request.window_idx].caption[model_variant] = caption
        if verbose:
            logger.info(
                f"Caption for clip {videos[request.video_idx].clips[request.clip_idx].uuid} window "
                f"{request.window_idx}: {caption}"
            )


def free_vllm_inputs(video: Video, model_variant: str) -> None:
    """Free unused memory for the model variant.

    Args:
        video: The video to free unused memory for.
        model_variant: The variant of the model.

    """
    vllm_plugin = _get_vllm_plugin(model_variant)
    for clip in video.clips:
        for window in clip.windows:
            window.model_input.pop(vllm_plugin.model_variant(), None)


def make_refined_llm_input(
    caption: str,
    prev_input: dict[str, Any],
    processor: AutoProcessor,
    model_variant: str,
    refine_prompt: str | None = None,
) -> dict[str, Any]:
    """Make a refined LLM input.

    Args:
        caption: The caption to refine.
        prev_input: The previous input.
        processor: The processor to use.
        model_variant: The variant of the model.
        refine_prompt: The prompt to use to refine the caption.

    Returns:
        A refined LLM input.

    """
    return _get_vllm_plugin(model_variant).make_refined_llm_input(caption, prev_input, processor, refine_prompt)


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
    vllm_plugin = _get_vllm_plugin(vllm_config.variant)
    finished: list[VllmCaptionRequest] = []

    for out in engine_output:
        if not isinstance(out, RequestOutput):
            msg = f"Expected RequestOutput, got {type(out)}. If you are using a pooling model, this is not supported."
            raise TypeError(msg)

        if out.finished:
            request = in_flight_requests[out.request_id]
            request.caption = vllm_plugin.decode(out)
            request.iterations += 1
            request.finished = True
            finished.append(request)

    return finished


def _caption_no_inflight_batching(  # noqa: PLR0913
    videos: list[Video],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    model_config: VllmConfig,
    *,
    verbose: bool = False,
) -> int:
    """Caption the videos without inflight batching.

    Args:
        videos: The videos to caption.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        model_config: The configuration for the vLLM model.
        verbose: whether to print verbose logs.

    Returns:
        The number of captions generated.

    """
    vllm_plugin = _get_vllm_plugin(model_config.variant)
    requests = list(gather_vllm_requests(videos, model_config))

    if not requests:
        logger.warning("No valid vLLM inputs found in any videos")
        return 0

    def _process_requests(requests: list[VllmCaptionRequest]) -> list[VllmCaptionRequest]:
        in_flight_requests: dict[str, VllmCaptionRequest] = {r.request_id: r for r in requests}
        outputs = vllm_generate(llm, sampling_params, requests, model_config.batch_size)
        finished_requests = process_vllm_output(outputs, in_flight_requests, model_config)  # type: ignore[arg-type]

        # Sanity check
        if len(finished_requests) != len(requests):
            msg = f"Expected {len(requests)} finished requests, got {len(finished_requests)}"
            raise RuntimeError(msg)

        return finished_requests

    # stage 1 captioning
    finished_requests = _process_requests(requests)

    if model_config.stage2_caption:
        # stage 2 captioning
        refined_requests = [
            vllm_plugin.make_refined_llm_request(r, processor, model_config.stage2_prompt_text)
            for r in finished_requests
        ]
        finished_requests = _process_requests(refined_requests)

    scatter_vllm_captions(model_config.variant, videos, finished_requests, verbose=verbose)

    return len(finished_requests)


def _caption_inflight_batching(  # noqa: PLR0913
    videos: list[Video],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    *,
    verbose: bool = False,
) -> int:
    """Caption the videos using inflight batching.

    Args:
        videos: The videos to caption.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        verbose: whether to print verbose logs.

    Returns:
        The number of captions generated.

    """
    if max_inflight_requests < 0:
        msg = f"{max_inflight_requests=} must be >= 0"
        raise ValueError(msg)

    vllm_plugin = _get_vllm_plugin(vllm_config.variant)
    request_q: Deque[VllmCaptionRequest] = deque()  # noqa: UP006, remove noqa when python 3.10 support is dropped
    in_flight_requests: dict[str, VllmCaptionRequest] = {}

    for request in gather_vllm_requests(videos, vllm_config):
        request_q.append(request)

    finished = 0
    total_requests = len(request_q)
    engine = llm.llm_engine

    while finished < total_requests:
        if request_q and (max_inflight_requests == 0 or len(in_flight_requests) < max_inflight_requests):
            request = request_q.popleft()
            # engine.add_request can accept a dictionary, but does not advertise this in its type hints
            engine.add_request(request.request_id, request.inputs, sampling_params)  # type: ignore[arg-type]
            in_flight_requests[request.request_id] = request

        engine_output = engine.step()
        finished_requests = process_vllm_output(engine_output, in_flight_requests, vllm_config)

        for request in finished_requests:
            del in_flight_requests[request.request_id]

        if vllm_config.stage2_caption:
            _finished_requests = [r for r in finished_requests if r.iterations >= _CAPTION_STAGE2_MAX_ITERATIONS]
            outstanding_requests = [r for r in finished_requests if r.iterations < _CAPTION_STAGE2_MAX_ITERATIONS]

            for request in outstanding_requests:
                refined_request = vllm_plugin.make_refined_llm_request(
                    request, processor, vllm_config.stage2_prompt_text
                )
                request_q.append(refined_request)

            finished_requests = _finished_requests

        scatter_vllm_captions(vllm_config.variant, videos, finished_requests, verbose=verbose)
        finished += len(finished_requests)

    return finished


def vllm_caption(  # noqa: PLR0913
    videos: list[Video],
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    vllm_config: VllmConfig,
    max_inflight_requests: int,
    *,
    inflight_batching: bool,
    verbose: bool = False,
) -> int:
    """Caption the videos using the vLLM model.

    Args:
        videos: The videos to caption.
        llm: The vLLM model.
        processor: The processor to use.
        sampling_params: The sampling parameters.
        vllm_config: The configuration for the VLLM model.
        max_inflight_requests: Maximum number of inflight requests to vLLM
           engine. Set to 0 for unlimited inflight requests.
        inflight_batching: Whether to use inflight batching.
        verbose: Whether to print verbose logs.

    Returns:
        The number of captions generated.

    """
    if inflight_batching:
        return _caption_inflight_batching(
            videos, llm, processor, sampling_params, vllm_config, max_inflight_requests, verbose=verbose
        )
    return _caption_no_inflight_batching(videos, llm, processor, sampling_params, vllm_config, verbose=verbose)
