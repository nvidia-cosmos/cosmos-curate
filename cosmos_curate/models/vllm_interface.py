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
encode_windows_for_vllm(windows, frames, config, processor, prompt)
llm_inputs, caption_mappings = gather_vllm_inputs(videos, model_variant)
vllm_outputs = vllm_generate(model, sampling_params, llm_inputs, batch_size, use_tqdm=use_tqdm)
captions = decode_vllm_outputs(vllm_outputs, model_variant)
scatter_vllm_captions(model_variant, videos, caption_mappings, captions)
free_vllm_inputs(video, model_variant)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from vllm import LLM, RequestOutput, SamplingParams

from cosmos_curate.core.utils.misc import grouping
from cosmos_curate.models.vllm_phi import VLLMPhi4

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from transformers import AutoProcessor

    from cosmos_curate.models.vllm_plugin import VLLMPlugin
    from cosmos_curate.pipelines.video.utils.data_model import (
        Video,
        VLLMConfig,
        Window,
    )


# Add new VLLM plugins to _VLLM_PLUGINS
_VLLM_PLUGINS = {VLLMPhi4.model_variant(): VLLMPhi4}


def _get_vllm_plugin(variant: str) -> VLLMPlugin:
    """Get the VLLM plugin for the model variant.

    Args:
        variant: The variant of the model.

    Returns:
        The VLLM plugin.

    Raises:
        ValueError: If the model variant is not supported.

    """
    plugin = _VLLM_PLUGINS.get(variant)
    if plugin is None:
        msg = f"VLLM Model variant {variant} not supported"
        raise ValueError(msg)
    return cast("VLLMPlugin", plugin)


def vllm_model(config: VLLMConfig) -> LLM:
    """Create a VLLM model instance.

    Args:
       config: Configuration for the VLLM model.

    Returns:
        A VLLM model instance.

    """
    return _get_vllm_plugin(config.variant).model(config)


def sampling_params(config: VLLMConfig) -> SamplingParams:
    """Create a sampling parameters object for the VLLM model.

    Args:
        config: Configuration for the VLLM model.

    Returns:
        A sampling parameters object.

    """
    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_output_tokens,
        stop_token_ids=[],
    )


def auto_processor(config: VLLMConfig) -> AutoProcessor:
    """Get the auto process for the model.

    Args:
        config: The configuration of the model.

    Returns:
        The auto processor for the model.

    """
    return _get_vllm_plugin(config.variant).processor()


def encode_windows_for_vllm(
    windows: list[Window],
    frames: list[torch.Tensor],
    config: VLLMConfig,
    processor: AutoProcessor,
    prompt: str,
) -> None:
    """Encode windows for the VLLM model.

    Takes a list of windows and frames and encodes them for the VLLM model.
    The windows are modified in-place by adding LLM inputs.

    Args:
        windows: The windows to encode.
        frames: The frames to encode.
        config: The configuration for the VLLM model.
        processor: The processor to use for the VLLM model.
        prompt: The prompt to use for the VLLM model.

    Raises:
        ValueError: If the windows and frames are not the same length.

    """
    vllm_plugin = _get_vllm_plugin(config.variant)

    if len(windows) != len(frames):
        msg = f"The number of windows ({len(windows)}) and frames ({len(frames)}) must be the same"
        raise ValueError(msg)

    for window, frame in zip(windows, frames, strict=True):
        llm_input = vllm_plugin.make_llm_input(prompt, frame, processor)
        vllm_plugin.add_llm_input_to_window(window, llm_input)


def gather_vllm_inputs(
    videos: list[Video],
    model_variant: str,
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    """Gather LLM inputs from a list of videos.

    Args:
        videos: The videos to gather LLM inputs from.
        model_variant: The variant of the model.

    Returns:
        * list of llm inputs
        * list of mappings of videos, clips, and windows to the captions, use
          this when calling scatter_vllm_captions

    """
    vllm_plugin = _get_vllm_plugin(model_variant)

    llm_inputs: list[dict[str, Any]] = []
    caption_mappings: list[tuple[int, int, int]] = []
    for video_idx, video in enumerate(videos):
        for clip_idx, clip in enumerate(video.clips):
            for window_idx, window in enumerate(clip.windows):
                llm_input = vllm_plugin.get_llm_input_from_window(window)
                llm_inputs.append(llm_input)
                caption_mappings.append((video_idx, clip_idx, window_idx))

    return llm_inputs, caption_mappings


def vllm_generate(
    llm: LLM,
    sampling_params: SamplingParams,
    data: Iterable[dict[str, Any]],
    batch_size: int,
    *,
    use_tqdm: bool = False,
) -> list[RequestOutput]:
    """Generate captions for the data using the vLLM model.

    Args:
        llm: The vLLM model.
        sampling_params: The sampling parameters.
        data: The data to generate captions for.
        batch_size: The batch size.
        use_tqdm: Whether to use tqdm.

    Returns:
        A list of captions.

    """
    all_outputs = []
    for batch_data in grouping.split_by_chunk_size(data, batch_size):
        # llm.generate can take a list of dicts, but does not advertize this in its type hints
        outputs = llm.generate(batch_data, sampling_params=sampling_params, use_tqdm=use_tqdm)  # type: ignore[arg-type]
        all_outputs.extend(outputs)
    return all_outputs


def decode_vllm_outputs(vllm_outputs: list[RequestOutput], model_variant: str) -> list[str]:
    """Decode the outputs from vllm_generate into a list of captions.

    Args:
        vllm_outputs: The output from vllm_generate
        model_variant: The variant of the model.


    Returns:
        List of captions

    """
    vllm_plugin = _get_vllm_plugin(model_variant)
    return [vllm_plugin.decode(out) for out in vllm_outputs]


def scatter_vllm_captions(
    model_variant: str,
    videos: list[Video],
    mapping: Iterable[tuple[int, int, int]],
    captions: Iterable[str],
    *,
    verbose: bool = False,
) -> None:
    """Scatter captions to the tasks / videos.

    Args:
        model_variant: The variant of the model.
        videos: The videos to assign captions to.
        mapping: The mapping of the [videos, clips, windows] to the captions.
        captions: The captions to assign to the windows
        verbose: whether to print verbose logs.

    """
    for (video_idx, clip_idx, window_idx), caption in zip(mapping, captions, strict=True):
        videos[video_idx].clips[clip_idx].windows[window_idx].caption[model_variant] = caption
        if verbose:
            logger.info(f"Caption for clip {videos[video_idx].clips[clip_idx].uuid} window {window_idx}: {caption}")


def free_vllm_inputs(video: Video, model_variant: str) -> None:
    """Free unused memory for the model variant.

    Args:
        video: The video to free unused memory for.
        model_variant: The variant of the model.

    """
    _get_vllm_plugin(model_variant).free_vllm_inputs(video)


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


def vllm_caption(  # noqa: PLR0913
    videos: list[Video],
    llm: LLM,
    processor: AutoProcessor,
    model_config: VLLMConfig,
    sampling_params: SamplingParams,
    *,
    use_tqdm: bool = False,
) -> int:
    """Caption the videos using the vLLM model.

    Args:
        videos: The videos to caption.
        llm: The vLLM model.
        processor: The processor to use.
        model_config: The configuration for the VLLM model.
        sampling_params: The sampling parameters.
        use_tqdm: Whether to use tqdm.

    Returns:
        The number of captions generated.

    """
    # stage 1 captioning
    llm_inputs, caption_mappings = gather_vllm_inputs(videos, model_config.variant)
    vllm_outputs = vllm_generate(llm, sampling_params, llm_inputs, model_config.batch_size, use_tqdm=use_tqdm)
    captions = decode_vllm_outputs(vllm_outputs, model_config.variant)

    # stage 2 captioning
    if model_config.stage2_caption:
        llm_inputs_refined = [
            make_refined_llm_input(
                caption, prev_input, processor, model_config.variant, model_config.stage2_prompt_text
            )
            for caption, prev_input in zip(captions, llm_inputs, strict=True)
        ]
        vllm_refined_outputs = vllm_generate(
            llm, sampling_params, llm_inputs_refined, model_config.batch_size, use_tqdm=use_tqdm
        )
        captions = decode_vllm_outputs(vllm_refined_outputs, model_config.variant)

    scatter_vllm_captions(model_config.variant, videos, caption_mappings, captions)

    return len(captions)
