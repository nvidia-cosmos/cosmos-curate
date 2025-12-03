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
"""Test vllm_interface.py."""

from unittest.mock import MagicMock, patch

import pytest

from cosmos_curate.core.utils.model import conda_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    VllmCaptionRequest,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)

if conda_utils.is_running_in_env("unified"):
    import torch
    from vllm import SamplingParams

    from cosmos_curate.models.vllm_interface import (
        _VLLM_PLUGINS,
        _caption_inflight_batching,
        _caption_no_inflight_batching,
        _get_vllm_plugin,
        auto_processor,
        make_metadata,
        make_model_inputs,
        process_vllm_output,
        sampling_params,
        vllm_caption,
        vllm_generate,
        vllm_model,
    )
    from tests.utils.vllm_mock import MockLLM, MockVllmPlugin, make_request_output

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:
    VALID_VARIANTS = []


@pytest.fixture(autouse=True)
def patch_vllm_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _VLLM_PLUGINS dict with {"mock": MockVllmPlugin} for every test in this module."""
    if conda_utils.is_running_in_env("unified"):
        monkeypatch.setitem(
            _VLLM_PLUGINS,
            "mock",
            MockVllmPlugin,
        )
        for k in list(_VLLM_PLUGINS.keys()):
            if k != "mock":
                monkeypatch.delitem(_VLLM_PLUGINS, k)


@pytest.mark.env("unified")
def test_get_vllm_plugin_raises() -> None:
    """Test _get_vllm_plugin raises ValueError for invalid variant."""
    with pytest.raises(ValueError, match=r".*"):
        _get_vllm_plugin("invalid")


@pytest.mark.env("unified")
def test_vllm_model() -> None:
    """vllm_model should return "llm"."""
    cfg = VllmConfig(model_variant="mock")
    assert isinstance(vllm_model(cfg), MockLLM)


@pytest.mark.env("unified")
def test_sampling_params() -> None:
    """Test sampling_params."""
    temperature = 0.1
    top_p = 0.2
    repetition_penalty = 1.3
    max_tokens = 1024
    vllm_config = VllmConfig(
        model_variant="mock",
        sampling_config=VllmSamplingConfig(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        ),
    )
    sp = sampling_params(vllm_config.sampling_config)
    assert isinstance(sp, SamplingParams)
    assert sp.temperature == temperature
    assert sp.top_p == top_p
    assert sp.repetition_penalty == repetition_penalty
    assert sp.max_tokens == max_tokens


@pytest.mark.env("unified")
def test_auto_processor() -> None:
    """Test auto_processor.

    This ensures that the plugin's auto_processor is called.
    """
    vllm_config = VllmConfig(model_variant="mock")
    assert auto_processor(vllm_config) is not None


@pytest.mark.env("unified")
def test_make_metadata() -> None:
    """Test make_metadata."""
    width = 32
    height = 64
    num_videos = 5
    num_frames = 2
    fps = 1.0
    frames = [torch.zeros((num_frames, 3, height, width)) for _ in range(num_videos)]
    window_config = WindowConfig(sampling_fps=fps)
    metadata = make_metadata(frames, window_config)
    assert len(metadata) == len(frames)
    for i in range(len(frames)):
        assert metadata[i]["fps"] == fps
        assert metadata[i]["duration"] == num_frames / fps
        assert metadata[i]["width"] == width
        assert metadata[i]["height"] == height
        assert metadata[i]["total_num_frames"] == num_frames
        assert metadata[i]["frames_indices"] == list(range(num_frames))
        assert not metadata[i]["do_sample_frames"]


@pytest.mark.env("unified")
def test_make_metadata_raises() -> None:
    """Test make_metadata raises ValueError for non-4D tensors."""
    with pytest.raises(ValueError, match=r".*"):
        make_metadata([torch.zeros((3, 32, 32))], WindowConfig(sampling_fps=1.0))


@pytest.mark.env("unified")
def test_make_model_inputs() -> None:
    """Test make_model_inputs."""
    videos = [torch.zeros((2, 3, 32, 32)) for _ in range(5)]
    metadata = [{"fps": 1.0}] * len(videos)
    config = VllmConfig(model_variant="mock")
    processor = MagicMock()
    prompt = "p"
    output = make_model_inputs(videos, metadata, config, processor, prompt)
    assert len(output) == len(videos)
    for i in range(len(videos)):
        ot = output[i]
        assert "prompt" in ot
        assert "multi_modal_data" in ot
        assert ot["prompt"] == prompt

        mm_data = ot["multi_modal_data"]
        assert mm_data["video"][0][0].shape == (2, 3, 32, 32)
        assert mm_data["video"][0][1] == metadata[i]


@pytest.mark.env("unified")
def test_vllm_generate() -> None:
    """Test vllm_generate."""
    llm = MockLLM()
    sampling_params = SamplingParams(max_tokens=100)
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    captions = vllm_generate(llm, sampling_params, requests, batch_size=2)  # type: ignore[arg-type]
    assert len(captions) == len(requests)
    assert [c.request_id for c in captions] == [r.request_id for r in requests]
    captions_text = [c.outputs[0].text for c in captions]
    expected_captions = [f"mock-caption-{i}" for i in range(len(requests))]
    assert captions_text == expected_captions


@pytest.mark.env("unified")
def test_process_vllm_output() -> None:
    """Test process_vllm_output."""
    vllm_plugin = _get_vllm_plugin("mock")
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    engine_output = [make_request_output(f"id{i}", f"mock-caption-{i}") for i in range(5)]
    inflight_requests = {r.request_id: r for r in requests}
    vllm_config = VllmConfig(model_variant="mock")
    finished = process_vllm_output(engine_output, inflight_requests, vllm_config)  # type: ignore[arg-type]
    assert len(finished) == len(engine_output)
    assert [r.request_id for r in finished] == [r.request_id for r in engine_output]
    assert [r.caption for r in finished] == [vllm_plugin.decode(r) for r in engine_output]


@pytest.mark.env("unified")
def test_process_vllm_output_raises() -> None:
    """Test process_vllm_output raises TypeError for non-RequestOutput output."""
    request_output = [object()]
    with pytest.raises(TypeError, match=r".*"):
        process_vllm_output(request_output, {}, VllmConfig(model_variant="mock"))  # type: ignore[arg-type]


@pytest.mark.env("unified")
def test_process_vllm_output_not_finished() -> None:
    """Test that no output is returned when no requests are finished."""
    requests = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    engine_output = [make_request_output(f"id{i}", f"mock-caption-{i}", finished=False) for i in range(5)]
    inflight_requests = {r.request_id: r for r in requests}
    vllm_config = VllmConfig(model_variant="mock")
    finished = process_vllm_output(engine_output, inflight_requests, vllm_config)  # type: ignore[arg-type]
    assert len(finished) == 0


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_no_inflight_batching(*, stage2: bool) -> None:
    """Test _caption_no_inflight_batching."""
    vllm_config = VllmConfig(model_variant="mock")
    model = vllm_model(vllm_config)
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    model_inputs = [{"a": 1}, {"b": 2}]
    stage2_prompts: list[str | None] = [None, None] if not stage2 else ["ref", "ref"]

    captions = _caption_no_inflight_batching(
        model_inputs=model_inputs,
        llm=model,
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
    )

    if stage2:
        off = len(model_inputs)
        expected_captions = [f"mock-caption-{i + off}" for i in range(len(model_inputs))]
    else:
        expected_captions = [f"mock-caption-{i}" for i in range(len(model_inputs))]

    assert captions == expected_captions


@pytest.mark.env("unified")
@patch("cosmos_curate.models.vllm_interface.process_vllm_output")
def test_caption_no_inflight_batching_raises(mock_process_vllm_output: MagicMock) -> None:
    """Test _caption_no_inflight_batching raises RuntimeError on length mismatch."""
    mock_process_vllm_output.return_value = []
    vllm_config = VllmConfig(model_variant="mock")
    llm = vllm_model(vllm_config)
    with pytest.raises(RuntimeError, match=r".*"):
        _caption_no_inflight_batching(
            model_inputs=[{"a": 1}],
            llm=llm,
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=vllm_config,
            stage2_prompts=[None],
        )


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
def test_caption_inflight_batching(*, stage2: bool) -> None:
    """Test _caption_inflight_batching."""
    vllm_config = VllmConfig(model_variant="mock")
    model = vllm_model(vllm_config)
    processor = auto_processor(vllm_config)
    sp = sampling_params(vllm_config.sampling_config)

    model_inputs = [{"a": 1}, {"b": 2}]
    stage2_prompts: list[str | None] = [None, None] if not stage2 else ["ref", "ref"]

    captions = _caption_inflight_batching(
        model_inputs=model_inputs,
        llm=model,
        processor=processor,
        sampling_params=sp,
        vllm_config=vllm_config,
        stage2_prompts=stage2_prompts,
        max_inflight_requests=0,
    )

    if stage2:
        off = len(model_inputs)
        expected_captions = [f"mock-caption-{i + off}" for i in range(len(model_inputs))]
    else:
        expected_captions = [f"mock-caption-{i}" for i in range(len(model_inputs))]

    assert captions == expected_captions


@pytest.mark.env("unified")
def test_vllm_caption_negative_inflight_raises() -> None:
    """vllm_caption should validate non-negative inflight param."""
    with pytest.raises(ValueError, match=r"must be >= 0"):
        vllm_caption(
            model_inputs=[{}],
            llm=MagicMock(),
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=VllmConfig(model_variant="qwen"),
            max_inflight_requests=-1,
            inflight_batching=True,
            stage2_prompts=[None],
        )


@pytest.mark.env("unified")
def test_vllm_caption_stage2_prompts_mismatch_raises() -> None:
    """vllm_caption should validate stage2_prompts length matches model_inputs length."""
    with pytest.raises(ValueError, match=r"must be same length"):
        vllm_caption(
            model_inputs=[{}],
            llm=MagicMock(),
            processor=MagicMock(),
            sampling_params=MagicMock(),
            vllm_config=VllmConfig(model_variant="qwen"),
            max_inflight_requests=0,
            inflight_batching=True,
            stage2_prompts=[None, None],
        )


@pytest.mark.env("unified")
@pytest.mark.parametrize("inflight", [False, True])
@patch("cosmos_curate.models.vllm_interface._caption_inflight_batching")
@patch("cosmos_curate.models.vllm_interface._caption_no_inflight_batching")
def test_vllm_caption_dispatch(mock_no_ifb: MagicMock, mock_ifb: MagicMock, *, inflight: bool) -> None:
    """vllm_caption dispatches to correct helper based on inflight flag.

    This is a happy path test that verifies that vllm_caption:
    1. runs without errors
    2. dispatches to the correct helper based on the inflight flag
    """
    mock_no_ifb.return_value = ["no_ifb"]
    mock_ifb.return_value = ["ifb"]
    out = vllm_caption(
        model_inputs=[{}],
        llm=MagicMock(),
        processor=MagicMock(),
        sampling_params=MagicMock(),
        vllm_config=VllmConfig(model_variant="any"),
        max_inflight_requests=0,
        inflight_batching=inflight,
    )
    assert out == (["ifb"] if inflight else ["no_ifb"])
