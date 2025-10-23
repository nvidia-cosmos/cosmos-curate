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
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest, VllmConfig

if conda_utils.is_running_in_env("unified"):
    from cosmos_curate.models.vllm_interface import (
        _VLLM_PLUGINS,
        _caption_no_inflight_batching,
        auto_processor,
        make_model_inputs,
        process_vllm_output,
        sampling_params,
        vllm_caption,
        vllm_generate,
        vllm_model,
    )

    VALID_VARIANTS = list(_VLLM_PLUGINS.keys())
else:
    VALID_VARIANTS = []


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
def test_sampling_params_maps_config(model_variant: str) -> None:
    """sampling_params should mirror fields from VllmConfig."""
    cfg = VllmConfig(
        model_variant=model_variant, temperature=0.7, top_p=0.9, repetition_penalty=1.1, max_output_tokens=42
    )
    sp = sampling_params(cfg)
    assert sp.temperature == cfg.temperature
    assert sp.top_p == cfg.top_p
    assert sp.repetition_penalty == cfg.repetition_penalty
    assert sp.max_tokens == cfg.max_output_tokens


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_auto_processor_and_vllm_model_use_plugin(mock_get_plugin: MagicMock, model_variant: str) -> None:
    """auto_processor and vllm_model delegate to plugin."""
    mock_plugin = MagicMock()
    mock_plugin.processor.return_value = "processor"
    mock_plugin.model.return_value = "llm"
    mock_get_plugin.return_value = mock_plugin

    cfg = VllmConfig(model_variant=model_variant)
    assert auto_processor(cfg) == "processor"
    assert vllm_model(cfg) == "llm"


@pytest.mark.env("unified")
@pytest.mark.parametrize("num_videos", [0, 1, 3])
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_make_model_inputs_invokes_plugin(mock_get_plugin: MagicMock, num_videos: int) -> None:
    """make_model_inputs calls plugin for each video."""
    mock_plugin = MagicMock()
    mock_plugin.make_llm_input.side_effect = lambda _prompt, frames, _proc: {"ok": True, "frames": frames}
    mock_get_plugin.return_value = mock_plugin

    cfg = VllmConfig(model_variant="any")
    videos = [object() for _ in range(num_videos)]
    out = make_model_inputs(videos, cfg, processor=MagicMock(), prompt="p")
    assert len(out) == num_videos
    for i, frames in enumerate(videos):
        assert out[i]["frames"] is frames


@pytest.mark.env("unified")
def test_vllm_generate_batches_and_rewrites_ids() -> None:
    """vllm_generate should batch by size and rewrite request_ids to match."""

    class DummyOut:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def generate(self, batch: list[object], **kwargs: object) -> list[DummyOut]:  # noqa: ARG002
            self.calls.append(len(batch))
            return [DummyOut(str(i)) for i in range(len(batch))]

    llm = DummyLLM()
    reqs = [VllmCaptionRequest(request_id=f"id{i}", inputs={"i": i}) for i in range(5)]
    outs = vllm_generate(llm, sampling_params=MagicMock(), requests=reqs, batch_size=2)  # type: ignore[arg-type]
    assert [o.request_id for o in outs] == [r.request_id for r in reqs]
    assert llm.calls == [2, 2, 1]


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", VALID_VARIANTS)
def test_process_vllm_output_filters_and_decodes(model_variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """process_vllm_output decodes finished outputs and ignores unfinished."""

    class DummyRO:
        def __init__(self, request_id: str, *, finished: bool) -> None:
            self.request_id = request_id
            self.finished = finished

    # Swap the RequestOutput type used for isinstance checks
    import cosmos_curate.models.vllm_interface as vi  # noqa: PLC0415

    monkeypatch.setattr(vi, "RequestOutput", DummyRO, raising=False)

    # Patch plugin to provide decode
    mock_plugin = MagicMock()
    mock_plugin.decode.side_effect = lambda out: f"dec-{out.request_id}"
    monkeypatch.setattr(vi, "_get_vllm_plugin", lambda _variant: mock_plugin, raising=False)

    cfg = VllmConfig(model_variant=model_variant)
    inflight = {f"r{i}": VllmCaptionRequest(request_id=f"r{i}", inputs={}) for i in range(3)}
    engine_out = [DummyRO("r0", finished=True), DummyRO("r1", finished=False), DummyRO("r2", finished=True)]

    finished = process_vllm_output(engine_out, inflight, cfg)  # type: ignore[arg-type]
    assert [r.request_id for r in finished] == ["r0", "r2"]
    assert [r.caption for r in finished] == ["dec-r0", "dec-r2"]


@pytest.mark.env("unified")
@pytest.mark.parametrize("stage2", [False, True])
@patch("cosmos_curate.models.vllm_interface.vllm_generate")
@patch("cosmos_curate.models.vllm_interface.process_vllm_output")
@patch("cosmos_curate.models.vllm_interface._get_vllm_plugin")
def test_caption_no_inflight_batching_flow(
    mock_get_plugin: MagicMock, mock_process: MagicMock, mock_generate: MagicMock, *, stage2: bool
) -> None:
    """_caption_no_inflight_batching should return decoded captions; optionally stage2."""
    mock_plugin = MagicMock()
    mock_plugin.decode.side_effect = lambda r: r.caption
    mock_plugin.make_refined_llm_request.side_effect = lambda r, _proc, _prmpt: VllmCaptionRequest(
        request_id=r.request_id, inputs=r.inputs
    )
    mock_get_plugin.return_value = mock_plugin

    def _finish(reqs: list[VllmCaptionRequest]) -> list[VllmCaptionRequest]:
        for i, r in enumerate(reqs):
            r.caption = ("cap" if not stage2 else "cap1") + str(i)
            r.stage2_prompt = "ref" if stage2 else None
        return list(reqs)

    mock_process.side_effect = lambda engine_out, inflight, cfg: _finish(inflight.values())  # noqa: ARG005
    mock_generate.return_value = []

    cfg = VllmConfig(model_variant="qwen", stage2_caption=stage2, stage2_prompt_text="ref")
    outputs = _caption_no_inflight_batching(
        model_inputs=[{"a": 1}, {"b": 2}],
        llm=MagicMock(),
        processor=MagicMock(),
        sampling_params=MagicMock(),
        vllm_config=cfg,
        stage2_prompts=[None, None],
    )
    assert outputs == [("cap" if not stage2 else "cap1") + "0", ("cap" if not stage2 else "cap1") + "1"]


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
