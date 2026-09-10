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

"""Tests for the per-event VLM caption stage.

After the protocol refactor, ``PerEventCaptionStage`` no longer owns
backend dispatch — it consumes a :class:`SingleInferenceCaptionStage` and
delegates inference. The tests below stub the inner stage (no real
Qwen / Gemini / OpenAI / vLLM engines are exercised) so the suite stays
CPU-only and runs in the default cosmos-curator environment.
"""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.captioning.per_event_caption_stage import (
    PerEventCaptionStage,
    _build_instances_block,
    _build_prompt,
    _extract_events_payload,
)
from cosmos_curator.pipelines.video.captioning.single_inference import SingleInferenceCaptionStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

# ---------------------------------------------------------------------------
# A pluggable fake ``SingleInferenceCaptionStage`` used as the ``inner=`` arg.
# ---------------------------------------------------------------------------


class _FakeInner(SingleInferenceCaptionStage):
    """Minimal ``SingleInferenceCaptionStage`` stand-in for use as ``inner=``.

    Subclasses the real ABC so the forwarding-property tests on
    ``PerEventCaptionStage`` (resources, conda_env_name, model,
    secondary_name, stage_setup_on_node, stage_setup, destroy) hit the
    same straight-through call paths the per-event stage takes in
    production.

    Records every ``caption_single`` call (prompt + bytes) so tests can
    assert the per-event stage built the prompt correctly. Returns a
    canned ``response`` by default; tests can override per call by
    setting ``responses`` to a list (popped left-to-right) or
    ``side_effect`` to an exception.
    """

    def __init__(
        self,
        *,
        response: str = '{"events": [{"event_id": "e0"}]}',
        responses: list[str] | None = None,
        side_effect: BaseException | None = None,
        resources: CuratorStageResource | None = None,
        conda_env_name: str | None = "fake-env",
        secondary: str = "fake-inner",
        last_finish_reasons: list[str] | None = None,
        last_usage_metadata: object | None = None,
    ) -> None:
        super().__init__()
        self.response = response
        self.responses = list(responses) if responses is not None else []
        self.side_effect = side_effect
        self._resources = resources or CuratorStageResource(cpus=2.0, gpus=0)
        self._conda_env_name = conda_env_name
        self._secondary = secondary
        self.calls: list[tuple[str, bytes]] = []
        self.setup_on_node_called = False
        self.setup_called = False
        self.destroy_called = False
        self.last_finish_reasons: list[str] = last_finish_reasons or []
        self.last_usage_metadata: object | None = last_usage_metadata

    @property
    def resources(self) -> CuratorStageResource:
        return self._resources

    @property
    def conda_env_name(self) -> str | None:
        return self._conda_env_name

    @property
    def model(self) -> ModelInterface | None:  # type: ignore[override]
        return None

    def secondary_name(self) -> str:
        return self._secondary

    def caption_single(self, prompt: str, video_bytes: bytes) -> str:
        self.calls.append((prompt, video_bytes))
        if self.side_effect is not None:
            raise self.side_effect
        if self.responses:
            return self.responses.pop(0)
        return self.response

    def stage_setup_on_node(self) -> None:
        super().stage_setup_on_node()
        self.setup_on_node_called = True

    def stage_setup(self) -> None:
        super().stage_setup()
        self.setup_called = True

    def destroy(self) -> None:
        self.destroy_called = True


# ---------------------------------------------------------------------------
# _extract_events_payload — JSON-shape robustness
# ---------------------------------------------------------------------------


def test_extract_events_payload_wrapped_dict() -> None:
    """Top-level ``{"events": [...]}`` is unwrapped."""
    raw = '{"events": [{"event_id": "e0"}, {"event_id": "e1"}]}'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}, {"event_id": "e1"}]


def test_extract_events_payload_bare_array() -> None:
    """A bare top-level array is accepted as-is."""
    raw = '[{"event_id": "e0"}]'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_dict_extra_keys_ignored() -> None:
    """Extra sibling keys next to ``events`` are tolerated."""
    raw = '{"meta": "blah", "events": [{"e": 1}], "trailing": [1, 2]}'
    assert _extract_events_payload(raw) == [{"e": 1}]


def test_extract_events_payload_dict_without_events_returns_empty() -> None:
    """A JSON object missing ``events`` returns an empty list."""
    assert _extract_events_payload('{"foo": [1, 2, 3]}') == []


def test_extract_events_payload_dict_events_not_list_returns_empty() -> None:
    """``events`` of the wrong type returns an empty list rather than crashing."""
    assert _extract_events_payload('{"events": "oops"}') == []


def test_extract_events_payload_strips_markdown_json_fence() -> None:
    """Markdown ```json fences are stripped before parsing."""
    raw = '```json\n{"events": [{"event_id": "e0"}]}\n```'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_strips_bare_fence() -> None:
    """Markdown ``` fences without a language tag are also stripped."""
    raw = '```\n[{"event_id": "e0"}]\n```'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_recovers_object_after_prose() -> None:
    """The greedy regex fallback recovers a JSON object embedded in prose."""
    raw = 'Here is the result:\n{"events": [{"event_id": "e0"}]}\n\nThanks!'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_returns_empty_on_garbage() -> None:
    """Non-JSON / empty input returns ``[]`` instead of raising."""
    assert _extract_events_payload("not json at all") == []
    assert _extract_events_payload("") == []


def test_extract_events_payload_tolerates_unescaped_newlines() -> None:
    r"""``strict=False`` lets us recover when a VLM forgets to escape ``\n`` in a caption."""
    raw = '{"events": [{"event_caption": "line1\nline2"}]}'
    out = _extract_events_payload(raw)
    assert len(out) == 1
    assert "line1" in out[0]["event_caption"]
    assert "line2" in out[0]["event_caption"]


# ---------------------------------------------------------------------------
# _build_instances_block / _build_prompt — pure helpers
# ---------------------------------------------------------------------------


def _make_clip(span: tuple[float, float] = (0.0, 5.0), instances: list[dict[str, Any]] | None = None) -> Clip:
    clip = Clip(uuid=uuid4(), source_video="src.mp4", span=span)
    clip.sam3_instances = instances
    return clip


def test_build_instances_block_filters_non_int_object_ids() -> None:
    """Entries with a non-int ``object_id`` are dropped from the prompt block."""
    clip = _make_clip(
        instances=[
            {"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0},
            {"object_id": "garbage", "prompt": "a bus", "start_time_s": 0.0, "end_time_s": 1.0},
            {"object_id": 7, "prompt": "a cat"},
        ],
    )
    payload = json.loads(_build_instances_block(clip))
    assert [e["object_id"] for e in payload["instances"]] == [1, 7]
    assert payload["instances"][0]["class"] == "a car"
    # ``class`` falls through to the prompt; missing time fields surface as None.
    assert payload["instances"][1]["class"] == "a cat"
    assert payload["instances"][1]["start_time_s"] is None


def test_build_instances_block_handles_missing_instances() -> None:
    """``sam3_instances=None`` produces an empty instances payload."""
    clip = _make_clip(instances=None)
    payload = json.loads(_build_instances_block(clip))
    assert payload == {"instances": []}


def test_build_prompt_appends_duration_and_instances() -> None:
    """The user template is followed by clip duration and the instances block."""
    clip = _make_clip(
        span=(0.0, 5.0),
        instances=[{"object_id": 24, "prompt": "a car", "start_time_s": 0.5, "end_time_s": 4.5}],
    )
    out = _build_prompt("USER PROMPT", clip)
    assert "USER PROMPT" in out
    assert "CLIP DURATION: 5.00 seconds" in out
    assert "[0.0, 5.00]" in out
    assert '"object_id": 24' in out


# ---------------------------------------------------------------------------
# PerEventCaptionStage._looks_truncated
# ---------------------------------------------------------------------------


def test_looks_truncated_completed_object() -> None:
    """A response that ends in ``}`` is treated as complete."""
    assert PerEventCaptionStage._looks_truncated('{"events": []}') is False


def test_looks_truncated_completed_array() -> None:
    """A response that ends in ``]`` is treated as complete."""
    assert PerEventCaptionStage._looks_truncated("[1, 2, 3]") is False


def test_looks_truncated_dangling_object() -> None:
    """A response that ends mid-object is flagged as truncated."""
    assert PerEventCaptionStage._looks_truncated('{"events": [{"x":') is True


def test_looks_truncated_empty_string() -> None:
    """An empty response is not flagged as truncated (it's ``empty_or_unparseable``)."""
    assert PerEventCaptionStage._looks_truncated("") is False


# ---------------------------------------------------------------------------
# PerEventCaptionStage construction / lifecycle forwarding
# ---------------------------------------------------------------------------


def test_per_event_construction_uses_default_prompt_when_unset() -> None:
    """``prompt_text=None`` falls through to the bundled traffic-surveillance default."""
    stage = PerEventCaptionStage(inner=_FakeInner())
    # The packaged default contains the literal string from the resource file.
    assert "events" in stage._prompt_template.lower()


def test_per_event_construction_uses_explicit_prompt() -> None:
    """A non-None ``prompt_text`` overrides the default."""
    stage = PerEventCaptionStage(inner=_FakeInner(), prompt_text="MY CUSTOM PROMPT")
    assert stage._prompt_template == "MY CUSTOM PROMPT"


def test_per_event_stage_setup_on_node_forwards_to_inner() -> None:
    """``stage_setup_on_node`` runs the inner stage's per-node hook.

    Load-bearing on multi-node Slurm where ``VllmCaptionStage`` copies
    weights to local SSD in this hook only.
    """
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner)
    stage.stage_setup_on_node()
    assert inner.setup_on_node_called is True


def test_per_event_stage_setup_forwards_to_inner() -> None:
    """``stage_setup`` initialises the inner stage."""
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner)
    stage.stage_setup()
    assert inner.setup_called is True


def test_per_event_destroy_forwards_to_inner() -> None:
    """``destroy`` shuts the inner stage down."""
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner)
    stage.destroy()
    assert inner.destroy_called is True


def test_per_event_resources_forward_from_inner() -> None:
    """``resources`` reflects the inner stage's footprint, not a hardcoded value."""
    custom = CuratorStageResource(cpus=4.0, gpus=1)
    inner = _FakeInner(resources=custom)
    stage = PerEventCaptionStage(inner=inner)
    assert stage.resources is custom


def test_per_event_conda_env_forwards_from_inner() -> None:
    """``conda_env_name`` propagates from the inner stage so weights download in the right env."""
    inner = _FakeInner(conda_env_name="default")
    stage = PerEventCaptionStage(inner=inner)
    assert stage.conda_env_name == "default"


def test_per_event_secondary_name_includes_inner() -> None:
    """``secondary_name`` is namespaced with ``per_event:`` and the inner secondary."""
    inner = _FakeInner(secondary="qwen3_vl_30b")
    stage = PerEventCaptionStage(inner=inner)
    assert stage.secondary_name() == "per_event:qwen3_vl_30b"


# ---------------------------------------------------------------------------
# _process_clip orchestration via the fake inner.
# ---------------------------------------------------------------------------


def _make_clip_with_annotated(
    instances: list[dict[str, Any]] | None,
    annotated_bytes: bytes | None,
) -> tuple[SplitPipeTask, Clip]:
    """Build a ``SplitPipeTask`` whose only clip optionally carries SAM3 outputs."""
    clip = Clip(uuid=uuid4(), source_video="src.mp4", span=(0.0, 5.0))
    if instances is not None:
        clip.sam3_instances = instances
    if annotated_bytes is not None:
        clip.sam3_annotated_video = bytes_to_numpy(annotated_bytes)  # type: ignore[assignment]
    video = Video(input_video=Path("src.mp4"), clips=[clip])
    task = SplitPipeTask(session_id="t", videos=[video])
    return task, clip


def test_process_clip_skips_when_no_instances() -> None:
    """Clips with ``sam3_instances=None`` are skipped silently (no errors recorded)."""
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner, verbose=True)
    _, clip = _make_clip_with_annotated(instances=None, annotated_bytes=b"\x00")
    stage._process_clip(clip)
    assert clip.sam3_events is None
    assert clip.errors == {}
    assert inner.calls == []


def test_process_clip_skips_on_empty_instances() -> None:
    """Clips with an empty instances list are skipped silently."""
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner, verbose=True)
    _, clip = _make_clip_with_annotated(instances=[], annotated_bytes=b"\x00")
    stage._process_clip(clip)
    assert clip.sam3_events is None
    assert clip.errors == {}
    assert inner.calls == []


def test_process_clip_records_missing_annotated_video() -> None:
    """Instances without an annotated video record ``missing_annotated_video`` and skip."""
    inner = _FakeInner()
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=None,
    )
    stage._process_clip(clip)
    assert clip.errors["per_event_caption"] == "missing_annotated_video"
    assert clip.sam3_events is None
    assert inner.calls == []


def test_process_clip_populates_events_on_success() -> None:
    """A well-formed VLM response populates ``clip.sam3_events`` and records no errors."""
    inner = _FakeInner(response='{"events": [{"event_id": "event_000000", "category": "collision"}]}')
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    stage._process_clip(clip)
    assert clip.sam3_events == [{"event_id": "event_000000", "category": "collision"}]
    assert clip.errors == {}
    # The annotated mp4 bytes were forwarded verbatim.
    assert len(inner.calls) == 1
    sent_prompt, sent_bytes = inner.calls[0]
    assert sent_bytes == b"fake-mp4-bytes"
    # Prompt includes the rendered SAM3 instances block (object_id 1).
    assert '"object_id": 1' in sent_prompt
    assert "CLIP DURATION" in sent_prompt


def test_process_clip_records_empty_response() -> None:
    """A complete-but-eventless response records ``empty_or_unparseable_response``."""
    # Valid JSON dict that ends in ``}`` (not truncated) but has no ``events`` key.
    inner = _FakeInner(response='{"foo": "bar"}')
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    stage._process_clip(clip)
    assert clip.sam3_events == []
    assert clip.errors["per_event_caption"] == "empty_or_unparseable_response"


def test_process_clip_records_truncated_response() -> None:
    """A response chopped mid-JSON records ``truncated_response``."""
    # No closing brace — trips ``_looks_truncated``.
    inner = _FakeInner(response='{"events": [{"x":')
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    stage._process_clip(clip)
    assert clip.sam3_events == []
    assert clip.errors["per_event_caption"] == "truncated_response"


def test_process_clip_records_api_error() -> None:
    """Exceptions from the inner stage bubble into ``clip.errors`` as ``api_error``."""
    inner = _FakeInner(side_effect=RuntimeError("kaboom"))
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    stage._process_clip(clip)
    assert "api_error" in clip.errors["per_event_caption"]
    assert "kaboom" in clip.errors["per_event_caption"]
    assert clip.sam3_events is None


def test_process_data_iterates_all_clips() -> None:
    """``process_data`` should call ``caption_single`` once per clip across all tasks."""
    inner = _FakeInner(responses=['{"events": [{"event_id": "e0"}]}', '{"events": [{"event_id": "e1"}]}'])
    stage = PerEventCaptionStage(inner=inner)
    task, _ = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    second_clip = Clip(uuid=uuid4(), source_video="src.mp4", span=(0.0, 5.0))
    second_clip.sam3_instances = [{"object_id": 2, "prompt": "a bus", "start_time_s": 0.0, "end_time_s": 1.0}]
    second_clip.sam3_annotated_video = bytes_to_numpy(b"second")  # type: ignore[assignment]
    task.videos[0].clips.append(second_clip)

    stage.process_data([task])
    assert len(inner.calls) == 2
    expected_events = [[{"event_id": "e0"}], [{"event_id": "e1"}]]
    assert [c.sam3_events for c in task.videos[0].clips] == expected_events


def test_per_event_stage_propagates_inner_diagnostics() -> None:
    """``last_finish_reasons`` / ``last_usage_metadata`` from the inner stage are surfaced.

    ``PerEventCaptionStage`` peeks at these duck-typed attrs on the
    inner when logging an empty-events warning, so the contract is part
    of the stage's behaviour even when the inner isn't a real
    ``GeminiCaptionStage``.
    """
    inner = _FakeInner(response='{"foo": "bar"}', last_finish_reasons=["MAX_TOKENS"], last_usage_metadata="usage-blob")
    stage = PerEventCaptionStage(inner=inner)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    # Should not raise — the diagnostic accessors are exercised inside
    # ``_log_empty_events`` via getattr.
    stage._process_clip(clip)
    assert clip.errors["per_event_caption"] == "empty_or_unparseable_response"
