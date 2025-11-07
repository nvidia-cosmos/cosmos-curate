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

"""Tests for the Gemini-backed API caption stage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cosmos_curate.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curate.pipelines.video.captioning import api_caption_stage
from cosmos_curate.pipelines.video.captioning.api_caption_stage import ApiCaptionStage
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window


class _DummyModels:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return SimpleNamespace(text="dummy caption")


class _DummyClient:
    def __init__(self, models: _DummyModels) -> None:
        self.models = models


class _EmptyResponseModels(_DummyModels):
    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])


def _make_task(mp4_bytes: bytes | None) -> SplitPipeTask:
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    clip.windows.append(Window(start_frame=0, end_frame=1, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(video=video)


def test_api_caption_stage_generates_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export captions into the window map when the API call succeeds."""
    monkeypatch.setattr(
        api_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    stage = ApiCaptionStage(
        model_variant="gemini",
        model_name="models/test",
        max_output_tokens=64,
    )
    dummy_models = _DummyModels()
    stage._client = _DummyClient(dummy_models)  # type: ignore[assignment]

    task = _make_task(b"\x00\x01")
    result = stage.process_data([task])

    window = result[0].video.clips[0].windows[0]
    assert window.caption["gemini"] == "dummy caption"
    assert dummy_models.last_kwargs is not None
    assert dummy_models.last_kwargs["model"] == "models/test"


def test_api_caption_stage_records_error_for_missing_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record an error when mp4 bytes are unavailable for a window."""
    monkeypatch.setattr(
        api_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    stage = ApiCaptionStage()
    stage._client = _DummyClient(_DummyModels())  # type: ignore[assignment]

    task = _make_task(None)
    stage.process_data([task])

    clip = task.video.clips[0]
    assert "gemini_caption_0" in clip.errors
    assert "gemini" not in clip.windows[0].caption


def test_stage_setup_loads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure stage_setup pulls the API key from the config file."""

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(api_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(api_caption_stage, "genai_types", object())
    monkeypatch.setattr(
        api_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="config-key")),
    )

    stage = ApiCaptionStage()
    stage.stage_setup()
    assert isinstance(stage._client, _FakeClient)
    assert stage._client.api_key == "config-key"


def test_api_caption_stage_logs_error_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture Gemini runtime errors when the response lacks text."""

    class _FakeClient:
        def __init__(self) -> None:
            self.models = _EmptyResponseModels()

    monkeypatch.setattr(api_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(
        api_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )

    stage = ApiCaptionStage(verbose=True)
    stage._client = _FakeClient()
    task = _make_task(b"\x00\x01")
    stage.process_data([task])
    clip = task.video.clips[0]
    assert "gemini_caption_0" in clip.errors


def test_stage_setup_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing config entry raises a runtime error."""

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(api_caption_stage, "genai", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(api_caption_stage, "genai_types", object())
    monkeypatch.setattr(api_caption_stage, "load_config", lambda: ConfigFileData())

    with pytest.raises(RuntimeError, match="Gemini API key missing"):
        ApiCaptionStage()


def test_extract_text_handles_block_reason() -> None:
    """Report Gemini block reasons in error messages."""
    response = SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
    with pytest.raises(RuntimeError, match="Gemini request blocked: SAFETY"):
        ApiCaptionStage._extract_text(response)


def test_extract_text_reports_finish_reason() -> None:
    """Surface finish reasons when Gemini returns empty content."""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[]),
                finish_reason="MAX_TOKENS",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="finish_reasons=\\['MAX_TOKENS'\\]"):
        ApiCaptionStage._extract_text(response)


def test_api_caption_stage_does_not_retry_invalid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid retry loops when the Gemini API rejects the key."""

    class _FakeClientError(Exception):
        pass

    class _FailingModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **_kwargs: object) -> SimpleNamespace:
            self.calls += 1
            msg = "API Key not found. Please pass a valid API key."
            raise _FakeClientError(msg)

    monkeypatch.setattr(
        api_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        api_caption_stage.genai,
        "errors",
        SimpleNamespace(ClientError=_FakeClientError),
        raising=False,
    )

    stage = ApiCaptionStage()
    failing_models = _FailingModels()
    stage._client = _DummyClient(failing_models)  # type: ignore[assignment]

    task = _make_task(b"\x00\x01")
    stage.process_data([task])

    clip = task.video.clips[0]
    assert failing_models.calls == 1
    assert "Gemini rejected the API key" in clip.errors["gemini_caption_0"]
