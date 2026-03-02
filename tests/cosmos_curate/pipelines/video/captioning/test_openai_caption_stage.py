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

"""Tests for the OpenAI-compatible API caption stage."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cosmos_curate.core.utils.config.config import ConfigFileData, OpenAIConfig
from cosmos_curate.pipelines.video.captioning import openai_caption_stage
from cosmos_curate.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(mp4_bytes: bytes | None, *, num_windows: int = 1) -> SplitPipeTask:
    """Create a minimal SplitPipeTask with one clip and the given windows."""
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    for i in range(num_windows):
        clip.windows.append(Window(start_frame=i * 10, end_frame=(i + 1) * 10, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _make_stage(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> OpenAICaptionStage:
    """Create a stage with the openai module patched so import-guarded code works."""
    # Ensure openai is importable in test (it may not live in "unified" env).
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    defaults: dict[str, object] = {"model_name": "test-model", "max_caption_retries": 1, "retry_delay_seconds": 0}
    defaults.update(kwargs)
    return OpenAICaptionStage(**defaults)  # type: ignore[arg-type]


class _FakeChoice:
    """Minimal stand-in for openai ChatCompletionChoice."""

    def __init__(self, text: str | None, finish_reason: str = "stop") -> None:
        self.message = SimpleNamespace(content=text)
        self.finish_reason = finish_reason


class _FakeResponse:
    """Minimal stand-in for openai ChatCompletion."""

    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


def _fake_openai_module() -> SimpleNamespace:
    """Return a fake openai module namespace with the error types the stage references."""

    class _AuthError(Exception):
        pass

    class _NotFoundError(Exception):
        pass

    class _BadRequestError(Exception):
        pass

    return SimpleNamespace(
        OpenAI=MagicMock,
        AuthenticationError=_AuthError,
        NotFoundError=_NotFoundError,
        BadRequestError=_BadRequestError,
    )


# ---------------------------------------------------------------------------
# stage_setup
# ---------------------------------------------------------------------------


def test_stage_setup_creates_client_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should create an OpenAI client using api_key from config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        openai_caption_stage,
        "maybe_load_config",
        lambda: ConfigFileData(openai=OpenAIConfig(api_key="test-key")),
    )

    stage = OpenAICaptionStage(model_name="m")
    stage.stage_setup()

    assert captured["api_key"] == "test-key"
    assert "base_url" not in captured


def test_stage_setup_passes_base_url_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should forward base_url when present in config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        openai_caption_stage,
        "maybe_load_config",
        lambda: ConfigFileData(openai=OpenAIConfig(api_key="k", base_url="http://localhost:8000/v1")),
    )

    stage = OpenAICaptionStage(model_name="m")
    stage.stage_setup()

    assert captured["base_url"] == "http://localhost:8000/v1"


def test_stage_setup_raises_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should raise RuntimeError when no config is loaded."""
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    monkeypatch.setattr(openai_caption_stage, "maybe_load_config", lambda: None)

    stage = OpenAICaptionStage(model_name="m")
    with pytest.raises(RuntimeError, match="OpenAI configuration not found"):
        stage.stage_setup()


def test_stage_setup_raises_when_openai_section_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should raise RuntimeError when openai section is absent."""
    monkeypatch.setattr(openai_caption_stage, "openai", _fake_openai_module(), raising=False)
    monkeypatch.setattr(openai_caption_stage, "maybe_load_config", lambda: ConfigFileData())

    stage = OpenAICaptionStage(model_name="m")
    with pytest.raises(RuntimeError, match="OpenAI configuration not found"):
        stage.stage_setup()


# ---------------------------------------------------------------------------
# _generate_caption
# ---------------------------------------------------------------------------


def test_generate_caption_encodes_mp4_as_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    """The video payload should be base64-encoded with the correct data URI prefix."""
    stage = _make_stage(monkeypatch)

    raw_bytes = b"\x00\x01\x02\x03"
    expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("caption")])
    stage._client = mock_client

    window = Window(start_frame=0, end_frame=1, mp4_bytes=raw_bytes)
    stage._generate_caption(window)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    content_parts = call_kwargs["messages"][0]["content"]
    video_url = content_parts[0]["video_url"]["url"]
    assert video_url == f"data:video/mp4;base64,{expected_b64}"


def test_generate_caption_returns_stripped_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption should be stripped of leading/trailing whitespace."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("  hello world  ")])
    stage._client = mock_client

    result = stage._generate_caption(Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))
    assert result == "hello world"


def test_generate_caption_raises_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError should be raised when the API returns no choices."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([])
    stage._client = mock_client

    with pytest.raises(RuntimeError, match="no choices"):
        stage._generate_caption(Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))


def test_generate_caption_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError should be raised when the caption text is empty."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("   ")])
    stage._client = mock_client

    with pytest.raises(RuntimeError, match="empty caption"):
        stage._generate_caption(Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))


def test_generate_caption_raises_when_client_not_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError should be raised when stage_setup was not called."""
    stage = _make_stage(monkeypatch)
    # _client stays None

    with pytest.raises(RuntimeError, match="not initialized"):
        stage._generate_caption(Window(start_frame=0, end_frame=1, mp4_bytes=b"\x00"))


def test_generate_caption_raises_when_mp4_bytes_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError should be raised when window has no mp4 bytes."""
    stage = _make_stage(monkeypatch)
    stage._client = MagicMock()

    with pytest.raises(RuntimeError, match="missing mp4 bytes"):
        stage._generate_caption(Window(start_frame=0, end_frame=1, mp4_bytes=None))


# ---------------------------------------------------------------------------
# _process_task / process_data
# ---------------------------------------------------------------------------


def test_process_task_stores_caption_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful caption should be stored in window.caption[model_variant]."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("A nice caption")])
    stage._client = mock_client

    task = _make_task(b"\x00\x01")
    stage.process_data([task])

    window = task.video.clips[0].windows[0]
    assert window.caption["openai"] == "A nice caption"


def test_process_task_records_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When captioning fails, the error should be stored in clip.errors."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")
    stage._client = mock_client

    task = _make_task(b"\x00\x01")
    stage.process_data([task])

    clip = task.video.clips[0]
    assert "openai_caption_0" in clip.errors
    assert "openai" not in clip.windows[0].caption


def test_process_task_continues_after_window_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in one window should not prevent subsequent windows from being captioned."""
    stage = _make_stage(monkeypatch)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("fail"),
        _FakeResponse([_FakeChoice("ok")]),
    ]
    stage._client = mock_client

    task = _make_task(b"\x00\x01", num_windows=2)
    stage.process_data([task])

    clip = task.video.clips[0]
    assert "openai_caption_0" in clip.errors
    assert clip.windows[1].caption["openai"] == "ok"


def test_process_data_returns_all_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """process_data should return every input task without dropping any."""
    stage = _make_stage(monkeypatch)
    stage._client = MagicMock()
    stage._client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("c")])

    tasks = [_make_task(b"\x00") for _ in range(3)]
    result = stage.process_data(tasks)

    assert len(result) == 3
    for orig, returned in zip(tasks, result, strict=False):
        assert orig is returned


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_generate_caption_does_not_retry_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """AuthenticationError should not be retried."""
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    stage = OpenAICaptionStage(model_name="m", max_caption_retries=3, retry_delay_seconds=0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_openai.AuthenticationError("bad key")
    stage._client = mock_client

    task = _make_task(b"\x00")
    stage.process_data([task])

    assert mock_client.chat.completions.create.call_count == 1
    assert "openai_caption_0" in task.video.clips[0].errors


def test_generate_caption_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """BadRequestError should not be retried."""
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(openai_caption_stage, "openai", fake_openai, raising=False)

    stage = OpenAICaptionStage(model_name="m", max_caption_retries=3, retry_delay_seconds=0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_openai.BadRequestError("invalid")
    stage._client = mock_client

    task = _make_task(b"\x00")
    stage.process_data([task])

    assert mock_client.chat.completions.create.call_count == 1


def test_generate_caption_retries_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient errors should be retried up to max_caption_retries times."""
    stage = _make_stage(monkeypatch, max_caption_retries=3, retry_delay_seconds=0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        ConnectionError("transient"),
        ConnectionError("transient"),
        _FakeResponse([_FakeChoice("recovered")]),
    ]
    stage._client = mock_client

    task = _make_task(b"\x00")
    stage.process_data([task])

    assert mock_client.chat.completions.create.call_count == 3
    assert task.video.clips[0].windows[0].caption["openai"] == "recovered"
