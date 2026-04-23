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

"""Tests for image OpenAI/Gemini endpoint caption stages."""

import base64
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cosmos_curate.core.sensors.data.image_data import ImageData, ImageMetadata
from cosmos_curate.core.utils.config.config import ConfigFileData, Gemini, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curate.pipelines.image.captioning import image_api_caption_stages
from cosmos_curate.pipelines.image.captioning.image_api_caption_stages import (
    ImageGeminiCaptionStage,
    ImageOpenAICaptionStage,
    ImageOpenAIPrepStage,
    _guess_media_type,
)
from cosmos_curate.pipelines.image.utils.data_model import Image, ImagePipeTask


def _make_task() -> ImagePipeTask:
    path = pathlib.Path("/fake/image.jpg")
    image = Image(
        input_image=path,
        relative_path="image.jpg",
        encoded_data=np.frombuffer(b"\xff\xd8\xff\xdbraw", dtype=np.uint8),
        image_data=ImageData.from_frames(
            np.full((1, 16, 20, 3), 127, dtype=np.uint8),
            ImageMetadata(height=16, width=20, image_format="jpg"),
        ),
    )
    return ImagePipeTask(session_id=str(path), image=image)


class _FakeChoice:
    def __init__(self, text: str | None, finish_reason: str = "stop") -> None:
        self.message = SimpleNamespace(content=text)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


def _fake_openai_module() -> SimpleNamespace:
    class _AuthError(Exception):
        pass

    class _NotFoundError(Exception):
        pass

    class _BadRequestError(Exception):
        pass

    class _APITimeoutError(Exception):
        pass

    return SimpleNamespace(
        OpenAI=MagicMock,
        AuthenticationError=_AuthError,
        NotFoundError=_NotFoundError,
        BadRequestError=_BadRequestError,
        APITimeoutError=_APITimeoutError,
    )


def test_openai_prep_stage_creates_png_payload() -> None:
    """Prep stage should cache a resized PNG payload for endpoint upload."""
    stage = ImageOpenAIPrepStage(caption_prep_min_pixels=16 * 20, caption_prep_max_pixels=16 * 20)
    task = _make_task()

    stage.process_data([task])

    payload = task.image.model_input["openai"]["payload_bytes"]
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert task.image.width == 28
    assert task.image.height == 28


def test_openai_stage_setup_creates_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """stage_setup should build the OpenAI client from config."""
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        class models:  # noqa: N801
            @staticmethod
            def list() -> SimpleNamespace:
                return SimpleNamespace(data=[SimpleNamespace(id="served-model")])

    fake_openai = _fake_openai_module()
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    monkeypatch.setattr(
        image_api_caption_stages,
        "maybe_load_config",
        lambda: ConfigFileData(openai=OpenAIConfig(caption=OpenAIEndpointConfig(api_key="test-key"))),
    )

    stage = ImageOpenAICaptionStage(model_name="auto")
    stage.stage_setup()

    assert captured["api_key"] == "test-key"
    assert stage._model_name == "served-model"


def test_openai_stage_uses_preprocessed_payload_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI caption stage should upload the cached PNG payload when available."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=1, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)

    task = _make_task()
    ImageOpenAIPrepStage(caption_prep_min_pixels=16 * 20, caption_prep_max_pixels=16 * 20).process_data([task])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("caption")])
    stage._client = mock_client

    stage.process_data([task])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    image_url = call_kwargs["messages"][0]["content"][0]["image_url"]["url"]
    payload = task.image.captions["openai"]
    assert image_url.startswith("data:image/png;base64,")
    assert payload == "caption"
    assert task.image.caption_status == "success"


def test_openai_stage_raw_mode_uses_original_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without prep payload, OpenAI captioning should upload the original image bytes."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=1, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    task = _make_task()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("caption")])
    stage._client = mock_client

    stage.process_data([task])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    image_url = call_kwargs["messages"][0]["content"][0]["image_url"]["url"]
    encoded = base64.b64encode(bytes(task.image.encoded_data.resolve())).decode("utf-8")
    assert image_url == f"data:image/jpeg;base64,{encoded}"


def test_guess_media_type_ignores_non_image_guess() -> None:
    """Non-image extensions should not override the payload fallback MIME type."""
    image = Image(
        input_image=pathlib.Path("/fake/not-really-json.bin"),
        relative_path="not-really-json.json",
        encoded_data=np.frombuffer(b"\xff\xd8\xff\xdbraw", dtype=np.uint8),
    )

    assert _guess_media_type(image, b"\xff\xd8\xff\xdbraw") == "image/jpeg"


def test_openai_stage_maps_length_to_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Length finish_reason with text should become truncated."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=1, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    task = _make_task()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse([_FakeChoice("partial", finish_reason="length")])
    stage._client = mock_client

    stage.process_data([task])

    assert task.image.caption_status == "truncated"
    assert task.image.caption == "partial"


def test_openai_stage_maps_content_filter_to_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """content_filter responses should mark the image blocked."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=1, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    task = _make_task()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _FakeResponse(
        [_FakeChoice("ignored", finish_reason="content_filter")]
    )
    stage._client = mock_client

    stage.process_data([task])

    assert task.image.caption_status == "blocked"
    assert "openai" not in task.image.captions


def test_openai_stage_does_not_retry_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authentication failures should not be retried."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=3, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    task = _make_task()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_openai.AuthenticationError("bad key")
    stage._client = mock_client

    stage.process_data([task])

    assert mock_client.chat.completions.create.call_count == 1
    assert task.image.caption_status == "error"
    assert task.image.errors["openai_caption"] == "bad key"


def test_openai_stage_retries_timeout_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient timeout failures should be retried and can recover."""
    stage = ImageOpenAICaptionStage(model_name="model", max_caption_retries=2, retry_delay_seconds=0)
    fake_openai = _fake_openai_module()
    monkeypatch.setattr(image_api_caption_stages, "openai", fake_openai, raising=False)
    task = _make_task()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        fake_openai.APITimeoutError("slow"),
        _FakeResponse([_FakeChoice("recovered")]),
    ]
    stage._client = mock_client

    stage.process_data([task])

    assert mock_client.chat.completions.create.call_count == 2
    assert task.image.caption == "recovered"
    assert task.image.caption_status == "success"


def test_gemini_stage_setup_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Gemini config should fail fast at construction."""
    monkeypatch.setattr(image_api_caption_stages, "load_config", lambda: ConfigFileData())

    with pytest.raises(RuntimeError, match="Gemini API key missing"):
        ImageGeminiCaptionStage()


def test_gemini_stage_generates_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful Gemini responses should populate the image caption fields."""
    monkeypatch.setattr(
        image_api_caption_stages,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        image_api_caption_stages,
        "genai_types",
        SimpleNamespace(
            Blob=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type),
            Part=lambda inline_data=None, text=None: SimpleNamespace(inline_data=inline_data, text=text),
            Content=lambda parts: SimpleNamespace(parts=parts),
            GenerateContentConfig=lambda max_output_tokens: SimpleNamespace(max_output_tokens=max_output_tokens),
        ),
        raising=False,
    )
    stage = ImageGeminiCaptionStage(max_caption_retries=1, retry_delay_seconds=0)
    stage._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text="caption"))
    )
    task = _make_task()

    stage.process_data([task])

    assert task.image.caption == "caption"
    assert task.image.caption_status == "success"


def test_gemini_stage_maps_max_tokens_to_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_TOKENS responses with text should become truncated."""
    monkeypatch.setattr(
        image_api_caption_stages,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        image_api_caption_stages,
        "genai_types",
        SimpleNamespace(
            Blob=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type),
            Part=lambda inline_data=None, text=None: SimpleNamespace(inline_data=inline_data, text=text),
            Content=lambda parts: SimpleNamespace(parts=parts),
            GenerateContentConfig=lambda max_output_tokens: SimpleNamespace(max_output_tokens=max_output_tokens),
        ),
        raising=False,
    )
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="partial")]), finish_reason="MAX_TOKENS"
            )
        ]
    )
    stage = ImageGeminiCaptionStage(max_caption_retries=1, retry_delay_seconds=0)
    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    task = _make_task()

    stage.process_data([task])

    assert task.image.caption_status == "truncated"
    assert task.image.caption == "partial"


def test_gemini_stage_maps_blocked_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt feedback block reasons should mark the image blocked."""
    monkeypatch.setattr(
        image_api_caption_stages,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        image_api_caption_stages,
        "genai_types",
        SimpleNamespace(
            Blob=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type),
            Part=lambda inline_data=None, text=None: SimpleNamespace(inline_data=inline_data, text=text),
            Content=lambda parts: SimpleNamespace(parts=parts),
            GenerateContentConfig=lambda max_output_tokens: SimpleNamespace(max_output_tokens=max_output_tokens),
        ),
        raising=False,
    )
    response = SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
    stage = ImageGeminiCaptionStage(max_caption_retries=1, retry_delay_seconds=0)
    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    task = _make_task()

    stage.process_data([task])

    assert task.image.caption_status == "blocked"
    assert "gemini" not in task.image.captions


def test_gemini_stage_records_empty_response_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty Gemini responses should become error results with detail."""
    monkeypatch.setattr(
        image_api_caption_stages,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="dummy-key")),
    )
    monkeypatch.setattr(
        image_api_caption_stages,
        "genai_types",
        SimpleNamespace(
            Blob=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type),
            Part=lambda inline_data=None, text=None: SimpleNamespace(inline_data=inline_data, text=text),
            Content=lambda parts: SimpleNamespace(parts=parts),
            GenerateContentConfig=lambda max_output_tokens: SimpleNamespace(max_output_tokens=max_output_tokens),
        ),
        raising=False,
    )
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
    stage = ImageGeminiCaptionStage(max_caption_retries=1, retry_delay_seconds=0)
    stage._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    task = _make_task()

    stage.process_data([task])

    assert task.image.caption_status == "error"
    assert task.image.errors["gemini_caption"].startswith("Gemini response did not contain caption text")
