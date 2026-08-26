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

"""Strict config and deterministic resolution for Curator Next ``video-split``."""

import copy
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.video_split.uris import join_s3_uri, normalize_s3_mp4_uri, normalize_s3_uri

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")

SchemaVersion = Literal[1]
VideoSplitKind = Literal["video-split"]
VideoEncoder = Literal["libopenh264"]
AudioMode = Literal["copy"]


class VideoSplitInputConfig(BaseModel):
    """Exactly one explicit object set or recursive S3 root."""

    model_config = _MODEL_CONFIG

    uris: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        description="Explicit S3 MP4 object URIs.",
        examples=[["s3://example-bucket/raw/a.mp4"]],
    )
    root_uri: str | None = Field(
        default=None,
        min_length=1,
        description="S3 prefix whose MP4 descendants are discovered recursively.",
        examples=["s3://example-bucket/raw/"],
    )

    @field_validator("uris", mode="before")
    @classmethod
    def _coerce_uris(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("uris")
    @classmethod
    def _normalize_uris(cls, uris: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if uris is None:
            return None
        return tuple(sorted({normalize_s3_mp4_uri(uri) for uri in uris}))

    @field_validator("root_uri")
    @classmethod
    def _normalize_root_uri(cls, root_uri: str | None) -> str | None:
        return None if root_uri is None else normalize_s3_uri(root_uri, strip_trailing_slash=True)

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        if (self.uris is None) == (self.root_uri is None):
            msg = "input must set exactly one of 'uris' or 'root_uri'"
            raise ValueError(msg)
        return self


class FixedStrideConfig(BaseModel):
    """Fixed-duration, fixed-stride span settings."""

    model_config = _MODEL_CONFIG

    duration_s: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)
    stride_s: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)
    min_duration_s: float = Field(
        default=2.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Minimum retained duration for a final short span.",
    )

    @model_validator(mode="after")
    def _validate_minimum(self) -> Self:
        if self.min_duration_s > self.duration_s:
            msg = "split.min_duration_s cannot be greater than split.duration_s"
            raise ValueError(msg)
        return self


class TranscodeConfig(BaseModel):
    """Settings that participate in the logical media contract.

    Every field here is hashed into clip identity, so this model holds only
    settings that decide what the output media *is*. Encoder thread count lives
    in execution config precisely because it does not.
    """

    model_config = _MODEL_CONFIG

    video_encoder: VideoEncoder = "libopenh264"
    video_bitrate: str = Field(default="4M", pattern=r"^[1-9][0-9]*(?:\.[0-9]+)?[KkMm]$")
    audio_mode: AudioMode = "copy"

    @field_validator("video_bitrate", mode="before")
    @classmethod
    def _normalize_bitrate(cls, value: object) -> object:
        if not isinstance(value, str) or not value[:-1]:
            return value
        try:
            number = Decimal(value[:-1])
        except ArithmeticError:
            return value
        return f"{format(number.normalize(), 'f')}{value[-1].upper()}"


class VideoSplitOutputConfig(BaseModel):
    """S3 media/error and local-or-S3 canonical Lance destinations."""

    model_config = _MODEL_CONFIG

    media_root: str = Field(min_length=1, examples=["s3://example-bucket/curated/video-split/"])
    clips_lance_uri: str = Field(default="", description="Canonical append-only clip table URI.")
    errors_uri: str = Field(default="", description="Run error report URI.")

    @model_validator(mode="before")
    @classmethod
    def _derive_dataset_uris(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        resolved = dict(value)
        media_root = resolved.get("media_root")
        if isinstance(media_root, str):
            resolved.setdefault("clips_lance_uri", join_s3_uri(media_root, "lance"))
            resolved.setdefault("errors_uri", join_s3_uri(media_root, "errors.json"))
        return resolved

    @field_validator("media_root")
    @classmethod
    def _normalize_media_root(cls, location: str) -> str:
        return normalize_s3_uri(location, strip_trailing_slash=True)

    @field_validator("clips_lance_uri")
    @classmethod
    def _normalize_lance_uri(cls, location: str) -> str:
        if not location or location != location.strip():
            msg = "Lance locations must be non-empty and cannot have surrounding whitespace"
            raise ValueError(msg)
        if "://" in location:
            return normalize_s3_uri(location, strip_trailing_slash=True)
        return str(Path(location).expanduser().resolve())

    @field_validator("errors_uri")
    @classmethod
    def _normalize_errors_uri(cls, location: str) -> str:
        normalized = normalize_s3_uri(location, strip_trailing_slash=True)
        if PurePosixPath(urlsplit(normalized).path).suffix.lower() != ".json":
            msg = f"Error report URIs must end in .json, got {location!r}"
            raise ValueError(msg)
        return normalized


class VideoSplitExecutionConfig(BaseModel):
    """Execution-only settings excluded from clip identity."""

    model_config = _MODEL_CONFIG

    storage_profile: str = Field(default="default", min_length=1)
    # Match the legacy CPU transcoder's empirically useful resource shape:
    # several single-threaded outputs share one FFmpeg process, while Ray
    # reserves enough cores to keep that bounded fan-out honest.
    transcode_cpus: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    # Threads per FFmpeg output encoder. Input decoders and simple filter
    # pipelines remain single-threaded because the batch already supplies
    # clip-level concurrency. Deliberately not part of TranscodeConfig: it
    # changes execution speed, not clip identity.
    encoder_threads: int = Field(default=1, ge=1)
    # Planned clips from one source are grouped into multi-output FFmpeg
    # invocations. This amortizes process startup and shared input access while
    # preserving one source download per transcode task.
    ffmpeg_batch_size: int = Field(default=16, ge=1)
    # Terminal records per publish task and therefore the upper bound for rows
    # in one fragment or one in-memory error batch. Each nonempty clip batch is
    # required to stage exactly one Lance fragment. A conservative default
    # bounds the all-errors case while producing few fragments at target scale.
    clips_per_publish_batch: int = Field(default=100_000, ge=1)
    storage_attempts: int = Field(default=3, ge=1)
    probe_attempts: int = Field(default=3, ge=1)
    transcode_attempts: int = Field(default=3, ge=1)
    probe_timeout_s: int = Field(default=120, ge=1)
    transcode_timeout_s: int = Field(default=600, ge=1)
    progress: bool = False


class ResolvedVideoSplitConfig(BaseModel):
    """Canonical v1 execution contract for Curator Next ``video-split``."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: VideoSplitKind
    input: VideoSplitInputConfig
    split: FixedStrideConfig = Field(default_factory=FixedStrideConfig)
    transcode: TranscodeConfig = Field(default_factory=TranscodeConfig)
    output: VideoSplitOutputConfig
    execution: VideoSplitExecutionConfig = Field(default_factory=VideoSplitExecutionConfig)


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "video-split",
    "input": {"uris": ["s3://example-bucket/raw/example.mp4"]},
    "output": {"media_root": "s3://example-bucket/curated/video-split/"},
}
_TEMPLATE = ResolvedVideoSplitConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json", exclude_none=True)
_TEMPLATE_PREAMBLE = """\
# All supported settings and their defaults are shown. Unchanged settings may be removed.
# To discover a prefix recursively, replace `uris` with `root_uri: s3://example-bucket/raw/`.
# The Lance and error-report URIs are optional; when omitted, they are derived from `media_root`.
"""


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config as a top-level mapping."""
    path = Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) if path.suffix.lower() in _YAML_SUFFIXES else json.load(config_file)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)
    return cast("dict[str, Any]", loaded)


def resolve_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> ResolvedVideoSplitConfig:
    """Load, override, and validate one config file."""
    return resolve_config_data(load_config_data(config_path), overrides=overrides)


def resolve_config_data(
    raw_data: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> ResolvedVideoSplitConfig:
    """Apply dotted YAML-valued overrides and return canonical typed config."""
    data = copy.deepcopy(dict(raw_data))
    apply_dotted_overrides(data, overrides)
    return ResolvedVideoSplitConfig.model_validate(data)


def resolved_config_to_json(config: ResolvedVideoSplitConfig, *, indent: int = 2) -> str:
    """Render canonical resolved JSON."""
    return json.dumps(config.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


def config_schema_json(*, indent: int = 2) -> str:
    """Return JSON Schema for v1 configs."""
    return json.dumps(ResolvedVideoSplitConfig.model_json_schema(), indent=indent, ensure_ascii=False) + "\n"


def config_template() -> dict[str, Any]:
    """Return a complete editable config template with every default."""
    return copy.deepcopy(_TEMPLATE)


def config_template_yaml() -> str:
    """Render the complete template as annotated YAML."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Return structured template metadata for agents."""
    return {
        "kind": "video-split",
        "description": "Split S3 MP4 sources into fixed-stride clips and publish a canonical Lance table.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "video-split"},
            {"path": "input.uris|input.root_uri", "example": ["s3://example-bucket/raw/example.mp4"]},
            {"path": "output.media_root", "example": "s3://example-bucket/curated/video-split/"},
        ],
        "config": config_template(),
    }
