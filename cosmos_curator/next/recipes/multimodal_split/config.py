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

"""Typed config and resolution for Curator Next ``multimodal-split``."""

import json
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")
_SUPPORTED_SCHEMES = frozenset({"s3", "file"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_PARENT_SEGMENT = ".."
# ``urlsplit`` deletes these from a URI outright rather than rejecting it, which
# WHATWG requires and RFC 3986 does not. All three are legal in POSIX filenames
# and S3 keys.
_URI_STRIPPED_WHITESPACE = ("\t", "\n", "\r")

SchemaVersion = Literal[1]
MultimodalSplitKind = Literal["multimodal-split"]


class MultimodalSplitInputConfig(BaseModel):
    """Candidate session selection resolved on the driver before Ray starts.

    Two discovery modes are expressed by this config. With ``input_path_prefix``
    alone, the immediate children of the prefix are the candidate sessions. When
    ``session_id_list_path`` is also set, the session IDs come from that file and
    are joined to the prefix instead.
    """

    model_config = _MODEL_CONFIG

    input_path_prefix: str = Field(
        min_length=1,
        description="Local directory or s3:// prefix holding one child per recording session.",
        examples=["s3://example-bucket/recordings"],
    )
    session_id_list_path: str | None = Field(
        default=None,
        description="Optional newline-delimited UTF-8 file of session IDs to join to input_path_prefix.",
        examples=["s3://example-bucket/sessions.txt"],
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap applied after session ID normalization, deduplication, and sorting.",
    )

    @field_validator("input_path_prefix")
    @classmethod
    def _validate_input_path_prefix(cls, prefix: str) -> str:
        return _validate_location(prefix, strip_trailing_slash=True)

    @field_validator("session_id_list_path")
    @classmethod
    def _validate_session_id_list_path(cls, path: str | None) -> str | None:
        if path is None:
            return None
        return _validate_location(path, strip_trailing_slash=False)


class ResolvedMultimodalSplitConfig(BaseModel):
    """Canonical execution contract for Curator Next ``multimodal-split``.

    Only the ``input`` section exists so far, because discovery is the only stage
    implemented. The splitting stage adds clip geometry, sensor selection, and
    output sections alongside it rather than replacing this one.
    """

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: MultimodalSplitKind
    input: MultimodalSplitInputConfig


def load_config(config_path: str | Path) -> ResolvedMultimodalSplitConfig:
    """Load and validate a ``multimodal-split`` config file."""
    return resolve_config(config_path)


def resolve_config(
    config_path: str | Path,
    overrides: tuple[str, ...] | list[str] = (),
) -> ResolvedMultimodalSplitConfig:
    """Load a config file and apply ``--set`` overrides before validation.

    Each override has the form ``"path.to.key=value"``, where the value is parsed
    with ``yaml.safe_load`` so numeric, boolean, and null literals become native
    types instead of strings. That matters here because the config is strict:
    a quoted ``"5"`` would be rejected for ``input.limit``.

    Example::

        resolve_config("config.yaml", overrides=["input.limit=10"])
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle) if path.suffix.lower() in _YAML_SUFFIXES else json.load(handle)
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)

    raw: dict[str, object] = loaded
    for override in overrides:
        _apply_override(raw, override)
    return ResolvedMultimodalSplitConfig.model_validate(raw)


def _apply_override(raw: dict[str, object], override: str) -> None:
    """Set one dotted ``path.to.key=value`` override in place."""
    if "=" not in override:
        msg = f"Override must have the form 'path.to.key=value', got {override!r}"
        raise ValueError(msg)
    key_path, _, value_str = override.partition("=")
    if not key_path:
        msg = f"Override has an empty key path: {override!r}"
        raise ValueError(msg)
    keys = key_path.split(".")
    if any(not key for key in keys):
        msg = f"Override path {key_path!r} contains an empty segment"
        raise ValueError(msg)

    node: dict[str, object] = raw
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            msg = f"Override path {key_path!r} passes through a non-dict at {key!r}"
            raise TypeError(msg)
        node = child
    node[keys[-1]] = yaml.safe_load(value_str)


def _validate_location(location: str, *, strip_trailing_slash: bool) -> str:
    """Canonicalize a local path, ``file://`` URI, or ``s3://`` URI.

    ``strip_trailing_slash`` applies to prefixes, whose trailing slashes are not
    meaningful, but never to object names such as the session ID list file.
    """
    if not location or not location.strip():
        msg = "Storage locations cannot be empty"
        raise ValueError(msg)
    if location != location.strip():
        msg = "Storage locations cannot have leading or trailing whitespace"
        raise ValueError(msg)

    if "://" not in location:
        _reject_parent_segments(location, location)
        return _strip_trailing_slash(location) if strip_trailing_slash else location

    raw_scheme, remainder = location.split("://", maxsplit=1)
    scheme = raw_scheme.lower()
    if scheme not in _SUPPORTED_SCHEMES:
        msg = f"Unsupported storage scheme in {location!r}; expected local paths, file:// URIs, or s3:// URIs"
        raise ValueError(msg)

    normalized = f"{scheme}://{remainder}"
    if any(character in normalized for character in _URI_STRIPPED_WHITESPACE):
        # Unlike '?' and '#', these do not truncate: they vanish, leaving a
        # location that names a different directory and looks entirely ordinary.
        # ``file:///data/rec\tord`` canonicalizes to ``file:///data/record``.
        msg = (
            f"Storage location {location!r} contains a tab, carriage return, or newline, which URI parsing "
            f"deletes as whitespace. Percent-encode them as %09, %0D, and %0A, or pass the location as a "
            f"plain path instead of a URI"
        )
        raise ValueError(msg)

    parsed = urlsplit(normalized)
    if parsed.query or parsed.fragment:
        # ``?`` and ``#`` are URI delimiters, so ``urlsplit`` drops everything from
        # the first one onwards out of ``path``. Both are legal in POSIX directory
        # names and S3 keys, so an unencoded one does not fail: it silently names a
        # shorter location than the author wrote. ``file:///data/my#dir`` reads
        # ``/data/my``. Rejecting is the only option that cannot lose data, since a
        # bare local path and a percent-encoded URI both express the real name.
        msg = (
            f"Storage location {location!r} contains an unencoded '?' or '#'. "
            f"Percent-encode them as %3F and %23, or pass the location as a plain path instead of a URI"
        )
        raise ValueError(msg)
    if scheme == "file":
        if parsed.netloc not in {"", "localhost"} or not parsed.path:
            msg = f"Unsupported local file URI: {location}"
            raise ValueError(msg)
        _reject_parent_segments(location, unquote(parsed.path))
    else:
        _reject_parent_segments(location, parsed.path)

    canonical = _strip_trailing_slash(normalized) if strip_trailing_slash else normalized
    if scheme == "s3":
        _validate_s3_location(canonical)
    return canonical


def _validate_s3_location(location: str) -> str:
    """Run the storage layer's own bucket and key validation at config time.

    ``S3Prefix`` is what discovery builds every S3 call from, and it rejects
    invalid bucket names and keys. Constructing one here moves that failure from
    the middle of a run to ``cosmos-curator pipeline validate``, where it costs
    nothing.

    The import is deferred purely for cost: ``s3_client`` pulls in boto3, about
    90ms of the ~120ms it takes to import, and the CLI reaches this module on
    plenty of paths that never name an S3 location. Deferring is not what makes
    the import work from the client-only ``tools`` environment -- ``s3_client``
    is responsible for its own importability there, and this function runs on
    every ``s3://`` prefix regardless.
    """
    from cosmos_curator.core.utils.storage.s3_client import S3Prefix  # noqa: PLC0415

    S3Prefix(location)
    return location


def _reject_parent_segments(location: str, path: str) -> None:
    """Reject ``..`` segments, which no reader here can agree on how to resolve.

    Locally, ``..`` after a symlink means two different directories at once: the
    kernel resolves the link first, so ``os.scandir`` lists the link target's
    parent, while ``os.path.abspath`` cancels the segment lexically and names the
    link's own parent. Discovery needs both -- it lists with one and builds
    ``session_uri`` with the other -- so a prefix like ``/data/latest/..`` would
    emit URIs under a directory whose children were never listed.

    In S3 there is no divergence but no meaning either: keys are opaque, so
    ``..`` is a literal name component rather than a parent reference.

    Neither reading is worth guessing at, and an absolute path always says what
    was meant, so ``..`` is rejected before any listing happens.
    """
    if _PARENT_SEGMENT in path.split("/"):
        msg = (
            f"Storage location {location!r} contains a '..' segment. "
            f"Write the location it resolves to instead: '..' names one directory before a symlink "
            f"and another after it, and is a literal key component in S3"
        )
        raise ValueError(msg)


def _strip_trailing_slash(location: str) -> str:
    """Drop trailing slashes from the location itself, never from its scheme.

    Stripping the whole string would turn ``s3://`` into ``s3:`` and ``file:///``
    into ``file:``. Neither is recognized as a remote path afterwards, so both
    would silently degrade into a path relative to the driver's working
    directory instead of failing.
    """
    scheme, separator, _ = location.partition("://")
    if not separator:
        return location.rstrip("/") or "/"

    parsed = urlsplit(location)
    if scheme != "file" and not parsed.netloc:
        msg = f"Storage location {location!r} names a scheme but no location"
        raise ValueError(msg)

    # Strip the path only. Stripping the whole string drops the authority too, so
    # ``file://localhost/`` would become ``file://localhost``, which has no path
    # and is rejected later by the very code meant to read it.
    stripped_path = parsed.path.rstrip("/")
    if scheme == "file" and not stripped_path:
        stripped_path = "/"
    return parsed._replace(path=stripped_path).geturl()
