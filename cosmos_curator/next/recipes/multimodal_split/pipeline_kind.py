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

"""Config-backed pipeline-kind surface for the ``multimodal-split`` recipe.

Only candidate session discovery is implemented, so a run enumerates sessions
and reports how many were found. The splitting stage replaces the body of
``run`` with real episode processing; the config, kind registration, and CLI
surface are unchanged by that.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _template_yaml() -> str:
    return """\
schema_version: 1
kind: multimodal-split

input:
  # Local directory or s3:// prefix holding one child directory per session.
  input_path_prefix: s3://example-bucket/recordings
  # Optional newline-delimited UTF-8 file of session IDs to use instead of
  # listing the prefix. Each ID is joined to input_path_prefix.
  session_id_list_path: null
  # Optional cap applied after deduplication and sorting.
  limit: null
"""


def _template_payload() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return {
        "kind": "multimodal-split",
        "description": "Discover candidate AV recording sessions beneath a storage prefix.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "multimodal-split"},
            {"path": "input.input_path_prefix", "example": "s3://example-bucket/recordings"},
        ],
        "config": yaml.safe_load(_template_yaml()),
    }


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=list(overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=list(overrides))
    return json.dumps(resolved.model_dump(mode="json"), indent=2) + "\n"


def _schema_json() -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.multimodal_split.config import (  # noqa: PLC0415
        ResolvedMultimodalSplitConfig,
    )

    return json.dumps(ResolvedMultimodalSplitConfig.model_json_schema(), indent=2) + "\n"


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.multimodal_split.discovery import (  # noqa: PLC0415
            discover_candidate_sessions,
        )

        table = discover_candidate_sessions(resolved.input)
        return PipelineRunOutput(
            json_payload={
                "input_path_prefix": resolved.input.input_path_prefix,
                "candidate_sessions": table.num_rows,
            },
            message=(
                f"multimodal-split: discovered {table.num_rows} candidate session(s) under "
                f"{resolved.input.input_path_prefix}. Episode splitting is not implemented yet."
            ),
        )

    return run


MULTIMODAL_SPLIT_KIND = PipelineKind(
    name="multimodal-split",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
