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

"""Tests for the ``multimodal-split`` config resolution and pipeline-kind surface."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS
from cosmos_curator.next.recipes.multimodal_split.config import (
    ResolvedMultimodalSplitConfig,
    resolve_config,
)
from cosmos_curator.next.recipes.multimodal_split.pipeline_kind import MULTIMODAL_SPLIT_KIND


def _write_config(root: Path, prefix: str, **input_fields: object) -> Path:
    path = root / "config.yaml"
    payload = {
        "schema_version": 1,
        "kind": "multimodal-split",
        "input": {"input_path_prefix": prefix, **input_fields},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _make_sessions(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True)


def test_the_kind_is_registered_under_its_hyphenated_name() -> None:
    """The CLI resolves configs by their ``kind`` discriminator."""
    assert BUILTIN_PIPELINE_KINDS.get("multimodal-split") is MULTIMODAL_SPLIT_KIND
    assert "multimodal-split" in BUILTIN_PIPELINE_KINDS.names()


def test_the_packaged_template_is_a_valid_config(tmp_path: Path) -> None:
    """`pipeline template` output must be usable without hand-editing to make it parse."""
    path = tmp_path / "template.yaml"
    path.write_text(MULTIMODAL_SPLIT_KIND.template_yaml(), encoding="utf-8")

    resolved = resolve_config(path)

    assert resolved.kind == "multimodal-split"
    assert resolved.input.input_path_prefix == "s3://example-bucket/recordings"
    assert resolved.input.session_id_list_path is None
    assert resolved.input.limit is None


def test_the_template_payload_agrees_with_the_template_yaml() -> None:
    """The JSON and YAML template surfaces must not drift apart."""
    payload = MULTIMODAL_SPLIT_KIND.template_payload()

    assert payload["kind"] == "multimodal-split"
    assert payload["config"] == yaml.safe_load(MULTIMODAL_SPLIT_KIND.template_yaml())


def test_validate_accepts_a_good_config_and_reports_the_failure_for_a_bad_one(tmp_path: Path) -> None:
    """`pipeline validate` is the operator's pre-flight check."""
    good = _write_config(tmp_path, "s3://example-bucket/recordings")
    assert MULTIMODAL_SPLIT_KIND.validate(good, []) == {"ok": True}

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"schema_version": 1, "kind": "multimodal-split", "input": {"input_path_prefix": "gs://x/y"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Unsupported storage scheme"):
        MULTIMODAL_SPLIT_KIND.validate(bad, [])


def test_render_emits_the_canonicalized_config(tmp_path: Path) -> None:
    """Render shows the values that will actually execute, after canonicalization."""
    path = _write_config(tmp_path, "s3://example-bucket/recordings///")

    rendered = json.loads(MULTIMODAL_SPLIT_KIND.render(path, []))

    assert rendered["input"]["input_path_prefix"] == "s3://example-bucket/recordings"


def test_overrides_are_parsed_as_yaml_scalars_not_strings(tmp_path: Path) -> None:
    """The config is strict, so a --set value must arrive as an int, not "10"."""
    path = _write_config(tmp_path, "s3://example-bucket/recordings")

    resolved = resolve_config(path, overrides=["input.limit=10"])

    assert resolved.input.limit == 10


def test_an_override_may_target_a_nested_key_that_is_absent(tmp_path: Path) -> None:
    """Setting an unset optional field must not require it to be present already."""
    path = _write_config(tmp_path, "s3://example-bucket/recordings")

    resolved = resolve_config(path, overrides=["input.session_id_list_path=s3://example-bucket/s.txt"])

    assert resolved.input.session_id_list_path == "s3://example-bucket/s.txt"


@pytest.mark.parametrize("override", ["nokeypath", "=value", "input..limit=1"])
def test_malformed_overrides_are_rejected(tmp_path: Path, override: str) -> None:
    """A mistyped --set is a config error rather than a silently ignored flag."""
    path = _write_config(tmp_path, "s3://example-bucket/recordings")

    with pytest.raises(ValueError, match="Override"):
        resolve_config(path, overrides=[override])


def test_a_missing_config_file_is_reported_by_path(tmp_path: Path) -> None:
    """The operator gets the path they typed, not a parser error."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        resolve_config(tmp_path / "absent.yaml")


def test_a_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    """A YAML list or scalar at the top level fails before Pydantic sees it."""
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="mapping at the top level"):
        resolve_config(path)


def test_the_wrong_kind_is_rejected(tmp_path: Path) -> None:
    """The discriminator must match, so a misrouted config fails loudly."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "kind": "robot-action-split", "input": {"input_path_prefix": "/data"}},
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        resolve_config(path)


def test_the_schema_documents_the_input_section() -> None:
    """`pipeline schema` is how an operator discovers the fields."""
    schema = json.loads(MULTIMODAL_SPLIT_KIND.schema_json())

    assert schema["$defs"]["MultimodalSplitInputConfig"]["required"] == ["input_path_prefix"]
    assert set(schema["required"]) == {"schema_version", "kind", "input"}


def test_a_run_discovers_sessions_and_reports_the_count(tmp_path: Path) -> None:
    """The run path exercises real discovery rather than standing in for it."""
    sessions = tmp_path / "recordings"
    sessions.mkdir()
    _make_sessions(sessions, ["session-b", "session-a"])
    path = _write_config(tmp_path, str(sessions))

    output = MULTIMODAL_SPLIT_KIND.prepare_run(path, set_overrides=[])()

    assert output.json_payload == {"input_path_prefix": str(sessions), "candidate_sessions": 2}
    assert "discovered 2 candidate session(s)" in output.message
    assert "not implemented yet" in output.message


def test_a_run_applies_overrides_before_discovering(tmp_path: Path) -> None:
    """--set must reach discovery, not just config validation."""
    sessions = tmp_path / "recordings"
    sessions.mkdir()
    _make_sessions(sessions, ["session-a", "session-b", "session-c"])
    path = _write_config(tmp_path, str(sessions))

    output = MULTIMODAL_SPLIT_KIND.prepare_run(path, set_overrides=["input.limit=1"])()

    assert output.json_payload["candidate_sessions"] == 1


def test_preparing_a_run_does_not_execute_it(tmp_path: Path) -> None:
    """prepare_run resolves config eagerly but defers the work until called."""
    path = _write_config(tmp_path, str(tmp_path / "does-not-exist"))

    prepared = MULTIMODAL_SPLIT_KIND.prepare_run(path, set_overrides=[])

    with pytest.raises(FileNotFoundError, match="Input path prefix does not exist"):
        prepared()


def test_the_resolved_config_is_frozen(tmp_path: Path) -> None:
    """The executing contract cannot drift after resolution."""
    path = _write_config(tmp_path, "s3://example-bucket/recordings")
    resolved = resolve_config(path)

    assert isinstance(resolved, ResolvedMultimodalSplitConfig)
    with pytest.raises(ValidationError, match="frozen"):
        resolved.schema_version = 1
