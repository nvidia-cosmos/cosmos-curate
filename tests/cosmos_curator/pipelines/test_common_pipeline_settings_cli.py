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
"""Tests for CommonPipelineSettings CLI wiring."""

import argparse

import attrs
import pytest

from cosmos_curator.pipelines.common_pipeline_settings import (
    PROFILING_CLI_FIELDS,
    CommonPipelineSettings,
    add_settings_cli_arguments,
)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_settings_cli_arguments(parser, CommonPipelineSettings)
    return parser


def _action_for_dest(parser: argparse.ArgumentParser, dest: str) -> argparse.Action | None:
    for action in parser._actions:
        if action.dest == dest:
            return action
    return None


def _user_dests(parser: argparse.ArgumentParser) -> set[str]:
    skip = frozenset({"help"})
    out: set[str] = set()
    for action in parser._actions:
        d = action.dest
        if d is None or d is argparse.SUPPRESS or d in skip:
            continue
        out.add(d)
    return out


def _expected_cli_field_names() -> set[str]:
    return {f.name for f in attrs.fields(CommonPipelineSettings)}


def test_common_parser_registered_dests_match_settings_fields() -> None:
    """Each attrs field with cli metadata has a parser action with the same dest name."""
    parser = _common_parser()
    assert _user_dests(parser) == _expected_cli_field_names()


def test_common_parser_action_dest_is_field_name() -> None:
    """Namespace attribute names match attrs field names (dest is the field name)."""
    parser = _common_parser()
    for field in attrs.fields(CommonPipelineSettings):
        action = _action_for_dest(parser, field.name)
        assert action is not None, field.name
        assert action.dest == field.name


def test_common_parser_scalar_action_type_matches_field_annotation() -> None:
    """Scalar flags get argparse type str or int matching the field annotation."""
    parser = _common_parser()
    expectations: list[tuple[str, type]] = [
        ("input_s3_profile_name", str),
        ("output_s3_profile_name", str),
        ("execution_mode", str),
        ("limit", int),
        ("model_weights_path", str),
        ("profile_cpu_exclude", str),
        ("profile_memory_exclude", str),
        ("profile_gpu_exclude", str),
    ]
    for dest, expected_type in expectations:
        action = _action_for_dest(parser, dest)
        assert action is not None, dest
        assert action.type is expected_type, f"{dest}: expected action.type {expected_type}, got {action.type!r}"


def test_common_parser_boolean_flags_use_action_not_scalar_type() -> None:
    """Boolean flags use action= and do not set a scalar type converter."""
    parser = _common_parser()
    for dest in (
        "verbose",
        "perf_profile",
        "profile_tracing",
        "profile_cpu",
        "profile_memory",
        "profile_gpu",
    ):
        action = _action_for_dest(parser, dest)
        assert action is not None, dest
        assert action.type is None, f"{dest}: expected no type= converter, got {action.type!r}"


def test_add_settings_cli_arguments_only_fields_registers_subset() -> None:
    """only_fields registers a subset of flags (add_profiling_args uses this)."""
    parser = argparse.ArgumentParser()
    add_settings_cli_arguments(parser, CommonPipelineSettings, only_fields=PROFILING_CLI_FIELDS)
    assert _user_dests(parser) == PROFILING_CLI_FIELDS


def test_common_parse_coerces_limit_and_from_namespace_roundtrip() -> None:
    """Parse coerces types; from_namespace builds settings from the namespace."""
    parser = _common_parser()
    args = parser.parse_args(
        [
            "--limit",
            "42",
            "--execution-mode",
            "STREAMING",
        ],
    )
    assert args.limit == 42
    assert type(args.limit) is int
    assert args.execution_mode == "STREAMING"

    settings = CommonPipelineSettings.from_namespace(args)
    assert settings.limit == 42
    assert settings.execution_mode == "STREAMING"


def test_otlp_run_attributes_map_parses_json_object() -> None:
    """The map flag parses a JSON object into a dict[str, str]."""
    parser = _common_parser()
    args = parser.parse_args(
        [
            "--otlp-run-attributes-map",
            '{"customer":"nvidia","cluster":"cs-oci-ord:interactive_singlenode"}',
        ],
    )
    settings = CommonPipelineSettings.from_namespace(args)
    assert settings.otlp_run_attributes_map == {
        "customer": "nvidia",
        "cluster": "cs-oci-ord:interactive_singlenode",
    }


def test_execution_mode_has_argparse_choices() -> None:
    """execution_mode exposes AUTO, BATCH, and STREAMING as argparse choices."""
    parser = _common_parser()
    action = _action_for_dest(parser, "execution_mode")
    assert action is not None
    assert action.choices is not None
    assert set(action.choices) == {"AUTO", "BATCH", "STREAMING"}


def test_execution_mode_default_is_auto() -> None:
    """With no flag, the CLI default preserves auto-selection behavior."""
    parser = _common_parser()
    args = parser.parse_args([])
    assert args.execution_mode == "AUTO"


def test_profile_tracing_sampling_default_is_one() -> None:
    """No flag -> sample every trace.

    The decision is made once at the trace root and inherited by the whole run,
    so a fractional default drops entire runs rather than thinning spans within
    one -- leaving tracing enabled but usually producing nothing to look at.
    """
    parser = _common_parser()
    args = parser.parse_args([])
    assert args.profile_tracing_sampling == 1.0


def test_xenna_streaming_scheduler_flag_is_kebab_case() -> None:
    """The auto-derived flag follows the field-name -> kebab-case convention."""
    parser = _common_parser()
    action = _action_for_dest(parser, "xenna_streaming_scheduler")
    assert action is not None
    assert "--xenna-streaming-scheduler" in action.option_strings


def test_xenna_streaming_scheduler_has_argparse_choices() -> None:
    """The flag exposes both scheduler kinds as argparse choices."""
    parser = _common_parser()
    action = _action_for_dest(parser, "xenna_streaming_scheduler")
    assert action is not None
    assert action.choices is not None
    assert set(action.choices) == {"FRAGMENTATION_BASED", "SATURATION_AWARE"}


def test_xenna_streaming_scheduler_default_is_fragmentation_based() -> None:
    """No flag -> FRAGMENTATION_BASED so behaviour is unchanged for existing operators."""
    parser = _common_parser()
    args = parser.parse_args([])
    assert args.xenna_streaming_scheduler == "FRAGMENTATION_BASED"


def test_xenna_streaming_scheduler_saturation_aware_value_round_trips() -> None:
    """Operator-supplied SATURATION_AWARE round-trips through ``from_namespace``."""
    parser = _common_parser()
    args = parser.parse_args(["--xenna-streaming-scheduler", "SATURATION_AWARE"])
    assert args.xenna_streaming_scheduler == "SATURATION_AWARE"

    settings = CommonPipelineSettings.from_namespace(args)
    assert settings.xenna_streaming_scheduler == "SATURATION_AWARE"


def test_xenna_streaming_scheduler_rejects_unknown_value_at_parse_time() -> None:
    """Argparse choices reject unknown values before the validator is reached."""
    parser = _common_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--xenna-streaming-scheduler", "not-a-scheduler"])


def test_xenna_streaming_scheduler_validator_rejects_unknown_value() -> None:
    """The attrs ``validators.in_`` rejects unknown values when constructed directly."""
    with pytest.raises(ValueError, match="xenna_streaming_scheduler"):
        CommonPipelineSettings(
            input_s3_profile_name="default",
            output_s3_profile_name="default",
            execution_mode="AUTO",
            xenna_streaming_scheduler="not-a-scheduler",
            limit=0,
            verbose=False,
            model_weights_path="weights",
            perf_profile=True,
            profile_tracing=False,
            profile_tracing_sampling=0.01,
            profile_tracing_otlp_endpoint="",
            otlp_metrics_push=False,
            otlp_metrics_push_endpoint="",
            otlp_metrics_push_interval=30,
            otlp_metrics_push_drop_regex="",
            otlp_run_attributes=True,
            profile_cpu=False,
            profile_memory=False,
            profile_gpu=False,
            profile_cpu_exclude="_root",
            profile_memory_exclude="_root",
            profile_gpu_exclude="_root",
            otlp_run_attributes_map={},
        )
