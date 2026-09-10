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

"""Tests for the ``curate`` pipeline-kind adapter's promises to its callers.

``prepare_run`` resolves the config NOW and imports the runtime LATER. Both
halves are contracts rather than optimizations: resolving eagerly is what lets
the generic runtime tell a config fault from a run fault by where it arose, and
deferring the runtime import is what keeps lance, ray and cuml off the path of a
plain ``--help``.

The template and the payload are pinned here because they are the adapter's
promises to a human and to a script respectively, and neither has another
surface that would notice it drifting.
"""

import pathlib

import pytest
import yaml

from cosmos_curator.next.recipes.curation.pipeline import CurateResult, MergeStats
from cosmos_curator.next.recipes.curation.pipeline_kind import CURATE_KIND, _run_message, _run_payload

from .conftest import RunChild

_RUNTIME_MODULE = "cosmos_curator.next.recipes.curation.pipeline"


def _write_config(directory: pathlib.Path, **overrides: object) -> pathlib.Path:
    """Write a minimal valid Curate config, applying keyword overrides."""
    config_path = directory / "curate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "kind": "curate", "clips_lance_uri": "s3://bucket/clips.lance"} | overrides,
        ),
        encoding="utf-8",
    )
    return config_path


def test_a_bad_config_fails_at_prepare_time_rather_than_at_run_time(tmp_path: pathlib.Path) -> None:
    """The config is resolved before ``prepare_run`` returns, not inside the closure.

    This is what makes the two fault classes distinguishable by origin. Were the
    resolve to happen inside the returned closure, every config error would reach
    the generic runtime from the same place a mid-run failure does, and be
    reported as one.
    """
    config_path = _write_config(tmp_path, dedup_eps=0.0)

    with pytest.raises(ValueError, match="dedup_eps"):
        CURATE_KIND.prepare_run(config_path, set_overrides=[])


def test_preparing_a_run_does_not_import_the_curate_runtime(tmp_path: pathlib.Path, run_child: RunChild) -> None:
    """``prepare_run`` returns without importing the module that holds lance and ray.

    Probed in a subprocess because this test session has already imported the
    runtime module for the other curation tests, which would make an in-process
    assertion unfailable. ``run_child`` owns anchoring that child to this checkout.
    """
    config_path = _write_config(tmp_path)
    probe = (
        "import pathlib, sys;"
        "from cosmos_curator.next.recipes.curation.pipeline_kind import CURATE_KIND;"
        f"CURATE_KIND.prepare_run(pathlib.Path({str(config_path)!r}), set_overrides=[]);"
        f"print({_RUNTIME_MODULE!r} in sys.modules)"
    )

    result = run_child(probe)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_the_shipped_template_is_accepted_by_validate(tmp_path: pathlib.Path) -> None:
    """``template`` emits a config that ``validate`` accepts.

    The documented launch path is template, then edit, then validate, so a
    template that cannot survive its own next step breaks the first thing an
    operator does. The template is a hand-written string, which is what makes
    this reachable at all.
    """
    config_path = tmp_path / "curate.yaml"
    config_path.write_text(CURATE_KIND.template_yaml(), encoding="utf-8")

    assert CURATE_KIND.validate(config_path, []) == {"ok": True}


def _result(**overrides: object) -> CurateResult:
    """Build a plausible run summary, applying keyword overrides."""
    fields: dict[str, object] = {
        "clips_lance_uri": "s3://bucket/clips.lance",
        "read_version": 46,
        "committed_version": 47,
        "eligible_rows": 1000,
        "written_rows": 1000,
        "requested_k": 8,
        "effective_k": 8,
        "subtask_k": 4,
        "fit_rows": 1000,
        "fairness_groups": 12,
        "unfunded_groups": 0,
        "target": 500,
        "reason_counts": {"selected": 500},
        "merge_stats": MergeStats(labels_in=3, labels_out=2, clips_moved=7, seconds=0.5),
        "centroids_uri": "s3://bucket/clips.lance__curate_centroids/v47.npz",
    }
    return CurateResult(**(fields | overrides))  # type: ignore[arg-type]


def test_the_reported_payload_publishes_every_field_under_its_result_name() -> None:
    """The ``--json`` payload names each field exactly as ``CurateResult`` does.

    These keys are what operators script against and what the runbook documents,
    so a rename on either side is a silent breakage: a reader of the JSON gets
    ``null`` rather than an error. Deriving the expectation from the result's own
    attribute names cross-checks two independently written lists, since
    ``_run_payload`` spells all fifteen keys by hand.
    """
    payload = _run_payload(_result())

    assert set(payload) == {field.name for field in CurateResult.__attrs_attrs__}
    assert payload["merge_stats"] == {"labels_in": 3, "labels_out": 2, "clips_moved": 7, "seconds": 0.5}


def test_the_one_line_summary_reports_unfunded_groups_only_when_there_are_some() -> None:
    """The summary names unfunded groups when any exist, and stays quiet otherwise.

    A group funded at zero contributed nothing, and which of its siblings were
    funded instead was decided by the seeded residual order - which no column
    preserves, because fairness groups are never persisted. So the line is the
    only place that fact is ever stated, and it has to be absent when it would
    be untrue.
    """
    assert "UNFUNDED" not in _run_message(_result(unfunded_groups=0))
    assert "2 of 12 fairness group(s) UNFUNDED, which ones decided by fairness_residual_seed alone" in _run_message(
        _result(unfunded_groups=2)
    )
