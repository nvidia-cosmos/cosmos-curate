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

"""Config-backed pipeline-kind surface for the ``curate`` recipe.

One resolver serves every operation that reads a config file, so ``validate``,
``render``, and ``run-pipeline`` cannot disagree about what one means. ``schema``
takes no config and is published by the model itself::

    config.yaml --> resolve_config --> CurateConfig
                                            |
              +------------+----------------+
              v            v                v
           validate      render         prepare_run
                                             |
                                             v
                                         run_curate

    (schema) ------------------> CurateConfig.model_json_schema()

The two-phase ``prepare_run``, the catch-nothing rule and the envelope contract
are shared across every kind and stated once in ``next.core.pipeline_kind``. What
is specific here is the weight of the deferral: the Curate runtime reaches lance,
ray, cuml and cupy, so a module-level import would put the whole GPU stack on the
path of a plain ``--help``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun

if TYPE_CHECKING:
    from cosmos_curator.next.recipes.curation.pipeline import CurateResult


def _template_yaml() -> str:
    return """\
# Curate widens the SAME clips.lance it reads, adding two nullable columns
# (curate_selection_reason, curate_cluster_id) in one commit. There is no output
# URI and no staging tree: the table it reads is the table it writes.
# All other settings and their defaults: cosmos-curator pipeline schema curate
schema_version: 1
kind: curate

# Read source AND write target. Its embedding column groups are filled upstream
# by the embeddings pipeline; a weighted group that is missing fails preflight.
clips_lance_uri: s3://example-bucket/robot_clips/lance/clips.lance

# Credential profile for the table and for the centroids artifact beside it.
storage_profile: default

# How many de-duplication SURVIVORS to keep. Use target_fraction for a share of
# them instead, or leave both unset to keep every survivor.
target:
  target_count: 50000

# A row is a duplicate when it scores above 1 - dedup_eps against a row earlier
# in retention order, which runs farthest-from-centroid first rather than in table
# order. Set it to null to skip de-duplication entirely, which makes every
# eligible row WITH A USABLE EMBEDDING a selection candidate.
dedup_eps: 0.01

# `weights` is absent on purpose. It defines the distance the WHOLE corpus is
# clustered and de-duplicated on, and a zero weight also drops that block from
# the eligibility predicate, so two runs differing in it are not comparable and
# the eligible set itself moves. Change it in this file under review; `--set
# weights.*` reaches it too, but an override leaves no reviewable record of what
# a committed version was selecting for. They must sum to 1; `pipeline schema
# curate` lists the defaults.
"""


def _template_payload() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return {
        "kind": "curate",
        "description": "Cluster, de-duplicate and task-balance clips.lance, writing the verdict back onto its rows.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "curate"},
            {"path": "clips_lance_uri", "example": "s3://example-bucket/robot_clips/lance/clips.lance"},
        ],
        "config": yaml.safe_load(_template_yaml()),
    }


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.curation.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=list(overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.curation.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=list(overrides))
    return json.dumps(resolved.model_dump(mode="json"), indent=2) + "\n"


def _schema_json() -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.curation.config import CurateConfig  # noqa: PLC0415

    return json.dumps(CurateConfig.model_json_schema(), indent=2) + "\n"


def _list_presets() -> list[PipelinePreset]:
    return []


def _run_payload(result: "CurateResult") -> dict[str, object]:
    """Project one run's summary onto the machine-readable ``--json`` payload.

    Assembled here rather than in the recipe so the reported shape is owned by
    the CLI surface that promises it, and the recipe keeps returning its own
    verification summary unchanged. Every field is ``O(1)``, which is what keeps
    the payload a report rather than a second copy of the verdict column.
    """
    merge = result.merge_stats
    return {
        "clips_lance_uri": result.clips_lance_uri,
        "read_version": result.read_version,
        "committed_version": result.committed_version,
        "eligible_rows": result.eligible_rows,
        "written_rows": result.written_rows,
        "requested_k": result.requested_k,
        "effective_k": result.effective_k,
        "subtask_k": result.subtask_k,
        "fit_rows": result.fit_rows,
        "fairness_groups": result.fairness_groups,
        "unfunded_groups": result.unfunded_groups,
        "target": result.target,
        "reason_counts": result.reason_counts,
        "merge_stats": {
            "labels_in": merge.labels_in,
            "labels_out": merge.labels_out,
            "clips_moved": merge.clips_moved,
            "seconds": merge.seconds,
        },
        "centroids_uri": result.centroids_uri,
    }


def _run_message(result: "CurateResult") -> str:
    """Summarize one run in a single line: the selection, the versions, then any unfunded groups.

    Unfunded groups are part of the verdict rather than a detail. A group that
    received a quota of zero contributed nothing, and which of its siblings were
    funded instead was decided by ``fairness_residual_seed`` alone - a fact no
    column preserves, because fairness groups are never persisted.
    """
    from cosmos_curator.next.recipes.curation.columns import CurateReason  # noqa: PLC0415

    selected = result.reason_counts.get(CurateReason.SELECTED.value, 0)
    unfunded = (
        f"; {result.unfunded_groups} of {result.fairness_groups} fairness group(s) UNFUNDED, "
        "which ones decided by fairness_residual_seed alone"
        if result.unfunded_groups
        else ""
    )
    return (
        f"curate: {selected} of {result.eligible_rows} eligible row(s) selected at target {result.target}; "
        f"clips.lance v{result.read_version} -> v{result.committed_version}; "
        f"k={result.effective_k} over {result.fairness_groups} fairness group(s); "
        f"centroids -> {result.centroids_uri}{unfunded}"
    )


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.curation.config import resolve_config  # noqa: PLC0415

    # Resolved eagerly, so a bad config fails as a config fault before the
    # runtime starts Ray, and the closure never reparses the file.
    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.curation.pipeline import run_curate  # noqa: PLC0415

        result = run_curate(resolved)
        return PipelineRunOutput(json_payload=_run_payload(result), message=_run_message(result))

    return run


CURATE_KIND = PipelineKind(
    name="curate",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
