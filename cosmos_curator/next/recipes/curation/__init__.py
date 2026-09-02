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

"""Curate recipe: cluster, de-duplicate, and balance the rows of ``clips.lance``.

Wide table in, wide table out. One version-pinned scan reads the source labels
and the embedding vectors, one GPU pass applies the retention rule inside each
cluster, one driver-side water-fill sets per-group quotas, and one commit writes
two nullable columns - ``curate_selection_reason`` and ``curate_cluster_id`` -
back onto the rows they were computed from. No side table, no fused staging, no
report sidecar.

::

    columns    the persisted + in-flight column contract, and the
    |          eligibility predicate. Imports nothing but pyarrow.
    |
    config     CurateConfig: the operator-settable surface.
    |
    vectors    eligibility classification, the weighted working vector,
    dedup      the retention kernel, the fairness quota and cut. Pure, and
    fairness   importable with lance / ray / cuml / cupy absent.
    |
    pipeline   the Lance and Ray layer: schema widening, the fit task, the
               scan, the group stages, and the write-back commit.

    pipeline_kind  the CLI adapter, off to one side of the ordering above: it
                   is imported at startup, so every import it needs sits
                   inside the callback that needs it.

Nothing is re-exported here, not even ``CurateConfig``. A package ``__init__``
runs before any submodule import, so every name listed here becomes a dependency
of importing ANY module in the package - including ``pipeline_kind``, which the
CLI composition root imports at startup to render ``--help``. ``pipeline`` would
put Lance and Ray on that path and break the CPU-only import the pure kernels are
required to keep; ``config`` would put pydantic on it and defeat the startup
laziness the pipeline-kind adapters exist to preserve. This applies to the plain
value classes too, not just the entry points: ``CurateResult`` carries no Lance
or Ray content of its own, but importing it EXECUTES the module that defines it.

So every symbol is reached through the module that owns it -
``curation.config.CurateConfig``, ``curation.pipeline.run_curate`` - and this
file stays a docstring, like every other recipe package's ``__init__``.

See docs/curator/design/curator-next-curation.md.
"""
