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

"""Driver-side reconciliation of freshly discovered spans with canonical clip rows.

Unlike ``video_split``, span discovery here (``discovery.discover_spans``) is
cheap parquet-only metadata reading with no video decode. Reconciliation does
not need to reconstruct "what should exist" from stored Lance fields the way
``video_split.recovery`` reconstructs a fixed-stride plan from stored source
duration to avoid re-probing — the freshly discovered ``SpanWorkItem.clip_id``
values already are the expected set for this run. Reconciliation is therefore
a direct diff: scan the canonical table for the discovered ``clip_id``s, then
drop already-committed items from each batch before any chunk MP4 download.
"""

from collections import Counter
from dataclasses import dataclass

import lance

from cosmos_curator.next.recipes.robot_action_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
)
from cosmos_curator.next.recipes.robot_action_split.discovery import ChunkSpanBatch
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal

_RECOVERY_COLUMNS = ("clip_id", "record_schema_version", "media_contract_version")


@dataclass(frozen=True, slots=True)
class ReconciledBatches:
    """Chunk batches with already-committed spans removed."""

    batches: tuple[ChunkSpanBatch, ...]
    complete_batches: int
    partial_batches: int
    unknown_batches: int
    committed_clip_rows: int


def reconcile_batches(batches: list[ChunkSpanBatch], *, dataset: lance.LanceDataset) -> ReconciledBatches:
    """Drop already-committed spans, returning only batches with missing work."""
    _validate_committed_versions(dataset)
    all_clip_ids = tuple(item.clip_id for batch in batches for item in batch.items)
    duplicates = sorted(clip_id for clip_id, count in Counter(all_clip_ids).items() if count > 1)
    if duplicates:
        msg = f"Discovery emitted duplicate clip_id value(s): {', '.join(duplicates)}"
        raise ValueError(msg)
    committed_versions = _scan_committed(all_clip_ids, dataset=dataset)

    reconciled: list[ChunkSpanBatch] = []
    complete_batches = 0
    partial_batches = 0
    unknown_batches = 0
    for batch in batches:
        missing_items = [item for item in batch.items if item.clip_id not in committed_versions]
        if not missing_items:
            complete_batches += 1
            continue
        if len(missing_items) == len(batch.items):
            unknown_batches += 1
        else:
            partial_batches += 1
        reconciled.append(
            ChunkSpanBatch(
                chunk_mp4_uri=batch.chunk_mp4_uri,
                data_parquet_uri=batch.data_parquet_uri,
                items=missing_items,
            )
        )

    return ReconciledBatches(
        batches=tuple(reconciled),
        complete_batches=complete_batches,
        partial_batches=partial_batches,
        unknown_batches=unknown_batches,
        committed_clip_rows=len(committed_versions),
    )


def _validate_committed_versions(dataset: lance.LanceDataset) -> None:
    """Reject a table containing any row from an incompatible schema/contract version.

    Must run independently of, and before, any ``clip_id``-filtered scan:
    ``make_clip_id`` bakes ``MEDIA_CONTRACT_VERSION`` into the digest itself, so
    a row written under an older media contract has a ``clip_id`` that this
    run's freshly discovered set never contains. A ``clip_id IN (...)`` filter
    built from that freshly discovered set would silently exclude every such
    row from a version check performed after filtering — a table containing
    only stale-contract rows for this run's spans would then look like zero
    committed rows instead of a version conflict, and reconciliation would
    append a second generation of clips instead of raising. Checking both
    version columns against the whole table up front, independent of which
    clip_ids this run happens to be looking for, is what makes the check
    actually reachable.
    """
    mismatched_schema = dataset.count_rows(filter=f"record_schema_version != {CLIP_RECORD_SCHEMA_VERSION}")
    if mismatched_schema:
        msg = (
            f"Canonical table contains {mismatched_schema} row(s) with record_schema_version "
            f"other than {CLIP_RECORD_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    mismatched_contract = dataset.count_rows(filter=f"media_contract_version != {MEDIA_CONTRACT_VERSION}")
    if mismatched_contract:
        msg = (
            f"Canonical table contains {mismatched_contract} row(s) with media_contract_version "
            f"other than {MEDIA_CONTRACT_VERSION}"
        )
        raise ValueError(msg)


def _scan_committed(clip_ids: tuple[str, ...], *, dataset: lance.LanceDataset) -> dict[str, tuple[int, int]]:
    """Return ``clip_id -> (record_schema_version, media_contract_version)`` for committed rows.

    Callers must run ``_validate_committed_versions`` first: by the time this
    scan runs, the whole table is already known to carry only the current
    schema/contract versions, so every row this filtered scan returns is safe
    to trust without re-checking its version columns here.
    """
    if not clip_ids:
        return {}
    # Use Lance's SQL filter parser rather than a PyArrow Expression, matching
    # lance_fragment_recovery._candidate_presence: Lance's Substrait bridge
    # cannot currently lower a string Expression once the table also contains
    # nullable struct enrichment fields, even though only these columns are
    # projected here.
    literals = ",".join(sql_string_literal(clip_id) for clip_id in clip_ids)
    id_filter = f"clip_id IN ({literals})"
    scanner = dataset.scanner(columns=list(_RECOVERY_COLUMNS), filter=id_filter)
    committed: dict[str, tuple[int, int]] = {}
    for record_batch in scanner.to_batches():
        for row in record_batch.to_pylist():
            clip_id = str(row["clip_id"])
            if clip_id in committed:
                msg = f"Canonical table contains duplicate clip_id {clip_id}"
                raise ValueError(msg)
            committed[clip_id] = (int(row["record_schema_version"]), int(row["media_contract_version"]))
    return committed
