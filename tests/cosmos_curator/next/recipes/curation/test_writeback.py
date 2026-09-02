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

"""Falsification tests for the two claims the wide-table architecture rests on.

Curate writes two columns into a table it shares with every other leg. Every
other design choice - no side table, no fused staging, no run id - is affordable
only if that write is non-destructive to the columns it does not own: same rows,
same row addresses, no tombstone, other columns untouched.

The second claim is about the columns it DOES own. The write is total: every row
of every fragment is written on every run, so a NULL means "the latest run did
not claim this row" and nothing else. That is what a re-run under narrower
weights depends on - without it a verdict computed against an earlier population
would survive indistinguishable from a current one.

These tests exercise both on real Lance rather than asserting them, so a pylance
version that changes the behaviour fails here instead of at the first 250M run.

::

    build a wide clips.lance      conftest: 3 fragments, 3 embedding groups
        |
    ensure_curate_columns         one metadata-only add_columns
        |
    update_one_fragment (x N)     the fragments carrying verdicts
    blank_one_fragment            the fragments no verdict group named
        |
    _collect -> _commit           one Update transaction
        |
    read back and compare against the pre-write table

The last test probes the ``cuml`` pixi environment and needs a real GPU, so it
carries ``env``. Every other test is CPU-only and runs by default.
"""

import json
import math
import pathlib
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence

import lance
import pyarrow as pa
import pytest
import yaml

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    KEY_COLUMN,
    TEXT_COLUMN_GROUP,
)
from cosmos_curator.next.recipes.curation.columns import (
    CURATE_CLUSTER_ID,
    CURATE_COLUMNS,
    CURATE_SELECTION_REASON,
    DEDUP_KEY_COLUMN,
    FRAGMENT_COLUMN,
    FUSED_BLOCKS,
    NO_DEDUP_GROUP,
    VERDICT_ROW,
    CurateReason,
    eligibility_filter,
)
from cosmos_curator.next.recipes.curation.config import CurateConfig, ModalityWeights
from cosmos_curator.next.recipes.curation.pipeline import (
    _FRAGMENT_ID_COLUMN,
    _GPU_ENV_NAME,
    CurateWriteError,
    _collect,
    _CollectedWrite,
    _commit,
    _commit_properties,
    _total_payloads,
    _total_rows,
    _uncovered_fragments,
    blank_one_fragment,
    ensure_curate_columns,
    update_one_fragment,
)

from .conftest import (
    POISONED_IMPORTS,
    ClipsTable,
    ClipsTableSpec,
    GroupState,
    RunChild,
    assert_poisoning_fired,
    poisoning,
)

BuildTable = Callable[[ClipsTableSpec], ClipsTable]

# Stand-in for the content hash of a published basis. These tests exercise the
# write-back and the commit, not the fit, so no basis is written here; the value's
# only job is to be the one the commit is expected to carry back out.
_STUB_FINGERPRINT = "0" * 64

# The session Ray cluster is pinned to num_gpus=0, so the probe needs a cluster
# of its own; a second ray.init() in this process would collide with it. The
# child reports through its exit status because a Ray driver writes freely to
# stderr and the JSON payload has to stay parseable.
_PROBE_NO_RAY_GPU = 3  # cluster started but exposes no device
_PROBE_TIMEOUT_S = 120  # cluster start + runtime_env resolution, generously


def _verdicts(fragment_id: int, rows: Sequence[tuple[str, int, str]]) -> pa.Table:
    """Build one fragment's verdict group from ``(clip_id, dedup_key, reason)`` triples."""
    return pa.table(
        {
            KEY_COLUMN: pa.array([clip_id for clip_id, _, _ in rows], type=pa.string()),
            FRAGMENT_COLUMN: pa.array([fragment_id] * len(rows), type=pa.int32()),
            DEDUP_KEY_COLUMN: pa.array([key for _, key, _ in rows], type=pa.int32()),
            CURATE_SELECTION_REASON: pa.array([reason for _, _, reason in rows], type=pa.string()),
        },
        schema=VERDICT_ROW,
    )


def _widened(table: ClipsTable) -> lance.LanceDataset:
    """Add the curate columns and return a handle pinned at the widened version."""
    ensure_curate_columns(lance.dataset(table.uri))
    return lance.dataset(table.uri)


def _properties(dataset: lance.LanceDataset, **overrides: object) -> dict[str, str]:
    """Build the commit's identity through the production builder.

    Routed through ``_commit_properties`` rather than written as a literal dict so
    a change to what a commit records reaches these tests instead of passing
    beside them.
    """
    config = CurateConfig(schema_version=1, kind="curate", clips_lance_uri=dataset.uri, **overrides)
    return _commit_properties(config, centroids_fingerprint=_STUB_FINGERPRINT)


def _write(dataset: lance.LanceDataset, groups: Sequence[pa.Table]) -> int:
    """Run the VERDICT pass only over the given groups; return the committed version.

    Deliberately not the whole write-back: this reaches only the fragments named by
    a group, which is what isolates per-fragment behaviour. Use ``_write_all`` to
    observe what a complete run publishes.
    """
    payloads = [
        str(update_one_fragment(group, uri=dataset.uri, read_version=int(dataset.version)).column(0)[0].as_py())
        for group in groups
    ]
    return _commit(dataset, _collect(payloads), properties=_properties(dataset))


def _write_all(dataset: lance.LanceDataset, groups: Sequence[pa.Table]) -> int:
    """Run both write-back passes over ``groups``, without Ray.

    The verdict groups first, then blanking for every fragment none of them named,
    so a test sees the table a whole run publishes rather than only the half the
    shuffle reaches. Fragment selection and the fragment-axis refusal are the
    production functions rather than a local restatement of them.

    The row-axis refusal ``_total_rows`` is deliberately absent: callers MAY pass
    partial verdicts, and those claim fewer rows than the table stores, so the
    identity would refuse a run the test means to publish. It is pinned directly
    instead. ``_write_back``'s own composition of the three needs Ray, so no test
    here covers it.
    """
    version = int(dataset.version)
    claimed = [
        str(update_one_fragment(group, uri=dataset.uri, read_version=version).column(0)[0].as_py()) for group in groups
    ]
    fragment_ids = [fragment.fragment_id for fragment in dataset.get_fragments()]
    uncovered = _uncovered_fragments(fragment_ids, claimed)
    blanked: list[str] = []
    if uncovered:
        batch = pa.table({_FRAGMENT_ID_COLUMN: pa.array(uncovered, type=pa.int64())})
        written = blank_one_fragment(batch, uri=dataset.uri, read_version=version)
        blanked = [str(written.column(0)[index].as_py()) for index in range(written.num_rows)]
    return _commit(dataset, _collect(_total_payloads(uncovered, claimed, blanked)), properties=_properties(dataset))


def _curate_values(uri: str) -> dict[str, tuple[str | None, int | None]]:
    """Return each ``clip_id``'s ``(reason, cluster_id)`` from the committed table.

    Keyed by ``clip_id``, so a caller asserting on a table built with a duplicate
    key must read per fragment instead.
    """
    rows = lance.dataset(uri).to_table(columns=[KEY_COLUMN, CURATE_SELECTION_REASON, CURATE_CLUSTER_ID]).to_pydict()
    return dict(
        zip(rows[KEY_COLUMN], zip(rows[CURATE_SELECTION_REASON], rows[CURATE_CLUSTER_ID], strict=True), strict=True)
    )


def _first_two_of_each_fragment(table: ClipsTable) -> list[pa.Table]:
    """Verdicts covering only the first two rows of every fragment, so most rows stay uncovered."""
    return [
        _verdicts(
            fragment_id,
            [(clip_ids[0], 5, CurateReason.SELECTED), (clip_ids[1], 5, CurateReason.DUPLICATE)],
        )
        for fragment_id, clip_ids in enumerate(table.clip_ids)
    ]


def test_write_back_preserves_the_row_count(build_clips_table: BuildTable) -> None:
    """A partial update writes no row, so the table's cardinality cannot move."""
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)
    before = dataset.count_rows()

    _write(dataset, _first_two_of_each_fragment(table))

    assert lance.dataset(table.uri).count_rows() == before


def test_write_back_preserves_fragment_identity(build_clips_table: BuildTable) -> None:
    """Fragment ids survive the update, which is what keeps row addresses stable."""
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)
    before = [fragment.fragment_id for fragment in dataset.get_fragments()]

    _write(dataset, _first_two_of_each_fragment(table))

    assert [fragment.fragment_id for fragment in lance.dataset(table.uri).get_fragments()] == before


def test_write_back_creates_no_deletion_vector(build_clips_table: BuildTable) -> None:
    """No fragment gains a tombstone.

    ``update_columns`` writes a new file for the updated fields only and rebinds
    the fragment to it; it does not replace rows. A non-zero count here would mean
    the write was a delete-and-append, which would invalidate every stable row
    address and make repeated runs grow the table.
    """
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)

    _write(dataset, _first_two_of_each_fragment(table))

    assert [fragment.num_deletions for fragment in lance.dataset(table.uri).get_fragments()] == [0, 0, 0]


def test_write_back_leaves_source_and_embedding_columns_identical(build_clips_table: BuildTable) -> None:
    """Every column Curate does not own reads back unchanged, from its original file.

    The value comparison proves the data is the same; the data-file comparison
    proves it was not rewritten to prove it, which is what makes the update cheap
    on a table whose embedding columns dominate its bytes.
    """
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)
    untouched = [name for name in dataset.schema.names if name not in CURATE_COLUMNS.names]
    before = dataset.to_table(columns=untouched)
    before_files = {file.path for fragment in dataset.get_fragments() for file in fragment.data_files()}

    _write(dataset, _first_two_of_each_fragment(table))

    after_dataset = lance.dataset(table.uri)
    assert after_dataset.to_table(columns=untouched).equals(before)
    after_files = {file.path for fragment in after_dataset.get_fragments() for file in fragment.data_files()}
    assert before_files <= after_files


def test_a_row_the_second_run_does_not_claim_is_blanked(build_clips_table: BuildTable) -> None:
    """A row inside a written fragment that the run does not claim comes back NULL.

    This can only be proved from non-NULL data. Starting from an all-NULL column,
    "the write blanked me" and "the left-outer join preserved my NULL" predict the
    same result, so the discriminating case needs a first run that gives the row a
    value the second run must clear. If it were preserved, a verdict computed
    against an earlier population would be indistinguishable from a current one.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    clip_ids = table.clip_ids[0]
    both = [(clip_ids[0], 1, CurateReason.SELECTED), (clip_ids[1], 2, CurateReason.DUPLICATE)]
    _write(_widened(table), [_verdicts(0, both)])

    _write(lance.dataset(table.uri), [_verdicts(0, [(clip_ids[0], 3, CurateReason.UNFUNDED)])])

    values = _curate_values(table.uri)
    assert values[clip_ids[0]] == (CurateReason.UNFUNDED, 3)
    assert values[clip_ids[1]] == (None, None)


def test_writing_one_fragment_leaves_another_fragments_verdicts_alone(build_clips_table: BuildTable) -> None:
    """Fragments are written independently: the verdict pass cannot reach across one.

    The within-run property the total write is built on top of, not in place of. A
    group names one fragment and ``update_columns`` is scoped to it, so a run's
    fragments cannot corrupt each other and may be written in any order. It is also
    exactly why the blanking pass has to exist: this isolation means a fragment no
    group names is not visited at all, so something else must clear it.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=1))
    first = [
        _verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)]),
        _verdicts(1, [(table.clip_ids[1][0], 2, CurateReason.DUPLICATE)]),
    ]
    _write(_widened(table), first)

    _write(lance.dataset(table.uri), [_verdicts(0, [(table.clip_ids[0][0], 9, CurateReason.BELOW_QUOTA)])])

    values = _curate_values(table.uri)
    assert values[table.clip_ids[0][0]] == (CurateReason.BELOW_QUOTA, 9)
    assert values[table.clip_ids[1][0]] == (CurateReason.DUPLICATE, 2)


def test_a_fragment_no_verdict_group_names_is_blanked(build_clips_table: BuildTable) -> None:
    """The blanking pass clears a whole fragment the verdict pass never reaches.

    The fragment-granularity half of totality, and the one a shuffle keyed on
    ``__frag`` can never cover: with no eligible row there is no group to key on.
    The table is appended as per-dataset slabs, so a modality missing for one whole
    dataset takes entire fragments out of the eligible set at once.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=1))
    first = [
        _verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)]),
        _verdicts(1, [(table.clip_ids[1][0], 2, CurateReason.DUPLICATE)]),
    ]
    _write(_widened(table), first)

    _write_all(lance.dataset(table.uri), [_verdicts(0, [(table.clip_ids[0][0], 9, CurateReason.BELOW_QUOTA)])])

    values = _curate_values(table.uri)
    assert values[table.clip_ids[0][0]] == (CurateReason.BELOW_QUOTA, 9)
    assert values[table.clip_ids[1][0]] == (None, None)


def test_a_narrowed_rerun_publishes_no_verdict_from_the_previous_run(build_clips_table: BuildTable) -> None:
    """The reviewed hazard end to end: nothing selected by run one survives run two.

    Run one claims every row. Run two claims a single row, as a re-run under weights
    that require a modality most rows lack would. Afterwards the table must hold
    exactly one verdict, so a consumer counting ``selected`` sees the target run two
    was given rather than the union of two runs' selections.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2))
    everything = [
        _verdicts(fragment_id, [(clip_id, 4, CurateReason.SELECTED) for clip_id in clip_ids])
        for fragment_id, clip_ids in enumerate(table.clip_ids)
    ]
    _write_all(_widened(table), everything)
    assert sum(reason is not None for reason, _ in _curate_values(table.uri).values()) == 4

    survivor = table.clip_ids[0][0]
    _write_all(lance.dataset(table.uri), [_verdicts(0, [(survivor, 6, CurateReason.SELECTED)])])

    values = _curate_values(table.uri)
    assert values[survivor] == (CurateReason.SELECTED, 6)
    assert [clip_id for clip_id, (reason, _) in values.items() if reason is not None] == [survivor]


def test_the_blanking_pass_targets_exactly_the_fragments_no_payload_named() -> None:
    """``_uncovered_fragments`` reads the covered set out of the payloads, not the verdicts.

    Which fragments get blanked decides whether the write is total, and the two-run
    tests above reach this only through a helper that supplies both arguments
    correctly. Called directly it must take the set complement over the table's own
    fragment ids, so a payload naming one absent from them removes nothing, and it
    must emit MANIFEST order rather than sorted order, because the blanking pass
    consumes the result as work items and compaction can leave the manifest
    unsorted.
    """
    assert _uncovered_fragments([0, 1, 2, 3], ['{"fragment_id": 2}', '{"fragment_id": 0}']) == [1, 3]
    assert _uncovered_fragments([5, 7], ['{"fragment_id": 5}', '{"fragment_id": 7}']) == []
    assert _uncovered_fragments([4, 6], []) == [4, 6]
    assert _uncovered_fragments([0, 1], ['{"fragment_id": 5}']) == [0, 1]
    assert _uncovered_fragments([7, 3, 5], []) == [7, 3, 5]


def test_a_blanking_pass_that_answers_for_fewer_fragments_is_refused() -> None:
    """``_total_payloads`` is the only thing that makes the write total rather than intended.

    The verdict pass alone commits a valid transaction over a subset of fragments,
    so nothing downstream can tell a complete write from one whose second pass was
    skipped or wired away - the commit succeeds either way and the unvisited
    fragments quietly keep a previous run's verdicts.
    """
    claimed = ['{"fragment_id": 0}']
    assert _total_payloads([], claimed, []) == claimed
    assert _total_payloads([3], claimed, ['{"fragment_id": 3}']) == [*claimed, '{"fragment_id": 3}']

    with pytest.raises(CurateWriteError, match="returned 0 payload"):
        _total_payloads([3], claimed, [])


def test_a_write_claiming_a_different_row_count_than_was_eligible_is_refused() -> None:
    """``_total_rows`` sees the loss the payload count structurally cannot.

    The uncovered set is derived from the payloads that came back, so a verdict
    payload that never arrives moves its fragment into the uncovered set and the
    blanking pass answers for it: the payload arithmetic balances while that
    fragment's rows go to NULL, which reads downstream as rows this run chose not
    to claim rather than as a failure. Asserted in both directions because the
    guard is an identity, not a floor - a surplus means a stage duplicated rows,
    and committing it would publish two verdicts for one clip. The fixture names
    no fragment because the guard reads only the row count.
    """
    collected = _CollectedWrite(fragments=(), field_ids=(), rows=4)
    assert _total_rows(collected, 4) is collected

    with pytest.raises(CurateWriteError, match="claimed 4 verdict row\\(s\\) for 5 eligible"):
        _total_rows(collected, 5)

    with pytest.raises(CurateWriteError, match="claimed 4 verdict row\\(s\\) for 3 eligible"):
        _total_rows(collected, 3)


def test_a_blanked_fragment_adds_no_rows_to_the_claimed_total(build_clips_table: BuildTable) -> None:
    """The blanking pass reports zero rows, which is what makes ``_total_rows`` add up.

    ``_total_rows`` compares the summed payloads against the ELIGIBLE count, so a
    blanking payload that reported the rows it NULLed would look like a surplus and
    abort every run over a table holding a fragment with no eligible row - not a
    rare shape, since the table is appended as per-dataset slabs. The two passes
    report different things on purpose: the verdict pass counts what it claimed,
    the blanking pass claims nothing.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
    dataset = _widened(table)
    version = int(dataset.version)
    claimed = [
        str(
            update_one_fragment(
                _verdicts(0, [(clip_id, 1, CurateReason.SELECTED) for clip_id in table.clip_ids[0]]),
                uri=dataset.uri,
                read_version=version,
            )
            .column(0)[0]
            .as_py()
        )
    ]
    uncovered = _uncovered_fragments([fragment.fragment_id for fragment in dataset.get_fragments()], claimed)
    written = blank_one_fragment(
        pa.table({_FRAGMENT_ID_COLUMN: pa.array(uncovered, type=pa.int64())}), uri=dataset.uri, read_version=version
    )
    blanked = [str(written.column(0)[index].as_py()) for index in range(written.num_rows)]

    collected = _collect(_total_payloads(uncovered, claimed, blanked))

    assert uncovered == [1], "fragment 1 carries no verdict group, so the blanking pass must reach it"
    assert collected.rows == 3, "the three claimed rows only; the blanked fragment's three rows are not claimed"
    assert _total_rows(collected, 3) is collected


def test_the_stamp_holds_exactly_four_properties(build_clips_table: BuildTable) -> None:
    """The property set is closed, so nothing joins or leaves a published commit unnoticed.

    Asserted as an exact set rather than by presence, because both directions are
    real. A property that silently DISAPPEARS leaves older versions carrying a key
    newer ones do not, so a reader cannot tell an absent value from an old table.
    A property that silently APPEARS is worse: the stamp is what a consumer keys
    attribution on, and an added key is published on every commit from then on.

    This is not hypothetical. A local edit adding an ``operator_note`` key to this
    dict survived the whole suite and reached a commit.

    All four are IDENTITY. ``centroids_fingerprint`` earns its place on the same
    terms as the digest: it names an object, not a row or a measurement, and it is
    the only value that can link a version to the basis its cluster ids were
    assigned against, because that object is named by its own content hash.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))

    assert set(_properties(_widened(table))) == {
        "kind",
        "schema_version",
        "config_digest",
        "centroids_fingerprint",
    }


def test_the_committed_version_records_the_leg_and_the_rules_that_produced_it(
    build_clips_table: BuildTable,
) -> None:
    """A committed version carries its own identity, readable from the table alone.

    Without this the two columns are anonymous: the same table is written by
    several legs and rewritten by every re-run, so two versions holding a
    selection cannot be told apart, and a reader comparing them has to trust
    bookkeeping that nothing verifies.

    The basis reference is part of that identity and is asserted by key, not only
    through the equality above: the basis object is named by its own content hash
    and so carries no version, which leaves the commit as the only thing that can
    say which basis a version's cluster ids were assigned against.
    """
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)

    version = _write(dataset, _first_two_of_each_fragment(table))

    stamped = lance.dataset(table.uri).read_transaction(version).transaction_properties
    assert stamped == _properties(dataset)
    assert stamped["kind"] == "curator-next-curation"
    assert stamped["config_digest"].startswith("sha256:")
    assert stamped["centroids_fingerprint"] == _STUB_FINGERPRINT


def test_the_input_version_is_recoverable_although_it_is_never_stamped(
    build_clips_table: BuildTable,
) -> None:
    """Lance persists the read version itself, which is why the stamp omits it.

    Pinned because the omission looks like a gap: a reader asking which table
    state a selection was computed against gets the answer from the transaction
    without us copying it, and a copy is the one form of this value that could
    disagree with the version actually read.
    """
    table = build_clips_table(ClipsTableSpec())
    dataset = _widened(table)
    read_version = int(dataset.version)

    version = _write(dataset, _first_two_of_each_fragment(table))

    transaction = lance.dataset(table.uri).read_transaction(version)
    assert transaction.read_version == read_version
    assert "read_version" not in transaction.transaction_properties


def test_a_first_run_leaves_uncovered_rows_null(build_clips_table: BuildTable) -> None:
    """On a freshly widened table only the covered rows gain a reason.

    Weaker than the blanking tests above - every row starts NULL here, so this
    cannot tell blank from preserve - but it pins the first-run state the
    post-commit completeness count is measured against.
    """
    table = build_clips_table(ClipsTableSpec(rows_per_fragment=4))
    dataset = _widened(table)
    covered = {clip_ids[0] for clip_ids in table.clip_ids} | {clip_ids[1] for clip_ids in table.clip_ids}

    _write(dataset, _first_two_of_each_fragment(table))

    values = _curate_values(table.uri)
    assert {clip_id for clip_id, (reason, _) in values.items() if reason is not None} == covered
    assert all(values[clip_id] == (None, None) for clip_id in values if clip_id not in covered)


def test_covered_rows_receive_their_own_verdict(build_clips_table: BuildTable) -> None:
    """Each covered row lands with the reason and cluster it was given, not a neighbour's."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=3))
    dataset = _widened(table)
    clip_ids = table.clip_ids[0]
    group = _verdicts(
        0,
        [(clip_ids[0], 7, CurateReason.SELECTED), (clip_ids[2], 11, CurateReason.BELOW_QUOTA)],
    )

    _write(dataset, [group])

    values = _curate_values(table.uri)
    assert values[clip_ids[0]] == (CurateReason.SELECTED, 7)
    assert values[clip_ids[2]] == (CurateReason.BELOW_QUOTA, 11)


def test_negative_dedup_key_lands_as_a_null_cluster_id(build_clips_table: BuildTable) -> None:
    """The in-flight sentinel never reaches storage as a cluster id.

    ``NO_DEDUP_GROUP`` routes a row past the similarity pass. Persisted as -1 it
    would be readable as a cluster and would corrupt every per-cluster
    aggregation; NULL says "scored against no cluster", which is the truth.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    dataset = _widened(table)
    clip_ids = table.clip_ids[0]
    group = _verdicts(
        0,
        [
            (clip_ids[0], NO_DEDUP_GROUP, CurateReason.INVALID_EMBEDDING),
            (clip_ids[1], 4, CurateReason.SELECTED),
        ],
    )

    _write(dataset, [group])

    values = _curate_values(table.uri)
    assert values[clip_ids[0]] == (CurateReason.INVALID_EMBEDDING, None)
    assert values[clip_ids[1]] == (CurateReason.SELECTED, 4)


def test_a_second_run_overwrites_the_previous_verdicts(build_clips_table: BuildTable) -> None:
    """Re-running replaces values in place rather than failing add-only or duplicating rows.

    Curate is re-runnable by design - it has no output directory to clear - so a
    second run over the same rows must be a plain overwrite.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    clip_ids = table.clip_ids[0]
    first = _widened(table)
    _write(first, [_verdicts(0, [(clip_ids[0], 1, CurateReason.SELECTED)])])

    second = lance.dataset(table.uri)
    _write(second, [_verdicts(0, [(clip_ids[0], 2, CurateReason.UNFUNDED)])])

    assert lance.dataset(table.uri).count_rows() == 2
    assert _curate_values(table.uri)[clip_ids[0]] == (CurateReason.UNFUNDED, 2)


def test_rerunning_identical_verdicts_is_idempotent(build_clips_table: BuildTable) -> None:
    """Two identical runs leave the same values, so a retried run is safe to repeat."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    clip_ids = table.clip_ids[0]
    group = _verdicts(0, [(clip_ids[0], 3, CurateReason.SELECTED)])
    _write(_widened(table), [group])
    after_first = _curate_values(table.uri)

    _write(lance.dataset(table.uri), [group])

    assert _curate_values(table.uri) == after_first


def test_duplicate_clip_id_within_a_fragment_aborts_before_committing(build_clips_table: BuildTable) -> None:
    """A repeated key inside one fragment raises and leaves the table untouched.

    Lance would resolve the duplicate by taking one matching row's value, and
    Curate legitimately gives identical-vector rows different verdicts - the second
    is a duplicate of the first - so the write would be plausible and wrong.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    dataset = _widened(table)
    version_before = int(dataset.version)
    shared = table.clip_ids[0][0]
    group = _verdicts(0, [(shared, 1, CurateReason.SELECTED), (shared, 1, CurateReason.DUPLICATE)])

    with pytest.raises(CurateWriteError, match="distinct clip_id"):
        _write(dataset, [group])

    after = lance.dataset(table.uri)
    assert int(after.version) == version_before
    assert all(reason is None for reason in after.to_table(columns=[CURATE_SELECTION_REASON])[0].to_pylist())


def test_duplicate_clip_id_in_the_persisted_fragment_aborts_before_write(build_clips_table: BuildTable) -> None:
    """A duplicate key already stored in the fragment is refused before ``update_columns``.

    The verdict group can be unique while the table still holds two rows under one
    ``clip_id``. Joining on that key would publish one verdict onto both rows and
    break the partial-subset write guarantee.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2, duplicate_within_fragment=0))
    dataset = _widened(table)
    version_before = int(dataset.version)
    shared = table.clip_ids[0][0]
    group = _verdicts(0, [(shared, 1, CurateReason.SELECTED)])

    with pytest.raises(CurateWriteError, match="update_columns match one verdict to multiple rows"):
        _write(dataset, [group])

    after = lance.dataset(table.uri)
    assert int(after.version) == version_before
    assert all(reason is None for reason in after.to_table(columns=[CURATE_SELECTION_REASON])[0].to_pylist())


def test_null_clip_id_in_the_persisted_fragment_aborts_before_write(
    build_clips_table: BuildTable,
    tmp_path: pathlib.Path,
) -> None:
    """A NULL key already stored in the fragment is refused before ``update_columns``.

    A unique verdict row cannot safely join onto a fragment that holds a NULL
    ``clip_id``; the write would be ambiguous about which rows it claims.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    patched_uri = str(tmp_path / "null-key.lance")
    original = lance.dataset(table.uri).to_table()
    clip_ids = original.column(KEY_COLUMN).to_pylist()
    clip_ids[1] = None
    patched = original.set_column(
        original.schema.get_field_index(KEY_COLUMN),
        KEY_COLUMN,
        pa.array(clip_ids, type=pa.string()),
    )
    lance.write_dataset(patched, patched_uri, mode="overwrite")
    patched_table = ClipsTable(uri=patched_uri, spec=table.spec, clip_ids=table.clip_ids)
    dataset = _widened(patched_table)
    version_before = int(dataset.version)
    group = _verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)])

    with pytest.raises(CurateWriteError, match="NULL clip_id"):
        _write(dataset, [group])

    after = lance.dataset(patched_uri)
    assert int(after.version) == version_before
    assert all(reason is None for reason in after.to_table(columns=[CURATE_SELECTION_REASON])[0].to_pylist())


def test_verdict_clip_id_absent_from_the_persisted_fragment_aborts_before_write(
    build_clips_table: BuildTable,
) -> None:
    """A verdict key missing from the fragment is refused before ``update_columns``.

    ``update_columns`` would leave the stored row unchanged when the right side has
    no match, so a wrong ``__frag`` can succeed while stale verdicts remain.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    dataset = _widened(table)
    version_before = int(dataset.version)
    group = _verdicts(0, [("clip-not-in-fragment", 1, CurateReason.SELECTED)])

    with pytest.raises(CurateWriteError, match="refusing a partial fragment update"):
        _write(dataset, [group])

    after = lance.dataset(table.uri)
    assert int(after.version) == version_before
    assert all(reason is None for reason in after.to_table(columns=[CURATE_SELECTION_REASON])[0].to_pylist())


def test_a_verdict_group_naming_two_fragments_is_refused(build_clips_table: BuildTable) -> None:
    """A group whose ``__frag`` is not constant aborts instead of dropping rows.

    ``groupby("__frag")`` already guarantees this, so the check exists for the
    failure it would otherwise hide: a row belonging to another fragment matches
    no ``clip_id`` in this one, so ``update_columns`` would drop its verdict with
    no error at all and the run would report it as written.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=1))
    dataset = _widened(table)
    mixed = pa.concat_tables(
        [
            _verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)]),
            _verdicts(1, [(table.clip_ids[1][0], 2, CurateReason.SELECTED)]),
        ]
    )

    with pytest.raises(CurateWriteError, match="exactly one fragment"):
        _write(dataset, [mixed])

    assert int(lance.dataset(table.uri).version) == int(dataset.version)


def test_a_verdict_group_naming_an_absent_fragment_is_refused(build_clips_table: BuildTable) -> None:
    """A fragment id the pinned version does not contain aborts rather than silently no-ops."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
    dataset = _widened(table)

    with pytest.raises(CurateWriteError, match="not present"):
        _write(dataset, [_verdicts(7, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)])])


def test_duplicate_clip_id_across_fragments_resolves_per_fragment(build_clips_table: BuildTable) -> None:
    """A key shared by two fragments is not an error: each write is fragment-scoped.

    Each ``update_columns`` call joins only against its own fragment's rows, so the
    two rows take their own verdicts. This is why the uniqueness check is
    intra-fragment and no corpus-wide scan is needed.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2, duplicate_across_fragments=(0, 1)))
    dataset = _widened(table)
    shared = table.clip_ids[0][0]
    groups = [
        _verdicts(0, [(shared, 1, CurateReason.SELECTED)]),
        _verdicts(1, [(shared, 2, CurateReason.DUPLICATE)]),
    ]

    _write(dataset, groups)

    per_fragment = [
        fragment.to_table(columns=[KEY_COLUMN, CURATE_SELECTION_REASON, CURATE_CLUSTER_ID]).to_pydict()
        for fragment in lance.dataset(table.uri).get_fragments()
    ]
    assert (per_fragment[0][CURATE_SELECTION_REASON][0], per_fragment[0][CURATE_CLUSTER_ID][0]) == (
        CurateReason.SELECTED,
        1,
    )
    assert (per_fragment[1][CURATE_SELECTION_REASON][0], per_fragment[1][CURATE_CLUSTER_ID][0]) == (
        CurateReason.DUPLICATE,
        2,
    )


def test_a_fragment_holding_no_eligible_row_is_never_written(build_clips_table: BuildTable) -> None:
    """Fragments absent from the update keep their metadata and read NULL.

    ``Update`` takes a partial fragment list, so a fragment whose rows all failed
    the eligibility predicate costs nothing and needs no empty write.
    """
    table = build_clips_table(ClipsTableSpec(fragments=3, empty_fragments=frozenset({1})))
    dataset = _widened(table)
    idle_files = {file.path for file in dataset.get_fragments()[1].data_files()}
    groups = [
        _verdicts(fragment_id, [(table.clip_ids[fragment_id][0], 1, CurateReason.SELECTED)]) for fragment_id in (0, 2)
    ]

    _write(dataset, groups)

    after = lance.dataset(table.uri)
    assert {file.path for file in after.get_fragments()[1].data_files()} == idle_files
    values = _curate_values(table.uri)
    assert all(values[clip_id] == (None, None) for clip_id in table.clip_ids[1])


def test_a_freshly_produced_table_carries_no_curate_columns(build_clips_table: BuildTable) -> None:
    """The producer's table has neither curate column, so Curate always widens it first.

    If the producer already emitted these names, ``ensure_curate_columns`` would
    be reading someone else's field and the ownership claim behind the
    single-writer design would be false.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))

    names = set(lance.dataset(table.uri).schema.names)

    assert names.isdisjoint(CURATE_COLUMNS.names)


def test_ensure_curate_columns_adds_both_columns_once(build_clips_table: BuildTable) -> None:
    """The first call adds two fields; a second call is a no-op on the same table."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))

    added = ensure_curate_columns(lance.dataset(table.uri))
    again = ensure_curate_columns(lance.dataset(table.uri))

    assert (added, again) == (2, 0)
    assert set(CURATE_COLUMNS.names) <= set(lance.dataset(table.uri).schema.names)


def test_ensure_curate_columns_advances_the_handle_it_was_given(build_clips_table: BuildTable) -> None:
    """The caller's handle moves to the widened version, which is what the write must pin.

    ``add_columns`` mutates the dataset object in place rather than returning a new
    one. The write-back reads ``dataset.version`` for the transaction's
    ``read_version``, so a caller that kept a pre-widening handle would pin a
    version whose manifest has no ``curate_*`` fields to rebind.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
    dataset = lance.dataset(table.uri)
    before = int(dataset.version)

    ensure_curate_columns(dataset)

    assert int(dataset.version) > before
    assert set(CURATE_COLUMNS.names) <= set(dataset.schema.names)


def test_ensure_curate_columns_adds_no_data_file(build_clips_table: BuildTable) -> None:
    """Widening is metadata-only: nullable fields read NULL without a file being written."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    before = {file.path for file in lance.dataset(table.uri).get_fragments()[0].data_files()}

    ensure_curate_columns(lance.dataset(table.uri))

    assert {file.path for file in lance.dataset(table.uri).get_fragments()[0].data_files()} == before


def test_ensure_curate_columns_rejects_a_partially_present_group(build_clips_table: BuildTable) -> None:
    """One column present without its sibling is a corrupt schema, not a state to repair."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
    lance.dataset(table.uri).add_columns(pa.schema([pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True)]))

    with pytest.raises(ValueError, match="partially present"):
        ensure_curate_columns(lance.dataset(table.uri))


def test_collect_refuses_two_metadata_versions_of_one_fragment(build_clips_table: BuildTable) -> None:
    """Two payloads naming one fragment abort, because the commit's outcome would depend on order."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
    dataset = _widened(table)
    group = _verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)])
    payload = str(update_one_fragment(group, uri=dataset.uri, read_version=int(dataset.version)).column(0)[0].as_py())

    with pytest.raises(CurateWriteError, match="two metadata versions"):
        _collect([payload, payload])


def test_update_one_fragment_returns_no_row_payload(build_clips_table: BuildTable) -> None:
    """The worker-to-driver channel is one string, so driver state is O(fragments)."""
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=4))
    dataset = _widened(table)
    group = _verdicts(0, [(clip_id, 1, CurateReason.SELECTED) for clip_id in table.clip_ids[0]])

    result = update_one_fragment(group, uri=dataset.uri, read_version=int(dataset.version))

    assert result.num_rows == 1
    assert result.num_columns == 1


def test_write_back_works_when_a_modality_group_is_absent(build_clips_table: BuildTable) -> None:
    """A table the action-embedding leg never ran for still accepts the curate columns.

    The write is bound to Curate's own field ids, so it cannot depend on which
    embedding groups a table happens to carry.
    """
    table = build_clips_table(
        ClipsTableSpec(fragments=1, rows_per_fragment=2, action=GroupState.ABSENT, image=GroupState.NULL)
    )
    dataset = _widened(table)

    _write(dataset, [_verdicts(0, [(table.clip_ids[0][0], 1, CurateReason.SELECTED)])])

    assert _curate_values(table.uri)[table.clip_ids[0][0]] == (CurateReason.SELECTED, 1)


def test_the_fixture_gives_each_modality_of_a_row_a_different_direction(build_clips_table: BuildTable) -> None:
    """No two modalities of one row share a vector, or fusion tests could not fail.

    If every block of a row carried the same direction, swapping two blocks in
    ``FUSED_BLOCKS`` or moving weight between them would be a no-op, so a test
    asserting that the block order or the weights change the fused result would
    pass whatever the implementation did. The fixture must not be able to hide
    that, so the per-column seed is pinned here rather than left to inspection.
    """
    table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
    columns = [
        TEXT_COLUMN_GROUP.primary_vector,
        "embedding_text_task",
        IMAGE_COLUMN_GROUP.primary_vector,
        ACTION_COLUMN_GROUP.primary_vector,
    ]
    rows = lance.dataset(table.uri).to_table(columns=columns).to_pydict()

    for index in range(2):
        drawn = [rows[column][index] for column in columns]
        assert len({tuple(vector) for vector in drawn}) == len(drawn), "two embedding blocks of one row are identical"


def test_the_fixture_rejects_a_knob_it_could_not_honour() -> None:
    """An index past the end of the table is refused, not silently dropped.

    A fixture that accepts ``empty_fragments={5}`` on a three-fragment table hands
    back a table with no empty fragment and no complaint, so the test asking for
    one passes having verified nothing. Loud beats convenient here.
    """
    with pytest.raises(ValueError, match="outside range"):
        ClipsTableSpec(fragments=3, empty_fragments=frozenset({5}))
    with pytest.raises(ValueError, match="outside range"):
        ClipsTableSpec(fragments=2, rows_per_fragment=2, zero_norm_vector_rows=frozenset({4}))
    with pytest.raises(ValueError, match="one index twice"):
        ClipsTableSpec(duplicate_vectors=(1, 1))
    with pytest.raises(ValueError, match="rows_per_fragment must be at least"):
        ClipsTableSpec(rows_per_fragment=1, duplicate_within_fragment=0)


def test_the_fixture_rejects_a_duplicate_pair_it_would_leave_null() -> None:
    """A duplicate pair inside an empty fragment, or with no filled group, is refused.

    Both states satisfy the knob's letter and not its purpose: the two rows exist
    but their vectors read NULL, so there is no pair for a de-duplication test to
    find. Refusing them keeps ``duplicate_vectors`` meaning one thing.
    """
    with pytest.raises(ValueError, match="empty fragment"):
        ClipsTableSpec(fragments=2, rows_per_fragment=2, empty_fragments=frozenset({1}), duplicate_vectors=(0, 3))
    with pytest.raises(ValueError, match="needs a filled group"):
        ClipsTableSpec(text=GroupState.NULL, image=GroupState.ABSENT, action=GroupState.NULL, duplicate_vectors=(0, 1))


def test_the_fixture_rejects_an_unusable_vector_it_would_leave_null() -> None:
    """The two pathology knobs are refused wherever no vector is written at all.

    Same rule as the duplicate pair, and it has to be stated for these knobs too:
    a NULL vector and a zero-norm one are both unusable, so the row still reaches
    ``invalid_embedding`` and the test passes without ever exercising the cause it
    named. Nothing downstream can tell the two apart, which is why the refusal has
    to happen here.
    """
    with pytest.raises(ValueError, match=r"zero_norm_vector_rows.*empty fragment"):
        ClipsTableSpec(
            fragments=2, rows_per_fragment=2, empty_fragments=frozenset({1}), zero_norm_vector_rows=frozenset({2})
        )
    with pytest.raises(ValueError, match=r"non_finite_vector_rows.*needs a filled group"):
        ClipsTableSpec(
            text=GroupState.NULL,
            image=GroupState.ABSENT,
            action=GroupState.NULL,
            non_finite_vector_rows=frozenset({0}),
        )


def test_the_fixture_rejects_a_duplicate_pair_whose_other_half_is_unusable() -> None:
    """A row cannot be both a duplicate and broken, so asking for both is refused.

    The two knobs disagree about one row rather than about the table: the pathology
    wins in ``_row_vector``, so the pair silently loses its second half while the
    spec still reads as though it placed one. Reached only on a table that CAN hold
    both properties, since a spec that would write NULL for the row is refused
    before this.
    """
    with pytest.raises(ValueError, match="also pathological"):
        ClipsTableSpec(fragments=1, rows_per_fragment=4, zero_norm_vector_rows=frozenset({0}), duplicate_vectors=(0, 1))


def test_the_fixture_can_place_a_duplicate_vector_pair(build_clips_table: BuildTable) -> None:
    """``duplicate_vectors`` yields two distinct clips whose vectors are identical.

    Workers C and E need a pair that is a de-duplication duplicate at any ``eps``
    while still being two separate rows, which no ``clip_id`` knob can express.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2, duplicate_vectors=(0, 3)))
    rows = lance.dataset(table.uri).to_table(columns=[KEY_COLUMN, IMAGE_COLUMN_GROUP.primary_vector]).to_pydict()

    vectors = dict(zip(rows[KEY_COLUMN], rows[IMAGE_COLUMN_GROUP.primary_vector], strict=True))

    assert table.clip_id_at(0) != table.clip_id_at(3)
    assert vectors[table.clip_id_at(0)] == vectors[table.clip_id_at(3)]


def test_the_fixture_can_place_non_finite_and_zero_norm_vectors(build_clips_table: BuildTable) -> None:
    """The two unusable-vector states are buildable and land on the named rows only.

    They are distinct failures: a non-finite coordinate would propagate NaN
    through a similarity matrix, while a zero vector merely has no direction.
    Both must reach the write as ``invalid_embedding`` rather than be dropped.
    """
    table = build_clips_table(
        ClipsTableSpec(
            fragments=1,
            rows_per_fragment=3,
            non_finite_vector_rows=frozenset({0}),
            zero_norm_vector_rows=frozenset({1}),
        )
    )
    rows = lance.dataset(table.uri).to_table(columns=[KEY_COLUMN, IMAGE_COLUMN_GROUP.primary_vector]).to_pydict()
    vectors = dict(zip(rows[KEY_COLUMN], rows[IMAGE_COLUMN_GROUP.primary_vector], strict=True))

    assert math.isinf(vectors[table.clip_id_at(0)][0])
    assert all(coordinate == 0.0 for coordinate in vectors[table.clip_id_at(1)])
    assert all(math.isfinite(coordinate) and coordinate != 0.0 for coordinate in vectors[table.clip_id_at(2)])


def test_eligibility_filter_selects_exactly_the_rows_with_every_weighted_vector(
    build_clips_table: BuildTable,
) -> None:
    """The predicate really pushes down: only rows with all weighted vectors come back."""
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2, empty_fragments=frozenset({1})))
    weights = {"subtask": 0.6, "image": 0.2, "action": 0.2}

    eligible = lance.dataset(table.uri).to_table(columns=[KEY_COLUMN], filter=eligibility_filter(weights))

    assert set(eligible[KEY_COLUMN].to_pylist()) == set(table.clip_ids[0])


def test_a_double_quoted_identifier_would_match_every_row(build_clips_table: BuildTable) -> None:
    """The bare-identifier rule is a correctness trap, demonstrated rather than cited.

    Lance parses a double-quoted name as a string LITERAL, so ``"embedding_image"
    IS NOT NULL`` is a non-null constant that matches every row - a silent
    full-corpus pass that would hand NULL vectors to the fusion kernel. The
    generated predicate must therefore contain no double quote at all.
    """
    table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2, empty_fragments=frozenset({1})))
    dataset = lance.dataset(table.uri)
    column = IMAGE_COLUMN_GROUP.primary_vector

    bare = dataset.to_table(columns=[KEY_COLUMN], filter=f"{column} IS NOT NULL").num_rows
    quoted = dataset.to_table(columns=[KEY_COLUMN], filter=f'"{column}" IS NOT NULL').num_rows

    assert bare == 2
    assert quoted == dataset.count_rows()
    assert '"' not in eligibility_filter({"subtask": 0.6, "image": 0.2, "action": 0.2})


def test_a_zero_weight_drops_its_modality_from_the_predicate() -> None:
    """A zero-weight block stops being required, which is the escape for identifier-shaped labels."""
    predicate = eligibility_filter({"subtask": 0.0, "image": 0.5, "action": 0.5})

    assert TEXT_COLUMN_GROUP.primary_vector not in predicate
    assert IMAGE_COLUMN_GROUP.primary_vector in predicate
    assert ACTION_COLUMN_GROUP.primary_vector in predicate


def test_eligibility_filter_rejects_an_all_zero_weight_vector() -> None:
    """With no weighted block no row could be evaluated, so an empty predicate is refused.

    Returning "" would push down as "no filter" and claim the entire corpus under
    a metric with no dimensions.
    """
    with pytest.raises(ValueError, match="at least one modality"):
        eligibility_filter({"subtask": 0.0, "image": 0.0, "action": 0.0})


def test_curate_reason_values_are_the_five_persisted_strings() -> None:
    """Pin the persisted reason vocabulary, including the absence of a missing-embedding value.

    These strings are a storage format: every downstream export filters on
    ``curate_selection_reason = 'selected'``, so renaming one silently empties a
    caller's result. There is deliberately no ``missing_embedding`` - a row whose
    vector is NULL is never claimed by a run, and ``embedding_action IS NULL``
    already answers the question, so a stored value would mirror a predicate.
    """
    assert [reason.value for reason in CurateReason] == [
        "selected",
        "duplicate",
        "below_quota",
        "unfunded",
        "invalid_embedding",
    ]


def test_curate_columns_are_nullable_string_and_int32() -> None:
    """Pin the persisted column types; nullability is what makes widening metadata-only."""
    assert [(field.name, str(field.type), field.nullable) for field in CURATE_COLUMNS] == [
        ("curate_selection_reason", "string", True),
        ("curate_cluster_id", "int32", True),
    ]


def test_fused_blocks_pin_the_concatenation_order_and_widths() -> None:
    """The fused layout is result-defining, so its order and widths are a stored contract.

    ``FUSED_BLOCKS`` fixes which coordinate range of the 865-wide fused vector
    belongs to which modality. A centroids artifact is only readable against the
    order that produced it, so reordering the blocks - or a modality changing its
    own width upstream - silently reinterprets every persisted centroid rather
    than failing. Widths are read off each group's own ``primary_vector`` field.
    """
    widths = [group.schema.field(group.primary_vector).type.list_size for group, _ in FUSED_BLOCKS]

    assert [(group.primary_vector, field) for group, field in FUSED_BLOCKS] == [
        ("embedding_text_subtask", "subtask"),
        ("embedding_image", "image"),
        ("embedding_action", "action"),
    ]
    assert widths == [384, 384, 97]
    assert sum(widths) == 865


def test_every_fused_block_weight_names_a_modality_weights_field() -> None:
    """The weight names are matched by string, so a config rename must fail here.

    ``eligibility_filter`` looks each block's weight up by name in a mapping the
    caller builds from ``ModalityWeights``. Nothing type-checks that hop: renaming
    a config field would surface as a ``KeyError`` on the first real run rather
    than at import.
    """
    assert {field for _, field in FUSED_BLOCKS} == set(ModalityWeights.model_fields)


def test_the_poison_set_names_every_driver_dependency_it_claims_to() -> None:
    """The poisoned set is pinned against a literal, so shrinking it is a deliberate edit.

    No purity test and no control can notice a name leaving ``POISONED_IMPORTS``:
    the purity test simply stops blocking that dependency, and the driver imports
    enough of the others that every control still fails. The guarantee would
    quietly cover one dependency fewer with nothing red, which is why the list is
    asserted here rather than trusted.

    Lives in this file rather than in ``conftest.py`` - which pytest does not
    collect tests from - beside the other literals this suite pins for the same
    reason, the persisted column types and the fused block layout.
    """
    assert POISONED_IMPORTS == ("ray", "cuml", "cudf", "cupy", "lance", "pylance")


def test_columns_and_config_import_without_lance_ray_or_gpu_libraries(run_child: RunChild) -> None:
    """The frozen contracts import on a CPU-only host with the driver libraries blocked.

    Two modules, for two opposite reasons. ``columns`` is the leaf every pure
    kernel imports - vectors, dedup and fairness all depend on it, and it imports
    nothing from the package itself - so one driver dependency landing there
    breaks all three CPU-only imports at once. ``config`` is imported by no
    kernel; that edge runs the other way, ``config`` importing ``fairness``. It is
    here because the pipeline-kind adapter resolves a config while PREPARING a
    run, on a host that has no runtime yet, and because importing it covers
    ``fairness`` transitively.

    Both also cover the package ``__init__``, which runs first and must therefore
    not re-export anything from ``pipeline``.
    """
    result = run_child(
        poisoning(
            "cosmos_curator.next.recipes.curation",
            "cosmos_curator.next.recipes.curation.columns",
            "cosmos_curator.next.recipes.curation.config",
        )
    )

    assert result.returncode == 0, result.stderr


def test_the_poisoned_import_harness_would_notice_a_driver_dependency(run_child: RunChild) -> None:
    """Negative control: the harness above passes for a reason, not by accident.

    ``sys.modules[name] = None`` is a subtle mechanism - it makes an import raise
    rather than removing the module - so a green purity test proves nothing until
    the same harness is shown to fail on a module that DOES import a driver
    dependency. Without this, a typo in the poison list would read as a passing
    purity guarantee.

    Why the check matches the mechanism's own message rather than a dependency
    name is in ``assert_poisoning_fired``.
    """
    assert_poisoning_fired(run_child(poisoning("cosmos_curator.next.recipes.curation.pipeline")))


def test_the_gpu_environment_resolves_pylance_next_to_cuml(repo_root: pathlib.Path) -> None:
    """The lock file composes Lance INTO the GPU environment, checkable without a device.

    The fit task reads ``clips.lance`` on the GPU, so ``pylance`` has to resolve
    inside the environment ``_GPU_ENV_NAME`` names - not merely in the driver's.
    Nothing about that is implied by the cuML dependency, and if a lock
    regeneration dropped it the failure would surface as an ImportError inside a
    remote task on a GPU host, which is the most expensive place to learn it.

    Complements the GPU probe below rather than replacing it: this proves the
    packages are SELECTED, the probe proves they import on a real device.
    """
    lock = yaml.safe_load((repo_root / "pixi.lock").read_text())
    environments = lock["environments"]

    assert _GPU_ENV_NAME in environments, f"pixi.lock defines no {_GPU_ENV_NAME!r} environment"
    by_platform = environments[_GPU_ENV_NAME]["packages"]
    # Asserted before the loop, because an environment resolved for no platform
    # would make every check below vacuous and the test green.
    assert by_platform, f"{_GPU_ENV_NAME} is resolved for no platform"
    for platform, entries in by_platform.items():
        selected = [str(url).rsplit("/", 1)[-1] for entry in entries for url in entry.values()]
        for package in ("pylance-", "cuml-", "cupy-"):
            assert any(name.startswith(package) for name in selected), (
                f"{_GPU_ENV_NAME}/{platform} selects no {package.rstrip('-')}"
            )


@pytest.mark.env("cuml")
def test_cuml_environment_provides_cupy_cuml_and_lance(repo_root: pathlib.Path) -> None:
    """The GPU environment named by the pipeline can import cuML, cuPy AND Lance.

    Lance is the load-bearing half: the k-means fit reads the clips table
    directly, so ``pylance`` must resolve inside the ``cuml`` pixi environment and
    not merely in the driver's. A ``runtime_env`` naming that environment is what
    makes the task independent of where the driver was launched.

    Skipped without a GPU, which makes the environment claim UNVERIFIED rather
    than proven. To run it on a GPU host, launch the default pixi environment and
    select this file with ``-m env``::

        cosmos-curator local launch --curator-path . --
            pixi run --as-is -e default python -m pytest -m env -v -rs THIS_FILE

    joined onto one line, with THIS_FILE replaced by this file's path.
    """
    # Checked here, before the child is spawned, so the skip costs nothing on a
    # CPU host: starting a whole Ray cluster in a subprocess only to discover
    # there is no device would slow the default run down for no answer.
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no NVIDIA GPU on this host; run on a GPU host to verify the cuml runtime_env")

    probe = textwrap.dedent(
        """
        import json
        import sys

        import ray

        from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
        from cosmos_curator.next.recipes.curation.pipeline import _GPU_ENV_NAME

        ray.init(include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
        if ray.cluster_resources().get("GPU", 0) < 1:
            sys.exit(3)

        @ray.remote(num_gpus=1, runtime_env=ray_data_gpu_runtime_env(_GPU_ENV_NAME))
        def _probe() -> dict[str, str]:
            # Deferred by necessity: cuML and cuPy exist only inside the
            # runtime_env, so a top-level import would fail on the driver and
            # defeat the probe.
            import cuml
            import cupy
            import lance as gpu_lance
            from cuml.cluster import KMeans

            KMeans(n_clusters=1)
            return {
                "cuml": cuml.__version__,
                "cupy": cupy.__version__,
                "lance": gpu_lance.__version__,
            }

        print(json.dumps(ray.get(_probe.remote())))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
        cwd=repo_root,
    )
    if result.returncode == _PROBE_NO_RAY_GPU:
        pytest.skip("Ray cluster exposes no GPU on this host")
    assert result.returncode == 0, result.stderr

    versions = json.loads(result.stdout.strip())

    assert set(versions) == {"cuml", "cupy", "lance"}
    assert all(versions.values())
