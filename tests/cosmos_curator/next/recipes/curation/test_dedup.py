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

"""Tests for the SemDeDup retention rule and the group stage that applies it.

Every fixture here is a hand-placed set of 2-D unit vectors, because the rule is
a statement about which pairs are compared and that is only checkable when the
similarity of every pair is known exactly. At ``eps = 0.01`` the duplicate
threshold is a cosine above 0.99, which two unit vectors clear when the angle
between them is under about 0.1416 rad; the angles below are chosen around that
boundary and their cosines are named in each test.

The two tests that matter most are the ones separating the shipped rule from the
two formulations it is routinely confused with. Neither would fail if the rule
were merely *implemented differently*; they fail only if it is a *different rule*:

    angles          0.00      0.10      0.20        cos(0.10) = 0.9950 > 0.99
                     a ------- b ------- c          cos(0.20) = 0.9801 < 0.99
    retention order  1st       2nd       3rd

    shipped rule     keep      drop      drop   c's nearest earlier row is the
    survivor-only    keep      drop      KEEP   already-dropped b, which
                                                survivor-only candidacy hides.

    angles          0.00      0.20      0.10
                     p         q         r        edges (cos > 0.99): p-r, q-r
    retention order  1st       2nd       3rd      so p, q, r are ONE component

    shipped rule     keep      keep      drop   q has no near EARLIER row, so
    components       keep      DROP      drop   only the component rule drops it.
"""

import inspect
import math
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.curation import dedup

# Weights and eps are chosen together, so a test that moved eps would be
# describing a different metric. Everything here uses the shipped default.
_EPS = 0.01

# Column names are spelled out rather than imported: these tests own their
# fixture, and a pass-through assertion is only meaningful if the name it checks
# is one the module under test never mentions. The last two are that pair - the
# fairness group keys, which dedup never names and can therefore only carry.
_CLIP_ID = "clip_id"
_FRAGMENT = "__frag"
_DEDUP_KEY = "__dedup_key"
_REASON = "curate_selection_reason"
_DISTANCE = "distance_to_centroid"
_SCORE = "max_earlier_similarity"
_VECTOR = "__vector"
_CANONICAL_TASK = "__canonical_task"
_SUBTASK_CLUSTER = "__subtask_cluster"

_VECTOR_TYPE = pa.list_(pa.float32(), 2)
_FRAGMENT_ID = 7

# The reference device and fused width the memory model is documented against.
# Spelled out rather than imported so these tests pin the model: if the geometry
# changes, the pinned numbers must be re-derived rather than quietly follow it.
_EIGHTY_GIB = 80 * 1024**3
_FUSED_DIM = 865

# Reported as the stub device's free memory, far below any total a test asks for,
# so a lookup reading the wrong half of ``mem_info`` cannot pass by coincidence.
_STUB_FREE_BYTES = 1024

# arccos(1 - eps): the exact angle at which a pair's cosine equals the duplicate
# threshold. The fan below straddles it by ``_BOUNDARY_MARGIN`` radians, which puts
# every one of its scores about 7e-5 of cosine from the verdict boundary.
_THRESHOLD_ANGLE = math.acos(1.0 - _EPS)
_BOUNDARY_MARGIN = 5e-4
_FAN_ROWS = 16

_Row = tuple[str, float, Sequence[float] | None]


def _unit(angle: float) -> list[float]:
    """Return the unit vector at ``angle`` radians, so pair cosines are ``cos(da)``."""
    return [math.cos(angle), math.sin(angle)]


def _group(
    rows: Sequence[_Row],
    *,
    dedup_key: int | Sequence[int] = 0,
    fragment_ids: int | Sequence[int] = _FRAGMENT_ID,
) -> pa.Table:
    """Build one dedup group from ``(clip_id, distance_to_centroid, vector)`` triples.

    Carries two columns the module under test never names - the fairness group
    keys - so a pass-through assertion cannot accidentally be checking a column
    the module knows about. Both vary per row, so a stage that rebuilt either
    from its own state could not reproduce them by coincidence.
    """
    ids = [clip_id for clip_id, _, _ in rows]
    keys = [dedup_key] * len(rows) if isinstance(dedup_key, int) else list(dedup_key)
    fragments = [fragment_ids] * len(rows) if isinstance(fragment_ids, int) else list(fragment_ids)
    return pa.table(
        {
            _CLIP_ID: pa.array(ids, type=pa.string()),
            _FRAGMENT: pa.array(fragments, type=pa.int32()),
            _DEDUP_KEY: pa.array(keys, type=pa.int32()),
            _CANONICAL_TASK: pa.array([f"task of {clip_id}" for clip_id in ids], type=pa.string()),
            _SUBTASK_CLUSTER: pa.array(range(len(ids)), type=pa.int32()),
            _REASON: pa.array([None] * len(rows), type=pa.string()),
            _DISTANCE: pa.array([distance for _, distance, _ in rows], type=pa.float32()),
            _VECTOR: pa.array([vector for _, _, vector in rows], type=_VECTOR_TYPE),
        }
    )


def _reasons(table: pa.Table) -> list[str | None]:
    """Return the verdict column as a plain list, in the table's own row order."""
    return list(table.column(_REASON).to_pylist())


def _score(rows: Sequence[_Row], *, tile: int = 4096, fragment_ids: int | Sequence[int] = _FRAGMENT_ID) -> np.ndarray:
    """Score ``rows`` with the pure kernel, returning input-order similarities."""
    fragments = [fragment_ids] * len(rows) if isinstance(fragment_ids, int) else list(fragment_ids)
    return dedup.max_earlier_similarity(
        [clip_id for clip_id, _, _ in rows],
        fragments,
        np.asarray([vector for _, _, vector in rows], dtype=np.float32),
        np.asarray([distance for _, distance, _ in rows], dtype=np.float32),
        tile=tile,
    )


def _boundary_fan() -> list[_Row]:
    """Return a fan of rows whose every score sits a hair from the duplicate threshold.

    Consecutive gaps alternate just inside and just outside the threshold angle, so
    the verdicts alternate with them and the duplicate mask is mixed rather than
    uniform - which is what stops a claim about verdicts from holding vacuously. On
    a monotone fan a row's nearest earlier row is always the one immediately before
    it, so each score is exactly ``cos(gap)`` and is known in advance. Distances
    descend, making retention order the list order.
    """
    angles = [0.0]
    for index in range(_FAN_ROWS - 1):
        offset = -_BOUNDARY_MARGIN if index % 2 == 0 else _BOUNDARY_MARGIN
        angles.append(angles[-1] + _THRESHOLD_ANGLE + offset)
    return [(f"clip-{index:02d}", float(_FAN_ROWS - index), _unit(angle)) for index, angle in enumerate(angles)]


def _stub_cuda(total_bytes: int) -> types.SimpleNamespace:
    """Return a ``cupy.cuda``-shaped handle whose single device reports ``total_bytes``."""
    device = types.SimpleNamespace(mem_info=(_STUB_FREE_BYTES, total_bytes))
    return types.SimpleNamespace(Device=lambda: device)


class _HostXpWithDevice:
    """``numpy`` plus a fabricated ``cuda`` handle, so the device branch runs on a CPU host.

    Every array operation is numpy's own; only the memory query is invented. That
    makes the derived tile, and the refusal, testable without a GPU - the stage's
    arithmetic is written once against an array-module handle, so the only thing
    the real device adds is the number this stub supplies.
    """

    def __init__(self, total_bytes: int) -> None:
        self.cuda = _stub_cuda(total_bytes)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - stands in for numpy or cupy
        return getattr(np, name)


def test_a_chained_row_scores_against_the_flagged_row_not_the_survivor() -> None:
    """A row's score is its similarity to the nearest EARLIER row, flagged or not.

    This is the rule's load-bearing detail, and the score is where it is visible:
    ``c`` sits 0.10 rad from the already-flagged ``b`` and 0.20 rad from the
    survivor ``a``, so the shipped rule scores it 0.9950 while any formulation
    that only considers survivors would score it 0.9801.
    """
    rows: list[_Row] = [("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.10)), ("c", 1.0, _unit(0.20))]

    scores = _score(rows)

    assert scores[2] == pytest.approx(math.cos(0.10), abs=1e-6)


def test_a_chained_row_is_a_duplicate_of_a_row_that_is_itself_a_duplicate() -> None:
    """The verdict follows the score: ``c`` is dropped for matching the dropped ``b``.

    Under survivor-only candidacy ``c`` would compare against ``a`` alone, score
    0.9801, clear the 0.99 threshold and be kept - the under-de-duplication this
    rule exists to avoid.
    """
    group = _group([("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.10)), ("c", 1.0, _unit(0.20))])

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == [None, "duplicate", "duplicate"]


def test_a_row_whose_only_near_neighbour_comes_later_survives() -> None:
    """Two survivors can share one connected component of the similarity graph.

    ``r`` is within the threshold of both ``p`` and ``q`` while ``p`` and ``q``
    are 0.20 rad apart, so all three are one component. Collapsing the component
    would keep a single representative and drop ``q``; the shipped rule keeps it,
    because ``q``'s only earlier row is the far ``p``.
    """
    group = _group([("p", 3.0, _unit(0.0)), ("q", 2.0, _unit(0.20)), ("r", 1.0, _unit(0.10))])

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == [None, None, "duplicate"]


def test_the_farthest_from_centroid_row_scores_zero() -> None:
    """The first row in retention order has no earlier neighbour, so it scores 0.0.

    Forced rather than computed: its masked similarity row is empty, and an
    unforced maximum over nothing would be ``-inf``.
    """
    rows: list[_Row] = [("near", 1.0, _unit(0.0)), ("far", 9.0, _unit(0.0))]

    assert _score(rows)[1] == pytest.approx(0.0)


def test_the_retention_order_is_distance_descending() -> None:
    """Of two identical vectors, the one farther from the centroid survives.

    Passed in nearest-first, so an implementation ordering by arrival rather than
    by distance would keep the wrong row.
    """
    group = _group([("near", 1.0, _unit(0.0)), ("far", 5.0, _unit(0.0))])

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == ["duplicate", None]


def test_an_exact_distance_tie_keeps_the_lower_clip_id() -> None:
    """Identical vectors at an identical distance resolve by ``clip_id`` ascending.

    Without the tie-break the survivor would depend on the order the rows were
    read in, making a re-run over a re-ordered table produce different verdicts.
    """
    group = _group([("b", 4.0, _unit(0.0)), ("a", 4.0, _unit(0.0))])

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == ["duplicate", None]


def test_an_exact_distance_tie_keeps_the_lower_fragment_id_across_fragments() -> None:
    """Identical ``clip_id`` and distance on different fragments resolve by fragment id.

    ``clip_id`` is unique only within a fragment, so the same id on two fragments
    must not fall back to Ray input order when the distance also ties.
    """
    group = _group(
        [("same", 4.0, _unit(0.0)), ("same", 4.0, _unit(0.0))],
        fragment_ids=[9, 3],
    )

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == ["duplicate", None]


def test_scores_are_returned_in_the_input_row_order() -> None:
    """Scoring reorders internally but reports back aligned to the rows it was given.

    The farthest row is passed second here, so its 0.0 must land in slot 1; an
    implementation returning retention order would put it in slot 0.
    """
    rows: list[_Row] = [("c", 1.0, _unit(0.20)), ("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.10))]

    scores = _score(rows)

    assert scores[1] == pytest.approx(0.0)


def test_the_boundary_fixture_straddles_the_duplicate_threshold() -> None:
    """The premise every tile claim below rests on: the fan really does sit on the boundary.

    A fixture whose rows are all comfortably far from ``1 - eps`` would make
    "tiling moves no verdict" hold no matter what the tile did, because no verdict
    was ever close to moving. Pinned separately so a fixture that drifted off the
    boundary is reported as a broken premise instead of silently emptying the
    invariance test below.
    """
    scores = _score(_boundary_fan())[1:]
    mask = dedup.duplicate_mask(scores, _EPS)

    assert sorted(set(mask.tolist())) == [False, True]
    assert float(np.max(np.abs(scores - (1.0 - _EPS)))) < 1e-4


def test_the_tile_size_changes_no_verdict_and_no_score_beyond_float32_rounding() -> None:
    """Tiling is a memory bound, not a result: a tiny tile and a huge one agree.

    Agreement is exact in the verdicts and only float32-exact in the scores. The
    tile IS the GEMM's row count, so BLAS picks a different microkernel per block
    shape and a pair's cosine can move by an ulp. Bit-equality is therefore the
    wrong assertion - it would pass or fail on the host's BLAS - while a moved
    verdict would mean the cluster-size tail silently decides what is a duplicate.
    The tolerance is 1e-6, three orders of magnitude below the fixture's ~7e-5
    margin from the threshold and four above one ulp at 0.99: tight enough that a
    real arithmetic change fails it, loose enough to absorb the documented drift.
    """
    rows = _boundary_fan()

    small, large = _score(rows, tile=3), _score(rows, tile=1000)

    assert np.array_equal(dedup.duplicate_mask(small, _EPS), dedup.duplicate_mask(large, _EPS))
    assert np.allclose(small, large, rtol=0.0, atol=1e-6)


def test_the_device_budget_reads_the_total_and_not_the_free_memory() -> None:
    """Total, so the tile - and the ulp of drift it carries - is a property of the card.

    Free memory depends on whatever else the card is doing, so a tile derived from
    it would differ between two runs of the same job over the same data, turning a
    documented per-GPU-model property into a per-run one.
    """
    assert dedup._device_total_bytes(_HostXpWithDevice(_EIGHTY_GIB)) == _EIGHTY_GIB


def test_a_host_array_module_reports_no_device_budget() -> None:
    """Plain numpy has no device to budget against, so the derivation is skipped.

    Answering with a number instead would let a CPU-only host silently take the
    shrinking or the refusal path against a budget that means nothing there.
    """
    assert dedup._device_total_bytes(np) is None


def test_a_cluster_at_the_configured_mean_size_keeps_the_default_tile() -> None:
    """The cap invariant: a cluster that fits today is scored exactly as it was.

    200,000 rows is the shipped ``target_mean_cluster_rows``, and the reserve is
    sized so the tile holds at the cap until roughly 9.6x that. Returning anything
    else here would re-block every GEMM the leg has ever run, which is precisely
    what makes the derivation an extension rather than a change.
    """
    tile = dedup._group_tile_rows(rows=200_000, width=_FUSED_DIM, dedup_key=0, device_total_bytes=_EIGHTY_GIB)

    assert tile == 4096


def test_the_reference_device_holds_about_eleven_million_fused_rows() -> None:
    """The ceiling is set by the vectors being held TWICE, not once.

    865 float32 is 3,460 bytes a row, and ``_unit_normalize`` allocates its result
    while its input is still referenced, so a row costs 6,920 bytes for the
    duration of that call plus 12 of whole-group index - which puts the reserved
    90% of an 80 GiB device at 11,152,540 rows. A model counting a single copy
    would put the refusal at twice that, which is to say after the card is
    already gone.
    """
    assert dedup.max_group_rows(width=_FUSED_DIM, device_total_bytes=_EIGHTY_GIB) == 11_152_540


@pytest.mark.parametrize(("rows", "tile"), [(3_400_000, 2048), (3_800_000, 1024)])
def test_a_cluster_too_large_for_the_default_tile_gets_a_smaller_one(rows: int, tile: int) -> None:
    """A cluster the cap cannot hold shrinks its tile instead of exhausting the card.

    Both row counts are 17-19x the configured mean and past the size at which the
    cap stops fitting. Neither is round, because together they bracket the per-cell
    coefficient from both sides: at 3.4M the exact quotient is 2140, so any model
    charging 10 bytes or more - the 13 a redundant cast would cost - drops to 1024;
    at 3.8M it is 1875, so any model charging 8 or less rises to 2048. Only 9
    answers both.
    """
    assert dedup._group_tile_rows(rows=rows, width=_FUSED_DIM, dedup_key=0, device_total_bytes=_EIGHTY_GIB) == tile


@pytest.mark.parametrize("rows", [2_000_000, 3_000_000, 4_000_000, 6_000_000, 9_000_000])
def test_the_derived_tile_is_quantized_down_to_a_power_of_two(rows: int) -> None:
    """Quantized for the same reason the budget is total memory: stability.

    The exact quotient at each of these row counts is a ragged number (3909, 2477,
    1761, 1045, 568), so a derivation that skipped the quantization fails every
    case. Rounding to a power of two means a cluster one row larger scores
    identically instead of re-blocking its GEMM for a few kilobytes.
    """
    tile = dedup._group_tile_rows(rows=rows, width=_FUSED_DIM, dedup_key=0, device_total_bytes=_EIGHTY_GIB)

    assert tile & (tile - 1) == 0


def test_the_largest_feasible_cluster_still_gets_a_tile_of_one() -> None:
    """At the refusal boundary the derivation still returns a usable tile.

    Deliberately unclamped: the feasible bound already charges a tile-of-one block
    against the same budget, so a tile below one is unreachable unless those two
    formulas stop agreeing. The width is 2 rather than the fused 865 because that
    is where the similarity block, not the normalize copy, is the binding term -
    which is the only case in which the boundary is tight.
    """
    total_bytes = 1_000_000
    feasible = dedup.max_group_rows(width=2, device_total_bytes=total_bytes)

    tile = dedup._group_tile_rows(rows=feasible, width=2, dedup_key=0, device_total_bytes=total_bytes)

    assert tile == 1


def test_a_cluster_too_large_for_any_tile_is_refused_by_name_and_by_number() -> None:
    """One row past the feasible bound the stage refuses, and says everything needed to act.

    The refusal happens inside a Ray UDF, where the exception type is erased and
    only the message reaches the operator - so the message carries the cluster's
    row count, the device it was measured against, and the maximum that device
    holds, rather than expecting the reader to have the model to hand.
    """
    rows = dedup.max_group_rows(width=_FUSED_DIM, device_total_bytes=_EIGHTY_GIB) + 1

    with pytest.raises(ValueError, match=rf"holds {rows} rows.*80\.0 GiB device holds at most {rows - 1} rows"):
        dedup._group_tile_rows(rows=rows, width=_FUSED_DIM, dedup_key=11, device_total_bytes=_EIGHTY_GIB)


def test_the_refusal_names_the_cluster_and_reaches_the_stage() -> None:
    """The guard is wired into the group stage, and names the cluster to look at.

    Without the wiring the derivation would be dead code and the group would reach
    the GEMM to fail as a CUDA out-of-memory error, which names nothing.
    """
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))], dedup_key=4)

    with pytest.raises(ValueError, match=r"__dedup_key=4 holds 2 rows"):
        dedup.mark_duplicates(group, eps=_EPS, xp=_HostXpWithDevice(50))


def test_the_stage_scoring_a_shrunk_tile_reaches_the_same_verdicts() -> None:
    """End to end: a device too small for the cap re-blocks the GEMM and agrees anyway.

    A 1000-byte device over the 16-row fan derives a tile of 4, so the group is
    scored in four blocks rather than one - asserted first, because a stub device
    roomy enough to keep the cap would make the comparison below compare one tile
    size with itself. Every verdict in that fan is decided by about 7e-5 of
    cosine, so a re-blocking that moved the arithmetic materially would move a
    verdict here.
    """
    group = _group(_boundary_fan())

    shrunk = dedup.mark_duplicates(group, eps=_EPS, xp=_HostXpWithDevice(1000))

    assert dedup._group_tile_rows(rows=_FAN_ROWS, width=2, dedup_key=0, device_total_bytes=1000) == 4
    assert _reasons(shrunk) == _reasons(dedup.mark_duplicates(group, eps=_EPS))


def test_a_row_at_exactly_one_minus_eps_survives_and_a_tighter_eps_drops_it() -> None:
    """The duplicate boundary is strict: the verdict flips one float32 step below the score.

    Read the score first and derive ``eps`` from it, so the boundary is hit
    exactly instead of approached. Both comparisons happen in float32, which is
    what stops a row on the boundary from being judged one way here and the other
    way by a reader working in float64.
    """
    scores = _score([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.05))])
    score = scores[1]

    assert not bool(dedup.duplicate_mask(scores, 1.0 - float(score))[1])
    assert bool(dedup.duplicate_mask(scores, 1.0 - float(np.nextafter(score, np.float32(0.0))))[1])


def test_a_zero_norm_row_is_never_a_duplicate() -> None:
    """A zero vector normalizes to zero, whose cosine to anything is 0, not NaN.

    Such a row should have been routed around this stage as invalid, so reaching
    here is a defect upstream; scoring it 0.0 keeps the defect to one row instead
    of propagating NaN through the whole group's GEMM.
    """
    group = _group([("real", 2.0, _unit(0.0)), ("zero", 1.0, [0.0, 0.0])])

    assert _reasons(dedup.mark_duplicates(group, eps=_EPS)) == [None, None]


def test_embeddings_that_do_not_match_the_id_count_are_rejected() -> None:
    """A vector matrix with the wrong row count pairs every row with another's vector."""
    with pytest.raises(ValueError, match="must be an"):
        dedup.max_earlier_similarity(
            ["a", "b"],
            [0, 1],
            np.asarray([_unit(0.0)], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
        )


def test_every_input_row_yields_exactly_one_verdict_row() -> None:
    """The stage is cardinality-preserving, so no eligible row can miss the write.

    Duplicates are marked, never dropped: a filtered row would reach storage with
    a NULL reason, indistinguishable from a row the run never claimed.
    """
    group = _group([("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.10)), ("c", 1.0, _unit(0.20))])

    assert dedup.mark_duplicates(group, eps=_EPS).num_rows == group.num_rows


def test_the_working_vector_does_not_survive_the_stage() -> None:
    """The vector's last reader drops it, keeping it out of the fairness shuffle."""
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))])

    assert _VECTOR not in dedup.mark_duplicates(group, eps=_EPS).schema.names


def test_the_score_the_verdict_was_taken_from_is_emitted_per_row() -> None:
    """The threshold's own input reaches the report, so ``dedup_eps`` is interpretable.

    Asserted against the kernel that produced the verdict rather than against a
    literal: the column has to be THAT similarity, not merely a plausible float,
    or the distribution an operator reads their eps against describes nothing.
    """
    rows = [("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.05)), ("c", 1.0, _unit(1.00))]
    group = _group(rows)

    emitted = dedup.mark_duplicates(group, eps=_EPS).column(_SCORE).to_pylist()

    assert emitted == pytest.approx([float(score) for score in _score(rows)])


def test_the_fragment_column_survives_the_stage() -> None:
    """``__frag`` rides through unchanged, because the write is fragment-scoped."""
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))])

    result = dedup.mark_duplicates(group, eps=_EPS)

    assert result.column(_FRAGMENT).to_pylist() == [_FRAGMENT_ID, _FRAGMENT_ID]


def test_the_fairness_group_keys_and_the_distance_survive_the_stage() -> None:
    """Columns the stage does not write ride through untouched, values included.

    Fairness keys every group on the ``(canonical task, subtask cluster)`` pair and
    ranks a funded group's survivors by the distance, so a column lost here would
    leave it unable to count a group's capacity or to order the cut inside one.
    """
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))])

    result = dedup.mark_duplicates(group, eps=_EPS)

    for name in (_CANONICAL_TASK, _SUBTASK_CLUSTER, _DISTANCE):
        assert result.column(name).to_pylist() == group.column(name).to_pylist()


def test_a_bypassed_group_keeps_every_column_and_value_but_the_vector() -> None:
    """A group keyed below zero is returned as it arrived, minus the vector, plus a NULL score.

    Its vectors are non-finite - that is why it was routed here - so nothing may
    read them. Comparing against the input with the vector dropped and the score
    appended proves no GEMM ran and no reason was rewritten.
    """
    group = _group(
        [("bad", 2.0, [float("nan"), 0.0]), ("worse", 1.0, [float("inf"), 0.0])],
        dedup_key=-1,
    )

    result = dedup.mark_duplicates(group, eps=_EPS)

    assert result.drop_columns([_SCORE]).equals(group.drop_columns([_VECTOR]))
    assert result.column(_SCORE).to_pylist() == [None, None]


def test_a_bypassed_group_and_a_scored_group_emit_the_same_schema() -> None:
    """Both branches must agree, or their blocks cannot be concatenated downstream.

    The bypass is the whole reason this can go wrong: it is the one path that
    returns without touching the reason column.
    """
    bypassed = dedup.mark_duplicates(_group([("bad", 1.0, [float("nan"), 0.0])], dedup_key=-1), eps=_EPS)
    scored = dedup.mark_duplicates(_group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))]), eps=_EPS)

    assert bypassed.schema.equals(scored.schema)


def test_a_sliced_group_scores_its_own_rows() -> None:
    """A group carrying a buffer offset reads its own vectors, not the ones before it.

    ``b`` and ``c`` are 0.05 rad apart and both far from ``a``, so reading the
    slice's coordinates from the start of the underlying buffer would score the
    pair as ``(a, b)`` - 0.5 rad apart - and keep both.
    """
    group = _group([("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.50)), ("c", 1.0, _unit(0.55))])

    assert _reasons(dedup.mark_duplicates(group.slice(1, 2), eps=_EPS)) == [None, "duplicate"]


def test_a_group_naming_two_dedup_keys_is_rejected() -> None:
    """Rows from two clusters in one group would be scored against each other.

    Grouping on the key already guarantees this, and the check stays because the
    failure is otherwise silent: the verdicts would be plausible and unjustified
    by any cluster boundary.
    """
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, _unit(0.10))], dedup_key=[0, 1])

    with pytest.raises(ValueError, match="exactly one"):
        dedup.mark_duplicates(group, eps=_EPS)


def test_a_group_missing_the_distance_column_is_rejected() -> None:
    """A missing column is named up front rather than surfacing from inside the GEMM."""
    group = _group([("a", 2.0, _unit(0.0))]).drop_columns([_DISTANCE])

    with pytest.raises(ValueError, match=_DISTANCE):
        dedup.mark_duplicates(group, eps=_EPS)


def test_a_null_vector_inside_a_scored_group_is_rejected() -> None:
    """A NULL vector belongs in the bypass; flattening past one would shift later rows."""
    group = _group([("a", 2.0, _unit(0.0)), ("b", 1.0, None)])

    with pytest.raises(ValueError, match="NULL"):
        dedup.mark_duplicates(group, eps=_EPS)


def test_the_group_udf_exposes_a_name() -> None:
    """``map_groups`` reads ``__name__`` off the callable, so a partial or lambda fails."""
    assert getattr(dedup.dedup_group, "__name__", "") == "dedup_group"


def test_the_launch_arguments_run_the_udf_inside_the_named_gpu_environment() -> None:
    """The GPU task names its pixi environment, instead of inheriting the driver's.

    Without this the task imports cuPy only when the driver happens to have been
    launched in that same environment - a run that works on one host and fails on
    the next with an ImportError from inside a worker.
    """
    args = dedup.dedup_launch_args(gpu_env_name="probe-env", concurrency=None)

    assert "probe-env" in args["runtime_env"]["py_executable"]


def test_the_launch_arguments_reserve_a_whole_gpu_and_ask_for_arrow_batches() -> None:
    """One card per group, and pyarrow rather than the default pandas batch format."""
    args = dedup.dedup_launch_args(gpu_env_name="probe-env", concurrency=None)

    assert (args["num_gpus"], args["batch_format"]) == (1, "pyarrow")


def test_the_launch_arguments_bind_to_the_installed_group_stage_signature() -> None:
    """Every argument is accepted by ``map_groups`` as it is actually installed.

    A structural check against the live API, not against a recorded parameter
    list: ``runtime_env`` is not a declared parameter but reaches the task
    through the variadic remote-args tail, so a Ray release that stopped
    forwarding it would silently drop the GPU environment. Binding fails loudly
    instead.
    """
    from ray.data.grouped_data import GroupedData  # noqa: PLC0415 - deferred; Ray is heavy to import

    args = dedup.dedup_launch_args(gpu_env_name="probe-env", concurrency=1)

    inspect.signature(GroupedData.map_groups).bind(object(), dedup.dedup_group, fn_kwargs={"eps": _EPS}, **args)


def test_the_concurrency_cap_passes_through_to_the_launch_arguments() -> None:
    """The cap is forwarded verbatim; it bounds GPU tasks and nothing about the result."""
    assert dedup.dedup_launch_args(gpu_env_name="probe-env", concurrency=3)["concurrency"] == 3


@pytest.mark.env("cuml")
def test_the_gpu_path_marks_the_same_duplicates_as_the_cpu_kernel() -> None:
    """The cuPy and numpy paths reach the same verdicts on the chain fixture.

    The only claim needing a device: the tiled GEMM is written once against an
    array-module handle, so this checks that cuPy honours the same masking and
    float32 comparison rather than re-checking the rule.
    """
    group = _group([("a", 3.0, _unit(0.0)), ("b", 2.0, _unit(0.10)), ("c", 1.0, _unit(0.20))])

    assert _reasons(dedup.dedup_group(group, eps=_EPS)) == _reasons(dedup.mark_duplicates(group, eps=_EPS))
