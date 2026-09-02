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

"""Fairness tests: canonicalization, label merging, nested quotas, the within-group cut.

Pure arithmetic and Arrow; no filesystem, no Ray, no GPU. Every input is built
here, so only a change in the mechanism under test can turn one of these red.

Three fixtures are load-bearing rather than incidental:

- Label fixtures are instruction PROSE, not ``task_4`` identifiers. On an
  identifier all four canonicalization steps are no-ops, so an identifier fixture
  would pass whatever the function did.
- Quota fixtures are UNEVEN. A uniform population never asks a group for more
  than it holds, which makes max-min water-fill a no-op and leaves its
  redistribution branch unreachable while the suite still passes.
- Merge fixtures state their cosine similarities explicitly and are asserted
  against a permuted input order, because a merge whose leaders depend on row
  order is unreproducible and nothing about it is persisted.

Level-2 keys here are small integers because that is what a level-2 key IS: a
cell of the subtask-text partition, not a label. This module never assigns one --
``vectors`` does, and ``test_vectors`` covers it - so these tests state the ids
directly and stay independent of how a centroid basis was fitted.
"""

import json
import random
import re
import textwrap
from collections.abc import Hashable, Sequence
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import KEY_COLUMN
from cosmos_curator.next.recipes.curation import fairness as curation_fairness
from cosmos_curator.next.recipes.curation.columns import (
    CANONICAL_TASK_COLUMN,
    CURATE_SELECTION_REASON,
    DEDUP_KEY_COLUMN,
    DISTANCE_COLUMN,
    FRAGMENT_COLUMN,
    NO_SUBTASK_CLUSTER,
    SUBTASK_CLUSTER_COLUMN,
    TASK_COLUMN,
    VERDICT_ROW,
    CurateReason,
    WithinGroupOrder,
)
from cosmos_curator.next.recipes.curation.fairness import (
    _TRAILING_PUNCTUATION,
    FairnessQuota,
    Level2Key,
    _pair_digest_keys,
    apply_label_merge,
    canonicalization_contract,
    canonicalize_label,
    canonicalize_labels,
    merge_labels,
    residual_ranks,
    select_within_quota,
    survivor_group_counts,
)

from .conftest import RunChild, assert_poisoning_fired, poisoning

# A pair-keyed count map is the shape the driver holds after the group count. The
# quota tests use short task strings and small cluster ids because only their
# ORDER matters to the maths.
Counts = dict[Level2Key, int]

_RAY_COUNT_COLUMN = "count()"


def _quotas(counts: Counts, target: int) -> dict[Level2Key, int]:
    """Build a FairnessQuota from a pair-count map and return its level-2 quotas.

    ``unfunded_groups`` is called on the way out because that is where the
    degeneracy warning lives, and calling it is what the driver does with every
    allocation; a helper that skipped it would leave the threshold untested.
    """
    keys = list(counts)
    quota = FairnessQuota.build(
        level2_keys=keys,
        level2_counts=[counts[key] for key in keys],
        target=target,
    )
    quotas = quota.quotas()
    quota.unfunded_groups(quotas)
    return quotas


def _funded(counts: Counts, target: int, *, residual_seed: int = 0) -> frozenset[Level2Key]:
    """Return the groups drawing a non-zero quota at ``residual_seed``."""
    keys = list(counts)
    quota = FairnessQuota.build(
        level2_keys=keys,
        level2_counts=[counts[key] for key in keys],
        target=target,
        residual_seed=residual_seed,
    )
    return frozenset(key for key, allocated in quota.quotas().items() if allocated > 0)


def _reference_level[K: Hashable](keys_ascending: Sequence[K], capacity: dict[K, int], target: int) -> dict[K, int]:
    """Naive one-level max-min: round-robin one row at a time in key-ascending order."""
    quota = dict.fromkeys(keys_ascending, 0)
    remaining = min(target, sum(capacity.values()))
    while remaining > 0:
        progressed = False
        for key in keys_ascending:
            if remaining == 0:
                break
            if quota[key] < capacity[key]:
                quota[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quota


def _by_residual[K: Hashable](keys_ascending: Sequence[K], digest_keys: Sequence[str], seed: int) -> list[K]:
    """Return ``keys_ascending`` reordered by the production residual rank.

    Round-robin is an independent derivation of max-min, which is the point of the
    reference. The ORDER it walks is deliberately not re-derived: it calls the
    production ``residual_ranks``, because a second digest implementation here
    would only pin that two copies of the same hash agree.
    """
    ranks = residual_ranks(list(digest_keys), seed)
    return [keys_ascending[index] for index in np.argsort(ranks, kind="stable")]


def _reference_nested(counts: Counts, target: int, residual_seed: int = 0) -> dict[Level2Key, int]:
    """Naive nested reference: level-1 round-robin, then per-parent level-2 round-robin.

    Both tiers walk in residual order, level 1 keyed on the task label and level 2
    on the ``(task, cell)`` pair, so the reference derives the ARITHMETIC
    independently while sharing the production comparator.
    """
    level1: dict[str, int] = {}
    for (parent, _child), count in counts.items():
        level1[parent] = level1.get(parent, 0) + count
    parents_ascending = sorted(level1)
    level1_quota = _reference_level(
        _by_residual(parents_ascending, parents_ascending, residual_seed),
        level1,
        target,
    )

    result: dict[Level2Key, int] = {}
    for parent in parents_ascending:
        children = {key: count for key, count in counts.items() if key[0] == parent}
        cells_ascending = sorted(children)
        result.update(
            _reference_level(
                _by_residual(cells_ascending, _pair_digest_keys(cells_ascending), residual_seed),
                children,
                level1_quota[parent],
            )
        )
    return result


def _group(
    rows: Sequence[tuple[str, float, str | None]],
    *,
    task: str = "wash the dishes",
    cluster: int = 4,
    fragment: int | Sequence[int] = 3,
    dedup_key: int = 7,
) -> pa.Table:
    """Build one post-dedup fairness group from ``(clip_id, distance, reason)`` triples.

    Carries every column the cut reads plus the two it must drop, so a projection
    that leaked a label or a distance would be visible.
    """
    fragments = [fragment] * len(rows) if isinstance(fragment, int) else list(fragment)
    return pa.table(
        {
            KEY_COLUMN: pa.array([clip_id for clip_id, _, _ in rows], pa.string()),
            FRAGMENT_COLUMN: pa.array(fragments, pa.int32()),
            DEDUP_KEY_COLUMN: pa.array([dedup_key] * len(rows), pa.int32()),
            CANONICAL_TASK_COLUMN: pa.array([task] * len(rows), pa.string()),
            SUBTASK_CLUSTER_COLUMN: pa.array([cluster] * len(rows), pa.int32()),
            DISTANCE_COLUMN: pa.array([distance for _, distance, _ in rows], pa.float32()),
            CURATE_SELECTION_REASON: pa.array([reason for _, _, reason in rows], pa.string()),
        }
    )


def _group_key(group: pa.Table) -> Level2Key:
    """Read one group's level-2 key the way the cut itself composes it."""
    return (
        group.column(CANONICAL_TASK_COLUMN)[0].as_py(),
        int(group.column(SUBTASK_CLUSTER_COLUMN)[0].as_py()),
    )


def _reasons(group: pa.Table, quota: int, order: WithinGroupOrder = "farthest") -> dict[str, str | None]:
    """Run the cut over one group and return ``clip_id -> reason``."""
    verdicts = select_within_quota(group, {_group_key(group): quota}, order)
    return dict(
        zip(
            verdicts.column(KEY_COLUMN).to_pylist(),
            verdicts.column(CURATE_SELECTION_REASON).to_pylist(),
            strict=True,
        )
    )


def _unit(cosine: float) -> list[float]:
    """Return a 2-D unit vector whose cosine against ``[1, 0]`` is exactly ``cosine``."""
    return [cosine, float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))]


def _merge(labelled: Sequence[tuple[str, int, float]], *, theta: float) -> dict[str, str]:
    """Merge ``(label, clip_count, cosine_against_the_x_axis)`` triples at ``theta``."""
    vectors = np.array([_unit(cosine) for _, _, cosine in labelled], dtype=np.float32)
    return merge_labels(
        [label for label, _, _ in labelled],
        [count for _, count, _ in labelled],
        vectors,
        theta=theta,
    )


# Cosines against the x-axis whose closest pair still sits at ~0.904, below every
# theta these tests use. Members of one group share a cosine and so always merge;
# two groups never do. That pins n_out to the group count exactly.
_SEPARATED_COSINES = (1.0, 0.6, 0.2, -0.2, -0.6)


def _merge_groups(group_sizes: Sequence[int]) -> dict[str, str]:
    """Merge a vocabulary of ``sum(group_sizes)`` labels forming ``len(group_sizes)`` groups.

    Lets a test state the collapse ratio the warning is computed from - ``n_in``
    is the sum, ``n_out`` the length - without hand-building vectors per case.
    """
    labelled = [
        (f"group {index} wording {member}", 10 - index, _SEPARATED_COSINES[index])
        for index, size in enumerate(group_sizes)
        for member in range(size)
    ]
    return _merge(labelled, theta=0.95)


def _warnings(records: Sequence[dict[str, Any]]) -> list[str]:
    """Return the WARNING messages captured so far."""
    return [record["message"] for record in records if record["level"].name == "WARNING"]


def _count_rows(rows: Sequence[tuple[str, int, str | None, int]]) -> pa.Table:
    """Build the O(G) group-count table Ray's groupby-count produces."""
    return pa.table(
        {
            CANONICAL_TASK_COLUMN: pa.array([task for task, _, _, _ in rows], pa.string()),
            SUBTASK_CLUSTER_COLUMN: pa.array([cluster for _, cluster, _, _ in rows], pa.int32()),
            CURATE_SELECTION_REASON: pa.array([reason for _, _, reason, _ in rows], pa.string()),
            _RAY_COUNT_COLUMN: pa.array([count for _, _, _, count in rows], pa.int64()),
        }
    )


def test_canonicalization_folds_a_trailing_period_onto_the_bare_instruction() -> None:
    """Two writings of one instruction differing only by its full stop reach one group."""
    assert canonicalize_label("add the dough to the portion on the scale.") == canonicalize_label(
        "add the dough to the portion on the scale"
    )


def test_canonicalization_folds_case() -> None:
    """A sentence-cased instruction and a lowercase one reach one group."""
    assert canonicalize_label("Add The Dough To The Scale") == canonicalize_label("add the dough to the scale")


def test_canonicalization_collapses_interior_whitespace() -> None:
    """Runs of spaces, tabs, and newlines inside an instruction collapse to one space each."""
    assert canonicalize_label("add the\tdough  to\nthe scale") == canonicalize_label("add the dough to the scale")


def test_canonicalization_keeps_two_different_instructions_apart() -> None:
    """Folding must not be so aggressive that distinct instructions collide."""
    assert canonicalize_label("add dough to the scale.") != canonicalize_label("remove dough from the scale.")


def test_canonicalization_normalizes_equivalent_unicode_sequences() -> None:
    """Composed and decomposed accents denote the same characters and must compare equal."""
    assert canonicalize_label("cr\u00eape the batter") == canonicalize_label("cre\u0302pe the batter")


def test_canonicalization_strips_a_mark_separated_by_a_space() -> None:
    """The space is inside the stripped set, so a detached trailing mark folds away too."""
    assert canonicalize_label("open the folder . ") == "open the folder"


def test_canonicalization_keeps_a_leading_mark() -> None:
    """Stripping is trailing-only: a leading mark is part of the label, not decoration."""
    assert canonicalize_label("...restack the plates") == "...restack the plates"


def test_a_wholly_punctuation_label_folds_to_the_blank_group() -> None:
    """A label with no letters has a canonical form, and it is the (legitimate) blank key."""
    assert canonicalize_label(" ... ") == ""


def test_canonicalization_is_a_no_op_on_identifier_shaped_labels() -> None:
    """Why the fixtures above are prose: on an identifier every step does nothing.

    NFC and whitespace collapse have nothing to act on, casefold cannot change an
    already-lowercase id, and the trailing strip finds a digit. A suite built on
    ``task_4`` labels would therefore pass no matter what the four steps did.
    """
    assert canonicalize_label("subtask_6186") == "subtask_6186"


@pytest.mark.parametrize(
    ("probe", "expected", "step"),
    [
        ("e\u0301", "\u00e9", "NFC must compose the combining acute"),
        ("\u00b2", "\u00b2", "NFC, not NFKC, must leave a compatibility-only decomposition alone"),
        ("a \t\n b", "a b", "a run of mixed whitespace must collapse to one space"),
        (" a.\t", "a", "the collapse must trim the ends before the strip runs, leaving no mark"),
        ("\u00df", "ss", "casefold, not lower, must fold the sharp s"),
        (f".a {_TRAILING_PUNCTUATION}", ".a", "every trailing mark strips, a leading one survives"),
        (_TRAILING_PUNCTUATION, "", "a label of nothing but marks folds to the blank group"),
    ],
)
def test_every_folding_step_is_observable_in_the_published_contract(probe: str, expected: str, step: str) -> None:
    """Each probe still folds the way the step it witnesses requires.

    Two runs that fold labels differently build different fairness groups, and the
    run identity hashes these pairs to say so. That holds only while each step
    leaves a mark, and a finite probe set earns that per step rather than in
    general: these pin every step's PRESENCE, plus the two SUBSTITUTIONS a
    near-equivalent step would otherwise slip through unseen. Membership of the
    strip set is the next test's concern, which is why two probes derive their
    text from it rather than spelling it out.

    Parametrized so a step that regresses is named on its own, instead of hiding
    behind whichever assertion happened to be written first.
    """
    assert dict(canonicalization_contract())[probe] == expected, step


def test_the_published_contract_probes_no_label_twice() -> None:
    """Distinct probes, so reading the contract as a mapping loses no step.

    The per-step test above looks each probe up by key. Two probes folding from
    the same text would collapse into one entry, and the step whose entry lost
    would then be asserted against its twin's value while still reporting a pass.
    """
    pairs = canonicalization_contract()

    assert len(dict(pairs)) == len(pairs), pairs


def test_dropping_one_trailing_mark_changes_the_published_contract() -> None:
    """Every mark in the strip set must be witnessed, not just the ones spelled out.

    The set decides which spellings reach one fairness key, so removing a mark
    regroups the corpus. Probes that named a mark by hand exercised only that
    mark, leaving the rest of the set free to change under an identity that
    reported no difference.
    """
    before = canonicalization_contract()

    for mark in _TRAILING_PUNCTUATION:
        thinner = _TRAILING_PUNCTUATION.replace(mark, "")
        with mock.patch.object(curation_fairness, "_TRAILING_PUNCTUATION", thinner):
            assert canonicalization_contract() != before, f"dropping {mark!r} left the contract unmoved"


def test_the_batch_helper_appends_the_canonical_task_column() -> None:
    """The scan's entry point folds the task label and leaves the raw one intact."""
    batch = pa.table({TASK_COLUMN: pa.array(["Wash  Dishes."])})

    folded = canonicalize_labels(batch)

    assert folded.column(CANONICAL_TASK_COLUMN).to_pylist() == ["wash dishes"]
    assert folded.column(TASK_COLUMN).to_pylist() == ["Wash  Dishes."]


def test_the_batch_helper_leaves_the_subtask_label_alone() -> None:
    """Level 2 is a cluster over the subtask EMBEDDING, so no canonical subtask column exists.

    A fold of ``subtask_name`` would be dead output: nothing keys a group on it,
    and a surviving column would invite a reader to group on the prose again.
    """
    batch = pa.table({TASK_COLUMN: pa.array(["Wash Dishes"]), "subtask_name": pa.array(["Place ON the Scale "])})

    folded = canonicalize_labels(batch)

    assert folded.column("subtask_name").to_pylist() == ["Place ON the Scale "]
    assert "canonical_subtask" not in folded.schema.names


def test_the_batch_helper_rejects_a_null_label() -> None:
    """A row with no label has no fairness group, so it must never reach the count silently."""
    batch = pa.table({TASK_COLUMN: pa.array([None], pa.string())})

    with pytest.raises(ValueError, match="NULL task_name"):
        canonicalize_labels(batch)


def test_a_label_within_theta_joins_the_higher_count_label() -> None:
    """Wording variants collapse onto the frequent label, which is what merging is for."""
    merged = _merge([("wash the dishes", 100, 1.0), ("wash dishes", 3, 0.99)], theta=0.95)

    assert merged == {"wash the dishes": "wash the dishes", "wash dishes": "wash the dishes"}


def test_the_representative_is_the_higher_count_label_not_the_earlier_one() -> None:
    """Frequency, not input position, decides who absorbs whom."""
    merged = _merge([("a rare wording", 1, 1.0), ("the common wording", 500, 0.99)], theta=0.95)

    assert merged["a rare wording"] == "the common wording"


def test_equal_counts_are_broken_by_the_ascending_label() -> None:
    """Without the label tie-break, two equally frequent labels would merge by scan order."""
    merged = _merge([("zebra crossing", 7, 1.0), ("apple crumble", 7, 0.99)], theta=0.95)

    assert merged["zebra crossing"] == "apple crumble"


def test_a_label_far_from_every_representative_keeps_its_own_group() -> None:
    """Merging is opt-in per pair: dissimilar labels stay separate fairness groups."""
    merged = _merge([("wash the dishes", 10, 1.0), ("sand the tabletop", 4, 0.10)], theta=0.95)

    assert merged == {"wash the dishes": "wash the dishes", "sand the tabletop": "sand the tabletop"}


def test_similarity_must_strictly_exceed_theta_to_merge() -> None:
    """The boundary is strict, matching the retention rule's own ``>`` convention."""
    similarity = float(np.dot(np.array(_unit(1.0), dtype=np.float32), np.array(_unit(0.98), dtype=np.float32)))
    labelled = [("first wording", 9, 1.0), ("second wording", 2, 0.98)]

    assert _merge(labelled, theta=similarity)["second wording"] == "second wording"
    assert _merge(labelled, theta=similarity - 1e-6)["second wording"] == "first wording"


def test_a_chain_of_near_neighbours_cannot_collapse_into_one_group() -> None:
    """Every member is within theta of its REPRESENTATIVE, never of a transitive chain.

    ``b`` is close to ``a`` and ``c`` is close to ``b``, but ``c`` is far from
    ``a``. Connected components would put all three in one group - the failure
    mode that silently disables fairness - so ``c`` must stay separate.
    """
    merged = _merge(
        [("a wording", 30, 1.0), ("b wording", 20, 0.96), ("c wording", 10, 0.84)],
        theta=0.95,
    )

    assert merged["b wording"] == "a wording"
    assert merged["c wording"] == "c wording"
    assert len(set(merged.values())) == 2


def test_the_merge_is_invariant_to_input_row_order() -> None:
    """Nothing records the groups, so reproducibility rests entirely on this.

    A merge that depended on the order labels arrive in would give a different
    fairness partition on a re-run of the same corpus, and neither the run nor
    the table would carry any evidence of it.
    """
    labelled = [
        ("wash the dishes", 40, 1.0),
        ("wash dishes", 9, 0.99),
        ("sand the tabletop", 25, 0.10),
        ("sand tabletop", 5, 0.12),
        ("stack the plates", 25, 0.60),
    ]
    shuffler = random.Random(20260824)  # noqa: S311 (deterministic fixture, not cryptography)
    baseline = _merge(labelled, theta=0.95)

    for _ in range(20):
        permuted = list(labelled)
        shuffler.shuffle(permuted)
        assert _merge(permuted, theta=0.95) == baseline


def test_an_extreme_collapse_is_reported_at_warning(loguru_records: list[dict[str, Any]]) -> None:
    """A merge that folds a vocabulary onto one group must say so.

    The grouping is never persisted, so this line is the only evidence that a
    threshold set too low redistributed the corpus. The counts and the theta are
    all in it, because the reader's next question is which knob to move.
    """
    _merge_groups([6])

    assert _warnings(loguru_records) == ["fairness merge collapsed task groups 6 -> 1 (theta=0.95)"]


def test_a_normal_merge_reports_nothing(loguru_records: list[dict[str, Any]]) -> None:
    """Folding a couple of wording variants is the intended outcome, not an anomaly.

    Stated because a warning that fires on every run carries no information: six
    labels down to five is the ordinary case this must stay silent for.
    """
    _merge_groups([2, 1, 1, 1, 1])

    assert _warnings(loguru_records) == []


def test_the_collapse_warning_names_the_task_level_it_can_only_come_from(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The merge is single-level, so the line must say which vocabulary folded.

    Two warnings can fire from the fairness module and they mean opposite things
    - this one that the TASK merge produced too few groups, the degenerate-quota
    one that level 2 produced too many. A message that named neither level would
    leave a reader guessing which knob to move.
    """
    _merge_groups([4])

    assert _warnings(loguru_records) == ["fairness merge collapsed task groups 4 -> 1 (theta=0.95)"]


def test_the_collapse_trigger_is_below_half_not_at_half(loguru_records: list[dict[str, Any]]) -> None:
    """Halving exactly is quiet; going past it warns.

    Pins the comparison itself. An inverted test, or a ``<=`` where the contract
    says ``<``, changes exactly one of these two assertions.
    """
    _merge_groups([2, 2, 2])
    assert _warnings(loguru_records) == []

    _merge_groups([3, 3])
    assert _warnings(loguru_records) == ["fairness merge collapsed task groups 6 -> 2 (theta=0.95)"]


def test_a_directionless_label_neither_joins_nor_absorbs() -> None:
    """A zero-norm embedding has no direction, so cosine cannot place it in any group."""
    labels = ["blank embedding", "wash the dishes"]
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    merged = merge_labels(labels, [500, 1], vectors, theta=0.0)

    assert merged == {"blank embedding": "blank embedding", "wash the dishes": "wash the dishes"}


def test_no_threshold_however_permissive_merges_a_directionless_label() -> None:
    """A directionless label draws its own quota share no matter how low theta is set.

    Theta sits at the bottom of the cosine range, so every label with a direction
    joins the leading representative. That leaves directionlessness as the only
    possible reason a label still stands alone, which is what makes this assertion
    a pin rather than a restatement of the threshold: refusing to merge on
    unmeasurable evidence must not depend on ``theta`` being high.
    """
    labels = ["wash the dishes", "wash up the dishes", "blank embedding"]
    vectors = np.array([_unit(1.0), _unit(0.5), [0.0, 0.0]], dtype=np.float32)

    merged = merge_labels(labels, [9, 5, 1], vectors, theta=-1.0)

    assert merged == {
        "wash the dishes": "wash the dishes",
        "wash up the dishes": "wash the dishes",
        "blank embedding": "blank embedding",
    }


def test_raw_vectors_are_renormalized_before_comparison() -> None:
    """Magnitude carries no label information, so a rescaled row must merge identically."""
    labels = ["first wording", "second wording"]
    unit = np.array([_unit(1.0), _unit(0.99)], dtype=np.float32)

    assert merge_labels(labels, [9, 2], unit, theta=0.95) == merge_labels(labels, [9, 2], unit * 17.0, theta=0.95)


def test_merge_treats_overflow_risk_magnitude_like_unit_scale() -> None:
    """A float32 norm past ~1.8e19 must not zero a row before merge.

    The expected mapping is written out rather than taken from the unit-scale
    call: comparing two live calls cannot tell "both correct" from "both
    degenerate", since a norm floor above every input silences each side alike.
    """
    labels = ["first wording", "second wording"]
    unit = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    huge = np.full((2, 4), 1e19, dtype=np.float32)
    merged = {"first wording": "first wording", "second wording": "first wording"}

    assert merge_labels(labels, [9, 2], unit, theta=0.95) == merged
    assert merge_labels(labels, [9, 2], huge, theta=0.95) == merged


def test_merge_treats_a_norm_past_the_float32_ceiling_like_unit_scale() -> None:
    """A row norm above the float32 ceiling must still merge on direction.

    Distinct from the ~1.8e19 case, which is where the SUM OF SQUARES overflows
    float32 and is held off by the float64 accumulator. Here the accumulated norm
    itself (4e38) exceeds what float32 can hold, so narrowing the DENOMINATOR is
    what turns it into inf. The input is finite float32, so the finiteness check
    passes it through, and the zeroed row still reports as directed - it merges
    with nothing and absorbs nothing, keeping a quota of its own.
    """
    labels = ["first wording", "second wording"]
    over_ceiling = np.full((2, 4), 2e38, dtype=np.float32)

    assert np.isfinite(over_ceiling).all()
    assert merge_labels(labels, [9, 2], over_ceiling, theta=0.95) == {
        "first wording": "first wording",
        "second wording": "first wording",
    }


def test_merge_rejects_a_repeated_label() -> None:
    """The merge runs over the DISTINCT label set; a repeat means the caller passed rows."""
    with pytest.raises(ValueError, match="must be distinct"):
        _merge([("same wording", 2, 1.0), ("same wording", 1, 0.99)], theta=0.95)


def test_merge_rejects_vectors_that_do_not_align_one_row_per_label() -> None:
    """A misaligned matrix would silently group labels by another label's direction."""
    with pytest.raises(ValueError, match=r"\(2, dim\) matrix"):
        merge_labels(["first", "second"], [2, 1], np.zeros((3, 4), dtype=np.float32), theta=0.95)


def test_merge_rejects_non_finite_label_vectors() -> None:
    """NaN or inf embeddings must fail before merge can silently isolate a label."""
    vectors = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="vectors must be finite"):
        merge_labels(["first", "second"], [2, 1], vectors, theta=0.95)


def test_merge_of_no_labels_is_empty() -> None:
    """An empty vocabulary is legal and merges to nothing."""
    assert merge_labels([], [], np.zeros((0, 4), dtype=np.float32), theta=0.95) == {}


def test_applying_the_merge_rewrites_the_task_column_in_place() -> None:
    """Downstream stages group on the merged key, so the rewrite must land on the column."""
    group = _group([("clip-0", 0.5, None)], task="wash dishes")

    merged = apply_label_merge(group, {"wash dishes": "wash the dishes"})

    assert merged.column(CANONICAL_TASK_COLUMN).to_pylist() == ["wash the dishes"]
    assert merged.num_rows == group.num_rows
    assert merged.column(KEY_COLUMN).to_pylist() == ["clip-0"]


def test_applying_the_merge_leaves_the_subtask_cluster_untouched() -> None:
    """The level-2 cell is assigned in the scan, so the merge must not be able to move it.

    Level 1 is the only merged key. A rewrite that also touched the cluster
    column would let a task threshold silently change the level-2 partition the
    centroid artifact fixed.
    """
    group = _group([("clip-0", 0.5, None)], task="wash dishes", cluster=9)

    merged = apply_label_merge(group, {"wash dishes": "wash the dishes"})

    assert merged.column(SUBTASK_CLUSTER_COLUMN).to_pylist() == [9]


def test_merging_two_tasks_merges_their_subtask_cluster_namespaces() -> None:
    """One semantic task holding one subtask cell becomes ONE level-2 group.

    A consequence of level-1 merging rather than of the level-2 partition: the
    two rows carry the same cluster cell under two task spellings, and after the
    task merge their group keys are equal.
    """
    variant_a = _group([("clip-0", 0.5, None)], task="wash the dishes", cluster=2)
    variant_b = _group([("clip-1", 0.5, None)], task="wash dishes", cluster=2)
    task_merge = {"wash the dishes": "wash the dishes", "wash dishes": "wash the dishes"}

    keys = {_group_key(apply_label_merge(group, task_merge)) for group in (variant_a, variant_b)}

    assert keys == {("wash the dishes", 2)}


def test_applying_the_merge_rejects_a_label_the_map_does_not_name() -> None:
    """A label with no representative would form a group the quota never funded."""
    group = _group([("clip-0", 0.5, None)], task="wash dishes")

    with pytest.raises(ValueError, match="absent from the label merge map"):
        apply_label_merge(group, {"other task": "other task"})


def test_applying_the_merge_rejects_a_batch_that_was_never_canonicalized() -> None:
    """The merge consumes canonical labels, so a raw batch is a wiring error, not a default."""
    batch = pa.table({TASK_COLUMN: pa.array(["Wash Dishes"])})

    with pytest.raises(ValueError, match=f"carries no {re.escape(CANONICAL_TASK_COLUMN)}"):
        apply_label_merge(batch, {})


def test_group_counts_exclude_rows_that_already_carry_a_reason() -> None:
    """Duplicates and invalid rows neither consume a quota nor enlarge the group funding one."""
    counts = _count_rows(
        [
            ("wash", 0, None, 4),
            ("wash", 0, CurateReason.DUPLICATE, 11),
            ("wash", 0, CurateReason.INVALID_EMBEDDING, 2),
        ]
    )

    assert survivor_group_counts(counts) == {("wash", 0): 4}


def test_group_counts_report_the_observed_group_count() -> None:
    """G is the number of groups holding a survivor, which is the map's own length."""
    counts = _count_rows(
        [
            ("wash", 0, None, 4),
            ("wash", 2, None, 1),
            ("sand", 1, None, 9),
            ("sand", 3, CurateReason.DUPLICATE, 6),
        ]
    )

    assert len(survivor_group_counts(counts)) == 3


def test_group_counts_of_an_all_duplicate_corpus_are_empty() -> None:
    """With no survivor there is no population to allocate, and no group to fund."""
    counts = _count_rows([("wash", 0, CurateReason.DUPLICATE, 12)])

    assert survivor_group_counts(counts) == {}


def test_quotas_sum_to_min_of_target_and_total() -> None:
    """Property 1: the allocation sums to min(target, total survivors)."""
    counts: Counts = {("a", 0): 30, ("a", 1): 20, ("b", 2): 10}
    assert sum(_quotas(counts, target=25).values()) == 25
    assert sum(_quotas(counts, target=999).values()) == 60  # clamped to total


def test_quota_never_exceeds_capacity_and_is_never_negative() -> None:
    """Property 2: 0 <= quota[g] <= capacity[g] for every group."""
    counts: Counts = {("a", 0): 5, ("a", 1): 1, ("b", 2): 3}
    quotas = _quotas(counts, target=7)
    for key, quota in quotas.items():
        assert 0 <= quota <= counts[key]


def test_every_group_is_covered_before_any_is_doubled() -> None:
    """Property 3, within one tier: at target == group count every group gets exactly one.

    One task, so the level-2 pass alone allocates and no level-1 remainder can
    reach the result. Across tasks the property is weaker; the next test pins how.
    """
    counts: Counts = {("t", 0): 9, ("t", 1): 9, ("t", 2): 9}
    assert _quotas(counts, target=3) == {("t", 0): 1, ("t", 1): 1, ("t", 2): 1}


def test_a_single_cell_task_can_spend_its_parent_share_twice() -> None:
    """Property 3 does not survive nesting: a target equal to G can fund fewer than G.

    Level 1 equalizes TASKS, so a task holding one cell hands that cell its whole
    parent share. Over two tasks at target 3 the level-1 fill line is one with a
    row left over; when that row goes to the single-cell task, the cell draws two
    and a sibling task's cell draws none. Which way it falls is the seed's, so
    this is a property of the nesting rather than of the comparator - the sweep
    misses the two-group outcome only if all twelve seeds agree, about one run in
    four thousand.
    """
    counts: Counts = {("a", 0): 9, ("a", 1): 9, ("b", 2): 9}

    funded = {len(_funded(counts, 3, residual_seed=seed)) for seed in range(12)}

    assert funded == {2, 3}


def test_unsaturated_quotas_differ_by_at_most_one() -> None:
    """Property 4: the residual is uniform, not proportional to size."""
    counts: Counts = {("t", 0): 5, ("t", 1): 5, ("t", 2): 5}
    assert sorted(_quotas(counts, target=7).values()) == [2, 2, 3]  # uniform +/- 1, never 5:1:1


def test_the_level2_remainder_lands_on_one_cell_chosen_by_the_seed() -> None:
    """Property 5, at level 2: the leftover row is placed by the residual order, uniformly.

    One task, so this is the level-2 pass alone. Three equal capacities leave
    exactly one leftover row, and the cell it lands on moves with the seed - which
    is possible only if the comparator is consulted at all. Under the ascending
    cell id this tier used to fund by, cell 0 would win at every seed.
    """
    counts: Counts = {("t", 0): 5, ("t", 1): 5, ("t", 2): 5}

    # At target 1 the fill line is zero, so the residual IS the whole allocation
    # and the funded set names the winner directly.
    winners = {_funded(counts, 1, residual_seed=seed) for seed in range(8)}

    assert sorted(_quotas(counts, target=7).values()) == [2, 2, 3]
    assert len(winners) > 1


def test_the_level2_digest_key_is_the_published_pair_encoding() -> None:
    """The pair encoding is result-defining, so it is pinned by name and not only by outcome.

    Changing it redraws every scarce-budget selection while every invariant still
    holds, so no property test can call it wrong. The cell leads so that the
    free-form label is the unambiguous tail: a label holding the delimiter itself
    still decodes to one pair, because ``str(int)`` never can.
    """
    assert _pair_digest_keys([("wipe table", 3), ("a|b", NO_SUBTASK_CLUSTER)]) == ["3|wipe table", "-1|a|b"]


def test_all_quota_values_are_python_ints() -> None:
    """Property 6: no float leaks into the result; every value is an int."""
    quotas = _quotas({("a", 0): 7, ("b", 1): 3}, target=4)
    assert all(type(value) is int for value in quotas.values())


def test_swapping_two_capacities_does_not_change_the_funded_set() -> None:
    """Capacity has left the comparator: which cell is funded is independent of its size.

    Moving the 100 rows to the other cell must not move the place, so in exactly
    one of these two arrangements the single-row cell is the funded one.
    """
    funded_small_first = _funded({("t", 0): 1, ("t", 1): 100}, target=1)
    funded_large_first = _funded({("t", 0): 100, ("t", 1): 1}, target=1)

    assert funded_small_first == funded_large_first
    assert len(funded_small_first) == 1


def test_rescaling_all_capacities_does_not_change_the_funded_set() -> None:
    """Funding never consults size, so a uniform capacity scale is invisible to it."""
    base = _quotas({("t", 0): 2, ("t", 1): 5}, target=1)
    scaled = _quotas({("t", 0): 200, ("t", 1): 500}, target=1)
    assert {k for k, v in base.items() if v > 0} == {k for k, v in scaled.items() if v > 0}


def test_the_reserved_cell_gets_no_privilege_from_its_negative_id() -> None:
    """A negative cell id is not a priority: the reserved cell competes like any other.

    ``NO_SUBTASK_CLUSTER`` sorts ahead of every fitted cell in KEY order, so an
    allocator handing its remainder out positionally would give the rows with no
    usable subtask direction their task's first place at every seed. Production
    never seats a survivor there beside a fitted cell - ``test_vectors`` pins the
    single predicate that guarantees it - so what is asserted here is the
    comparator's indifference, not a reachable corpus.
    """
    cells: Counts = {("t", NO_SUBTASK_CLUSTER): 10, ("t", 0): 10, ("t", 1): 10}

    winners = {_funded(cells, 1, residual_seed=seed) for seed in range(8)}

    assert winners != {frozenset({("t", NO_SUBTASK_CLUSTER)})}


# Verb-initial task labels, which is the shape this project's own annotations take
# ("Make coffee", "Pick up capsule" in the embeddings design doc). The family is the
# leading word, so an allocator that funds an alphabetical prefix funds whole
# families and starves whole families. One cell per task, so the level-1 pass
# decides everything and level 2 is a passthrough.
#
# Four families of six rather than a smaller grid, because the assertion below is
# that a funded set of half the tasks is not the alphabetical half. Two random
# halves of 12 coincide once in 924 tries - often enough that seed 0 actually hit
# it - while two halves of 24 coincide once in 2.7 million.
_VERB_FAMILY_TARGET = 12
_VERB_FAMILY_GROUPS: Counts = {
    (f"{verb} object {index}", 0): 10 for verb in ("close", "open", "pour", "wipe") for index in range(6)
}


def _funded_tasks(residual_seed: int) -> frozenset[str]:
    """Return the task labels drawing a non-zero quota at ``residual_seed``."""
    funded = _funded(_VERB_FAMILY_GROUPS, _VERB_FAMILY_TARGET, residual_seed=residual_seed)
    return frozenset(task for task, _cell in funded)


def test_a_target_below_the_task_count_does_not_fund_an_alphabetical_prefix() -> None:
    """The funded tasks are not the alphabetically first ones, so representation does not track spelling.

    At ``target`` below the task count the fill line is zero and the residual IS
    the whole allocation, so the order decides which tasks appear in the output at
    all. Under key-ascending order the funded set is exactly the leading prefix of
    the sorted labels, which for verb-initial prose means leading ACTION families:
    every "wipe ..." task is starved together, on every run.

    Asserted as "not the prefix" rather than as family coverage, because coverage
    is NOT what the digest provides. It decorrelates the loss from spelling; it
    does not bound it. Every family does happen to place here, but a small enough
    family still draws nothing by luck, and asserting otherwise would pin this
    test to one hash output. Guaranteeing each family a place needs a
    coverage-first allocator, not a different hash.
    """
    quotas = _quotas(_VERB_FAMILY_GROUPS, target=_VERB_FAMILY_TARGET)

    funded = {task for (task, _cell), quota in quotas.items() if quota > 0}
    alphabetical_prefix = set(sorted({task for task, _cell in _VERB_FAMILY_GROUPS})[:_VERB_FAMILY_TARGET])
    assert len(funded) == _VERB_FAMILY_TARGET
    assert funded != alphabetical_prefix


def test_within_one_pass_the_unfunded_count_does_not_depend_on_the_residual_seed() -> None:
    """Within a single pass the seed moves WHICH tasks are funded and never HOW MANY.

    The fill line is a function of the capacity multiset and the target alone, so
    every seed starves the same number of groups. That invariance is also why the
    unfunded count cannot be the signal that detects a spelling-correlated
    allocation - it is identical in the good case and the bad one.

    One cell per task, so this fixture IS a single pass. The invariance does not
    survive nesting; the next test pins how far it moves.
    """
    funded_per_seed = [len(_funded_tasks(seed)) for seed in range(5)]

    assert funded_per_seed == [_VERB_FAMILY_TARGET] * 5


# Two tasks occupying UNEQUAL cell counts, which is exactly what the one-cell-per-task
# fixture above cannot express. Level-1 capacities are 50 and 20, so at target 9 the
# level-1 fill line is 4 with one row left over: whichever parent wins it draws 5.
# A on 5 cells needs 5 to cover them, B on 2 needs only 2 - so the leftover row
# decides whether a cell starves anywhere in the corpus.
_UNEQUAL_CELL_GROUPS: Counts = {("A", cell): 10 for cell in range(5)} | {("B", cell): 10 for cell in range(2)}
_UNEQUAL_CELL_TARGET = 9
_UNEQUAL_CELL_LEVEL1_REMAINDER = 1


def test_nesting_lets_the_seed_move_the_unfunded_count_within_the_level1_remainder() -> None:
    """Across tiers the unfunded count is seed-dependent, bounded by the level-1 remainder.

    The single-pass invariance above is what the degeneracy threshold rests on, so
    the residue nesting leaves has to be pinned rather than assumed. A level-1
    reseed that moves the leftover row between parents of different cell counts
    moves the level-2 total, because a parent starves ``occupied cells - budget``
    of them. The count is still a scarcity reading: it cannot drift further than
    the number of rows the level-1 residual has to place.

    The bound is exercised at a remainder of ONE, so what this pins is that the
    drift exists and respects the remainder - NOT that it is small. On a corpus
    mixing large and small tasks the remainder runs to hundreds and the reported
    count moves by tens of groups.
    """
    total_groups = len(_UNEQUAL_CELL_GROUPS)
    unfunded_per_seed = {
        total_groups - len(_funded(_UNEQUAL_CELL_GROUPS, _UNEQUAL_CELL_TARGET, residual_seed=seed)) for seed in range(8)
    }

    assert max(unfunded_per_seed) - min(unfunded_per_seed) <= _UNEQUAL_CELL_LEVEL1_REMAINDER
    # Both outcomes, not merely two of them. A one-cell-per-task fixture reports
    # zero at every seed and cannot reach either half of this assertion.
    assert unfunded_per_seed == {0, 1}


def test_the_same_seed_funds_the_same_tasks_twice() -> None:
    """Reseeding is the only way to move the funded set: at a fixed seed the allocation is reproducible."""
    first = _quotas(_VERB_FAMILY_GROUPS, target=_VERB_FAMILY_TARGET)
    second = _quotas(_VERB_FAMILY_GROUPS, target=_VERB_FAMILY_TARGET)

    assert first == second


def test_the_residual_order_is_the_same_in_every_process(run_child: RunChild) -> None:
    """Two interpreters reading one config must fund the same tasks.

    ``hash()`` is salted per interpreter, so a rank built on it would order these
    labels differently under each ``PYTHONHASHSEED`` - while every same-process
    test in this module still passed, because a salted hash is self-consistent
    within one run. Two hostile salts are the only way to hold that apart, and
    the salt is fixed at interpreter startup, so this cannot be done in-process.
    """
    labels = [f"task {index}" for index in range(32)]
    program = textwrap.dedent(
        f"""
        import json
        from cosmos_curator.next.recipes.curation.fairness import residual_ranks
        print(json.dumps(residual_ranks({labels!r}, 7).tolist()))
        """
    )

    children = [run_child(program, env={"PYTHONHASHSEED": salt}) for salt in ("1", "2")]

    assert [child.returncode for child in children] == [0, 0], children[0].stderr
    assert [json.loads(child.stdout) for child in children] == [residual_ranks(labels, 7).tolist()] * 2


def test_a_different_seed_funds_a_different_task_set() -> None:
    """The seed is result-defining, so it must be able to change the result.

    Pinned because a seed threaded through but never reaching the comparator
    would leave every other test here passing: the coverage test above would
    still hold under key order for some fixtures, and reproducibility holds
    trivially for a constant.
    """
    funded_by_seed = {_funded_tasks(seed) for seed in range(8)}

    assert len(funded_by_seed) > 1


# Every task populates every cell, and the per-task budget covers a quarter of
# them - the regime where the level-2 residual decides three cells out of four.
#
# Cell ids are GLOBAL, so ascending cell id is the same order under every parent:
# the funded cells would be the same low ids in all forty tasks, and the rest of
# the partition would reach the selected set with no clips at all. The task count
# is what makes that visible; a single-task fixture cannot express a CROSS-TASK
# effect, and neither can the unfunded count, which is equal under both orders.
_EVERY_TASK_EVERY_CELL: Counts = {(f"task{task:02d}", cell): 8 for task in range(40) for cell in range(16)}
_EVERY_CELL_PER_TASK_BUDGET = 4


def test_no_cell_is_starved_in_every_task_at_once() -> None:
    """Under a scarce level-2 budget the loss is spread across cells, not fixed on the same ones.

    The residual is hashed over the ``(task, cell)`` PAIR rather than the cell, so
    a cell that loses its place under one task can win under another and every
    region of subtask meaning still reaches the output. Hashing the cell alone
    would rank it identically everywhere and reproduce the defect this replaced.

    Coverage is the measurable consequence, not a guarantee the digest offers: it
    holds here because forty independent draws of four cells from sixteen leave a
    given cell unfunded everywhere with probability 0.75 ** 40, about 1 in 10^5.
    Asserting it on a fixture this size is a statement about the ORDER, not luck.
    """
    tasks = len({task for task, _cell in _EVERY_TASK_EVERY_CELL})
    target = tasks * _EVERY_CELL_PER_TASK_BUDGET

    funded = _funded(_EVERY_TASK_EVERY_CELL, target)

    assert {cell for _task, cell in funded} == set(range(16))
    assert len(funded) == target  # every funded group holds exactly one clip at this budget


def test_nested_worked_example_equalizes_parents_despite_row_ratio() -> None:
    """Two parents receive equal totals despite a 10:1 row ratio."""
    counts: Counts = {
        ("A", 0): 600,
        ("A", 1): 300,
        ("A", 2): 100,
        ("B", 3): 60,
        ("B", 4): 40,
    }
    quotas = _quotas(counts, target=200)
    # A's three cells split 100 as evenly as integers allow; WHICH of them takes
    # the odd row is the residual order's call and is asserted elsewhere. B is
    # saturated, so both its cells return their full capacity.
    assert sorted(v for k, v in quotas.items() if k[0] == "A") == [33, 33, 34]
    assert {k: v for k, v in quotas.items() if k[0] == "B"} == {("B", 3): 60, ("B", 4): 40}
    assert sum(v for k, v in quotas.items() if k[0] == "A") == 100
    assert sum(v for k, v in quotas.items() if k[0] == "B") == 100


def test_level2_quotas_sum_to_their_parent_level1_quota() -> None:
    """Each parent's children sum to the parent's level-1 quota by construction."""
    quotas = _quotas({("A", 0): 600, ("A", 1): 300, ("B", 2): 90}, target=120)
    parent_a = quotas[("A", 0)] + quotas[("A", 1)]
    parent_b = quotas[("B", 2)]
    assert parent_a + parent_b == 120
    assert parent_a == parent_b


def test_strongly_unequal_parents_receive_equal_totals_up_to_capacity() -> None:
    """A 100-child 10M parent and a 2-child 10k parent split the target equally."""
    counts: Counts = {("big", i): 100_000 for i in range(100)}
    counts.update({("small", 0): 5_000, ("small", 1): 5_000})
    quotas = _quotas(counts, target=4_000)
    assert sum(v for k, v in quotas.items() if k[0] == "big") == 2_000
    assert sum(v for k, v in quotas.items() if k[0] == "small") == 2_000


def test_same_child_key_under_two_parents_is_two_independent_groups() -> None:
    """Hierarchy separation: one subtask cluster cell under two tasks gets two budgets."""
    quotas = _quotas({("A", 0): 10, ("B", 0): 10}, target=4)
    assert quotas[("A", 0)] == 2
    assert quotas[("B", 0)] == 2


def test_an_underfunded_group_hands_its_unused_share_to_its_peers() -> None:
    """The redistribution branch: a group smaller than its uniform share releases the rest.

    Three groups and a target of 12 give a uniform share of 4, which the first
    group cannot fill - it holds one row. Its three unused places must go to the
    other two, so together they draw 11 rather than 8. This is the branch a
    uniform fixture population can never reach.
    """
    quotas = _quotas({("t", 0): 1, ("t", 1): 10, ("t", 2): 10}, target=12)

    assert quotas[("t", 0)] == 1  # capped at capacity, so its share is not the uniform 4
    assert sorted((quotas[("t", 1)], quotas[("t", 2)])) == [5, 6]
    assert quotas[("t", 1)] + quotas[("t", 2)] == 11
    # Both peers sit strictly above the uniform share they would draw with no
    # redistribution, whichever of them took the odd row.
    assert min(quotas[("t", 1)], quotas[("t", 2)]) > 12 // 3


def test_a_uniform_population_leaves_the_uniform_share_untouched() -> None:
    """The contrast case: with equal capacities nothing is released and nothing redistributes.

    Stated because it is why the fixtures above are uneven: this input exercises
    the fill line alone, so a suite built only on it would pass with the
    redistribution step deleted.
    """
    assert _quotas({("t", 0): 10, ("t", 1): 10, ("t", 2): 10}, target=12) == {
        ("t", 0): 4,
        ("t", 1): 4,
        ("t", 2): 4,
    }


def test_a_capacity_bound_parent_releases_its_share_to_the_other_parent() -> None:
    """Redistribution at level 1: a one-row task cannot hold half the target."""
    quotas = _quotas({("small", 1): 1, ("big", 0): 20}, target=12)

    assert quotas == {("small", 1): 1, ("big", 0): 11}


def test_matches_independent_reference_over_randomized_nested_inputs() -> None:
    """Randomized nested cases agree with the naive round-robin reference."""
    rng = random.Random(19872704)  # noqa: S311 (deterministic test fixture, not cryptography)
    for _ in range(400):
        counts: Counts = {}
        for parent_index in range(rng.randint(1, 5)):
            for child_index in range(rng.randint(1, 6)):
                counts[(f"p{parent_index}", child_index)] = rng.randint(1, 50)
        target = rng.randint(0, sum(counts.values()) + 5)
        assert _quotas(counts, target) == _reference_nested(counts, target)


def test_target_zero_funds_nothing() -> None:
    """A resolved target of zero is legal and selects no row anywhere."""
    assert _quotas({("a", 0): 5, ("b", 1): 5}, target=0) == {("a", 0): 0, ("b", 1): 0}


def test_target_at_or_above_total_returns_every_capacity() -> None:
    """A target that covers the population returns each group's full capacity."""
    counts: Counts = {("a", 0): 5, ("a", 1): 3, ("b", 2): 2}
    assert _quotas(counts, target=10) == counts


def test_single_group_holding_everything() -> None:
    """One pair receives min(target, capacity)."""
    assert _quotas({("only", 0): 100}, target=30) == {("only", 0): 30}


def test_every_group_holds_exactly_one_row() -> None:
    """When every group has one row, a scarce target funds exactly ``target`` of them."""
    quotas = _quotas({("t", i): 1 for i in range(5)}, target=3)
    assert sum(quotas.values()) == 3
    assert sorted(quotas.values()) == [0, 0, 1, 1, 1]


def test_large_group_count_agrees_with_reference_and_is_never_rejected() -> None:
    """A group count far larger than the usual fixtures is served with no limit."""
    counts: Counts = {("t", i): 1 for i in range(2_000)}
    quotas = _quotas(counts, 1_000)
    assert sum(quotas.values()) == 1_000
    assert quotas == _reference_nested(counts, 1_000)


# Four level-2 groups under ONE task, each holding exactly two survivors. Used
# for the regimes the reporting boundary is not the subject of: every group
# funded, and a target of zero.
#
# A single task cannot exhibit a CROSS-TASK effect, which is why the tail fixture
# below exists rather than more cases over this one.
_TWO_DEEP_GROUPS: Counts = {("t", 0): 2, ("t", 1): 2, ("t", 2): 2, ("t", 3): 2}

# Twenty-five level-2 cells under ONE task, two survivors each. The cell count is
# what makes the BOUNDARY expressible: with one task the level-1 quota is the
# whole target, so a target below the cell count funds exactly ``target`` cells
# and leaves the rest at zero. Twenty-five cells therefore move the unfunded
# share in 4% steps, which is fine enough to sit one row either side of a fifth -
# a four-cell fixture can only step in 25% and cannot express the boundary at all.
_TWENTY_FIVE_CELLS: Counts = {("t", cell): 2 for cell in range(25)}

# Two six-cell tasks plus one task whose two rows occupy a single cell - the
# shape a long-tailed corpus produces, because a task with fewer rows than
# subtask_clusters cannot populate them all.
#
# Load-bearing property, and the reason this fixture is MULTI-task: at target 6
# every task draws a level-1 quota of 2, which the six-cell tasks split into
# ones (funding 2 of 6 cells each) while the single-cell task absorbs both rows
# into its one cell. The largest level-2 quota in the corpus is therefore 2 even
# though 8 of the 13 groups receive nothing - so any rule reading the MAXIMUM
# quota is silent here, and one task out of three silences it. A single-task
# fixture cannot express that at all.
_SMALL_TASK_TAIL_GROUPS: Counts = {
    **{(f"big{task}", cell): 3 for task in range(2) for cell in range(6)},
    ("tiny", 0): 2,
}


def test_an_allocation_that_leaves_most_level2_groups_unfunded_is_reported(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A corpus allocated by the residual seed says so even when one task is funded generously.

    The counts are in the line because the reader's next question is how far
    ``subtask_clusters`` overshot: the group count is a run's own choice, and
    nothing about the group set is persisted to reconstruct it from afterwards.

    Every task is funded here, so the level-1 line must stay silent: this fixture
    is the pure level-2 cause, and a message naming the target would be pointing
    at a knob that is already correct.
    """
    quotas = _quotas(_SMALL_TASK_TAIL_GROUPS, target=6)

    assert max(quotas.values()) == 2  # the tail task's cell, which no maximum can see past
    assert _warnings(loguru_records) == [
        "fairness quota is degenerate at level 2: 8 of 13 group(s) received nothing, so which cells are "
        "funded is decided by fairness_residual_seed and not by the corpus; lower subtask_clusters or "
        "raise the target"
    ]


def test_a_target_below_the_task_count_is_reported_as_a_level1_shortfall(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A starved TASK names the target and the merge, not subtask_clusters.

    The two starvation causes have different remedies, and a single message
    covering both sent a reader to the wrong knob: these tasks already hold one
    cell each, so lowering ``subtask_clusters`` leaves every one of them starved.
    """
    _quotas(_VERB_FAMILY_GROUPS, target=_VERB_FAMILY_TARGET)  # 24 single-cell tasks, so 12 get nothing

    assert _warnings(loguru_records) == [
        "fairness target 12 is below the 24 task group(s), so 12 task(s) draw no selected clips and every "
        "survivor in them lands as unfunded; raise the target above the task count, or LOWER "
        "merge_theta_task to fold the task vocabulary harder. Lowering subtask_clusters cannot help, "
        "because the shortfall is at the task level, so the level-2 line is suppressed until every task "
        "is funded"
    ]


def test_a_starved_task_suppresses_the_level2_line(
    loguru_records: list[dict[str, Any]],
) -> None:
    """One cause is named per run, so the two remedies never contradict each other.

    Ten of these twelve groups hold nothing, far past the level-2 threshold, but
    one whole task is starved - and at a target below the task count every funded
    task holds a quota of exactly one, which makes the level-2 share a function of
    ``subtask_clusters`` rather than of the corpus. Reporting it would tell the
    reader to lower the very knob the level-1 line says cannot help.
    """
    groups = {(f"task{task}", cell): 10 for task in range(3) for cell in range(4)}

    quotas = _quotas(groups, target=2)  # 3 tasks, so one draws nothing

    assert sum(1 for quota in quotas.values() if quota == 0) == 10  # 83%, past a fifth
    assert [message.split(",")[0] for message in _warnings(loguru_records)] == [
        "fairness target 2 is below the 3 task group(s)"
    ]


def test_an_allocation_that_funds_every_group_reports_nothing(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A target equal to the group count is the fairest allocation there is, not a collapsed one.

    Every group receives exactly one clip, so nothing at all was decided by the
    residual order - the regime the warning names is precisely absent. Stated
    because a line that fires on the ideal case carries no information.
    """
    quotas = _quotas(_TWO_DEEP_GROUPS, target=4)

    assert set(quotas.values()) == {1}
    assert _warnings(loguru_records) == []


def test_exactly_a_fifth_of_the_groups_unfunded_is_quiet(loguru_records: list[dict[str, Any]]) -> None:
    """The boundary is inclusive on the quiet side, so a fifth exactly does not warn.

    The companion to the test below, and the pair pins the comparison itself: one
    row of the target separates them, so a ``<`` where the contract says ``<=``
    flips exactly one of the two.
    """
    _quotas(_TWENTY_FIVE_CELLS, target=20)  # 5 of 25 unfunded, exactly a fifth

    assert _warnings(loguru_records) == []


def test_a_quarter_of_the_groups_unfunded_is_reported(loguru_records: list[dict[str, Any]]) -> None:
    """Just past a fifth warns, which is the regime a majority rule could not see.

    Six of twenty-five is 24% - comfortably under half, so a rule that fired only
    once MOST groups went unfunded stayed silent here while three quarters of the
    partition was still being allocated by the residual seed.
    """
    _quotas(_TWENTY_FIVE_CELLS, target=19)  # 6 of 25 unfunded, just past a fifth

    assert _warnings(loguru_records) == [
        "fairness quota is degenerate at level 2: 6 of 25 group(s) received nothing, so which cells are "
        "funded is decided by fairness_residual_seed and not by the corpus; lower subtask_clusters or "
        "raise the target"
    ]


def test_a_target_of_zero_is_not_reported_as_a_degenerate_allocation(
    loguru_records: list[dict[str, Any]],
) -> None:
    """Selecting nothing is what the operator asked for, not a collapsed allocation.

    Every group is unfunded at target zero, so the share alone would report the
    worst possible collapse; the target guard is what keeps that from firing.
    """
    _quotas(_TWO_DEEP_GROUPS, target=0)

    assert _warnings(loguru_records) == []


def test_a_corpus_with_no_surviving_group_is_not_reported_as_degenerate(
    loguru_records: list[dict[str, Any]],
) -> None:
    """A run whose every row was already reasoned has no groups to report on.

    Reachable rather than theoretical: ``survivor_group_counts`` returns an empty
    map when de-duplication and the invalid-embedding check between them claimed
    every row, and "0 of 0 groups received nothing" would be the one message that
    names no decision at all.
    """
    assert _quotas({}, target=4) == {}

    assert _warnings(loguru_records) == []


def test_the_degenerate_allocation_warning_is_distinct_from_the_merge_collapse_warning(
    loguru_records: list[dict[str, Any]],
) -> None:
    """The two warnings report opposite vocabulary failures, so neither can cover the other.

    A merge that folds nothing away emits no collapse line, and the allocation
    over its unfolded vocabulary is exactly what turns out to be degenerate. One
    vocabulary therefore reaches one warning and not the other.
    """
    merged = _merge_groups([1, 1, 1, 1])
    assert len(set(merged.values())) == 4  # nothing collapsed
    _quotas({("t", label): 2 for label in merged}, target=1)

    assert [message.split(":")[0] for message in _warnings(loguru_records)] == [
        "fairness quota is degenerate at level 2"
    ]


def test_the_unfunded_count_is_returned_and_not_only_logged() -> None:
    """The count the warning is decided on is the count the caller receives.

    One owner for the zero-quota predicate: the driver reports the same number in
    its INFO line and in ``CurateResult``, so a second derivation of it could
    disagree with the threshold that fired here.
    """
    keys = list(_TWENTY_FIVE_CELLS)
    quota = FairnessQuota.build(
        level2_keys=keys,
        level2_counts=[_TWENTY_FIVE_CELLS[key] for key in keys],
        target=19,
    )
    quotas = quota.quotas()

    assert quota.unfunded_groups(quotas) == sum(1 for allotted in quotas.values() if allotted == 0) == 6


def test_the_unfunded_count_is_returned_below_the_warning_threshold() -> None:
    """A quiet allocation still reports its count, because the INFO line is unconditional."""
    keys = list(_TWENTY_FIVE_CELLS)
    quota = FairnessQuota.build(
        level2_keys=keys,
        level2_counts=[_TWENTY_FIVE_CELLS[key] for key in keys],
        target=20,
    )

    assert quota.unfunded_groups(quota.quotas()) == 5


def test_build_rejects_duplicate_pair_keys() -> None:
    """Observed pairs come from a group count, so a duplicate key is a caller defect."""
    with pytest.raises(ValueError, match="distinct observed pairs"):
        FairnessQuota.build(level2_keys=[("a", 0), ("a", 0)], level2_counts=[1, 2], target=1)


def test_build_rejects_non_positive_counts() -> None:
    """Already-reasoned rows are excluded upstream, so a zero or negative count is corruption."""
    with pytest.raises(ValueError, match="strictly positive"):
        FairnessQuota.build(level2_keys=[("a", 0)], level2_counts=[0], target=1)


def test_build_rejects_negative_target() -> None:
    """A negative target has no meaning; the resolved target is never below zero."""
    with pytest.raises(ValueError, match="target must be >= 0"):
        FairnessQuota.build(level2_keys=[("a", 0)], level2_counts=[5], target=-1)


def test_build_rejects_length_mismatch() -> None:
    """Keys and counts are parallel arrays and must have equal length."""
    with pytest.raises(ValueError, match="equal length"):
        FairnessQuota.build(level2_keys=[("a", 0), ("a", 1)], level2_counts=[1], target=1)


def test_direct_construction_with_unsorted_keys_is_rejected() -> None:
    """quotas() trusts ascending keys, so a hand-built unsorted instance fails loudly."""
    with pytest.raises(ValueError, match="strictly ascending"):
        FairnessQuota(
            level1_keys=("a",),
            level1_capacity=np.array([3], dtype=np.int64),
            level2_keys=(("a", 1), ("a", 0)),
            level2_capacity=np.array([1, 2], dtype=np.int64),
            target=1,
        )


def test_direct_construction_rejects_mismatched_level1_capacity() -> None:
    """level1_capacity must equal the sum of each parent's contiguous level-2 slice."""
    with pytest.raises(ValueError, match="level1_capacity must equal"):
        FairnessQuota(
            level1_keys=("a",),
            level1_capacity=np.array([99], dtype=np.int64),
            level2_keys=(("a", 0), ("a", 1)),
            level2_capacity=np.array([1, 2], dtype=np.int64),
            target=1,
        )


def test_direct_construction_rejects_non_positive_level2_capacity() -> None:
    """A zero child capacity would make quotas() divide by an empty slice."""
    with pytest.raises(ValueError, match="level2_capacity must be strictly positive"):
        FairnessQuota(
            level1_keys=("a",),
            level1_capacity=np.array([1], dtype=np.int64),
            level2_keys=(("a", 0),),
            level2_capacity=np.array([0], dtype=np.int64),
            target=1,
        )


def test_quota_ignores_input_iteration_order() -> None:
    """Keys are normalized to ascending order on build, so input order cannot reach the result."""
    counts: Counts = {("b", 2): 4, ("a", 1): 3, ("a", 0): 6}
    forward = _quotas(counts, target=8)
    reversed_input = _quotas(dict(reversed(list(counts.items()))), target=8)
    assert forward == reversed_input


def test_the_cut_selects_the_farthest_rows_and_reasons_the_rest_below_quota() -> None:
    """A funded group keeps its quota by ordering and gives the losers below_quota."""
    group = _group([("clip-a", 0.1, None), ("clip-b", 0.9, None), ("clip-c", 0.5, None)])

    assert _reasons(group, quota=2) == {
        "clip-b": CurateReason.SELECTED,
        "clip-c": CurateReason.SELECTED,
        "clip-a": CurateReason.BELOW_QUOTA,
    }


def test_nearest_ordering_inverts_which_rows_win() -> None:
    """The ordering mode is result-defining, so the same group yields the opposite set."""
    group = _group([("clip-a", 0.1, None), ("clip-b", 0.9, None), ("clip-c", 0.5, None)])

    reasons = _reasons(group, quota=1, order="nearest")

    assert reasons["clip-a"] == CurateReason.SELECTED
    assert reasons["clip-b"] == CurateReason.BELOW_QUOTA


def test_neutral_ordering_ignores_distance_and_ranks_by_clip_id() -> None:
    """With no distance preference the cut is by identity alone, and still total."""
    group = _group([("clip-z", 0.99, None), ("clip-a", 0.01, None)])

    reasons = _reasons(group, quota=1, order="neutral")

    assert reasons["clip-a"] == CurateReason.SELECTED
    assert reasons["clip-z"] == CurateReason.BELOW_QUOTA


def test_clip_id_breaks_a_distance_tie() -> None:
    """Equal distances must not leave the winner to read order, which is not reproducible."""
    group = _group([("clip-z", 0.5, None), ("clip-a", 0.5, None)])

    reasons = _reasons(group, quota=1)

    assert reasons["clip-a"] == CurateReason.SELECTED
    assert reasons["clip-z"] == CurateReason.BELOW_QUOTA


def test_an_unfunded_group_reasons_every_survivor_unfunded() -> None:
    """A zero quota means the group was never funded, which is a different verdict from losing."""
    group = _group([("clip-a", 0.1, None), ("clip-b", 0.9, None)])

    assert set(_reasons(group, quota=0).values()) == {CurateReason.UNFUNDED}


def test_an_already_reasoned_row_passes_through_and_never_consumes_a_place() -> None:
    """A duplicate ranks first by distance yet must not take a survivor's place.

    The row keeps its own verdict and is removed from the ranking, so the quota
    is spent entirely on rows that could still be selected.
    """
    group = _group(
        [("clip-dup", 0.99, CurateReason.DUPLICATE), ("clip-a", 0.5, None), ("clip-b", 0.1, None)],
    )

    assert _reasons(group, quota=1) == {
        "clip-dup": CurateReason.DUPLICATE,
        "clip-a": CurateReason.SELECTED,
        "clip-b": CurateReason.BELOW_QUOTA,
    }


def test_an_invalid_embedding_row_keeps_its_verdict_through_the_cut() -> None:
    """The scan's verdict is final; fairness only fills the rows that have none."""
    group = _group([("clip-bad", 0.0, CurateReason.INVALID_EMBEDDING), ("clip-a", 0.5, None)])

    assert _reasons(group, quota=1)["clip-bad"] == CurateReason.INVALID_EMBEDDING


def test_a_group_holding_no_survivor_passes_through_without_a_quota() -> None:
    """An all-duplicate group is never counted, so it must not demand a quota entry."""
    group = _group([("clip-a", 0.5, CurateReason.DUPLICATE), ("clip-b", 0.1, CurateReason.DUPLICATE)])

    verdicts = select_within_quota(group, {}, "farthest")

    assert verdicts.column(CURATE_SELECTION_REASON).to_pylist() == [CurateReason.DUPLICATE] * 2


def test_a_group_with_survivors_but_no_quota_entry_raises() -> None:
    """The count and the cut must agree on the population, or the allocation is meaningless."""
    group = _group([("clip-a", 0.5, None)])

    with pytest.raises(ValueError, match="never counted"):
        select_within_quota(group, {}, "farthest")


def test_a_group_naming_two_subtask_cells_is_rejected() -> None:
    """Two cells in one group would be cut against one cell's quota.

    Grouping on both key columns already guarantees one key, and the check stays
    because the failure is otherwise silent: row 0 names the key, so the second
    cell's rows would be ranked against - and written under - a quota counted for
    a group they are not in. Funding row 0's key leaves the key check as the only
    thing that can reject this input.
    """
    first = _group([("clip-a", 0.5, None)], cluster=4)
    group = pa.concat_tables([first, _group([("clip-b", 0.1, None)], cluster=9)])

    with pytest.raises(ValueError, match="exactly one"):
        select_within_quota(group, {_group_key(first): 1}, "farthest")


def test_a_group_naming_two_canonical_tasks_is_rejected() -> None:
    """Two merged labels in one group would be cut against one label's quota.

    The task column is checked separately from the cell because it carries the
    level-1 merge: a guard that only looked at the cell would pass a group whose
    rows answer to two different labels.
    """
    first = _group([("clip-a", 0.5, None)], task="wash the dishes")
    group = pa.concat_tables([first, _group([("clip-b", 0.1, None)], task="fold the laundry")])

    with pytest.raises(ValueError, match="exactly one"):
        select_within_quota(group, {_group_key(first): 1}, "farthest")


def test_the_cut_preserves_cardinality() -> None:
    """One reasoned row out per row in, whatever the quota."""
    group = _group([("clip-a", 0.1, None), ("clip-b", 0.9, CurateReason.DUPLICATE), ("clip-c", 0.5, None)])

    for quota in (0, 1, 2):
        assert _reasons(group, quota=quota).keys() == {"clip-a", "clip-b", "clip-c"}


def test_the_cut_propagates_the_fragment_and_dedup_keys() -> None:
    """The write regroups by fragment and derives the cluster id, so both must survive."""
    group = _group([("clip-a", 0.5, None)], fragment=11, dedup_key=4)
    key = _group_key(group)

    verdicts = select_within_quota(group, {key: 1}, "farthest")

    assert verdicts.column(FRAGMENT_COLUMN).to_pylist() == [11]
    assert verdicts.column(DEDUP_KEY_COLUMN).to_pylist() == [4]


def test_equal_distance_and_clip_id_tiebreaks_on_fragment() -> None:
    """Same clip_id and distance on two fragments: lower fragment_id wins quota=1."""
    group = _group(
        [("same-id", 1.0, None), ("same-id", 1.0, None)],
        fragment=[8, 2],
    )
    key = _group_key(group)

    verdicts = select_within_quota(group, {key: 1}, "farthest")

    assert list(
        zip(
            verdicts.column(FRAGMENT_COLUMN).to_pylist(),
            verdicts.column(CURATE_SELECTION_REASON).to_pylist(),
            strict=True,
        )
    ) == [(8, str(CurateReason.BELOW_QUOTA)), (2, str(CurateReason.SELECTED))]


def test_the_cut_emits_exactly_the_verdict_row() -> None:
    """Fairness is the last stage before the write, so the labels and distance stop here."""
    group = _group([("clip-a", 0.5, None)])
    key = _group_key(group)

    verdicts = select_within_quota(group, {key: 1}, "farthest")

    assert verdicts.schema == VERDICT_ROW


def test_fairness_imports_without_lance_ray_or_gpu_libraries(run_child: RunChild) -> None:
    """The kernel is pure, so it must import on a CPU-only host with the driver libraries blocked."""
    result = run_child(poisoning("cosmos_curator.next.recipes.curation.fairness"))

    assert result.returncode == 0, result.stderr


def test_the_poisoned_import_harness_would_notice_a_driver_dependency(run_child: RunChild) -> None:
    """Negative control: the harness above passes for a reason, not by accident.

    ``sys.modules[name] = None`` makes an import raise rather than removing the
    module, which is subtle enough that a green purity test proves nothing until
    the same harness is shown to fail on a module that DOES import a driver
    dependency.

    Why the check matches the mechanism's own message rather than a dependency
    name is in ``assert_poisoning_fired``.
    """
    assert_poisoning_fired(run_child(poisoning("cosmos_curator.next.recipes.curation.pipeline")))
