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

"""Fused-geometry tests: the two identities, the block layout, and assignment.

Pure NumPy and Arrow - no Lance table, no Ray, no GPU - because the whole module
under test is. Batches are built here from synthetic vectors rather than from the
shared clips fixture, so a geometric claim is asserted against numbers the test
chose and can reason about.

One property makes several of these tests able to fail at all: every modality of
a row must carry a DIFFERENT direction. Permuting or reweighting two identical
blocks is a no-op, so an order- or weight-sensitivity test built on equal blocks
passes no matter what the kernel does. ``_vectors`` seeds per modality and the
tests that depend on it assert the distinctness they rely on.
"""

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.curation.vectors import (
    _MIN_NORM,
    BLOCK_DIMS,
    FUSED_DIM,
    CentroidAssigner,
    WorkingVectors,
    assign_clusters,
    unit_rows,
    vector_column_to_matrix,
    working_vectors,
)

from .conftest import RunChild, assert_poisoning_fired, poisoning

# The shipped default, and one skewed alternative, so an identity is never proved
# only under the weights the code was written against.
_DEFAULT_WEIGHTS = {"subtask": 0.6, "image": 0.2, "action": 0.2}
_SKEWED_WEIGHTS = {"subtask": 0.5, "image": 0.3, "action": 0.2}

# The block layout the fused vector is asserted against, stated here in the order
# a reader of a centroids artifact would assume. Written out rather than derived
# from FUSED_BLOCKS so a production reordering fails the layout test instead of
# being mirrored by it.
_EXPECTED_LAYOUT: tuple[tuple[EmbeddingColumnGroup, str], ...] = (
    (TEXT_COLUMN_GROUP, "subtask"),
    (IMAGE_COLUMN_GROUP, "image"),
    (ACTION_COLUMN_GROUP, "action"),
)


def _dim(group: EmbeddingColumnGroup) -> int:
    """Return the stored width of one group's primary vector."""
    return int(group.schema.field(group.primary_vector).type.list_size)


def _vectors(group: EmbeddingColumnGroup, rows: int, seed: int) -> npt.NDArray[np.float32]:
    """Return ``rows`` distinct vectors of one group's width.

    The seed is the caller's, so two modalities of one row only share a direction
    when a test deliberately asks them to.
    """
    return np.random.default_rng(seed).standard_normal((rows, _dim(group))).astype(np.float32)


def _column(group: EmbeddingColumnGroup, rows: npt.NDArray[np.float32]) -> pa.Array:
    """Build one group's ``fixed_size_list<float32, dim>`` vector column."""
    return pa.array([list(row) for row in rows], type=pa.list_(pa.float32(), _dim(group)))


def _batch(
    *,
    text: npt.NDArray[np.float32] | None = None,
    image: npt.NDArray[np.float32] | None = None,
    action: npt.NDArray[np.float32] | None = None,
) -> pa.Table:
    """Build a scanned-batch stand-in holding whichever vector columns are given.

    A ``None`` block leaves its column out of the batch entirely, which is the
    state a zero-weight modality is allowed to be in.
    """
    columns = {
        group.primary_vector: _column(group, rows)
        for group, rows in ((TEXT_COLUMN_GROUP, text), (IMAGE_COLUMN_GROUP, image), (ACTION_COLUMN_GROUP, action))
        if rows is not None
    }
    return pa.table(columns)


def _full_batch(rows: int, *, seed: int = 0) -> pa.Table:
    """Build a batch whose three modalities carry independent directions."""
    return _batch(
        text=_vectors(TEXT_COLUMN_GROUP, rows, seed + 1),
        image=_vectors(IMAGE_COLUMN_GROUP, rows, seed + 2),
        action=_vectors(ACTION_COLUMN_GROUP, rows, seed + 3),
    )


def _drop_counts(fused: WorkingVectors) -> tuple[int, int]:
    """Return ``(non_finite, zero_norm)`` rows, the way a caller of the kernel derives them.

    The kernel returns the zero-norm MASK, because the cause decides which bypass
    sentinel a dropped row is routed on and a count cannot say which row it
    counted. The counts are then this arithmetic, which is only correct while the
    two causes partition the dropped rows - the property the tests below pin.
    """
    return int((~fused.keep & ~fused.zero_norm).sum()), int(fused.zero_norm.sum())


def _unit(vector: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Return ``vector`` as a float64 unit vector."""
    wide = np.asarray(vector, dtype=np.float64)
    return wide / np.linalg.norm(wide)


def _first_row(batch: pa.Table, group: EmbeddingColumnGroup) -> npt.NDArray[np.float64]:
    """Return the batch's first row of one group's vector, as a float64 array."""
    return np.asarray(batch.column(group.primary_vector)[0].values, dtype=np.float64)


def _block_bounds() -> list[tuple[int, int]]:
    """Return the ``(start, stop)`` coordinate range of each block in fused order."""
    edges = np.cumsum((0, *BLOCK_DIMS))
    return [(int(start), int(stop)) for start, stop in itertools.pairwise(edges)]


def _weight_samples() -> list[dict[str, float]]:
    """Return weight splits spanning the simplex, for the two identity tests.

    Both identities are claimed for EVERY split summing to 1, so proving them at
    two hand-picked splits under-tests the claim. The Dirichlet draw is deliberately
    skewed (alpha < 1) to reach near-degenerate splits, where one block is scaled by
    a ``sqrt(w)`` small enough to expose a cancellation the balanced cases hide.
    """
    fields = ("subtask", "image", "action")
    samples = [_DEFAULT_WEIGHTS, _SKEWED_WEIGHTS, {"subtask": 0.999, "image": 0.0005, "action": 0.0005}]
    drawn = np.random.default_rng(2026).dirichlet((0.3, 0.3, 0.3), size=7)
    samples.extend(dict(zip(fields, (float(w) for w in row), strict=True)) for row in drawn)
    return samples


@pytest.mark.parametrize("weights", _weight_samples())
def test_every_working_vector_is_unit_norm(weights: dict[str, float]) -> None:
    """Weights summing to 1 make ``||working|| == 1``, whatever their split.

    The identity that lets a fused cosine be a cosine at all, and the reason the
    per-block scale is ``sqrt(w)``: under a plain ``w`` the norm would be
    ``sqrt(sum w_m^2)``, which is below 1 for every non-degenerate split.
    """
    fused = working_vectors(_full_batch(16), weights)

    assert fused.keep.all()
    np.testing.assert_allclose(np.linalg.norm(fused.working.astype(np.float64), axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("weights", _weight_samples())
def test_fused_distance_equals_the_weighted_sum_of_per_block_distances(weights: dict[str, float]) -> None:
    """``1 - cos(A, B)`` decomposes into ``sum_m w_m (1 - cos_m)``, exactly.

    This is what makes ``dedup_eps`` interpretable: an operator can reason about
    the per-modality distances a duplicate is allowed to have, instead of about an
    opaque number in an 865-dimensional space.
    """
    left = _full_batch(1, seed=10)
    right = _full_batch(1, seed=20)

    fused_left = working_vectors(left, weights).working[0].astype(np.float64)
    fused_right = working_vectors(right, weights).working[0].astype(np.float64)

    expected = 0.0
    for group, field in _EXPECTED_LAYOUT:
        block_cosine = _unit(_first_row(left, group)) @ _unit(_first_row(right, group))
        expected += weights[field] * (1.0 - float(block_cosine))
    assert (1.0 - float(fused_left @ fused_right)) == pytest.approx(expected, abs=1e-5)


def test_each_block_contributes_a_sub_vector_of_norm_sqrt_of_its_weight() -> None:
    """The per-block scale is ``sqrt(w)``, not ``w``.

    Pinned directly, because the two scalings differ by a monotone factor and so
    leave every ordering intact while silently changing every distance: a
    ``w``-scaled vector still clusters, just against a metric nobody chose.
    """
    batch = _full_batch(1)

    working = working_vectors(batch, _DEFAULT_WEIGHTS).working

    for (start, stop), (_, field) in zip(_block_bounds(), _EXPECTED_LAYOUT, strict=True):
        norm = float(np.linalg.norm(working[0, start:stop].astype(np.float64)))
        assert norm == pytest.approx(np.sqrt(_DEFAULT_WEIGHTS[field]), abs=1e-6)


def test_the_block_layout_follows_the_declared_fused_order() -> None:
    """Which coordinate range holds which modality is result-defining, so it is pinned.

    A centroid is only interpretable against the order that produced it, and a
    permutation would leave every norm and every identity intact while assigning
    rows to different clusters. The expected vector is rebuilt here in the
    documented order rather than read from the production tuple.
    """
    batch = _full_batch(1)
    blocks = [_first_row(batch, group) for group, _ in _EXPECTED_LAYOUT]
    # Without this the test could not fail: permuting equal blocks is a no-op.
    assert not np.allclose(_unit(blocks[0]), _unit(blocks[1]))

    working = working_vectors(batch, _DEFAULT_WEIGHTS).working

    expected = np.concatenate(
        [
            np.sqrt(_DEFAULT_WEIGHTS[field]) * _unit(block)
            for block, (_, field) in zip(blocks, _EXPECTED_LAYOUT, strict=True)
        ]
    )
    assert working.shape == (1, FUSED_DIM)
    np.testing.assert_allclose(working[0].astype(np.float64), expected, atol=1e-6)


def test_swapping_two_block_weights_changes_the_working_vector() -> None:
    """The per-modality weights are result-defining, not a tuning detail.

    Two runs at different weights cluster and de-duplicate differently, so a
    weight change must be visible in the vector itself. The distinctness assertion
    is what gives the test teeth: reweighting two identical blocks is a no-op.
    """
    batch = _full_batch(4)
    assert not np.allclose(_unit(_first_row(batch, TEXT_COLUMN_GROUP)), _unit(_first_row(batch, IMAGE_COLUMN_GROUP)))

    default = working_vectors(batch, _DEFAULT_WEIGHTS).working
    swapped = working_vectors(batch, {"subtask": 0.2, "image": 0.6, "action": 0.2}).working

    assert not np.allclose(default, swapped, atol=1e-6)


def test_scaling_the_action_vector_leaves_the_working_vector_unchanged() -> None:
    """Action is de-duplicated on gesture SHAPE, and amplitude is DELIBERATELY erased.

    The stored action vector's length encodes gesture size, and L2-normalizing it
    throws that away: the same motion performed large and small becomes one point.
    That is the decided contract - the fused distance answers "same motion
    pattern", never "same motion size" - so this test exists to make any future
    attempt to preserve amplitude fail loudly here rather than quietly change
    every duplicate verdict in the corpus.
    """
    text = _vectors(TEXT_COLUMN_GROUP, 1, 1)
    image = _vectors(IMAGE_COLUMN_GROUP, 1, 2)
    action = _vectors(ACTION_COLUMN_GROUP, 1, 3)
    batch = _batch(
        text=np.vstack([text, text]),
        image=np.vstack([image, image]),
        action=np.vstack([action, 10.0 * action]),
    )
    # The amplitude difference is real in the input, which is the premise of the
    # claim that the output erases it.
    assert np.linalg.norm(10.0 * action) == pytest.approx(10.0 * float(np.linalg.norm(action)), rel=1e-6)

    working = working_vectors(batch, _DEFAULT_WEIGHTS).working

    np.testing.assert_allclose(working[0], working[1], atol=1e-6)


def test_changing_the_action_direction_changes_the_working_vector() -> None:
    """Action is direction-only, not weightless.

    The complement of the magnitude-erasure test: without this, a kernel that
    dropped the action block entirely would pass that test and lose the
    motion-redundancy separation the block's weight is spent on.
    """
    text = _vectors(TEXT_COLUMN_GROUP, 1, 1)
    image = _vectors(IMAGE_COLUMN_GROUP, 1, 2)
    action = _vectors(ACTION_COLUMN_GROUP, 2, 3)
    batch = _batch(text=np.vstack([text, text]), image=np.vstack([image, image]), action=action)

    working = working_vectors(batch, _DEFAULT_WEIGHTS).working

    assert not np.allclose(working[0], working[1], atol=1e-6)


def test_a_non_finite_coordinate_drops_the_row_as_non_finite() -> None:
    """An ``inf`` row is unusable and is counted as non-finite, never as zero-norm.

    Finiteness must be tested before the division, because ``norm(inf) == inf``
    passes a norm-only gate and then normalizes to NaN - which would survive into
    a centroid and from there into every row's assignment.
    """
    image = _vectors(IMAGE_COLUMN_GROUP, 2, 2)
    image[1, 0] = np.inf
    batch = _batch(text=_vectors(TEXT_COLUMN_GROUP, 2, 1), image=image, action=_vectors(ACTION_COLUMN_GROUP, 2, 3))

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True, False]
    assert _drop_counts(fused) == (1, 0)
    assert np.isfinite(fused.working).all()


def test_a_nan_coordinate_drops_the_row_as_non_finite() -> None:
    """A NaN is unusable for the same reason an ``inf`` is, and is counted the same way."""
    action = _vectors(ACTION_COLUMN_GROUP, 2, 3)
    action[0, 5] = np.nan
    batch = _batch(text=_vectors(TEXT_COLUMN_GROUP, 2, 1), image=_vectors(IMAGE_COLUMN_GROUP, 2, 2), action=action)

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [False, True]
    assert _drop_counts(fused) == (1, 0)


def test_an_all_zero_block_drops_the_row_as_zero_norm() -> None:
    """A finite vector with no direction cannot be normalized, so its row is unusable."""
    text = _vectors(TEXT_COLUMN_GROUP, 3, 1)
    text[1] = 0.0
    batch = _batch(text=text, image=_vectors(IMAGE_COLUMN_GROUP, 3, 2), action=_vectors(ACTION_COLUMN_GROUP, 3, 3))

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True, False, True]
    assert _drop_counts(fused) == (0, 1)


def test_a_huge_magnitude_row_still_lands_on_the_unit_sphere() -> None:
    """A row whose sum of squares exceeds the float32 range is still normalized correctly.

    Norms are accumulated in float64 precisely for this row: a float32 sum of
    squares saturates to ``inf``, the division then yields all zeros, and the row
    is kept as a finite vector sitting at the origin - an off-sphere point that
    every identity here assumes cannot exist, produced without any error.
    """
    # 384 * (1e19)^2 = 3.8e38, past the float32 maximum, from coordinates well
    # inside it.
    huge = np.full((1, _dim(TEXT_COLUMN_GROUP)), 1e19, dtype=np.float32)
    assert not np.isfinite(np.float32(np.square(np.float64(1e19)) * _dim(TEXT_COLUMN_GROUP)))
    batch = _batch(text=huge, image=_vectors(IMAGE_COLUMN_GROUP, 1, 2), action=_vectors(ACTION_COLUMN_GROUP, 1, 3))

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True]
    assert _drop_counts(fused) == (0, 0)
    assert float(np.linalg.norm(fused.working[0].astype(np.float64))) == pytest.approx(1.0, abs=1e-5)


def test_a_row_whose_norm_passes_the_float32_ceiling_still_lands_on_the_unit_sphere() -> None:
    """A norm too large for float32 must not be narrowed into the divisor.

    The second of two thresholds, and the accumulator only covers the first. Past
    ~1.8e19 the SUM OF SQUARES overflows, which float64 accumulation handles; past
    ~3.4e38 the resulting NORM is itself unrepresentable in float32, so casting it
    back down to divide re-creates the same ``inf`` one step later. ``keep`` is
    computed from the wide norm and still selects the row, so the failure is a
    fused vector at the origin rather than a dropped row.
    """
    huge = np.full((1, _dim(TEXT_COLUMN_GROUP)), 2e37, dtype=np.float32)
    assert np.isfinite(huge).all()
    assert float(np.linalg.norm(huge[0].astype(np.float64))) > float(np.finfo(np.float32).max)
    batch = _batch(text=huge, image=_vectors(IMAGE_COLUMN_GROUP, 1, 2), action=_vectors(ACTION_COLUMN_GROUP, 1, 3))

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True]
    assert float(np.linalg.norm(fused.working[0].astype(np.float64))) == pytest.approx(1.0, abs=1e-5)


def test_the_zero_norm_mask_names_which_dropped_row_it_claims() -> None:
    """The mask attributes each dropped row to exactly one cause, by POSITION.

    The caller routes a dropped row to a bypass sentinel chosen from this mask, so
    a mask that merely counted correctly - claiming the non-finite row instead of
    the zero-norm one - would send both rows to the wrong sentinel while every
    tally still balanced.
    """
    image = _vectors(IMAGE_COLUMN_GROUP, 4, 2)
    image[1, 0] = np.inf
    image[2] = 0.0
    batch = _batch(text=_vectors(TEXT_COLUMN_GROUP, 4, 1), image=image, action=_vectors(ACTION_COLUMN_GROUP, 4, 3))

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True, False, False, True]
    assert fused.zero_norm.tolist() == [False, False, True, False]
    assert (~fused.keep & ~fused.zero_norm).tolist() == [False, True, False, False]


def test_working_rows_align_with_the_kept_rows_and_not_with_the_batch() -> None:
    """The returned matrix is indexed by KEPT row, in input order.

    The one contract a caller can get wrong silently: reading ``working[i]`` as
    row ``i`` of the batch would attach every verdict after a dropped row to the
    wrong clip. Each surviving row is compared against fusing it on its own.
    """
    text = _vectors(TEXT_COLUMN_GROUP, 3, 1)
    image = _vectors(IMAGE_COLUMN_GROUP, 3, 2)
    action = _vectors(ACTION_COLUMN_GROUP, 3, 3)
    image[1] = 0.0
    batch = _batch(text=text, image=image, action=action)

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    assert fused.keep.tolist() == [True, False, True]
    assert fused.working.shape == (2, FUSED_DIM)
    for position, row in enumerate([0, 2]):
        one_row = _batch(text=text[row : row + 1], image=image[row : row + 1], action=action[row : row + 1])
        np.testing.assert_allclose(
            fused.working[position], working_vectors(one_row, _DEFAULT_WEIGHTS).working[0], atol=1e-6
        )


def test_a_zero_weight_block_contributes_zeros_and_keeps_the_vector_unit_norm() -> None:
    """A zero weight removes a modality from the metric without breaking the identity.

    The documented escape for a corpus whose labels are identifiers rather than
    language: the block's coordinates go to zero and the remaining weights still
    sum to 1, so the vector stays on the unit sphere and every distance stays
    decomposable.
    """
    weights = {"subtask": 0.0, "image": 0.5, "action": 0.5}
    text_start, text_stop = _block_bounds()[0]

    working = working_vectors(_full_batch(4), weights).working

    assert np.count_nonzero(working[:, text_start:text_stop]) == 0
    np.testing.assert_allclose(np.linalg.norm(working.astype(np.float64), axis=1), 1.0, atol=1e-5)


def test_a_zero_weight_block_is_never_read_from_the_batch() -> None:
    """A zero-weight modality is not required to be present at all.

    The eligibility predicate drops that column's ``IS NOT NULL`` clause, so rows
    with no text embedding reach this kernel. Reading the column anyway would
    decode NULL placeholder values, or fail outright on a batch that never
    projected it.
    """
    weights = {"subtask": 0.0, "image": 0.5, "action": 0.5}
    batch = _batch(image=_vectors(IMAGE_COLUMN_GROUP, 2, 2), action=_vectors(ACTION_COLUMN_GROUP, 2, 3))
    assert TEXT_COLUMN_GROUP.primary_vector not in batch.schema.names

    fused = working_vectors(batch, weights)

    assert fused.keep.all()
    assert fused.working.shape == (2, FUSED_DIM)


def test_a_nan_in_a_zero_weight_block_keeps_the_row() -> None:
    """An unweighted modality's broken vector cannot make a row invalid.

    Both degeneracy tests skip a zero-weight block, so a corpus with unusable
    text still curates on image and action alone.
    """
    weights = {"subtask": 0.0, "image": 0.5, "action": 0.5}
    text = np.full((1, _dim(TEXT_COLUMN_GROUP)), np.nan, dtype=np.float32)
    batch = _batch(text=text, image=_vectors(IMAGE_COLUMN_GROUP, 1, 2), action=_vectors(ACTION_COLUMN_GROUP, 1, 3))

    fused = working_vectors(batch, weights)

    assert fused.keep.tolist() == [True]
    assert _drop_counts(fused) == (0, 0)
    assert np.isfinite(fused.working).all()
    assert float(np.linalg.norm(fused.working[0].astype(np.float64))) == pytest.approx(1.0, abs=1e-5)


def test_an_empty_batch_returns_typed_empty_results() -> None:
    """A batch whose every row was filtered out is not an error.

    A fragment can legitimately hold no eligible row, and the scan calls this
    kernel per fragment, so the empty case has to be total rather than guarded at
    every call site.
    """
    fused = working_vectors(_full_batch(0), _DEFAULT_WEIGHTS)

    assert fused.keep.shape == (0,)
    assert fused.keep.dtype == np.bool_
    assert fused.working.shape == (0, FUSED_DIM)
    assert fused.working.dtype == np.float32
    assert _drop_counts(fused) == (0, 0)


def test_a_missing_weight_raises_rather_than_silently_unweighting_a_block() -> None:
    """A block with no weight is a caller error, not a zero.

    Defaulting it to zero would drop the modality out of the metric while the
    eligibility predicate still required its vector - two halves of the contract
    disagreeing, with nothing to say so.
    """
    with pytest.raises(KeyError, match="action"):
        working_vectors(_full_batch(1), {"subtask": 0.6, "image": 0.4})


def test_a_block_stored_at_an_unexpected_width_is_rejected() -> None:
    """A table embedded against a different block geometry fails instead of curating.

    ``fixed_size_list`` guarantees one width per column, never that it is the
    width this run fuses, and the two numbers come from different places: the
    schema module here, the embed leg that wrote the table there. Accepted
    silently, a narrower block shifts every later coordinate into the wrong
    modality and the run clusters, de-duplicates and commits against a basis
    nobody chose, reporting success the whole way.
    """
    batch = pa.table(
        {
            TEXT_COLUMN_GROUP.primary_vector: pa.array([[1.0] * 8], type=pa.list_(pa.float32(), 8)),
            IMAGE_COLUMN_GROUP.primary_vector: _column(IMAGE_COLUMN_GROUP, _vectors(IMAGE_COLUMN_GROUP, 1, 2)),
            ACTION_COLUMN_GROUP.primary_vector: _column(ACTION_COLUMN_GROUP, _vectors(ACTION_COLUMN_GROUP, 1, 3)),
        }
    )

    with pytest.raises(ValueError, match="different block geometry"):
        working_vectors(batch, _DEFAULT_WEIGHTS)


def test_the_exported_fused_width_matches_the_vector_the_kernel_builds() -> None:
    """``FUSED_DIM`` describes the real output width, not a stale copy.

    The centroids artifact records it so a saved basis can be interpreted, and the
    basis shape is validated against it; a constant that drifted from the kernel
    would validate the wrong thing.
    """
    working = working_vectors(_full_batch(2), _DEFAULT_WEIGHTS).working

    assert working.shape[1] == FUSED_DIM
    assert sum(BLOCK_DIMS) == FUSED_DIM


def test_the_exported_block_widths_match_the_columns_they_describe() -> None:
    """``BLOCK_DIMS`` follows each group's own column width, in fused order.

    The block boundaries are what make a centroid coordinate mean a modality, so
    a width read from the wrong group would mis-slice every saved basis.
    """
    assert list(BLOCK_DIMS) == [_dim(group) for group, _ in _EXPECTED_LAYOUT]


def test_vector_column_to_matrix_reads_a_fixed_size_list_column() -> None:
    """A fixed-size-list column decodes to an ``(n, dim)`` float32 matrix."""
    column = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2))

    matrix = vector_column_to_matrix(column)

    assert matrix.dtype == np.float32
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_vector_column_to_matrix_rebases_a_sliced_column_offset() -> None:
    """A sliced column decodes its OWN rows, not the child buffer's first rows.

    ``FixedSizeListArray.values`` ignores the slice offset, so without the rebase
    a batched read would pair every clip with another clip's vector - silently, and
    with every downstream shape still correct.
    """
    column = pa.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], type=pa.list_(pa.float32(), 2))

    matrix = vector_column_to_matrix(column.slice(1))

    assert matrix.tolist() == [[2.0, 3.0], [4.0, 5.0]]


def test_vector_column_to_matrix_reads_a_chunked_column() -> None:
    """A multi-chunk column decodes as one matrix, in chunk order.

    A table column is a ``ChunkedArray``, and a batch assembled from two reads
    arrives with more than one chunk.
    """
    chunk = pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2))

    matrix = vector_column_to_matrix(pa.chunked_array([chunk, chunk]))

    assert matrix.tolist() == [[1.0, 2.0], [1.0, 2.0]]


def test_vector_column_to_matrix_rejects_a_non_fixed_size_list_column() -> None:
    """A variable-width list column has no fixed vector width and is refused.

    Reshaping it would need an offsets walk this decoder does not do, so the
    mismatch fails immediately instead of producing a plausible wrong matrix.
    """
    column = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32()))

    with pytest.raises(TypeError, match="fixed_size_list"):
        vector_column_to_matrix(column)


def _unit_reference(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-normalize a float32 matrix to unit L2 norm, independently of the module."""
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def test_assignment_matches_a_brute_force_reference() -> None:
    """The cluster and the distance match an independent full-matrix computation."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((4, 8)).astype(np.float32)
    block = _unit_reference(rng.standard_normal((16, 8)).astype(np.float32))

    cluster_id, distance = CentroidAssigner.from_raw(raw).score(block)

    sims = block @ _unit_reference(raw).T
    expected_id = np.argmax(sims, axis=1)
    assert np.array_equal(cluster_id, expected_id.astype(np.int32))
    best_sim = np.clip(sims[np.arange(block.shape[0]), expected_id], -1.0, 1.0)
    np.testing.assert_allclose(distance, 1.0 - best_sim, atol=1e-6)


def test_distance_stays_non_negative_when_similarity_exceeds_one() -> None:
    """Float32 round-off can push the winning dot product above 1.0; distance must stay >= 0."""
    raw = np.array([[1.0, 0.0]], dtype=np.float32)
    block = np.array([[1.0001, 0.0]], dtype=np.float32)

    _, distance = CentroidAssigner.from_raw(raw).score(block)

    assert distance[0] >= 0.0


def test_raw_and_normalized_centroids_disagree_on_the_nearest_cluster() -> None:
    """The normalization gate changes the ANSWER, not just the scale.

    Point ``[1, 0]`` is nearest in angle to centroid ``[1, 0]``, but centroid
    ``[2, 2]`` is longer and so wins the raw dot product from 45 degrees away.
    Skipping the gate therefore assigns rows to the wrong clusters, which is why
    the fit's array stays raw on disk and is normalized exactly once.
    """
    raw = np.array([[1.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    block = np.array([[1.0, 0.0]], dtype=np.float32)

    cluster_id, _ = CentroidAssigner.from_raw(raw).score(block)

    assert cluster_id[0] == 0
    assert int(np.argmax(block @ raw.T, axis=1)[0]) == 1


def test_assignment_ties_break_to_the_lowest_cluster_id() -> None:
    """Two equidistant centroids resolve to the lower id, so the result is reproducible."""
    raw = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    block = _unit_reference(np.array([[1.0, 1.0]], dtype=np.float32))

    cluster_id, _ = CentroidAssigner.from_raw(raw).score(block)

    assert cluster_id[0] == 0


def test_a_zero_norm_centroid_is_rejected_at_construction() -> None:
    """A degenerate basis fails loud instead of capturing the whole corpus.

    cuML genuinely emits empty clusters; normalizing one yields NaN, and NaN wins
    every ``argmax``, so the failure mode is one cluster silently holding every
    row rather than an error.
    """
    raw = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="degenerate"):
        CentroidAssigner.from_raw(raw)


def test_an_empty_centroid_basis_is_rejected_at_construction() -> None:
    """An empty (0, d) basis must fail in from_raw, not in score's argmax."""
    raw = np.empty((0, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"non-empty \(k, d\) matrix"):
        CentroidAssigner.from_raw(raw)


def test_large_float32_centroids_normalize_with_float64_norms() -> None:
    """A row whose float32 norm overflows must not divide to all zeros.

    ``np.linalg.norm`` on float32 rows can square coordinates to ``inf`` while the
    values themselves stay finite, so the zero-norm gate passes and the division
    silently wipes the row. Accumulating in float64 keeps the norm finite and the
    unit vector on the unit sphere.
    """
    raw = np.array([[1e20, 0.0]], dtype=np.float32)
    assert float(np.linalg.norm(raw, axis=1, keepdims=True)[0, 0]) == float("inf")

    assigner = CentroidAssigner.from_raw(raw)
    unit = assigner.centroids_unit

    np.testing.assert_allclose(unit, np.array([[1.0, 0.0]], dtype=np.float32), rtol=0, atol=1e-6)


def test_a_centroid_whose_norm_passes_the_float32_ceiling_normalizes_to_unit() -> None:
    """A centroid norm above the float32 maximum must not be narrowed into the divisor.

    Every construction guard upstream passes such a row - it is finite, and its
    norm is far above the zero-norm floor - so narrowing the divisor seats a row
    of zeros in a basis that just reported itself clean. A zero centroid scores 0
    against every vector, so it wins nothing and is never assigned: k cells were
    fitted and only k-1 remain reachable.
    """
    raw = np.array([[2e38, 3e38], [1.0, 0.0]], dtype=np.float32)
    assert np.isfinite(raw).all()
    assert float(np.linalg.norm(raw[0].astype(np.float64))) > float(np.finfo(np.float32).max)

    unit = CentroidAssigner.from_raw(raw).centroids_unit

    np.testing.assert_allclose(np.linalg.norm(unit.astype(np.float64), axis=1), 1.0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 3), (0,)])
def test_scoring_an_empty_block_returns_empty_typed_arrays(shape: tuple[int, ...]) -> None:
    """An all-invalid batch scores to empty int32/float32 arrays, not an error.

    Both empty shapes are accepted, so a caller shortcutting a batch with no
    surviving row does not have to carry the fused width along just to be scored;
    a widthless empty block would otherwise fail in the matmul.
    """
    assigner = CentroidAssigner.from_raw(np.eye(3, dtype=np.float32))

    cluster_id, distance = assigner.score(np.empty(shape, dtype=np.float32))

    assert (cluster_id.shape, cluster_id.dtype) == ((0,), np.int32)
    assert (distance.shape, distance.dtype) == ((0,), np.float32)


# The reserved cell id a row with no usable vector lands in. Negative on purpose,
# so it can never collide with a real cell index whatever k the run chose.
_ABSENT = -1


@pytest.fixture(name="axes")
def fixture_axes() -> CentroidAssigner:
    """Return an assigner over the two coordinate axes, so a cell id is readable by eye.

    Cell 0 is ``+x`` and cell 1 is ``+y``, which is what lets the assignment tests
    state an expected id rather than compare against a second implementation of
    the kernel they are checking.
    """
    return CentroidAssigner.from_raw(np.eye(2, dtype=np.float32))


def test_unit_rows_leaves_a_directionless_row_as_zeros(axes: CentroidAssigner) -> None:
    """A zero row must not divide, because 0/0 is NaN and NaN wins every argmax.

    Asserted on the returned matrix rather than through an assignment, because the
    caller drops unusable rows before scoring - so a NaN here would be invisible
    downstream right up until some other caller scored the full matrix.
    """
    unit, usable = unit_rows(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))

    assert np.array_equal(unit[0], np.zeros(2, dtype=np.float32))
    assert usable.tolist() == [False, True]
    assert axes.score(unit[usable])[0].tolist() == [0]


def test_a_row_whose_norm_passes_the_float32_ceiling_reaches_its_true_cell(axes: CentroidAssigner) -> None:
    """A row too long for a float32 norm is assigned on its direction, not on its length.

    Length carries no cell information, so this row belongs in the same cell as
    its unit-scale twin. Narrowing the divisor zeros it while ``usable`` still
    holds, and an all-zero row scores 0 against every centroid - so ``argmax``
    hands it to cell 0. It lands in a real cell, never the reserved one, which is
    why the mis-assignment leaves no trace for a reader to find.
    """
    row = np.array([[2e38, 3e38]], dtype=np.float32)
    assert np.isfinite(row).all()
    assert float(np.linalg.norm(row[0].astype(np.float64))) > float(np.finfo(np.float32).max)
    present = np.array([True])

    at_unit_scale = assign_clusters(np.array([[2.0, 3.0]], dtype=np.float32), present, axes, _ABSENT)

    assert at_unit_scale.tolist() == [1]
    assert assign_clusters(row, present, axes, _ABSENT).tolist() == [1]


def test_a_row_with_no_vector_lands_in_the_reserved_cell(axes: CentroidAssigner) -> None:
    """Absence is read from the column's validity, not from the buffer's contents.

    Arrow leaves the values under a null slot undefined, and a null slot commonly
    still holds a perfectly finite vector. The row's cell must come from
    ``present`` alone, or a NULL subtask embedding silently joins a real cell and
    competes for its quota.
    """
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    present = np.array([False, True])

    cells = assign_clusters(matrix, present, axes, _ABSENT)

    assert cells.tolist() == [_ABSENT, 1]


def test_a_directionless_row_lands_in_the_reserved_cell(axes: CentroidAssigner) -> None:
    """A stored zero vector is present but has no position, so no cell is nearest to it."""
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    cells = assign_clusters(matrix, np.array([True, True]), axes, _ABSENT)

    assert cells.tolist() == [_ABSENT, 0]


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_row_lands_in_the_reserved_cell(axes: CentroidAssigner, bad: float) -> None:
    """A non-finite coordinate must be excluded BEFORE the division, not after.

    An infinite coordinate has a finite-looking norm test - ``inf > 1e-12`` is
    True - so a usable mask built from the norm alone admits it, and ``inf/inf``
    then makes the unit row NaN. NaN wins ``argmax``, so the row would capture
    cell 0 instead of being reported as having no cell at all.
    """
    matrix = np.array([[bad, 0.0], [0.0, 1.0]], dtype=np.float32)

    cells = assign_clusters(matrix, np.array([True, True]), axes, _ABSENT)

    assert cells.tolist() == [_ABSENT, 1]


def test_unusable_rows_do_not_shift_the_cells_of_their_neighbours(axes: CentroidAssigner) -> None:
    """Cells are returned in INPUT order, so a gap must not slide later rows onto it.

    The kernel scores only the usable subset, so its output is shorter than the
    batch. Scattering that back by the mask is what keeps row i's cell on row i;
    a positional copy would assign every row after a gap its predecessor's cell.
    """
    matrix = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    cells = assign_clusters(matrix, np.array([True, True, True]), axes, _ABSENT)

    assert cells.tolist() == [0, _ABSENT, 1]


def test_a_batch_with_no_usable_row_is_all_reserved_and_typed(axes: CentroidAssigner) -> None:
    """Every row unusable is legal: the whole batch takes the reserved cell, as int32.

    The column type is part of the contract because this array becomes the
    level-2 group key, and a shuffle key whose type varies with the batch would
    split one group in two.
    """
    matrix = np.zeros((3, 2), dtype=np.float32)

    cells = assign_clusters(matrix, np.array([True, True, True]), axes, _ABSENT)

    assert cells.tolist() == [_ABSENT] * 3
    assert cells.dtype == np.int32


def _text_rows_at_norms(norms: Sequence[np.float32]) -> npt.NDArray[np.float32]:
    """Return one subtask-text row per norm, each a single axis scaled to that norm.

    One non-zero component, so the row's norm is the given value exactly and no
    summation rounding sits between the fixture and the floor it straddles.
    """
    rows = np.zeros((len(norms), _dim(TEXT_COLUMN_GROUP)), dtype=np.float32)
    rows[:, 0] = np.asarray(norms, dtype=np.float32)
    return rows


def test_the_subtask_cell_and_the_fusion_gate_reject_the_same_rows() -> None:
    """Both gates admit exactly the same rows, to one float32 step.

    The agreement is what keeps the reserved level-2 cell empty of survivors in a
    run that WEIGHTS the subtask block, and the fairness bound depends on it: a
    row that took that cell while still being a fused survivor would draw a
    sibling's share of its task's quota while standing for no region of subtask
    meaning at all - silently, since neither the unfunded nor the below-quota
    population is attributable from a written row. Unweighted, no basis is fitted
    and EVERY row takes that cell, which is safe for the other reason: it is then
    each task's only cell, with no sibling to take a share from.

    Both verdicts route through one ``_finite_and_directed`` predicate, so the
    agreement is true by construction; what this pins is the construction, and it
    fails if either surface regains a norm comparison of its own. The fixture is
    a LADDER across the floor because degenerate fills alone would not do: 0, inf
    and nan are rejected at any threshold, so a fixture of those passes even when
    the two gates disagree. The rungs one step either side are what fail then.

    Equality rather than implication because the text norm is the only cause
    varying here, and only equality catches both directions: a stricter fusion
    gate rejects a row the assigner still places, which implication misses.
    """
    floor = np.float32(_MIN_NORM)
    norms = [
        np.float32(0.0),
        np.nextafter(floor, np.float32(0.0)),
        floor,
        np.nextafter(floor, np.float32(1.0)),
        floor * np.float32(2.0),
        np.float32(1.0),
        np.float32(np.inf),
        np.float32(np.nan),
    ]
    text = _text_rows_at_norms(norms)
    rows = len(norms)
    batch = _batch(
        text=text,
        image=_vectors(IMAGE_COLUMN_GROUP, rows, 2),
        action=_vectors(ACTION_COLUMN_GROUP, rows, 3),
    )
    # All-present, because the eligibility predicate requires the subtask column
    # non-NULL whenever the block carries weight: absence is a separate cause,
    # pinned by test_a_row_with_no_vector_lands_in_the_reserved_cell, and would
    # reach the reserved cell without the row's own degenerate direction being
    # what put it there.
    subtask_cells = assign_clusters(
        text,
        np.ones(rows, dtype=bool),
        CentroidAssigner.from_raw(np.eye(2, _dim(TEXT_COLUMN_GROUP), dtype=np.float32)),
        _ABSENT,
    )

    fused = working_vectors(batch, _DEFAULT_WEIGHTS)

    unplaceable = (subtask_cells == _ABSENT).tolist()
    # Reachability: an equality over a ladder that never crosses the floor would
    # hold with both gates deleted.
    assert set(unplaceable) == {True, False}
    assert unplaceable == [not kept for kept in fused.keep.tolist()]


def test_vectors_imports_without_lance_ray_or_gpu_libraries(run_child: RunChild) -> None:
    """The geometry kernel imports on a CPU-only host with the driver libraries blocked.

    Purity is what makes every identity above assertable without a cluster: the
    module may reach ``numpy``, ``pyarrow`` and the column contract, and nothing
    else. A transitive Lance or Ray import would move these tests behind a
    driver's dependency set.
    """
    result = run_child(poisoning("cosmos_curator.next.recipes.curation.vectors"))

    assert result.returncode == 0, result.stderr


def test_the_poisoned_import_harness_would_notice_a_driver_dependency(run_child: RunChild) -> None:
    """Negative control: the purity test above passes for a reason, not by accident.

    ``sys.modules[name] = None`` makes an import raise rather than removing the
    module, which is subtle enough that a typo in the poison list would read as a
    passing purity guarantee. Pointing the same harness at a module that DOES
    import a driver dependency is what rules that out.

    Why the check matches the mechanism's own message rather than a dependency
    name is in ``assert_poisoning_fired``.
    """
    assert_poisoning_fired(run_child(poisoning("cosmos_curator.next.recipes.curation.pipeline")))
