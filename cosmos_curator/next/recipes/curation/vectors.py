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

"""Where a row sits: its working vector and locality cell, and its subtask cell.

Curate clusters and de-duplicates on ONE metric, and this module is the only
place that builds it. Every block is L2-normalized, scaled by ``sqrt(w)``, and
concatenated in ``FUSED_BLOCKS`` order::

    working = [ sqrt(w_0) * u_0 | sqrt(w_1) * u_1 | sqrt(w_2) * u_2 ]
                 subtask 384        image 384        action 97      -> 865
                                        |
                                        v
                             CentroidAssigner.score
                                        |
                                        v
                          (cluster_id, distance_to_centroid)

Two identities hold because the weights sum to 1, and together they are what let
one vector stand in for three modalities::

    ||working||^2 = sum_m w_m ||u_m||^2 = sum_m w_m = 1
    1 - cos(A, B) = sum_m w_m (1 - cos_m)

``sqrt(w)`` rather than ``w`` is what makes the second identity exact: the dot
product of concatenated blocks is additive, so each block contributes
``w_m (u_mA . u_mB)`` instead of ``w_m^2 (...)``. Scaling by ``w`` would leave a
vector of norm ``sqrt(sum w_m^2)`` and a fused cosine that decomposes into
nothing an operator could reason about when choosing ``dedup_eps``.

Normalizing the action block DISCARDS gesture magnitude, which is the decision
rather than an oversight. Stored action vectors are raw PCA coordinates whose
length encodes gesture size, so one direction at a tenfold amplitude ratio
becomes one point: the fused distance answers "same motion pattern", never "same
motion size". Preserving amplitude would take a global action rescale, which
breaks ``||working|| = 1``, which in turn makes the fused cosine
non-decomposable, ``dedup_eps`` uninterpretable, and spherical k-means invalid.
It is therefore a different architecture, not a knob.

``assign_clusters`` runs the same nearest-centroid kernel over ONE stored column
rather than over the fused vector, and that is how the level-2 fairness group is
computed. It is deliberately the same code and deliberately not the same
geometry: the fused basis answers "is this row near that one", the subtask text
basis answers "which region of instruction meaning is this row in", and the only
thing that keeps them apart is which vectors each assigner is handed. Nothing
here derives a fairness group from the fused vector, which measurement forbids -
the fused metric carries no task signal (same-task-closer AUC 0.543).

Pure NumPy and Arrow - no Ray, Lance, cuML or cuPy, directly or transitively -
so every geometric claim above is testable on a CPU-only host.

See docs/curator/design/curator-next-curation.md.
"""

from collections.abc import Mapping

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.next.recipes.curation.columns import FUSED_BLOCKS, block_width

# A row vector at or below this norm has no direction, so no normalization can
# rescue it; the row is invalid_embedding rather than a point in the space.
_MIN_NORM: float = 1e-12

# A fitted centroid is a mean of unit vectors, so its norm lies in (0, 1]. A row
# at or below this is a broken basis (an empty or re-initialized cuML cluster, a
# truncated artifact); normalizing it would emit NaN, which np.argmax then treats
# as the winner and which would silently capture every row in the corpus.
_MIN_CENTROID_NORM: float = 1e-12


BLOCK_DIMS: tuple[int, ...] = tuple(block_width(group) for group, _ in FUSED_BLOCKS)
"""Per-block width, in fused order, read off each group's own ``fixed_size_list``.

Derived rather than declared, so the widths cannot drift from the columns they
describe. Together with the order they make a set of centroids readable: a
coordinate of a fused vector means nothing without the block boundaries and the
weights that produced it.
"""

FUSED_DIM: int = sum(BLOCK_DIMS)
"""Width of one working vector, and therefore of one centroid."""


def vector_column_to_matrix(col: pa.ChunkedArray | pa.Array) -> npt.NDArray[np.float32]:
    """Convert a ``fixed_size_list<float32>`` column to an ``(n, dim)`` matrix.

    Keeps the slice-offset rebase, which is load-bearing rather than defensive:
    ``FixedSizeListArray.values`` returns the whole child buffer and ignores a
    nonzero slice offset, so a sliced column - which a batched read can hand you -
    would decode the buffer's first rows instead of its own.

    Raises:
        TypeError: If ``col`` is not a fixed-size-list column.

    """
    array = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if not pa.types.is_fixed_size_list(array.type):
        msg = f"vector_column_to_matrix expects a fixed_size_list column, got {array.type}"
        raise TypeError(msg)
    list_size = array.type.list_size
    n_rows = len(array)
    child = array.values.slice(array.offset * list_size, n_rows * list_size)
    flat = np.asarray(child.to_numpy(zero_copy_only=False), dtype=np.float32)
    return flat.reshape(n_rows, list_size)


def _block_weights(weights: Mapping[str, float]) -> tuple[float, ...]:
    """Return the weight funding each fused block, in fused order.

    Raises:
        KeyError: If a block has no weight, since a silently unweighted modality
            would drop out of the metric and out of the eligibility predicate.

    """
    return tuple(float(weights[field]) for _, field in FUSED_BLOCKS)


def _row_norms(block: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Return each row's L2 norm, accumulated in float64.

    The accumulation width is load-bearing rather than cautious. A float32 sum of
    squares overflows to ``inf`` once a row's norm passes ~1.8e19 - far inside the
    float32 range its coordinates live in - and the row then normalizes to all
    zeros: kept, finite, and off the unit sphere. That is the one input class that
    could break this module's identities without raising anywhere.

    The float64 return is half of that guarantee and callers own the other half:
    divide by it as-is. Narrowing it back to float32 first reinstates the same
    failure at the float32 ceiling, which a row norm can exceed while every
    coordinate stays finite.
    """
    squares: npt.NDArray[np.float64] = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    return np.sqrt(squares)


def _finite_and_directed(
    block: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Return ``(finite, directed, norms)`` for one raw block.

    The single site that judges a row usable. Both the fusion classifier and the
    centroid assigner route through it, so the reserved level-2 cell and the
    fused-survivor set cannot drift apart: this is the only comparison against
    ``_MIN_NORM`` in the module.

    Args:
        block: ``(n, d)`` raw vectors.

    Returns:
        ``finite`` marks rows whose every coordinate is finite; ``directed`` marks
        rows that are finite AND carry a direction, so normalizing them is
        meaningful; ``norms`` are the guarded row norms, safe to divide by
        wherever ``directed`` holds.

    """
    finite: npt.NDArray[np.bool_] = np.asarray(np.isfinite(block).all(axis=1), dtype=np.bool_)
    # Finiteness is tested BEFORE any division, and that ordering is the whole
    # point of the check: an inf row has norm = inf, which passes a norm-only
    # gate, and then normalizes to NaN and poisons whichever centroid it is
    # averaged into. A NaN row is caught here for the same reason.
    #
    # Substituting 1.0 on the non-finite rows is what makes the returned norms
    # finite on EVERY row. Those rows are already excluded by ``finite``, so the
    # substitution cannot change a verdict; what it buys is that a caller's
    # arithmetic over the full array - a float32 cast, a masked divide - carries
    # no NaN or inf lanes. The copy is skipped when nothing needs substituting,
    # which is the common case.
    guarded = block if bool(finite.all()) else np.where(finite[:, None], block, np.float32(1.0))
    norms = _row_norms(guarded)
    directed: npt.NDArray[np.bool_] = finite & (norms > _MIN_NORM)
    return finite, directed, norms


def _decode_blocks(
    batch: pa.Table,
    block_weights: tuple[float, ...],
) -> list[npt.NDArray[np.float32] | None]:
    """Decode each weighted block's vectors; ``None`` marks a zero-weight block.

    A zero-weight block is not read at all, so its column may be absent from the
    batch or entirely NULL - the same escape the eligibility predicate gives it.

    Raises:
        ValueError: If a stored block is not the width this run fuses.
            ``fixed_size_list`` pins that a column has ONE width, never which one,
            and the two numbers have independent origins: ``BLOCK_DIMS`` follows
            the schema module, the column follows whatever the embed leg wrote. A
            corpus embedded against a different width would otherwise cluster,
            de-duplicate and commit on a basis nobody chose, reporting success
            throughout.

    """
    blocks: list[npt.NDArray[np.float32] | None] = []
    for (group, _), weight, width in zip(FUSED_BLOCKS, block_weights, BLOCK_DIMS, strict=True):
        if weight == 0.0:
            blocks.append(None)
            continue
        matrix = vector_column_to_matrix(batch.column(group.primary_vector))
        if matrix.shape[1] != width:
            msg = (
                f"{group.primary_vector} stores {matrix.shape[1]}-wide vectors but this run fuses "
                f"{width}; the table was embedded against a different block geometry"
            )
            raise ValueError(msg)
        blocks.append(matrix)
    return blocks


def _classify(
    blocks: list[npt.NDArray[np.float32] | None],
    n_rows: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Return ``(keep, zero_norm)`` over the decoded blocks.

    The two causes are disjoint - a non-finite row is never also zero-norm - so a
    caller attributes every dropped row to exactly one cause, deriving the
    non-finite rows as ``~keep & ~zero_norm``.
    """
    finite = np.ones(n_rows, dtype=bool)
    unnormalizable = np.zeros(n_rows, dtype=bool)
    for block in blocks:
        if block is None:
            continue
        block_finite, block_directed, _ = _finite_and_directed(block)
        finite &= block_finite
        # ``~block_directed`` also covers the rows non-finite in THIS block, which
        # the finiteness test has already claimed. The AND with ``finite`` below
        # drops them again, which is what keeps the two causes disjoint.
        unnormalizable |= ~block_directed
    zero_norm = finite & unnormalizable
    return finite & ~zero_norm, zero_norm


def _weighted_unit_blocks(
    blocks: list[npt.NDArray[np.float32] | None],
    block_weights: tuple[float, ...],
    keep: npt.NDArray[np.bool_],
) -> list[npt.NDArray[np.float32]]:
    """Normalize and scale every block, restricted to the kept rows.

    Only kept rows are divided, so the zero-norm rows classification already
    rejected never reach a division at all.
    """
    n_kept = int(keep.sum())
    scaled: list[npt.NDArray[np.float32]] = []
    for width, weight, block in zip(BLOCK_DIMS, block_weights, blocks, strict=True):
        if block is None:
            # A plain zero block, never sqrt(0) * NaN: the modality leaves the
            # metric without its (possibly unusable) values reaching the vector.
            scaled.append(np.zeros((n_kept, width), dtype=np.float32))
            continue
        kept = block[keep]
        # Accumulate wide, divide wide, STORE narrow. Narrowing the norm first
        # would undo the wide accumulation: a norm past the float32 ceiling casts
        # to inf and zeros a row that keep still selects, emitting a non-unit
        # fused row. Writing into a float32 out is what keeps that promotion from
        # materializing a float64 copy of a corpus-scale block - the ufunc buffers
        # it in fixed-size chunks instead.
        unit = np.empty_like(kept)
        np.divide(kept, _row_norms(kept)[:, None], out=unit)
        unit *= np.sqrt(np.float32(weight))
        scaled.append(unit)
    return scaled


@attrs.frozen(eq=False)
class WorkingVectors:
    """One batch's fused vectors and the two per-row masks that explain the rest.

    ``eq=False`` because it holds ndarrays.

    Attributes:
        keep: Per input row, whether the row has a usable fused vector.
        working: ``(keep.sum(), FUSED_DIM)`` float32 in input order, aligned to
            the rows ``keep`` selects.
        zero_norm: Per input row, whether the row was dropped for having no
            direction. DISJOINT from ``keep``, and the two together partition the
            causes: a dropped row is non-finite exactly when
            ``~keep & ~zero_norm``.

    """

    keep: npt.NDArray[np.bool_]
    working: npt.NDArray[np.float32]
    zero_norm: npt.NDArray[np.bool_]


def working_vectors(batch: pa.Table, weights: Mapping[str, float]) -> WorkingVectors:
    """Build the weighted unit vector of every usable row in ``batch``.

    Args:
        batch: One scanned batch carrying each weighted block's primary vector.
        weights: Weight per ``FUSED_BLOCKS`` field; ``CurateConfig`` owns sum-to-1.

    Returns:
        The fused rows and their two masks; see ``WorkingVectors``.

    Raises:
        KeyError: If a fused block has no entry in ``weights``.
        ValueError: If a weighted block is stored at a width this run cannot fuse.

    """
    # Rows arrive already filtered on IS NOT NULL, so the only verdict left to
    # reach is usable or invalid_embedding: a missing vector is never seen here
    # and therefore needs no reason of its own (see columns.py).
    #
    # The masks rather than their counts, because the caller needs more than the
    # tally: the cause decides WHICH bypass sentinel a dropped row is routed on,
    # and a count cannot say which row it counted.
    block_weights = _block_weights(weights)
    blocks = _decode_blocks(batch, block_weights)
    keep, zero_norm = _classify(blocks, batch.num_rows)
    working = np.concatenate(_weighted_unit_blocks(blocks, block_weights, keep), axis=1)
    return WorkingVectors(keep=keep, working=working, zero_norm=zero_norm)


@attrs.frozen(eq=False)
class CentroidAssigner:
    """Score one block of unit-norm row vectors against a centroid basis.

    Width-agnostic on purpose, and used at two widths: the fused basis at
    ``FUSED_DIM`` for locality, and the subtask text basis at ``TEXT_DIM`` for the
    level-2 fairness group. The kernel is the same because the contract is - unit
    rows, so ``argmax`` of the dot product IS the nearest centroid - and the two
    bases are kept apart by which vectors are handed to which assigner, never by
    the assigner knowing its purpose.

    ``eq=False`` because it holds an ndarray. Construct via ``from_raw`` so the
    raw centroid array the fit produces is unit-normalized exactly once; that
    normalization is a correctness gate, not a scaling nicety (see ``from_raw``).

    Attributes:
        centroids_unit: ``(k, d)`` unit-normalized centroids.

    """

    centroids_unit: npt.NDArray[np.float32]

    @classmethod
    def from_raw(cls, raw: npt.NDArray[np.float32]) -> "CentroidAssigner":
        """Build an assigner from the RAW centroid array the fit produces.

        Unit-normalization here is load-bearing: working vectors are unit-norm, so
        ``X @ C_unit.T`` IS cosine similarity and its argmax IS the nearest
        centroid. Handed raw centroids the kernel returns the WRONG cluster - not
        merely a mis-scaled distance - because a longer centroid wins the dot
        product without being nearer in angle.

        Args:
            raw: ``(k, d)`` raw (non-unit) centroids from the k-means fit.

        Returns:
            An assigner holding the unit-normalized basis.

        Raises:
            ValueError: If ``raw`` is not a non-empty ``(k, d)`` matrix, any
                centroid row holds a non-finite value, or has ~zero norm; either
                degenerate case would otherwise fail opaquely in ``score``.

        """
        raw_f32 = np.asarray(raw, dtype=np.float32)
        if raw_f32.ndim != 2 or raw_f32.shape[0] == 0:  # noqa: PLR2004 -- matrix rank is always 2
            msg = f"centroids must be a non-empty (k, d) matrix, got shape {raw_f32.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(raw_f32)):
            bad = np.argwhere(~np.isfinite(raw_f32))
            row = int(bad[0, 0])
            col = int(bad[0, 1])
            msg = f"centroid row {row} has a non-finite value at column {col}; the fitted basis is degenerate"
            raise ValueError(msg)
        norms = _row_norms(raw_f32)
        if not np.all(np.isfinite(norms)):
            row = int(np.argwhere(~np.isfinite(norms))[0, 0])
            msg = f"centroid row {row} has a non-finite norm; the fitted basis is degenerate"
            raise ValueError(msg)
        if np.any(norms <= _MIN_CENTROID_NORM):
            worst = int(np.argmin(norms))
            msg = f"centroid row {worst} has ~zero norm ({float(norms[worst]):.3e}); the fitted basis is degenerate"
            raise ValueError(msg)
        # Divided at float64 and stored at float32, never narrowed first: a norm
        # past the float32 ceiling would cast to inf and zero the row, seating a
        # centroid that wins nothing in the basis the guards above just cleared.
        centroids_unit = np.empty_like(raw_f32)
        np.divide(raw_f32, norms[:, None], out=centroids_unit)
        return cls(centroids_unit=centroids_unit)

    def score(
        self,
        block: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """Assign every row in ``block`` to its nearest centroid.

        Args:
            block: ``(n, d)`` working vectors, already unit-norm. The caller bounds
                ``n``: one ``n x k`` score matrix is materialized per call and this
                kernel does not chunk.

        Returns:
            ``(cluster_id, distance_to_centroid)`` of shapes ``(n,)`` int32 and
            ``(n,)`` float32. Ties break to the lowest ``cluster_id``.

        """
        block_f32 = np.asarray(block, dtype=np.float32)
        # A (0, d) block already flows through the kernel below; the guard is what
        # additionally accepts a WIDTHLESS empty array, which the matmul rejects
        # with an opaque gufunc dimension error. An all-invalid batch therefore
        # never has to carry FUSED_DIM just to be scored.
        if block_f32.shape[0] == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        sims = block_f32 @ self.centroids_unit.T
        best = np.argmax(sims, axis=1)
        cluster_id = best.astype(np.int32)
        best_sim = np.clip(
            sims[np.arange(block_f32.shape[0]), best],
            np.float32(-1.0),
            np.float32(1.0),
        )
        cosine_dist = (np.float32(1.0) - best_sim).astype(np.float32, copy=False)
        return cluster_id, cosine_dist


def unit_rows(matrix: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Row-normalize ``matrix``; return the unit rows and the mask of rows that had a direction.

    A row that is non-finite or at or below ``_MIN_NORM`` is left as zeros rather
    than divided, so it never becomes NaN - which ``np.argmax`` would then treat
    as the winning centroid.

    Args:
        matrix: ``(n, d)`` raw vectors.

    Returns:
        ``(unit, usable)``: the normalized matrix, and the per-row mask of rows a
        centroid assignment is meaningful for.

    """
    block = np.asarray(matrix, dtype=np.float32)
    _, usable, norms = _finite_and_directed(block)
    unit = np.zeros_like(block)
    # The float64 norm divides as it was accumulated. Narrowed to float32 first, a
    # norm past the float32 ceiling becomes inf and zeros the row while ``usable``
    # still holds - and a zeroed-but-usable row reaches ``score`` as an all-zero
    # similarity, which argmax hands to centroid 0 rather than to ``absent_id``.
    np.divide(block, norms[:, None], out=unit, where=usable[:, None])
    return unit, usable


def assign_clusters(
    matrix: npt.NDArray[np.float32],
    present: npt.NDArray[np.bool_],
    assigner: CentroidAssigner,
    absent_id: int,
) -> npt.NDArray[np.int32]:
    """Assign every row of a RAW vector column to its nearest centroid cell.

    Unlike ``CentroidAssigner.score``, this takes raw rows and owns the
    normalization, because its input is a stored column rather than a working
    vector the fusion path already normalized. A row that is absent, non-finite,
    or directionless gets ``absent_id`` instead of a cell: it has no position in
    the space, so the nearest centroid to it is not a defined question.

    Args:
        matrix: ``(n, d)`` raw vectors; rows where ``present`` is False are not read.
        present: Per-row presence mask, from the column's own validity bitmap.
        assigner: The basis to score against; its width must match ``matrix``.
        absent_id: Cell id for a row with no usable vector.

    Returns:
        ``(n,)`` int32 cell ids, in input order.

    """
    n_rows = int(matrix.shape[0])
    result = np.full(n_rows, absent_id, dtype=np.int32)
    unit, usable = unit_rows(matrix)
    usable &= present
    if not bool(usable.any()):
        return result
    cells, _distance = assigner.score(unit[usable])
    result[usable] = cells
    return result
