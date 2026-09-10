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

"""The SemDeDup retention rule, and the one group stage that applies it.

Two layers, deliberately separable: the kernel is array math with no Arrow, Ray
or device in it, and the stage is the Arrow-in / Arrow-out transform the GPU
group stage runs. The split is what lets the rule - the highest-risk thing in
the leg - be pinned by ordinary CPU tests.

::

    retention_order        distance DESC, clip_id ASC, fragment_id ASC
        |
    max_earlier_similarity per row, max cosine to any STRICTLY EARLIER row
        |                  (tiled GEMM; numpy on CPU, cupy on the GPU)
    duplicate_mask         score > 1 - eps
        |
    mark_duplicates        one group -> same rows, vector dropped, reason set,
        |                  score EMITTED (NULL where no GEMM ran) for the report
        |                  fold; derives the tile, refuses what no tile can fit
    dedup_group            the named UDF: imports cupy, calls the above
    dedup_launch_args      how that UDF must be launched (GPU + pixi env)

THE RETENTION RULE. A row is a duplicate exactly when its maximum cosine
similarity to any strictly earlier row in retention order - **including earlier
rows already flagged as duplicates** - exceeds ``1 - eps``. Drawing the maximum
over ALL earlier rows rather than over survivors only is the load-bearing
detail, and it is not a stylistic choice between equivalent formulations:

- Restricting the candidates to survivors turns this into greedy
  maximal-independent-set, which UNDER-de-duplicates. Given a chain a-b-c where
  ``b`` is near ``a`` and ``c`` is near ``b`` but far from ``a``, the correct
  rule drops ``c`` because its nearest earlier row is the flagged ``b``;
  survivor-only candidacy compares ``c`` against ``a`` alone and keeps it.
- Collapsing the similarity graph into connected components instead
  OVER-de-duplicates, because it keeps one row per component however wide the
  component is. Three rows where the last one in order sits near both of the
  first two form a single component, yet the rule here keeps both of the first
  two: each was scored against its own earlier rows and neither had one nearby.

Retention order is farthest-from-centroid first, so a cluster's survivor is its
atypical, information-rich row rather than the one nearest its centroid, with
``clip_id`` ascending and then ``fragment_id`` ascending as tie-breaks so an
exact-distance pair resolves the same way on every re-run and under any input
row order.

WHAT ``eps`` MEANS HERE. The similarity is the FUSED cosine, and the fused
vector is unit-norm with weights summing to 1, which makes
``1 - cos(a, b) = sum_m w_m (1 - cos_m(a, b))`` exact. The duplicate criterion
is therefore a weighted sum of per-modality distances falling under ``eps``, so
``eps`` and ``ModalityWeights`` are chosen TOGETHER: at ``subtask=0.6`` and
``eps=0.01`` the subtask term alone must stay under 0.017, which makes the
effective rule a CONJUNCTION - a duplicate describes the same subtask AND looks
alike AND moves alike - rather than a blend that one very close modality could
carry on its own. Re-weighting without re-choosing ``eps`` silently moves the
duplicate boundary for every modality at once.

ONE CLUSTER MUST FIT ONE CARD. The stage loads a whole cluster onto one GPU, and
``k`` is chosen from a MEAN cluster size, so the largest cluster is a property of
the data rather than of the config. The group stage therefore derives its GEMM
tile from the cluster's row count against the device's total memory - see THE
DEVICE MEMORY MODEL below - shrinking the tile so a cluster larger than the
default tile can hold still runs, and REFUSING with an actionable message when
the cluster's vectors alone exceed the card, which no tile choice can fix.

See docs/curator/design/curator-next-curation.md.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from cosmos_curator.next.embeddings.schemas import KEY_COLUMN
from cosmos_curator.next.recipes.curation.columns import (
    CURATE_SELECTION_REASON,
    DEDUP_KEY_COLUMN,
    DEDUP_SCORE_COLUMN,
    DISTANCE_COLUMN,
    FRAGMENT_COLUMN,
    WORKING_VECTOR_COLUMN,
    CurateReason,
)
from cosmos_curator.next.recipes.curation.vectors import vector_column_to_matrix

# Rows scored per GEMM tile. Bounds the ``tile x n`` similarity block so a large
# cluster never materializes the whole ``n x n`` matrix.
#
# A CEILING, not a setting: ``_group_tile_rows`` derives the tile per group and
# may return less, never more. Capping at the value the leg has always used is
# load-bearing rather than cautious - every group that fits at this tile keeps it
# and is therefore bitwise unchanged, so the derivation only ever changes the
# arithmetic of groups that previously could not run at all.
#
# A memory knob, and verdict-invariant rather than bit-invariant: the tile is the
# GEMM's row count, so BLAS selects a different microkernel for a small block
# than for a large one and the same pair's cosine can differ by a float32 ulp
# between tile sizes (measured at exactly 1 ulp for a 40-row group under Apple
# Accelerate). That moves no verdict unless a row's score sits within an ulp of
# ``1 - eps``, which the pinning test states as the contract - scores equal to
# within float32 rounding, duplicate mask identical.
_DEDUP_TILE_ROWS_CAP = 4096

# THE DEVICE MEMORY MODEL, read off the allocation sites in ``_unit_normalize``
# and ``_tiled_max_similarity``. For a group of ``m`` rows at fused width ``d``:
#
#   normalize peak   m * (2 * 4d + 12)          the unit copy alongside its input
#   loop peak        m * (9 * tile + 4d + 12)   three (tile, m) blocks at once
#
# ::
#
#     (m, d) float32 vectors  ---------------> 4dm, held for the whole call
#         |
#         +-- _unit_normalize: result allocated while its input is still
#         |   referenced, so the vectors cost DOUBLE for that call     8dm
#         |
#         +-- per tile: sims (4) | earlier mask (1) | xp.where (4)  9 * tile * m
#
# The 9 is per CELL of one (tile, m) block, and it is the loop's high-water at
# TWO different instants rather than one - which is why the loop body must not
# gain a fourth block:
#
#   the xp.where line     4 float32 sims | 1 boolean mask | 4 the where result
#   the next iteration    4 the PREVIOUS sims | 1 the previous mask | 4 the new
#                         product - because Python holds both bindings until the
#                         assignment that replaces them completes
#
# The second instant is the reason ``sims`` is not cast: an ``astype`` on an
# already-float32 product adds a fourth live block there, measured at 13 bytes
# per cell against 9 without it.
#
# The 12 is per ROW of the two whole-group index arrays - 4 bytes of float32
# ``max_sim`` plus 8 of the int64 ``cols``. The doubling in the normalize term is
# the reason the ceiling is what it is: shrinking the tile drives the loop term
# toward zero, which leaves the NORMALIZE peak binding, and that peak is two
# copies of the vectors rather than one.
#
# The model is written against total-and-reserved rather than free memory, and
# the reserve is what absorbs an allocator that holds more than the live bytes:
# cuPy's pool does not return freed blocks to the driver, so a peak reached once
# stays reserved for the rest of the call.
#
# HOST memory is a separate cost and is NOT bounded here: the caller holds the
# Arrow-decoded matrix and its retention-order gather at once, so a group also
# costs about ``2 * 4dm`` bytes of node RAM. Nothing refuses on that - at the
# device ceiling it is ~77 GB, so a node sized for one cluster per GPU is the
# operator's assumption to check.
_FLOAT32_BYTES = 4
_SIM_BLOCK_BYTES_PER_CELL = 9
_GROUP_INDEX_BYTES_PER_ROW = 12
_NORMALIZE_LIVE_COPIES = 2

# Fraction of TOTAL device memory left to the CUDA context, cuBLAS workspaces and
# pool fragmentation - costs the analytic model above cannot see. A fraction
# rather than a fixed subtraction because fragmentation scales with the pool
# while the context does not, and 10% covers both on every card the leg targets
# (~8 GiB on an 80 GiB device against a context of well under 1 GiB). It is not
# free, but it is cheap: it moves the row count at which the tile starts
# shrinking from ~2.1M to ~1.9M, still roughly 9.6x the default mean cluster, so
# no realistic run pays for the reserve with a smaller tile.
_DEVICE_RESERVE_FRACTION = 0.10

# A whole GPU per group: each cluster's tiled cosine GEMM wants a card to itself.
# A scheduling reservation, not a numeric knob.
_DEDUP_TASK_GPUS = 1

# The stage reads and returns pyarrow tables, so it must be handed them; under
# the default batch format the UDF would receive pandas and every column access
# below would fail.
_DEDUP_BATCH_FORMAT = "pyarrow"

# What one group must carry to be scorable. Presence is checked up front so a
# producer mistake is reported against the missing name, rather than surfacing as
# a KeyError from inside the tiled GEMM.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    KEY_COLUMN,
    FRAGMENT_COLUMN,
    DEDUP_KEY_COLUMN,
    CURATE_SELECTION_REASON,
    DISTANCE_COLUMN,
    WORKING_VECTOR_COLUMN,
)


def retention_order(
    clip_ids: Sequence[str],
    fragment_ids: Sequence[int],
    distance_to_centroid: npt.NDArray[np.float32],
) -> npt.NDArray[np.intp]:
    """Return the retention order: distance DESC, ``clip_id`` ASC, ``fragment_id`` ASC.

    The rule keeps the FIRST occurrence in this order, so putting the
    farthest-from-centroid row first makes the atypical row the survivor and the
    row nearest the centroid the one dropped. The ``clip_id`` and ``fragment_id``
    tie-breaks make an exact-distance pair deterministic, independent of the
    order the rows happened to be read in.

    Args:
        clip_ids: The group's clip ids, in any order, parallel to
            ``fragment_ids`` and ``distance_to_centroid``.
        fragment_ids: The group's fragment ids, parallel to ``clip_ids``.
        distance_to_centroid: Fused cosine distance to the centroid.

    Returns:
        A permutation of ``range(len(clip_ids))`` in retention order.

    Raises:
        ValueError: If the inputs have different lengths, which would pair each
            id with another row's distance and order the group by nothing.

    """
    distances = np.asarray(distance_to_centroid, dtype=np.float64)
    if len(clip_ids) != distances.shape[0] or len(fragment_ids) != distances.shape[0]:
        msg = (
            "clip_ids, fragment_ids and distance_to_centroid must be parallel, "
            f"got {len(clip_ids)}, {len(fragment_ids)} and {distances.shape[0]}"
        )
        raise ValueError(msg)
    if len(clip_ids) == 0:
        return np.empty(0, dtype=np.intp)
    # np.lexsort's LAST key is primary: primary = -distance (higher distance
    # sorts first), then clip_id ascending, then fragment_id ascending.
    order = np.lexsort(
        (
            np.asarray(fragment_ids, dtype=np.int64),
            np.asarray(clip_ids, dtype=np.str_),
            -distances,
        )
    )
    return np.asarray(order, dtype=np.intp)


def _unit_normalize(vectors: Any, *, xp: Any) -> Any:  # noqa: ANN401 - xp is numpy or cupy
    """Row-normalize ``vectors`` to unit L2 norm, mapping a zero row to zero.

    Fused vectors arrive unit-norm by construction, so this is defensive: a zero
    row maps to the zero vector, whose cosine to anything is 0, keeping it
    non-duplicate instead of producing ``NaN`` across its whole group.
    """
    norms = xp.sqrt(xp.sum(vectors * vectors, axis=1, keepdims=True))
    safe = xp.where(norms > 0, norms, xp.asarray(1.0, dtype=vectors.dtype))
    return vectors / safe


def _to_host(array: Any, *, xp: Any) -> npt.NDArray[Any]:  # noqa: ANN401 - xp is numpy or cupy
    """Return a host ``numpy`` view of ``array`` (``cupy.asnumpy`` on the GPU)."""
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(array)  # type: ignore[no-any-return]
    return np.asarray(array)


def _tiled_max_similarity(unit: Any, *, xp: Any, tile: int) -> Any:  # noqa: ANN401 - xp is numpy or cupy
    """For unit-norm rows already in retention order, return each row's max earlier cosine.

    Entry ``j`` is the maximum cosine similarity of row ``j`` to any strictly
    earlier row ``i < j``. Row 0 has no earlier neighbour and is forced to
    ``0.0`` so a group's representative is never its own duplicate.

    Computed in row tiles, so the ``n x n`` similarity matrix is never
    materialized whole: the ``xp.where`` line holds three ``(tile, n)`` blocks at
    once, which is the loop term of THE DEVICE MEMORY MODEL in the module header.
    ``xp`` is ``numpy`` on the CPU and ``cupy`` on the GPU; the arithmetic is
    identical.
    """
    if tile < 1:
        msg = f"tile must be >= 1, got {tile}"
        raise ValueError(msg)
    n = int(unit.shape[0])
    max_sim = xp.zeros(n, dtype=xp.float32)
    cols = xp.arange(n)
    neg_inf = xp.asarray(float("-inf"), dtype=xp.float32)
    zero = xp.asarray(0.0, dtype=xp.float32)
    for start in range(0, n, tile):
        stop = min(start + tile, n)
        block = unit[start:stop]  # (b, d)
        # Already float32: unit rows are float32 / float32. Casting here would
        # copy the whole block for nothing and cost a fourth live (b, n) array at
        # the moment the next iteration allocates - see THE DEVICE MEMORY MODEL.
        sims = block @ unit.T  # (b, n) cosine (unit rows)
        global_rows = xp.arange(start, stop)[:, None]  # (b, 1) global row index
        earlier = cols[None, :] < global_rows  # strictly-earlier mask
        block_max = xp.max(xp.where(earlier, sims, neg_inf), axis=1)
        # Row 0 (and only row 0) has an all-masked block row, whose max is -inf.
        has_earlier = global_rows[:, 0] > 0
        max_sim[start:stop] = xp.where(has_earlier, block_max, zero)
    return max_sim


def max_earlier_similarity(  # noqa: PLR0913 - parallel id columns are load-bearing
    clip_ids: Sequence[str],
    fragment_ids: Sequence[int],
    embeddings: npt.NDArray[np.float32],
    distance_to_centroid: npt.NDArray[np.float32],
    *,
    xp: Any = np,  # noqa: ANN401 - xp is numpy or cupy
    tile: int = _DEDUP_TILE_ROWS_CAP,
) -> npt.NDArray[np.float32]:
    """Score one group: each row's max cosine to any strictly earlier row, IN INPUT ORDER.

    "Earlier" is retention order, duplicates included; scores return aligned to the input rows.

    Args:
        clip_ids: The group's clip ids, in any order.
        fragment_ids: The group's fragment ids, parallel to ``clip_ids``.
        embeddings: ``(n, d)`` fused vectors parallel to ``clip_ids``.
        distance_to_centroid: ``(n,)`` retention-order key, parallel to the ids.
        xp: Array module - ``numpy`` on the CPU and in tests, ``cupy`` in the UDF.
        tile: Rows scored per GEMM tile; a memory knob, verdict-invariant. Defaults
            to the ceiling, which is what a group small enough to fit it gets;
            ``mark_duplicates`` derives a smaller one for a group that does not.

    Raises:
        ValueError: If ``embeddings`` is not an ``(n, d)`` matrix, or ``tile`` < 1.

    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    n = len(clip_ids)
    if vectors.ndim != 2 or vectors.shape[0] != n:  # noqa: PLR2004 - a matrix has 2 dimensions
        msg = f"embeddings must be an (n, d) matrix with n == {n} rows, got shape {vectors.shape}"
        raise ValueError(msg)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    # Every earlier row is a candidate, including ones whose own score already marks
    # them duplicates: see THE RETENTION RULE in the module docstring for why that IS
    # the rule rather than one formulation of it.
    order = retention_order(clip_ids, fragment_ids, distance_to_centroid)
    unit = _unit_normalize(xp.asarray(vectors[order], dtype=xp.float32), xp=xp)
    ordered_scores = np.asarray(_to_host(_tiled_max_similarity(unit, xp=xp, tile=tile), xp=xp), dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)
    scores[order] = ordered_scores
    return scores


def duplicate_mask(scores: npt.NDArray[np.float32], eps: float) -> npt.NDArray[np.bool_]:
    """Return which rows are duplicates: those scoring STRICTLY above ``1 - eps``.

    The comparison is done in float32 against float32 scores, so a row sitting
    exactly on the boundary survives and cannot be judged differently by a second
    reader working in float64.

    Args:
        scores: Max-earlier-similarity scores from ``max_earlier_similarity``.
        eps: The duplicate threshold; ``CurateConfig.dedup_eps``.

    Returns:
        A boolean mask parallel to ``scores``; True means duplicate.

    """
    threshold = np.float32(1.0 - eps)
    return np.asarray(scores, dtype=np.float32) > threshold


def _require_columns(group: pa.Table) -> None:
    """Reject a group missing a column this stage reads, or a non-string reason.

    Raises:
        ValueError: Naming the missing columns, or the reason column's actual
            type. The reason column is written in place here, so its type is part
            of the contract rather than something to coerce: widening it would
            hand the write a column it cannot store.

    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in group.schema.names]
    if missing:
        msg = f"dedup group is missing required column(s) {missing}; has {group.schema.names}"
        raise ValueError(msg)
    reason_type = group.schema.field(CURATE_SELECTION_REASON).type
    if not pa.types.is_string(reason_type):
        msg = f"{CURATE_SELECTION_REASON} must be a string column, got {reason_type}"
        raise ValueError(msg)


def _one_dedup_key(group: pa.Table) -> int:
    """Return the group's single ``__dedup_key``.

    Guaranteed by grouping on that key, checked anyway because a mixed group
    fails SILENTLY: rows from two clusters would be scored against each other,
    producing duplicate verdicts that no cluster boundary justifies.

    Raises:
        ValueError: If the group names anything other than exactly one key.

    """
    keys = pc.unique(group.column(DEDUP_KEY_COLUMN))  # type: ignore[attr-defined]
    if len(keys) != 1:
        msg = f"one dedup group must name exactly one {DEDUP_KEY_COLUMN}, got {keys.to_pylist()}"
        raise ValueError(msg)
    return int(keys[0].as_py())


def _scored_vectors(column: pa.ChunkedArray | pa.Array) -> npt.NDArray[np.float32]:
    """Return the ``(n, d)`` float32 matrix a scored group's GEMM runs on.

    Decoding - including the slice-offset rebase a batched read makes necessary -
    belongs to ``vectors``, which owns the fused vector's layout. What is added
    here is the one condition specific to SCORING: every row must actually have a
    vector, because the matrix is positional and a row without one has nothing to
    be scored against.

    Raises:
        ValueError: If the column carries a NULL. Such a row belongs in the
            negative-key bypass, never in a scored group.
        TypeError: If the column is not a fixed-size list, raised by the decoder.

    """
    if column.null_count:
        msg = f"{WORKING_VECTOR_COLUMN} carries {column.null_count} NULL(s) inside a scored dedup group"
        raise ValueError(msg)
    return vector_column_to_matrix(column)


def _device_total_bytes(xp: Any) -> int | None:  # noqa: ANN401 - xp is numpy or cupy
    """Return the total memory of the device ``xp`` allocates on, or None for host numpy.

    TOTAL rather than free, deliberately. Free memory depends on whatever else the
    card is doing, so a tile derived from it would differ between two runs of the
    same job - and with it the ~1-ulp cross-tile cosine drift, turning a
    documented per-GPU-model property into a per-run one. Keyed to total, a given
    cluster on a given card always scores identically.
    """
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return None
    return int(cuda.Device().mem_info[1])


def _vector_bytes_per_row(width: int) -> int:
    """Return the per-row bytes held for a whole scoring call at fused ``width``."""
    return width * _FLOAT32_BYTES + _GROUP_INDEX_BYTES_PER_ROW


def _device_budget_bytes(device_total_bytes: int) -> int:
    """Return the bytes of a device this stage will actually allocate against."""
    return int(device_total_bytes * (1.0 - _DEVICE_RESERVE_FRACTION))


def max_group_rows(*, width: int, device_total_bytes: int) -> int:
    """Return the rows of a ``width``-wide group one device can score at ANY tile size.

    The bound is the peak at ``tile = 1``, the smallest block the loop can take:
    below it the only remaining lever is gone, so this is the row count above
    which the stage must refuse instead of shrinking further. Which of the two
    terms binds depends on the width - the normalize copy at the fused 865, the
    similarity block at the narrow widths tests use - so the model takes both.

    Public because the driver evaluates the same bound: at ``k == 1`` a run's
    single cluster holds every eligible row, and the cluster-count warning says so
    against this number. The driver has no device to ask, so it passes an assumed
    total; the refusal computed here against the real card stays the binding check.
    """
    vector_bytes = width * _FLOAT32_BYTES
    per_row = (
        max(_NORMALIZE_LIVE_COPIES * vector_bytes, _SIM_BLOCK_BYTES_PER_CELL + vector_bytes)
        + _GROUP_INDEX_BYTES_PER_ROW
    )
    return _device_budget_bytes(device_total_bytes) // per_row


def _floor_to_power_of_two(value: int) -> int:
    """Return the largest power of two at or below ``value``, which must be >= 1."""
    return 1 << (value.bit_length() - 1)


def _group_tile_rows(*, rows: int, width: int, dedup_key: int, device_total_bytes: int | None) -> int:
    """Return the GEMM tile for one group, or refuse a group no tile size can fit.

    Args:
        rows: The group's row count, which must be >= 1; a group with no rows is never scored.
        width: Fused vector width.
        dedup_key: The group's cluster id, named in the refusal so an operator
            reading it knows which cluster to look at.
        device_total_bytes: Total device memory, or None on a host array module,
            where there is no device budget to derive against and the cap stands.

    Returns:
        Rows per GEMM tile: at most ``_DEDUP_TILE_ROWS_CAP``, so a group that fits
        at the cap is scored exactly as it was before the tile became derived, and
        at least one, because ``max_group_rows`` charges a tile-of-one block
        against the same budget that decides the refusal.

    Raises:
        ValueError: If the group cannot be scored at any tile size - the stage
            refuses rather than truncating the group. The message is
            self-contained - row count, device total, feasible maximum, and the
            remedies - because this raise happens inside a Ray UDF, where the
            exception type is erased and only the message reaches the operator.

    """
    if device_total_bytes is None:
        return _DEDUP_TILE_ROWS_CAP
    budget_bytes = _device_budget_bytes(device_total_bytes)
    feasible_rows = max_group_rows(width=width, device_total_bytes=device_total_bytes)
    if rows > feasible_rows:
        msg = (
            f"curate dedup: {DEDUP_KEY_COLUMN}={dedup_key} holds {rows} rows, which no GEMM tile size "
            f"can fit: a {device_total_bytes / 1024**3:.1f} GiB device holds at most {feasible_rows} "
            f"rows of width {width}, because a cluster's vectors are held twice while they are "
            f"normalized and {_DEVICE_RESERVE_FRACTION:.0%} of the card is reserved for the CUDA "
            f"context and allocator fragmentation. "
            f"Lower target_mean_cluster_rows to raise k and split the corpus more finely, "
            f"or widen the fit sample so the basis is not concentrating rows into few clusters. "
            f"Splitting this cluster is not a remedy: retention is the maximum similarity over all "
            f"strictly earlier rows WITHIN one cluster, so a split changes verdicts."
        )
        raise ValueError(msg)
    spare_bytes = budget_bytes - _vector_bytes_per_row(width) * rows
    # Quantized down to a power of two for the same reason the budget is total
    # rather than free memory: the tile decides the GEMM's blocking and therefore
    # the last bit of a cosine, so it must be a function of the card and the
    # cluster, not of a byte count that drifts between two runs of the same job.
    return min(_DEDUP_TILE_ROWS_CAP, _floor_to_power_of_two(spare_bytes // (_SIM_BLOCK_BYTES_PER_CELL * rows)))


def with_scores(table: pa.Table, scores: npt.NDArray[np.float32] | None) -> pa.Table:
    """Append the transient retention-score column, all-NULL when no GEMM ran.

    One owner for the column's name, type and position, because it has two
    producers: this stage on the scored path, and the projection the pipeline
    substitutes for this stage when ``dedup_eps`` is unset. Two producers that
    disagreed on any of the three would hand the report fold a block it cannot
    read, and the bypass path here would break block concatenation outright.

    Args:
        table: Rows already shed of the working vector.
        scores: Per-row max-earlier similarity, or None for rows never compared.

    """
    values = pa.nulls(table.num_rows, type=pa.float32()) if scores is None else pa.array(scores, type=pa.float32())
    return table.append_column(pa.field(DEDUP_SCORE_COLUMN, pa.float32(), nullable=True), values)


def mark_duplicates(
    group: pa.Table,
    *,
    eps: float,
    xp: Any = np,  # noqa: ANN401 - xp is numpy or cupy
) -> pa.Table:
    """Apply the retention rule to one cluster; return the same rows without the vector.

    Cardinality- and order-preserving: one row out per row in, in arrival order, so ``__frag``,
    the fairness group keys and the distance ride through. It drops the vector, writes the
    reason - ``duplicate`` on rejected rows, as it arrived on the rest - and appends each
    row's score.

    Args:
        group: One cluster's rows, carrying the verdict columns, distance and vector.
        eps: The duplicate threshold; ``CurateConfig.dedup_eps``.
        xp: Array module for the retention GEMM - ``numpy`` or ``cupy``.

    Raises:
        ValueError: A missing column, a non-string reason column, more than one key, a NULL vector,
            or a cluster too large for the device at any tile size (``_group_tile_rows``).
        TypeError: The vector column is not a fixed-size list.

    """
    _require_columns(group)
    without_vector = group.drop_columns([WORKING_VECTOR_COLUMN])
    if group.num_rows == 0:
        logger.info(f"curate dedup: empty group, {DEDUP_KEY_COLUMN} unresolved")
        return with_scores(without_vector, None)
    dedup_key = _one_dedup_key(group)
    # The only instrument for the cluster-size tail SHORT of the refusal below: k
    # is chosen from a mean, so a run's largest cluster - the one that sizes the
    # GEMM's peak memory - is unknown until these lines are read back.
    logger.info(f"curate dedup: {DEDUP_KEY_COLUMN}={dedup_key} rows={group.num_rows}")
    # The bypass for rows the scan already judged invalid_embedding: no GEMM, so an
    # unusable vector cannot propagate NaN. Negativity rather than equality against
    # either sentinel, because both causes have the same fate here. It still sheds
    # the vector and still emits the score - NULL, since nothing was compared -
    # because two groups of one stage that disagreed on their schema cannot
    # concatenate.
    if dedup_key < 0:
        return with_scores(without_vector, None)
    matrix = _scored_vectors(group.column(WORKING_VECTOR_COLUMN))
    scores = max_earlier_similarity(
        group.column(KEY_COLUMN).to_pylist(),
        group.column(FRAGMENT_COLUMN).to_pylist(),
        matrix,
        np.asarray(group.column(DISTANCE_COLUMN).to_numpy(zero_copy_only=False), dtype=np.float32),
        xp=xp,
        tile=_group_tile_rows(
            rows=int(matrix.shape[0]),
            width=int(matrix.shape[1]),
            dedup_key=dedup_key,
            device_total_bytes=_device_total_bytes(xp),
        ),
    )
    reason = pc.if_else(  # type: ignore[attr-defined]
        pa.array(duplicate_mask(scores, eps)),
        pa.scalar(CurateReason.DUPLICATE.value, type=pa.string()),
        without_vector.column(CURATE_SELECTION_REASON),
    )
    index = without_vector.schema.get_field_index(CURATE_SELECTION_REASON)
    return with_scores(without_vector.set_column(index, without_vector.schema.field(index), reason), scores)


def dedup_group(group: pa.Table, *, eps: float) -> pa.Table:
    """Score one cluster on the GPU. The named UDF the retention stage runs.

    A module-level function rather than a lambda or a ``partial`` because
    ``map_groups`` reads ``__name__`` off the callable it is given and fails on
    one that has none. ``cupy`` is imported here so this module keeps importing on
    a CPU-only host, where the kernel above is fully testable.

    Args:
        group: One cluster's rows, keyed on ``__dedup_key``.
        eps: The duplicate threshold; ``CurateConfig.dedup_eps``.

    Returns:
        The group's rows with the vector dropped, duplicates marked and each row's
        score appended; see ``mark_duplicates``.

    """
    import cupy  # type: ignore[import-not-found]  # noqa: PLC0415 - deferred so this module imports GPU-free

    return mark_duplicates(group, eps=eps, xp=cupy)


def dedup_launch_args(*, gpu_env_name: str, concurrency: int | None) -> dict[str, Any]:
    """Return the ``map_groups`` arguments ``dedup_group`` must be launched with.

    Bundled with the UDF instead of spelled out at the call site so the
    ``runtime_env`` cannot be forgotten. ``cupy`` lives in the pixi environment
    named here and not in the driver's, so a task without this argument imports
    it only when the driver happens to have been launched inside that same
    environment - which is a latent failure, not a missing feature.

    Args:
        gpu_env_name: The pixi environment holding cuPy. Taken from the caller,
            which already owns that name, so it is not restated here.
        concurrency: Cap on simultaneous GPU tasks, or None to let the scheduler
            use every free GPU. A scheduling knob only: the retention result is
            invariant to it.

    Returns:
        Keyword arguments for ``Dataset.groupby(...).map_groups``.

    """
    # Deferred: this module is imported by CPU-only tests of the kernel above,
    # and the runtime-env helper imports Ray at module scope.
    from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env  # noqa: PLC0415

    return {
        "batch_format": _DEDUP_BATCH_FORMAT,
        "num_gpus": _DEDUP_TASK_GPUS,
        "concurrency": concurrency,
        "runtime_env": ray_data_gpu_runtime_env(gpu_env_name),
    }
