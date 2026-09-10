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

"""The Curate leg's column contract: what it persists, and what it moves in flight.

Curate is wide-table in, wide-table out. It reads ``clips.lance`` and writes two
nullable columns back onto the same rows, so its whole data model is statable
here: two persisted names and types, the five reason values one of them may
carry, the three embedding blocks whose vectors decide a row's verdict, the seven
transient names stages route rows by, and every answer to "which columns does
this run read" - the eligibility predicate among them.

::

    one clips.lance row
      |
      +-- source fields        task_name / subtask_name / ... (robot_action_split)
      |
      +-- embedding_* fields   FUSED_BLOCKS names the three Curate fuses, each
      |                        block being its group's primary_vector
      |
      +-- curate_* fields      CURATE_COLUMNS - owned here, written by one
                               LanceOperation.Update

Declarations plus pure derivations over them, and nothing else: no Lance, Ray,
cuML or cuPy import. The pure kernels (vectors, dedup, fairness) sit above this
module and must import on a CPU-only host with those four poisoned, which they
cannot do if the contract they share drags a driver dependency in with it.

The read set has ONE owner here, because the leg asks the question in four
shapes - a required-column list, a scanner projection, an eligibility predicate,
and a producer-attribution probe - and an answer that drifted between them would
either read a column nothing attributes or attribute one nothing reads::

    weights
      |
      v
    _weighted_blocks          the only weight-filtered view of FUSED_BLOCKS
      |
      +--> weighted_vectors   the FUSED set: fit projection, scan, eligibility
      |
      +--> consumed_groups    weighted groups + text, which is always read
             |
             +--> consumed_vectors    per group; the text asymmetry lives HERE
                    |
                    +--> consumed_columns   the union: what preflight requires
                                            and the producer gate must attribute

There is deliberately no ``missing_embedding`` reason. A row whose required
vector is NULL is never claimed by a run, and ``embedding_action IS NULL``
already answers "why is this row uncurated", so a stored value restating a
queryable predicate would be a mirror field. ``invalid_embedding`` earns a value
for the opposite reason: discovering it means reading the vector and testing it,
which no predicate can do.

See docs/curator/design/curator-next-curation.md.
"""

import enum
from collections.abc import Mapping
from typing import Literal

import pyarrow as pa

from cosmos_curator.next.embeddings.schemas import (
    EMBEDDING_COLUMN_GROUPS,
    KEY_COLUMN,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)


class CurateReason(enum.StrEnum):
    """Why a row landed in - or out of - the curated subset.

    The five string values are a persisted format contract: they are written to
    ``curate_selection_reason`` and read by every downstream export, whose query
    is ``WHERE curate_selection_reason = 'selected'``. NULL is not a member and
    never will be, because it means the run did not claim the row at all - see
    the module docstring for why that state needs no value of its own.
    """

    SELECTED = "selected"
    DUPLICATE = "duplicate"
    BELOW_QUOTA = "below_quota"
    UNFUNDED = "unfunded"
    INVALID_EMBEDDING = "invalid_embedding"


WithinGroupOrder = Literal["farthest", "nearest", "neutral"]
"""Which survivors win inside a funded fairness group at a fixed target.

Result-defining, so two runs that differ here are not comparable: ``farthest``
keeps the atypical rows (the same ordering the retention rule already uses to
pick a cluster's representative), ``nearest`` inverts it, and ``neutral`` orders
by ``clip_id`` alone with no distance preference.

Declared with the columns rather than with the config so the config model and the
fairness kernel each reach it without an import edge to the other.
"""

CURATE_SELECTION_REASON = "curate_selection_reason"
CURATE_CLUSTER_ID = "curate_cluster_id"

CURATE_COLUMNS: pa.Schema = pa.schema(
    [
        pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True),
        pa.field(CURATE_CLUSTER_ID, pa.int32(), nullable=True),
    ]
)
"""The two columns Curate owns on ``clips.lance``, added in one metadata commit.

Both are nullable, which is what makes an all-nullable ``add_columns`` a
metadata-only widening: every pre-existing row reads NULL without a data file
being rewritten. They are atomic siblings - same pass, same commit.

A NULL means exactly one thing: the run that committed last did not claim this
row. The write is total, so every row of every fragment is written on every run,
and a row excluded by ``eligibility_filter()`` because a weighted embedding is
missing is written NULL rather than left holding an earlier run's verdict. The
one asymmetry between the two columns is that ``curate_cluster_id`` is also NULL
for a claimed ``invalid_embedding`` row, which was scored against no centroid.

``curate_cluster_id`` is a computational LOCALITY partition, never a semantic
category. It records which k-means cell a row was scored against so a
``duplicate`` verdict is interpretable; two rows sharing it are near each other
in the fused space and nothing more.

It is therefore NOT the same thing as ``SUBTASK_CLUSTER_COLUMN``, and the
distinction is the reason only one of the two is persisted. This one partitions
the FUSED metric, whose k is derived from the corpus size and which is measurably
not a task signal, so it can only ever mean locality. That one partitions the
subtask TEXT embedding at a fixed, operator-set k, and it is a fairness group.
Same algorithm, different space, different purpose, and neither is readable as
the other.
"""

CURATE_UPDATE_SCHEMA: pa.Schema = pa.schema([pa.field(KEY_COLUMN, pa.string(), nullable=False), *CURATE_COLUMNS])
"""The join key plus the two owned columns, in the order ``update_columns`` reads.

``update_columns`` joins on the ordinary persisted ``clip_id`` with left-outer
semantics: a fragment row absent from an update table keeps its previous value,
while one present carrying NULL is overwritten with NULL. The write relies on the
second half - it passes every stored row, blank where unclaimed - because keeping
a previous value is precisely what must not happen across runs.

The ``string`` key type is an ASSERTION about the producer, not a coercion:
``update_columns`` rejects a key-type mismatch outright (``Index column type
mismatch: expected Utf8, got LargeUtf8``), and it does so inside the write
worker, after the fit and the whole de-duplication pass. Preflight compares this
type against the table's own so the run fails on the driver in a second instead.
"""

# In-flight routing columns, never persisted and never added to the table. They
# sit beside the persisted contract because each is spelled by one stage and read
# by another: a producer and a consumer that disagreed about one of these names
# would hand each other a table it cannot read, so the name belongs to neither of
# them.
#
# __frag carries the id of the fragment a row was read from, stamped by the
# reader that owns that fragment, so the write can regroup verdicts by their
# physical home without deriving a fragment from a row address.
#
# __dedup_key carries the cluster a row is scored within, or one of the two
# negative sentinels below for a row that must reach the write WITHOUT entering
# the similarity pass. A sentinel is a routing value and not a cluster id, which
# is why it lives on a private column and becomes a NULL curate_cluster_id on the
# way to storage.
#
# __subtask_cluster carries the level-2 fairness group: which of
# CurateConfig.subtask_clusters cells the row's subtask text embedding was
# assigned to, or NO_SUBTASK_CLUSTER when the row carries no usable one.
FRAGMENT_COLUMN = "__frag"
DEDUP_KEY_COLUMN = "__dedup_key"

NO_DEDUP_GROUP = -1
NO_DEDUP_GROUP_ZERO_NORM = -2
"""The two bypass keys, by the cause that produced them: non-finite, zero-norm.

Both route identically. Every consumer tests NEGATIVITY rather than equality -
the write's ``pc.less(keys, 0)``, the retention stage's ``dedup_key < 0``, and the
scan's fill - because a bypassed row has the same fate whatever made its vector
unusable: no similarity is computed, no ``duplicate`` verdict can be reached, and
``curate_cluster_id`` is stored NULL under the ``invalid_embedding`` reason.

The split exists for observability alone, and it is free. The retention stage
runs one group per distinct key and already logs the size of a bypassed group, so
splitting the key splits that line into one per cause; because the grouping is
corpus-wide those counts are exact and total, with no reduction and no second
pass. A single sentinel could only ever report the two causes summed. When that
stage is skipped there is no per-group line to split, and the scan's own warning -
per scan task, so summed by the reader - is what keeps the causes apart.
"""

SUBTASK_CLUSTER_COLUMN = "__subtask_cluster"
NO_SUBTASK_CLUSTER = -1
"""The level-2 fairness group of a row whose subtask text embedding is unusable.

ONE reserved group, not one group per such row, and that is the bound doing its
job: these rows still need a fairness group, and giving each its own would
reintroduce exactly the unbounded key ``subtask_clusters`` exists to remove.

Reached by any row whose subtask text vector is absent, non-finite or
directionless, whatever the fusion weights are. Weighting the block narrows the
causes but does not remove them: the eligibility predicate requires a weighted
column non-NULL, yet presence is not usability, so a stored vector can still be
non-finite or zero-norm and a fully weighted run can produce this cell for one
row. At ``subtask=0.0`` - the documented escape for a corpus whose labels are
identifiers rather than language - the predicate stops requiring the column at
all and every row may take it.

So the cell reports a per-row data-quality state and not a config choice, and it
needs no ``CurateReason`` of its own for the same reason ``missing_embedding``
does not exist: the state is queryable from the source column. It is RESERVED,
and the key is never NULL - the fairness stage groups on it.

Negative like the ``__dedup_key`` sentinels and for the same reason - a sentinel
must not be confusable with a cell index - but a SEPARATE constant, because the
two route on different geometries and a shared name would invite one being tested
for the other.
"""

# The raw task annotation the clips table carries and the canonical key the scan
# folds it into with fairness.canonicalize_label. Level-1 fairness groups on
# __canonical_task; level-2 groups on the (__canonical_task, __subtask_cluster)
# pair, whose second element vectors.SubtaskClusterAssigner computes, at the scan.
# Neither grouping key is persisted: __canonical_task exists in flight only - it
# carries the private prefix for that reason, as its level-2 counterpart does -
# and task_name is read from the table and never written back.
TASK_COLUMN = "task_name"
CANONICAL_TASK_COLUMN = "__canonical_task"

RAY_COUNT_COLUMN = "count()"
"""The column name Ray Data's ``GroupedData.count()`` gives its output.

Verified against Ray 2.55.1. Here rather than in either module that touches it
because it is the same producer/consumer pair the routing names above exist for:
the driver takes the group counts and ``fairness.survivor_group_counts`` reads
them, so a framework rename must be one edit and not two that can drift. No Ray
import is implied by depending on the name.

It is also why both reductions are taken via ``count()`` rather than an
``aggregate`` whose output column would be named differently.
"""

DISTANCE_COLUMN = "distance_to_centroid"
"""Each row's fused cosine distance to its cluster's centroid. In flight only.

Produced by the scan's assignment step, read by the retention pass as its
ordering key, and carried through to fairness, which orders within a funded
group by it. It is never persisted; the cluster it is a distance TO reaches
storage as ``curate_cluster_id``.
"""

DEDUP_SCORE_COLUMN = "max_earlier_similarity"
"""Each row's cosine to the nearest row that outranked it. In flight only.

Produced by the retention pass - it is the value the duplicate threshold is
applied to - read by the report fold, which bins it corpus-wide, and dropped by
the projection onto ``VERDICT_ROW`` before the write. NULL for a row that
bypassed the similarity pass, and NULL on every row when the pass is skipped
entirely, because no comparison was made.

Nothing about it is persisted. The distribution is the only thing that makes
``dedup_eps`` interpretable after the fact - a stored per-row score would be an
O(N) column answering a question that is already answered by O(bins) of counts.
"""

WORKING_VECTOR_COLUMN = "__vector"
"""The fused, unit-norm, weighted vector every duplicate verdict is computed on.

In flight only, and it stops at the retention pass: that stage is the vector's
last reader, so it drops the column, which is what keeps the widest thing in the
run out of the fairness shuffle and out of the write.
"""

TRANSACTION_KIND = "curator-next-curation"
KIND_PROPERTY = "kind"
FINGERPRINT_PROPERTY = "centroids_fingerprint"
CENTROIDS_ROOT_SUFFIX = "__curate_centroids"
"""The four names that make a committed version's basis findable from outside.

Here rather than in the pipeline for the same producer/consumer reason as
``RAY_COUNT_COLUMN``, and it binds harder: the pipeline writes the basis object
under ``{clips_uri}{CENTROIDS_ROOT_SUFFIX}/{sha256}.npz`` and stamps
``{KIND_PROPERTY: TRANSACTION_KIND, FINGERPRINT_PROPERTY: <that sha256>}`` on the
commit, while a reader has to reverse exactly that to find the geometry a
``curate_cluster_id`` means anything against. A reader cannot import the pipeline
to learn these - it pulls in Ray and cuML - so before they lived here every
reader mirrored the literals, and a rename produced a tool that resolves nothing
rather than a failing import.

``TRANSACTION_KIND`` is the LANCE COMMIT property value and is not the CLI
pipeline kind (``curate``); the two are different namespaces that both spell
their key "kind".
"""

CENTROID_ARCHIVE_KEYS = frozenset(
    {
        "centroids",
        "effective_k",
        "fused_dim",
        "block_dims",
        "block_columns",
        "block_weights",
        "read_version",
        "fit_rows",
        "kmeans_random_state",
        "fit_fragment_ids",
        "subtask_centroids",
        "subtask_k",
        "subtask_column",
        "producer_columns",
        "producer_identities",
        "resolved_config",
        "config_digest",
    }
)
"""Every key the centroids artifact carries, as the writer and readers agree on it.

The set is shared so that a reader can tell "a key I do not consume" apart from
"a key no Curate run wrote", which is the only distinction worth reporting about
an archive's shape. A reader holding its own idea of the full set cannot: once
the writer adds a key, every healthy artifact looks foreign to it.

Adding a key therefore means adding it here too, which the writer's own test
enforces rather than leaving to a reader to discover in the field.
"""

VERDICT_ROW: pa.Schema = pa.schema(
    [
        pa.field(KEY_COLUMN, pa.string(), nullable=False),
        pa.field(FRAGMENT_COLUMN, pa.int32(), nullable=False),
        pa.field(DEDUP_KEY_COLUMN, pa.int32(), nullable=False),
        # Nullable in flight, unlike at rest: the scan reasons only about the
        # rows it can already judge invalid, and dedup and fairness fill the
        # rest, so a row in transit legitimately carries no reason yet.
        pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True),
    ]
)
"""The MINIMUM a verdict row must carry to be writable. Never stored.

A floor, not the exact row: a stage may carry more, and the ones between the scan
and the write do. Dedup emits the canonical labels and each row's distance for
fairness to order by, and none of that reaches storage. The write reads these
four columns by name and ignores the rest, so widening a row in flight is safe
and dropping one of these four is not.

The int32 widths are the Ray shuffle key's type, not decoration - stamping
``__frag`` as int64 would still write correctly here and then fail as a group-key
mismatch somewhere else entirely.
"""

# Which ModalityWeights field funds each embedding group. Text is the one that
# differs: its group carries two vectors and only the subtask one is fused, so
# the weight is named after the vector rather than after the group. A new group
# reaching FUSED_BLOCKS without an entry here raises KeyError at import, which is
# correct - a new modality has to decide its weight before it can be fused.
_WEIGHT_FIELD_BY_GROUP: dict[str, str] = {"text": "subtask", "image": "image", "action": "action"}

FUSED_BLOCKS: tuple[tuple[EmbeddingColumnGroup, str], ...] = tuple(
    (group, _WEIGHT_FIELD_BY_GROUP[group.name]) for group in EMBEDDING_COLUMN_GROUPS
)
"""The fused blocks, in the order they are concatenated: subtask, image, action.

The order is result-defining, because it fixes which coordinate range of the
fused vector belongs to which modality; a centroids artifact is only readable
against the order that produced it. Each block's column and width come from its
group's ``primary_vector``, so nothing here re-declares a name or a dimension.
The second element of each pair names the ``ModalityWeights`` field that funds
that block.
"""


def block_width(group: EmbeddingColumnGroup) -> int:
    """Return the stored width of one block's primary vector."""
    return int(group.schema.field(group.primary_vector).type.list_size)


def _task_vector_column() -> str:
    """Return the text vector column whose embedding represents a TASK label.

    Derived from the text group rather than spelled out: the subtask vector is its
    ``primary_vector`` (the block Curate fuses), so the task vector is the one
    remaining field once the primary and the provenance columns are removed. A
    group that stops having exactly one such field raises here, at import, so a
    schema change cannot silently leave the merge pointing at the wrong column.

    Raises:
        ValueError: If the text group does not carry exactly one non-primary,
            non-provenance vector field.

    """
    group = TEXT_COLUMN_GROUP
    remaining = [
        name for name in group.field_names if name != group.primary_vector and name not in group.provenance_columns
    ]
    if len(remaining) != 1:
        msg = (
            f"the text embedding group must carry exactly one non-primary, non-provenance vector for the "
            f"task label merge, found {remaining} in {list(group.field_names)}"
        )
        raise ValueError(msg)
    return remaining[0]


TASK_VECTOR_COLUMN: str = _task_vector_column()
"""The stored embedding of a task label; the level-1 merge's only vector input."""

SUBTASK_VECTOR_COLUMN: str = TEXT_COLUMN_GROUP.primary_vector
"""The stored embedding of a subtask label; the level-2 partition's only input.

Also the fused subtask block, which is what makes the level-2 basis free to fit:
whenever the block carries weight, the fit sample's scanner already projects this
column, so partitioning it costs no additional read.
"""


def _subtask_weight_field() -> str:
    """Return the ``ModalityWeights`` field funding the subtask block.

    Raises:
        ValueError: If no fused block is the subtask vector, which would leave the
            level-2 partition unable to tell whether its input is funded.

    """
    for group, field in FUSED_BLOCKS:
        if group.primary_vector == SUBTASK_VECTOR_COLUMN:
            return field
    msg = (
        f"no fused block carries {SUBTASK_VECTOR_COLUMN}, so nothing funds the level-2 partition; "
        f"blocks are {[group.primary_vector for group, _ in FUSED_BLOCKS]}"
    )
    raise ValueError(msg)


SUBTASK_WEIGHT_FIELD: str = _subtask_weight_field()
"""Which ``ModalityWeights`` field funds the subtask block, looked up not spelled.

The text group is the only one whose weight is named after a vector rather than
after the group, so this is the single place that asymmetry is resolved.

Also read to decide whether a level-2 basis is fitted at all: a zero weight is the
documented escape for a corpus whose subtask labels are identifiers rather than
language, and it removes the column from both the metric and the predicate, so
there is nothing left to partition.
"""


def _weighted_blocks(weights: Mapping[str, float]) -> tuple[tuple[EmbeddingColumnGroup, str], ...]:
    """Return the fused blocks this run funds, in ``FUSED_BLOCKS`` order.

    The only place the block list is FILTERED by weight, so every question of the
    form "which columns does this run read" resolves through here and a de-weighted
    modality cannot drop out of one caller's answer while surviving in another's.
    Asking whether ONE named block is funded is a different question, answered by
    testing ``SUBTASK_WEIGHT_FIELD`` directly.

    Raises:
        KeyError: If a fused block has no weight entry; an unweighted modality
            would silently drop out of the metric and the predicate alike.

    """
    return tuple((group, field) for group, field in FUSED_BLOCKS if weights[field] > 0.0)


def weighted_vectors(weights: Mapping[str, float]) -> tuple[str, ...]:
    """Return the primary vector of every block this run funds, in fused-block order.

    The FUSED read set: what the fit projects, what the scan reads, and what a row
    must carry to be eligible. It deliberately excludes ``TASK_VECTOR_COLUMN``,
    which no weight funds - for the set that includes it see ``consumed_columns``.
    """
    return tuple(group.primary_vector for group, _ in _weighted_blocks(weights))


def consumed_groups(weights: Mapping[str, float]) -> tuple[EmbeddingColumnGroup, ...]:
    """Return the embedding groups this run reads a vector out of, in fused-block order.

    The text group is always consumed, because the task merge reads
    ``TASK_VECTOR_COLUMN`` on every run whatever the weights. Every other group is
    consumed exactly when its block carries weight, so a de-weighted modality is
    not read even on a table that carries it.
    """
    return tuple(dict.fromkeys([TEXT_COLUMN_GROUP, *(group for group, _ in _weighted_blocks(weights))]))


def consumed_vectors(group: EmbeddingColumnGroup, weights: Mapping[str, float]) -> tuple[str, ...]:
    """Return the vectors this run reads out of ``group``, in schema order.

    The text group is the asymmetric one: ``TASK_VECTOR_COLUMN`` is read on every
    run because the level-1 merge needs it, and the fused ``primary_vector`` only
    when its block carries weight. Every other group is read exactly through its
    ``primary_vector``.

    Args:
        group: A group ``consumed_groups`` returned for these weights.
        weights: Per-block weight keyed by the ``FUSED_BLOCKS`` weight field.

    """
    if group.name != TEXT_COLUMN_GROUP.name:
        return (group.primary_vector,)
    read = {TASK_VECTOR_COLUMN}
    if weights[SUBTASK_WEIGHT_FIELD] > 0.0:
        read.add(group.primary_vector)
    return tuple(name for name in group.field_names if name in read)


def consumed_columns(weights: Mapping[str, float]) -> tuple[str, ...]:
    """Return every vector column this run reads, across all consumed groups.

    The union of ``consumed_vectors`` over ``consumed_groups``: the weighted
    primaries plus ``TASK_VECTOR_COLUMN``. Preflight requires exactly this set to
    exist and the producer gate proves exactly this set attributed, so neither
    restates the text asymmetry on its own.
    """
    return tuple(dict.fromkeys(name for group in consumed_groups(weights) for name in consumed_vectors(group, weights)))


def eligibility_filter(weights: Mapping[str, float]) -> str:
    """Return the Lance predicate selecting the rows a run is able to evaluate.

    A row is claimable only if every block carrying weight has a vector, so this ANDs one
    ``IS NOT NULL`` clause per non-zero-weight block. A zero-weight block contributes no
    clause: the supported escape for a corpus whose labels are identifiers, not language.

    Note it tests the WEIGHTED blocks, so it never names ``TASK_VECTOR_COLUMN``:
    eligibility cannot speak for a text group that only the level-1 merge reads.

    Args:
        weights: Per-block weight keyed by the ``FUSED_BLOCKS`` weight field.

    Raises:
        KeyError: If a fused block has no weight entry; an unweighted modality would
            silently drop out of the metric and the predicate alike.
        ValueError: If no block carries weight, so no row could be evaluated.

    """
    # Identifiers are written BARE. Lance parses a double-quoted name as a string
    # LITERAL, so "embedding_image" IS NOT NULL is a non-null constant that
    # matches every row instead of filtering any - a silent full-corpus pass.
    clauses = [f"{name} IS NOT NULL" for name in weighted_vectors(weights)]
    if not clauses:
        msg = f"no fused block carries weight in {dict(weights)}; at least one modality must be weighted"
        raise ValueError(msg)
    return " AND ".join(clauses)
