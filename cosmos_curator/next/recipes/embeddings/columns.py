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

"""Driver-side schema, state, and narrow reads over the embedding groups of ``clips.lance``.

Everything about the table that is neither model computation nor the physical
column write lives here, as functions over a dataset the CALLER opened. There is
no bound URI and no private opener, so which version a call reads is visible at
the call site: a caller that must see its own commit re-opens and passes the new
handle, rather than trusting shared state to have refreshed itself.

::

    dataset (opened by the caller - its version is the version read)
        |
        +--> SCHEMA   widen one group in, or detach one group out; each is a
        |             single metadata commit and groups stay independent, so a
        |             text-only run never creates the image or action columns
        |
        +--> STATE    a filled group has at most one producer, and it is the
        |             configured one - checked before any compute is spent
        |
        +--> READ     the pending predicate, filled counts, and a one-column
                      stream

The reads are deliberately cheap. The pending predicate is only BUILT here and is
evaluated later, per fragment, by the worker that owns it; the filled count is
resolved inside Lance and returns a bounded result whatever the table's length;
and the one function that streams rows projects a single column. The producer
identifies the state check reads come from ``lance_utils``, which owns the bounded
distinct-value read.

The physical write and its commit are NOT here; they belong to the fill path.
Splitting them lets schema and state be tested with no dependency on the
distributed write.

All failures raise a plain, informative ``ValueError`` (the CLI's
``except ValueError -> exit(1)`` handler catches them); no custom exception
hierarchy is introduced.
"""

from collections.abc import Iterator, Sequence

import lance
import pyarrow as pa
from loguru import logger

from cosmos_curator.next.embeddings.schemas import EmbeddingColumnGroup
from cosmos_curator.next.utils.lance_utils import distinct_non_null_values

# Validation only has to tell "one producer" from "more than one", so it asks for
# one value beyond the single one it tolerates and never for the whole column.
_MAX_PRODUCERS_INSPECTED = 2


def _assert_present_group_matches(schema: pa.Schema, group: EmbeddingColumnGroup) -> None:
    """Assert every field of an already-present group matches the group's schema.

    Compares type (which pins the fixed-size-list width) and nullability, so a
    table carrying a differently-shaped column under one of the group's names is
    rejected before any compute runs rather than failing inside a worker's cast.

    Raises:
        ValueError: On any type or nullability mismatch.

    """
    for expected in group.schema:
        stored = schema.field(expected.name)
        if not stored.type.equals(expected.type):
            msg = (
                f"embedding column group {group.name!r}: column {expected.name!r} has type {stored.type} "
                f"but the group requires {expected.type}"
            )
            raise ValueError(msg)
        if stored.nullable != expected.nullable:
            msg = f"embedding column group {group.name!r}: column {expected.name!r} must be nullable"
            raise ValueError(msg)


def _group_presence(schema_names: set[str], group: EmbeddingColumnGroup) -> str:
    """Classify a group as ``absent`` (no fields), ``present`` (all fields), or ``partial``."""
    present = [name for name in group.field_names if name in schema_names]
    if not present:
        return "absent"
    if len(present) == len(group.field_names):
        return "present"
    return "partial"


def ensure_embedding_columns(
    dataset: lance.LanceDataset, groups: Sequence[EmbeddingColumnGroup]
) -> tuple[int, int | None]:
    """Add the missing fields of only the given groups in one metadata commit.

    ``add_columns`` with an all-nullable schema is a METADATA-ONLY operation:
    Lance records the new fields in the manifest and reads them as NULL for
    every existing row. No data file is rewritten and no row is replaced, so
    this cannot create a tombstone.

    Each group must be absent as a whole (all its fields are added) or present
    and exactly matching its schema (nothing added). A partially present group
    is a corrupt schema and raises. Groups not passed are left untouched, so a
    text-only run never creates the image or action columns.

    Args:
        dataset: The clips table to widen.
        groups: The enabled modalities' column groups.

    Returns:
        ``(fields_added, commit_version)``. ``commit_version`` is the widening
        metadata commit's version, captured on the handle immediately after
        ``add_columns``; ``None`` when every group was already present.

    Raises:
        ValueError: If a group is partially present, or a present group does not
            match its schema exactly.

    """
    schema = dataset.schema
    names = set(schema.names)
    fields_to_add: list[pa.Field] = []
    for group in groups:
        presence = _group_presence(names, group)
        if presence == "absent":
            fields_to_add.extend(group.schema)
        elif presence == "present":
            _assert_present_group_matches(schema, group)
        else:
            missing = [name for name in group.field_names if name not in names]
            found = [name for name in group.field_names if name in names]
            msg = (
                f"embedding column group {group.name!r} is partially present on {dataset.uri}: "
                f"has {found}, missing {missing}; a group must be absent as a whole or present and complete"
            )
            raise ValueError(msg)
    if fields_to_add:
        dataset.add_columns(pa.schema(fields_to_add))
        return len(fields_to_add), int(dataset.version)
    return 0, None


def drop_embedding_group(dataset: lance.LanceDataset, group: EmbeddingColumnGroup) -> int:
    """Detach one group's columns, taking it back to absent so a later run refills it.

    ``drop_columns`` detaches the group's data file from every fragment and a
    refill writes a brand-new one, which is what makes a partial refill safe: the
    fragments it never reaches read NULL rather than pre-drop values.

    Returns:
        The number of columns dropped (0 when the group was already absent).

    """
    names = set(dataset.schema.names)
    # Narrow to the fields that are really there: drop_columns rejects a name the
    # schema does not carry and then drops NOTHING, so passing a partially present
    # group whole would fail the entire reset instead of clearing what exists.
    present = [name for name in group.field_names if name in names]
    if not present:
        # drop_columns([]) does not raise, but it does commit a version. Skipping
        # the call keeps a no-op reset off the table's version history entirely.
        return 0
    dataset.drop_columns(present)
    logger.info(f"detached embedding column group {group.name!r} from {dataset.uri}: dropped {present}")
    return len(present)


def validate_embedding_group(
    dataset: lance.LanceDataset,
    group: EmbeddingColumnGroup,
    expected_provenance: dict[str, str] | None,
) -> None:
    """Validate a group is safe to fill: at most one (expected) producer.

    The group's rows are all-empty or all-complete by construction, so a mixed
    NULL/non-NULL partial row is unreachable and is not checked here. The
    guarantee comes from the producer, not from the write: the group-batch
    builders in ``embeddings.schemas`` derive every field of a row from ONE
    per-row validity mask, so a failed row's vector and provenance go NULL
    together, and ``update_columns`` then writes that whole group row at once.

    Enforces that the complete rows carry at most one distinct producer identity
    per provenance column, and that identity equals the configured one in
    ``expected_provenance``. Because a single distinct tuple over the provenance
    columns is equivalent to each provenance column having at most one distinct
    non-null value, this is checked column by column.

    Args:
        dataset: The clips table to read.
        group: The group to validate.
        expected_provenance: ``{provenance_column: expected_value}`` for the
            staleness comparison, or ``None`` to enforce only the single-producer
            rule. The action leg passes ``None`` here and reads the single
            surviving fingerprint via ``distinct_non_null_values``.

    Raises:
        ValueError: If the group is absent, has more than one producer, or is
            stale versus ``expected_provenance``.

    """
    missing = [name for name in group.field_names if name not in set(dataset.schema.names)]
    if missing:
        msg = (
            f"embedding column group {group.name!r} is not present on {dataset.uri} (missing {missing}); "
            f"ensure_embedding_columns must run before validation"
        )
        raise ValueError(msg)
    for column in group.provenance_columns:
        producers = distinct_non_null_values(dataset, column, max_values=_MAX_PRODUCERS_INSPECTED)
        if len(producers) > 1:
            msg = (
                f"embedding column group {group.name!r} provenance column {column!r} has multiple producers "
                f"{list(producers)}; reset the group with --reset-group {group.name} and rerun to replace it"
            )
            raise ValueError(msg)
        if producers and expected_provenance is not None:
            expected = expected_provenance.get(column)
            if expected is not None and producers[0] != expected:
                msg = (
                    f"embedding column group {group.name!r} is stale: {column!r} is {producers[0]!r} but the "
                    f"configured producer is {expected!r}; reset the group with --reset-group {group.name} "
                    f"and rerun to recompute it"
                )
                raise ValueError(msg)


def pending_filter(group: EmbeddingColumnGroup, applicability_filter: str | None) -> str:
    """Return the "pending" predicate: applicable rows whose primary vector is NULL.

    This is the single definition of "a row this modality still owes work on", used
    to build the incremental row filter. For text (``applicability_filter is
    None``, every row applicable) it degrades to the bare "primary vector IS NULL"
    clause.
    """
    primary_null = f"{group.primary_vector} IS NULL"
    if applicability_filter:
        return f"({applicability_filter}) AND {primary_null}"
    return primary_null


def count_filled(dataset: lance.LanceDataset, group: EmbeddingColumnGroup) -> int:
    """Return the number of rows whose primary vector is non-NULL."""
    # Zero, not an error, when the group's columns have not been added yet: a
    # caller asking "how much of this group is filled" before the widening commit
    # must read nothing rather than reference a column the SQL planner cannot
    # resolve.
    if group.primary_vector not in dataset.schema.names:
        return 0
    return int(dataset.count_rows(filter=f"{group.primary_vector} IS NOT NULL"))


def scan_column(
    dataset: lance.LanceDataset, column: str, *, row_filter: str | None, batch_size: int
) -> Iterator[pa.Table]:
    """Stream one narrow column of the table, batch by batch.

    The driver-side equivalent of a worker's fragment scan: it projects a single
    column and pushes the predicate into Lance, so a caller that needs a
    corpus-wide view of one field (the action leg's PCA candidate sampling) never
    materializes the table. Yields nothing when the column has no matching rows.

    Args:
        dataset: The clips table to read.
        column: The single column to project.
        row_filter: SQL predicate pushed into the scan, or ``None`` for all rows.
        batch_size: Rows per yielded table.

    """
    scanner = dataset.scanner(columns=[column], filter=row_filter, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        if record_batch.num_rows:
            yield pa.Table.from_batches([record_batch])
