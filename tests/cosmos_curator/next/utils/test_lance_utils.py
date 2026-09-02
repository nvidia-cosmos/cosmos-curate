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

"""Read + version utilities: open_dataset error classification, count_rows, distinct values.

``lance_utils`` is deliberately read-only, so this file is its whole suite: it
exercises the recipe-agnostic helpers against a local generic table, with no
embeddings schema involved. Each recipe's own write path is covered by that
recipe's tests.

The distinct-value read is tested here rather than in either recipe because both
consume it - the embeddings leg to refuse filling a group that already carries two
producer identities, the curation leg to refuse fusing one - so a behaviour change
must redden one shared file instead of depending on which recipe happens to cover
it.
"""

import pathlib
from collections.abc import Sequence

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_split import contracts, lance_sink
from cosmos_curator.next.utils import lance_utils
from cosmos_curator.next.utils.lance_utils import count_rows, distinct_non_null_values

# Distinct values a producer-identity read asks for: one is the legal state, and
# the second exists only so "more than one" is distinguishable from it.
_PRODUCERS_INSPECTED = 2

_IDENTITY_COLUMN = "producer"

# Rows per data file, so every multi-row table below spans several fragments and a
# read that stopped at the first one would be visibly wrong.
_ROWS_PER_FILE = 3


def _write_identities(path: pathlib.Path, values: Sequence[str | None]) -> lance.LanceDataset:
    """Write a one-column table of producer identities across several fragments."""
    uri = str(path)
    lance.write_dataset(
        pa.table({_IDENTITY_COLUMN: pa.array(list(values), type=pa.string())}),
        uri,
        max_rows_per_file=_ROWS_PER_FILE,
        data_storage_version=lance_utils.LANCE_DATA_STORAGE_VERSION,
    )
    return lance.dataset(uri)


def test_count_rows_is_zero_for_absent_table(tmp_path: pathlib.Path) -> None:
    """A table that does not exist yet counts as zero rows, not an error."""
    assert count_rows(str(tmp_path / "missing.lance")) == 0


def test_count_rows_returns_written_row_count(tmp_path: pathlib.Path) -> None:
    """An existing table reports its row count from fragment metadata."""
    uri = str(tmp_path / "t.lance")
    lance.write_dataset(pa.table({"clip_id": ["a", "b", "c"]}), uri, data_storage_version="2.2")
    assert count_rows(uri) == 3


def test_open_dataset_propagates_non_absence_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A permissions-style error is re-raised, not misread as an absent table.

    Absence is classified by message substring; this pins that an error whose text
    is not "not found" / "does not exist" propagates rather than reading as None.
    """

    def raise_permission_error(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        message = "Access Denied: caller lacks permission"
        raise OSError(message)

    monkeypatch.setattr(lance, "dataset", raise_permission_error)
    with pytest.raises(OSError, match="Access Denied"):
        lance_utils.open_dataset(str(tmp_path / "t.lance"))


def test_open_dataset_or_raise_raises_for_absent_table(tmp_path: pathlib.Path) -> None:
    """A caller that cannot proceed gets a ValueError (not None) when the table is absent.

    ``open_dataset`` returns None for absence; ``open_dataset_or_raise`` is the
    raising counterpart, so the absent path must surface as the documented
    ValueError rather than a None a caller could dereference.
    """
    with pytest.raises(ValueError, match="not found"):
        lance_utils.open_dataset_or_raise(str(tmp_path / "missing.lance"))


def test_open_dataset_or_raise_translates_unreadable_oserror_to_valueerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A non-absence storage OSError is translated to ValueError, so one ``except`` covers both failures.

    Unlike absence (which ``open_dataset`` maps to None), a permissions-style
    OSError propagates out of ``open_dataset``; the raising helper wraps it as
    ValueError so callers need not also catch OSError.
    """

    def raise_permission_error(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        message = "Access Denied: caller lacks permission"
        raise OSError(message)

    monkeypatch.setattr(lance, "dataset", raise_permission_error)
    with pytest.raises(ValueError, match="could not be opened"):
        lance_utils.open_dataset_or_raise(str(tmp_path / "t.lance"))


def test_distinct_non_null_values_returns_the_single_identity(tmp_path: pathlib.Path) -> None:
    """A column filled by one producer reads back as exactly that one value."""
    dataset = _write_identities(tmp_path / "single.lance", ["fp-abc"] * 3)
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == ("fp-abc",)


def test_distinct_non_null_values_detects_an_identity_confined_to_the_final_row(tmp_path: pathlib.Path) -> None:
    """A second identity present in only the last row is still surfaced.

    This is the case that separates a bounded read from a truncated one: every
    fragment but the last carries a single identity, so the answer is only settled
    once the whole column has been considered. Stopping early here would hide
    exactly the mixed-producer corruption both callers exist to catch.
    """
    rows = 32
    dataset = _write_identities(tmp_path / "late.lance", ["fp-first"] * (rows - 1) + ["fp-last"])
    values = distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)
    assert values == ("fp-first", "fp-last")


def test_distinct_non_null_values_caps_the_result_at_max_values(tmp_path: pathlib.Path) -> None:
    """A column holding more identities than asked for yields exactly ``max_values``.

    What keeps the read bounded for a caller that only needs to tell one identity
    from several: a corpus column of arbitrary variety must not arrive in the
    driver row by row.

    The bound must be filled by DISTINCT values, so the set is asserted alongside
    the count. Both callers read "more than one value came back" as "more than one
    producer wrote this column", so a cap reached by the same identity twice would
    abort a run over a perfectly well formed table.

    Each identity is written twice in CONSECUTIVE rows for that assertion to be
    reachable, and the adjacency is the load-bearing half: it is what makes the
    first two rows a non-deduplicating read would return the same value. Over one
    row per identity - or the same nine pairs interleaved - such a read still hands
    back distinct values and the assertion can no longer fail.
    """
    dataset = _write_identities(tmp_path / "many.lance", [f"fp-{index}" for index in range(9) for _ in range(2)])
    values = distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)
    assert len(values) == _PRODUCERS_INSPECTED
    assert len(set(values)) == _PRODUCERS_INSPECTED


def test_distinct_non_null_values_ignores_null_rows(tmp_path: pathlib.Path) -> None:
    """Unfilled rows contribute nothing, so a partly filled column names one identity."""
    dataset = _write_identities(tmp_path / "partial.lance", ["fp-abc", None, "fp-abc", None])
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == ("fp-abc",)


def test_distinct_non_null_values_reads_an_identity_containing_a_quote(tmp_path: pathlib.Path) -> None:
    """An identity containing a single quote is read back verbatim.

    The read is expressed as a SQL statement, so a value able to terminate a
    string literal must never reach one. Were it to, a lone legal identity could
    be misread as two and abort a run that was in fact well formed.
    """
    quoted = "fp-o'brien"
    dataset = _write_identities(tmp_path / "quoted.lance", [quoted] * 6)
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == (quoted,)


def test_distinct_non_null_values_names_a_column_the_table_lacks(tmp_path: pathlib.Path) -> None:
    """An absent column raises a named error rather than a planner one.

    The name is interpolated into the statement, so the schema check is what
    stands in for quoting it; without the check the caller would get whatever
    DataFusion says about an unresolvable identifier.
    """
    dataset = _write_identities(tmp_path / "absent.lance", ["fp-abc"])
    with pytest.raises(ValueError, match="no such column"):
        distinct_non_null_values(dataset, "not_a_column", max_values=_PRODUCERS_INSPECTED)


def test_distinct_non_null_values_refuses_a_non_string_column(tmp_path: pathlib.Path) -> None:
    """A non-string column is refused, because its values would not be the promised str.

    Lance yields each value as its own Python type, so an int column would return
    ints under a ``tuple[str, ...]`` annotation - satisfying the type checker and
    no caller. Refusing is what keeps the annotation true at runtime.
    """
    uri = str(tmp_path / "int-identity.lance")
    lance.write_dataset(pa.table({_IDENTITY_COLUMN: pa.array([1, 2], type=pa.int64())}), uri)
    with pytest.raises(ValueError, match="not a string"):
        distinct_non_null_values(lance.dataset(uri), _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)


def test_distinct_non_null_values_refuses_a_bound_that_reads_nothing(tmp_path: pathlib.Path) -> None:
    """A bound below one is a caller bug: it would report "unfilled" for any column.

    ``LIMIT 0`` is valid SQL returning no rows, so the read would succeed and the
    caller would conclude the column names no producer whatever it holds.
    """
    dataset = _write_identities(tmp_path / "zero-bound.lance", ["fp-abc"])
    with pytest.raises(ValueError, match="must be at least 1"):
        distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=0)


def test_lance_sink_and_utils_share_one_storage_version() -> None:
    """The Extract sink writes at the version lance_utils owns, not a private copy.

    Executed coverage for the single-owner arrangement: robot_action_split's own
    suite skips entirely without a libopenh264 encoder, so it cannot gate this.
    """
    assert lance_sink.LANCE_DATA_STORAGE_VERSION == lance_utils.LANCE_DATA_STORAGE_VERSION


def test_contracts_no_longer_defines_a_storage_version() -> None:
    """The duplicate definition is gone, so the two next/ tables cannot disagree."""
    assert not hasattr(contracts, "LANCE_DATA_STORAGE_VERSION")
