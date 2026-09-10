# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic canonical-table bootstrap and incremental append primitives.

Uses a small two-column schema unrelated to any recipe to prove these primitives
do not depend on clip-specific (or any other recipe-specific) row shape.
"""

from pathlib import Path
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.utils import lance_fragment_recovery

_DATA_STORAGE_VERSION = "2.2"
_KIND = "test-recipe"
_OPERATION = "append-rows"

_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]
)


def _rows(*row_ids: str) -> pa.Table:
    return pa.table({"row_id": list(row_ids), "value": [f"v-{row_id}" for row_id in row_ids]}, schema=_SCHEMA)


def _bootstrap(uri: str) -> lance.LanceDataset:
    return lance_fragment_recovery.open_or_create_table(
        uri=uri,
        storage_profile="default",
        schema=_SCHEMA,
        data_storage_version=_DATA_STORAGE_VERSION,
        kind=_KIND,
    )


def _stage(uri: str, *row_ids: str) -> str:
    candidate = lance_fragment_recovery.write_row_fragment(
        _rows(*row_ids),
        uri=uri,
        storage_profile="default",
        schema=_SCHEMA,
        id_column="row_id",
        data_storage_version=_DATA_STORAGE_VERSION,
    )
    assert candidate is not None
    return candidate


def _append(uri: str, candidate: str, *, attempts: int = 1) -> int:
    return lance_fragment_recovery.append_row_fragment(
        candidate,
        uri=uri,
        storage_profile="default",
        schema=_SCHEMA,
        id_column="row_id",
        kind=_KIND,
        operation=_OPERATION,
        data_storage_version=_DATA_STORAGE_VERSION,
        attempts=attempts,
    )


def test_absent_table_is_bootstrapped_once_with_zero_rows(tmp_path: Path) -> None:
    """The durable schema exists before the first row and is never recreated."""
    uri = str(tmp_path / "rows.lance")

    created = _bootstrap(uri)
    reopened = _bootstrap(uri)

    assert created.version == reopened.version == 1
    assert reopened.count_rows() == 0
    assert reopened.schema == _SCHEMA


def test_every_staged_fragment_appends_as_a_new_version(tmp_path: Path) -> None:
    """The schema bootstrap and each fragment have separate visible versions."""
    uri = str(tmp_path / "rows.lance")
    bootstrap_version = _bootstrap(uri).version

    first_version = _append(uri, _stage(uri, "a", "b"))
    second_version = _append(uri, _stage(uri, "c"))

    dataset = lance.dataset(uri)
    assert first_version == bootstrap_version + 1
    assert second_version == first_version + 1
    assert sorted(dataset.to_table()["row_id"].to_pylist()) == ["a", "b", "c"]


def test_replaying_a_committed_fragment_is_a_noop(tmp_path: Path) -> None:
    """Re-appending an already-canonical fragment skips the commit."""
    uri = str(tmp_path / "rows.lance")
    _bootstrap(uri)
    candidate = _stage(uri, "a")
    first_version = _append(uri, candidate)

    replay_version = _append(uri, candidate)

    assert replay_version == first_version
    assert lance.dataset(uri).to_table()["row_id"].to_pylist() == ["a"]


def test_ambiguous_commit_is_confirmed_via_presence_check_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit that raises after actually succeeding server-side must be confirmed.

    It must be confirmed by re-checking candidate presence, not treated as a
    failure and retried into a duplicate append.
    """
    uri = str(tmp_path / "rows.lance")
    _bootstrap(uri)
    candidate = _stage(uri, "a")
    real_commit = lance.LanceDataset.commit

    def _commit_then_raise_ambiguously(*args: object, **kwargs: object) -> None:
        real_commit(*args, **kwargs)  # the write actually lands...
        msg = "ambiguous network failure after commit"
        raise OSError(msg)  # ...but the caller sees this as a failure.

    commit = Mock(side_effect=_commit_then_raise_ambiguously)
    monkeypatch.setattr(lance.LanceDataset, "commit", commit)

    version = _append(uri, candidate, attempts=3)

    assert commit.call_count == 1  # confirmed via presence check, never retried
    assert version == lance.dataset(uri).version
    assert lance.dataset(uri).to_table()["row_id"].to_pylist() == ["a"]


def test_partial_candidate_presence_fails_instead_of_appending_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight rejects partial visibility before attempting another append."""
    uri = str(tmp_path / "rows.lance")
    _bootstrap(uri)
    _append(uri, _stage(uri, "a"))
    candidate = _stage(uri, "a", "b")

    commit = Mock(side_effect=AssertionError("partial candidate unexpectedly reached commit"))
    monkeypatch.setattr(lance.LanceDataset, "commit", commit)

    with pytest.raises(RuntimeError, match="1 of 2 candidate row_id values"):
        _append(uri, candidate, attempts=2)

    commit.assert_not_called()
    assert lance.dataset(uri).to_table(columns=["row_id"])["row_id"].to_pylist() == ["a"]


def test_existing_table_must_contain_compatible_producer_fields(tmp_path: Path) -> None:
    """A table with a different canonical row contract cannot be reused."""
    uri = str(tmp_path / "rows.lance")
    incompatible_schema = _SCHEMA.remove(_SCHEMA.get_field_index("row_id"))
    lance.write_dataset(
        pa.Table.from_batches([], schema=incompatible_schema),
        uri,
        mode="create",
        data_storage_version=_DATA_STORAGE_VERSION,
    )

    with pytest.raises(ValueError, match=r"missing producer-owned field.*row_id"):
        _bootstrap(uri)


def test_existing_table_must_have_producer_fields_in_the_same_order(tmp_path: Path) -> None:
    """A same-name, same-type, differently-ordered producer schema is rejected.

    ``write_row_fragment`` stages a fragment using the caller's schema's own
    field order, which Lance's ``write_fragments`` binds positionally rather
    than by name against the already-open table. A reordered producer schema
    would otherwise pass every other check here and silently misalign values
    into the wrong columns on the next append.
    """
    uri = str(tmp_path / "rows.lance")
    reordered_schema = pa.schema([_SCHEMA.field("value"), _SCHEMA.field("row_id")])
    lance.write_dataset(
        pa.Table.from_batches([], schema=reordered_schema),
        uri,
        mode="create",
        data_storage_version=_DATA_STORAGE_VERSION,
    )

    with pytest.raises(ValueError, match=r"producer-owned fields in a different order"):
        _bootstrap(uri)


def test_existing_extension_fields_must_be_nullable(tmp_path: Path) -> None:
    """Every extension must accept null from future producer-only fragments."""
    uri = str(tmp_path / "rows.lance")
    incompatible_schema = _SCHEMA.append(pa.field("enrichment__test_v1", pa.string(), nullable=False))
    lance.write_dataset(
        pa.Table.from_batches([], schema=incompatible_schema),
        uri,
        mode="create",
        data_storage_version=_DATA_STORAGE_VERSION,
    )

    with pytest.raises(ValueError, match=r"non-nullable extension field.*enrichment__test_v1"):
        _bootstrap(uri)


def test_existing_nullable_extension_fields_are_preserved_on_append(tmp_path: Path) -> None:
    """Producer fragments omit extension fields and Lance presents them as null."""
    uri = str(tmp_path / "rows.lance")
    _bootstrap(uri).add_columns(pa.field("enrichment__test_v1", pa.string(), nullable=True))

    _append(uri, _stage(uri, "a"))

    dataset = lance.dataset(uri)
    assert dataset.schema.field("enrichment__test_v1").nullable
    row = dataset.to_table().to_pylist()[0]
    assert row["row_id"] == "a"
    assert row["enrichment__test_v1"] is None


def test_fragment_writer_requires_the_canonical_schema(tmp_path: Path) -> None:
    """Rows that do not match the canonical schema cannot be staged."""
    with pytest.raises(ValueError, match="canonical schema"):
        lance_fragment_recovery.write_row_fragment(
            pa.table({"row_id": ["a"]}),
            uri=str(tmp_path / "rows.lance"),
            storage_profile="default",
            schema=_SCHEMA,
            id_column="row_id",
            data_storage_version=_DATA_STORAGE_VERSION,
        )


def test_fragment_writer_rejects_duplicate_ids_in_one_batch(tmp_path: Path) -> None:
    """Two rows with the same id in one batch cannot both stage as one fragment."""
    with pytest.raises(ValueError, match="duplicate row_id values"):
        lance_fragment_recovery.write_row_fragment(
            _rows("a", "a"),
            uri=str(tmp_path / "rows.lance"),
            storage_profile="default",
            schema=_SCHEMA,
            id_column="row_id",
            data_storage_version=_DATA_STORAGE_VERSION,
        )
