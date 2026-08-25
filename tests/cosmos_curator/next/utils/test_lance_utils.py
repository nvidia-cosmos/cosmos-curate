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

"""Read + version utilities: open_dataset error classification, count_rows, storage version.

``lance_utils`` is deliberately read-only, so this file is its whole suite: it
exercises the recipe-agnostic helpers against a local generic table, with no
embeddings schema involved. Each recipe's own write path is covered by that
recipe's tests.
"""

import pathlib

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_split import contracts, lance_sink
from cosmos_curator.next.utils import lance_utils
from cosmos_curator.next.utils.lance_utils import count_rows


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


def test_lance_sink_and_utils_share_one_storage_version() -> None:
    """The Extract sink writes at the version lance_utils owns, not a private copy.

    Executed coverage for the single-owner arrangement: robot_action_split's own
    suite skips entirely without a libopenh264 encoder, so it cannot gate this.
    """
    assert lance_sink.LANCE_DATA_STORAGE_VERSION == lance_utils.LANCE_DATA_STORAGE_VERSION


def test_contracts_no_longer_defines_a_storage_version() -> None:
    """The duplicate definition is gone, so the two next/ tables cannot disagree."""
    assert not hasattr(contracts, "LANCE_DATA_STORAGE_VERSION")
