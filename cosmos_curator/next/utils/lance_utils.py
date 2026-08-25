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

"""Shared Lance utilities for Curator Next recipes.

Deliberately small: the two things every Curator Next recipe needs from Lance
before it can decide anything - "does this table exist yet, and how many rows
does it hold" - plus the one constant every sink that writes its own fragments
must agree on. Absence is answerable either way: ``open_dataset`` returns
``None`` for a caller that treats a missing table as a valid state, and
``open_dataset_or_raise`` raises for one that cannot proceed without it.
Everything table-shaped (schemas, key columns, what a vector means) belongs to
the owning recipe, not here.

Write paths are NOT here on purpose. Each recipe owns its own commit semantics:
``video_split`` and ``robot_action_split`` append fragments, the embeddings recipe
updates column groups in place. A shared "write a Lance table" helper would have
to encode one of those choices for all of them.
"""

from typing import Literal

import lance

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options

# The Lance data-storage version every Curator Next table is written at, so they
# all share one fragment granularity. Public because other sinks that write their
# own fragments must use the same value; a second definition there would let the
# two tables silently diverge.
#
# Independent of the video pipeline's CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
# which looks identical but is an enforced per-table read gate: that pipeline
# raises when an existing dataset's version differs, so sharing a symbol would
# turn a bump made here into a breaking read change there. Kept separate
# deliberately.
LANCE_DATA_STORAGE_VERSION: Literal["2.2"] = "2.2"


def open_dataset(uri: str, *, storage_options: dict[str, str] | None = None) -> lance.LanceDataset | None:
    """Open a Lance dataset, returning ``None`` when it does not exist.

    Lance (9.x) exposes no dataset-exists predicate, so absence is classified by
    the error message: a "not found" / "does not exist" ``(OSError, ValueError)``
    reads as ``None`` and anything else re-raises. The message match cannot tell a
    permissions or bucket error whose text happens to contain "not found" apart
    from a genuinely absent table.
    """
    try:
        return lance.dataset(uri, storage_options=storage_options)
    except (OSError, ValueError) as exc:
        message = str(exc).lower()
        if "not found" in message or "does not exist" in message:
            return None
        raise


def open_dataset_or_raise(uri: str, *, storage_options: dict[str, str] | None = None) -> lance.LanceDataset:
    """Open a Lance dataset, raising ``ValueError`` when it is absent or unreadable.

    The raising counterpart to ``open_dataset``, for callers that cannot proceed
    without the table. A storage-level ``OSError`` is translated too, so one
    ``except ValueError`` covers both failure modes.

    Raises:
        ValueError: If no table exists at ``uri``, or it cannot be opened.

    """
    try:
        dataset = open_dataset(uri, storage_options=storage_options)
    except OSError as exc:
        msg = f"Lance table at {uri} could not be opened: {exc}"
        raise ValueError(msg) from exc
    if dataset is None:
        msg = f"Lance table not found at {uri}"
        raise ValueError(msg)
    return dataset


def count_rows(uri: str, *, storage_profile: str = "default") -> int:
    """Return the row count of a Lance table, or 0 when it does not exist yet.

    Reads fragment metadata only (no column scan). A missing table is 0 while a
    real storage / I/O error propagates.
    """
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        return 0
    return int(dataset.count_rows())
