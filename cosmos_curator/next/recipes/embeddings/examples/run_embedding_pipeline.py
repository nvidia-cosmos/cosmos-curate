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

r"""Maintenance tool: reset an embedding column group on a ``clips.lance`` table.

Embedding RUNS are configured, not flagged: write a ``kind: embeddings`` config
and launch it with ``pixi run --as-is run-pipeline <config>``. This tool covers
the one operation that deliberately has no config field, because a reset is
destructive and must be typed once rather than persisted in a YAML that a
scheduler can replay.

Resetting drops the named groups' columns and re-adds them empty, then exits
without embedding anything, so replacing a group is always two deliberate steps:
reset, then an ordinary run that refills. Between them the group reads all-NULL,
which is exactly the state the pending predicate reads as "everything is owed".

Example (shell line continuations render as a single backslash)::

    # Replace the text group under new weights: reset it, then re-run to refill
    python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
        --clips-lance-uri s3://bucket/run/clips.lance \
        --reset-group text
    pixi run --as-is run-pipeline embeddings.yaml
"""

import argparse
import sys

from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.embeddings.columns import drop_embedding_group, ensure_embedding_columns
from cosmos_curator.next.recipes.embeddings.config import Modality
from cosmos_curator.next.utils.lance_utils import open_dataset_or_raise

# Which column group each modality names on the command line. The reset
# operation takes a URI and a group rather than a run spec, so it cannot read the
# mapping off a ModalityFill.
_GROUPS_BY_MODALITY: dict[Modality, EmbeddingColumnGroup] = {
    Modality.TEXT: TEXT_COLUMN_GROUP,
    Modality.IMAGE: IMAGE_COLUMN_GROUP,
    Modality.ACTION: ACTION_COLUMN_GROUP,
}


def _parse_args() -> argparse.Namespace:
    """Parse the maintenance tool's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MAINTENANCE: drop and re-add embedding column groups on a clips Lance table.",
    )
    parser.add_argument(
        "--clips-lance-uri",
        required=True,
        help="URI of the shared clips Lance table whose embedding columns are reset.",
    )
    parser.add_argument("--storage-profile", default="default", help="Storage profile for tables / media / artifacts.")
    parser.add_argument(
        "--reset-group",
        nargs="+",
        required=True,
        choices=[modality.value for modality in Modality],
        help="Drop these modalities' embedding columns and re-add them empty, then exit without embedding. The "
        "group reads all-NULL until an ordinary run refills it from scratch - which is how a group is replaced "
        "under new weights or a new PCA basis.",
    )
    return parser.parse_args()


def _reset_groups(args: argparse.Namespace) -> None:
    """Drop the named groups' columns and re-add them empty, then return.

    Dropping alone would leave the table in a state no ordinary run distinguishes
    from "this modality was never enabled", so the columns are re-added in the
    same operation: the group ends up present and entirely NULL, which is exactly
    the state the pending predicate reads as "everything is owed".

    Raises:
        ValueError: If the table is absent or unreadable.

    """
    # Order-preserving dedup: a repeated ``--reset-group text text`` must stage each
    # group once. ensure_embedding_columns snapshots the schema names a single time,
    # so a duplicated group would be seen as "absent" twice and stage duplicate
    # fields, failing on the duplicate column name instead of being idempotent.
    groups = [_GROUPS_BY_MODALITY[Modality(name)] for name in dict.fromkeys(args.reset_group)]
    storage_options = get_lance_storage_options(args.clips_lance_uri, profile_name=args.storage_profile)
    dataset = open_dataset_or_raise(args.clips_lance_uri, storage_options=storage_options)
    for group in groups:
        drop_embedding_group(dataset, group)
    # Re-opened between the drops and the re-add: each drop is its own commit, and
    # ensure_embedding_columns decides what to add from the schema of the handle it
    # is given, so it must read the post-drop version rather than the pre-drop one.
    dataset = open_dataset_or_raise(args.clips_lance_uri, storage_options=storage_options)
    added, _ = ensure_embedding_columns(dataset, groups)
    logger.info(
        f"reset {[group.name for group in groups]} on {args.clips_lance_uri}: {added} column(s) re-added empty; "
        f"run the embeddings pipeline to refill them"
    )


def main() -> None:
    """Reset the named embedding groups, reporting a driver-side failure as one line.

    Catch a driver-side ``ValueError`` (a missing or unreadable clips table),
    log it at ERROR, and exit non-zero so the operator sees the actionable
    message as the last line rather than the tail of a traceback that buries it.
    """
    args = _parse_args()
    try:
        _reset_groups(args)
    except ValueError as error:
        logger.error(str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
