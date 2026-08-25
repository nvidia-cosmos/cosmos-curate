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

r"""Standalone development runner for the embedding recipe.

Runs ONLY embedding generation over an existing ``clips.lance`` table (updating
its ``embedding_*`` columns in place) and prints a per-modality fill summary. It
bypasses the production runner and the full pipeline wiring - a small, reviewable
way to exercise the recipe end to end.

``--reset-group`` is a separate MAINTENANCE operation rather than a run mode: it
drops the named groups' columns and exits without embedding anything, so
replacing a group is always two deliberate invocations (reset, then an ordinary
run that refills). Between them the group reads all-NULL.

Examples (shell line continuations render as a single backslash)::

    # Quick smoke over a few fragments
    python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
        --clips-lance-uri s3://bucket/run/clips.lance \
        --modalities text image \
        --max-fragments 2

    # Full run including action (fits and persists the PCA basis if none exists yet)
    python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
        --clips-lance-uri s3://bucket/run/clips.lance \
        --modalities text image action \
        --storage-profile default

    # Replace the text group under a new model: reset it, then re-run to refill
    python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
        --clips-lance-uri s3://bucket/run/clips.lance \
        --reset-group text
    python -m cosmos_curator.next.recipes.embeddings.examples.run_embedding_pipeline \
        --clips-lance-uri s3://bucket/run/clips.lance \
        --modalities text
"""

import argparse
import sys
from typing import Any

import ray
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.embeddings.columns import drop_embedding_group, ensure_embedding_columns
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality
from cosmos_curator.next.recipes.embeddings.embed import EmbeddingRunResult, run_embedding_pipeline
from cosmos_curator.next.utils.lance_utils import open_dataset_or_raise

# Which column group each modality names on the command line. The reset
# operation runs before any config is built - it takes a URI and a group, not a
# run spec - so it cannot read the mapping off a ModalityFill.
_GROUPS_BY_MODALITY: dict[Modality, EmbeddingColumnGroup] = {
    Modality.TEXT: TEXT_COLUMN_GROUP,
    Modality.IMAGE: IMAGE_COLUMN_GROUP,
    Modality.ACTION: ACTION_COLUMN_GROUP,
}


def _parse_args() -> argparse.Namespace:
    """Parse the development runner's command-line arguments."""
    parser = argparse.ArgumentParser(description="Run only the embedding modalities over a clips Lance table.")
    parser.add_argument(
        "--clips-lance-uri",
        required=True,
        help="URI of the shared clips Lance table (read source AND write target; base rows appended upstream).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=[modality.value for modality in Modality],
        choices=[modality.value for modality in Modality],
        help="Modalities to embed (each runs independently).",
    )
    parser.add_argument("--storage-profile", default="default", help="Storage profile for tables / media / artifacts.")
    parser.add_argument("--model-weights-path", default=None, help="Override the weights base for download_models.")
    parser.add_argument(
        "--reset-group",
        nargs="+",
        default=[],
        choices=[modality.value for modality in Modality],
        help="MAINTENANCE: drop these modalities' embedding columns and re-add them empty, then exit without "
        "embedding. The group reads all-NULL until an ordinary run refills it from scratch - which is how a "
        "group is replaced under new weights or a new PCA basis.",
    )
    parser.add_argument(
        "--max-fragments",
        type=int,
        default=None,
        help="Cap the Lance fragments each modality visits, for a smoke test. Caps only the fill, not the action "
        "leg's whole-table PCA candidate scan.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> EmbeddingPipelineConfig:
    """Translate CLI args into a validated ``EmbeddingPipelineConfig``."""
    overrides: dict[str, Any] = {
        "clips_lance_uri": args.clips_lance_uri,
        "modalities": tuple(args.modalities),
        "storage_profile": args.storage_profile,
        "max_fragments": args.max_fragments,
    }
    if args.model_weights_path is not None:
        overrides["model_weights_path"] = args.model_weights_path
    return EmbeddingPipelineConfig(**overrides)


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
    added = ensure_embedding_columns(dataset, groups)
    logger.info(
        f"reset {[group.name for group in groups]} on {args.clips_lance_uri}: {added} column(s) re-added empty; "
        f"re-run without --reset-group to refill them"
    )


def _log_summary(result: EmbeddingRunResult) -> None:
    """Log the run's per-modality verification metrics (counts and versions; never vectors)."""
    # ending_version is the LAST modality's commit (each modality commits its own
    # group separately), which - versions being monotonic on one table - is the
    # table's final version after the run. None means nothing was committed.
    version_text = f"v{result.ending_version}" if result.ending_version is not None else "(nothing committed)"
    logger.info(f"clips.lance final version (last modality commit): {version_text}")
    for modality_result in result.modalities:
        # The line also reports skipped_fragments, and the two would
        # read as one fact when they are unrelated - no commit versus lost work.
        committed = (
            f"v{modality_result.committed_version}" if modality_result.committed_version is not None else "(none)"
        )
        ratio = modality_result.failed / modality_result.selected if modality_result.selected else 0.0
        logger.info(
            f"{modality_result.modality}: selected={modality_result.selected} filled={modality_result.filled} "
            f"failed={modality_result.failed} ({ratio:.0%}) "
            f"skipped_fragments={modality_result.skipped_fragments} -> {committed}"
        )
    if result.action is not None:
        action = result.action
        fit = "loaded" if action.samples_used is None else f"fit on {action.samples_used} descriptors"
        logger.info(f"action PCA: fingerprint={action.fingerprint[:12]} ({fit})")


def _enable_rich_progress_ui() -> None:
    """Switch Ray Data to its rich progress UI for this driver process.

    Ray Data logs a startup hint that its classic ``tqdm`` bars can be replaced
    by a richer progress UI. Applying the switch on the driver's ``DataContext``
    here - before ``run_embedding_pipeline`` opens the first Lance scan - makes
    every modality's seed / read / fill pass render with the rich UI.
    """
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False


def main() -> None:
    """Enable the rich progress UI, then either reset groups or run the recipe.

    Catch a driver-side ``ValueError`` (invalid CLI inputs such as
    ``--max-fragments 0`` - pydantic's ``ValidationError`` is a ``ValueError``
    subclass - a stale group needing ``--reset-group``, a modality that committed
    nothing while its fragments failed, or a missing clips table), log it at ERROR,
    and exit non-zero so the operator sees the actionable message as the last line
    rather than the tail of a traceback that buries it. Config assembly is inside
    the guarded block so a bad CLI value exits cleanly too.
    """
    _enable_rich_progress_ui()
    args = _parse_args()
    try:
        if args.reset_group:
            _reset_groups(args)
            return
        result = run_embedding_pipeline(_build_config(args))
    except ValueError as error:
        logger.error(str(error))
        sys.exit(1)
    _log_summary(result)


if __name__ == "__main__":
    main()
