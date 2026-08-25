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

"""Direct-``clips.lance`` embedding driver: widen the schema, fill each group in place.

``run_embedding_pipeline`` resolves the table's credentials once, adds the enabled
modalities' ``embedding_*`` columns in one metadata commit, then fills one group
per modality. A modality is DATA (a ``ModalityFill`` built by a small function),
so text and image share a plain loop with no branch; action is the one modality
with genuinely different BEHAVIOUR - it must bind one PCA basis before any worker
exists and its outcome is judged afterwards - and that lives in an explicit
branch rather than in a lifecycle hook::

    storage_options = get_lance_storage_options(uri, profile)   resolved once
        |
        v
    ensure_embedding_columns(enabled groups)   (one add_columns metadata commit)
        |
        v
    _stage_weights(text / image fills)         (pre-Ray fork)
        |
        v
    for fill in (text, image):
        reopen -> validate_embedding_group -> fill_embedding_group
        |
        v
    if action enabled:
        reopen -> validate_embedding_group -> resolve_action_pca
        |    |
        |    +-- None: no clip carries action data -> skipped, no actor pool
        |    |
        |    +-- basis: build_action_fill -> fill_embedding_group
        |                                     -> check_action_outcome(reopened)
        v
    EmbeddingRunResult(one ModalityResult per enabled modality, + the basis)

The modality boundary is sequential on purpose: Ray Data already fans each modality
across the cluster, and Lance treats same-fragment ``Update`` commits as conflicting
even on disjoint groups, so overlapping modalities would only serialize or repeat
fragment writes. Each modality re-opens the table, so image and action see the
columns and rows text just committed - which is why the driver passes an open
dataset around rather than a URI, making "which version is this reading" visible
at each call site.

"Incremental" means only rows whose primary vector is still NULL are computed, so a
top-up embeds exactly the new clips and an unchanged re-run finds every fragment
already complete, rewrites nothing, and commits nothing. Replacing a whole group is
NOT a run mode: it is the separate ``--reset-group`` maintenance operation, which
drops the group's columns and re-adds them empty so the next ordinary run refills
them.
"""

from collections.abc import Callable, Sequence

import attrs
import lance
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import download_models
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.embeddings.action.pca import action_pca_root_uri
from cosmos_curator.next.embeddings.schemas import ACTION_COLUMN_GROUP, EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.action_pca import ActionPca, check_action_outcome, resolve_action_pca
from cosmos_curator.next.recipes.embeddings.columns import ensure_embedding_columns, validate_embedding_group
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality
from cosmos_curator.next.recipes.embeddings.fill import fill_embedding_group
from cosmos_curator.next.recipes.embeddings.modalities import (
    ModalityFill,
    ModalityResult,
    build_action_fill,
    build_image_fill,
    build_text_fill,
)
from cosmos_curator.next.utils.lance_utils import open_dataset_or_raise

# The modalities whose whole run is described by their spec, in execution order.
# Action is absent by construction, not by omission: its fill cannot be built
# until a basis is resolved, so it needs the explicit branch below rather than a
# row in this table.
_GENERIC_FILL_BUILDERS: tuple[tuple[Modality, Callable[[EmbeddingPipelineConfig], ModalityFill]], ...] = (
    (Modality.TEXT, build_text_fill),
    (Modality.IMAGE, build_image_fill),
)


@attrs.frozen
class EmbeddingRunResult:
    """Verification summary for one run: per-modality counts and the action basis.

    Attributes:
        modalities: One result per enabled modality, in text -> image -> action
            order (including any that embedded nothing).
        action: The PCA basis the action modality bound, or ``None`` when action
            was not enabled or had nothing to embed. Carried here rather than on
            a ``ModalityResult`` because the fingerprint and the fit / reuse
            distinction are action-specific.

    """

    modalities: tuple[ModalityResult, ...]
    action: ActionPca | None

    @property
    def ending_version(self) -> int | None:
        """Return the version of this run's last commit, or ``None`` if it committed nothing.

        Derived rather than stored: every modality writes the same table, so the
        last non-``None`` ``committed_version`` IS the run's end state, and a
        second recorded field could only disagree with it.
        """
        committed = [result.committed_version for result in self.modalities if result.committed_version is not None]
        return committed[-1] if committed else None


def run_embedding_pipeline(config: EmbeddingPipelineConfig) -> EmbeddingRunResult:
    """Fill every enabled embedding column group directly on the shared ``clips.lance``.

    Args:
        config: Fully resolved run configuration.

    Returns:
        A verification summary (per-modality counts, the versions committed, and
        the action basis; never vectors).

    Raises:
        ValueError: If the table is absent or unreadable, a group is unsafe to
            fill, a modality committed nothing while its fragments failed, or a
            commit fails.

    """
    storage_options = get_lance_storage_options(config.clips_lance_uri, profile_name=config.storage_profile)
    fills = _generic_fills(config)
    logger.info(f"embedding clips at {config.clips_lance_uri}; modalities={list(config.modalities)}")

    dataset = _open_clips(config, storage_options=storage_options)
    added = ensure_embedding_columns(dataset, _enabled_groups(fills, config))
    logger.info(f"ensured embedding columns for {list(config.modalities)}: {added} field(s) added")

    # Before the first fill, because a fill starts Ray Data and download_models
    # must fork from a Ray-free process.
    _stage_weights(fills, config)

    results = [_run_generic_fill(fill, config=config, storage_options=storage_options) for fill in fills]
    action: ActionPca | None = None
    if Modality.ACTION in config.modalities:
        action_result, action = _run_action_fill(config, storage_options=storage_options)
        results.append(action_result)
    return EmbeddingRunResult(modalities=tuple(results), action=action)


def _generic_fills(config: EmbeddingPipelineConfig) -> tuple[ModalityFill, ...]:
    """Build the enabled spec-driven modalities' fills, in execution order."""
    return tuple(build(config) for modality, build in _GENERIC_FILL_BUILDERS if modality in config.modalities)


def _enabled_groups(fills: Sequence[ModalityFill], config: EmbeddingPipelineConfig) -> list[EmbeddingColumnGroup]:
    """Return every enabled modality's column group, so one commit widens them all.

    Action's group is named directly rather than read off a fill: its fill does
    not exist yet (it needs a basis, and resolving one reads the group's own
    provenance columns), so the columns must be added first.
    """
    groups = [fill.group for fill in fills]
    if Modality.ACTION in config.modalities:
        groups.append(ACTION_COLUMN_GROUP)
    return groups


def _stage_weights(fills: Sequence[ModalityFill], config: EmbeddingPipelineConfig) -> None:
    """Download the enabled modalities' checkpoints before any Ray Data execution.

    ``download_models`` forks a child that initialises and tears down its own Ray,
    so it must run from a process that has never initialised Ray - i.e. before the
    first fill, and before the action leg's PCA candidate pass. Only the
    spec-driven fills are inspected, which is complete rather than partial: the
    action modality loads no checkpoint, so an action-only run stages nothing.
    """
    names = [fill.weights_name for fill in fills if fill.weights_name is not None]
    if not names:
        return
    logger.info(f"staging weights for {names} from {config.model_weights_path}")
    download_models(names, config.model_weights_path)


def _run_generic_fill(
    fill: ModalityFill, *, config: EmbeddingPipelineConfig, storage_options: dict[str, str] | None
) -> ModalityResult:
    """Validate and fill one modality whose entire run is described by its spec."""
    dataset = _open_clips(config, storage_options=storage_options)
    validate_embedding_group(dataset, fill.group, expected_provenance=fill.expected_provenance)
    return fill_embedding_group(dataset, fill, storage_options=storage_options, max_fragments=config.max_fragments)


def _run_action_fill(
    config: EmbeddingPipelineConfig, *, storage_options: dict[str, str] | None
) -> tuple[ModalityResult, ActionPca | None]:
    """Bind the run's one PCA basis, then fill the action group and judge the outcome.

    The ``None`` basis is load-bearing rather than defensive: with no clip
    carrying action data there is nothing to load and nothing to fit, so the run
    reports a skip WITHOUT starting an actor pool.

    Returns:
        The action modality's result, and the basis it bound (``None`` when the
        leg was skipped).

    """
    dataset = _open_clips(config, storage_options=storage_options)
    # No expected provenance: action's producer identity is the PCA fingerprint
    # its own rows carry, so validation enforces only single-producer uniqueness
    # and resolve_action_pca then reads that fingerprint back.
    validate_embedding_group(dataset, ACTION_COLUMN_GROUP, expected_provenance=None)
    pca = resolve_action_pca(dataset, config, root_uri=action_pca_root_uri(config.clips_lance_uri))
    if pca is None:
        return ModalityResult.nothing_to_do(Modality.ACTION), None

    result = fill_embedding_group(
        dataset,
        build_action_fill(config, pca.artifact),
        storage_options=storage_options,
        max_fragments=config.max_fragments,
    )
    # Re-opened, so the "the group holds nothing at all" test sees this run's own
    # commit rather than the pre-fill version the fill was planned against.
    check_action_outcome(result, _open_clips(config, storage_options=storage_options))
    return result, pca


def _open_clips(config: EmbeddingPipelineConfig, *, storage_options: dict[str, str] | None) -> lance.LanceDataset:
    """Open the clips table at its latest version, raising if it is absent."""
    return open_dataset_or_raise(config.clips_lance_uri, storage_options=storage_options)
