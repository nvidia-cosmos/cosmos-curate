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

"""Validation tests for the Curate config: the two cross-field invariants.

Pins what ``CurateConfig`` refuses to build - weights that break the unit-norm
fusion identity, a target expressed twice, a removed key, a blank table URI - the
envelope that makes a file routable, and the keep-count arithmetic of
``SelectionTarget.resolve``. One behavior per test.
"""

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.curation import config as curation_config
from cosmos_curator.next.recipes.curation import fairness as curation_fairness
from cosmos_curator.next.recipes.curation.columns import (
    FUSED_BLOCKS,
    TASK_COLUMN,
    TASK_VECTOR_COLUMN,
    CurateReason,
)
from cosmos_curator.next.recipes.curation.config import (
    _MIN_DEDUP_EPS,
    CurateConfig,
    ModalityWeights,
    SelectionTarget,
    contract_fingerprint,
    resolve_config,
)
from cosmos_curator.next.recipes.curation.fairness import canonicalization_contract
from cosmos_curator.next.recipes.curation.vectors import BLOCK_DIMS


def _config(**overrides: object) -> CurateConfig:
    """Build the smallest valid Curate config, applying keyword overrides."""
    base: dict[str, object] = {
        "schema_version": 1,
        "kind": "curate",
        "clips_lance_uri": "s3://bucket/clips.lance",
    }
    base.update(overrides)
    return CurateConfig(**base)


class TestModalityWeights:
    """The weights are an interface, and the sum-to-1 rule is what makes it one."""

    def test_defaults_are_the_pinned_interface_values(self) -> None:
        """The subtask text block is the intended dominant curation term."""
        weights = ModalityWeights()
        assert (weights.subtask, weights.image, weights.action) == (0.6, 0.2, 0.2)

    def test_weights_that_do_not_sum_to_one_are_rejected(self) -> None:
        """Off-sum weights break the unit-norm identity, so they fail on the driver."""
        with pytest.raises(ValidationError, match="sum to 1"):
            ModalityWeights(subtask=0.5, image=0.2, action=0.2)

    def test_a_negative_weight_is_rejected(self) -> None:
        """A negative weight makes sqrt(w) NaN, so it never reaches a GPU task.

        The rejection comes from the field's ``ge=0.0`` bound, not from the
        sum-to-1 validator, so these weights sum to exactly 1 and keep every
        other field inside its bounds: the lower bound is then the only thing
        that can refuse them.
        """
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ModalityWeights(subtask=1.0, image=-0.2, action=0.2)

    def test_a_zero_weight_block_is_legal(self) -> None:
        """Dropping a modality is a supported config, not a validation error."""
        weights = ModalityWeights(subtask=0.0, image=0.5, action=0.5)
        assert weights.subtask == 0.0

    def test_a_sum_error_far_above_the_tolerance_is_rejected(self) -> None:
        """The tolerance is a float-noise allowance, not a slack budget.

        These weights are off by 1e-5 -- ten times the tolerance, but small
        enough to read as a typo rather than a mistake. Pinning an error of
        this size is what constrains the tolerance's MAGNITUDE; a fixture off
        by a visible margin would still pass with the tolerance loosened by
        several orders of magnitude.
        """
        with pytest.raises(ValidationError, match="sum to 1"):
            ModalityWeights(subtask=0.6, image=0.2, action=0.19999)


class TestSelectionTarget:
    """At most one form of the keep-count may be set."""

    def test_neither_form_is_set_by_default(self) -> None:
        """An unset target is legal and means "keep every survivor"."""
        target = SelectionTarget()
        assert (target.target_count, target.target_fraction) == (None, None)

    def test_setting_both_forms_is_rejected(self) -> None:
        """A target expressed twice has no single answer."""
        with pytest.raises(ValidationError, match="at most one"):
            SelectionTarget(target_count=10, target_fraction=0.5)

    def test_a_fraction_above_one_is_rejected(self) -> None:
        """The denominator is the survivor population, so no fraction exceeds it."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SelectionTarget(target_fraction=1.5)

    def test_a_negative_count_is_rejected(self) -> None:
        """A negative count would survive ``resolve`` and reach the fairness quota.

        Unlike the fraction, zero is legal here and means "keep nothing", so the
        bound guards only the negative side -- and it has to, because
        ``min(target_count, survivor_count)`` propagates a negative unchanged
        rather than clamping it to zero.
        """
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SelectionTarget(target_count=-5)

    def test_a_zero_fraction_is_rejected(self) -> None:
        """Zero is not a way to say "keep nothing", because resolve cannot honor it.

        The floor-of-one clamp would lift a zero fraction back to one kept row,
        so accepting 0.0 would mean silently keeping a row the operator asked
        not to keep. The strict floor refuses the request instead of
        reinterpreting it; a run that wants nothing selected does not run.
        """
        with pytest.raises(ValidationError, match=r"target_fraction[\s\S]*greater than 0"):
            SelectionTarget(target_fraction=0.0)


class TestSelectionTargetResolve:
    """``resolve`` turns a target into a concrete keep-count for one population."""

    def test_an_unset_target_keeps_every_survivor(self) -> None:
        """No target means the whole survivor population is kept."""
        assert SelectionTarget().resolve(7) == 7

    def test_a_count_target_clamps_to_the_population(self) -> None:
        """Asking for more rows than survived keeps only what survived."""
        assert SelectionTarget(target_count=100).resolve(7) == 7

    def test_a_fraction_target_rounds_half_up(self) -> None:
        """Half of 5 is 2.5 and resolves to 3; banker's rounding would give 2."""
        assert SelectionTarget(target_fraction=0.5).resolve(5) == 3

    def test_a_fraction_target_keeps_at_least_one_row(self) -> None:
        """A fraction that rounds to zero over a non-empty population keeps one row."""
        assert SelectionTarget(target_fraction=0.01).resolve(10) == 1

    def test_an_empty_population_resolves_to_zero(self) -> None:
        """No survivors means nothing to keep, whatever the target says.

        The fraction form is what pins the empty-population guard: a count
        clamps to zero arithmetically, but a fraction would be lifted back to
        one row by the floor-of-one clamp if the guard stopped firing.
        """
        assert SelectionTarget(target_fraction=0.5).resolve(0) == 0
        assert SelectionTarget(target_count=5).resolve(0) == 0


class TestCurateConfig:
    """The flat surface: strict, frozen, and ``extra="forbid"``."""

    def test_the_sub_models_carry_their_own_defaults(self) -> None:
        """The top-level config composes the two invariant carriers by default."""
        cfg = _config()
        assert cfg.weights.subtask == 0.6
        assert cfg.target.target_count is None
        assert cfg.storage_profile == "default"

    def test_within_group_order_defaults_to_farthest(self) -> None:
        """The abundant-data regime default keeps atypical (farthest) rows."""
        assert _config().within_group_order == "farthest"

    def test_an_unknown_within_group_order_is_rejected(self) -> None:
        """An unrecognized ordering is a config error, not a silent fallback."""
        with pytest.raises(ValidationError, match="'farthest', 'nearest' or 'neutral'"):
            _config(within_group_order="random")

    def test_an_unknown_key_is_rejected_and_named_in_the_error(self) -> None:
        """Any key this model does not declare is refused, and the error names it.

        ``staging_root`` is illustrative -- it is a field the wide-table
        migration removed, so it is what a stale config is likely to carry --
        but the mechanism is name-agnostic: every unknown key is refused
        identically. What the assertion pins is the second half, that the
        offending key appears in the message, which is what makes a stale
        config actionable rather than merely rejected.
        """
        with pytest.raises(ValidationError, match="staging_root"):
            _config(staging_root="/lustre/scratch")

    def test_a_quoted_number_is_not_silently_coerced(self) -> None:
        """Strict mode refuses a string where a number is declared.

        Lax mode would parse ``"200000"`` into the int and run with a value the
        operator never checked. Under strict mode a quoted number in a config
        file is a parse error naming the field, which is the same bargain as
        ``extra="forbid"``: a migration fails loudly instead of proceeding on a
        value nobody chose.
        """
        with pytest.raises(ValidationError, match="target_mean_cluster_rows"):
            _config(target_mean_cluster_rows="200000")

    @pytest.mark.parametrize(
        "field",
        ["target_mean_cluster_rows", "fit_sample_rows", "dedup_concurrency"],
    )
    def test_a_count_field_rejects_zero(self, field: str) -> None:
        """Each of these three counts has no meaning at zero, and fails late if allowed.

        ``target_mean_cluster_rows`` is the divisor in the cluster-count
        derivation, so zero is a division by zero on the driver.
        ``fit_sample_rows`` at zero asks the k-means fit to derive centroids
        from no rows. ``dedup_concurrency`` at zero asks for a GPU stage that
        can never run a task -- and ``None``, not zero, is how a run says
        "use every device". Each bound converts a failure deep in a remote task
        into a config error.
        """
        with pytest.raises(ValidationError, match=field):
            _config(**{field: 0})

    def test_a_merge_theta_above_one_is_rejected(self) -> None:
        """A threshold above the maximum possible cosine silently disables merging.

        Theta is compared against a cosine similarity, which cannot exceed 1,
        so 1.5 is not a strict setting but a no-op: no label pair ever clears
        it, every wording variant keeps its own fairness quota, and the run
        reports success with a merge that did nothing. The bound is what makes
        that unreachable rather than undetectable.
        """
        with pytest.raises(ValidationError, match="merge_theta_task"):
            _config(merge_theta_task=1.5)

    def test_a_subtask_cluster_count_below_one_is_rejected(self) -> None:
        """Level 2 needs at least one cell, because zero cells is a partition of nothing.

        ``subtask_clusters`` is the number of centroids the fit produces and
        therefore the level-2 group count per task. At zero the fit would be
        asked for an empty basis and every row would fall to the reserved
        absent cell, collapsing level-2 fairness to level 1 while the run still
        reported success.
        """
        with pytest.raises(ValidationError, match="subtask_clusters"):
            _config(subtask_clusters=0)

    def test_the_previous_spelling_of_the_subtask_cell_count_is_refused_by_name(self) -> None:
        """The rename carries NO alias, so a stale config fails at parse time.

        This is the whole migration story for the field: ``extra="forbid"`` names
        the offending key, so an operator learns their config is stale before a
        GPU is claimed. An alias would instead let the old key keep working and
        the two spellings drift.
        """
        with pytest.raises(ValidationError, match="subtask_cluster_k"):
            _config(subtask_cluster_k=16)

    def test_an_unset_threshold_skips_de_duplication(self) -> None:
        """``None`` is how a run asks for no de-duplication, and it is accepted.

        The two spellings of "drop nothing" mean different things and only one of
        them is honest, because the retention test is a strict
        ``score > 1 - eps`` and an identical pair scores exactly 1.0. ``None``
        skips the GPU pass; ``0.0`` would pay for it and drop nothing, including
        the exact copies a reader expects it to catch. The paired test below pins
        the other half.
        """
        assert _config(dedup_eps=None).dedup_eps is None

    def test_a_zero_threshold_is_rejected_by_its_lower_bound(self) -> None:
        """Zero is refused by the bound on the float, not by the optional union.

        The distinction is the contract: were the union to reject it, ``0.0``
        would be indistinguishable from any other ill-typed value, and the
        message would not tell an operator that the neighbouring ``None`` is the
        spelling they wanted. The bound belongs to the operator surface only --
        ``dedup.duplicate_mask`` is still callable at 0.0, which is what lets its
        own boundary test read a threshold off a score and hit it exactly.
        """
        with pytest.raises(ValidationError, match=r"dedup_eps[\s\S]*greater than 0"):
            _config(dedup_eps=0.0)

    def test_an_eps_too_small_for_float32_is_rejected_rather_than_silently_inert(self) -> None:
        """A positive eps whose threshold rounds back to 1.0f is refused at parse time.

        The retention pass compares float32 scores against ``float32(1 - eps)``.
        At 1e-9 that cast lands back on exactly 1.0, which no cosine exceeds, so
        the run would schedule a whole-GPU pass per cluster and drop nothing. It
        is caught here rather than in ``duplicate_mask`` because a raise inside a
        ``map_groups`` UDF arrives with its type erased, after the GPU stage has
        already been scheduled.
        """
        assert np.float32(1.0 - 1e-9) == np.float32(1.0)

        with pytest.raises(ValidationError, match=r"dedup_eps[\s\S]*set it to null"):
            _config(dedup_eps=1e-9)

    def test_the_smallest_accepted_eps_still_reaches_a_usable_threshold(self) -> None:
        """The floor is one float32 step below 1.0, so its own threshold is usable.

        Pinned against the cast rather than restated as a literal: a floor that
        drifted down toward the exact round-off boundary would rest on how a
        float64 subtraction rounds, and one that drifted up would turn away
        thresholds a whole step clear of 1.0.
        """
        assert _config(dedup_eps=_MIN_DEDUP_EPS).dedup_eps == _MIN_DEDUP_EPS
        assert np.float32(1.0 - _MIN_DEDUP_EPS) < np.float32(1.0)

    def test_a_blank_table_uri_is_rejected(self) -> None:
        """A whitespace-only table URI is rejected before any read is attempted."""
        with pytest.raises(ValidationError, match="must not be blank or whitespace-only"):
            _config(clips_lance_uri="   ")

    def test_the_config_is_frozen(self) -> None:
        """The assembled config is immutable, so a run cannot be re-aimed mid-flight."""
        cfg = _config()
        with pytest.raises(ValidationError, match="frozen"):
            cfg.storage_profile = "other"


class TestConfigEnvelope:
    """The two fields that make a file routable, and the resolver every surface shares."""

    def test_a_config_naming_another_pipeline_is_refused(self) -> None:
        """A misrouted file is rejected rather than read under Curate's field meanings.

        Both trees use ``extra="forbid"``, so the fields of another kind would
        already fail; pinning the discriminator makes the FIRST error name the
        actual mistake instead of listing every unknown key.
        """
        with pytest.raises(ValidationError, match="Input should be 'curate'"):
            _config(kind="embeddings")

    def test_a_future_schema_generation_is_refused(self) -> None:
        """A version this generation cannot read fails instead of being reinterpreted.

        This is why the field is required rather than defaulted: a default would
        make a generation-2 file parse under generation-1 field meanings, which
        is the one failure the envelope exists to prevent.
        """
        with pytest.raises(ValidationError, match="Input should be 1"):
            _config(schema_version=2)

    def test_a_dotted_override_reaches_a_nested_field(self, tmp_path: Path) -> None:
        """``--set`` assigns into a sub-model that the file never mentions.

        The path is created on the way down, so reaching ``target.target_fraction``
        does not require the file to carry a ``target`` block at all.
        """
        config_path = tmp_path / "curate.yaml"
        config_path.write_text(
            yaml.safe_dump({"schema_version": 1, "kind": "curate", "clips_lance_uri": "s3://bucket/clips.lance"}),
            encoding="utf-8",
        )

        resolved = resolve_config(config_path, overrides=["target.target_fraction=0.25"])

        assert resolved.target.target_fraction == 0.25

    def test_an_override_is_checked_by_the_cross_field_rules(self, tmp_path: Path) -> None:
        """An override lands before validation, so it cannot slip past a cross-field rule.

        Ordering is the whole contract: were the override applied to the built
        model instead, patching one weight would leave the summed-to-1 invariant
        unchecked, and the fused vector silently non-unit-norm.
        """
        config_path = tmp_path / "curate.yaml"
        config_path.write_text(
            yaml.safe_dump({"schema_version": 1, "kind": "curate", "clips_lance_uri": "s3://bucket/clips.lance"}),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="must sum to 1"):
            resolve_config(config_path, overrides=["weights.subtask=0.9"])

    def test_a_malformed_file_fails_as_a_value_error_not_as_the_parser_s_own_type(self, tmp_path: Path) -> None:
        """An unparseable config raises ValueError rather than a yaml or json exception.

        Every surface that presents config faults cleanly catches ValueError, so a
        parser's own exception type would escape as an unhandled error. The routed
        CLI path cannot show this -- it parses the file once already, to read
        ``kind``, and wraps the fault there -- so only a direct caller of this
        resolver is exposed, which is the path the recipe README documents.
        """
        config_path = tmp_path / "curate.yaml"
        config_path.write_text('clips_lance_uri: "unclosed\n', encoding="utf-8")

        with pytest.raises(ValueError, match="not well-formed"):
            resolve_config(config_path)

    def test_a_top_level_list_is_a_config_fault_not_a_type_error(self, tmp_path: Path) -> None:
        """A YAML sequence at the root must surface as ValueError like parser faults."""
        config_path = tmp_path / "curate.yaml"
        config_path.write_text("- not a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="mapping at the top level"):
            resolve_config(config_path)

    def test_a_directory_path_is_a_config_fault_not_is_a_directory_error(self, tmp_path: Path) -> None:
        """A directory config path must raise ValueError, not IsADirectoryError from open()."""
        config_dir = tmp_path / "curate.yaml"
        config_dir.mkdir()

        with pytest.raises(ValueError, match="Config path must be a file"):
            resolve_config(config_dir)


class TestRunIdentity:
    """A config's digest is what makes two committed selections comparable."""

    def test_the_identity_covers_every_field_except_the_named_exclusions(self) -> None:
        """The digested text holds all config fields but the non-result-defining ones.

        This is the failsafe direction, and the reason to pin it: a field added to
        the model later joins the identity without anyone remembering to list it.
        Were the set built the other way round, a new threshold would be omitted
        and two runs applying different rules would report one identity.
        """
        digested = set(json.loads(_config().result_defining_json()))

        assert digested == (set(CurateConfig.model_fields) | {"__contract__"}) - {
            "clips_lance_uri",
            "storage_profile",
            "dedup_concurrency",
        }

    def test_changing_a_threshold_changes_the_identity(self) -> None:
        """A different duplicate threshold is a different run."""
        assert _config(dedup_eps=0.01).result_defining_digest() != _config(dedup_eps=0.02).result_defining_digest()

    def test_turning_de_duplication_off_changes_the_identity(self) -> None:
        """``dedup_eps=None`` is a rule change, not an absent value.

        Worth its own case because None is the one result-defining value that is
        not a number, so a serialization that dropped empty fields would let a
        run with de-duplication disabled collide with one that had it on.
        """
        assert _config(dedup_eps=None).result_defining_digest() != _config(dedup_eps=0.01).result_defining_digest()

    def test_changing_only_scheduling_or_location_keeps_the_identity(self) -> None:
        """Concurrency, credentials, and the table's address are not rules.

        Two runs differing only in these selected for the same thing, so they must
        report the same identity -- otherwise a digest comparison answers "these
        differ" for a table that was merely copied elsewhere or run on a bigger
        cluster.
        """
        moved = _config(
            clips_lance_uri="s3://other-bucket/clips.lance",
            storage_profile="secondary",
            dedup_concurrency=4,
        )

        assert moved.result_defining_digest() == _config().result_defining_digest()

    def test_the_digest_is_the_hash_of_the_text_that_gets_archived(self) -> None:
        """Digest and archived text agree, so the two published artifacts cross-check.

        The commit carries only the digest and the sidecar carries only the text.
        Recomputing one from the other is how a reader confirms a sidecar belongs
        to the version it sits beside.
        """
        config = _config()

        recomputed = hashlib.sha256(config.result_defining_json().encode("utf-8")).hexdigest()

        assert config.result_defining_digest() == f"sha256:{recomputed}"

    def test_the_canonical_text_is_a_fixed_published_format(self) -> None:
        """The exact bytes are pinned, because a digest of them is published.

        Sorted keys and tight separators are not style: a digest is only
        comparable against digests built the same way, so dropping either
        re-bases every value already committed while any test that merely
        compares two live configs keeps passing.

        Editing this literal is therefore a format change, and a field added to
        the model lands here on purpose -- the new digest is not comparable with
        anything committed before it.
        """
        config = _config(weights=ModalityWeights(subtask=0.5, image=0.3, action=0.2), dedup_eps=0.02)

        assert config.result_defining_json() == (
            '{"__contract__":{"canonical_task":[["e\\u0301","\\u00e9"],["\\u00b2","\\u00b2"],'
            '["a \\t\\n b","a b"],[" a.\\t","a"],["\\u00df","ss"],[".a .!?,;: ",".a"],[".!?,;: ",""]],'
            '"fused_blocks":[["subtask","embedding_text_subtask",384],'
            '["image","embedding_image",384],["action","embedding_action",97]],'
            '"reasons":["selected","duplicate","below_quota","unfunded","invalid_embedding"],'
            '"task_label":"task_name","task_vector":"embedding_text_task"},'
            '"dedup_eps":0.02,"fairness_residual_seed":0,"fit_sample_rows":4000000,"kind":"curate",'
            '"kmeans_random_state":42,"merge_theta_task":0.95,"schema_version":1,"subtask_clusters":16,'
            '"target":{"target_count":null,"target_fraction":null},"target_mean_cluster_rows":200000,'
            '"weights":{"action":0.2,"image":0.3,"subtask":0.5},"within_group_order":"farthest"}'
        )

    def test_a_contract_change_alone_changes_the_identity(self) -> None:
        """Two runs under different code contracts are not comparable, config aside.

        The block order, the block widths and the reason vocabulary decide the
        verdicts as surely as any config field does, but no config file names
        them. Were the digest to cover only the fields an operator writes, a
        release that reordered the fused blocks would leave every previously
        committed digest looking equal to the new incomparable ones.
        """
        config = _config()
        before = config.result_defining_digest()

        with mock.patch.object(
            curation_config,
            "contract_fingerprint",
            return_value={"fused_blocks": [["image", "embedding_image", 384]], "reasons": ["selected"]},
        ):
            after = config.result_defining_digest()

        assert before != after

    def test_the_contract_is_read_off_the_declarations_it_describes(self) -> None:
        """The fingerprint is derived, so a moved declaration cannot leave it stale.

        A hand-maintained contract version would satisfy the digest test above
        while still reporting one identity for two block orders, because nothing
        forces the bump. Pinning the payload against ``FUSED_BLOCKS``,
        ``BLOCK_DIMS``, ``CurateReason`` and the two task columns is what makes
        forgetting impossible rather than merely discouraged.

        The canonicalization key is the exception: both sides of that assertion
        re-derive through the same rule, so it pins only WHICH function feeds the
        key and cannot notice the fold changing. What covers the fold itself is
        ``test_a_changed_label_fold_alone_changes_the_identity`` below.
        """
        contract = contract_fingerprint()

        assert [field for _, field in FUSED_BLOCKS] == [field for field, _, _ in contract["fused_blocks"]]
        assert [group.primary_vector for group, _ in FUSED_BLOCKS] == [
            column for _, column, _ in contract["fused_blocks"]
        ]
        assert list(BLOCK_DIMS) == [width for _, _, width in contract["fused_blocks"]]
        assert [reason.value for reason in CurateReason] == contract["reasons"]
        assert contract["task_label"] == TASK_COLUMN
        assert contract["task_vector"] == TASK_VECTOR_COLUMN
        assert canonicalization_contract() == contract["canonical_task"]

    def test_a_changed_label_fold_alone_changes_the_identity(self) -> None:
        """Two runs that group labels differently are not comparable, config aside.

        Canonicalization decides which spellings share a fairness group, so it
        decides which rows compete for one quota and therefore which are selected.
        No config field names it, so without it in the contract a release that
        changed a folding step would report the old and new runs as one identity.
        """
        config = _config()
        before = config.result_defining_digest()

        with mock.patch.object(curation_fairness, "canonicalize_label", lambda text: text):
            after = config.result_defining_digest()

        assert before != after
