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

"""Fit / project / fingerprint / content-addressed-store tests for the action PCA basis.

The store is content-addressed: a basis lives at ``<root>/<fingerprint>.npz`` and
``load(fingerprint)`` reads exactly that object. The tests cover the pure numpy
fit and projection numerics, the immutable fingerprint, and the store's
idempotent ``save_if_absent`` / verified ``load`` contract (missing artifact,
fingerprint/content mismatch, and archive-validation rejections).
"""

import pathlib
from typing import Any

import numpy as np
import pytest

from cosmos_curator.next.embeddings.action.pca import (
    _MIN_EXPLAINED_VARIANCE_WARN,
    PcaArtifact,
    PcaArtifactStore,
    action_pca_artifact_uri,
    action_pca_root_uri,
    fit_action_pca,
)
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.schemas import ACTION_DIM


def _fittable(rows: int, *, seed: int = 0) -> np.ndarray:
    """Return a full-rank ``(rows, 600)`` sample with real variance."""
    return np.random.default_rng(seed).standard_normal((rows, DESCRIPTOR_DIM))


def _valid_archive_members() -> dict[str, Any]:
    """Return a complete, current-format set of ``.npz`` members (eight keys)."""
    return {
        "mean": np.zeros(DESCRIPTOR_DIM),
        "components": np.zeros((ACTION_DIM, DESCRIPTOR_DIM)),
        "descriptor_dim": np.int64(DESCRIPTOR_DIM),
        "n_components": np.int64(ACTION_DIM),
        "n_fit_rows": np.int64(100),
        "descriptor_version": np.asarray(DESCRIPTOR_VERSION),
        "fit_timestamp": np.asarray("x"),
        "explained_variance_ratio": np.float64(0.9),
    }


def _write_artifact(root: pathlib.Path, fingerprint: str, members: dict[str, Any]) -> str:
    """Write ``members`` as the ``.npz`` object named by ``fingerprint`` under ``root``.

    Bypasses ``save_if_absent`` on purpose so a test can plant a hand-crafted
    (possibly invalid or content-mismatched) archive at the exact URI ``load``
    resolves for that fingerprint.
    """
    uri = action_pca_artifact_uri(str(root), fingerprint)
    with pathlib.Path(uri).open("wb") as handle:
        np.savez(handle, **members)
    return uri


def test_fit_produces_expected_shapes_and_provenance() -> None:
    """A healthy fit yields (97, 600) components, a (600,) mean, and recorded provenance."""
    pca = fit_action_pca(_fittable(200))
    assert pca.components.shape == (ACTION_DIM, DESCRIPTOR_DIM)
    assert pca.mean.shape == (DESCRIPTOR_DIM,)
    assert pca.n_components == ACTION_DIM
    assert pca.n_fit_rows == 200
    assert pca.descriptor_version == DESCRIPTOR_VERSION


def test_project_maps_batch_and_single_vector() -> None:
    """Projection reduces (N, 600) -> (N, 97) and (600,) -> (97,)."""
    pca = fit_action_pca(_fittable(200))
    assert pca.project(_fittable(5, seed=1)).shape == (5, ACTION_DIM)
    assert pca.project(_fittable(1, seed=2)[0]).shape == (ACTION_DIM,)


def test_fit_rejects_too_few_rows() -> None:
    """K <= n_components cannot support a full basis (mean-centering costs a DOF)."""
    with pytest.raises(ValueError, match="need >"):
        fit_action_pca(_fittable(ACTION_DIM))


def test_fit_rejects_zero_variance() -> None:
    """Identical descriptors have no basis to learn."""
    with pytest.raises(ValueError, match="zero total variance"):
        fit_action_pca(np.ones((200, DESCRIPTOR_DIM)))


def test_fit_rejects_rank_deficient_sample() -> None:
    """A sample spanning fewer than n_components directions is rejected."""
    rng = np.random.default_rng(3)
    low_rank = rng.standard_normal((200, 50)) @ rng.standard_normal((50, DESCRIPTOR_DIM))
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_action_pca(low_rank)


def test_fit_rejects_wrong_width() -> None:
    """Descriptors must be exactly 600 wide."""
    with pytest.raises(ValueError, match=r"must be \(K, 600\)"):
        fit_action_pca(np.zeros((200, 500)))


def test_fit_rejects_non_2d() -> None:
    """A 1-D input is not a (K, 600) matrix."""
    with pytest.raises(ValueError, match="2-D"):
        fit_action_pca(np.zeros(DESCRIPTOR_DIM))


def test_fit_rejects_non_finite_descriptors() -> None:
    """A NaN in the sample fails with a finiteness message, not a misleading rank error."""
    sample = _fittable(200)
    sample[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_action_pca(sample)


def test_project_rejects_broadcastable_but_wrong_shape() -> None:
    """A (600, 1) column vector broadcasts against the mean but is not a descriptor."""
    pca = fit_action_pca(_fittable(200))
    with pytest.raises(ValueError, match=r"must be \(600,\) or \(N, 600\)"):
        pca.project(np.zeros((DESCRIPTOR_DIM, 1)))


def test_components_form_an_orthonormal_basis() -> None:
    """PCA rows are mutually orthogonal unit vectors, so the projection is a rotation.

    A random matrix (which every shape/metadata test would still accept) fails
    this: it is the identity that makes the output a PCA basis, not an arbitrary
    linear map.
    """
    pca = fit_action_pca(_fittable(200))
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(ACTION_DIM), atol=1e-9)


def test_projecting_the_fit_mean_yields_the_origin() -> None:
    """The mean of the fit population projects to zero (centering is applied, once)."""
    pca = fit_action_pca(_fittable(200))
    np.testing.assert_allclose(pca.project(pca.mean), np.zeros(ACTION_DIM), atol=1e-9)


def test_leading_components_capture_the_most_variance() -> None:
    """Projected coordinates have non-increasing variance: the basis keeps the TOP directions.

    Orthonormality and the centering identity both hold for the bottom-k singular
    vectors too, so only this ordering identity catches a components-reversed bug
    (storing the least-variance directions while still reporting the top-k ratio).
    """
    sample = _fittable(400)
    pca = fit_action_pca(sample)
    per_component_variance = pca.project(sample).var(axis=0)
    assert np.all(np.diff(per_component_variance) <= 1e-9)


def test_fit_is_sign_reproducible_and_canonical() -> None:
    """A deliberate re-fit of the same sample reproduces the same basis with a pinned sign.

    Same-process determinism alone does not prove the convention (LAPACK returns
    stable signs within one process); the canonical clause -- each component's
    largest-magnitude entry is positive -- is what makes the basis reproducible
    across BLAS builds.
    """
    first = fit_action_pca(_fittable(200))
    second = fit_action_pca(_fittable(200))
    np.testing.assert_array_equal(first.components, second.components)
    rows = np.arange(first.components.shape[0])
    largest = np.abs(first.components).argmax(axis=1)
    assert np.all(first.components[rows, largest] > 0.0)


def test_fit_records_explained_variance_ratio() -> None:
    """A healthy fit records a finite retained-variance fraction in (0, 1]."""
    pca = fit_action_pca(_fittable(150))
    assert 0.0 < pca.explained_variance_ratio <= 1.0


def test_poor_fit_warns_below_floor(loguru_records: list[dict[str, Any]]) -> None:
    """A fit whose retained variance is below the floor warns and records the low ratio.

    A wide, high-rank population spreads variance across far more than ACTION_DIM
    directions, so 97 components capture well under the floor; the warning is how
    an operator learns the basis may be inadequate for that population.
    """
    pca = fit_action_pca(_fittable(400))
    assert pca.explained_variance_ratio < _MIN_EXPLAINED_VARIANCE_WARN
    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert any("below the" in message for message in warnings)


def test_fit_rejects_n_components_above_descriptor_dim() -> None:
    """Requesting more components than the descriptor is wide is impossible, not a data problem."""
    with pytest.raises(ValueError, match=r"n_components must be in \[1, 600\]"):
        fit_action_pca(_fittable(200), n_components=DESCRIPTOR_DIM + 1)


def test_action_pca_root_uri_is_a_sibling_directory_regardless_of_trailing_slash() -> None:
    """The PCA root is a directory beside the clips table, not nested inside it, slash or not."""
    expected = "s3://b/clips.lance__action_pca"
    assert action_pca_root_uri("s3://b/clips.lance/") == expected
    assert action_pca_root_uri("s3://b/clips.lance") == expected


def test_action_pca_artifact_uri_names_the_object_by_fingerprint() -> None:
    """One basis is addressed by ``<root>/<fingerprint>.npz`` (content addressing)."""
    assert action_pca_artifact_uri("s3://b/clips.lance__action_pca", "abc123") == (
        "s3://b/clips.lance__action_pca/abc123.npz"
    )


def test_save_if_absent_persists_at_fingerprint_uri_and_load_round_trips(tmp_path: pathlib.Path) -> None:
    """save_if_absent writes ``<root>/<fingerprint>.npz`` and load(fingerprint) restores the basis."""
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))

    uri = store.save_if_absent(pca)

    assert uri == action_pca_artifact_uri(str(tmp_path), pca.fingerprint)
    loaded = store.load(pca.fingerprint)
    np.testing.assert_array_equal(loaded.mean, pca.mean)
    np.testing.assert_array_equal(loaded.components, pca.components)
    assert loaded.descriptor_version == pca.descriptor_version
    assert loaded.explained_variance_ratio == pytest.approx(pca.explained_variance_ratio)


def test_save_if_absent_is_idempotent(tmp_path: pathlib.Path) -> None:
    """Saving the same basis twice returns the same URI without error (content addressing)."""
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))

    first = store.save_if_absent(pca)
    second = store.save_if_absent(pca)

    assert first == second
    assert store.load(pca.fingerprint).fingerprint == pca.fingerprint


def test_save_if_absent_and_load_round_trip_under_a_file_uri_root(tmp_path: pathlib.Path) -> None:
    """A ``file://`` root persists the basis where load() looks for it.

    The root is derived from the recipe's clips URI, which an operator may give in
    the ``file://`` form that local artifact URIs are otherwise recorded in. The
    object must land at the path the URI names so the fingerprint a Lance row
    references still resolves.
    """
    pca = fit_action_pca(_fittable(200))
    root = tmp_path / "clips.lance__action_pca"
    store = PcaArtifactStore(root.as_uri())

    uri = store.save_if_absent(pca)

    assert uri == action_pca_artifact_uri(root.as_uri(), pca.fingerprint)
    assert (root / f"{pca.fingerprint}.npz").is_file()
    assert store.load(pca.fingerprint).fingerprint == pca.fingerprint


def test_save_if_absent_under_a_file_uri_root_skips_a_redundant_write(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The presence gate short-circuits the second save of an already-persisted basis.

    The gate is what keeps the write idempotent under a ``file://`` root; if it
    failed to see the existing object, every rerun would rewrite an immutable
    artifact that other runs may be reading.
    """
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore((tmp_path / "root").as_uri())
    first = store.save_if_absent(pca)

    def _fail_on_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("save_if_absent rewrote an artifact that was already present")

    monkeypatch.setattr(PcaArtifactStore, "_write", _fail_on_write)

    assert store.save_if_absent(pca) == first


def test_save_if_absent_rejects_divergent_bytes_at_the_same_fingerprint(tmp_path: pathlib.Path) -> None:
    """A pre-existing object whose content differs from the basis being saved is a corruption error.

    Content addressing means the object name IS its content hash, so bytes at that
    name that do not hash back to it are corruption; save_if_absent refuses to
    overwrite rather than silently trust or clobber them.
    """
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))
    # Plant a different (but valid) archive at the URI pca would occupy.
    _write_artifact(tmp_path, pca.fingerprint, _valid_archive_members())

    with pytest.raises(ValueError, match="different content fingerprint"):
        store.save_if_absent(pca)


def test_load_missing_fingerprint_fails_with_rebuild_hint(tmp_path: pathlib.Path) -> None:
    """Loading a fingerprint with no object present names the missing artifact and suggests rebuild."""
    store = PcaArtifactStore(str(tmp_path))
    with pytest.raises(ValueError, match="missing at"):
        store.load("deadbeef")


def test_load_rejects_content_fingerprint_mismatch(tmp_path: pathlib.Path) -> None:
    """An object whose bytes do not hash to the requested name is rejected (not silently trusted)."""
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))
    # Write a valid basis under the WRONG name; its own fingerprint != "wrongname".
    store._write(action_pca_artifact_uri(str(tmp_path), "wrongname"), pca)

    with pytest.raises(ValueError, match="does not match the requested"):
        store.load("wrongname")


def test_load_rejects_stale_descriptor_version(tmp_path: pathlib.Path) -> None:
    """A basis fit under the old descriptor semantics is refused on load."""
    members = _valid_archive_members()
    members["descriptor_version"] = np.asarray("dual-wrist-v1")
    _write_artifact(tmp_path, "stale", members)
    with pytest.raises(ValueError, match="descriptor-version mismatch"):
        PcaArtifactStore(str(tmp_path)).load("stale")


def test_load_rejects_shape_mismatch(tmp_path: pathlib.Path) -> None:
    """A basis whose mean width disagrees with the descriptor width is refused."""
    members = _valid_archive_members()
    members["mean"] = np.zeros(500)
    _write_artifact(tmp_path, "badshape", members)
    with pytest.raises(ValueError, match="shape mismatch"):
        PcaArtifactStore(str(tmp_path)).load("badshape")


def test_load_rejects_wrong_component_count(tmp_path: pathlib.Path) -> None:
    """A basis retaining a different component count than this code produces is refused."""
    wrong = ACTION_DIM + 1
    members = _valid_archive_members()
    members["components"] = np.zeros((wrong, DESCRIPTOR_DIM))
    members["n_components"] = np.int64(wrong)
    _write_artifact(tmp_path, "wide", members)
    with pytest.raises(ValueError, match="component-count mismatch"):
        PcaArtifactStore(str(tmp_path)).load("wide")


def test_load_rejects_non_finite_mean(tmp_path: pathlib.Path) -> None:
    """A corrupted basis whose mean contains NaN is refused before projection."""
    mean = np.zeros(DESCRIPTOR_DIM)
    mean[0] = np.nan
    members = _valid_archive_members()
    members["mean"] = mean
    _write_artifact(tmp_path, "nanmean", members)
    with pytest.raises(ValueError, match="artifact validation failed: mean contains non-finite"):
        PcaArtifactStore(str(tmp_path)).load("nanmean")


def test_load_rejects_non_finite_components(tmp_path: pathlib.Path) -> None:
    """A corrupted basis whose components contain inf is refused before projection."""
    components = np.zeros((ACTION_DIM, DESCRIPTOR_DIM))
    components[0, 0] = np.inf
    members = _valid_archive_members()
    members["components"] = components
    _write_artifact(tmp_path, "infcomp", members)
    with pytest.raises(ValueError, match="artifact validation failed: components contain non-finite"):
        PcaArtifactStore(str(tmp_path)).load("infcomp")


def test_load_rejects_descriptor_dim_mismatch(tmp_path: pathlib.Path) -> None:
    """A basis recording a descriptor_dim that disagrees with its arrays is refused on load."""
    members = _valid_archive_members()
    members["descriptor_dim"] = np.int64(12345)
    _write_artifact(tmp_path, "bogus_dim", members)
    with pytest.raises(ValueError, match="descriptor-width mismatch"):
        PcaArtifactStore(str(tmp_path)).load("bogus_dim")


def test_load_rejects_foreign_archive_missing_keys(tmp_path: pathlib.Path) -> None:
    """A foreign .npz missing required members raises ValueError naming them, not KeyError."""
    _write_artifact(tmp_path, "foreign", {"mean": np.zeros(DESCRIPTOR_DIM)})
    with pytest.raises(ValueError, match="missing required key"):
        PcaArtifactStore(str(tmp_path)).load("foreign")


def test_load_accepts_seven_key_artifact_without_variance_ratio(tmp_path: pathlib.Path) -> None:
    """A pre-change 7-key artifact still loads; explained_variance_ratio reads as not-recorded.

    The stored archive omits ``explained_variance_ratio``; its own content
    fingerprint is what ``load`` verifies, so the object is planted under that
    exact fingerprint rather than an arbitrary name.
    """
    members = _valid_archive_members()
    del members["explained_variance_ratio"]
    seven_key = PcaArtifact(mean=members["mean"], components=members["components"])
    _write_artifact(tmp_path, seven_key.fingerprint, members)
    loaded = PcaArtifactStore(str(tmp_path)).load(seven_key.fingerprint)
    assert np.isnan(loaded.explained_variance_ratio)


def test_fitted_artifact_arrays_are_read_only() -> None:
    """A fitted basis cannot be mutated in place -- one instance is shared across projector workers."""
    pca = fit_action_pca(_fittable(200))
    assert not pca.mean.flags.writeable
    assert not pca.components.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        pca.components[0, 0] = 999.0


def test_loaded_artifact_arrays_are_read_only(tmp_path: pathlib.Path) -> None:
    """A loaded basis is equally immutable: both construction paths freeze the arrays."""
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))
    store.save_if_absent(pca)
    loaded = store.load(pca.fingerprint)
    assert not loaded.mean.flags.writeable
    assert not loaded.components.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        loaded.mean[0] = 1.0


def _basis(*, seed: int = 0, version: str = DESCRIPTOR_VERSION) -> PcaArtifact:
    """Build a PcaArtifact directly from deterministic arrays (fingerprint tests only).

    Construction runs no shape/rank validation, so arbitrary but real-shaped arrays
    are enough to exercise the fingerprint; a matching ``seed`` yields byte-identical
    arrays, which is what the equality tests rely on.
    """
    mean = np.random.default_rng(seed).standard_normal(DESCRIPTOR_DIM)
    components = np.random.default_rng(seed + 1).standard_normal((ACTION_DIM, DESCRIPTOR_DIM))
    return PcaArtifact(mean=mean, components=components, descriptor_version=version)


def test_fingerprint_matches_for_identical_arrays_and_version() -> None:
    """Two bases with the same arrays and version share one fingerprint."""
    assert _basis(seed=5).fingerprint == _basis(seed=5).fingerprint


def test_fingerprint_differs_when_a_component_element_changes() -> None:
    """A one-element change to ``components`` changes the fingerprint."""
    base = _basis(seed=5)
    perturbed = np.array(base.components)
    perturbed[0, 0] += 1.0
    other = PcaArtifact(mean=np.array(base.mean), components=perturbed, descriptor_version=base.descriptor_version)
    assert base.fingerprint != other.fingerprint


def test_fingerprint_differs_on_descriptor_version_change() -> None:
    """Same arrays but a different ``descriptor_version`` produce different fingerprints."""
    assert _basis(seed=5, version="dual-wrist-v1").fingerprint != _basis(seed=5, version="dual-wrist-v2").fingerprint


def test_fingerprint_survives_save_then_load(tmp_path: pathlib.Path) -> None:
    """The fingerprint is stable across a save/load round-trip (recomputed from persisted bytes)."""
    pca = fit_action_pca(_fittable(200))
    store = PcaArtifactStore(str(tmp_path))
    store.save_if_absent(pca)
    assert store.load(pca.fingerprint).fingerprint == pca.fingerprint


def test_fingerprint_is_never_persisted_in_the_npz(tmp_path: pathlib.Path) -> None:
    """The archive carries no ``fingerprint`` key: it is a derived value, never stored."""
    pca = fit_action_pca(_fittable(200))
    uri = PcaArtifactStore(str(tmp_path)).save_if_absent(pca)
    with np.load(uri) as archive:
        assert "fingerprint" not in archive.files
