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
"""Parity: the vectorized clip_measurements_columnar vs the row-at-a-time oracle.

Production computes clip measurements with the columnar
:func:`clip_measurements_columnar` (a flat-scalar full-outer join + pyarrow.compute
diffs). This test pins it to an independent, obviously-correct reference
implementation (``_oracle_clip_measurement`` below, the original per-row code) so a
future refactor of the columnar path can't silently drift from the intended
semantics -- especially the subtle ones: Python ``==``-with-None
(``None == None -> True``), non-finite scores collapsing to null, and one-sided
clips leaving the absent side / its equality flags null.

The oracle lives here, not in production, precisely because it exists to check the
columnar code rather than to run. It reuses ``_video_uuid`` / ``_windows`` from the
module under test (shared with the still-live window path).
"""

import math
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.load import index_by_clip
from cosmos_curator.pipelines.video.split_comparison.measure.core import (
    _video_uuid,
    _window_measurements,
    _windows,
    clip_measurements_columnar,
    measure_stats,
    merge_caption_rollups,
)
from cosmos_curator.pipelines.video.split_comparison.measure.schema import (
    CLIP_MEASUREMENT_SCHEMA,
    WINDOW_MEASUREMENT_SCHEMA,
)

# --------------------------------------------------------------------------- #
# Oracle: the reference per-row clip measurement the columnar path must match.
# --------------------------------------------------------------------------- #


def _oracle_clip_measurement(
    clip_uuid: str,
    row_a: Mapping[str, Any] | None,
    row_b: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one clip measurement row: raw per-side values plus diffs / equality flags."""
    present_a = row_a is not None
    present_b = row_b is not None
    a: Mapping[str, Any] = row_a or {}
    b: Mapping[str, Any] = row_b or {}

    aesthetic_a = _finite(a.get("aesthetic_score"))
    aesthetic_b = _finite(b.get("aesthetic_score"))
    motion_a = _as_mapping(a.get("motion_score"))
    motion_b = _as_mapping(b.get("motion_score"))
    gmean_a = _finite(motion_a.get("global_mean"))
    gmean_b = _finite(motion_b.get("global_mean"))
    ppm_a = _finite(motion_a.get("per_patch_min_256"))
    ppm_b = _finite(motion_b.get("per_patch_min_256"))

    return {
        "clip_uuid": clip_uuid,
        "video_uuid": _video_uuid(row_a, row_b),
        "present_a": present_a,
        "present_b": present_b,
        **_score_triple("aesthetic", aesthetic_a, aesthetic_b),
        **_score_triple("motion_global_mean", gmean_a, gmean_b),
        **_score_triple("motion_per_patch_min_256", ppm_a, ppm_b),
        "valid_a": _bool_or_none(a.get("valid")) if present_a else None,
        "valid_b": _bool_or_none(b.get("valid")) if present_b else None,
        "valid_equal": _equal_or_none(a.get("valid"), b.get("valid"), present_a=present_a, present_b=present_b),
        "has_caption_a": _bool_or_none(a.get("has_caption")) if present_a else None,
        "has_caption_b": _bool_or_none(b.get("has_caption")) if present_b else None,
        "has_caption_equal": _equal_or_none(
            a.get("has_caption"), b.get("has_caption"), present_a=present_a, present_b=present_b
        ),
        "rejection_stage_a": _str_or_none(a.get("rejection_stage")) if present_a else None,
        "rejection_stage_b": _str_or_none(b.get("rejection_stage")) if present_b else None,
        "rejection_stage_equal": _equal_or_none(
            a.get("rejection_stage"), b.get("rejection_stage"), present_a=present_a, present_b=present_b
        ),
        "num_windows_a": len(_windows(a)) if present_a else None,
        "num_windows_b": len(_windows(b)) if present_b else None,
        "min_caption_similarity": None,
        "mean_caption_similarity": None,
    }


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    return None


def _abs_diff(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return abs(value_a - value_b)


def _score_triple(prefix: str, value_a: float | None, value_b: float | None) -> dict[str, float | None]:
    return {f"{prefix}_a": value_a, f"{prefix}_b": value_b, f"{prefix}_abs_diff": _abs_diff(value_a, value_b)}


def _bool_or_none(value: object) -> bool | None:
    return bool(value) if value is not None else None


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _equal_or_none(value_a: object, value_b: object, *, present_a: bool, present_b: bool) -> bool | None:
    if not (present_a and present_b):
        return None
    return value_a == value_b


# --------------------------------------------------------------------------- #
# Fixtures: a source-table pair exercising the clip measurement edge cases.
# --------------------------------------------------------------------------- #

_MOTION = pa.struct([("global_mean", pa.float64()), ("per_patch_min_256", pa.float64())])
_WINDOW = pa.struct(
    [
        ("start_ns", pa.int64()),
        ("end_ns", pa.int64()),
        ("captions", pa.map_(pa.string(), pa.large_string())),
        ("enhanced_captions", pa.map_(pa.string(), pa.large_string())),
    ]
)
_SOURCE_SCHEMA = pa.schema(
    [
        ("clip_uuid", pa.string()),
        ("video_uuid", pa.string()),
        ("aesthetic_score", pa.float64()),
        ("motion_score", _MOTION),
        ("valid", pa.bool_()),
        ("has_caption", pa.bool_()),
        ("rejection_stage", pa.string()),
        ("windows", pa.list_(_WINDOW)),
    ]
)

_UNSET = object()


def _windows_list(count: int) -> list[dict[str, Any]]:
    return [
        {"start_ns": i, "end_ns": i + 1, "captions": {"qwen": f"caption-{i}"}, "enhanced_captions": {}}
        for i in range(count)
    ]


def _row(
    clip_uuid: str,
    *,
    video_uuid: str = "video-1",
    aesthetic: float | None = 4.0,
    motion: object = _UNSET,
    valid: bool | None = True,
    has_caption: bool | None = True,
    rejection_stage: str | None = None,
    windows: object = 1,
) -> dict[str, Any]:
    """Build one source clip row; ``windows`` may be a count, an explicit list, or None."""
    motion_value = {"global_mean": 0.1, "per_patch_min_256": 0.2} if motion is _UNSET else motion
    windows_value = _windows_list(windows) if isinstance(windows, int) else windows
    return {
        "clip_uuid": clip_uuid,
        "video_uuid": video_uuid,
        "aesthetic_score": aesthetic,
        "motion_score": motion_value,
        "valid": valid,
        "has_caption": has_caption,
        "rejection_stage": rejection_stage,
        "windows": windows_value,
    }


def _edge_case_tables() -> tuple[pa.Table, pa.Table]:
    """Two source tables aligned by clip_uuid, each clip probing one measurement edge."""
    a_rows = [
        _row("identical"),
        _row("aesthetic_diff", aesthetic=6.0),
        _row("motion_diff", motion={"global_mean": 0.9, "per_patch_min_256": 0.5}),
        _row("valid_diff", valid=False),
        _row("caption_flag_diff", has_caption=False),
        _row("rejection_set_one_side", rejection_stage="qwen_semantic"),
        _row("rejection_both_none"),  # None == None -> equal True
        _row("valid_null_one_side", valid=None),  # present clip with a null scalar
        _row("nan_aesthetic", aesthetic=float("nan")),  # non-finite -> null
        _row("inf_motion", motion={"global_mean": float("inf"), "per_patch_min_256": 0.2}),
        _row("null_motion", motion=None),  # null struct -> null sub-scores
        _row("zero_windows", windows=0),
        _row("null_windows", windows=None),  # null list -> num_windows 0
        _row("window_count_diff", windows=3),
        _row("video_uuid_diff", video_uuid="video-a"),
        _row("a_only"),  # one-sided: present in A only
    ]
    b_rows = [
        _row("identical"),
        _row("aesthetic_diff", aesthetic=4.0),
        _row("motion_diff"),
        _row("valid_diff", valid=True),
        _row("caption_flag_diff", has_caption=True),
        _row("rejection_set_one_side", rejection_stage=None),
        _row("rejection_both_none"),
        _row("valid_null_one_side", valid=True),
        _row("nan_aesthetic", aesthetic=4.0),
        _row("inf_motion"),
        _row("null_motion"),
        _row("zero_windows", windows=0),
        _row("null_windows", windows=None),
        _row("window_count_diff", windows=1),
        _row("video_uuid_diff", video_uuid="video-b"),
        _row("b_only"),  # one-sided: present in B only
    ]
    return (
        pa.Table.from_pylist(a_rows, schema=_SOURCE_SCHEMA),
        pa.Table.from_pylist(b_rows, schema=_SOURCE_SCHEMA),
    )


def _oracle_table(table_a: pa.Table, table_b: pa.Table) -> pa.Table:
    rows_a = index_by_clip(table_a)
    rows_b = index_by_clip(table_b)
    union = sorted(set(rows_a) | set(rows_b))
    rows = [_oracle_clip_measurement(uuid, rows_a.get(uuid), rows_b.get(uuid)) for uuid in union]
    return pa.Table.from_pylist(rows, schema=CLIP_MEASUREMENT_SCHEMA)


def _norm(table: pa.Table) -> list[dict[str, Any]]:
    return sorted(table.to_pylist(), key=lambda row: row["clip_uuid"])


def _by_uuid(table: pa.Table, clip_uuid: str) -> dict[str, Any]:
    return next(row for row in table.to_pylist() if row["clip_uuid"] == clip_uuid)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_columnar_clip_measurement_matches_oracle() -> None:
    """The vectorized path is byte-identical to the per-row oracle across all edge cases."""
    table_a, table_b = _edge_case_tables()
    columnar = clip_measurements_columnar(table_a, table_b)
    assert columnar.schema.equals(CLIP_MEASUREMENT_SCHEMA)
    assert _norm(columnar) == _norm(_oracle_table(table_a, table_b))


def test_rejection_stage_none_on_both_sides_is_equal() -> None:
    """None == None must resolve to True (Python equality), not Arrow null."""
    row = _by_uuid(clip_measurements_columnar(*_edge_case_tables()), "rejection_both_none")
    assert row["rejection_stage_a"] is None
    assert row["rejection_stage_b"] is None
    assert row["rejection_stage_equal"] is True


def test_one_sided_clip_leaves_absent_side_and_equality_null() -> None:
    """A clip present on only one side records presence but nothing to compare."""
    row = _by_uuid(clip_measurements_columnar(*_edge_case_tables()), "a_only")
    assert row["present_a"] is True
    assert row["present_b"] is False
    assert row["aesthetic_b"] is None
    assert row["aesthetic_abs_diff"] is None
    assert row["valid_b"] is None
    assert row["valid_equal"] is None


def test_non_finite_scores_collapse_to_null() -> None:
    """NaN / inf scores become null, so their abs diff is null too."""
    columnar = clip_measurements_columnar(*_edge_case_tables())
    nan_row = _by_uuid(columnar, "nan_aesthetic")
    assert nan_row["aesthetic_a"] is None
    assert nan_row["aesthetic_abs_diff"] is None
    inf_row = _by_uuid(columnar, "inf_motion")
    assert inf_row["motion_global_mean_a"] is None
    assert inf_row["motion_global_mean_abs_diff"] is None


def test_null_or_empty_windows_count_as_zero() -> None:
    """A null or empty windows list yields num_windows 0, not null, for a present clip."""
    columnar = clip_measurements_columnar(*_edge_case_tables())
    assert _by_uuid(columnar, "zero_windows")["num_windows_a"] == 0
    assert _by_uuid(columnar, "null_windows")["num_windows_a"] == 0


# --------------------------------------------------------------------------- #
# Window-table consumers: merge_caption_rollups + measure_stats read columns
# off the columnar window table (no full pylist round-trip).
# --------------------------------------------------------------------------- #


def _window_row(clip_uuid: str, similarity: float | None, *, identical: bool | None) -> dict[str, Any]:
    return {
        "clip_uuid": clip_uuid,
        "video_uuid": "v",
        "start_ns": 0,
        "end_ns": 1,
        "model": "qwen",
        "kind": "caption",
        "present_a": True,
        "present_b": True,
        "identical": identical,
        "similarity": similarity,
        "len_a": 1,
        "len_b": 1,
    }


def _window_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=WINDOW_MEASUREMENT_SCHEMA)


def test_merge_caption_rollups_fills_min_mean_per_clip() -> None:
    """min/mean caption similarity are rolled up per clip from the window table; nulls are skipped."""
    src = pa.Table.from_pylist([_row("c1"), _row("c2")], schema=_SOURCE_SCHEMA)
    clip_table = clip_measurements_columnar(src, src)  # rollups start null
    windows = _window_table(
        [
            _window_row("c1", 0.8, identical=False),
            _window_row("c1", 0.6, identical=False),
            _window_row("c2", 1.0, identical=True),
            _window_row("c2", None, identical=None),  # one-sided window: no similarity, skipped
        ]
    )
    rows = {r["clip_uuid"]: r for r in merge_caption_rollups(clip_table, windows).to_pylist()}
    assert rows["c1"]["min_caption_similarity"] == pytest.approx(0.6)
    assert rows["c1"]["mean_caption_similarity"] == pytest.approx(0.7)
    assert rows["c2"]["min_caption_similarity"] == pytest.approx(1.0)
    assert rows["c2"]["mean_caption_similarity"] == pytest.approx(1.0)


def test_merge_caption_rollups_no_similarities_leaves_nulls() -> None:
    """With no computed similarities the clip rollups stay null (table returned unchanged)."""
    src = pa.Table.from_pylist([_row("c1")], schema=_SOURCE_SCHEMA)
    clip_table = clip_measurements_columnar(src, src)
    merged = merge_caption_rollups(clip_table, _window_table([_window_row("c1", None, identical=None)]))
    assert merged.column("min_caption_similarity").to_pylist() == [None]


def test_measure_stats_counts_off_the_window_table() -> None:
    """Stats read clip union + per-side presence + the window table's identical column."""
    union = ["c1", "c2", "c3"]
    index_a = {"c1": 0, "c2": 1, "c3": 2}
    index_b = {"c1": 0, "c2": 1}  # c3 is one-sided (only in A)
    windows = _window_table(
        [
            _window_row("c1", 1.0, identical=True),
            _window_row("c1", 0.5, identical=False),
            _window_row("c2", None, identical=None),
        ]
    )
    stats = measure_stats(union, index_a, index_b, windows)
    assert stats["num_clips"] == 3
    assert stats["num_clips_one_sided"] == 1
    assert stats["num_windows"] == 3
    assert stats["num_windows_identical"] == 1
    assert stats["num_caption_pairs_embedded"] == 1


def test_window_measurements_sorts_null_bounds_without_typeerror() -> None:
    """A window with null start/end bounds alongside a bounded one must not crash the bounds sort.

    ``_windows_by_bounds`` keys on ``(start_ns, end_ns)`` which the type admits as ``int | None``;
    sorting that union with the default comparator raises ``TypeError`` (None vs int). The clip
    should measure cleanly, emitting one row per (window, model, kind) across both bounds.
    """
    row = {
        "windows": [
            {"start_ns": None, "end_ns": None, "captions": {"qwen": "a"}},
            {"start_ns": 0, "end_ns": 100, "captions": {"qwen": "b"}},
        ],
    }
    rows, _jobs = _window_measurements("c1", "v", row, None)
    assert {(r["start_ns"], r["end_ns"]) for r in rows} == {(None, None), (0, 100)}
