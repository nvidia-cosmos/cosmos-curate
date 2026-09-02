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

"""Unit tests for the session-grain half of the engine.

The per-stream half needs an open sensor and is exercised through the recipe's
fixtures; ``run_session_metrics`` needs nothing but numbers, so it is tested here.
"""

import pytest

from cosmos_curator.core.sensors.data_integrity.engine import run_session_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import (
    NAME_SENSOR_OVERLAP,
    NAME_SENSOR_SPREAD,
    SESSION_INSTRUMENTS,
    Thresholds,
)
from cosmos_curator.core.sensors.data_integrity.results import CheckStatus


def _by_name(bounds: list[tuple[int, int]], **kwargs: object) -> dict[str, CheckStatus]:
    thresholds = Thresholds(**kwargs)  # type: ignore[arg-type]
    return {result.name: result.status for result in run_session_metrics(bounds, thresholds=thresholds)}


def test_every_session_metric_is_judged_in_registry_order() -> None:
    """The report's order is the registry's, so a reader sees the same metrics in the same places."""
    results = run_session_metrics([(0, 1_000), (100, 1_100)])
    assert [result.name for result in results] == [spec.name for spec in SESSION_INSTRUMENTS]


def test_an_aligned_session_passes_both_metrics() -> None:
    """The rig every check is measured against: same window, same start, same stop."""
    assert _by_name([(0, 1_000_000_000), (0, 1_000_000_000)]) == {
        NAME_SENSOR_SPREAD: CheckStatus.PASS,
        NAME_SENSOR_OVERLAP: CheckStatus.PASS,
    }


def test_a_sensor_that_started_late_fails_the_spread_it_exceeds() -> None:
    """One camera two seconds behind the rest is what the spread threshold is for."""
    late = [(0, 10_000_000_000), (2_000_000_000, 10_000_000_000)]
    assert _by_name(late, max_sensor_spread_ns=1_000_000_000)[NAME_SENSOR_SPREAD] is CheckStatus.FAIL
    assert _by_name(late, max_sensor_spread_ns=3_000_000_000)[NAME_SENSOR_SPREAD] is CheckStatus.PASS


def test_overlap_is_judged_as_the_share_of_the_session_that_is_not_shared() -> None:
    """Stated as the complement because a spec can only ask for a value to stay below a limit."""
    # 10% of the span has one sensor missing: 1s of a 10s session.
    ragged = [(0, 10_000_000_000), (1_000_000_000, 10_000_000_000)]
    assert _by_name(ragged, max_non_overlap_percent=5.0)[NAME_SENSOR_OVERLAP] is CheckStatus.FAIL
    assert _by_name(ragged, max_non_overlap_percent=20.0)[NAME_SENSOR_OVERLAP] is CheckStatus.PASS


@pytest.mark.parametrize("bounds", [[], [(0, 1_000)]])
def test_a_session_with_nothing_to_compare_is_skipped_not_passed(bounds: list[tuple[int, int]]) -> None:
    """A single-sensor session must not read as a session whose sensors agree."""
    statuses = _by_name(bounds)
    assert set(statuses.values()) == {CheckStatus.SKIPPED}


def test_a_skipped_session_metric_says_how_many_sensors_it_had() -> None:
    """A bare "insufficient data" leaves an operator guessing; the count is the whole explanation."""
    (spread, _) = run_session_metrics([(0, 1_000)])
    assert "num_sensors=1" in spread.reason


def test_bounds_are_taken_in_any_order() -> None:
    """A session runner reports streams as they finish, so the engine cannot require sorting them."""
    forward = run_session_metrics([(0, 1_000), (300, 1_400), (150, 1_250)])
    backward = run_session_metrics([(150, 1_250), (300, 1_400), (0, 1_000)])
    assert [(r.name, r.status, r.measurement) for r in forward] == [(r.name, r.status, r.measurement) for r in backward]
