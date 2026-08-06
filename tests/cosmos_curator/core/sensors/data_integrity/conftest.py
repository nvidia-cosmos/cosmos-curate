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

"""Shared fixtures for the data-integrity store tests.

These build results by running the *real* engine over a synthetic timeline rather
than hand-assembling measurements. Half of what the store tests assert is that a
value survives the trip from the kernel through Arrow and back, so a faked
measurement would fake the thing under test.

Exposed as fixtures because the test tree is not a package (pytest runs with
``--import-mode=importlib``), so one test module cannot import another's helpers.
"""

import pathlib
from collections.abc import Callable, Iterator
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.cli_common import DEFAULT_THRESHOLDS, run_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    ResolvedConfig,
    StreamResult,
    VideoInfo,
    stream_result,
)

HZ_100_PERIOD_NS = 10_000_000  # one sample every 10 ms at 100 Hz

#: A header rate of zero is the sentinel for "the container declares nothing", which
#: is what makes rate / gap / jitter never run at all.
NO_HEADER_RATE = Fraction(0, 1)


class FakeSensor:
    """Minimal stand-in satisfying the shared engine's ``IntegritySensor`` surface."""

    def __init__(
        self,
        timestamps: list[int],
        *,
        has_bframes: bool = False,
        codec_name: str = "h264",
        avg_frame_rate: Fraction = Fraction(100, 1),
    ) -> None:
        """Hold a synthetic timeline and the container facts the engine reads off it."""
        self._ts = np.array(timestamps, dtype=np.int64)
        self.has_bframes = has_bframes
        self.codec_name = codec_name
        self.video_metadata = SimpleNamespace(avg_frame_rate=avg_frame_rate)

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        """The full synthetic timeline."""
        return self._ts

    @property
    def start_ns(self) -> int:
        """First timestamp, or zero for an empty timeline."""
        return int(self._ts[0]) if len(self._ts) else 0

    @property
    def end_ns(self) -> int:
        """Last timestamp, or zero for an empty timeline."""
        return int(self._ts[-1]) if len(self._ts) else 0

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        """Yield the timeline in ``batch_size`` windows (``0`` = one window)."""
        step = batch_size or len(self._ts) or 1
        for start in range(0, len(self._ts), step):
            yield self._ts[start : start + step]


@pytest.fixture
def perfect() -> Callable[..., list[int]]:
    """Build ``n`` timestamps exactly one 100 Hz period apart."""

    def _make(n: int = 10) -> list[int]:
        return [i * HZ_100_PERIOD_NS for i in range(n)]

    return _make


@pytest.fixture
def drifting() -> Callable[..., list[int]]:
    """Build ``n`` timestamps at a uniformly wrong rate, ``percent`` off 100 Hz."""

    def _make(n: int = 10, *, percent: float = 1.0) -> list[int]:
        period = int(HZ_100_PERIOD_NS * (1.0 + percent / 100.0))
        return [i * period for i in range(n)]

    return _make


@pytest.fixture
def make_stream() -> Callable[..., StreamResult]:
    """Run the real engine over a synthetic timeline and package the result."""

    def _make(  # noqa: PLR0913 -- every knob shapes a different synthetic stream
        source: str,
        timestamps: list[int],
        *,
        expected_hz: float | None = 100.0,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        avg_frame_rate: Fraction = Fraction(100, 1),
        has_bframes: bool = False,
        selector_value: str = identity.DEFAULT_SELECTOR_VALUE,
    ) -> StreamResult:
        sensor = FakeSensor(timestamps, avg_frame_rate=avg_frame_rate, has_bframes=has_bframes)
        metrics, info, resolved = run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds)
        return stream_result(source, metrics, info, resolved, selector_value=selector_value)

    return _make


@pytest.fixture
def run_engine() -> Callable[..., tuple[list[CheckResult], VideoInfo, ResolvedConfig]]:
    """Run the engine and return its output unpackaged, the shape the single-video CLI works in.

    :func:`make_stream` packages the same run; this returns it unpackaged, for tests
    that stand in for ``run_checks`` itself.
    """

    def _make(
        timestamps: list[int],
        *,
        expected_hz: float | None = 100.0,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        avg_frame_rate: Fraction = Fraction(100, 1),
    ) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
        sensor = FakeSensor(timestamps, avg_frame_rate=avg_frame_rate)
        return run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds)

    return _make


@pytest.fixture
def make_errored_stream() -> Callable[..., StreamResult]:
    """Build a stream that could not be opened at all -- no metrics, no measurements."""

    def _make(source: str, message: str = "moov atom not found") -> StreamResult:
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error=message,
        )

    return _make


@pytest.fixture
def store_root(tmp_path: pathlib.Path) -> str:
    """Point at a local store root that does not exist yet, so the first write has to create it."""
    return str(tmp_path / "di-store")
