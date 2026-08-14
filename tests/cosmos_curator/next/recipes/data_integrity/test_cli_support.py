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

"""Unit tests for the argparse surface and Ctrl-C plumbing shared by both data-integrity CLIs."""

import io
import os
import threading

import pytest

from cosmos_curator.core.sensors.utils.io import open_data_source
from cosmos_curator.next.recipes.data_integrity.cli_support import available_cpu_count, cancellable_reader


def test_prefers_the_affinity_mask_over_the_host_core_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a cpuset the mask is narrower than the machine, and the mask is what binds."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert available_cpu_count() == 3


def test_falls_back_to_the_host_core_count_without_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """sched_getaffinity is Linux-only, so macOS and Windows take the cpu_count path."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert available_cpu_count() == 8


@pytest.mark.parametrize("cpu_count", [None, 0])
def test_never_returns_less_than_one(monkeypatch: pytest.MonkeyPatch, cpu_count: int | None) -> None:
    """An undetectable CPU count must still yield a usable worker count, not 0 or None."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    assert available_cpu_count() == 1


def test_reports_a_plausible_count_on_the_real_host() -> None:
    """Unmocked, the helper returns something a thread pool can actually be sized with."""
    assert available_cpu_count() >= 1


class _NotSeekable(io.BufferedIOBase):
    """A readable but unseekable stream, which the sensor library refuses to accept."""

    def read(self, size: int | None = -1) -> bytes:
        return b"x" * (16 if size in (None, -1) else int(size))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def test_the_wrapper_reports_its_streams_own_capabilities() -> None:
    """Capabilities are delegated, not asserted, so the sensor's guard still sees the truth.

    The sensor library rejects a stream that is not readable and seekable. A wrapper
    claiming both would smuggle an unusable stream past that check and turn a clear
    up-front error into an opaque failure inside libav.
    """
    assert not cancellable_reader(_NotSeekable(), threading.Event()).seekable()  # type: ignore[arg-type]

    seekable = cancellable_reader(io.BytesIO(b"x"), threading.Event())
    assert seekable.seekable()
    assert seekable.readable()


def test_an_unseekable_stream_is_still_rejected_up_front() -> None:
    """The wrapper must not let a stream the sensor cannot use reach libav."""
    wrapped = cancellable_reader(_NotSeekable(), threading.Event())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seekable"), open_data_source(wrapped):  # type: ignore[arg-type]
        pass
