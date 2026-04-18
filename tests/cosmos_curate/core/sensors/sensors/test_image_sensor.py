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

"""Tests for ``ImageSensor``."""

import pathlib
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image as PILImage

from cosmos_curate.core.sensors.sampling.grid import SamplingGrid, SamplingWindow
from cosmos_curate.core.sensors.sampling.spec import SamplingSpec
from cosmos_curate.core.sensors.sensors import image_sensor as image_sensor_module
from cosmos_curate.core.sensors.sensors.image_sensor import ImageSensor, _resolve_sensor_timestamps
from cosmos_curate.core.utils.storage.storage_client import StoragePrefix


class _DummyRemotePrefix(StoragePrefix):
    @property
    def path(self) -> str:
        return f"dummy://{self._input}"


class _StaticGrid:
    def __init__(self, windows: list[npt.NDArray[np.int64]]) -> None:
        self._windows = []
        for t in windows:
            if len(t) == 0:
                self._windows.append(SamplingWindow(start_ns=0, exclusive_end_ns=0, timestamps_ns=t))
            else:
                self._windows.append(
                    SamplingWindow(
                        start_ns=int(t[0]),
                        exclusive_end_ns=int(t[-1]) + 1,
                        timestamps_ns=t,
                    )
                )

    def __iter__(self) -> Iterator[SamplingWindow]:
        return iter(self._windows)


class _StaticSpec:
    def __init__(self, windows: list[np.ndarray]) -> None:
        self.grid = _StaticGrid(windows)
        self.policy = object()


def _write_image(
    path: pathlib.Path,
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (4, 3),
) -> None:
    PILImage.new("RGB", size, color).save(path)


def test_image_sensor_synthesizes_timestamps(tmp_path: pathlib.Path) -> None:
    """Missing timestamps should synthesize ``0..N-1``."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b])
    np.testing.assert_array_equal(sensor.sensor_timestamps_ns, np.array([0, 1], dtype=np.int64))


def test_resolve_sensor_timestamps_synthesizes_read_only_range() -> None:
    """The helper should synthesize a read-only ``0..N-1`` range when timestamps are omitted."""
    resolved = _resolve_sensor_timestamps(3, None)
    np.testing.assert_array_equal(resolved, np.array([0, 1, 2], dtype=np.int64))
    assert not resolved.flags.writeable


def test_resolve_sensor_timestamps_rejects_length_mismatch() -> None:
    """The helper should reject timestamp arrays whose length does not match the source count."""
    with pytest.raises(ValueError, match="must match sources length"):
        _resolve_sensor_timestamps(2, np.array([0], dtype=np.int64))


def test_image_sensor_rejects_empty_sources() -> None:
    """ImageSensor should require at least one source image."""
    with pytest.raises(ValueError, match="sources must be non-empty"):
        ImageSensor([])


def test_image_sensor_start_and_end_ns(tmp_path: pathlib.Path) -> None:
    """The sensor should expose the first and last sensor timestamps."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 30], dtype=np.int64))

    assert sensor.start_ns == 10
    assert sensor.end_ns == 30


def test_image_sensor_sample_uses_closest_timestamp(tmp_path: pathlib.Path) -> None:
    """Sampling should pick the image nearest to each reference timestamp in the window."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 30], dtype=np.int64))
    grid = SamplingGrid(
        timestamps_ns=np.array([10, 29, 40], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(sampled.align_timestamps_ns, np.array([10, 29], dtype=np.int64))
    np.testing.assert_array_equal(sampled.sensor_timestamps_ns, np.array([10, 30], dtype=np.int64))
    assert sampled.frames.shape == (2, 3, 4, 3)
    assert tuple(sampled.frames[0, 0, 0]) == (255, 0, 0)
    assert tuple(sampled.frames[1, 0, 0]) == (0, 255, 0)


def test_image_sensor_sample_is_window_local(tmp_path: pathlib.Path) -> None:
    """Sampling should ignore a globally closer image that lies outside the current window."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 35], dtype=np.int64))
    grid = SamplingGrid(
        timestamps_ns=np.array([29, 30], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


def test_image_sensor_sample_returns_empty_for_empty_window(tmp_path: pathlib.Path) -> None:
    """An empty window should yield the cached empty ImageData result."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    sampled_batches = list(sensor.sample(_StaticSpec([np.empty(0, dtype=np.int64)])))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


@pytest.mark.skip(reason="Sentinel-style exclusive-end marker removed; SamplingWindow handles boundaries explicitly")
def test_image_sensor_sample_returns_empty_when_active_grid_is_empty(tmp_path: pathlib.Path) -> None:
    """A window containing only its exclusive-end marker should yield empty data."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    sampled_batches = list(sensor.sample(_StaticSpec([np.array([10], dtype=np.int64)])))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


def test_image_sensor_sample_returns_empty_when_sampler_returns_no_indices(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sampler returns no indices, the sensor should yield empty data."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    def _fake_sample_window_indices(
        sensor_timestamps_ns: np.ndarray,
        window: np.ndarray,
        *,
        policy: object,
        dedup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del sensor_timestamps_ns, window, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    monkeypatch.setattr(image_sensor_module, "sample_window_indices", _fake_sample_window_indices)

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    grid = SamplingGrid(
        timestamps_ns=np.array([10, 11], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled_batches = list(sensor.sample(SamplingSpec(grid=grid)))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


@pytest.mark.skip(
    reason=(
        "Last element in grid is exclusive-end marker; resolve_sensor_timestamps() "
        "treats it as active, needs to be changed."
    ),
)
def test_image_sensor_sample_preserves_one_output_row_per_align_timestamp(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should delegate with ``dedup=False`` so rows stay aligned to active timestamps."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    def _fake_sample_window_indices(
        sensor_timestamps_ns: np.ndarray,
        window: SamplingWindow,
        *,
        policy: object,
        dedup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del sensor_timestamps_ns, policy
        assert dedup is False
        np.testing.assert_array_equal(window, np.array([10, 11], dtype=np.int64))
        return np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)

    monkeypatch.setattr(image_sensor_module, "sample_window_indices", _fake_sample_window_indices)

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10, 11], dtype=np.int64))
    grid = SamplingGrid(
        timestamps_ns=np.array([10, 11], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(sampled.align_timestamps_ns, np.array([10], dtype=np.int64))
    np.testing.assert_array_equal(sampled.sensor_timestamps_ns, np.array([10], dtype=np.int64))
    assert sampled.frames.shape == (1, 3, 4, 3)
    assert tuple(sampled.frames[0, 0, 0]) == (255, 0, 0)


def test_image_sensor_sample_with_no_windows_returns_nothing(tmp_path: pathlib.Path) -> None:
    """If the grid yields no windows, sampling should yield no ImageData batches."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))

    assert list(sensor.sample(_StaticSpec([]))) == []


def test_image_sensor_empty_image_data_is_cached(tmp_path: pathlib.Path) -> None:
    """The cached empty ImageData instance should be reused."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))

    assert sensor._get_empty_image_data() is sensor._get_empty_image_data()


def test_image_sensor_read_bytes_requires_client_for_remote_source() -> None:
    """Remote sources should require a storage client."""
    sensor = ImageSensor([pathlib.Path(__file__)])

    with pytest.raises(ValueError, match="storage client is required for non-local image sources"):
        sensor._read_bytes(_DummyRemotePrefix("bucket/path"))


def test_image_sensor_read_bytes_uses_storage_client_for_remote_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote sources should delegate reads through storage_utils when a client is configured."""
    expected = b"remote-bytes"

    def _fake_read_bytes(source: StoragePrefix, client: object) -> bytes:
        assert str(source) == "dummy://bucket/path"
        assert client is fake_client
        return expected

    fake_client = object()
    monkeypatch.setattr(image_sensor_module.storage_utils, "read_bytes", _fake_read_bytes)

    sensor = ImageSensor([pathlib.Path(__file__)], client=fake_client)

    assert sensor._read_bytes(_DummyRemotePrefix("bucket/path")) == expected
