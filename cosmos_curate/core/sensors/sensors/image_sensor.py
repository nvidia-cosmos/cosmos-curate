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

"""Sensor wrapper for timestamped still images."""

import io
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

from cosmos_curate.core.sensors.data.image_data import ImageData, ImageMetadata
from cosmos_curate.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curate.core.sensors.sampling.spec import SamplingSpec
from cosmos_curate.core.sensors.utils.validation import require_strictly_increasing
from cosmos_curate.core.utils.storage import storage_client, storage_utils


def _resolve_sensor_timestamps(
    num_sources: int,
    sensor_timestamps_ns: npt.NDArray[np.int64] | None,
) -> npt.NDArray[np.int64]:
    """Resolve validated, read-only sensor timestamps for the provided sources."""
    resolved_sensor_timestamps: npt.NDArray[np.int64]
    if sensor_timestamps_ns is None or len(sensor_timestamps_ns) == 0:
        resolved_sensor_timestamps = np.arange(num_sources, dtype=np.int64)
    else:
        resolved_sensor_timestamps = np.array(sensor_timestamps_ns, dtype=np.int64, copy=True)
    if len(resolved_sensor_timestamps) != num_sources:
        msg = f"sensor_timestamps_ns length {len(resolved_sensor_timestamps)} must match sources length {num_sources}"
        raise ValueError(msg)
    require_strictly_increasing("sensor_timestamps_ns", resolved_sensor_timestamps)
    resolved_sensor_timestamps.flags.writeable = False
    return resolved_sensor_timestamps


class ImageSensor:
    """Image sensor with optional timestamps and nearest-neighbor sampling."""

    def __init__(
        self,
        sources: Sequence[storage_client.StoragePrefix | Path],
        sensor_timestamps_ns: npt.NDArray[np.int64] | None = None,
        *,
        client: storage_client.StorageClient | None = None,
    ) -> None:
        """Initialize with image sources, optional sensor timestamps, and optional storage client."""
        if len(sources) == 0:
            msg = "sources must be non-empty"
            raise ValueError(msg)

        self._sources = list(sources)
        self._client = client
        self._provided_sensor_timestamps_ns = sensor_timestamps_ns
        self._sensor_timestamps_ns: npt.NDArray[np.int64] | None = None
        self._empty_image_data: ImageData | None = None

    @property
    def sensor_timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Return native per-image sensor timestamps in nanoseconds."""
        if self._sensor_timestamps_ns is None:
            self._sensor_timestamps_ns = _resolve_sensor_timestamps(
                len(self._sources),
                self._provided_sensor_timestamps_ns,
            )
        return self._sensor_timestamps_ns

    @property
    def start_ns(self) -> int:
        """Return the earliest sensor timestamp."""
        return int(self.sensor_timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        """Return the latest sensor timestamp."""
        return int(self.sensor_timestamps_ns[-1])

    def sample(self, spec: SamplingSpec) -> Generator[ImageData, None, None]:
        """Yield sampled ``ImageData`` batches for each window in ``spec.grid``."""
        for window in spec.grid:
            if window.size == 0:
                yield self._get_empty_image_data()
                continue

            active_grid = window[window < window[-1]]
            if len(active_grid) == 0:
                yield self._get_empty_image_data()
                continue
            indices, counts = sample_window_indices(
                self.sensor_timestamps_ns,
                window,
                policy=spec.policy,
                dedup=False,
            )

            sampled_frames: list[npt.NDArray[np.uint8]] = []
            sampled_canonical_ts: list[int] = []
            metadata: ImageMetadata | None = None

            for idx, count in zip(indices, counts, strict=True):
                frame, metadata = self._load_frame(int(idx))
                sensor_ts = int(self.sensor_timestamps_ns[int(idx)])
                for _ in range(int(count)):
                    sampled_frames.append(frame)
                    sampled_canonical_ts.append(sensor_ts)

            if metadata is None:
                yield self._get_empty_image_data()
                continue

            frames = np.stack(sampled_frames, axis=0)
            yield ImageData(
                align_timestamps_ns=np.array(active_grid, dtype=np.int64),
                sensor_timestamps_ns=np.array(sampled_canonical_ts, dtype=np.int64),
                frames=frames,
                metadata=metadata,
            )

    def _get_empty_image_data(self) -> ImageData:
        """Return a cached empty ``ImageData`` preserving image geometry."""
        if self._empty_image_data is None:
            _, metadata = self._load_frame(0)
            empty_ts = np.empty(0, dtype=np.int64)
            empty_frames = np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8)
            self._empty_image_data = ImageData(
                align_timestamps_ns=empty_ts,
                sensor_timestamps_ns=empty_ts,
                frames=empty_frames,
                metadata=metadata,
            )
        return self._empty_image_data

    def _load_frame(self, idx: int) -> tuple[npt.NDArray[np.uint8], ImageMetadata]:
        """Read and decode one image frame."""
        raw = self._read_bytes(self._sources[idx])
        with PILImage.open(io.BytesIO(raw)) as image:
            rgb_image = image.convert("RGB")
            frame = np.array(rgb_image, dtype=np.uint8)
            fmt = image.format.lower() if image.format is not None else None
        metadata = ImageMetadata(height=int(frame.shape[0]), width=int(frame.shape[1]), image_format=fmt)
        return frame, metadata

    def _read_bytes(self, source: storage_client.StoragePrefix | Path) -> bytes:
        """Read image bytes from local or remote storage."""
        if isinstance(source, Path):
            return source.read_bytes()
        if self._client is None:
            msg = "storage client is required for non-local image sources"
            raise ValueError(msg)
        return storage_utils.read_bytes(source, self._client)
