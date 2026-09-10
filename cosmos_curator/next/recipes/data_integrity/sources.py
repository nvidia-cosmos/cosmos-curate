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

"""Turn a URI into an open sensor and run the metrics over it.

The IO wrapper around the per-stream engine
(:mod:`cosmos_curator.core.sensors.data_integrity.engine`): the engine takes an
already-open sensor and performs no IO, so resolving a local path or cloud URI into
a stream, and constructing the sensor on it, happens here. That split is what keeps
the sensor library backend-agnostic -- it never accepts a URI and imports no cloud
client (see :mod:`cosmos_curator.core.sensors.utils.io`) -- and it is why these two
functions sit in the recipe rather than beside the metrics they call.

Both entry points build on :func:`run_checks`: the single-video ``di-check`` CLI
(:mod:`.cli`) and the concurrent session runner (:mod:`.session_runner`).
"""

import pathlib
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import BinaryIO, cast

from cosmos_curator.core.sensors.data_integrity.engine import run_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, ResolvedConfig, VideoInfo
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import open_storage_source


def _as_data_source(stream: BinaryIO) -> DataSource:
    """Cast a ``BinaryIO`` from :func:`open_source` to a ``DataSource``.

    ``smart_open``'s S3 / Azure readers and a plain ``Path.open("rb")`` handle are all
    seekable ``io.BufferedIOBase`` subclasses, so they satisfy the ``DataSource`` union
    at runtime even though static typing only sees ``BinaryIO`` (mirrors
    ``check_video_index``).
    """
    return cast("DataSource", stream)


@contextmanager
def open_source(
    source: str,
    *,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> Generator[BinaryIO]:
    """Yield a fresh readable stream for ``source``, cloud URI or local path alike.

    ``stream_wrapper``, when given, wraps that stream before it is yielded (a
    byte-counting reader for download progress, a cancellable one for Ctrl-C).

    Local paths are opened here rather than handed to the sensor as a :class:`Path` so
    that they get the same wrappers: the sensor library opens a ``Path`` into a Python
    handle anyway (``open_file``), so libav sees the same callbacks either way and this
    costs nothing measurable, while a path on a shared filesystem can be every bit as
    slow to read as a cloud object. The trade is that the sensor now borrows one
    stateful handle instead of being able to re-open the file, which matches what every
    cloud source has always given it.
    """
    if is_remote_path(source):
        with open_storage_source(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        ) as remote_stream:
            yield stream_wrapper(remote_stream) if stream_wrapper is not None else remote_stream
    else:
        with pathlib.Path(source).open("rb") as local_stream:
            yield stream_wrapper(local_stream) if stream_wrapper is not None else local_stream


def run_checks(  # noqa: PLR0913
    source: str,
    *,
    expected_hz: float | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    stream_idx: int = 0,
    batch_size: int = 0,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    stats: dict[str, float] | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Open ``source``, build a :class:`CameraSensor`, and run every metric on it.

    The I/O wrapper around :func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics`:
    it resolves the local path or cloud stream, constructs the sensor, and delegates
    the compute. Any open / decode failure propagates as an exception for the caller
    to classify (the single-video CLI maps it to exit code 2; the session tool records
    a per-stream ERROR).

    Args:
        source: local path, ``s3://`` URI, or ``az://`` URI.
        expected_hz: expected sample rate for the rate-dependent metrics.
        thresholds: pass/fail policy (see :class:`Thresholds`).
        stream_idx: which video stream to open (default 0).
        batch_size: timestamps per metric update; 0 feeds the whole array at once.
        s3_profile_name: optional AWS profile forwarded to ``open_storage_source``.
        azure_profile_name: Azure profile forwarded to ``open_storage_source``.
        endpoint_url: optional S3 endpoint override for S3-compatible stores.
        stats: optional out-parameter; ``sensor_init_ms`` is recorded here in
            addition to the ``stream_ms`` / ``evaluate_ms`` from ``run_metrics``.
        stream_wrapper: optional wrapper applied to the stream before the sensor
            reads it (e.g. a byte-counting reader for progress), for local paths
            as well as cloud URIs.

    Returns:
        The tuple returned by ``run_metrics``.

    """
    with open_source(
        source,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stream_wrapper=stream_wrapper,
    ) as src:
        t0 = time.perf_counter()
        sensor = CameraSensor(_as_data_source(src), stream_idx=stream_idx)
        if stats is not None:
            stats["sensor_init_ms"] = (time.perf_counter() - t0) * 1000
        return run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size, stats=stats)
