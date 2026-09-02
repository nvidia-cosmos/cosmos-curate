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

"""What a data-integrity run *found*: the vocabulary every other module speaks.

The kernel (:mod:`.metrics` / :mod:`.evaluation`) only judges a well-defined
measurement as ``PASS`` / ``FAIL``. These types wrap that with everything a caller
also has to express: a metric that could not be judged at all (:class:`CheckStatus`
``SKIPPED``), a stream that could not be opened (:class:`OverallStatus` ``ERROR``),
where the expected rate came from (:class:`ExpectedHzSource`), and JSON-safe
renderings of the kernel's own objects.

Deliberately a leaf: it imports the kernel and nothing else in this package. The
engine, the reports and the store all produce or consume these types, so anything
they share belongs here rather than in whichever of them happened to define it
first -- otherwise the store ends up importing the CLI to learn what a verdict is.
"""

import enum
import math
from collections.abc import Iterator
from typing import Protocol

import attrs
import numpy as np
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.evaluation import EvaluationResult
from cosmos_curator.core.sensors.data_integrity.metrics import Measurement


class CheckStatus(enum.Enum):
    """Per-metric check status, deliberately distinct from :class:`EvaluationStatus`.

    Kernel evaluators only ever return PASS/FAIL over a well-defined measurement.
    ``SKIPPED`` lives here because it covers the cases the kernel cannot evaluate
    at all: an undefined measurement (a kernel invariant not to evaluate) or a
    missing prerequisite (no usable expected rate from either ``--expected-hz`` or
    the container header).
    """

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class OverallStatus(enum.Enum):
    """Rolled-up verdict for a stream or a session.

    ``ERROR`` is distinct from ``FAIL``: ``FAIL`` means the integrity checks ran
    and something was out of bounds, while ``ERROR`` means the stream could not
    be opened or decoded so no judgment was possible. Because "unmeasured" is the
    weaker guarantee, ``ERROR`` is the more severe of the two when several streams
    are rolled up; see :meth:`SessionReport.status`.
    """

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    ERROR = "ERROR"


class ExpectedHzSource(enum.Enum):
    """Origin of the effective expected sample rate used by rate-dependent metrics.

    Variants, in fallback priority:

    * ``USER`` -- user-supplied ``--expected-hz``; the authoritative baseline.
    * ``HEADER`` -- from :attr:`VideoMetadata.avg_frame_rate`; best-effort only,
      and not a sound basis for the rate check (see :func:`resolve_expected_hz`).
    * ``UNAVAILABLE`` -- neither is usable; rate-dependent metrics SKIP.

    Reported alongside the rate itself so a reader can tell which of those three
    situations produced a given verdict.
    """

    USER = "user"
    HEADER = "header"
    UNAVAILABLE = "unavailable"


@attrs.define(frozen=True)
class CheckResult:
    """Result of one metric on one stream.

    Attributes:
        name: metric identifier (one of the ``NAME_*`` constants).
        status: PASS / FAIL / SKIPPED for this metric on this stream.
        reason: human-readable one-line summary (value and threshold, or why it
            was skipped), shown verbatim in the human report.
        measurement: JSON-safe dict of the raw measurement, or ``None`` when the
            metric was skipped before a measurement existed.
        evaluation: JSON-safe dict of the kernel evaluation (status + margin), or
            ``None`` when the measurement was undefined / skipped.
        raw_measurement: the frozen measurement itself, or ``None`` when the metric
            never ran. Carried for the store, which persists typed columns and so
            needs the real values -- ``measurement`` above has already been through
            :func:`_json_safe`, which flattens ``NaN`` to ``None`` and would turn
            "measured but undefined" into "not measured". Renderers build their own
            dicts field by field, so this is never serialised by them.

    """

    name: str
    status: CheckStatus
    reason: str
    measurement: dict[str, object] | None
    evaluation: dict[str, object] | None
    raw_measurement: Measurement | None = None


@attrs.define(frozen=True)
class ResolvedConfig:
    """Effective expected rate resolved from user args + sensor metadata.

    ``expected_hz`` is ``None`` iff ``expected_hz_source`` is ``UNAVAILABLE``; the
    invariant is enforced by :func:`~.engine.resolve_expected_hz`.
    """

    expected_hz: float | None
    expected_hz_source: ExpectedHzSource


@attrs.define(frozen=True)
class VideoInfo:
    """Small snapshot of sensor-level facts shared by both reports."""

    codec_name: str
    has_bframes: bool
    num_samples: int
    start_ns: int | None
    end_ns: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a plain JSON-serialisable dict of the fields."""
        return {
            "codec_name": self.codec_name,
            "has_bframes": self.has_bframes,
            "num_samples": self.num_samples,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
        }


class IntegritySensor(Protocol):  # pragma: no cover
    """Structural sensor surface :func:`~.engine.run_metrics` needs (``CameraSensor`` satisfies it)."""

    @property
    def codec_name(self) -> str:
        """Video codec name (e.g. ``h264``)."""
        ...

    @property
    def has_bframes(self) -> bool:
        """Whether the stream signals frame reordering (B-frames)."""
        ...

    @property
    def start_ns(self) -> int:
        """First timestamp in nanoseconds."""
        ...

    @property
    def end_ns(self) -> int:
        """Last timestamp in nanoseconds."""
        ...

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        """The full decoded timeline in ``int64`` nanoseconds."""
        ...

    @property
    def video_metadata(self) -> VideoMetadata:
        """Scalar stream metadata (carries the nominal ``avg_frame_rate``)."""
        ...

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        """Yield the timeline in ``int64`` ns batches (``0`` = one batch)."""
        ...


def _json_safe(value: object) -> object:
    """Convert numpy scalars, NaN, and infinities into JSON-serialisable equivalents.

    NaN and infinity have no JSON representation; ``allow_nan=False`` in
    :func:`json.dumps` would otherwise raise, silently promoting a rare corruption
    into a hard crash on reporting. Both map to ``None`` so the report survives an
    undefined field while staying loudly wrong (rather than ``0.0``, which would
    look defined).
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def measurement_to_dict(measurement: Measurement) -> dict[str, object]:
    """Serialise a metric measurement into a JSON-safe dict."""
    raw = attrs.asdict(measurement)  # type: ignore[arg-type]  # protocol vs. attrs class
    return _json_safe(raw)  # type: ignore[return-value]


def evaluation_to_dict(result: EvaluationResult[int] | EvaluationResult[float]) -> dict[str, object]:
    """Serialise an :class:`EvaluationResult` into a JSON-safe dict."""
    return {"status": result.status.value, "margin": _json_safe(result.margin)}


def overall_status(results: list[CheckResult]) -> CheckStatus:
    """FAIL if any check failed; SKIPPED never fails the run, otherwise PASS."""
    return CheckStatus.FAIL if any(r.status is CheckStatus.FAIL for r in results) else CheckStatus.PASS


@attrs.define(frozen=True)
class StreamResult:
    """Integrity result for a single stream (one video / one camera).

    A stream that failed to open or decode carries ``error`` set and an empty
    ``metrics`` list; its :attr:`status` is then ``ERROR``. Otherwise the status
    is ``FAIL`` if any metric failed and ``PASS`` if none did (``SKIPPED``
    metrics do not fail the stream, but are reported).

    Attributes:
        source: the stream's source path or URI.
        codec_name: codec as reported by the sensor, or ``None`` on error.
        has_bframes: B-frame (frame-reordering) flag, or ``None`` on error.
        num_samples: number of timestamps analyzed, or ``None`` on error.
        start_ns / end_ns: first / last timestamp in nanoseconds, or ``None``.
        metrics: per-metric results (empty on error).
        error: failure message if the stream could not be opened/decoded.
        expected_hz: effective expected sample rate for this stream, or ``None``
            when unavailable or on error. Resolved per stream, since each stream
            carries its own header rate.
        expected_hz_source: where ``expected_hz`` came from, or ``None`` on error.
            Reported because it explains the rate / gap / jitter verdicts: an
            ``UNAVAILABLE`` rate is why those three checks SKIP.
        selector_type / selector_value: which stream *inside* ``source`` this is
            (see :mod:`~cosmos_curator.core.sensors.data_integrity.identity`).
            Carried so the store can key on it: two runs of ``di-check`` over one
            file with different ``--stream-idx`` are two streams, and without this
            they would overwrite each other's rows.

    """

    source: str
    codec_name: str | None
    has_bframes: bool | None
    num_samples: int | None
    start_ns: int | None
    end_ns: int | None
    metrics: list[CheckResult]
    error: str | None = None
    expected_hz: float | None = None
    expected_hz_source: ExpectedHzSource | None = None
    selector_type: str = identity.DEFAULT_SELECTOR_TYPE
    selector_value: str = identity.DEFAULT_SELECTOR_VALUE

    @property
    def status(self) -> OverallStatus:
        """Roll the metric statuses (or an open/decode error) into one verdict."""
        if self.error is not None:
            return OverallStatus.ERROR
        if any(m.status is CheckStatus.FAIL for m in self.metrics):
            return OverallStatus.FAIL
        return OverallStatus.PASS


def stream_result(
    source: str,
    metrics: list[CheckResult],
    video_info: VideoInfo,
    resolved_cfg: ResolvedConfig,
    *,
    selector_value: str = identity.DEFAULT_SELECTOR_VALUE,
) -> StreamResult:
    """Package one successful engine run as a :class:`StreamResult`.

    Shared so the single-video CLI and the session runner describe a checked stream
    identically -- which is what lets both write the same rows to the store.

    ``selector_value`` is the stream index that was opened: the single-video CLI
    passes its ``--stream-idx``, while the session runner takes the default because
    it opens one stream per file.
    """
    return StreamResult(
        source=source,
        codec_name=video_info.codec_name,
        has_bframes=video_info.has_bframes,
        num_samples=video_info.num_samples,
        start_ns=video_info.start_ns,
        end_ns=video_info.end_ns,
        metrics=metrics,
        expected_hz=resolved_cfg.expected_hz,
        expected_hz_source=resolved_cfg.expected_hz_source,
        selector_value=selector_value,
    )


@attrs.define(frozen=True)
class SessionReport:
    """Integrity results for one session (all of its streams).

    A future multi-session run is simply a ``list[SessionReport]``; this type
    deliberately models a single session only.

    Attributes:
        session_path: the session path / prefix the streams were discovered under.
        streams: per-stream results, in discovery order.
        metrics: session-grain results -- the checks whose subject is the session
            rather than any one stream, such as whether its sensors were recording
            over the same interval. Empty when nothing session-grain was run.

    """

    session_path: str
    streams: list[StreamResult]
    metrics: list[CheckResult] = attrs.field(factory=list)

    @property
    def status(self) -> OverallStatus:
        """Session verdict: ``ERROR`` if any stream errored, else ``FAIL`` if anything failed, else ``PASS``.

        ``ERROR`` outranks ``FAIL`` because the two are different kinds of
        statement. A ``FAIL`` is a completed measurement judged against the
        thresholds in force, so it can be re-judged if that policy changes, and
        may then pass. An ``ERROR`` stream was never measured, so no
        re-evaluation can complete it: the session's verdict stays partial until
        that stream is read again. Ranking ``ERROR`` first is what lets a
        partially measured session be recognised, and re-queued, from its verdict
        alone. Both are still counted in the report, so neither is hidden. An
        empty session (nothing discovered) is ``ERROR`` for the same reason:
        nothing was measured.

        A failing session-grain metric is a ``FAIL`` on the same terms as a failing
        stream: it is a judged measurement, so it re-judges. It cannot raise the
        verdict to ``ERROR``, because a session metric that could not be measured is
        ``SKIPPED`` rather than errored -- an unmeasurable session is already an
        ``ERROR`` by way of the streams that made it one.
        """
        if not self.streams:
            return OverallStatus.ERROR
        statuses = {s.status for s in self.streams}
        if OverallStatus.ERROR in statuses:
            return OverallStatus.ERROR
        if OverallStatus.FAIL in statuses or any(check.status is CheckStatus.FAIL for check in self.metrics):
            return OverallStatus.FAIL
        return OverallStatus.PASS
