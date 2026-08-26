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

"""Unit tests for the function the Ray Data stage runs on one session."""

import datetime
import pathlib
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from cosmos_curator.core.sensors.data_integrity.instruments import INSTRUMENTS
from cosmos_curator.core.sensors.data_integrity.results import StreamResult
from cosmos_curator.next.recipes.data_integrity import processing, store_schema
from cosmos_curator.next.recipes.data_integrity.config import ResolvedDataIntegrityConfig

RUN_ID = "0" * 32
CREATED_AT = datetime.datetime(2026, 8, 20, 12, 0, tzinfo=datetime.UTC)

#: How many metrics a stream with a usable declared rate produces, and therefore how
#: many measurement rows and how many evaluations it contributes.
NUM_METRICS = len(INSTRUMENTS)


def _config(session: pathlib.Path, **execution: Any) -> ResolvedDataIntegrityConfig:  # noqa: ANN401
    """Resolve a config covering one local session."""
    return ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)]},
            "output": {"store_root": str(session.parent / "di-store")},
            "execution": execution,
        }
    )


@pytest.fixture
def session(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> pathlib.Path:
    """Build a session directory holding one readable video and one unreadable one."""
    path = tmp_path / "clips" / "session"
    path.mkdir(parents=True)
    (path / "front.mp4").write_bytes(h264_video())
    (path / "rear.mp4").write_bytes(b"not a video")
    return path


def _check(session: pathlib.Path, **execution: Any) -> dict[str, object]:  # noqa: ANN401
    """Measure one session the way the Ray stage does."""
    return processing.check_session(
        {"session_path": str(session)},
        config=_config(session, **execution),
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )


def _rows(record: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Unpack the opaque payload the driver appends."""
    return pickle.loads(record["rows"])  # type: ignore[arg-type]  # noqa: S301 -- written by check_session


def test_a_session_lists_its_streams_in_discovery_order(session: pathlib.Path) -> None:
    """Listing is the first half of the task, and it is ordinary discovery."""
    assert processing.discover_session(str(session), config=_config(session)) == [
        str(session / "front.mp4"),
        str(session / "rear.mp4"),
    ]


def test_listing_honours_the_per_session_stream_cap(session: pathlib.Path) -> None:
    """``limit`` caps streams within a session; it never drops a session."""
    config = ResolvedDataIntegrityConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "data-integrity",
            "input": {"sessions": [str(session)], "limit": 1},
            "output": {"store_root": str(session.parent / "di-store")},
        }
    )

    assert processing.discover_session(str(session), config=config) == [str(session / "front.mp4")]


def test_a_failed_listing_escapes_for_rays_retry(tmp_path: pathlib.Path) -> None:
    """Nothing has been measured yet, so the error belongs to Ray, not to a row."""
    missing = tmp_path / "absent"

    with pytest.raises(OSError, match="absent"):
        processing.check_session(
            {"session_path": str(missing)},
            config=_config(missing),
            run_id=RUN_ID,
            created_at=CREATED_AT,
        )


def test_a_session_yields_the_rows_of_every_stream_it_holds(session: pathlib.Path) -> None:
    """One task covers the whole session: a row per stream, plus the readable one's metrics."""
    record = _check(session)
    rows = _rows(record)

    assert record["streams"] == 2
    assert len(rows[store_schema.STREAM_DATASET]) == 2
    assert len(rows[store_schema.EVALUATION_DATASET]) == NUM_METRICS
    assert sum(len(rows[store_schema.metric_dataset_path(spec.name)]) for spec in INSTRUMENTS) == NUM_METRICS


def test_an_unreadable_stream_does_not_cost_its_siblings(session: pathlib.Path) -> None:
    """One broken video must not take out the rest of its session, or the run."""
    record = _check(session)
    stream_rows = _rows(record)[store_schema.STREAM_DATASET]

    assert record["unreadable"] == 1
    assert record["failed_metrics"] == 0
    errored = [row for row in stream_rows if row["error"] is not None]
    assert [row["source"] for row in errored] == [str(session / "rear.mp4")]
    # The readable sibling is still measured, and only it is judged: an errored stream
    # was never measured, so there is nothing to judge.
    assert len(stream_rows) == 2


def test_every_row_carries_the_runs_identity_and_its_session(session: pathlib.Path) -> None:
    """One run id across the invocation, one session path on every row of it."""
    for dataset_rows in _rows(_check(session)).values():
        for row in dataset_rows:
            assert row["run_id"] == RUN_ID
            assert row["session_path"] == str(session)
            assert row["created_at"] == CREATED_AT


def test_an_empty_session_produces_a_result_but_no_rows(tmp_path: pathlib.Path) -> None:
    """An empty session is a fact about the input, not a failure of the task."""
    empty = tmp_path / "empty"
    empty.mkdir()
    record = _check(empty)

    assert record["streams"] == 0
    assert record["unreadable"] == 0
    assert _rows(record) == {}


def test_the_configured_attempt_budget_reaches_every_stream(
    session: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execution.stream_attempts`` is the retry knob, and it is not decorative."""
    seen: list[object] = []

    def _record(source: str, **kwargs: object) -> StreamResult:
        seen.append(kwargs["max_attempts"])
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error="stubbed",
        )

    monkeypatch.setattr(processing, "run_one_stream", _record)

    _check(session, stream_attempts=5)

    assert seen == [5, 5]


def test_the_payload_names_only_datasets_the_driver_can_write(session: pathlib.Path) -> None:
    """The payload's keys and the schemas they are written under are one contract."""
    assert set(_rows(_check(session))) <= set(processing.DATASET_SCHEMAS)
