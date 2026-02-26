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
"""Unit tests for CurationPhase protocol and PipelineBuilder."""

import pytest

from cosmos_curate.core.interfaces.phase_interface import CurationPhase, PipelineBuilder
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec, PipelineTask

# ---------------------------------------------------------------------------
# Minimal concrete helpers
# ---------------------------------------------------------------------------


class _PassthroughStage(CuratorStage):
    """Minimal stage that passes tasks through unchanged."""

    def process_data(self, task: list[PipelineTask]) -> list[PipelineTask]:
        """Return tasks unmodified."""
        return task


def _make_stage() -> CuratorStage:
    """Return a fresh no-op passthrough stage."""
    return _PassthroughStage()


class _Phase(CurationPhase):
    """Parameterised test phase with configurable requires/populates and stage count."""

    def __init__(
        self,
        name: str,
        requires: frozenset[str],
        populates: frozenset[str],
        num_stages: int = 1,
    ) -> None:
        """Initialise the test phase with explicit field declarations."""
        self._name = name
        self._requires = requires
        self._populates = populates
        self._num_stages = num_stages

    @property
    def name(self) -> str:
        """Return the phase name."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the set of required field tokens."""
        return self._requires

    @property
    def populates(self) -> frozenset[str]:
        """Return the set of field tokens this phase populates."""
        return self._populates

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Return a list of no-op passthrough stages."""
        return [_make_stage() for _ in range(self._num_stages)]


# ---------------------------------------------------------------------------
# PipelineBuilder tests
# ---------------------------------------------------------------------------


def test_empty_builder_returns_empty_list() -> None:
    """A builder with no phases produces an empty stage list."""
    assert PipelineBuilder().build() == []


def test_single_phase_no_requirements() -> None:
    """A phase with no requirements can be added and contributes its stages."""
    phase = _Phase("alpha", frozenset(), frozenset({"a"}), num_stages=2)
    stages = PipelineBuilder().add_phase(phase).build()
    assert len(stages) == 2


def test_chained_phases_satisfied_requirements() -> None:
    """Phases added in dependency order produce a concatenated stage list."""
    p1 = _Phase("p1", frozenset(), frozenset({"a"}), num_stages=1)
    p2 = _Phase("p2", frozenset({"a"}), frozenset({"b"}), num_stages=2)
    p3 = _Phase("p3", frozenset({"a", "b"}), frozenset({"c"}), num_stages=1)

    stages = PipelineBuilder().add_phase(p1).add_phase(p2).add_phase(p3).build()
    assert len(stages) == 4  # 1 + 2 + 1


def test_missing_requirement_raises_value_error() -> None:
    """Adding a phase whose requirements are not satisfied raises ValueError naming the phase."""
    p1 = _Phase("p1", frozenset(), frozenset({"a"}))
    # p2 requires "b" which p1 does not provide
    p2 = _Phase("p2", frozenset({"b"}), frozenset({"c"}))

    builder = PipelineBuilder()
    builder.add_phase(p1)
    with pytest.raises(ValueError, match="'p2'"):
        builder.add_phase(p2)


def test_error_message_names_missing_fields() -> None:
    """The ValueError message includes the names of the unsatisfied field tokens."""
    p1 = _Phase("p1", frozenset(), frozenset({"x"}))
    p2 = _Phase("p2", frozenset({"x", "y", "z"}), frozenset())

    builder = PipelineBuilder()
    builder.add_phase(p1)
    with pytest.raises(ValueError, match="requires fields") as exc_info:
        builder.add_phase(p2)

    msg = str(exc_info.value)
    # both missing fields should be named in message
    assert "y" in msg
    assert "z" in msg


def test_adding_phase_out_of_order_fails() -> None:
    """Adding a phase before its prerequisite phase is available raises ValueError."""
    transcode = _Phase("transcode", frozenset({"split"}), frozenset({"transcoded"}))
    ingest = _Phase("ingest", frozenset(), frozenset({"remuxed"}))

    builder = PipelineBuilder()
    builder.add_phase(ingest)
    with pytest.raises(ValueError, match="split"):
        builder.add_phase(transcode)


def test_method_chaining_returns_builder() -> None:
    """add_phase() returns the builder instance to support method chaining."""
    p = _Phase("p", frozenset(), frozenset({"x"}))
    builder = PipelineBuilder()
    result = builder.add_phase(p)
    assert result is builder


def test_phases_with_overlapping_populates() -> None:
    """Two phases may populate the same field token without conflict."""
    p1 = _Phase("p1", frozenset(), frozenset({"a"}))
    p2 = _Phase("p2", frozenset({"a"}), frozenset({"a", "b"}))  # re-populates "a"
    p3 = _Phase("p3", frozenset({"b"}), frozenset())

    stages = PipelineBuilder().add_phase(p1).add_phase(p2).add_phase(p3).build()
    assert len(stages) == 3


def test_stage_order_preserved() -> None:
    """Stages are emitted in the order phases were added to the builder."""
    sentinel_stages: list[CuratorStage] = [_make_stage() for _ in range(3)]

    class _OrderedPhase(CurationPhase):
        """Test phase that returns a specific sentinel stage."""

        def __init__(self, idx: int) -> None:
            """Store the index of the sentinel stage to return."""
            self._idx = idx

        @property
        def name(self) -> str:
            """Return the phase name (the sentinel index as a string)."""
            return str(self._idx)

        @property
        def requires(self) -> frozenset[str]:
            """Return an empty set of requirements."""
            return frozenset()

        @property
        def populates(self) -> frozenset[str]:
            """Return an empty set of populated fields."""
            return frozenset()

        def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
            """Return the sentinel stage at this phase's index."""
            return [sentinel_stages[self._idx]]

    builder = PipelineBuilder()
    for i in range(3):
        builder.add_phase(_OrderedPhase(i))

    built = builder.build()
    assert built == sentinel_stages


def test_empty_phase_contributes_no_stages() -> None:
    """A phase with zero stages passes through without affecting the stage count."""
    p1 = _Phase("noops", frozenset(), frozenset({"x"}), num_stages=0)
    p2 = _Phase("real", frozenset({"x"}), frozenset(), num_stages=2)
    stages = PipelineBuilder().add_phase(p1).add_phase(p2).build()
    assert len(stages) == 2


def test_build_called_multiple_times_returns_independent_lists() -> None:
    """build() can be called multiple times and returns independent lists each time."""
    phase = _Phase("p", frozenset(), frozenset({"x"}), num_stages=2)
    builder = PipelineBuilder().add_phase(phase)

    first = builder.build()
    second = builder.build()

    assert len(first) == len(second)  # same number of stages
    assert first is not second  # distinct list objects
