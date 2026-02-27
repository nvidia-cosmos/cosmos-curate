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
"""CurationPhase protocol and PipelineBuilder for composable video curation pipelines.

A CurationPhase is a named, self-contained unit that produces a list of pipeline stages
and declares which Clip fields it requires as inputs and which it populates as outputs.

PipelineBuilder validates that the composition of phases is internally consistent at
construction time — before any data flows. If a phase requires fields not yet populated
by prior phases, the builder raises immediately with a descriptive error.

Usage::

    builder = PipelineBuilder()
    builder.add_phase(IngestPhase(ingest_cfg))
    builder.add_phase(SplitPhase(split_cfg))     # requires "remuxed" from IngestPhase
    builder.add_phase(TranscodePhase(xcode_cfg)) # requires "split" from SplitPhase
    stages = builder.build()

The requires/populates sets are abstract field tokens. For phases wrapping the existing
splitting pipeline (Spike 0), they represent logical pipeline states. Starting from
Spike 1, new phases can additionally refer to optional attrs fields on the Clip data model.
"""

from abc import ABC, abstractmethod
from typing import Self

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec


class CurationPhase(ABC):
    """A named, self-contained pipeline phase.

    Subclasses declare which Clip field tokens they consume (requires) and produce
    (populates). PipelineBuilder uses these declarations to validate that no phase is
    added before the fields it depends on have been made available by a prior phase.

    The field tokens are plain strings — they do not need to correspond to actual
    attribute names on Clip for Spike 0 phases. Once Spike 1 adds optional Clip
    fields, new phases should use the matching attribute name as the token.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this phase, used in error messages."""

    @property
    @abstractmethod
    def requires(self) -> frozenset[str]:
        """Field tokens that must be populated by prior phases before this phase runs."""

    @property
    @abstractmethod
    def populates(self) -> frozenset[str]:
        """Field tokens that this phase will populate on Clip objects."""

    @abstractmethod
    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Construct and return the ordered list of stages for this phase."""


class PipelineBuilder:
    """Validates and assembles a stage list from an ordered sequence of CurationPhase instances.

    Call add_phase() for each phase in execution order. Validation happens at add_phase()
    time — an error is raised immediately if a phase declares requirements that no prior
    phase has satisfied. Call build() to obtain the flat, ordered stage list.

    The builder is a construction-time tool only; it does not participate in data flow
    at runtime. The stage list it produces is structurally identical to a manually
    assembled list.
    """

    def __init__(self) -> None:
        """Initialise an empty builder with no phases added."""
        self._phases: list[CurationPhase] = []
        self._available_fields: set[str] = set()

    def add_phase(self, phase: CurationPhase) -> Self:
        """Add a phase to the pipeline, validating its requirements first.

        Args:
            phase: The CurationPhase to add.

        Returns:
            self, to enable method chaining.

        Raises:
            ValueError: If phase.requires contains tokens not yet in available_fields.

        """
        missing = phase.requires - self._available_fields
        if missing:
            available = sorted(self._available_fields) or ["(none)"]
            msg = (
                f"Phase '{phase.name}' requires fields {sorted(missing)} "
                f"that have not been populated by any prior phase. "
                f"Available fields: {available}"
            )
            raise ValueError(msg)
        self._phases.append(phase)
        self._available_fields |= phase.populates
        return self

    def build(self) -> list[CuratorStage | CuratorStageSpec]:
        """Build and return the complete, ordered stage list.

        Returns:
            Stages from all added phases, in the order phases were added.

        """
        stages: list[CuratorStage | CuratorStageSpec] = []
        for phase in self._phases:
            stages.extend(phase.build_stages())
        return stages
