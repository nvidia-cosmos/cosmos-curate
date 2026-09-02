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

"""Core contracts and registration for config-backed pipeline kinds.

A registered kind's config model MUST declare ``schema_version`` and ``kind`` as
required fields with no default, and its ``kind`` must be a per-recipe ``Literal``
equal to ``PipelineKind.name``. That equality is checked twice, in layers that do
not see each other: the CLI reads ``kind`` from the file to pick the adapter, and
the model's ``Literal`` re-checks it, so a routed-but-mismatched file and a direct
``resolve_config`` caller both fail rather than one of them slipping through. No
default, because a defaulted ``schema_version`` would let a later generation's
file parse under this generation's field meanings.

``prepare_run`` is two-phase by contract, not as an optimization. It resolves the
config eagerly and returns a closure that runs it, which is what lets a caller
classify a failure by where it arose -- a config fault surfaces from
``prepare_run``, anything else from the closure -- without any adapter catching or
re-labelling exceptions. Adapters therefore catch nothing.

Adapters are imported at CLI startup, so an adapter's module scope must stay free
of its recipe's config model and runtime dependencies; every such import belongs
inside the callback that needs it.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NotRequired, Protocol, TypedDict


@dataclass(frozen=True)
class PipelineRunOutput:
    """Human- and machine-readable results from a config-backed pipeline run."""

    json_payload: dict[str, object]
    message: str


PreparedPipelineRun = Callable[[], PipelineRunOutput]


class PipelinePreset(TypedDict):
    """Metadata required for a discoverable pipeline preset."""

    name: str
    qualified_name: str
    fragment: dict[str, Any]
    section: NotRequired[str]


class PipelineRunPreparer(Protocol):
    """Resolve one user config and return its deferred runtime invocation."""

    def __call__(
        self,
        config: Path,
        *,
        set_overrides: list[str],
    ) -> PreparedPipelineRun:
        """Prepare one deferred pipeline execution."""
        ...


@dataclass(frozen=True)
class PipelineKind:
    """Config and runtime surface registered for one pipeline discriminator."""

    name: str
    template_yaml: Callable[[], str]
    template_payload: Callable[[], dict[str, Any]]
    validate: Callable[[Path, Sequence[str]], dict[str, object]]
    render: Callable[[Path, Sequence[str]], str]
    schema_json: Callable[[], str]
    list_presets: Callable[[], list[PipelinePreset]]
    prepare_run: PipelineRunPreparer


class PipelineKindRegistry:
    """Immutable collection of uniquely named pipeline-kind strategies."""

    def __init__(self, pipeline_kinds: Iterable[PipelineKind]) -> None:
        """Index pipeline kinds and reject empty or duplicate names."""
        by_name: dict[str, PipelineKind] = {}
        for pipeline_kind in pipeline_kinds:
            if not pipeline_kind.name:
                msg = "Pipeline kind names must not be empty"
                raise ValueError(msg)
            if pipeline_kind.name in by_name:
                msg = f"Duplicate pipeline kind: {pipeline_kind.name}"
                raise ValueError(msg)
            by_name[pipeline_kind.name] = pipeline_kind
        self._by_name: Mapping[str, PipelineKind] = MappingProxyType(by_name)

    def get(self, name: str) -> PipelineKind:
        """Return one registered kind with a useful unsupported-name error."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            valid = ", ".join(self.names())
            msg = f"Unknown pipeline kind {name!r}. Valid pipeline kinds: {valid}"
            raise ValueError(msg) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered discriminator names in stable order."""
        return tuple(sorted(self._by_name))

    def __iter__(self) -> Iterator[PipelineKind]:
        """Iterate over registered kinds in stable name order."""
        return (self._by_name[name] for name in self.names())
