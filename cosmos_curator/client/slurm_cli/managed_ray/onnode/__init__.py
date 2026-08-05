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
"""Modules uploaded verbatim into a run directory and executed on cluster nodes.

Submission copies these files into the private run directory, where a login or compute host runs them as plain
scripts. That host has no Cosmos Curator installation and no Pixi environment, which imposes two contracts the
rest of the codebase does not carry. Both are enforced by ``test_onnode.py``.

**Standard library only at module scope.** Anything heavier belongs in an inline import inside the function that
needs it.

**Python 3.8, not 3.12.** The interpreter is whatever ``python3`` resolves to on a login or compute node, which
across surveyed Slurm clusters ranges from 3.8.10 to 3.12.x. 3.8 is the floor because ``typing.TypedDict`` and
``typing.Literal`` arrived there, and below it this state model would need ``typing_extensions``, which the rule
above forbids. Each module therefore opens with ``from __future__ import annotations``, which the project
otherwise bans, keeping PEP 585 and PEP 604 annotations unevaluated. That import covers annotations only:
everything else must avoid post-3.8 syntax and APIs, and a module-level type alias is the trap worth naming
because it looks like an annotation but is an ordinary assignment.

Their filenames are part of the deployed contract: the modules import each other by flat name when executed
standalone, so they keep distinctive names rather than taking short ones from this package.
"""
