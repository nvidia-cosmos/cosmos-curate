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

"""Pure text preprocessing for the BGE leg (no torch, deterministic).

The task / subtask strings on the clips table carry incidental whitespace from
their many upstream sources (tabs, newlines, doubled spaces). BGE is
whitespace-sensitive, so two clips with the same instruction but different
spacing would otherwise land at slightly different points in the embedding
space. Collapsing runs of whitespace to a single space makes the text input a
deterministic function of the instruction's words alone.

``format_task`` and ``format_subtask`` apply the identical normalization today;
the two names exist for call-site readability at the two source fields, not
because the transforms differ. Should task and subtask ever need distinct
handling, the split already exists to carry it.
"""


def _collapse_whitespace(text: str | None) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends.

    ``None`` (an absent field) formats to the empty string so the row still
    produces a (zero-content) embedding rather than crashing the batch.
    """
    if not text:
        return ""
    return " ".join(text.split())


def format_task(task_name: str | None) -> str:
    """Return the whitespace-normalized task instruction for embedding."""
    return _collapse_whitespace(task_name)


def format_subtask(subtask_name: str | None) -> str:
    """Return the whitespace-normalized subtask instruction for embedding."""
    return _collapse_whitespace(subtask_name)
