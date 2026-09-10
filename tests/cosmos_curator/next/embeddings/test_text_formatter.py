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

"""Pure text-formatter tests (deterministic whitespace normalization)."""

from cosmos_curator.next.embeddings.text.formatter import format_subtask, format_task


def test_collapses_mixed_whitespace_to_single_spaces() -> None:
    """Tabs, newlines, and doubled spaces collapse to single spaces."""
    assert format_task("  pick   up\tthe\nblock ") == "pick up the block"


def test_none_formats_to_empty_string() -> None:
    """An absent field formats to '' rather than crashing the batch."""
    assert format_task(None) == ""
    assert format_subtask(None) == ""


def test_formatting_is_idempotent() -> None:
    """Formatting an already-clean string is a no-op (stable input to BGE)."""
    once = format_subtask("stack the red cube")
    assert format_subtask(once) == once


def test_spacing_variants_map_to_one_canonical_string() -> None:
    """Same words, different spacing, produce identical model input (deterministic)."""
    assert format_task("open  the   drawer") == format_task("open the drawer")
