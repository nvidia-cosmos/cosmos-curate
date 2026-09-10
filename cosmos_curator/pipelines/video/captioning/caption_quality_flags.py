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
"""Deterministic structural-health annotations for caption windows.

The ``flag_*`` fields are monitoring signals for suspicious caption shapes, not
semantic caption-quality scores, output gates, or policy decisions.
"""

import string
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from cosmos_curator.pipelines.video.utils.data_model import Window

type _EvaluableCaption = tuple["Window", str, list[str]]

TRIGRAM_SIZE = 3
REPEATED_WORD_SHARE_THRESHOLD = 0.5
REPEATED_WORD_DOMINANCE_MIN_TOTAL_WORDS = 6
LOW_UNIQUE_WORD_RATIO_THRESHOLD = 0.4
LOW_UNIQUE_WORD_RATIO_MIN_TOTAL_WORDS = 8
# Near-duplicate checks use word-set Jaccard, intentionally ignoring order and multiplicity.

_EVALUABLE_CAPTION_STATUSES = frozenset({"success", "truncated"})


@attrs.define(frozen=True)
class CaptionQualityThresholdConfig:
    """Operator-configurable thresholds for caption quality flags."""

    length_floor_words: int = attrs.field(default=4, validator=attrs.validators.ge(0))
    # Conservative runaway-output guard, not a calibrated "long caption" threshold.
    length_ceiling_words: int = attrs.field(default=1024)
    # Keep trigram detection conservative: long structured captions can repeat benign phrasing.
    repeated_trigram_min_count: int = attrs.field(default=5, validator=attrs.validators.ge(2))
    near_duplicate_jaccard_threshold: float = attrs.field(
        default=0.9,
        validator=attrs.validators.and_(attrs.validators.gt(0), attrs.validators.le(1)),
    )

    @length_ceiling_words.validator
    def _validate_length_ceiling_words(self, _attribute: "attrs.Attribute[int]", value: int) -> None:
        minimum = max(self.length_floor_words, 1)
        if value < minimum:
            msg = f"length_ceiling_words must be at least {minimum}"
            raise ValueError(msg)


DEFAULT_CAPTION_QUALITY_THRESHOLDS = CaptionQualityThresholdConfig()


def _normalize_caption_text(text: str) -> str:
    """Normalize case and whitespace, stripping only ASCII trailing punctuation."""
    normalized = " ".join(text.lower().strip().split())
    return normalized.rstrip(string.punctuation).strip()


def _is_length_outlier(words: Sequence[str], thresholds: CaptionQualityThresholdConfig) -> bool:
    return len(words) < thresholds.length_floor_words or len(words) > thresholds.length_ceiling_words


def _has_repeated_trigram(words: Sequence[str], thresholds: CaptionQualityThresholdConfig) -> bool:
    if len(words) < TRIGRAM_SIZE:
        return False
    trigrams = Counter(tuple(words[index : index + TRIGRAM_SIZE]) for index in range(len(words) - TRIGRAM_SIZE + 1))
    return any(count >= thresholds.repeated_trigram_min_count for count in trigrams.values())


def _has_repetition(words: Sequence[str], thresholds: CaptionQualityThresholdConfig) -> bool:
    if len(words) >= REPEATED_WORD_DOMINANCE_MIN_TOTAL_WORDS:
        _, most_common_count = Counter(words).most_common(1)[0]
        if most_common_count / len(words) >= REPEATED_WORD_SHARE_THRESHOLD:
            return True

    if _has_repeated_trigram(words, thresholds):
        return True

    if len(words) >= LOW_UNIQUE_WORD_RATIO_MIN_TOTAL_WORDS:
        unique_word_ratio = len(set(words)) / len(words)
        if unique_word_ratio < LOW_UNIQUE_WORD_RATIO_THRESHOLD:
            return True

    return False


def _jaccard_similarity(left_words: Sequence[str], right_words: Sequence[str]) -> float:
    left = set(left_words)
    right = set(right_words)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _is_near_duplicate(
    left_text: str,
    left_words: Sequence[str],
    right_text: str,
    right_words: Sequence[str],
    thresholds: CaptionQualityThresholdConfig,
) -> bool:
    return (
        left_text == right_text
        or _jaccard_similarity(left_words, right_words) >= thresholds.near_duplicate_jaccard_threshold
    )


def _get_evaluable_caption(window: "Window", model_variant: str) -> tuple[str, list[str]] | None:
    if window.caption_status not in _EVALUABLE_CAPTION_STATUSES:
        return None

    caption = window.caption.get(model_variant)
    if caption is None:
        return None

    normalized = _normalize_caption_text(caption)
    if not normalized:
        return None

    return normalized, normalized.split()


def apply_caption_quality_flags(
    window_groups: Iterable[Sequence["Window"]],
    model_variant: str,
    thresholds: CaptionQualityThresholdConfig = DEFAULT_CAPTION_QUALITY_THRESHOLDS,
) -> None:
    """Apply heuristic caption quality flags to grouped caption windows in place.

    This is a captioning-path post-processing utility rather than a standalone
    stage by design: it runs after captions are scattered onto ``clip.windows``,
    uses only the active ``model_variant`` caption key, respects caption-stage
    gating such as filter-window mode, and avoids adding a separate stage for
    metadata-only annotations.
    """
    for windows in window_groups:
        indexed_windows = list(enumerate(windows))
        evaluable: dict[int, _EvaluableCaption] = {}

        for original_index, window in indexed_windows:
            window.flag_length_outlier = None
            window.flag_repetition = None
            window.flag_near_duplicate = None

            caption = _get_evaluable_caption(window, model_variant)
            if caption is None:
                continue

            normalized_text, words = caption
            window.flag_length_outlier = _is_length_outlier(words, thresholds)
            window.flag_repetition = _has_repetition(words, thresholds)
            window.flag_near_duplicate = False
            evaluable[original_index] = (window, normalized_text, words)

        sorted_windows = sorted(indexed_windows, key=lambda item: (item[1].start_frame, item[1].end_frame, item[0]))
        for (left_index, _), (right_index, _) in pairwise(sorted_windows):
            left = evaluable.get(left_index)
            right = evaluable.get(right_index)
            if left is None or right is None:
                continue

            left_window, left_text, left_words = left
            right_window, right_text, right_words = right
            if _is_near_duplicate(left_text, left_words, right_text, right_words, thresholds):
                left_window.flag_near_duplicate = True
                right_window.flag_near_duplicate = True
