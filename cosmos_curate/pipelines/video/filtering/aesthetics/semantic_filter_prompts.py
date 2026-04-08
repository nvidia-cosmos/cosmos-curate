# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prompts and categories for Qwen-based filtering and video-type classification.

Used by VllmFilteringStage (semantic filter) and VllmVideoClassifierStage (type allow/block).
"""

from loguru import logger

# 27 video type categories (underscores, no spaces) for CLI-safe single-token arguments.
VIDEO_TYPE_LABELS: tuple[str, ...] = (
    "movie/film_scene",
    "tv_drama/tv_series",
    "animation",
    "cartoon",
    "video_game",
    "news_reporting",
    "talk_show",
    "product_review/unboxing",
    "how_to/demo",
    "digital_illustration/digital_text",
    "visual_pattern",
    "egocentric_video/walking_POV",
    "vehicle_POV",
    "aerial_footage",
    "travel/luggage/plane",
    "food/drink",
    "cooking",
    "sports/exercise",
    "clothing/fashion/makeup",
    "exterior_building/cityscape",
    "interior_spaces",
    "nature_environment",
    "animal/insect/wildlife",
    "pets",
    "person/crowd",
    "talking_head",
    "other",
)

_PROMPTS = {
    "custom": """
    Can you answer the following questions about this video:
    """,
    "default": """Can you answer the following questions about this video:
        1. Is this a slideshow (e.g., only showing static images, animated text / image, etc.)
        or a video with slide transitions?
        2. Is this a synthetic video (e.g., screen recording, motion graphics, AI-generated video,
        stop motion, slideshow, etc.)
        as opposed to a video captured by an optical camera sensor?
        3. Does this video have visual filters (e.g., editing properties
        like grain, noise, saturation, color, aliasing, simulated weather effect, etc.)?
        4. Does this video have text overlaid in post-production (e.g., watermark, subtitles, logo, graphics, etc.)?
        Text that is part of the original video content is not considered as post-production text.
        5. Is this video a video in video (e.g., video overlay/collage, etc.)?
        6. Does this video have bad photographic artifacts(e.g., over/under exposure, lens flare, poor focus, etc.)?
        7. Does this video have distorted view (e.g., fisheye effect form wide field of view)?
        8. Is the video rotated to an uncommon view?
        9. Does this video have low resolution or is it blurry for all or some of the frames?
        10. Does this video contain any blurred or pixelated region
        (e.g., on specific human faces or objects, background, logo region, etc.)?
        11. Does this video mainly involve camera movement and little scene dynamics?
        12. Does this video contain abrupt / very large camera motion or camera shake (e.g., in some
        frames the camera is moving or rotated so fast that you cannot see clearly what's happening).
        13. Does this video involve fast zoom in or zoom out of the camera?
        14. Does this video have unnatural speed (e.g., slow motion, time lapse, frame skipping, etc.)?
        Answer format:
        {
        "slideshow": "yes" or "no",
        "synthetic video": "yes" or "no",
        "visual filter": "yes" or "no",
        "post-production text": "yes" or "no",
        "video in video": "yes" or "no",
        "photographic artifacts": "yes" or "no",
        "distorted view": "yes" or "no",
        "rotated view": "yes" or "no",
        "low resolution": "yes" or "no",
        "blurred region": "yes" or "no",
        "little scene dynamics": "yes" or "no",
        "abrupt camera motion": "yes" or "no",
        "fast zoom": "yes" or "no",
        "unnatural speed": "yes" or "no"
}""",
    "type": (
        "Classify this video into the following categories. For each category, answer yes or no "
        "depending on whether the video content matches that category. "
        "Please provide text in the specified json format.\n\n"
        + "\n".join(f"Is this video: {label}?" for label in VIDEO_TYPE_LABELS)
        + "\n\nAnswer format: {\n"
        + ",\n".join(f'"{label}": "yes" or "no"' for label in VIDEO_TYPE_LABELS)
        + "\n}"
        + "\n\nPlease provide text in the specified json format.\n\n"
    ),
}

# Default filter criteria per prompt variant. For "type", use VIDEO_TYPE_LABELS.
# When using a custom prompt with "--qwen-filter-categories" the criteria are generated from that.
FILTER_CRITERIA: dict[str, list[str]] = {
    "default": [
        "slideshow",
        "synthetic video",
        "visual filter",
        "post-production text",
        "video in video",
        "photographic artifacts",
        "distorted view",
        "rotated view",
        "low resolution",
        "blurred region",
        "abrupt camera motion",
        "fast zoom",
        "unnatural speed",
    ],
    "type": list(VIDEO_TYPE_LABELS),
}


def get_qwen_filter_prompt(
    prompt_variant: str,
    filter_categories: str | None,
    *,
    verbose: bool = False,
) -> str:
    """Return the filtering/classification prompt for the Qwen model.

    Args:
        prompt_variant: One of _PROMPTS keys (e.g. "default", "type").
        filter_categories: Optional comma-separated categories for custom prompt.
        verbose: Log the chosen prompt.

    Returns:
        The prompt string.

    Raises:
        ValueError: If prompt_variant is not in _PROMPTS when filter_categories is None.

    """
    if filter_categories is not None:
        try:
            categories = filter_categories.split(",")
            prompt = _PROMPTS["custom"]
            for category in categories:
                prompt += f"Is there {category} in the video?\n"
            prompt += "\nAnswer format: {\n"
            for i, category in enumerate(categories):
                comma = "," if i < len(categories) - 1 else ""  # No comma for last item
                prompt += f""""{category}": "yes" or "no"{comma}\n"""
            prompt += "}"
        except AttributeError:
            logger.warning(f"Prompt text is not a comma separated list: {filter_categories}")
            prompt = _PROMPTS["default"]
    else:
        if prompt_variant not in _PROMPTS:
            error_msg = f"Invalid prompt variant: {prompt_variant}"
            raise ValueError(error_msg)
        prompt = _PROMPTS[prompt_variant]
    if verbose:
        logger.debug(f"Filtering prompt: {prompt}")
    return prompt
