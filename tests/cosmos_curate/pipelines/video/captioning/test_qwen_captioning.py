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

"""Test the QWEN result."""

import pathlib
import uuid
from collections import Counter

import pytest
from scipy.spatial.distance import cosine  # type: ignore[import-untyped]

from cosmos_curate.pipelines.video.captioning.captioning_stages import (  # type: ignore[import-untyped]
    QwenCaptionStage,
    QwenInputPreparationStage,
)
from cosmos_curate.pipelines.video.clipping.clip_extraction_stages import (  # type: ignore[import-untyped]
    ClipTranscodingStage,
)
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video  # type: ignore[import-untyped]
from tests.utils.sequential_runner import run_pipeline

_THRESHOLD = 0.8
_NUM_CLIPS = 2


def tf_vector(sentence: str) -> dict[str, float]:
    """Compute term frequency vector for a sentence.

    Args:
        sentence: Sentence to compute term frequency vector for.

    Returns:
        Term frequency vector for the sentence.

    """
    words = sentence.lower().split()
    tf = Counter(words)
    total = sum(tf.values())
    return {word: count / total for word, count in tf.items()}


def cosine_similarity(s1: str, s2: str) -> float:
    """Compute cosine similarity between two sentences.

    Args:
        s1: First sentence.
        s2: Second sentence.

    Returns:
        Cosine similarity between the two sentences.

    """
    tf1 = tf_vector(s1)
    tf2 = tf_vector(s2)
    all_words = list(set(tf1.keys()) | set(tf2.keys()))

    v1 = [tf1.get(w, 0) for w in all_words]
    v2 = [tf2.get(w, 0) for w in all_words]

    return 1 - cosine(v1, v2)  # type: ignore[no-any-return]


@pytest.fixture
def sample_captioning_task(sample_video_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample embedding task."""
    clips = []
    for start, end in [(2.5, 6.5), (8.0, 14.0)]:
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"sample_video.mp4#{start}-{end}"),
            source_video="sample_video.mp4",
            span=(start, end),
        )
        clips.append(clip)

    video = Video(
        input_video=pathlib.Path("sample_video.mp4"),
        source_bytes=sample_video_data,
        clips=clips,
    )
    return SplitPipeTask(
        video=video,
    )


@pytest.mark.env("vllm")
def test_generate_embedding(sample_captioning_task: SplitPipeTask) -> None:
    """Test the QwenCaptioning result."""
    stages = [
        ClipTranscodingStage(encoder="libopenh264"),
        QwenInputPreparationStage(sampling_fps=2.0),
        QwenCaptionStage(),
    ]
    tasks = run_pipeline([sample_captioning_task], stages)

    # Validate that captions were generated
    assert tasks is not None
    assert len(tasks) > 0
    assert len(tasks[0].video.clips) == _NUM_CLIPS

    # Define expected captions for validation (adjust based on your video content)
    expected_captions = [
        (
            "The video begins with a scene featuring a red pickup truck parked on a cobblestone street. "
            'The truck is adorned with various decals, including "Black Magic," "419," and logos for '
            '"Pepsi Max" and "Bullrun.com." A man is seated on the back of the truck, holding a yellow cup, '
            "and appears to be engaged in conversation or gesturing with his hand. The background shows a brick "
            "building with green shutters and a few other individuals standing nearby, suggesting an "
            "urban setting.\n\n"
            "The scene then transitions to a different setting where the same red pickup truck is now in motion "
            "on a road. The truck is driving at high speed, as indicated by the blurred background, which consists "
            "of greenery and a clear sky. The vehicle's wheels are spinning rapidly, emphasizing its speed and "
            "movement. The truck's decals remain visible, reinforcing the branding and sponsorship details from "
            "the previous scene. This transition suggests a shift from a stationary, casual moment to an "
            "action-packed, dynamic one, possibly indicating a change in the narrative or context of the video."
        ),
        (
            "The video begins with an interior shot of a vehicle, focusing on the driver's side. The driver, "
            "a bald man wearing glasses and a dark jacket, is seen from the side as he speaks into a microphone "
            "attached to his headgear. The microphone is connected to a device mounted on the dashboard, "
            "suggesting that he might be recording or communicating via radio. The car's interior is visible, "
            "including the steering wheel and dashboard, which have a modern "
            "design with various controls and displays.\n\n"
            "The scene then transitions to an exterior view of a busy street. The camera captures a series of cars "
            'driving in traffic, with some vehicles prominently displaying the "Bullrun" logo on their sides. '
            "These cars appear to be part of a rally or event, as indicated by their racing decals and the presence "
            "of other high-performance vehicles in the background. The street is lined with trees and buildings, "
            "and there are construction cones visible, indicating ongoing roadwork or maintenance.\n\n"
            "As the camera continues to pan, it focuses on a blue Mitsubishi Lancer Evolution, a well-known rally car, "
            "driving alongside the other vehicles. This car is particularly noticeable due to its vibrant blue color "
            'and the "Bullrun" branding on its hood and doors. The driver of this car is seen through the window, '
            "adding a human element to the scene. The surrounding traffic includes a mix of sedans, SUVs, and other "
            "rally cars, all moving at a steady pace. The overall "
            "atmosphere suggests a lively and dynamic environment, "
            "typical of a rally or motorsport event.\n\n"
            "The video effectively combines the intimate perspective of the driver inside the car with the broader "
            "context of the event, providing a comprehensive view of the rally experience. The transition between "
            "the interior and exterior shots helps to build a narrative that captures both the personal aspect of "
            "driving and the excitement of the event itself."
        ),
    ]

    for clip_idx, clip in enumerate(tasks[0].video.clips):
        assert len(clip.windows) > 0, f"Clip {clip_idx} should have at least one window"

        for window_idx, window in enumerate(clip.windows):
            assert "qwen" in window.caption, f"Clip {clip_idx} window {window_idx} should have qwen caption"

            generated_caption = window.caption["qwen"]
            expected_caption = (
                expected_captions[clip_idx] if clip_idx < len(expected_captions) else expected_captions[0]
            )

            similarity = cosine_similarity(generated_caption, expected_caption)
            assert similarity >= _THRESHOLD, (
                f"Caption similarity {similarity:.3f} below threshold for clip {clip_idx} window {window_idx}"
            )
