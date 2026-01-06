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

"""Test vLLM captioning results."""

import pathlib
import uuid
from collections import Counter

import pytest
from scipy.spatial.distance import cosine

from cosmos_curate.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curate.core.interfaces.runner_interface import RunnerInterface
from cosmos_curate.pipelines.video.captioning.vllm_caption_stage import (
    VllmCaptionStage,
    VllmPrepStage,
)
from cosmos_curate.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)  # type: ignore[import-untyped]

_THRESHOLDS = {
    "qwen": 0.9,
    "cosmos_r1": 0.8,
    "cosmos_r2": 0.7,
}
_NUM_CLIPS = 1
_VLLM_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "cosmos_r2": {
        "preprocess": True,
    },
}
_WINDOW_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "qwen": {
        "sampling_fps": 2.0,
        "model_does_preprocess": False,
    },
    "cosmos_r1": {
        "sampling_fps": 4.0,
        "model_does_preprocess": False,
    },
    "cosmos_r2": {
        "sampling_fps": 4.0,
        "model_does_preprocess": True,
    },
}
_EXPECTED_CAPTIONS: dict[str, list[str | None]] = {
    "qwen": [
        (
            "The video begins with a close-up shot of a tablet placed on a wooden table, displaying a scene from a "
            "fantasy series. The tablet screen shows a character with long blonde hair, dressed in a blue outfit, "
            "standing amidst a crowd of people. The background features a sky with clouds and some flags, "
            "suggesting an outdoor setting, possibly a marketplace or a gathering place.\n"
            "\n"
            "As the camera pans to the right, it reveals a hand reaching out to interact with the tablet. The hand "
            "taps the screen, causing the scene to transition. The next shot is a close-up of a dragon breathing fire, "
            'which is a dramatic and iconic moment often associated with fantasy series like "Game of Thrones."\n'
            "\n"
            "The scene then shifts to a television screen, where the same character from the tablet is now shown in a "
            'different context. The character is depicted in a fiery environment, with the words "FOR BIGGER BLAZES" '
            "prominently displayed in bold blue letters across the screen. This text suggests that the content being "
            "viewed is enhanced for larger screens, likely referring to the high-definition quality of the TV "
            "display.\n"
            "\n"
            "The video continues with a shot of a person's hand plugging in a Chromecast device into an HDMI port on a "
            "television. This action indicates the process of streaming content from a device (like a smartphone or "
            "tablet) to a larger screen, such as a TV. The Chromecast device is a small, rectangular device designed "
            "to connect digital media devices to a television via HDMI.\n"
            "\n"
            'Following this, the video transitions to a white screen with the HBO GO logo and the text "right on your '
            'TV." This message emphasizes the availability of HBO GO content on television screens, highlighting the '
            "convenience of watching premium content on a larger display.\n"
            "\n"
            'The final frames show a white screen with the Google Chrome logo and the text "For $35. For everyone." '
            "This suggests a promotional offer related to the Google Chromecast service, indicating that it can be "
            "purchased for $35 and is accessible to all users. The video concludes with the Google logo and the "
            'website address "google.com/chromecast," directing viewers to learn more about the product.\n'
            "\n"
            "Overall, the video effectively showcases the integration of digital content with traditional television "
            "viewing, emphasizing the benefits of using a Chromecast device to enhance the viewing experience."
        ),
    ],
    "cosmos_r1": [
        (
            "### Detailed Description and Analysis of the Video\n"
            "\n"
            "#### Opening Scene\n"
            "- **Setting**: A wooden table with a tablet displaying a woman in a blue outfit (likely from a fantasy "
            "series, hinted by the HBO GO watermark). A green coffee mug is visible nearby, suggesting a relaxed, "
            "everyday environment.\n"
            "\n"
            "#### Tablet Interaction\n"
            "- **Action**: A finger taps the tablet screen twice, triggering a transition. This gesture implies user "
            "engagement with digital content.\n"
            "- **Result**: The tablet exits the frame, replaced by a larger TV screen showing a high-intensity scene "
            "of a dragon breathing fire, scaling up the viewing experience.\n"
            "\n"
            "#### Dragon Scene\n"
            "- **Symbolism**: The dragon (from *Game of Thrones*-inspired visuals) represents epic fantasy content, "
            "aligning with HBO GO's branding. The fiery backdrop emphasizes drama and excitement.\n"
            '- **Text Overlay**: "FOR BIGGER BLAZES" doubles as a pun ("blazes" vs. "fires"), teasing bigger, '
            "more thrilling content available on the platform.\n"
            "\n"
            "#### Return to Woman\n"
            "- **Focus**: The tablet reappears, zooming in on the woman's serious expression against a fiery "
            "background. Her role as a central character or spokesperson for the service is highlighted.\n"
            "\n"
            "#### Chromecast Setup\n"
            "- **Instructional Segment**: A hand plugs a Chromecast device into a TV port, demonstrating how users "
            "can stream content to their TVs. Close-ups of the port and hand movements emphasize simplicity and "
            "accessibility.\n"
            '- **Text Overlays**: Clear subscription details ("Everything you love, now on your TV." "$35/mo.") '
            "and a note about required broadband appear, clarifying costs and technical requirements.\n"
            "\n"
            "#### Branding and Conclusion\n"
            '- **White Screens**: HBO GO and Google logos reappear, followed by a tagline ("Everything you love, now '
            'on your TV.") and a promotional price.\n'
            '- **Final Shot**: The Chromecast logo and website URL ("google.com/chromecast") encourage viewer '
            "action, ending with a clean, minimalist design.\n"
            "\n"
            "#### Narrative and Symbolic Elements\n"
            "1. **Transition from Casual to Immersive**: The shift from tablet to TV underscores HBO GO's value "
            "proposition-translating personalized content into a larger, shared viewing experience.\n"
            '2. **Symbolic Dragons and Excitement**: The dragon scene visually reinforces the idea of "big, bold '
            'entertainment," while the pun on "blazes" ties HBO GO to iconic fantasy tropes.\n'
            "3. **User-Centric Instruction**: The Chromecast demo emphasizes ease of use, appealing to tech-savvy and "
            "casual users alike.\n"
            "4. **Branding Consistency**: Repeated logos and taglines ensure brand recognition, while the final white "
            "screens maintain a polished, professional tone.\n"
            "\n"
            "**Overall Purpose**: The video promotes HBO GO as a platform for high-quality, immersive content, "
            "leveraging both emotional appeal (excitement, drama) and practical guidance (setup instructions, pricing "
            "clarity) to attract subscribers."
        ),
    ],
    "cosmos_r2": [
        (
            "The video showcases a seamless streaming experience using a Chromecast device. It starts with a tablet on "
            "a wooden surface, displaying a scene from a fantasy series. A hand taps the screen, transitioning to a "
            'fiery dragon scene on a larger TV, emphasizing immersive content. The text "FOR BIGGER BLAZES" appears, '
            "highlighting the scale of the visuals. The focus returns to the tablet as another hand connects it via "
            "HDMI, demonstrating easy casting. The video concludes with pricing, availability details, and a call to "
            'action directing viewers to "google.com/chromecast," reinforcing the service\'s accessibility and '
            "branding."
        ),
    ],
}


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
def sample_captioning_task(sample_clip_data: bytes) -> SplitPipeTask:
    """Fixture to create a sample captioning task."""
    clip = Clip(
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, "sample_clip.mp4#0.0-15.0"),
        source_video="sample_clip.mp4",
        span=(0.0, 15.0),
        encoded_data=sample_clip_data,
    )

    video = Video(
        input_video=pathlib.Path("sample_clip.mp4"),
        clips=[clip],
    )
    return SplitPipeTask(
        video=video,
    )


@pytest.mark.env("unified")
@pytest.mark.parametrize("model_variant", ["qwen", "cosmos_r1", "cosmos_r2"])
def test_vllm_caption_generation(
    sample_captioning_task: SplitPipeTask, sequential_runner: RunnerInterface, model_variant: str
) -> None:
    """Test the vLLM captioning result."""
    vllm_config = VllmConfig(
        model_variant=model_variant,
        sampling_config=VllmSamplingConfig(temperature=0.0, max_tokens=2048),
        **_VLLM_CONFIG_OVERRIDES.get(model_variant, {}),
    )
    window_config = WindowConfig(
        **_WINDOW_CONFIG_OVERRIDES.get(model_variant, {}),
    )
    stages = [
        VllmPrepStage(vllm_config=vllm_config, window_config=window_config),
        VllmCaptionStage(vllm_config=vllm_config),
    ]
    tasks = run_pipeline([sample_captioning_task], stages, runner=sequential_runner)

    # Validate that captions were generated
    assert tasks is not None
    assert len(tasks) > 0
    assert len(tasks[0].video.clips) == _NUM_CLIPS

    expected_captions = _EXPECTED_CAPTIONS[model_variant]

    for clip_idx, clip in enumerate(tasks[0].video.clips):
        assert len(clip.windows) > 0, f"Clip {clip_idx} should have at least one window"
        assert len(expected_captions) == len(clip.windows), (
            f"Clip {clip_idx} should have {len(expected_captions)} windows for model {model_variant}"
        )

        for window_idx, window in enumerate(clip.windows):
            assert model_variant in window.caption, (
                f"Clip {clip_idx} window {window_idx} should have {model_variant} caption"
            )

            generated_caption = window.caption[model_variant]
            assert generated_caption.strip(), (
                f"Clip {clip_idx} window {window_idx} should have non-empty {model_variant} caption"
            )

            expected_caption = expected_captions[window_idx]
            similarity = cosine_similarity(generated_caption, expected_caption)
            threshold = _THRESHOLDS[model_variant]
            assert similarity >= threshold, (
                f"Caption similarity {similarity:.3f} below threshold for clip {clip_idx} window {window_idx}: "
                f"[{generated_caption}] vs. [{expected_caption}]"
            )
