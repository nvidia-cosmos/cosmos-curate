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
"""Tests for the MCAP jsonschema constants and payload builders."""

import base64
import json

import numpy as np
import numpy.testing as npt

from cosmos_curator.pipelines.video.read_write import mcap_schemas


def test_topic_schemas_cover_all_topics() -> None:
    """Every topic constant maps to a titled schema."""
    assert set(mcap_schemas.TOPIC_SCHEMAS) == {
        mcap_schemas.TOPIC_IMAGE_RAW,
        mcap_schemas.TOPIC_SCENE_ANNOTATION,
        mcap_schemas.TOPIC_AUDIO,
        mcap_schemas.TOPIC_CAMERA_INFO,
        mcap_schemas.TOPIC_TF_STATIC,
        mcap_schemas.TOPIC_CLIP_EMBEDDING,
    }
    for schema in mcap_schemas.TOPIC_SCHEMAS.values():
        assert schema["title"]


def test_payload_builders_round_trip() -> None:
    """Payload builders emit valid JSON matching the schema field types."""
    ns = 1_500_000_000
    video_payload = json.loads(mcap_schemas.compressed_video_message(ns, "camera", b"\x00\x01", "h264"))
    assert video_payload["timestamp"] == {"sec": 1, "nsec": 500_000_000}
    assert base64.b64decode(video_payload["data"]) == b"\x00\x01"

    audio_payload = json.loads(mcap_schemas.raw_audio_message(0, b"\x00\x00", 48000, 1))
    assert audio_payload["format"] == "pcm-s16"

    calibration_payload = json.loads(mcap_schemas.camera_calibration_message(0, "camera", 1920, 1080))
    assert len(calibration_payload["K"]) == 9
    assert len(calibration_payload["P"]) == 12
    assert calibration_payload["width"] == 1920

    transforms_payload = json.loads(mcap_schemas.frame_transforms_message(0, "camera"))
    assert transforms_payload["transforms"][0]["child_frame_id"] == "camera"

    vector = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    embedding_payload = json.loads(mcap_schemas.clip_embedding_message(0, "m", "v", vector))
    decoded = np.frombuffer(base64.b64decode(embedding_payload["data"]), dtype="<f4")
    npt.assert_allclose(decoded, [1.0, 2.0, 3.0, 4.0])
