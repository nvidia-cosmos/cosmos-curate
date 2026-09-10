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
"""MCAP topic constants, jsonschema definitions, and message payload builders.

The channel layout mirrors the Foxglove-compatible session recordings the
curated MCAP output is meant to interoperate with: video frames on
``/camera/image-raw`` (`foxglove.CompressedVideo`), captions and detection
payloads on ``/scene-annotation`` (`midcentury.SceneAnnotation`), audio on
``/camera/audio`` (`foxglove.RawAudio`), one-shot calibration/transform
messages, and clip embeddings on ``/clip/embedding``
(`midcentury.ClipEmbedding`, curator-specific).

All payloads are JSON (schema encoding ``jsonschema``, message encoding
``json``); binary fields (video bitstream, audio samples, embedding vectors)
are base64 strings per the Foxglove convention.
"""

import base64
import json
from collections.abc import Buffer
from typing import Any

import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND

TOPIC_IMAGE_RAW = "/camera/image-raw"
TOPIC_SCENE_ANNOTATION = "/scene-annotation"
TOPIC_AUDIO = "/camera/audio"
TOPIC_CAMERA_INFO = "/camera/camera-info"
TOPIC_TF_STATIC = "/tf-static"
TOPIC_CLIP_EMBEDDING = "/clip/embedding"

SESSION_METADATA_RECORD_NAME = "session-metadata"

JSONSCHEMA_ENCODING = "jsonschema"
JSON_MESSAGE_ENCODING = "json"

_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "time",
    "properties": {
        "sec": {"type": "integer", "minimum": 0},
        "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
    },
}

COMPRESSED_VIDEO_SCHEMA_NAME = "foxglove.CompressedVideo"
COMPRESSED_VIDEO_SCHEMA: dict[str, Any] = {
    "title": COMPRESSED_VIDEO_SCHEMA_NAME,
    "description": "A single frame of a compressed video bitstream",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of video frame"},
        "frame_id": {"type": "string", "description": "Frame of reference for the video."},
        "data": {"type": "string", "contentEncoding": "base64", "description": "Compressed video frame data."},
        "format": {"type": "string", "description": "Video format. Supported values: h264, h265, vp9, av1."},
    },
    "required": ["timestamp", "frame_id", "data", "format"],
}

RAW_AUDIO_SCHEMA_NAME = "foxglove.RawAudio"
RAW_AUDIO_SCHEMA: dict[str, Any] = {
    "title": RAW_AUDIO_SCHEMA_NAME,
    "description": "A single block of an audio bitstream",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of the start of the audio block"},
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Audio data. The samples in the data must be interleaved and little-endian",
        },
        "format": {"type": "string", "description": "Audio format. Only 'pcm-s16' is currently supported"},
        "sample_rate": {"type": "integer", "minimum": 0, "description": "Sample rate in Hz"},
        "number_of_channels": {"type": "integer", "minimum": 0, "description": "Number of channels in the audio block"},
    },
    "required": ["timestamp", "data", "format", "sample_rate", "number_of_channels"],
}

CAMERA_CALIBRATION_SCHEMA_NAME = "foxglove.CameraCalibration"
CAMERA_CALIBRATION_SCHEMA: dict[str, Any] = {
    "title": CAMERA_CALIBRATION_SCHEMA_NAME,
    "description": "Camera calibration parameters",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of calibration data"},
        "frame_id": {"type": "string"},
        "width": {"type": "integer", "minimum": 0, "description": "Image width"},
        "height": {"type": "integer", "minimum": 0, "description": "Image height"},
        "distortion_model": {"type": "string", "description": "Name of distortion model"},
        "D": {"type": "array", "items": {"type": "number"}, "description": "Distortion parameters"},
        "K": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 9,
            "maxItems": 9,
            "description": "Intrinsic camera matrix (3x3 row-major)",
        },
        "R": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 9,
            "maxItems": 9,
            "description": "Rectification matrix (3x3 row-major)",
        },
        "P": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 12,
            "maxItems": 12,
            "description": "Projection matrix (3x4 row-major)",
        },
    },
    "required": ["timestamp", "frame_id", "width", "height", "distortion_model", "D", "K", "R", "P"],
}

FRAME_TRANSFORMS_SCHEMA_NAME = "foxglove.FrameTransforms"
FRAME_TRANSFORMS_SCHEMA: dict[str, Any] = {
    "title": FRAME_TRANSFORMS_SCHEMA_NAME,
    "description": "An array of FrameTransform messages",
    "type": "object",
    "properties": {
        "transforms": {
            "type": "array",
            "items": {
                "title": "foxglove.FrameTransform",
                "type": "object",
                "properties": {
                    "timestamp": _TIME_SCHEMA,
                    "parent_frame_id": {"type": "string"},
                    "child_frame_id": {"type": "string"},
                    "translation": {
                        "title": "foxglove.Vector3",
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                        "required": ["x", "y", "z"],
                    },
                    "rotation": {
                        "title": "foxglove.Quaternion",
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "w": {"type": "number"},
                        },
                        "required": ["x", "y", "z", "w"],
                    },
                },
                "required": ["timestamp", "parent_frame_id", "child_frame_id", "translation", "rotation"],
            },
            "description": "Array of transforms",
        },
    },
    "required": ["transforms"],
}

SCENE_ANNOTATION_SCHEMA_NAME = "midcentury.SceneAnnotation"
SCENE_ANNOTATION_SCHEMA: dict[str, Any] = {
    "title": SCENE_ANNOTATION_SCHEMA_NAME,
    "type": "object",
    "properties": {
        "timestamp": _TIME_SCHEMA,
        "data": {"type": "string", "description": "JSON-encoded detection payload or scene description."},
    },
    "required": ["timestamp", "data"],
}

CLIP_EMBEDDING_SCHEMA_NAME = "midcentury.ClipEmbedding"
CLIP_EMBEDDING_SCHEMA: dict[str, Any] = {
    "title": CLIP_EMBEDDING_SCHEMA_NAME,
    "type": "object",
    "properties": {
        "timestamp": _TIME_SCHEMA,
        "model_name": {"type": "string", "description": "Embedding model/algorithm name."},
        "model_version": {"type": "string", "description": "Embedding model version."},
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Little-endian float32 embedding vector.",
        },
    },
    "required": ["timestamp", "model_name", "model_version", "data"],
}


# Every topic carries exactly one schema; channel registration is driven by this map.
TOPIC_SCHEMAS: dict[str, dict[str, Any]] = {
    TOPIC_IMAGE_RAW: COMPRESSED_VIDEO_SCHEMA,
    TOPIC_SCENE_ANNOTATION: SCENE_ANNOTATION_SCHEMA,
    TOPIC_AUDIO: RAW_AUDIO_SCHEMA,
    TOPIC_CAMERA_INFO: CAMERA_CALIBRATION_SCHEMA,
    TOPIC_TF_STATIC: FRAME_TRANSFORMS_SCHEMA,
    TOPIC_CLIP_EMBEDDING: CLIP_EMBEDDING_SCHEMA,
}


def ns_to_time(ns: int) -> dict[str, int]:
    """Convert integer nanoseconds to a Foxglove ``time`` object."""
    return {"sec": ns // NS_PER_SECOND, "nsec": ns % NS_PER_SECOND}


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def compressed_video_message(ns: int, frame_id: str, data: Buffer, video_format: str) -> bytes:
    """Build a ``foxglove.CompressedVideo`` JSON payload."""
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "frame_id": frame_id,
            "data": base64.b64encode(data).decode("ascii"),
            "format": video_format,
        }
    )


def raw_audio_message(ns: int, data: Buffer, sample_rate: int, number_of_channels: int) -> bytes:
    """Build a ``foxglove.RawAudio`` JSON payload (interleaved little-endian pcm-s16)."""
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "data": base64.b64encode(data).decode("ascii"),
            "format": "pcm-s16",
            "sample_rate": sample_rate,
            "number_of_channels": number_of_channels,
        }
    )


def scene_annotation_message(ns: int, text: str) -> bytes:
    """Build a ``midcentury.SceneAnnotation`` JSON payload.

    ``text`` is either a plain-text scene description (caption) or a
    JSON-encoded detection payload; both share the topic.
    """
    return _encode({"timestamp": ns_to_time(ns), "data": text})


def clip_embedding_message(ns: int, model_name: str, model_version: str, vector: npt.NDArray[np.float32]) -> bytes:
    """Build a ``midcentury.ClipEmbedding`` JSON payload."""
    raw = vector.reshape(-1).astype("<f4", copy=False).tobytes()
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "model_name": model_name,
            "model_version": model_version,
            "data": base64.b64encode(raw).decode("ascii"),
        }
    )


def camera_calibration_message(ns: int, frame_id: str, width: int, height: int) -> bytes:
    """Build a ``foxglove.CameraCalibration`` JSON payload.

    No real calibration source exists in the pipeline; K/R/P are identity-style
    placeholders with the principal point at the image center so the message is
    well-formed for Foxglove without claiming a measured focal length.
    """
    cx = width / 2
    cy = height / 2
    k = [1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [1.0, 0.0, cx, 0.0, 0.0, 1.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "frame_id": frame_id,
            "width": width,
            "height": height,
            "distortion_model": "plumb_bob",
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "K": k,
            "R": r,
            "P": p,
        }
    )


def frame_transforms_message(ns: int, child_frame_id: str) -> bytes:
    """Build a ``foxglove.FrameTransforms`` JSON payload with one identity map->camera transform."""
    return _encode(
        {
            "transforms": [
                {
                    "timestamp": ns_to_time(ns),
                    "parent_frame_id": "map",
                    "child_frame_id": child_frame_id,
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            ]
        }
    )
