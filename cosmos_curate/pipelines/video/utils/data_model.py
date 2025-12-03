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

"""Data Model util."""

from __future__ import annotations

import pathlib
import sys
from collections import deque
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import PipelineTask
from cosmos_curate.core.utils.storage import storage_client
from cosmos_curate.pipelines.video.utils.decoder_utils import extract_video_metadata

try:
    import torch

    TensorType = getattr(torch, "Tensor", None)
except ImportError:
    TensorType = None

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable
    from uuid import UUID

    import numpy.typing as npt

    import cosmos_curate.pipelines.video.filtering.motion.motion_vector_backend as motion
    from cosmos_curate.core.utils.infra.performance_utils import StagePerfStats


def _get_object_size(obj: object) -> int:
    """Get the size of a single object.

    Lists, tuples, sets, and frozensets return 0 since we only count contents,
    not the container itself. The calling function is expected to extract the
    contents of the container and call this function on each item.

    Arguments:
        obj: The object to get the size of.

    Returns:
        The size of the object in bytes.

    """
    if obj is None:
        return 0
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.nbytes
    if TensorType is not None and isinstance(obj, TensorType):
        if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
            return int(obj.element_size() * obj.nelement())
        return sys.getsizeof(obj)  # Fallback for unexpected tensor types
    # For containers, only count contents, not the container itself
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        return 0
    return sys.getsizeof(obj)


def _add_children_to_queue(obj: object, q: deque[object], visited: set[int]) -> None:
    """Add child objects to the queue for processing."""
    children: Iterable[object] = iter([])
    # Skip transient performance tracking fields that shouldn't count toward data size
    _skip_fields = {"stage_perf"}

    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children = obj
    elif attrs.has(obj.__class__):
        children = (getattr(obj, field.name) for field in attrs.fields(obj.__class__) if field.name not in _skip_fields)

    for child in children:
        if id(child) not in visited:
            q.append(child)


def get_major_size(obj: object) -> int:
    """Get the memory size of an attrs instance in bytes.

    This function is used to get the memory size of an attrs instance in bytes.
    It can handle circular references and nested structures. It does not count
    the size of the containers themselves, only the contents.

    Args:
        obj: The object to get the memory size of.

    Returns:
        The memory size of the object in bytes.

    """
    size = 0
    visited: set[int] = set()
    q: deque[object] = deque([obj])

    while q:
        current_obj = q.popleft()
        if id(current_obj) in visited:
            continue
        visited.add(id(current_obj))

        size += _get_object_size(current_obj)
        _add_children_to_queue(current_obj, q, visited)

    return size


@attrs.define
class Window:
    """Container for captioning window."""

    # Start frame number of this window
    start_frame: int
    # End frame number of this window
    end_frame: int
    # MP4 bytes for this window
    mp4_bytes: bytes | None = None
    # Model input for this window: model_variant -> llm_input
    model_input: dict[str, dict[str, Any]] = attrs.Factory(dict)
    # `caption: {model_name: caption}`
    caption: dict[str, str] = attrs.Factory(dict)
    enhanced_caption: dict[str, str] = attrs.Factory(dict)
    # t5_xxl embeddings for this window
    t5_xxl_embedding: dict[str, npt.NDArray[np.int32]] = attrs.Factory(dict)
    # webp preview
    webp_bytes: bytes | None = None
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Calculate total memory size of the window.

        Returns:
            Total size in bytes.

        """
        return get_major_size(self)


@attrs.define
class Clip:
    """Container for video clip data including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    uuid: UUID
    source_video: str
    span: tuple[float, float]
    encoded_data: bytes | None = None
    extracted_frames: dict[str, npt.NDArray[np.uint8]] = attrs.Factory(dict)
    # motion
    decoded_motion_data: motion.DecodedData | None = None
    motion_score_global_mean: float | None = None
    motion_score_per_patch_min_256: float | None = None
    # aesthetic
    aesthetic_score: float | None = None
    # embedding
    cosmos_embed1_frames: npt.NDArray[np.float32] | None = None
    cosmos_embed1_embedding: npt.NDArray[np.float32] | None = None
    intern_video_2_frames: npt.NDArray[np.float32] | None = None
    intern_video_2_embedding: npt.NDArray[np.float32] | None = None
    # captioning
    windows: list[Window] = attrs.Factory(list)
    filter_windows: list[Window] = attrs.Factory(list)
    # for testing
    cosmos_embed1_text_match: tuple[str, float] | None = None
    intern_video_2_text_match: tuple[str, float] | None = None
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)

    def get_all_captions(self) -> list[str]:
        """Get all captions from the clip's windows.

        Returns:
            A list of all captions from the clip's windows.

        """
        captions: list[str] = []
        for window in self.windows:
            captions.extend(window.caption.values())
        return captions

    def extract_metadata(self) -> dict[str, Any] | None:
        """Extract metadata from the clip's encoded_data.

        Returns:
            A dictionary containing the extracted metadata (width, height, framerate,
            num_frames, video_codec, num_bytes) if encoded_data exists, None otherwise.

        Raises:
            Exception: Any exception from extract_video_metadata is propagated.

        """
        if self.encoded_data is None:
            return None

        metadata = extract_video_metadata(self.encoded_data)

        return {
            "width": metadata.width,
            "height": metadata.height,
            "framerate": metadata.fps,
            "num_frames": metadata.num_frames,
            "video_codec": metadata.video_codec,
            "num_bytes": len(self.encoded_data),
        }

    @property
    def duration(self) -> float:
        """Calculate the duration of the clip.

        Returns:
            Duration of the clip in seconds.

        """
        return self.span[1] - self.span[0]

    def get_major_size(self) -> int:
        """Calculate total memory size of the clip.

        Returns:
            Total size in bytes.

        """
        total_size = len(self.uuid.bytes)
        if self.encoded_data:
            total_size += len(self.encoded_data)
        if self.extracted_frames:
            for x in self.extracted_frames.values():
                total_size += x.nbytes
        if self.decoded_motion_data is not None:
            total_size += self.decoded_motion_data.get_major_size()
        if self.intern_video_2_frames is not None:
            total_size += self.intern_video_2_frames.nbytes
        if self.intern_video_2_embedding is not None:
            total_size += self.intern_video_2_embedding.nbytes
        for window in self.windows:
            total_size += window.get_major_size()
        return total_size


@attrs.define
class ClipStats:
    """Statistics for video clips including filtering, transcoding, and captioning results.

    This class accumulates statistics about the number of clips processed through
    different stages of the video processing pipeline, including motion filtering,
    aesthetic filtering, and captioning.
    """

    num_filtered_by_motion: int = 0
    num_filtered_by_aesthetic: int = 0
    num_passed: int = 0
    num_transcoded: int = 0
    num_with_embeddings: int = 0
    num_with_caption: int = 0
    num_with_webp: int = 0
    total_clip_duration: float = 0.0
    max_clip_duration: float = 0.0

    def combine(self, other: ClipStats) -> None:
        """Combine two ClipStats objects.

        Args:
            other: ClipStats object to combine with.

        """
        self.num_filtered_by_motion += other.num_filtered_by_motion
        self.num_filtered_by_aesthetic += other.num_filtered_by_aesthetic
        self.num_passed += other.num_passed
        self.num_transcoded += other.num_transcoded
        self.num_with_embeddings += other.num_with_embeddings
        self.num_with_caption += other.num_with_caption
        self.num_with_webp += other.num_with_webp
        self.total_clip_duration += other.total_clip_duration
        self.max_clip_duration = max(self.max_clip_duration, other.max_clip_duration)


@attrs.define
class VideoMetadata:
    """Metadata for video content including dimensions, timing, and codec information.

    This class stores essential video properties such as resolution, frame rate,
    duration, and encoding details.
    """

    size: int | None = None
    height: int | None = None
    width: int | None = None
    framerate: float | None = None
    num_frames: int | None = None
    duration: float | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    audio_codec: str | None = None
    bit_rate_k: int | None = None
    format_name: str | None = None


@attrs.define
class Video:
    """Container for video content including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    input_video: storage_client.StoragePrefix | pathlib.Path
    # encoded video bytes
    encoded_data: bytes | None = None
    # video metadata
    metadata: VideoMetadata = attrs.Factory(VideoMetadata)
    # decoded video frames
    frame_array: npt.NDArray[np.uint8] | None = None
    # clips
    clips: list[Clip] = attrs.Factory(list)
    filtered_clips: list[Clip] = attrs.Factory(list)
    # for cracking
    num_total_clips: int = 0
    num_clip_chunks: int = 0
    clip_chunk_index: int = 0
    # for last writer stage
    clip_stats: ClipStats = attrs.Factory(ClipStats)
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)

    def populate_metadata(self) -> None:
        """Extract and assign video metadata from encoded_data.

        This method extracts metadata from the video data in encoded_data and
        assigns it to self.metadata.

        Raises:
            ValueError: If encoded_data is None.
            Exception: Any exception from extract_video_metadata is propagated.

        """
        if self.encoded_data is None:
            error_msg = "No video data available: encoded_data is None"
            raise ValueError(error_msg)

        # Extract metadata using the existing function
        extracted_metadata = extract_video_metadata(self.encoded_data)

        # Set the size from encoded_data
        self.metadata.size = len(self.encoded_data)

        # Map the extracted metadata to our metadata object
        self.metadata.height = extracted_metadata.height
        self.metadata.width = extracted_metadata.width
        self.metadata.framerate = extracted_metadata.fps
        self.metadata.num_frames = extracted_metadata.num_frames
        self.metadata.duration = extracted_metadata.video_duration
        self.metadata.video_codec = extracted_metadata.video_codec
        self.metadata.pixel_format = extracted_metadata.pixel_format
        self.metadata.audio_codec = extracted_metadata.audio_codec
        self.metadata.bit_rate_k = extracted_metadata.bit_rate_k
        self.metadata.format_name = extracted_metadata.format_name

    @property
    def fraction(self) -> float:
        """Calculate the fraction of processed clips.

        Returns:
            Fraction of processed clips.

        """
        if self.num_total_clips == 0:
            return 1.0
        return (len(self.clips) + len(self.filtered_clips)) / self.num_total_clips

    @property
    def weight(self) -> float:
        """Calculate the weight of the video.

        Returns:
            Weight of the video.

        """
        if self.metadata.size is None:
            return 0
        # normalize to 5 min
        assert self.metadata.duration is not None
        weight = self.metadata.duration / 300
        # when clips are further chunked
        return weight * self.fraction

    @property
    def input_path(self) -> str:
        """Get the input path of the video.

        Returns:
            Input path of the video.

        """
        if isinstance(self.input_video, storage_client.StoragePrefix):
            return self.input_video.path
        return self.input_video.as_posix()

    def has_metadata(self) -> bool:
        """Check if all metadata fields are present.

        Returns:
            True if all metadata fields are present, False otherwise.

        """
        return all(
            [
                self.metadata.height,
                self.metadata.width,
                self.metadata.duration,
                self.metadata.framerate,
                self.metadata.num_frames,
                self.metadata.video_codec,
            ],
        )

    def nvdec_support(self) -> bool:
        """Heuristic function to switch between nvdec or CPU-fallback on V100/A100/H100.

        For detailed info on Video Codec SDK hardware support, see:
        https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
        """
        if self.metadata.video_codec is None or self.metadata.pixel_format is None:
            return False
        if self.metadata.video_codec == "h264" and (
            "nv16" in self.metadata.pixel_format or "420p" in self.metadata.pixel_format
        ):
            # h264 decoding supported only 8-bit surface format
            return True
        if self.metadata.video_codec == "hevc" and (
            "420p" in self.metadata.pixel_format or "444p" in self.metadata.pixel_format
        ):
            # h265/hevc decoding supports yuv420p and yuv444p pixel formats
            return True
        if self.metadata.video_codec not in ("mjpeg", "av1", "vp9", "vp8"):
            # - mjpeg is not supported
            # - av1 is not supported on A100/H100
            # - VP8/9 are not exposed in PyNvVideoCodec (but supported by VideoCodecSDK)
            logger.warning(f"Encountered new video codec [{self.metadata.video_codec}], assuming no NVDEC support.")
        return False

    def is_10_bit_color(self) -> bool | None:
        """Heuristic function to determine if the input video has 10-bit color."""
        if self.metadata.pixel_format is None:
            return None
        return "10le" in self.metadata.pixel_format or "10be" in self.metadata.pixel_format

    def get_major_size(self) -> int:
        """Calculate total memory size of the video.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        total_size += len(self.encoded_data) if self.encoded_data else 0
        total_size += sys.getsizeof(self.frame_array)
        for clip in self.clips:
            total_size += clip.get_major_size()
        total_size += self.frame_array.nbytes if self.frame_array is not None else 0
        return total_size


@attrs.define
class SplitPipeTask(PipelineTask):
    """The data we want to pass between stages of split-pipeline."""

    video: Video
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    @property
    def fraction(self) -> float:
        """Calculate fraction of processed video in the task.

        Returns:
            Fraction of processed video.

        """
        return self.video.fraction

    @property
    def weight(self) -> float:
        """Calculate weight of video in the task.

        Returns:
            Weight of video.

        """
        return self.video.weight

    def get_major_size(self) -> int:
        """Calculate memory size of video in the task.

        Returns:
            Total size in bytes.

        """
        return self.video.get_major_size()


@attrs.define
class ClipSample:
    """Container for video clip sample data including metadata, frames, and embeddings.

    This class stores information about a video clip sample, including its UUID, dimensions,
    frame count, byte size, and metadata.
    """

    uuid: str
    width: int
    height: int
    num_frames: int
    num_bytes: int
    clip_location: storage_client.StoragePrefix | pathlib.Path
    clip_metadata: dict[str, Any] = attrs.Factory(dict)
    encoded_data: bytes | None = None
    t5_xxl_embeddings: list[npt.NDArray[np.int32]] = attrs.Factory(list)

    def get_major_size(self) -> int:
        """Calculate total memory size of the clip sample.

        Returns:
            Total size in bytes.

        """
        total_size = sys.getsizeof(self.clip_metadata)
        total_size += len(self.encoded_data) if self.encoded_data else 0
        total_size += sum(x.nbytes for x in self.t5_xxl_embeddings)
        return total_size


@attrs.define
class ShardPipeTask(PipelineTask):
    """The data we want to pass between stages of sharding-pipeline."""

    bin_path: str
    part_num: int
    samples: list[ClipSample]
    output_tar_video: storage_client.StoragePrefix | pathlib.Path
    output_tar_metas: storage_client.StoragePrefix | pathlib.Path
    output_tar_t5_xxl: storage_client.StoragePrefix | pathlib.Path
    key_count: int
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Calculate total memory size of all samples in the task.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        for sample in self.samples:
            total_size += sample.get_major_size()
        return total_size


def get_video_from_task(task: PipelineTask) -> Video:
    """Extract the Video object attached to a pipeline task.

    Args:
        task: Task expected to expose a `video` attribute.

    Returns:
        The associated `Video` instance.

    Raises:
        TypeError: If the task is missing a `video` attribute or it has an unexpected type.

    """
    video = getattr(task, "video", None)
    if not isinstance(video, Video):
        msg = f"task.video type={type(video)}, expected `Video`"
        raise TypeError(msg)
    return video


@attrs.define
class VllmSamplingConfig:
    """Configuration for vLLM sampling parameters.

    Unless otherwise specified, the vLLM default values are used.
    The defaults differ to maintain compatibility with the previous configuration.

    Args:
        presence_penalty: Penalize tokens based on their presence in the generated text.
        frequency_penalty: Penalize tokens based on their frequency in the generated text.
        repetition_penalty: Penalize tokens that have been generated previously.
        temperature: Controls randomness in sampling (higher = more random).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter (0 = disabled).
        min_p: Minimum probability threshold for sampling.
        max_tokens: Maximum number of tokens to generate (None = no limit).

    """

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.05  # vLLM default is 1.0
    temperature: float = 0.1  # vLLM default is 1.0
    top_p: float = 0.001  # vLLM default is 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int | None = 512  # vLLM default is None


@attrs.define
class VllmConfig:
    """Configuration for a vLLM model.

    Args:
        model_variant: Name of the model variant to use.
        prompt_variant: Type of prompt to use.
        prompt_text: Custom prompt text if provided.
        batch_size: Number of samples to process in parallel.
        fp8: Whether to enable FP8 precision.
        preprocess: Whether model handles preprocessing.
        disable_mmcache: Whether to disable model cache.
        num_gpus_per_worker: Number of GPUs to allocate per worker.
        batch_size: Number of samples to process in parallel.
        stage2_caption: Whether to enable stage 2 captioning.
        stage2_prompt_text: Custom prompt text for stage 2 captioning.
        max_retries: Number of times to retry captioning failures.
        copy_weights_to: Optional custom directory to copy model weights to before loading.
            If set, model weights will be copied from the default cache location to this
            directory before the model is loaded. This is useful for copying weights to
            faster storage (e.g., local SSD) on compute nodes.
        sampling_config: Configuration for sampling parameters.

    """

    model_variant: str
    prompt_variant: str = "default"
    prompt_text: str | None = None
    fp8: bool = True
    preprocess: bool = False
    disable_mmcache: bool = False
    num_cpus_for_prepare: float = 2.0
    num_gpus: int = 1
    batch_size: int = 4
    stage2_caption: bool = False
    stage2_prompt_text: str | None = None
    max_retries: int = 3
    copy_weights_to: pathlib.Path | None = None
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)


@attrs.define
class WindowConfig:
    """Configuration for splitting a video into windows.

    Args:
        sampling_fps: Frames per second for sampling.
        window_size: Size of each window in frames.
        remainder_threshold: Minimum frames required for a remainder window.
        preprocess_dtype: Data type for preprocessing.
        model_does_preprocess: Whether model handles preprocessing.
        use_input_bit_rate: Whether to use the input video's bit rate for processing.

    """

    window_size: int = 256
    sampling_fps: float = 2.0
    remainder_threshold: int = 128
    model_does_preprocess: bool = False
    preprocess_dtype: str = "float32"
    use_input_bit_rate: bool = False


@attrs.define
class VllmCaptionRequest:
    """A vLLM captioning task for a single clip window.

    Args:
        request_id: The request ID.
        inputs: The inputs for the vLLM model.
        caption: The caption generated by the vLLM model.
           * If caption is None, this indicates that the caption should be generated
             by the vLLM model using the inputs
        stage2_prompt: A second-stage prompt used to refine the caption
           * If stage2_prompt is set, and caption is not None, this indicates
             that the caption should be refined using the stage2_prompt.
             A new request should be created with new inputs and with stage2_prompt set to None.
           * If stage2_prompt is not set, and caption is not None, this indicates
             that the caption should be used as is.

    """

    request_id: str
    inputs: dict[str, Any]
    caption: str | None = None
    stage2_prompt: str | None = None
