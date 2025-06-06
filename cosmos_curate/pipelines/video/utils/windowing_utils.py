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
"""Utilities which are used in multiple places in the pipeline and/or are unit-tested."""

import subprocess

import attrs
import torch

from cosmos_curate.core.utils import conda_utils
from cosmos_curate.core.utils.runtime.operation_utils import make_pipeline_named_temporary_file
from cosmos_curate.pipelines.video.utils.decoder_utils import get_frame_count

if conda_utils.is_running_in_env("vllm"):
    from cosmos_curate.pipelines.video.utils.vision_process import fetch_video


@attrs.define
class WindowFrameInfo:
    """Container for frame window information, storing start and end frame indices.

    This class represents a window of frames in a video, defined by its start and end frame positions.
    """

    start: int
    end: int


WINDOW_MIN_FRAMES = 4


def compute_windows(total_frames: int, window_size: int = 128, remainder_threshold: int = 64) -> list[WindowFrameInfo]:
    """Generate windows by splitting the video into segments of the specified size.

    Args:
        total_frames: total frames
        window_size: The size of each window in number of frames.
        remainder_threshold: The minimum number of frames required to create a new window from the remainder.

    Yields:
        Tuple of (start_frame, end_frame) representing each window.

    """
    if not total_frames or total_frames < WINDOW_MIN_FRAMES:
        return []
    if total_frames <= window_size:
        return [WindowFrameInfo(0, total_frames - 1)]
    # Calculate the number of full window_size windows
    num_full_windows = total_frames // window_size

    # Calculate the remainder frames after filling in window_size windows
    remainder = total_frames % window_size

    out: list[WindowFrameInfo] = []
    # Yield each full window
    for i in range(num_full_windows):
        start_frame = i * window_size
        end_frame = start_frame + window_size - 1
        out.append(WindowFrameInfo(start_frame, end_frame))

    # Handle the remainder
    if remainder >= remainder_threshold:
        out.append(WindowFrameInfo(total_frames - remainder, total_frames - 1))
    elif remainder > 0 and num_full_windows > 0:
        # Expand the last window with the remainder if it exists
        out[-1] = WindowFrameInfo(out[-1].start, total_frames - 1)
    return out


def split_video_into_windows(  # noqa: PLR0913
    mp4_bytes: bytes,
    window_size: int = 256,
    remainder_threshold: int = 128,
    sampling_fps: float = 2.0,
    *,
    model_does_preprocess: bool = False,
    preprocess_dtype: str = "uint8",
    flip_input: bool = False,
    num_frames_to_use: int = 0,
    return_bytes: bool = False,
    return_video_frames: bool = True,
    num_threads: int = 1,
) -> tuple[list[bytes], list[torch.Tensor | None], list[WindowFrameInfo]]:
    """Calculate windows and return video inputs for the Qwen language model from input clips.

    Processes video to determine the windows for a clip, decode in one shot and return processed frames
    for each window in a format suitable for consumption by the Qwen model.

    Args:
        mp4_bytes: input video in bytes
        fps: Frames per second of the input video.
        preprocess_dtype: Data type to use for preprocessing the video/image inputs.
        num_frames_to_use: Number of frames to extract from the video. If 0, uses all frames.
        flip_input: Whether to flip the input video/image horizontally.
        return_bytes: Whether to extract mp4 bytes for each window for use by PreviewStage
        model_does_preprocess: if the model does preprocessing
        num_threads: number of threads
        remainder_threshold: threshold for remainder
        return_video_frames: whether to return video frames
        sampling_fps: sampling fps
        window_size: window size

    Returns:
        Tuple containing:
            - "window_mp4_bytes": mp4 bytes corresponding to each window - only used when Preview stage is enabled
            - "window_frames": Decoded and per-window processed frames ready for use by Qwen model
            - "window info": start and end frame indices for each window in a clip

    """
    with make_pipeline_named_temporary_file(sub_dir="windowing") as input_file:
        input_file.write_bytes(mp4_bytes)
        total_frames = get_frame_count(mp4_bytes)
        windows = compute_windows(total_frames, window_size, remainder_threshold)
        video_frames: list[torch.Tensor | None] = []
        mp4_bytes_list: list[bytes] = []

        if not windows:
            return mp4_bytes_list, video_frames, windows

        if return_video_frames:
            video, frame_counts = fetch_video(
                str(input_file),
                sampling_fps=sampling_fps,
                window_range=windows,
                do_preprocess=not model_does_preprocess,
                preprocess_dtype=preprocess_dtype,
                num_frames_to_use=num_frames_to_use,
                flip_input=flip_input,
            )

            index = 0
            for count in frame_counts:
                video_frames.append(video[index : index + count - 1])
                index = count

        if return_bytes:
            if len(windows) == 1:
                return [mp4_bytes], video_frames, windows

            for window in windows:
                with make_pipeline_named_temporary_file(sub_dir="windowing") as tmp_file:
                    # Use ffmpeg to split the file on the frames.
                    command = [
                        "ffmpeg",
                        "-threads",
                        str(num_threads),
                        "-y",
                        "-i",
                        str(input_file),
                        "-loglevel",
                        "error",
                        "-vf",
                        f"select='between(n\\,{window.start}\\,{window.end})',setpts=PTS-STARTPTS",
                        "-threads",
                        str(num_threads),
                        "-f",
                        "mp4",
                        "-an",
                        str(tmp_file),
                    ]
                    subprocess.check_call(command)  # noqa: S603
                    mp4_bytes_list.append(tmp_file.read_bytes())
        return mp4_bytes_list, video_frames, windows
