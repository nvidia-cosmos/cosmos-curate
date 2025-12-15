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
"""Tests for the clip metadata writer stage."""

import base64
import json
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from cosmos_curate.core.utils.storage import storage_client, storage_utils
from cosmos_curate.pipelines.video.read_write.metadata_writer_stage import (
    ClipWriterStage,
    _archive_processed_sidecars,
    consolidate_lance_fragments,
)
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, VideoMetadata, Window


def _create_stage(output_dir: Path, input_dir: Path, **overrides: object) -> ClipWriterStage:
    """Instantiate the stage with default options and optional overrides."""
    params = {
        "output_path": str(output_dir),
        "input_path": str(input_dir),
        "output_s3_profile_name": "default",
        "upload_clips": True,
        "upload_clip_info_in_chunks": False,
        "upload_clip_info_in_lance": False,
        "upload_cds_parquet": True,
        "dry_run": False,
        "generate_embeddings": True,
        "embedding_algorithm": "internvideo2",
        "embedding_model_version": "v1",
        "generate_previews": False,
        "caption_models": ["qwen"],
        "enhanced_caption_models": ["qwen_plus"],
        "generate_cosmos_predict_dataset": "disable",
        "verbose": False,
        "log_stats": False,
    }
    params.update(overrides)
    stage = ClipWriterStage(**params)
    stage.stage_setup()
    return stage


class _FailingDeleteClient(storage_client.StorageClient):
    """Storage client stub that raises on delete to exercise cleanup errors."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})

    def object_exists(self, dest: storage_client.StoragePrefix) -> bool:
        return str(dest) in self.objects

    def upload_bytes(self, dest: storage_client.StoragePrefix, data: bytes) -> None:
        self.objects[str(dest)] = data

    def upload_bytes_uri(self, uri: str, data: bytes, _chunk_size_bytes: int = 100) -> None:
        self.objects[uri] = data

    def download_object_as_bytes(self, uri: storage_client.StoragePrefix, _chunk_size_bytes: int = 10) -> bytes:
        try:
            return self.objects[str(uri)]
        except KeyError as exc:
            raise FileNotFoundError(str(uri)) from exc

    def download_objects_as_bytes(self, uris: list[storage_client.StoragePrefix]) -> list[bytes]:
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(
        self, uri: storage_client.StoragePrefix, limit: int = 0
    ) -> list[storage_client.StoragePrefix]:
        _ = limit
        prefix = str(uri).rstrip("/") + "/"
        return [storage_utils.path_to_prefix(path) for path in sorted(self.objects) if path.startswith(prefix)]

    def list_recursive(
        self, prefix: storage_client.StoragePrefix, limit: int = 0
    ) -> list[dict[str, object]]:  # pragma: no cover - unused
        raise NotImplementedError

    def upload_file(
        self,
        local_path: str,
        remote_path: storage_client.StoragePrefix,
        _chunk_size: int = 100,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def sync_remote_to_local(
        self,
        remote_prefix: storage_client.StoragePrefix,
        local_dir: Path,
        *,
        delete: bool = False,
        _chunk_size_bytes: int = 10,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def make_background_uploader(
        self,
        _chunk_size_bytes: int = 100,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def delete_object(self, dest: storage_client.StoragePrefix) -> None:
        error_msg = f"delete failed for {dest}"
        raise RuntimeError(error_msg)


@pytest.fixture(autouse=True)
def fake_extract_video_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid invoking ffprobe in tests."""

    def _fake_extract_video_metadata(_: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            width=1920,
            height=1080,
            fps=30.0,
            num_frames=60,
            video_codec="h264",
        )

    monkeypatch.setattr(
        "cosmos_curate.pipelines.video.utils.data_model.extract_video_metadata",
        _fake_extract_video_metadata,
    )


def _build_video(
    video_path: Path,
    clip: Clip,
    *,
    clip_chunk_index: int = 0,
) -> Video:
    """Assemble Video metadata wrapper used by the stage."""
    metadata = VideoMetadata(
        height=1080,
        width=1920,
        framerate=30.0,
        num_frames=60,
        duration=2.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
    )
    return Video(
        input_video=video_path,
        metadata=metadata,
        clips=[clip],
        filtered_clips=[],
        num_total_clips=1,
        num_clip_chunks=1,
        clip_chunk_index=clip_chunk_index,
    )


def _stage_with_main_clip(tmp_path: Path) -> tuple[ClipWriterStage, SplitPipeTask, Clip, Window, Path]:
    """Create a standard stage/task/clip tuple for integration testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")

    stage = _create_stage(output_dir, input_dir)

    main_window = Window(
        start_frame=0,
        end_frame=30,
        caption={"qwen": "main caption"},
        enhanced_caption={"qwen_plus": "enhanced view"},
        webp_bytes=b"webp-content",
        t5_xxl_embedding={"default": np.array([1, 2, 3], dtype=np.int32)},
    )
    filtered_window = Window(
        start_frame=0,
        end_frame=30,
        caption={"qwen_rejection_reasons": "too blurry"},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=b"clip-bytes",
        windows=[main_window],
        filter_windows=[filtered_window],
    )
    clip.intern_video_2_embedding = np.array([0.1, 0.2], dtype=np.float32)

    video = _build_video(video_path, clip)
    task = SplitPipeTask(video=video)
    return stage, task, clip, main_window, output_dir


def _assert_payloads_cleared(clip: Clip, window: Window) -> None:
    """Ensure transient buffers are released after processing."""
    assert clip.encoded_data is None
    assert clip.intern_video_2_embedding is None
    assert window.webp_bytes is None
    assert window.caption == {}
    assert window.enhanced_caption == {}


def _read_json(path: Path) -> dict[str, object]:
    """Load JSON data from disk."""
    return json.loads(path.read_text())


def _assert_embeddings_written(output_dir: Path, clip: Clip, video_uuid: uuid.UUID) -> None:
    """Validate clip-level and chunk-level embedding outputs."""
    embedding_pickle_path = output_dir / "iv2_embd" / f"{clip.uuid}.pickle"
    with embedding_pickle_path.open("rb") as infile:
        stored_embedding = pickle.load(infile)  # noqa: S301 - reading data produced within the test
    npt.assert_array_equal(stored_embedding, np.array([0.1, 0.2], dtype=np.float32))

    embedding_parquet_path = output_dir / "iv2_embd_parquet" / f"{video_uuid}_0.parquet"
    embedding_df = pd.read_parquet(embedding_parquet_path)
    assert len(embedding_df) == 1
    assert embedding_df.iloc[0]["id"] == str(clip.uuid)
    npt.assert_allclose(np.array(embedding_df.iloc[0]["embedding"]), np.array([0.1, 0.2], dtype=np.float32))


def test_process_data_writes_expected_local_outputs(tmp_path: Path) -> None:
    """End-to-end validation for per-clip assets and metadata aggregation."""
    stage, task, clip, main_window, output_dir = _stage_with_main_clip(tmp_path)
    video = task.video

    result = stage.process_data([task])
    assert result == [task]

    _assert_payloads_cleared(clip, main_window)

    clip_mp4_path = output_dir / "clips" / f"{clip.uuid}.mp4"
    assert clip_mp4_path.read_bytes() == b"clip-bytes"

    preview_path = output_dir / "previews" / str(clip.uuid) / "0_30.webp"
    assert preview_path.read_bytes() == b"webp-content"

    clip_meta_path = output_dir / "metas" / "v0" / f"{clip.uuid}.json"
    clip_metadata = _read_json(clip_meta_path)
    assert clip_metadata["span_uuid"] == str(clip.uuid)
    assert clip_metadata["clip_location"].endswith(f"clips/{clip.uuid}.mp4")
    assert clip_metadata["filtered_windows"] == [
        {"start_frame": 0, "end_frame": 30, "qwen_rejection_reasons": "too blurry"}
    ]
    assert clip_metadata["windows"] == [
        {
            "start_frame": 0,
            "end_frame": 30,
            "qwen_caption": "main caption",
            "qwen_plus_enhanced_caption": "enhanced view",
        }
    ]
    assert clip_metadata["valid"] is True
    assert clip_metadata["has_caption"] is True

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    video_meta = _read_json(output_dir / "processed_videos" / "video.mp4.json")
    assert video_meta["video"] == video.input_path
    assert video_meta["num_total_clips"] == 1

    clip_chunk_meta = _read_json(output_dir / "processed_clip_chunks" / "video.mp4_0.json")
    assert clip_chunk_meta["num_clips_transcoded"] == 1
    assert clip_chunk_meta["num_clips_with_embeddings"] == 1
    assert clip_chunk_meta["num_clips_with_caption"] == 1
    assert clip_chunk_meta["num_clips_with_webp"] == 1
    assert clip_chunk_meta["max_clip_duration"] == pytest.approx(2.0)
    assert clip_chunk_meta["all_windows"][str(clip.uuid)] == {"0_30": "main caption"}
    assert clip_chunk_meta["all_windows_enhanced_caption"][str(clip.uuid)] == {"0_30": "enhanced view"}

    _assert_embeddings_written(output_dir, clip, video_uuid)

    cds_parquet_path = output_dir / "cds_parquet" / f"{video_uuid}_0.parquet"
    cds_df = pd.read_parquet(cds_parquet_path)
    assert len(cds_df) == 1
    assert cds_df.iloc[0]["id"] == str(clip.uuid)
    npt.assert_allclose(np.array(cds_df.iloc[0]["embedding"]), np.array([0.1, 0.2], dtype=np.float32))
    cds_meta = json.loads(cds_df.iloc[0]["$meta"])
    assert cds_meta["model_name"] == "internvideo2"
    assert cds_meta["model_version"] == "v1"
    assert cds_meta["caption"] == "main caption"
    assert cds_meta["clip_location"].endswith(f"clips/{clip.uuid}.mp4")


def test_chunked_metadata_writes_group_jsonl(tmp_path: Path) -> None:
    """Ensure chunked metadata buffering emits JSONL records."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_chunks=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    window = Window(
        start_frame=0,
        end_frame=15,
        caption={"qwen": "chunk caption"},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 1.5),
        encoded_data=b"data",
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(video=video)

    stage.process_data([task])
    consolidate_lance_fragments(str(output_dir), "default")

    per_clip_meta_path = output_dir / "metas" / "v0" / f"{clip.uuid}.json"
    assert not per_clip_meta_path.exists()

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    jsonl_path = output_dir / "metas_jsonl" / "v0" / f"{video_uuid}_0.jsonl"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1

    chunk_record = json.loads(lines[0])
    assert chunk_record["span_uuid"] == str(clip.uuid)
    assert chunk_record["has_caption"] is True
    assert chunk_record["windows"] == [{"start_frame": 0, "end_frame": 15, "qwen_caption": "chunk caption"}]
    assert chunk_record["clip_location"].endswith(f"clips/{clip.uuid}.mp4")

    chunk_stats = _read_json(output_dir / "processed_clip_chunks" / "video.mp4_0.json")
    assert chunk_stats["num_clips_with_embeddings"] == 0
    assert chunk_stats["num_clips_with_caption"] == 1


def test_chunked_metadata_writes_lance_dataset(tmp_path: Path) -> None:
    """Verify chunked metadata optionally writes to Lance alongside JSONL."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_lance=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    window = Window(
        start_frame=5,
        end_frame=25,
        caption={"qwen": "lance caption"},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.5),
        encoded_data=b"data",
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(video=video)

    stage.process_data([task])
    consolidate_lance_fragments(str(output_dir), "default")

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    lance_root = output_dir / "lance" / "v0"
    dataset = lance.dataset(lance_root.as_posix())
    rows = dataset.to_table().to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["span_uuid"] == str(clip.uuid)
    assert row["video_uuid"] == str(video_uuid)
    assert row["clip_chunk_index"] == 0
    assert row["windows"][0]["qwen_caption"] == "lance caption"


def test_lance_consolidation_is_idempotent(tmp_path: Path) -> None:
    """Verify calling consolidate_lance_fragments multiple times appends correctly and archives sidecars."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_lance=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    # First clip and consolidation
    window1 = Window(start_frame=0, end_frame=10, caption={"qwen": "first caption"})
    clip1 = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 1.0),
        encoded_data=b"data1",
        windows=[window1],
    )
    video1 = _build_video(video_path, clip1, clip_chunk_index=0)
    task1 = SplitPipeTask(video=video1)
    stage.process_data([task1])
    consolidate_lance_fragments(str(output_dir), "default")

    lance_root = output_dir / "lance" / "v0"
    dataset = lance.dataset(lance_root.as_posix())
    assert len(dataset.to_table()) == 1

    # Verify first sidecar was archived
    staging_dir = output_dir / "lance_fragments" / "v0"
    processed_dir = output_dir / "processed_lance_fragments" / "v0"
    assert len(list(staging_dir.glob("*.json"))) == 0
    assert len(list(processed_dir.glob("*.json"))) == 1

    # Second clip (different chunk) and consolidation
    window2 = Window(start_frame=10, end_frame=20, caption={"qwen": "second caption"})
    clip2 = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(1.0, 2.0),
        encoded_data=b"data2",
        windows=[window2],
    )
    video2 = _build_video(video_path, clip2, clip_chunk_index=1)
    task2 = SplitPipeTask(video=video2)
    stage.process_data([task2])
    consolidate_lance_fragments(str(output_dir), "default")

    # Verify dataset was appended (2 rows total)
    dataset = lance.dataset(lance_root.as_posix())
    rows = dataset.to_table().to_pylist()
    assert len(rows) == 2
    span_uuids = {row["span_uuid"] for row in rows}
    assert str(clip1.uuid) in span_uuids
    assert str(clip2.uuid) in span_uuids

    # Verify second sidecar was also archived
    assert len(list(staging_dir.glob("*.json"))) == 0
    assert len(list(processed_dir.glob("*.json"))) == 2

    # Third consolidation with no new sidecars should be a no-op
    consolidate_lance_fragments(str(output_dir), "default")
    dataset = lance.dataset(lance_root.as_posix())
    assert len(dataset.to_table()) == 2


def test_archive_processed_sidecars_raises_on_remote_delete_failure() -> None:
    """Ensure cleanup errors for remote sidecars surface instead of being swallowed."""
    staging_root = "s3://bucket/staging"
    processed_root = "s3://bucket/processed"
    sidecar_name = "chunk.json"
    payload = {"fragments": [], "schema_b64": base64.b64encode(b"schema").decode("ascii")}
    staged_path = storage_utils.get_full_path(staging_root, sidecar_name)
    staging_client = _FailingDeleteClient({str(staged_path): json.dumps(payload).encode("utf-8")})
    processed_client = _FailingDeleteClient()

    with pytest.raises(RuntimeError, match="Failed to delete remote sidecar"):
        _archive_processed_sidecars(
            [sidecar_name],
            staging_root,
            processed_root,
            staging_client,
            processed_client,
        )

    processed_path = storage_utils.get_full_path(processed_root, sidecar_name)
    assert str(processed_path) in processed_client.objects


def test_per_window_dataset_assets_written(tmp_path: Path) -> None:
    """Verify per-window dataset assets are written when enabled."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_embeddings=False,
        generate_cosmos_predict_dataset="enable",
    )

    window = Window(
        start_frame=0,
        end_frame=20,
        mp4_bytes=b"window-mp4",
        caption={"qwen": "dataset caption"},
        t5_xxl_embedding={"default": np.array([1, 2], dtype=np.int32)},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=b"clip-bytes",
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(video=video)

    stage.process_data([task])

    dataset_root = output_dir / "cosmos_predict2_video2world_dataset"
    video_file = dataset_root / "videos" / f"{clip.uuid}_0_20.mp4"
    meta_file = dataset_root / "metas" / f"{clip.uuid}_0_20.txt"
    t5_file = dataset_root / "t5_xxl" / f"{clip.uuid}_0_20.pickle"

    assert video_file.read_bytes() == b"window-mp4"
    assert meta_file.read_text() == "dataset caption"
    with t5_file.open("rb") as infile:
        stored_t5 = pickle.load(infile)  # noqa: S301 - reading data produced within the test
    assert len(stored_t5) == 1
    npt.assert_array_equal(stored_t5[0], np.array([1, 2], dtype=np.int32))

    assert window.mp4_bytes is None


def test_video_errors_written_to_error_path(tmp_path: Path) -> None:
    """Verify that videos with errors write to video_errors path and skip normal chunk metadata."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    errors = {"captioning": "test error"}
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    # Create a video with errors (no clips)
    metadata = VideoMetadata(
        height=1080,
        width=1920,
        framerate=30.0,
        num_frames=60,
        duration=2.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
    )
    video = Video(
        input_video=video_path,
        metadata=metadata,
        clips=[],
        filtered_clips=[],
        num_total_clips=0,
        num_clip_chunks=1,
        clip_chunk_index=0,
        errors=errors,
    )
    task = SplitPipeTask(video=video)

    stage.process_data([task])

    # Verify that normal chunk metadata is NOT written
    chunk_meta_path = output_dir / "processed_clip_chunks" / "video.mp4_0.json"
    assert not chunk_meta_path.exists()

    # Verify that error file IS written
    error_path = output_dir / "video_errors" / "video.mp4_0.json"
    assert error_path.exists()

    error_data = _read_json(error_path)
    assert error_data["video"] == video_path.as_posix()
    assert error_data["clip_chunk_index"] == 0
    assert error_data["errors"] == errors

    # Verify that video-level metadata is also NOT written (since errors exist)
    video_meta_path = output_dir / "processed_videos" / "video.mp4.json"
    assert not video_meta_path.exists()
