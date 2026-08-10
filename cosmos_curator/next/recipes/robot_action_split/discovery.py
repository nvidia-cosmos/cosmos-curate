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

"""Parquet-driven span provider for ``robot-action-split``.

Reads LeRobot/Mecka dataset shards and emits one SpanWorkItem per (subtask
span, camera view).  No video is decoded during discovery.

Discovery phases
----------------
1. Shard enumeration  -- list shard_*/ dirs under each input URI.
2. Per-shard prep     -- read meta/info.json, subtasks.parquet, tasks.parquet;
                         discover videos/ view dirs; list data chunk files.
3. Segment build      -- per data file: read data parquet + episode parquets;
                         numpy diff-based subtask boundary detection.
4. Filter + dedup     -- duration bounds, label skip, per-episode dedup.
"""

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pyarrow.parquet as pq

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client, path_exists, read_bytes
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig, SpanFilterConfig
from cosmos_curator.next.recipes.robot_action_split.identities import (
    make_clip_id,
    make_source_id,
    make_span_group_id,
)

_CHUNK_FILE_RE = re.compile(r"^chunk-(\d+)/file-(\d+)\.parquet$")
_VIEW_PREFIX = "observation.images."

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class SpanWorkItem:
    """One (subtask span, camera view) to cut and register."""

    # Identity
    source_id: str
    span_group_id: str
    clip_id: str
    view_name: str

    # Source location
    chunk_mp4_uri: str
    data_parquet_uri: str

    # Span geometry (integer frames)
    episode_index: int
    episode_frame_base: int
    frame_start: int
    frame_end: int
    native_fps: float
    episode_from_timestamp: float

    # Labels
    subtask_index: int
    subtask_name: str
    task_index: int
    task_name: str

    # Episode identity
    episode_id: str

    # Camera
    camera_intrinsics: list[float] | None

    @property
    def duration_s(self) -> float:
        """Span duration in seconds."""
        return (self.frame_end - self.frame_start) / self.native_fps

    @property
    def start_ns(self) -> int:
        """Span start as absolute nanoseconds within the chunk MP4."""
        chunk_frame_offset = round(self.episode_from_timestamp * self.native_fps)
        abs_start = chunk_frame_offset + (self.frame_start - self.episode_frame_base)
        return round(abs_start / self.native_fps * 1e9)

    @property
    def end_ns(self) -> int:
        """Span end (exclusive) as absolute nanoseconds within the chunk MP4."""
        chunk_frame_offset = round(self.episode_from_timestamp * self.native_fps)
        abs_end = chunk_frame_offset + (self.frame_end - self.episode_frame_base)
        return round(abs_end / self.native_fps * 1e9)


@attrs.define
class ChunkSpanBatch:
    """All SpanWorkItems sharing the same source chunk MP4 and data parquet."""

    chunk_mp4_uri: str
    data_parquet_uri: str
    items: list[SpanWorkItem]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _get_client(uri: str, storage_profile: str) -> StorageClient:
    """Return a storage client for *uri*, raising if the path is not remote."""
    client = get_storage_client(uri, profile_name=storage_profile)
    if client is None:
        msg = f"No storage client available for remote URI: {uri}"
        raise RuntimeError(msg)
    return client


def _s3_list_relative(uri: str, storage_profile: str) -> list[str]:
    """Return keys under *uri* relative to that prefix, via recursive S3 listing."""
    client = _get_client(uri, storage_profile)
    s3p = S3Prefix(uri.rstrip("/") + "/")
    root_key = s3p.prefix.rstrip("/") + "/"
    objects = client.list_recursive(s3p)
    result = []
    for obj in objects:
        key = obj["Key"]
        if key.startswith(root_key):
            result.append(key[len(root_key) :])
    return result


# ---------------------------------------------------------------------------
# Shard/filesystem helpers
# ---------------------------------------------------------------------------


def _list_shard_dirs(
    root: str,
    *,
    storage_profile: str = "default",
) -> list[tuple[str, str]]:
    """Return (path, name) pairs for shard directories under *root*.

    Each returned path has a ``meta/`` subdirectory (or S3 prefix).
    """
    if _is_s3(root):
        client = _get_client(root, storage_profile)
        s3p = S3Prefix(root.rstrip("/") + "/")
        root_key = s3p.prefix.rstrip("/") + "/"
        bucket = s3p.bucket
        try:
            all_objects = client.list_recursive(s3p)
        except Exception:  # noqa: BLE001
            return []

        subdir_names: dict[str, set[str]] = {}
        for obj in all_objects:
            key = obj["Key"]
            rel = key.removeprefix(root_key)
            parts = rel.split("/", 1)
            if len(parts) >= 2:  # noqa: PLR2004
                subdir_names.setdefault(parts[0], set()).add(parts[1])

        result = []
        for name in sorted(subdir_names):
            info_key = f"s3://{bucket}/{root_key}{name}/meta/info.json"
            if client.object_exists(S3Prefix(info_key)):
                uri = f"s3://{bucket}/{root_key}{name}"
                result.append((uri, name))

        # Flat layout: root itself is the shard.
        if not result:
            flat_info = f"s3://{bucket}/{root_key}meta/info.json"
            if client.object_exists(S3Prefix(flat_info)):
                name = root.rstrip("/").rsplit("/", 1)[-1]
                result.append((root.rstrip("/"), name))

        return result

    root_path = Path(root)
    dirs = [(str(p), p.name) for p in sorted(root_path.iterdir()) if p.is_dir() and (p / "meta").is_dir()]
    # Flat layout: root itself is the shard.
    if not dirs and (root_path / "meta").is_dir():
        dirs.append((root, root_path.name))
    return dirs


def _read_shard_meta(
    shard_path: str,
    *,
    storage_profile: str = "default",
) -> tuple[dict[int, str], dict[int, str], float]:
    """Read subtask map, task map, and fps from a shard's meta directory."""
    if _is_s3(shard_path):
        client = _get_client(shard_path, storage_profile)

        info_uri = shard_path.rstrip("/") + "/meta/info.json"
        info = json.loads(read_bytes(info_uri, client=client).decode("utf-8"))
        fps = float(info.get("fps", 30))

        subtask_map: dict[int, str] = {}
        subtask_uri = shard_path.rstrip("/") + "/meta/subtasks.parquet"
        if client.object_exists(S3Prefix(subtask_uri)):
            data = read_bytes(subtask_uri, client=client)
            subtask_df = pq.read_table(io.BytesIO(data)).to_pandas()
            _SUBTASK_LABEL_COLS = ("subtask", "subtask_name")
            subtask_col = next((c for c in _SUBTASK_LABEL_COLS if c in subtask_df.columns), None)
            for _, row in subtask_df.iterrows():
                subtask_map[int(row["subtask_index"])] = (
                    str(row[subtask_col]) if subtask_col else f"subtask_{int(row['subtask_index'])}"
                )

        tasks_uri = shard_path.rstrip("/") + "/meta/tasks.parquet"
        task_data = read_bytes(tasks_uri, client=client)
        task_df = pq.read_table(io.BytesIO(task_data)).to_pandas()
        text_col = (
            "task" if "task" in task_df.columns else next((c for c in task_df.columns if c != "task_index"), None)
        )
        task_map: dict[int, str] = {
            int(row["task_index"]): (str(row[text_col]) if text_col else f"task_{int(row['task_index'])}")
            for _, row in task_df.iterrows()
        }

        return subtask_map, task_map, fps

    meta_dir = Path(shard_path) / "meta"

    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))

    subtask_map = {}
    subtask_file = meta_dir / "subtasks.parquet"
    if subtask_file.is_file():
        subtask_df = pq.read_table(str(subtask_file)).to_pandas()
        _SUBTASK_LABEL_COLS = ("subtask", "subtask_name")
        subtask_col = next((c for c in _SUBTASK_LABEL_COLS if c in subtask_df.columns), None)
        for _, row in subtask_df.iterrows():
            subtask_map[int(row["subtask_index"])] = (
                str(row[subtask_col]) if subtask_col else f"subtask_{int(row['subtask_index'])}"
            )

    task_df = pq.read_table(str(meta_dir / "tasks.parquet")).to_pandas()
    text_col = "task" if "task" in task_df.columns else next((c for c in task_df.columns if c != "task_index"), None)
    task_map = {
        int(row["task_index"]): (str(row[text_col]) if text_col else f"task_{int(row['task_index'])}")
        for _, row in task_df.iterrows()
    }

    return subtask_map, task_map, fps


def _local_data_file_indices(data_dir: Path) -> set[tuple[int, int]]:
    """Scan a local ``data/`` directory for ``(chunk_index, file_index)`` pairs."""
    pairs: set[tuple[int, int]] = set()
    for chunk_dir in data_dir.iterdir():
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for fpath in chunk_dir.iterdir():
            if fpath.name.startswith("file-") and fpath.suffix == ".parquet":
                try:
                    ci = int(chunk_dir.name.split("-")[1])
                    fi = int(fpath.stem.split("-")[1])
                    pairs.add((ci, fi))
                except (IndexError, ValueError):
                    pass
    return pairs


def _list_data_files(
    shard_path: str,
    *,
    storage_profile: str = "default",
) -> list[tuple[int, int]]:
    """Return sorted (chunk_index, file_index) pairs under data/."""
    if _is_s3(shard_path):
        data_uri = shard_path.rstrip("/") + "/data/"
        try:
            relative_keys = _s3_list_relative(data_uri, storage_profile)
        except Exception:  # noqa: BLE001
            return []
        pairs: set[tuple[int, int]] = set()
        for rel in relative_keys:
            m = _CHUNK_FILE_RE.match(rel)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
        return sorted(pairs)

    data_dir = Path(shard_path) / "data"
    if not data_dir.is_dir():
        return []
    return sorted(_local_data_file_indices(data_dir))


def _discover_view_names(  # noqa: C901
    shard_path: str,
    requested: tuple[str, ...],
    *,
    storage_profile: str = "default",
) -> tuple[list[str], bool]:
    """Return (selected_views, source_is_multiview)."""
    if _is_s3(shard_path):
        videos_uri = shard_path.rstrip("/") + "/videos/"
        try:
            relative_keys = _s3_list_relative(videos_uri, storage_profile)
        except Exception:  # noqa: BLE001
            relative_keys = []
        views: set[str] = set()
        for rel in relative_keys:
            first = rel.split("/", 1)[0]
            if first.startswith(_VIEW_PREFIX):
                views.add(first)
        available = sorted(views) or ["observation.images.main"]
        source_is_multiview = len(available) > 1
        if requested:
            selected = [v for v in requested if v in set(available)]
            if not selected:
                msg = (
                    f"None of the requested views {list(requested)} are available "
                    f"in {shard_path}; available: {available}"
                )
                raise ValueError(msg)
            return (selected, source_is_multiview)
        return (available, source_is_multiview)

    vid_dir = Path(shard_path) / "videos"
    if not vid_dir.is_dir():
        return (["observation.images.main"], False)
    available = sorted(p.name for p in vid_dir.iterdir() if p.is_dir() and p.name.startswith("observation.images."))
    if not available:
        available = ["observation.images.main"]
    source_is_multiview = len(available) > 1
    if requested:
        selected = [v for v in requested if v in set(available)]
        if not selected:
            msg = f"None of the requested views {list(requested)} are available in {shard_path}; available: {available}"
            raise ValueError(msg)
        return (selected, source_is_multiview)
    return (available, source_is_multiview)


def _read_episode_meta(
    shard_path: str,
    view_names: list[str],
    *,
    storage_profile: str = "default",
) -> dict[int, dict[str, Any]]:
    """Build episode_index to metadata from meta/episodes/ parquets."""
    shard_name = shard_path.rstrip("/").rsplit("/", 1)[-1]
    meta: dict[int, dict[str, Any]] = {}
    if _is_s3(shard_path):
        client = _get_client(shard_path, storage_profile)
        ep_uri = shard_path.rstrip("/") + "/meta/episodes/"
        s3p = S3Prefix(ep_uri)
        root_key = s3p.prefix.rstrip("/") + "/"
        bucket = s3p.bucket
        try:
            relative_keys = _s3_list_relative(ep_uri, storage_profile)
        except Exception:  # noqa: BLE001
            return meta
        for rel in sorted(relative_keys):
            if not rel.endswith(".parquet"):
                continue
            uri = f"s3://{bucket}/{root_key}{rel}"
            data = read_bytes(uri, client=client)
            meta = _parse_episode_parquet(io.BytesIO(data), view_names, meta, shard_name)
        return meta

    ep_root = Path(shard_path) / "meta" / "episodes"
    if not ep_root.is_dir():
        return meta
    for fpath in sorted(ep_root.rglob("*.parquet")):
        meta = _parse_episode_parquet(str(fpath), view_names, meta, shard_name)
    return meta


def _synthetic_episode_id(shard_name: str, episode_index: int) -> str:
    """Return a synthetic episode_id when the column is absent.

    episode_id is a derived field added by the vendor to their parquet schema;
    it is not part of the standard LeRobot format. When absent, falling back
    to str(episode_index) alone would cause cross-shard collisions since
    episode_index is only unique within a shard. This synthetic id uses
    ``"{shard_name}:{episode_index}"`` which is unique within a dataset root:
    shards have distinct names and episode_index is unique within each shard.
    Cross-root uniqueness is already handled by source_id (hash of the dataset
    root URI) being part of span_group_id.
    """
    return f"{shard_name}:{episode_index}"


def _parse_episode_parquet(
    source: str | io.BytesIO,
    view_names: list[str],
    meta: dict[int, dict[str, Any]],
    shard_name: str = "",
) -> dict[int, dict[str, Any]]:
    """Parse one episode parquet file and merge its rows into *meta*."""
    schema_names = set(pq.read_schema(source).names)
    if isinstance(source, io.BytesIO):
        source.seek(0)
    cols = [c for c in ("episode_index", "episode_id", "camera_intrinsics", "dataset_from_index") if c in schema_names]
    for v in view_names:
        for suffix in ("chunk_index", "file_index", "from_timestamp"):
            col = f"videos/{v}/{suffix}"
            if col in schema_names:
                cols.append(col)
    if "from_timestamp" in schema_names:
        cols.append("from_timestamp")
    if "episode_index" not in cols:
        return meta
    d = pq.read_table(source, columns=cols).to_pydict()
    n = len(d["episode_index"])
    for i in range(n):
        ep = int(d["episode_index"][i])
        intrinsics = d.get("camera_intrinsics", [None] * n)[i]
        views_info: dict[str, dict[str, Any]] = {}
        for v in view_names:
            cc, fc, tc = f"videos/{v}/chunk_index", f"videos/{v}/file_index", f"videos/{v}/from_timestamp"
            views_info[v] = {
                "chunk_index": int(d[cc][i]) if cc in d and d[cc][i] is not None else None,
                "file_index": int(d[fc][i]) if fc in d and d[fc][i] is not None else None,
                "from_timestamp": (
                    float(d[tc][i])
                    if tc in d and d[tc][i] is not None
                    else float(d["from_timestamp"][i])
                    if "from_timestamp" in d
                    else 0.0
                ),
            }
        meta[ep] = {
            "episode_id": str(d["episode_id"][i]) if "episode_id" in d else _synthetic_episode_id(shard_name, ep),
            "camera_intrinsics": [float(x) for x in intrinsics] if intrinsics is not None else None,
            "dataset_from_index": int(d["dataset_from_index"][i]) if "dataset_from_index" in d else 0,
            "views": views_info,
        }
    return meta


def _view_mp4_path(shard_path: str, view: str, meta: dict[str, Any], data_chunk_index: int) -> str:
    vinfo = meta.get("views", {}).get(view, {})
    ci = vinfo.get("chunk_index") if vinfo.get("chunk_index") is not None else data_chunk_index
    fi = vinfo.get("file_index") if vinfo.get("file_index") is not None else 0
    if _is_s3(shard_path):
        return f"{shard_path.rstrip('/')}/videos/{view}/chunk-{ci:03d}/file-{fi:03d}.mp4"
    return str(Path(shard_path) / "videos" / view / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4")


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------


def _build_segments_for_file(  # noqa: C901, PLR0913
    shard_path: str,
    source_id: str,
    ci: int,
    fi: int,
    subtask_map: dict[int, str],
    task_map: dict[int, str],
    fps: float,
    episode_meta: dict[int, dict[str, Any]],
    view_names: list[str],
    video_bitrate: str,
    *,
    source_is_multiview: bool,
    storage_profile: str = "default",
) -> list[SpanWorkItem]:
    """Build SpanWorkItems from one data parquet file."""
    if _is_s3(shard_path):
        data_uri = f"{shard_path.rstrip('/')}/data/chunk-{ci:03d}/file-{fi:03d}.parquet"
        client = _get_client(shard_path, storage_profile)
        if not path_exists(data_uri, client=client):
            return []
        data_bytes = read_bytes(data_uri, client=client)
        parquet_source: str | io.BytesIO = io.BytesIO(data_bytes)
        data_parquet_uri = data_uri
    else:
        data_path = Path(shard_path) / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        if not data_path.is_file():
            return []
        parquet_source = str(data_path)
        data_parquet_uri = str(data_path)

    schema_names = set(pq.read_schema(parquet_source).names)
    if isinstance(parquet_source, io.BytesIO):
        parquet_source.seek(0)
    has_subtask = "subtask_index" in schema_names
    cols = ["episode_index", "frame_index", "task_index", *(["subtask_index"] if has_subtask else [])]
    t = pq.read_table(parquet_source, columns=cols)
    ep_arr = np.asarray(t.column("episode_index").to_numpy())
    fr_arr = np.asarray(t.column("frame_index").to_numpy())
    ti_arr = np.asarray(t.column("task_index").to_numpy())
    si_arr = np.asarray(t.column("subtask_index").to_numpy()) if has_subtask else ti_arr.copy()

    if len(ep_arr) == 0:
        return []

    if len(ep_arr) > 1 and np.any(np.diff(ep_arr) < 0):
        order = np.lexsort((fr_arr, ep_arr))
        ep_arr, fr_arr, ti_arr, si_arr = ep_arr[order], fr_arr[order], ti_arr[order], si_arr[order]

    ep_bounds = np.concatenate(([0], np.where(np.diff(ep_arr) != 0)[0] + 1, [len(ep_arr)]))
    items: list[SpanWorkItem] = []

    for i in range(len(ep_bounds) - 1):
        lo, hi = int(ep_bounds[i]), int(ep_bounds[i + 1])
        ep_idx = int(ep_arr[lo])
        meta = episode_meta.get(ep_idx)
        if meta is None:
            continue

        frames = fr_arr[lo:hi]
        subtasks = si_arr[lo:hi]
        tasks = ti_arr[lo:hi]
        ep_frame_base = int(frames[0])

        run_bounds = np.concatenate(([0], np.where(np.diff(subtasks) != 0)[0] + 1, [len(subtasks)]))
        for j in range(len(run_bounds) - 1):
            rs, re = int(run_bounds[j]), int(run_bounds[j + 1])
            st_idx = int(subtasks[rs])
            t_idx = int(tasks[rs])
            frame_s, frame_e = int(frames[rs]), int(frames[re - 1]) + 1
            subtask_name = (
                subtask_map.get(st_idx, task_map.get(t_idx, f"subtask_{st_idx}"))
                if has_subtask
                else task_map.get(t_idx, f"task_{t_idx}")
            )
            span_group_id = make_span_group_id(source_id, meta["episode_id"], st_idx, frame_s)
            for view in view_names:
                vinfo = meta.get("views", {}).get(view, {})
                items.append(
                    SpanWorkItem(
                        source_id=source_id,
                        span_group_id=span_group_id,
                        clip_id=make_clip_id(
                            span_group_id, view, video_bitrate, source_is_multiview=source_is_multiview
                        ),
                        view_name=view,
                        chunk_mp4_uri=_view_mp4_path(shard_path, view, meta, ci),
                        data_parquet_uri=data_parquet_uri,
                        episode_index=ep_idx,
                        episode_frame_base=ep_frame_base,
                        frame_start=frame_s,
                        frame_end=frame_e,
                        native_fps=fps,
                        episode_from_timestamp=float(vinfo.get("from_timestamp", 0.0)),
                        subtask_index=st_idx,
                        subtask_name=subtask_name,
                        task_index=t_idx,
                        task_name=task_map.get(t_idx, f"task_{t_idx}"),
                        episode_id=str(meta["episode_id"]),
                        camera_intrinsics=meta.get("camera_intrinsics"),
                    )
                )
    return items


# ---------------------------------------------------------------------------
# Filter + dedup
# ---------------------------------------------------------------------------


def _filter_by_label(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    skip_set = frozenset(s.lower() for s in cfg.skip_labels)
    prefixes = tuple(p.lower() for p in cfg.skip_label_prefixes)
    substrings = tuple(s.lower() for s in cfg.skip_label_substrings)
    return [
        item
        for item in items
        if (name := item.subtask_name.strip().lower()) not in skip_set
        and not (prefixes and name.startswith(prefixes))
        and not (substrings and any(sub in name for sub in substrings))
    ]


def _filter_by_duration(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    return [
        item
        for item in items
        if item.duration_s >= cfg.min_duration_s
        and (cfg.max_duration_s is None or item.duration_s <= cfg.max_duration_s)
    ]


def _dedup(items: list[SpanWorkItem], cfg: SpanFilterConfig) -> list[SpanWorkItem]:
    """Per-episode dedup: keep at most max_keep_per_description spans per subtask_name."""

    def ep_key(x: SpanWorkItem) -> tuple[str, str]:
        # episode_id is the stable vendor-assigned identifier; episode_index is
        # shard-local and is not unique across shards.
        return (x.source_id, x.episode_id)

    result: list[SpanWorkItem] = []
    for _, ep_iter in groupby(sorted(items, key=ep_key), key=ep_key):
        ep_items = list(ep_iter)
        by_desc: dict[str, list[SpanWorkItem]] = {}
        for item in ep_items:
            by_desc.setdefault(item.subtask_name, []).append(item)

        for group in by_desc.values():
            span_groups: dict[str, list[SpanWorkItem]] = {}
            for item in group:
                span_groups.setdefault(item.span_group_id, []).append(item)
            span_keys = list(dict.fromkeys(item.span_group_id for item in group))

            if len(span_keys) <= cfg.max_keep_per_description:
                result.extend(group)
                continue

            def _max_dur(k: str, _sg: dict[str, list[SpanWorkItem]] = span_groups) -> float:
                return max(i.duration_s for i in _sg[k])

            preferred = sorted(
                [
                    k
                    for k in span_keys
                    if any(cfg.dedup_prefer_min_s <= i.duration_s <= cfg.dedup_prefer_max_s for i in span_groups[k])
                ],
                key=lambda k: (-_max_dur(k), k),
            )
            if len(preferred) < cfg.max_keep_per_description:
                remaining = sorted(
                    [k for k in span_keys if k not in set(preferred)],
                    key=lambda k: (-_max_dur(k), k),
                )
                preferred = [*preferred, *remaining[: cfg.max_keep_per_description - len(preferred)]]

            for k in preferred[: cfg.max_keep_per_description]:
                result.extend(span_groups[k])

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discover_spans(config: ResolvedRobotActionSplitConfig) -> list[ChunkSpanBatch]:
    """Run span discovery for all dataset roots in the config.

    Returns a list of ChunkSpanBatch objects ready for Ray Data processing.
    Each batch groups all SpanWorkItems that share the same source chunk MP4
    and data parquet, so each Ray worker opens one video file.
    """
    all_items: list[SpanWorkItem] = []
    workers = config.execution.discovery_workers
    requested_views = config.output.views
    storage_profile = config.execution.storage_profile
    video_bitrate = config.output.video_bitrate
    chunk_limit = config.input.limit
    # seen_chunks is global across all input URIs so limit applies to the total
    # number of chunks processed, not per-URI.
    seen_chunks: set[str] = set()

    for dataset_root in config.input.uris:
        source_id = make_source_id(dataset_root)
        shard_dirs = _list_shard_dirs(dataset_root, storage_profile=storage_profile)
        if not shard_dirs:
            continue

        def _prep(
            args: tuple[str, str],
        ) -> tuple[
            str,
            dict[int, str],
            dict[int, str],
            float,
            list[tuple[int, int]],
            list[str],
            bool,
            dict[int, dict[str, Any]],
        ]:
            shard_path, _ = args
            subtask_map, task_map, fps = _read_shard_meta(shard_path, storage_profile=storage_profile)
            views, source_is_multiview = _discover_view_names(
                shard_path, requested_views, storage_profile=storage_profile
            )
            data_files = _list_data_files(shard_path, storage_profile=storage_profile)
            episode_meta = _read_episode_meta(shard_path, views, storage_profile=storage_profile)
            return shard_path, subtask_map, task_map, fps, data_files, views, source_is_multiview, episode_meta

        with ThreadPoolExecutor(max_workers=min(len(shard_dirs), workers)) as pool:
            futures = [pool.submit(_prep, sd) for sd in shard_dirs]
            shard_infos = [f.result() for f in futures]

        file_tasks = [
            (sp, ci, fi, stm, tm, fps, views, sim, ep_meta)
            for sp, stm, tm, fps, data_files, views, sim, ep_meta in shard_infos
            for ci, fi in data_files
        ]
        if chunk_limit is not None:
            filtered: list[tuple[Any, ...]] = []
            for task in file_tasks:
                key = f"{task[0]}/chunk-{task[1]:03d}"
                if key not in seen_chunks and len(seen_chunks) < chunk_limit:
                    seen_chunks.add(key)
                if key in seen_chunks:
                    filtered.append(task)
            file_tasks = filtered

        def _build(args: tuple[Any, ...], _sid: str = source_id, _vb: str = video_bitrate) -> list[SpanWorkItem]:
            sp, ci, fi, stm, tm, fps, views, sim, ep_meta = args
            return _build_segments_for_file(
                sp,
                _sid,
                ci,
                fi,
                stm,
                tm,
                fps,
                ep_meta,
                views,
                _vb,
                source_is_multiview=sim,
                storage_profile=storage_profile,
            )

        with ThreadPoolExecutor(max_workers=min(len(file_tasks) or 1, workers)) as pool:
            all_items.extend(
                item for r in as_completed({pool.submit(_build, t): t for t in file_tasks}) for item in r.result()
            )

    cfg = config.split
    all_items = _filter_by_label(all_items, cfg)
    all_items = _filter_by_duration(all_items, cfg)
    all_items = _dedup(all_items, cfg)

    by_chunk: dict[tuple[str, str], list[SpanWorkItem]] = {}
    for item in all_items:
        by_chunk.setdefault((item.chunk_mp4_uri, item.data_parquet_uri), []).append(item)

    max_per_batch = config.execution.max_segments_per_batch
    return [
        ChunkSpanBatch(
            chunk_mp4_uri=mp4_uri, data_parquet_uri=parquet_uri, items=chunk_items[start : start + max_per_batch]
        )
        for (mp4_uri, parquet_uri), chunk_items in by_chunk.items()
        for start in range(0, len(chunk_items), max_per_batch)
    ]
