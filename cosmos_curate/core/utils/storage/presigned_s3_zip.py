# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper utilities for working with presigned S3 URLs that reference zip archives.

This module provides a minimal set of helpers to:

1. Download a zip archive from a presigned HTTPS URL and extract it to a
   temporary location so the pipeline can treat the contents like a normal
   *input_video_path* directory.
2. Create a zip archive from a local directory and upload it to a presigned
   HTTPS URL so the caller can fetch the results without direct access to the
   backing object store.

The implementation intentionally avoids pulling in any extra heavy-weight
dependencies so that importing it has negligible impact on start-up time.

Note: This module was previously called ``presigned_zip_utils``. It was renamed
in favour of a more explicit name indicating that it handles presigned **S3**
URLs specifically. Imports using the old name should be updated accordingly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import argparse
import contextlib
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import ray
import requests
from loguru import logger

from cosmos_curate.core.utils.infra import ray_cluster_utils
from cosmos_curate.core.utils.storage.storage_utils import (
    get_files_relative,
    get_storage_client,
)
from cosmos_curate.pipelines.video.read_write.summary_writers import (
    _write_all_window_captions,
)

__all__ = [
    "download_and_extract_zip",
    "gather_and_upload_outputs",
    "gather_outputs_from_all_nodes",
    "handle_presigned_urls",
    "zip_and_upload_directory",
]


def _download_file(url: str, dst_path: Path) -> None:
    """Download *url* to *dst_path* in a streaming fashion.

    Args:
        url: A presigned HTTPS URL pointing to the remote zip archive.
        dst_path: Local filesystem path where the downloaded file will be
            written. All missing parent directories will be created.

    Raises:
        requests.RequestException: If the remote server responds with a non-2xx
            HTTP status code.
        OSError: If the destination file cannot be written.

    """
    logger.info(f"Downloading file from presigned URL to {dst_path} …")

    # Ensure the destination directory exists before we start writing data.
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream the response in manageable chunks so very large archives do not
    # need to fit entirely in memory.
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dst_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=8 * 1024):
                if chunk:  # Filter out keep-alive chunks.
                    fp.write(chunk)

    logger.info("Download completed.")


def _download_and_extract_zip_single_node(
    presigned_url: str,
    tmp_dir: str | None = None,
) -> str:
    """Download a presigned zip archive and extract its contents.

    The downloaded archive is saved into a temporary directory (or *tmp_dir* if
    provided) before extraction.  The function returns the directory that
    contains the extracted files so downstream pipeline stages can treat the
    returned path like a normal ``input_video_path``.

    Args:
        presigned_url: Presigned HTTPS URL that grants temporary access to the
            zip archive stored in S3.
        tmp_dir: Optional path to an existing directory that should be used as
            the base for all temporary files.  If *None*, a fresh directory is
            created via :pyfunc:`tempfile.mkdtemp`.

    Returns:
        Path (as ``str``) to the directory containing the extracted archive
        contents.

    Raises:
        requests.RequestException: If the download fails.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
        OSError: If the archive cannot be written or extracted.

    """
    base_tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="input_videos_"))
    base_tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download the archive.
    zip_path = base_tmp_dir / "archive.zip"
    _download_file(presigned_url, zip_path)

    # 2. Extract the archive into its own sub-directory so that the *.zip* file
    # itself will never be mistaken for an input video by downstream code.
    extract_dir = base_tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    logger.info("Extraction completed.")

    # Heuristic: if the archive contains a single top-level directory, return
    # that directory directly; otherwise return *extract_dir*.
    top_level_items = list(extract_dir.iterdir())
    if len(top_level_items) == 1 and top_level_items[0].is_dir():
        return str(top_level_items[0])

    return str(extract_dir)


def zip_and_upload_directory(directory: str, presigned_url: str) -> None:
    """Create a zip archive from *directory* and upload it to *presigned_url*.

    Args:
        directory: Local directory whose contents should be archived.
        presigned_url: Presigned HTTPS URL (``PUT``) that grants write access to
            the destination object in S3.

    Raises:
        ValueError: If *directory* does not exist or is not a directory.
        requests.RequestException: If the upload fails or S3 returns a non-2xx
            response.
        OSError: If the temporary zip archive cannot be created or read.

    """
    src_dir = Path(directory).expanduser().resolve()
    if not src_dir.is_dir():
        msg = f"Directory to zip does not exist: {src_dir}"
        raise ValueError(msg)

    # Create the zip archive in the same filesystem to avoid potential cross-
    # device issues when later moving/removing the file.
    fd, tmp_path = tempfile.mkstemp(prefix="output_archive_", suffix=".zip")
    os.close(fd)
    tmp_path_path = Path(tmp_path)

    # ``shutil.make_archive`` expects the *base_name* **without** the extension.
    base_name = tmp_path_path.with_suffix("")
    shutil.make_archive(str(base_name), "zip", root_dir=str(src_dir))
    archive_path = base_name.with_suffix(".zip")

    logger.info(f"Uploading zipped output ({archive_path}) to presigned URL …")

    with archive_path.open("rb") as fp:
        response = requests.put(presigned_url, data=fp, timeout=60)

    try:
        response.raise_for_status()
    except Exception as exc:
        logger.error(f"Failed to upload archive: {exc}\n{response.text}")
        raise
    finally:
        # Always attempt to clean-up the temporary archive—do not let clean-up
        # failures mask the underlying exception.
        with contextlib.suppress(OSError):
            archive_path.unlink(missing_ok=True)

    logger.info("Upload completed successfully.")


# Reserve CPU resources for Ray actors
_ZIP_DOWNLOADER_CPU_REQUEST: float = 1.0  # full core per node for download
_OUTPUT_GATHERER_CPU_REQUEST: float = 0.1  # fractional core for lightweight gather tasks


def _download_and_extract_zip_impl(presigned_url: str, base_tmp_dir: str) -> str:
    """Download *presigned_url* to *base_tmp_dir* and extract its contents."""
    tmp_dir = Path(base_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_dir / "archive.zip"
    _download_file(presigned_url, zip_path)

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting downloaded archive …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # If the archive contains a single top-level directory, return that; else the extraction dir
    top_items = list(extract_dir.iterdir())
    if len(top_items) == 1 and top_items[0].is_dir():
        return str(top_items[0])
    return str(extract_dir)


@ray.remote(num_cpus=_ZIP_DOWNLOADER_CPU_REQUEST)
class _ZipDownloader:
    """Ray actor that performs a single download/extract on its node."""

    def __init__(self) -> None:
        self._node = ray_cluster_utils.get_node_name()

    def run(self, url: str, base_tmp_dir: str) -> tuple[str, str]:
        path = _download_and_extract_zip_impl(url, base_tmp_dir)
        return self._node, path


def _download_and_extract_on_all_nodes(url: str, base_tmp_dir: str) -> str:
    """Ensure the archive is downloaded & extracted once per Ray node."""
    bundles = [{"CPU": _ZIP_DOWNLOADER_CPU_REQUEST} for _ in ray_cluster_utils.get_live_nodes()]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    actors = [_ZipDownloader.options(placement_group=pg).remote() for _ in bundles]  # type: ignore[attr-defined]
    results = ray.get([a.run.remote(url, base_tmp_dir) for a in actors])

    for node, path in results:
        logger.info(f"Archive extracted on node {node} at {path}")

    # Path is identical across nodes
    return str(Path(base_tmp_dir) / "extracted")


def _worker_download_and_extract(url: str, base_tmp_dir: str) -> str:
    ray_cluster_utils.init_or_connect_to_cluster()
    extracted = _download_and_extract_on_all_nodes(url, base_tmp_dir)
    time.sleep(1)  # Let Ray flush logs
    ray.shutdown()
    return extracted


def download_and_extract_zip(presigned_url: str) -> str:
    """Download & extract input videos on **all** Ray nodes, returning driver path."""
    base_tmp_dir = str(Path(tempfile.gettempdir()) / f"input_videos_{uuid.uuid4().hex}")
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_worker_download_and_extract, presigned_url, base_tmp_dir)
        return fut.result()


def handle_presigned_urls(pipeline_type: str, pipeline_args: argparse.Namespace) -> argparse.Namespace:
    """Update *pipeline_args* in-place based on any presigned URLs present."""
    if getattr(pipeline_args, "input_presigned_s3_url", None):
        logger.info("Input presigned URL detected - downloading …")
        extracted_path = download_and_extract_zip(pipeline_args.input_presigned_s3_url)
        logger.info(f"Extracted to temporary directory: {extracted_path}")
        if pipeline_type == "split":
            pipeline_args.input_video_path = extracted_path
        elif pipeline_type == "semantic-dedup":
            pipeline_args.input_embeddings_path = extracted_path
        else:
            logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned input URL.")

    if getattr(pipeline_args, "output_presigned_s3_url", None):
        if pipeline_type == "split":
            if not getattr(pipeline_args, "output_clip_path", None):
                pipeline_args.output_clip_path = tempfile.mkdtemp(prefix="output_split_")
                logger.warning(
                    f"No output_clip_path provided; using temporary directory {pipeline_args.output_clip_path}",
                )
        elif pipeline_type == "semantic-dedup":
            if not getattr(pipeline_args, "output_path", None):
                pipeline_args.output_path = tempfile.mkdtemp(prefix="output_dedup_")
                logger.warning(
                    f"No output_path provided; using temporary directory {pipeline_args.output_path}",
                )
        else:
            logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned output URL.")
    return pipeline_args


@ray.remote(num_cpus=_OUTPUT_GATHERER_CPU_REQUEST)
class _OutputGatherer:
    """Actor that zips local output directory and returns bytes."""

    def run(self, output_dir: str) -> tuple[str, Any | None]:
        node = ray_cluster_utils.get_node_name()
        out_path = Path(output_dir)
        if not out_path.exists():
            return node, None
        if sum(len(files) for _, _, files in os.walk(out_path)) == 0:
            return node, None

        fd, zip_path_str = tempfile.mkstemp(prefix="node_output_", suffix=".zip")
        os.close(fd)
        zip_path = Path(zip_path_str)
        shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", output_dir)
        zip_file = zip_path.with_suffix(".zip")
        with zip_file.open("rb") as fh:
            data = fh.read()
        zip_file.unlink(missing_ok=True)
        return node, ray.put(data)


def _extract_zip_bytes(buf: bytes, dest_dir: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(buf)
        tmp_path = Path(tmp.name)
    with zipfile.ZipFile(tmp_path, "r") as zf:
        zf.extractall(dest_dir)
    tmp_path.unlink(missing_ok=True)


def _gather_outputs_on_all_nodes(output_dir: str) -> None:
    bundles = [{"CPU": _OUTPUT_GATHERER_CPU_REQUEST} for _ in ray_cluster_utils.get_live_nodes()]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    actors = [_OutputGatherer.options(placement_group=pg).remote() for _ in bundles]  # type: ignore[attr-defined]
    results = ray.get([a.run.remote(output_dir) for a in actors])

    for node, obj in results:
        if obj is None:
            logger.info(f"No output data on node {node}")
            continue
        data = ray.get(obj)
        _extract_zip_bytes(data, output_dir)
        logger.info(f"Merged output from node {node}")


def _worker_gather_outputs(output_dir: str) -> None:
    ray_cluster_utils.init_or_connect_to_cluster()
    _gather_outputs_on_all_nodes(output_dir)
    time.sleep(1)  # Let Ray flush logs
    ray.shutdown()


def gather_outputs_from_all_nodes(output_directory: str) -> None:
    """Collect outputs from all Ray nodes into *output_directory*."""
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_worker_gather_outputs, output_directory)
        try:
            fut.result(timeout=300)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Gather outputs failed: {exc}")


def gather_and_upload_outputs(pipeline_type: str, args: argparse.Namespace) -> None:
    """Gather outputs, write metadata, zip, and upload via presigned URL."""
    if getattr(args, "output_presigned_s3_url", None) is None:
        return

    output_path: str | None = None
    if pipeline_type == "split":
        if getattr(args, "output_clip_path", None) is None:
            logger.warning("output_clip_path for split pipeline is not set?")
            return
        output_path = str(args.output_clip_path)
    elif pipeline_type == "semantic-dedup":
        if getattr(args, "output_path", None) is None:
            logger.warning("output_path for semantic-dedup pipeline is not set?")
            return
        output_path = str(args.output_path)
    else:
        logger.warning(f"Unsupported pipeline type '{pipeline_type}' for presigned output URL.")
        return

    try:
        logger.info("Gathering per-node outputs …")
        gather_outputs_from_all_nodes(output_path)

        if pipeline_type == "split":
            logger.info("Re-writing consolidated window-captions metadata …")
            input_client = get_storage_client(args.input_video_path)
            output_client = get_storage_client(output_path, can_overwrite=True)
            _write_all_window_captions(
                output_path,
                None,
                get_files_relative(args.input_video_path, input_client),
                0,
                output_client,
            )

        logger.info("Uploading zipped outputs …")
        zip_and_upload_directory(output_path, args.output_presigned_s3_url)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to gather/upload outputs: {exc}")
    finally:
        if "output_split_" in output_path or "output_dedup_" in output_path:
            with contextlib.suppress(OSError):
                shutil.rmtree(output_path)
