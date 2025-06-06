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

import contextlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
from loguru import logger

__all__ = [
    "download_and_extract_zip",
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


def download_and_extract_zip(presigned_url: str, tmp_dir: str | None = None) -> str:
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
    """Create a zip archive from *directory* and upload it via *presigned_url*.

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
