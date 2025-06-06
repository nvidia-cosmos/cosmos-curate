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

"""S3 Utilities."""

import json
import pathlib
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from cosmos_curate.core.utils import s3_client, storage_client
from cosmos_curate.core.utils.misc.retry_utils import do_with_retries
from cosmos_curate.core.utils.s3_client import is_s3path

if TYPE_CHECKING:
    from collections.abc import Sequence


def read_bytes(
    filepath: str | s3_client.S3Prefix | pathlib.Path,
    client: s3_client.S3Client | None = None,
) -> bytes:
    """Read bytes from a file or S3 object.

    Args:
        filepath: Path to the file or S3 object.
        client: Optional S3 client for S3 paths.

    Returns:
        The file contents as bytes.

    """
    if is_s3path(str(filepath)):
        s3_path = s3_client.S3Prefix(str(filepath))

        def func_to_call() -> bytes:
            assert client is not None
            return client.download_object_as_bytes(s3_path)

        return do_with_retries(func_to_call)
    assert not isinstance(filepath, s3_client.S3Prefix)
    with pathlib.Path(str(filepath)).open("rb") as fp:
        return fp.read()


def read_json_file(
    filepath: s3_client.S3Prefix | pathlib.Path,
    client: s3_client.S3Client | None = None,
) -> dict[str, object] | Any:  # noqa: ANN401
    """Read a JSON file from a file or S3 object.

    Args:
        filepath: Path to the JSON file or S3 object.
        client: Optional S3 client for S3 paths.

    Returns:
        The JSON data as a dictionary.

    """
    if is_s3path(str(filepath)):
        buffer = read_bytes(filepath, client)
        return json.loads(buffer.decode("utf-8"))
    assert not isinstance(filepath, s3_client.S3Prefix)
    with pathlib.Path(str(filepath)).open() as fp:
        return json.load(fp)


def read_txt_file(
    filepath: s3_client.S3Prefix | pathlib.Path,
    client: s3_client.S3Client | None = None,
) -> str:
    """Read a text file from a file or S3 object.

    Args:
        filepath: Path to the text file or S3 object.
        client: Optional S3 client for S3 paths.

    Returns:
        The text file contents as a string.

    """
    if is_s3path(str(filepath)):
        buffer = read_bytes(filepath, client)
        return buffer.decode("utf-8")
    assert not isinstance(filepath, s3_client.S3Prefix)
    with pathlib.Path(str(filepath)).open() as fp:
        return fp.read()


def path_exists(path: str | s3_client.S3Prefix | pathlib.Path, client: s3_client.S3Client | None) -> bool:
    """Check if a path exists.

    Args:
        path: Path to check.
        client: Optional S3 client for S3 paths.

    Returns:
        True if the path exists, False otherwise.

    """
    if is_s3path(str(path)):
        assert client is not None
        return client.object_exists(s3_client.S3Prefix(str(path)))
    assert isinstance(path, (str, pathlib.Path))
    return pathlib.Path(str(path)).exists()


def verify_path(path: str, level: int = 0) -> None:
    """Verify the input /output path exists.

    Args:
        path: Path to verify.
        level: Level of the path to verify.

    """
    if not is_s3path(path):
        path_to_check = pathlib.Path(path)
        while level > 0:
            path_to_check = path_to_check.parent
            level -= 1
        if not pathlib.Path(path_to_check).exists():
            error_msg = f"Local path {path_to_check} does not exist."
            raise FileNotFoundError(error_msg)


def create_path(path: str) -> None:
    """Create a path.

    Args:
        path: Path to create.

    """
    if not is_s3path(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def is_path_nested(path1: str, path2: str) -> bool:
    """Check if one path is nested within another.

    Args:
        path1: First path.
        path2: Second path.

    Returns:
        True if one path is nested within the other, False otherwise.

    """
    _path1 = path1.rstrip("/") + "/"
    _path2 = path2.rstrip("/") + "/"
    return _path1.startswith(_path2) or _path2.startswith(_path1)


def _get_s3_prefix(s3path: str | s3_client.S3Prefix, *args: str) -> s3_client.S3Prefix:
    prefix = [str(s3path).rstrip("/")]
    if args:
        prefix += [x.strip("/") for x in args]
    return s3_client.S3Prefix("/".join(prefix))


def _get_local_path(localpath: pathlib.Path, *args: str) -> pathlib.Path:
    return pathlib.Path(localpath, *args)


def get_full_path(path: str | s3_client.S3Prefix | pathlib.Path, *args: str) -> s3_client.S3Prefix | pathlib.Path:
    """Get the full path.

    Like os.path.join, but for S3 paths or strings or local paths.

    Args:
        path: Path to get.
        args: Additional arguments.

    Returns:
        The full path.

    """
    if not isinstance(path, pathlib.Path):
        error = f"Path must be a pathlib.Path, got {type(path)}"
        raise TypeError(error)
    return _get_local_path(path, *args)


def _get_objects_from_s3path_relative(
    s3path: s3_client.S3Prefix,
    client: s3_client.S3Client,
    limit: int = 0,
) -> list[str]:
    """Extract list of objects from the input s3path.

    Args:
        s3path: S3 path to get.
        client: S3 client.
        limit: Limit the number of objects to get.

    Returns:
        List of objects.

    """
    s3_prefix = _get_s3_prefix(s3path)

    def func_to_call() -> list[storage_client.StoragePrefix]:
        return client.list_recursive_directory(s3_prefix, limit)

    objects: Sequence[s3_client.S3Prefix] = do_with_retries(func_to_call)  # type: ignore[arg-type]
    s3_prefix_path = str(s3_prefix).rstrip("/") + "/"
    objects_relative = []
    for x in objects:
        if not str(x).startswith(s3_prefix_path):
            continue
        if str(x).endswith("/"):
            logger.warning(f"Skipping empty prefix {x}")
            continue
        objects_relative.append(str(x).replace(s3_prefix_path, ""))
    return sorted(objects_relative)


def _get_files_from_localpath_relative(localpath: pathlib.Path) -> list[str]:
    """Extract list of files from the input localpath.

    Args:
        localpath: Local path to get.

    Returns:
        List of files.

    """
    return sorted([str(x.relative_to(localpath)) for x in localpath.rglob("*") if x.is_file()])


def get_files_relative(path: str, client: s3_client.S3Client | None = None, limit: int = 0) -> list[str]:
    """Get list of files relative to the input path.

    Args:
        path: Path to get.
        client: S3 client.
        limit: Limit the number of files to get.

    Returns:
        List of files.

    """
    if is_s3path(path):
        assert client is not None
        return _get_objects_from_s3path_relative(s3_client.S3Prefix(path), client, limit)
    return _get_files_from_localpath_relative(pathlib.Path(path))


def get_directories_relative(path: str, client: s3_client.S3Client | None = None) -> list[str]:
    """Get list of directories relative to the input path.

    Args:
        path: Path to get.
        client: S3 client.

    Returns:
        List of directories.

    """
    objects = get_files_relative(path, client)
    dirs = set()
    for obj in objects:
        dirs.add(obj.split("/")[0])
    return sorted(dirs)


def get_next_file(
    filename: str,
    filetype: str,
    output_path: str,
    client: s3_client.S3Client | None = None,
) -> s3_client.S3Prefix | pathlib.Path:
    """Get the next file.

    Args:
        filename: Filename to get.
        filetype: Filetype to get.
        output_path: Output path to get.
        client: S3 client.

    Returns:
        The next file.

    """
    next_index = 0
    existing_files = get_files_relative(output_path, client)
    for item in existing_files:
        match = re.match(rf"{filename}_(\d+)\.{filetype}", item)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)
    next_file_name = f"{filename}_{next_index}.{filetype}"
    return get_full_path(output_path, next_file_name)


def backup_file(
    dest: s3_client.S3Prefix | pathlib.Path,
    client: s3_client.S3Client | None,
) -> None:
    """Backup a file.

    Args:
        dest: Destination to backup.
        client: S3 client.

    """
    if isinstance(dest, s3_client.S3Prefix):
        assert client is not None
        if client.object_exists(dest):

            def func_to_call() -> None:
                idx = 1
                data = client.download_object_as_bytes(dest)
                while True:
                    bak_dest = s3_client.S3Prefix(str(dest) + f".bak{idx}")
                    if not client.object_exists(bak_dest):
                        break
                    idx += 1
                client.upload_bytes(bak_dest, data)

            do_with_retries(func_to_call)
    elif dest.exists():
        idx = 1
        while True:
            bak_dest = dest.with_suffix(f"{dest.suffix}.bak{idx}")
            if not bak_dest.exists():
                break
            idx += 1
        dest.rename(bak_dest)
