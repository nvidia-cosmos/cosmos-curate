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
"""Tests for cosmos_curator.core.utils.storage.presigned_s3_zip."""

import argparse
import contextlib
import io
import math
import os
import socketserver
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from cosmos_curator.core.utils.storage.presigned_s3_zip import (
    _SERVER_INPUT_WORKSPACE_ATTR,
    _SERVER_OUTPUT_WORKSPACE_ATTR,
    _create_zip_archive,
    _download_and_extract_zip_single_node,
    _get_output_path,
    _validate_archive_size,
    _validate_upload_completion,
    _write_split_metadata,
    cleanup_server_input_workspace,
    cleanup_server_output_workspace,
    gather_and_upload_outputs,
    handle_presigned_urls,
    validate_local_input_paths,
    zip_and_upload_directory,
    zip_and_upload_directory_multipart,
)


def _server_owned_args(**kwargs: str) -> argparse.Namespace:
    """Build args as if handle_presigned_urls had created and recorded this workspace.

    ``kwargs`` must include exactly one of ``output_clip_path``/``output_path``; that
    same value is recorded as this request's server-owned workspace, matching what
    ``_use_server_output_workspace`` does for a real request.
    """
    workspace = kwargs.get("output_clip_path") or kwargs.get("output_path")
    args = argparse.Namespace(**kwargs)
    setattr(args, _SERVER_OUTPUT_WORKSPACE_ATTR, workspace)
    return args


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def _serve(handler_cls: type[BaseHTTPRequestHandler]) -> Iterator[HTTPServer]:
    """Spin up a simple threaded HTTP server for the duration of the context."""
    server: HTTPServer = _ThreadedHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def test_create_zip_archive_success(tmp_path: Path) -> None:
    """Create an archive from nested files and validate contents."""
    src_dir = tmp_path / "data"
    src_dir.mkdir()
    (src_dir / "top.txt").write_text("root file", encoding="utf-8")
    nested = src_dir / "nested"
    nested.mkdir()
    (nested / "inner.bin").write_bytes(b"\x00\x01\x02")

    archive_path = _create_zip_archive(str(src_dir))

    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        assert {"top.txt", "nested/", "nested/inner.bin"} == names
        assert zf.read("top.txt") == b"root file"
        assert zf.read("nested/inner.bin") == b"\x00\x01\x02"

    archive_path.unlink(missing_ok=True)


def test_create_zip_archive_invalid_directory(tmp_path: Path) -> None:
    """Raise when attempting to zip a directory that does not exist."""
    missing_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        _create_zip_archive(str(missing_dir))


def test_download_and_extract_zip_single_node_returns_inner_directory(tmp_path: Path) -> None:
    """Return the single top-level directory when extracting."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("single_dir/file.txt", "payload")
    zip_bytes = buf.getvalue()

    class DownloadHandler(BaseHTTPRequestHandler):
        payload: ClassVar[bytes] = zip_bytes

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(DownloadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
        extracted = _download_and_extract_zip_single_node(url, tmp_dir=str(tmp_path))

    extracted_path = Path(extracted)
    assert extracted_path.name == "single_dir"
    assert (extracted_path / "file.txt").read_text(encoding="utf-8") == "payload"


def test_download_and_extract_zip_single_node_multiple_top_level(tmp_path: Path) -> None:
    """Return extraction dir when multiple top-level entries exist."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root_a.txt", "a")
        zf.writestr("root_b.txt", "b")
    zip_bytes = buf.getvalue()

    class DownloadHandler(BaseHTTPRequestHandler):
        payload: ClassVar[bytes] = zip_bytes

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(DownloadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
        extracted = _download_and_extract_zip_single_node(url, tmp_dir=str(tmp_path))

    extracted_path = Path(extracted)
    assert extracted_path.name == "extracted"
    assert sorted(p.name for p in extracted_path.iterdir()) == ["root_a.txt", "root_b.txt"]


def test_zip_and_upload_directory_round_trip(tmp_path: Path) -> None:
    """Zip a directory, upload it, and validate round-trip."""
    src_dir = tmp_path / "upload"
    src_dir.mkdir()
    file_a = src_dir / "alpha.txt"
    file_b = src_dir / "beta" / "nested.bin"
    file_b.parent.mkdir()
    file_a.write_text("alpha", encoding="utf-8")
    expected_bytes = os.urandom(64)
    file_b.write_bytes(expected_bytes)

    class UploadHandler(BaseHTTPRequestHandler):
        received: ClassVar[bytes | None] = None

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            type(self).received = body
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(UploadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/upload"
        zip_and_upload_directory(str(src_dir), url)

    assert UploadHandler.received is not None
    with zipfile.ZipFile(io.BytesIO(UploadHandler.received)) as zf:
        assert set(zf.namelist()) == {"beta/", "beta/nested.bin", "alpha.txt"}
        assert zf.read("alpha.txt") == b"alpha"
        assert zf.read("beta/nested.bin") == expected_bytes


def test_write_split_metadata_skips_all_captions_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presigned split uploads should not rebuild aggregate captions unless opted in."""
    called = False

    def fake_write_all_window_captions(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip._write_all_window_captions",
        fake_write_all_window_captions,
    )

    _write_split_metadata(
        argparse.Namespace(input_video_path="/input", write_all_caption_json=False),
        "/output",
    )

    assert called is False


def test_write_split_metadata_rebuilds_all_captions_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive all-captions option should reach the presigned metadata rewrite path."""
    called = False

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.get_storage_client",
        lambda *args, **_kwargs: f"client:{args[0]}",
    )
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.get_files_relative",
        lambda *_args, **_kwargs: ["video.mp4"],
    )

    def fake_write_all_window_captions(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip._write_all_window_captions",
        fake_write_all_window_captions,
    )

    _write_split_metadata(
        argparse.Namespace(input_video_path="/input", write_all_caption_json=True),
        "/output",
    )

    assert called is True


def test_handle_presigned_urls_maps_annotate_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Annotate presigned input ZIPs should populate input_image_path."""
    extracted_path = str(tmp_path / "extracted_images")
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.download_and_extract_zip",
        lambda _url: extracted_path,
    )

    args = argparse.Namespace(input_presigned_s3_url="https://example.test/input.zip")

    result = handle_presigned_urls("annotate", args)

    assert result is args
    assert args.input_image_path == extracted_path


def test_handle_presigned_urls_records_input_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The extract dir must be recorded so validate_local_input_paths can recognize it."""
    extracted_path = str(tmp_path / "extracted_videos")
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.download_and_extract_zip",
        lambda _url: extracted_path,
    )

    args = argparse.Namespace(input_presigned_s3_url="https://example.test/input.zip")
    handle_presigned_urls("split", args)

    assert getattr(args, _SERVER_INPUT_WORKSPACE_ATTR) == extracted_path


def test_cleanup_server_input_workspace_removes_extract_dir(tmp_path: Path) -> None:
    """The presigned-input extract dir must be removed once the pipeline is done with it."""
    extract_dir = tmp_path / "input_videos_abc"
    extract_dir.mkdir()
    (extract_dir / "video1.mp4").write_bytes(b"")

    args = argparse.Namespace()
    setattr(args, _SERVER_INPUT_WORKSPACE_ATTR, str(extract_dir))

    cleanup_server_input_workspace(args)

    assert not extract_dir.exists()


def test_cleanup_server_input_workspace_is_a_noop_without_presigned_input(tmp_path: Path) -> None:
    """A request that never used input_presigned_s3_url has nothing to clean up."""
    unrelated_dir = tmp_path / "some_other_dir"
    unrelated_dir.mkdir()

    cleanup_server_input_workspace(argparse.Namespace(input_video_path=str(unrelated_dir)))

    assert unrelated_dir.exists()


def test_handle_presigned_urls_creates_annotate_output_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Annotate presigned outputs should use a temporary output_path when omitted."""
    output_path = str(tmp_path / "output_annotate_abc")

    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix == "output_annotate_"
        return output_path

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.tempfile.mkdtemp",
        fake_mkdtemp,
    )
    args = argparse.Namespace(output_presigned_s3_url="https://example.test/output.zip")

    result = handle_presigned_urls("annotate", args)

    assert result is args
    assert args.output_path == output_path


def test_handle_presigned_urls_ignores_caller_supplied_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller-supplied output_clip_path must not survive when a presigned URL is set."""
    workspace = str(tmp_path / "output_split_abc")

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.tempfile.mkdtemp",
        lambda *, prefix: workspace,  # noqa: ARG005
    )
    args = argparse.Namespace(
        output_clip_path="/var/secrets",
        output_presigned_s3_url="https://example.test/output.zip",
    )

    result = handle_presigned_urls("split", args)

    assert result is args
    assert args.output_clip_path == workspace
    assert args.output_clip_path != "/var/secrets"


def test_get_output_path_supports_annotate(tmp_path: Path) -> None:
    """Annotate presigned uploads should read output_path like semantic-dedup."""
    output_path = str(tmp_path / "annotate-output")
    args = _server_owned_args(output_path=output_path)

    assert _get_output_path("annotate", args) == output_path


def test_get_output_path_rejects_path_outside_server_workspace() -> None:
    """A path outside the server temp dir must not be returned for zipping/uploading."""
    args = argparse.Namespace(output_clip_path="/var/secrets")

    assert _get_output_path("split", args) is None


def test_get_output_path_rejects_a_different_requests_workspace(tmp_path: Path) -> None:
    """A path under the shared system temp dir, but not *this* request's, must be rejected.

    ``/tmp`` is shared by every concurrent request; checking only "is this under the
    system temp dir" would accept another request's leftover directory just as
    readily as an attacker-chosen one.
    """
    another_requests_workspace = tmp_path / "output_split_someone_else"
    another_requests_workspace.mkdir()
    args = argparse.Namespace(output_clip_path=str(another_requests_workspace))
    # No _SERVER_OUTPUT_WORKSPACE_ATTR recorded for *this* request at all.

    assert _get_output_path("split", args) is None


def test_get_output_path_rejects_mismatch_between_recorded_and_current_path(tmp_path: Path) -> None:
    """A request's own recorded workspace does not authorize a *different* real dir.

    Even if output_clip_path is reassigned after handle_presigned_urls ran -- to
    another real, existing temp dir, not an attacker fantasy path -- it must not be
    accepted just because *some* legitimate workspace was recorded for this request.
    """
    this_requests_workspace = tmp_path / "output_split_mine"
    this_requests_workspace.mkdir()
    a_different_real_dir = tmp_path / "output_split_not_mine"
    a_different_real_dir.mkdir()

    args = argparse.Namespace(output_clip_path=str(a_different_real_dir))
    setattr(args, _SERVER_OUTPUT_WORKSPACE_ATTR, str(this_requests_workspace))

    assert _get_output_path("split", args) is None


def test_gather_and_upload_outputs_skips_upload_for_path_outside_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/var/secrets must never reach gather/zip/upload, even if it somehow ends up in args."""
    called: dict[str, str] = {}
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.gather_outputs_from_all_nodes",
        lambda path: called.setdefault("gather", path),
    )
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.zip_and_upload_directory",
        lambda path, url: called.setdefault("upload", f"{path}|{url}"),
    )

    gather_and_upload_outputs(
        "split",
        argparse.Namespace(
            output_clip_path="/var/secrets",
            output_presigned_s3_url="https://example.test/output.zip",
        ),
    )

    assert called == {}


def test_gather_and_upload_outputs_cleans_up_recorded_workspace_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded workspace that no longer matches output_clip_path must still be removed.

    _get_output_path() rejects the mismatch and returns None -- which used to mean
    gather_and_upload_outputs returned before its cleanup ever ran, leaking the real
    on-disk directory that handle_presigned_urls actually created.
    """
    recorded_workspace = tmp_path / "output_split_recorded"
    recorded_workspace.mkdir()
    (recorded_workspace / "partial.mp4").write_bytes(b"")
    diverged_path = tmp_path / "output_split_diverged"
    diverged_path.mkdir()

    called: dict[str, str] = {}
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.gather_outputs_from_all_nodes",
        lambda path: called.setdefault("gather", path),
    )

    args = argparse.Namespace(
        output_clip_path=str(diverged_path),
        output_presigned_s3_url="https://example.test/output.zip",
    )
    setattr(args, _SERVER_OUTPUT_WORKSPACE_ATTR, str(recorded_workspace))

    gather_and_upload_outputs("split", args)

    assert called == {}  # never reached gather/zip/upload
    assert not recorded_workspace.exists()  # but the real workspace is still cleaned up


def test_gather_and_upload_outputs_cleans_annotate_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporary annotate output dirs should be removed after presigned upload."""
    output_path = tmp_path / "output_annotate_abc"
    output_path.mkdir()
    (output_path / "summary.json").write_text("{}", encoding="utf-8")
    called: dict[str, str] = {}

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.gather_outputs_from_all_nodes",
        lambda path: called.setdefault("gather", path),
    )
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.zip_and_upload_directory",
        lambda path, url: called.setdefault("upload", f"{path}|{url}"),
    )

    gather_and_upload_outputs(
        "annotate",
        _server_owned_args(
            output_path=str(output_path),
            output_presigned_s3_url="https://example.test/output.zip",
        ),
    )

    assert called["gather"] == str(output_path)
    assert called["upload"] == f"{output_path}|https://example.test/output.zip"
    assert not output_path.exists()


def test_gather_and_upload_outputs_raises_upload_failure_and_cleans_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload failures should be visible to callers while temp dirs are still cleaned."""
    output_path = tmp_path / "output_annotate_abc"
    output_path.mkdir()
    (output_path / "summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.gather_outputs_from_all_nodes",
        lambda _path: None,
    )

    def fail_upload(_path: str, _url: str) -> None:
        msg = "upload failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.zip_and_upload_directory",
        fail_upload,
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        gather_and_upload_outputs(
            "annotate",
            _server_owned_args(
                output_path=str(output_path),
                output_presigned_s3_url="https://example.test/output.zip",
            ),
        )

    assert not output_path.exists()


def test_cleanup_server_output_workspace_removes_orphaned_temp_dir(tmp_path: Path) -> None:
    """A request that never reached gather_and_upload_outputs must not leak its temp dir."""
    workspace = tmp_path / "output_split_abc"
    workspace.mkdir()
    (workspace / "partial.mp4").write_bytes(b"")

    cleanup_server_output_workspace(
        "split",
        _server_owned_args(
            output_clip_path=str(workspace),
            output_presigned_s3_url="https://example.test/output.zip",
        ),
    )

    assert not workspace.exists()


def test_cleanup_server_output_workspace_falls_back_to_recorded_workspace_on_mismatch(
    tmp_path: Path,
) -> None:
    """A recorded workspace that no longer matches output_clip_path must still be removed."""
    recorded_workspace = tmp_path / "output_split_recorded"
    recorded_workspace.mkdir()
    diverged_path = tmp_path / "output_split_diverged"
    diverged_path.mkdir()

    args = argparse.Namespace(
        output_clip_path=str(diverged_path),
        output_presigned_s3_url="https://example.test/output.zip",
    )
    setattr(args, _SERVER_OUTPUT_WORKSPACE_ATTR, str(recorded_workspace))

    cleanup_server_output_workspace("split", args)

    assert not recorded_workspace.exists()
    assert diverged_path.exists()  # not this request's workspace; must be left alone


def test_cleanup_server_output_workspace_ignores_requests_without_presigned_url(tmp_path: Path) -> None:
    """A request that never asked for a presigned output has nothing server-owned to clean up."""
    workspace = tmp_path / "output_split_abc"
    workspace.mkdir()

    cleanup_server_output_workspace("split", argparse.Namespace(output_clip_path=str(workspace)))

    assert workspace.exists()


def test_zip_and_upload_directory_multipart(tmp_path: Path) -> None:
    """Split large uploads across presigned part URLs and reassemble."""
    src_dir = tmp_path / "multipart"
    src_dir.mkdir()
    for idx in range(3):
        (src_dir / f"file_{idx}.bin").write_bytes(os.urandom(2048))

    archive_for_size = _create_zip_archive(str(src_dir))
    archive_bytes = archive_for_size.read_bytes()
    archive_for_size.unlink(missing_ok=True)

    received_parts: list[bytes] = []

    class MultipartHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received_parts.append(body)
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    part_count = 3
    chunk_size = max(1, math.ceil(len(archive_bytes) / part_count))
    assert len(archive_bytes) > chunk_size

    with _serve(MultipartHandler) as server:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        part_urls = [f"{base_url}/part/{idx}" for idx in range(part_count)]
        multipart_config = {"uploadId": "upload", "key": "output.zip", "parts": part_urls}
        zip_and_upload_directory_multipart(str(src_dir), multipart_config, chunk_size_bytes=chunk_size)

    assert len(received_parts) >= 2  # ensure multipart behaviour occurred
    combined = b"".join(received_parts)
    with zipfile.ZipFile(io.BytesIO(combined)) as zf:
        assert set(zf.namelist()) == {
            "file_0.bin",
            "file_1.bin",
            "file_2.bin",
        }
        for idx in range(3):
            assert zf.read(f"file_{idx}.bin") == (src_dir / f"file_{idx}.bin").read_bytes()


def test_validate_archive_size_raises_when_exceeding() -> None:
    """Detect when the archive size exceeds available parts."""
    with pytest.raises(ValueError, match="exceeds maximum expected size"):
        _validate_archive_size(archive_size=301, part_urls=["url1", "url2"], chunk_size_bytes=150)


def test_validate_upload_completion_detects_mismatch() -> None:
    """Detect unfinished uploads when bytes uploaded do not match."""
    with pytest.raises(ValueError, match="Upload size mismatch"):
        _validate_upload_completion(bytes_uploaded=99, archive_size=100)


def test_validate_local_input_paths_rejects_local_path_with_no_extract_dir() -> None:
    """A local path is rejected outright when this request never used input_presigned_s3_url."""
    with pytest.raises(ValueError, match="input_video_path"):
        validate_local_input_paths(argparse.Namespace(input_video_path="/var/secrets"))


@pytest.mark.parametrize("uri", ["s3://bucket/prefix", "az://container/prefix"])
def test_validate_local_input_paths_accepts_remote_paths(uri: str) -> None:
    """BYO media in the caller's own bucket is always allowed, extract dir or not."""
    validate_local_input_paths(argparse.Namespace(input_video_path=uri))


def test_validate_local_input_paths_accepts_this_requests_extract_dir(tmp_path: Path) -> None:
    """The exact directory this request's own presigned input was extracted into is allowed."""
    extract_dir = tmp_path / "input_videos_abc"
    extract_dir.mkdir()
    args = argparse.Namespace(input_video_path=str(extract_dir))
    setattr(args, _SERVER_INPUT_WORKSPACE_ATTR, str(extract_dir))

    validate_local_input_paths(args)  # must not raise


def test_validate_local_input_paths_rejects_a_different_requests_extract_dir(tmp_path: Path) -> None:
    """Having *an* extract dir recorded does not authorize a different real local path."""
    my_extract_dir = tmp_path / "input_videos_mine"
    my_extract_dir.mkdir()
    someone_elses_dir = tmp_path / "input_videos_not_mine"
    someone_elses_dir.mkdir()

    args = argparse.Namespace(input_video_path=str(someone_elses_dir))
    setattr(args, _SERVER_INPUT_WORKSPACE_ATTR, str(my_extract_dir))

    with pytest.raises(ValueError, match="input_video_path"):
        validate_local_input_paths(args)


def test_validate_local_input_paths_covers_input_video_list_json_path() -> None:
    """input_video_list_json_path gets the same treatment: local rejected, remote allowed."""
    with pytest.raises(ValueError, match="input_video_list_json_path"):
        validate_local_input_paths(argparse.Namespace(input_video_list_json_path="/etc/passwd"))

    validate_local_input_paths(  # must not raise
        argparse.Namespace(input_video_list_json_path="s3://bucket/manifest.json"),
    )


def test_validate_local_input_paths_ignores_absent_args() -> None:
    """A request with none of the input path args set is trivially fine."""
    validate_local_input_paths(argparse.Namespace())  # must not raise
