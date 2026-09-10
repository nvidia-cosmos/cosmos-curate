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

"""Read/write/removal contract for ``next.utils.storage``.

Covers ``write_media`` (including the ``file://`` locations ``artifact_uri``
hands out), ``remove_prefix`` (whole staging tree / prefix), ``remove_object``
(a single durable artifact), and ``read_media_if_present``, on both the local
and the remote branch. The removal and read helpers share the same
degenerate paths every cleanup call relies on: an absent target is a no-op,
and a non-absence failure propagates rather than being reported as a clean
removal. The remote branch is exercised through a fake storage client so no
network or credentials are needed.
"""

import pathlib

import pytest

from cosmos_curator.core.utils.storage.azure_client import AzurePrefix
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.next.utils import storage


class _FakeDeleteClient:
    """Minimal delete-capable storage client recording what it was asked to delete.

    Stands in for an ``S3Client`` / ``AzureClient`` on the remote branch: it
    yields a fixed listing for ``list_recursive_directory`` and remembers every
    ``delete_object`` target so a test can assert the recursive/single delete
    contract without a real backend. The listing is fixed but the requested
    prefix is recorded, because ``deleted`` alone cannot catch a caller that
    swept the wrong subtree - the canned reply comes back whatever prefix it was
    handed, so only ``listed`` distinguishes the two.

    Attributes:
        listed: Prefix of every ``list_recursive_directory`` call, in order.
        deleted: Target of every ``delete_object`` call, in order.

    """

    def __init__(self, listing: list[StoragePrefix], *, present: bool = True) -> None:
        self._listing = listing
        self._present = present
        self.listed: list[StoragePrefix] = []
        self.deleted: list[StoragePrefix] = []

    def list_recursive_directory(self, uri: StoragePrefix, _limit: int = 0) -> list[StoragePrefix]:
        self.listed.append(uri)
        return self._listing

    def object_exists(self, _dest: StoragePrefix) -> bool:
        return self._present

    def delete_object(self, dest: StoragePrefix) -> None:
        self.deleted.append(dest)


class _FakeReadClient:
    """Minimal read client: reports presence and yields fixed bytes for the object.

    Stands in for an ``S3Client`` on the remote read branch. ``object_exists``
    drives the not-found-to-None mapping (a real client returns False on a 404),
    and ``download_object_as_bytes`` returns the stored payload only when present.
    """

    def __init__(self, *, present: bool, data: bytes = b"") -> None:
        self._present = present
        self._data = data

    def object_exists(self, _dest: StoragePrefix) -> bool:
        return self._present

    def download_object_as_bytes(self, _uri: StoragePrefix, _chunk_size_bytes: int = 0) -> bytes:
        return self._data


def test_a_file_uri_destination_is_written_where_it_names(tmp_path: pathlib.Path) -> None:
    """A file:// location must land where it names, not under the scheme.

    ``Path`` reads the scheme as a directory, so the object goes into a relative
    ``file:/`` tree under whatever the worker's cwd was -- silently, so the run
    reports success and the bytes are somewhere else.
    """
    storage.write_media(f"{tmp_path.as_uri()}/nested/out.bin", b"payload")

    assert (tmp_path / "nested" / "out.bin").read_bytes() == b"payload"


def test_the_uri_this_package_hands_out_is_one_it_can_write_to(tmp_path: pathlib.Path) -> None:
    """``artifact_uri`` normalizes a local path to ``file://``; writing must round-trip it."""
    recorded = storage.artifact_uri(str(tmp_path / "clip.mp4"))
    # Checked before writing, not after: a plain path would write fine, so an
    # artifact_uri that stopped returning a URI would leave this test passing
    # for a shape it does not cover.
    assert recorded.startswith("file://")

    storage.write_media(recorded, b"payload")

    assert (tmp_path / "clip.mp4").read_bytes() == b"payload"


def test_remove_prefix_deletes_a_local_directory_tree(tmp_path: pathlib.Path) -> None:
    """A local prefix and everything under it is removed from disk."""
    root = tmp_path / "staging"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "shard.parquet").write_bytes(b"data")

    storage.remove_prefix(str(root))

    assert not root.exists()


def test_remove_prefix_absent_local_target_is_a_noop(tmp_path: pathlib.Path) -> None:
    """Removing a prefix that never existed is a no-op, not an error (re-run path)."""
    storage.remove_prefix(str(tmp_path / "never_written"))


def test_remove_prefix_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style failure surfaces; only table-absence is swallowed."""
    root = tmp_path / "staging"
    root.mkdir()

    def raise_permission_error(*_args: object, **_kwargs: object) -> None:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.shutil, "rmtree", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.remove_prefix(str(root))


def test_remove_prefix_remote_deletes_every_listed_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote branch lists exactly the prefix named and deletes each entry under it."""
    listing: list[StoragePrefix] = [S3Prefix("s3://bucket/run/a"), S3Prefix("s3://bucket/run/b")]
    client = _FakeDeleteClient(listing)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_prefix("s3://bucket/run")

    assert client.listed == [S3Prefix("s3://bucket/run")]
    assert client.deleted == listing


def test_remove_prefix_remote_ignores_per_object_missing_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listed object already gone at delete time is skipped without aborting the sweep.

    The miss sits in the middle of the listing on purpose. A helper that stopped
    at the first absent object would still empty a listing whose last entry is
    the missing one, so only an entry AFTER the miss can distinguish skipping it
    from aborting on it.
    """
    listing: list[StoragePrefix] = [
        S3Prefix("s3://bucket/run/a"),
        S3Prefix("s3://bucket/run/b"),
        S3Prefix("s3://bucket/run/c"),
    ]

    class _ClientMissingOnSecond(_FakeDeleteClient):
        def delete_object(self, dest: StoragePrefix) -> None:
            if dest.path.endswith("run/b"):
                raise FileNotFoundError(dest.path)
            super().delete_object(dest)

    client = _ClientMissingOnSecond(listing)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_prefix("s3://bucket/run")

    assert client.deleted == [listing[0], listing[2]]


def test_remove_prefix_remote_deletes_every_listed_object_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same remote contract holds for az:// prefixes."""
    listing: list[StoragePrefix] = [AzurePrefix("az://container/run/a"), AzurePrefix("az://container/run/b")]
    client = _FakeDeleteClient(listing)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_prefix("az://container/run")

    assert client.listed == [AzurePrefix("az://container/run")]
    assert client.deleted == listing


def test_remove_prefix_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote URI with no delete-capable client is a hard error, not a silent skip."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.remove_prefix("s3://bucket/run")


def test_remove_object_deletes_a_local_file(tmp_path: pathlib.Path) -> None:
    """A single local object is removed while its parent directory is untouched."""
    target = tmp_path / "report.json"
    target.write_text("{}")

    storage.remove_object(str(target))

    assert not target.exists()
    assert tmp_path.exists()


def test_remove_object_absent_local_target_is_a_noop(tmp_path: pathlib.Path) -> None:
    """Removing an object that never existed is a no-op, not an error."""
    storage.remove_object(str(tmp_path / "missing.json"))


def test_remove_object_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style failure on a single-object delete surfaces, not swallowed."""
    target = tmp_path / "report.json"
    target.write_text("{}")

    def raise_permission_error(_self: pathlib.Path, *_args: object, **_kwargs: object) -> None:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.Path, "unlink", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.remove_object(str(target))


def test_remove_object_refuses_a_local_directory(tmp_path: pathlib.Path) -> None:
    """remove_object deletes single objects, never a tree: a directory propagates.

    This is the invariant that distinguishes remove_object from remove_prefix.
    The real stdlib error must surface rather than be swallowed as an absent
    no-op; its exact type is platform-dependent (``IsADirectoryError`` on Linux,
    ``PermissionError`` on macOS), so this pins the shared ``OSError`` contract -
    critically, it is not the ``FileNotFoundError`` the helper tolerates.
    """
    directory = tmp_path / "staging"
    directory.mkdir()

    with pytest.raises(OSError):  # noqa: PT011 - type is platform-dependent; see docstring
        storage.remove_object(str(directory))

    assert directory.exists()


def test_remove_object_remote_absent_target_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote 404 (object_exists False) deletes nothing and raises no error."""
    client = _FakeDeleteClient(listing=[], present=False)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("s3://bucket/run/report.json")

    assert client.deleted == []


def test_remove_object_remote_deletes_the_single_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote branch deletes exactly the one object named, never a listing."""
    client = _FakeDeleteClient(listing=[])
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("s3://bucket/run/report.json")

    assert client.listed == []
    assert len(client.deleted) == 1
    assert client.deleted[0].path.endswith("report.json")


def test_remove_object_remote_deletes_the_single_object_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same single-object delete contract holds for az:// URIs."""
    client = _FakeDeleteClient(listing=[])
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("az://container/run/report.json")

    assert client.listed == []
    assert len(client.deleted) == 1
    assert client.deleted[0].path.endswith("report.json")


def test_remove_object_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote object URI with no delete-capable client is a hard error."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.remove_object("s3://bucket/run/report.json")


def test_read_media_if_present_returns_local_bytes(tmp_path: pathlib.Path) -> None:
    """A present local object yields its exact bytes."""
    target = tmp_path / "report.json"
    target.write_bytes(b'{"schema": 1}')

    assert storage.read_media_if_present(str(target)) == b'{"schema": 1}'


def test_read_media_if_present_absent_local_target_returns_none(tmp_path: pathlib.Path) -> None:
    """An absent local object collapses to None, not a FileNotFoundError."""
    assert storage.read_media_if_present(str(tmp_path / "missing.json")) is None


def test_read_media_if_present_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style read failure surfaces; only absence maps to None."""
    target = tmp_path / "report.json"
    target.write_bytes(b"{}")

    def raise_permission_error(_self: pathlib.Path) -> bytes:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.Path, "read_bytes", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.read_media_if_present(str(target))


def test_read_media_if_present_remote_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote 404 (object_exists False) reads as None without downloading."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=False))

    assert storage.read_media_if_present("s3://bucket/run/report.json") is None


def test_read_media_if_present_remote_returns_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present remote object yields its downloaded bytes."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=True, data=b"payload"))

    assert storage.read_media_if_present("s3://bucket/run/report.json") == b"payload"


def test_read_media_if_present_remote_returns_bytes_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same read contract holds for az:// URIs."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=True, data=b"payload"))

    assert storage.read_media_if_present("az://container/run/report.json") == b"payload"


def test_read_media_if_present_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote URI with no readable storage client is a hard error, not a silent None."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.read_media_if_present("s3://bucket/run/report.json")
