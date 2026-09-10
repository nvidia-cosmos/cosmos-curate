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
"""Characterization tests for this recipe's storage policies, carried over from their predecessor.

Written against the private cloud-source helper module that used to sit in
``cosmos_curator/core/sensors/scripts/``, while it was still unmodified, to pin what it
did before :mod:`cosmos_curator.core.utils.storage` replaced it. That module is now
deleted and these tests have moved here with it, rewired onto
:mod:`cosmos_curator.next.recipes.data_integrity.storage_io` -- so they no longer
describe a retirement plan, they are the suite for the helpers that replaced it. Their
siblings, pinning the ``storage_cli`` half of the same predecessor, live in
``tests/cosmos_curator/core/utils/test_storage_cli.py``.

**How this file is organised, and why.** Everything above the ``BEHAVIOUR`` banner is
mechanism: how a subject is constructed and which SDK entry point is faked. Everything
below it is behaviour: values, exception types, message substrings, call counts and
ordering. The move rewrote the mechanism block and left the behaviour block standing,
which is what shows the replacement preserves behaviour rather than merely compiling.

Seven tests are *stated* differently without asking anything different, because the
sentinel they read is gone. The retired stat type carried an ``is_empty`` property;
:class:`StorageStat` does not, and
:func:`~cosmos_curator.next.recipes.data_integrity.storage_io.object_stat` answers the
same question -- did this lookup learn anything -- by returning ``None`` instead. Those
tests now assert ``is None`` / ``is not None``; the contract they pin is unchanged.

The fakes both suites drive -- the Azure listing and blob shapes, and the two wrappers
that put a fake behind a real storage client -- live in
:mod:`tests.cosmos_curator.core.utils.storage_fakes`, imported below. The ones only
these tests drive are here, which is why the HeadObject and whole-object S3 fakes sit
in this file and the Azure ones do not. Nothing here performs network I/O.
"""

import io
from collections.abc import Callable
from typing import Any, BinaryIO

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils import storage_cli
from cosmos_curator.core.utils.storage.storage_client import StorageStat
from cosmos_curator.core.utils.storage_cli import StorageCliError
from cosmos_curator.next.recipes.data_integrity import storage_io
from tests.cosmos_curator.core.utils.storage_fakes import (
    WHEN,
    FakeAzureContainer,
    FakeAzureService,
    FakeBlobClient,
    FakeBlobProperties,
    azure_client,
    s3_client,
)

# ======================================================================================
# MECHANISM -- subject construction and hand-rolled backend fakes.
# The move to the storage layer rewrote this block. Nothing here asserts anything.
#
# One shape recurs, and it exists so the behaviour block below could stay put: the
# storage layer takes a ``StorageClient``, not a backend SDK client, so the SDK fakes
# are wrapped by ``s3_client`` / ``azure_client``. The fakes themselves are still
# SDK-shaped, because the request each Curator call makes on the wire is exactly what
# these tests exist to pin.
# ======================================================================================

_ObjectStatter = Callable[[str], StorageStat | None]

#: The error code AWS returns for a HeadObject on an absent key.
_S3_NOT_FOUND_CODE = "404"


class _FakeHeadS3:
    """HeadObject stand-in serving one canned answer and counting its calls.

    The call count is what proves a stat issues a single request rather than retrying.
    """

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.head_object_calls: list[dict[str, object]] = []

    def head_object(self, **kwargs: object) -> dict[str, Any]:
        self.head_object_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _FakeTextS3:
    """Whole-object S3 stand-in backed by a dict, recording every operation in order.

    Boto3-shaped rather than storage-shaped, which is what keeps the "no extra
    request" assertion meaningful: the upload runs the real ``S3Client.upload_bytes``,
    so a client built without ``can_overwrite`` would show its existence check here as
    a recorded ``head_object``.
    """

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.operations: list[str] = []

    def upload_fileobj(self, fileobj: BinaryIO, bucket: str, key: str, **_kwargs: object) -> None:
        self.operations.append("put_object")
        self.objects[bucket, key] = fileobj.read()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 - botocore's own parameter names
        self.operations.append("get_object")
        body = self.objects[Bucket, Key]
        # ``ContentLength`` keeps the download on its single-request path; without it
        # the client hands over to the managed transfer and this fake is not one.
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 - botocore's own parameter names
        self.operations.append("head_object")
        if (Bucket, Key) not in self.objects:
            raise _client_error(_S3_NOT_FOUND_CODE)
        return {"ContentLength": len(self.objects[Bucket, Key])}


def _client_error(code: str) -> ClientError:
    """Build the ClientError botocore raises for a HeadObject the store refused."""
    return ClientError({"Error": {"Code": code, "Message": "refused"}}, "HeadObject")


def _s3_statter(
    response: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> tuple[_ObjectStatter, _FakeHeadS3]:
    """Return a stat callable bound to a fake S3 serving one HeadObject answer."""
    fake = _FakeHeadS3(response, error)

    def _stat(source: str) -> StorageStat | None:
        return storage_io.object_stat(source, client=s3_client(fake))

    return _stat, fake


def _azure_statter(
    props: FakeBlobProperties | None = None,
    error: Exception | None = None,
) -> tuple[_ObjectStatter, FakeAzureContainer]:
    """Return a stat callable bound to a fake container serving one properties answer."""
    container = FakeAzureContainer(blob_client=FakeBlobClient(props, error))
    service = FakeAzureService(container)

    def _stat(source: str) -> StorageStat | None:
        return storage_io.object_stat(source, client=azure_client(service))

    return _stat, container


def _s3_text_store(monkeypatch: pytest.MonkeyPatch) -> _FakeTextS3:
    """Aim the text helpers' internally-built client at an in-memory object store.

    The whole-document helpers take no client argument, so the only seam is the factory
    they call -- which they reach through the module, hence the module-level patch.

    ``can_overwrite`` is forwarded rather than swallowed: the substitute has to be the
    client the factory would have built, or a write that stopped asking for overwrite
    permission would still pass the test named for it.
    """
    fake = _FakeTextS3()
    monkeypatch.setattr(
        storage_cli,
        "make_s3_client",
        lambda *_args, can_overwrite=False, **_kwargs: s3_client(fake, can_overwrite=can_overwrite),
    )
    return fake


# ======================================================================================
# BEHAVIOUR -- what the helpers promise their callers. This block survived the move to
# the storage layer, apart from the seven tests that read the retired ``is_empty``
# sentinel and now read ``None`` instead.
# ======================================================================================

# --------------------------------------------------------------------------------------
# 1. storage_io.object_stat
# --------------------------------------------------------------------------------------


def test_a_successful_s3_head_populates_size_etag_and_last_modified() -> None:
    """One HEAD answers all three questions, so enrichment costs no extra round trip.

    The ETag arrives quoted on the wire and is stored unquoted, so a value recorded here
    compares cleanly against one from another client.
    """
    stat_object, fake = _s3_statter({"ContentLength": 1234, "ETag": '"abc123"', "LastModified": WHEN})

    stat = stat_object("s3://bucket/root/a.mp4")

    assert stat == StorageStat(size_bytes=1234, etag="abc123", last_modified=WHEN)
    assert fake.head_object_calls == [{"Bucket": "bucket", "Key": "root/a.mp4"}]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_client_error("404"), id="a-missing-object"),
        pytest.param(_client_error("403"), id="a-forbidden-head"),
        pytest.param(_client_error("NoSuchKey"), id="a-store-specific-missing-key-code"),
        pytest.param(RuntimeError("something else entirely"), id="an-arbitrary-failure"),
    ],
)
def test_a_failed_s3_head_yields_nothing_learned_instead_of_raising(error: Exception) -> None:
    """The lookup must never raise, which is the exact opposite of the storage layer's ``stat``.

    Both callers -- a progress display and a staleness record -- are enrichment, so a failure
    has to degrade to "nothing learned" rather than break the check that was actually asked
    for. The 403 case is the important one: it is what AWS answers for an object that is
    merely absent when the caller lacks ``s3:ListBucket``.

    Restated: "nothing learned" used to be an all-fields-``None`` stat carrying
    ``is_empty``, and is now ``None``. Same contract, read off the return type.
    """
    stat_object, _ = _s3_statter(error=error)

    assert stat_object("s3://bucket/root/a.mp4") is None


def test_a_head_response_carrying_no_fields_yields_nothing_learned() -> None:
    """A backend that reports nothing is indistinguishable from a failure, deliberately.

    Restated as ``is None`` for the reason given above, and this is the assertion that
    pins it: the storage layer does return a stat here, all three fields empty, and
    collapsing that to ``None`` is a decision this helper makes on its callers' behalf.
    """
    stat_object, _ = _s3_statter({})

    assert stat_object("s3://bucket/root/a.mp4") is None


def test_a_zero_byte_object_is_not_reported_as_nothing_learned() -> None:
    """A truncated upload is a fact worth recording, and ``0`` must not read as "no answer"."""
    stat_object, _ = _s3_statter({"ContentLength": 0})

    stat = stat_object("s3://bucket/root/truncated.mp4")

    assert stat is not None
    assert stat.size_bytes == 0


def test_a_successful_azure_properties_lookup_populates_the_same_three_fields() -> None:
    """One ``get_blob_properties`` is the Azure equivalent, unquoted ETag included."""
    stat_object, container = _azure_statter(FakeBlobProperties(99, WHEN, '"deadbeef"'))

    stat = stat_object("az://container/root/a.mp4")

    assert stat == StorageStat(size_bytes=99, etag="deadbeef", last_modified=WHEN)
    assert container.requested_blob == "root/a.mp4"


def test_a_failed_azure_properties_lookup_yields_nothing_learned_instead_of_raising() -> None:
    """The never-raise contract is per-helper, not per-backend."""
    stat_object, _ = _azure_statter(error=RuntimeError("gone"))

    assert stat_object("az://container/root/a.mp4") is None


def test_a_non_cloud_uri_yields_nothing_learned_instead_of_raising() -> None:
    """A local path is not an error here either; it simply has nothing to report."""
    assert storage_io.object_stat("/data/sessions/a.mp4") is None


@pytest.mark.parametrize(
    ("response", "learned_something"),
    [
        pytest.param({}, False, id="nothing-at-all"),
        pytest.param({"ContentLength": 7}, True, id="only-a-size"),
        pytest.param({"ETag": '"abc"'}, True, id="only-an-etag"),
        pytest.param({"LastModified": WHEN}, True, id="only-a-timestamp"),
    ],
)
def test_a_stat_is_returned_only_when_it_learned_something(
    response: dict[str, Any],
    learned_something: bool,  # noqa: FBT001 - a parametrized expectation, not a mode switch
) -> None:
    """Callers must tell a failed HEAD apart from a real but sparse response.

    Passing an empty stat along as if it were a fact would record a null ETag for an object
    that was merely unlucky the first time. Any *one* field is enough to be worth passing on.

    Restated: the retired ``is_empty`` property is now the ``None`` in the return type, so
    the question is asked of the helper's result rather than of a stat built by hand. This
    drives real HEAD responses through the helper, which is a stronger test of the same
    rule -- a sparse stat now has to survive the round trip, not merely exist.
    """
    stat_object, _ = _s3_statter(response)

    assert (stat_object("s3://bucket/root/a.mp4") is not None) is learned_something


# --------------------------------------------------------------------------------------
# 2. storage_io.read_text / storage_io.write_text
# --------------------------------------------------------------------------------------


def test_text_written_to_s3_comes_back_as_the_same_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store's manifest is JSON with operator-supplied strings in it, so non-ASCII must survive."""
    _s3_text_store(monkeypatch)

    storage_io.write_text("s3://bucket/store/manifest.json", '{"note": "héllo — ok"}')

    assert storage_io.read_text("s3://bucket/store/manifest.json") == '{"note": "héllo — ok"}'


def test_text_is_written_as_utf_8_bytes_under_the_key_from_the_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoding is UTF-8 with no BOM, and the bucket/key split is the first slash."""
    fake = _s3_text_store(monkeypatch)

    storage_io.write_text("s3://bucket/store/manifest.json", "héllo")

    assert fake.objects == {("bucket", "store/manifest.json"): "héllo".encode()}


def test_writing_text_overwrites_without_first_checking_for_an_existing_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest is rewritten on every run, so a guarded write would need a second call.

    The recorded operation list is the proof: two PUTs and nothing else, no HEAD and no
    conditional request.
    """
    fake = _s3_text_store(monkeypatch)

    storage_io.write_text("s3://bucket/store/manifest.json", "first")
    storage_io.write_text("s3://bucket/store/manifest.json", "second")

    assert fake.operations == ["put_object", "put_object"]
    assert storage_io.read_text("s3://bucket/store/manifest.json") == "second"


@pytest.mark.parametrize("uri", ["az://container/manifest.json", "/data/manifest.json"])
def test_writing_text_anywhere_but_s3_is_a_storage_cli_error(monkeypatch: pytest.MonkeyPatch, uri: str) -> None:
    """These helpers are S3-only; Azure and local writes are somebody else's job."""
    _s3_text_store(monkeypatch)

    with pytest.raises(StorageCliError, match="expected an s3:// URI"):
        storage_io.write_text(uri, "payload")


@pytest.mark.parametrize("uri", ["az://container/manifest.json", "/data/manifest.json"])
def test_reading_text_from_anywhere_but_s3_is_a_storage_cli_error(monkeypatch: pytest.MonkeyPatch, uri: str) -> None:
    """The read side refuses the same set of URIs as the write side."""
    _s3_text_store(monkeypatch)

    with pytest.raises(StorageCliError, match="expected an s3:// URI"):
        storage_io.read_text(uri)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(storage_io.read_text, id="read_text"),
        pytest.param(lambda uri: storage_io.write_text(uri, "payload"), id="write_text"),
    ],
)
def test_a_bucket_the_prefix_type_rejects_arrives_as_a_storage_cli_error(
    operation: Callable[[str], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected bucket has to reach the operator through the CLIs' own error type.

    ``S3Prefix`` raises a bare ``ValueError``, which travels straight past the
    ``except StorageCliError`` handlers in ``cli`` and ``session_cli`` and lands as a
    traceback. ``object_stat`` already routes through ``storage_prefix``; these two do
    the same, so one module answers the same failure one way. An uppercase name is the
    cheapest trigger -- the bucket regex is the only lowercase-only one of the pair,
    ``AzurePrefix`` matching case-insensitively -- though what the rule *should* be is
    CVC-1257's question, not this test's.
    """
    _s3_text_store(monkeypatch)

    with pytest.raises(StorageCliError, match="malformed s3 URI"):
        operation("s3://MyBucket/store/manifest.json")
