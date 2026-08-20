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

"""Tests for the strict ``multimodal-split`` candidate session input config."""

import pytest
from pydantic import ValidationError

from cosmos_curator.next.recipes.multimodal_split.config import MultimodalSplitInputConfig


def test_prefix_only_mode_defaults_to_no_list_and_no_limit() -> None:
    """The minimal config selects prefix-listing mode."""
    config = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings")

    assert config.input_path_prefix == "s3://example-bucket/recordings"
    assert config.session_id_list_path is None
    assert config.limit is None


def test_trailing_slashes_are_stripped_so_joins_stay_predictable() -> None:
    """Prefixes that differ only by trailing slashes canonicalize to one value."""
    bare = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings")
    slashed = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings///")

    assert slashed.input_path_prefix == bare.input_path_prefix


def test_session_id_list_path_keeps_its_exact_object_name() -> None:
    """A list path names an object, so its trailing characters are never stripped."""
    config = MultimodalSplitInputConfig(
        input_path_prefix="/data/recordings",
        session_id_list_path="/data/sessions.txt",
    )

    assert config.session_id_list_path == "/data/sessions.txt"


def test_unknown_fields_are_rejected() -> None:
    """Typos in a recipe config fail loudly instead of being silently ignored."""
    with pytest.raises(ValidationError, match="recurse"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", recurse=True)


def test_config_is_frozen() -> None:
    """Discovery inputs cannot drift after resolution."""
    config = MultimodalSplitInputConfig(input_path_prefix="/data/recordings")

    with pytest.raises(ValidationError, match="frozen"):
        config.limit = 5


def test_strict_mode_rejects_a_stringly_typed_limit() -> None:
    """A YAML-quoted limit is a config error rather than a silent coercion."""
    with pytest.raises(ValidationError, match="valid integer"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", limit="5")


@pytest.mark.parametrize("limit", [0, -1])
def test_limit_must_be_positive(limit: int) -> None:
    """A non-positive limit would silently discover nothing."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", limit=limit)


@pytest.mark.parametrize("location", ["", "   ", " /data/recordings", "/data/recordings "])
def test_empty_or_padded_locations_are_rejected(location: str) -> None:
    """Whitespace in a location is a config authoring error, not something to trim silently."""
    with pytest.raises(ValidationError):
        MultimodalSplitInputConfig(input_path_prefix=location)


def test_unsupported_scheme_is_rejected() -> None:
    """Only local paths, file:// URIs, and s3:// URIs are discoverable."""
    with pytest.raises(ValidationError, match="Unsupported storage scheme"):
        MultimodalSplitInputConfig(input_path_prefix="gs://example-bucket/recordings")


def test_pathless_file_uri_is_rejected() -> None:
    """A file:// URI without a path would otherwise fall back to the working directory."""
    with pytest.raises(ValidationError, match="Unsupported local file URI"):
        MultimodalSplitInputConfig(input_path_prefix="file://")


def test_unsupported_scheme_is_rejected_for_the_session_id_list() -> None:
    """The list path gets the same scheme validation as the prefix."""
    with pytest.raises(ValidationError, match="Unsupported storage scheme"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="gs://example-bucket/sessions.txt",
        )


def test_a_local_prefix_may_be_paired_with_an_s3_session_id_list() -> None:
    """Mixed storage between the prefix and the list file is explicitly supported."""
    config = MultimodalSplitInputConfig(
        input_path_prefix="/data/recordings",
        session_id_list_path="s3://example-bucket/sessions.txt",
    )

    assert config.session_id_list_path == "s3://example-bucket/sessions.txt"


def test_a_scheme_without_a_location_is_rejected() -> None:
    """A templated prefix with empty variables must fail, not become a relative path.

    Stripping trailing slashes off the whole string would reduce 's3://' to 's3:',
    which is no longer recognized as remote and would silently resolve against the
    driver's working directory.
    """
    with pytest.raises(ValidationError, match="names a scheme but no location"):
        MultimodalSplitInputConfig(input_path_prefix="s3://")


def test_the_file_uri_root_survives_slash_stripping() -> None:
    """file:/// is the filesystem root and must not collapse to the bare scheme."""
    config = MultimodalSplitInputConfig(input_path_prefix="file:///")

    assert config.input_path_prefix == "file:///"


@pytest.mark.parametrize("prefix", ["/", "///"])
def test_the_filesystem_root_survives_slash_stripping(prefix: str) -> None:
    """A local root prefix reduces to a single separator rather than an empty string."""
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == "/"


def test_trailing_slashes_are_stripped_from_the_key_not_the_scheme() -> None:
    """Only the location part of a URI loses its trailing slashes."""
    config = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings///")

    assert config.input_path_prefix == "s3://example-bucket/recordings"


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("file:///", "file:///"),
        ("file://localhost/", "file://localhost/"),
        ("file:///data/recordings///", "file:///data/recordings"),
        ("s3://example-bucket/recordings///", "s3://example-bucket/recordings"),
        ("s3://example-bucket/", "s3://example-bucket"),
    ],
)
def test_only_the_path_component_loses_trailing_slashes(prefix: str, expected: str) -> None:
    """Stripping the whole string would drop the authority along with the slashes.

    ``file://localhost/`` would become ``file://localhost``, which has no path and
    is rejected by the very code meant to read it.
    """
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "file:///data/my#dir",
        "file:///data/my?dir",
        "file://localhost/data/my#dir",
        "s3://example-bucket/my#dir",
        "s3://example-bucket/my?dir",
    ],
)
def test_uris_with_an_unencoded_delimiter_are_rejected(prefix: str) -> None:
    """'#' and '?' are legal directory names but URI delimiters, so they truncate silently.

    ``file:///data/my#dir`` parses to the path ``/data/my``. Accepting it would
    curate a different directory than the one named, without any error.
    """
    with pytest.raises(ValidationError, match="unencoded"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_an_unencoded_delimiter_is_rejected_in_the_session_id_list_path() -> None:
    """The list path truncates the same way and gets the same guard."""
    with pytest.raises(ValidationError, match="unencoded"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="file:///data/lists#2026/sessions.txt",
        )


def test_a_percent_encoded_delimiter_is_the_supported_spelling() -> None:
    """Percent-encoding is how a URI names a directory containing '#'."""
    config = MultimodalSplitInputConfig(input_path_prefix="file:///data/my%23dir")

    assert config.input_path_prefix == "file:///data/my%23dir"


def test_a_plain_local_path_may_contain_a_delimiter_character() -> None:
    """Only URIs reinterpret '#'; a bare path is passed through to the filesystem."""
    config = MultimodalSplitInputConfig(input_path_prefix="/data/my#dir")

    assert config.input_path_prefix == "/data/my#dir"


@pytest.mark.parametrize(
    "prefix",
    [
        "/data/recordings/../other",
        "../recordings",
        "file:///data/latest/..",
        "file:///data/%2E%2E/other",
        "s3://example-bucket/recordings/../other",
    ],
)
def test_parent_directory_segments_are_rejected(prefix: str) -> None:
    """'..' means two different directories depending on who resolves it.

    Listing goes through the kernel, which resolves a symlink before the '..';
    ``session_uri`` is built with ``os.path.abspath``, which cancels the segment
    lexically. A prefix like ``/data/latest/..`` would therefore emit URIs under
    a directory whose children were never listed. Rejecting it at config time is
    the pre-flight failure, before any listing happens.
    """
    with pytest.raises(ValidationError, match=r"'\.\.' segment"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_parent_directory_segments_are_rejected_in_the_session_id_list_path() -> None:
    """The list path gets the same rule as the prefix."""
    with pytest.raises(ValidationError, match=r"'\.\.' segment"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="/data/../sessions.txt",
        )


@pytest.mark.parametrize("prefix", ["/data/..recordings", "/data/recordings..", "/data/.../recordings"])
def test_only_a_whole_parent_segment_is_rejected(prefix: str) -> None:
    """'..' inside a name is an ordinary directory, not a parent reference."""
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == prefix


@pytest.mark.parametrize("prefix", ["s3://Example-Bucket/recordings", "s3://ab/recordings"])
def test_an_invalid_s3_bucket_name_is_rejected_at_config_time(prefix: str) -> None:
    """Discovery builds every S3 call from ``S3Prefix``, so its rules apply here too.

    Constructing one during validation moves an uppercase or too-short bucket
    name from the middle of a run to ``cosmos-curator pipeline validate``. The
    match is on ``S3Prefix``'s own message, so this fails if the check is dropped
    rather than passing on some other validator's rejection.
    """
    with pytest.raises(ValidationError, match="Invalid S3 bucket name"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_an_invalid_s3_object_key_is_rejected_at_config_time() -> None:
    """``S3Prefix`` restricts key characters, and a run would otherwise hit that later."""
    with pytest.raises(ValidationError, match="Invalid S3 object key"):
        MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings/*/raw")


def test_an_s3_session_id_list_path_is_validated_by_the_storage_layer_too() -> None:
    """A bucket-less s3 list path used to be accepted and fail only at read time.

    The prefix field caught this already through slash stripping, but the list
    path skips that step, so ``s3://`` reached ``read_text`` unchallenged.
    """
    with pytest.raises(ValidationError, match="Invalid S3 bucket name"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="s3://",
        )


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r"])
def test_uris_containing_stripped_whitespace_are_rejected(whitespace: str) -> None:
    """``urlsplit`` deletes tab, LF, and CR from a URI outright, per WHATWG.

    ``file:///data/rec<TAB>ord`` canonicalizes to ``file:///data/record``: not a
    truncation but a different directory name, with no error and nothing in the
    output to show a character went missing. All three are legal in POSIX
    filenames and S3 keys.
    """
    with pytest.raises(ValidationError, match="whitespace"):
        MultimodalSplitInputConfig(input_path_prefix=f"file:///data/rec{whitespace}ord")


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r"])
def test_stripped_whitespace_is_rejected_in_an_s3_uri(whitespace: str) -> None:
    """S3 keys carry the same characters and lose them the same way."""
    with pytest.raises(ValidationError, match="whitespace"):
        MultimodalSplitInputConfig(input_path_prefix=f"s3://example-bucket/rec{whitespace}ord")
