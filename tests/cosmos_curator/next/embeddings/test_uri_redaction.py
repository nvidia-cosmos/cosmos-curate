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

r"""Contract of this module's two redactions, pinned where the redaction is defined.

It removes exactly three things - userinfo, query, fragment - and escapes
everything else. Two properties make it worth this much test surface, and each
has a matching failure mode that a naive implementation walks into:

- It must not DROP a character. Asserting a separator is merely *absent* from the
  output cannot tell escaping from deletion, and a deleted character prints a URI
  that matches nothing in storage and collapses two distinct malformed URIs into
  one string. So every escape case asserts an exact test-owned expected string.
- The secret removal must not depend on the value PARSING. ``urlparse`` populates
  ``netloc`` only when the scheme matches, so a userinfo check keyed on it stops
  firing for any value with an odd character in the scheme - while the transport,
  which ignores those characters, still resolves the credential. So the escape
  cases are swept across POSITIONS (before the scheme, inside it, in the
  authority, in the path), not only across characters.

The invariant those two properties add up to: no value from which ANY parser could
read authority userinfo may render that userinfo, and nothing else may be dropped.
An ``@`` that no parser reads as userinfo is a filename and is kept.

The escapes are written out here rather than derived from the production table, so
a change to that table reddens a test instead of silently agreeing with itself.

``redact_diagnostic_for_log`` carries the same invariant into text the caller did
not author, where the secret's position is unknown - so it removes by value, and
the cases below pin that it removes the secret without charging unrelated text
for the punctuation the secret happened to use.
"""

import pytest

from cosmos_curator.next.embeddings.uri_redaction import redact_diagnostic_for_log, redact_for_log

# Every character ``str.splitlines`` treats as a record boundary, plus tab, paired
# with the escape the redaction must emit for it. Tab is not a boundary; it is
# here because it is one of the three characters ``urlparse`` deletes from its
# input, which makes it the sharpest probe of whether a parse sits in the path.
_RECORD_BOUNDARIES = (
    ("\t", r"\u0009"),
    ("\n", r"\u000a"),
    ("\x0b", r"\u000b"),
    ("\x0c", r"\u000c"),
    ("\r", r"\u000d"),
    ("\x1c", r"\u001c"),
    ("\x1d", r"\u001d"),
    ("\x1e", r"\u001e"),
    ("\x7f", r"\u007f"),
    ("\x85", r"\u0085"),
    ("\u2028", r"\u2028"),
    ("\u2029", r"\u2029"),
)

# Characters that break scheme recognition without being control characters, so
# ``urlparse`` reports no authority for a URI that carries a live credential.
# Ordinary text, which is what makes them the cheapest way to reach the defect.
_SCHEME_BREAKERS = ("_", " ", "%", "!")


@pytest.mark.parametrize(("boundary", "escaped"), _RECORD_BOUNDARIES)
def test_redaction_escapes_a_record_boundary_in_a_secret_free_uri(boundary: str, escaped: str) -> None:
    """A URI with nothing to strip keeps its whole path with the boundary escaped."""
    rendered = redact_for_log(f"/data/action/a{boundary}b.bin")

    assert rendered == f"/data/action/a{escaped}b.bin"
    assert len(rendered.splitlines()) == 1


@pytest.mark.parametrize(("boundary", "escaped"), _RECORD_BOUNDARIES)
def test_redaction_escapes_a_record_boundary_in_a_secret_bearing_uri(boundary: str, escaped: str) -> None:
    """Stripping a query does not cost the path a character: the boundary is escaped, not dropped."""
    rendered = redact_for_log(f"s3://bucket/action/a{boundary}b.bin?sig=deadbeef")

    assert rendered == f"s3://bucket/action/a{escaped}b.bin"
    assert len(rendered.splitlines()) == 1


@pytest.mark.parametrize(("boundary", "escaped"), _RECORD_BOUNDARIES)
@pytest.mark.parametrize("template", ["{c}s3://bucket/a.bin?sig=x", "s3{c}://bucket/a.bin?sig=x"])
def test_redaction_escapes_a_record_boundary_in_the_scheme(template: str, boundary: str, escaped: str) -> None:
    """A boundary before or inside the scheme is escaped like any other.

    The position that a parse-based implementation loses: ``urlparse`` strips
    leading C0 characters and deletes tab / CR / LF outright, so a value rebuilt
    from its components renders several distinct inputs as one string.
    """
    rendered = redact_for_log(template.format(c=boundary))

    assert rendered == template.format(c=escaped).removesuffix("?sig=x")
    assert len(rendered.splitlines()) == 1


@pytest.mark.parametrize("odd", [*[pair[0] for pair in _RECORD_BOUNDARIES], *_SCHEME_BREAKERS])
@pytest.mark.parametrize("template", ["{c}s3://key:s3cret@host/a.bin?sig=x", "s3{c}://key:s3cret@host/a.bin?sig=x"])
def test_redaction_strips_userinfo_whatever_sits_in_the_scheme(template: str, odd: str) -> None:
    """No character in the scheme may defeat the userinfo stripping.

    The credential is live regardless of the wrapper it arrives in, so a redaction
    whose secret detection depends on the value parsing cleanly publishes it for
    every input the parser happens to reject - which is most of them.
    """
    rendered = redact_for_log(template.format(c=odd))

    assert "s3cret" not in rendered
    assert len(rendered.splitlines()) == 1


def test_redaction_renders_two_uris_differing_only_by_a_boundary_position_differently() -> None:
    """Two distinct malformed URIs must not render as the same string.

    Naming a resource someone can look up is the log's whole purpose, so a
    rendering that collapses distinct values defeats it even while staying safe.
    """
    first = redact_for_log("s3://bucket/x\n/y.bin?sig=1")
    second = redact_for_log("s3://bucket/x/\ny.bin?sig=1")

    assert first != second


def test_redaction_strips_userinfo_from_an_authority_only_a_parser_can_see() -> None:
    """An authority that exists only after the parser deletes a character is still stripped.

    ``s3:/<tab>/user:pw@host`` carries no ``//`` as written, but every URI parser
    deletes tab, so the transport resolves it as ``s3://user:pw@host`` and the
    credential is live. Surgery on the raw text alone cannot see that authority,
    which is why the rendering reconciles both readings.
    """
    assert redact_for_log("s3:/\t/key:s3cret@host/a.bin") == "s3://host/a.bin"


def test_redaction_keeps_an_at_sign_that_no_parser_reads_as_userinfo() -> None:
    """An ``@`` in a path is kept: it is a filename, not a credential.

    A URI with no authority has nowhere for a transport to read credentials from,
    so redacting around every ``@`` would destroy legitimate local paths to guard
    a value that is already inert.
    """
    assert redact_for_log("/data/action/user@host/a.bin") == "/data/action/user@host/a.bin"


def test_redaction_does_not_raise_on_a_malformed_authority() -> None:
    """A URI the standard parser rejects is still rendered, not raised on.

    Every caller invokes this from inside a per-row failure path - two of them
    from within an ``except`` block - so raising here would convert one unreadable
    payload into a failed batch.
    """
    assert redact_for_log("s3://key:s3cret@[::1/a.bin") == "s3://[::1/a.bin"


def test_redaction_keeps_a_plain_uri_intact() -> None:
    """A URI carrying no secret is returned whole - naming the resource IS the log's value."""
    assert redact_for_log("s3://bucket/action/a.bin") == "s3://bucket/action/a.bin"


def test_redaction_strips_a_presigned_query() -> None:
    """A query string is dropped: a presigned URL carries its signature there."""
    assert redact_for_log("https://bucket.example/action/a.bin?sig=deadbeef") == "https://bucket.example/action/a.bin"


def test_redaction_strips_a_fragment() -> None:
    """A fragment is dropped on its own, without a query or userinfo to trigger the branch."""
    assert redact_for_log("https://bucket.example/action/a.bin#tail") == "https://bucket.example/action/a.bin"


def test_redaction_strips_netloc_userinfo() -> None:
    """Embedded credentials are dropped on their own, keeping the host and path.

    Tested without a query so the ``@``-in-netloc arm of the branch is exercised
    on its own: bundled with a query, a regression that stopped checking the
    netloc would still take the branch and still strip the userinfo.
    """
    rendered = redact_for_log("https://key:s3cret@bucket.example/action/a.bin")

    assert rendered == "https://bucket.example/action/a.bin"


def test_redaction_distinguishes_a_literal_escape_from_the_character_it_denotes() -> None:
    r"""A URI containing the text ``\u000a`` renders differently from one containing a newline.

    The rendering is only worth its escapes if they can be read in reverse. With
    the backslash left unescaped both inputs print as ``...a\u000ab.bin``, so an
    operator handed the record cannot tell which of two real, distinct artifacts
    to go and look for - the same collapse the escapes exist to prevent.
    """
    literal = redact_for_log(r"s3://bucket/a\u000ab.bin")
    control = redact_for_log("s3://bucket/a\nb.bin")

    assert literal != control


def test_diagnostic_redaction_strips_a_presigned_query_quoted_back_by_the_message() -> None:
    """A signature the message quoted from the URI is removed, keeping the rest of the text.

    ``OSError`` renders the filename it failed on, so a message interpolated
    beside an already-redacted URI field reintroduces the secret the field
    dropped.
    """
    url = "s3://bucket/action/a.bin?X-Amz-Signature=deadbeef"

    rendered = redact_diagnostic_for_log(f"[Errno 2] No such file or directory: '{url}'", url)

    assert rendered == "[Errno 2] No such file or directory: 's3://bucket/action/a.bin'"


def test_diagnostic_redaction_strips_userinfo_quoted_back_by_the_message() -> None:
    """Credentials in the URI's authority are removed from the quoted text."""
    url = "s3://key:s3cret@bucket/action/a.bin"

    rendered = redact_diagnostic_for_log(f"HeadObject failed for {url}", url)

    assert rendered == "HeadObject failed for s3://bucket/action/a.bin"


def test_diagnostic_redaction_escapes_a_record_boundary_in_the_message() -> None:
    r"""A boundary anywhere in the message is escaped, whatever put it there.

    The message is not the caller's own text, so it can end the record early
    independently of the URI - a decoder that echoes a byte it rejected reaches
    the log without passing through the URI field at all.
    """
    rendered = redact_diagnostic_for_log("ValueError: bad byte\nWARNING forged line", "s3://bucket/a.bin")

    assert rendered == r"ValueError: bad byte\u000aWARNING forged line"


def test_diagnostic_redaction_keeps_a_question_mark_the_uri_did_not_contribute() -> None:
    """A URI with an empty query does not cost the message its own punctuation.

    The secret is removed by value, so a span holding nothing but its delimiter
    would otherwise match every such character in unrelated text.
    """
    rendered = redact_diagnostic_for_log("is the object there? unknown", "s3://bucket/a.bin?")

    assert rendered == "is the object there? unknown"


def test_diagnostic_redaction_leaves_a_message_that_never_quoted_the_uri() -> None:
    """A message with no URI in it is returned as written.

    The common case for a decode failure: it is handed bytes and never learns
    the URI, so removal must be a no-op rather than a guess at what to cut.
    """
    rendered = redact_diagnostic_for_log("ValueError: ACT2 payload too short: 19 bytes", "s3://k:s3cret@b/a.bin?sig=x")

    assert rendered == "ValueError: ACT2 payload too short: 19 bytes"
