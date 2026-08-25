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

r"""Render a URI that arrived as data into an operator-visible log record.

A URI a process READS rather than authors is untrusted input, and two distinct
things follow from that. It may embed a credential or a presigned signature, so
logging it verbatim publishes a secret to whatever aggregator collects the line.
And it may carry a character that ends the record early, so one bad value can
forge a second log line. ``redact_for_log`` answers both at one chokepoint, so
every caller inherits the same decision instead of re-deriving it.

A URI also leaks back out through text the caller did not write - an exception
message quoting the filename it failed on, a subprocess's stderr - so
``redact_diagnostic_for_log`` applies the same secret set to that text. Both
share one definition of what counts as a secret, because two copies of that
decision would drift and the divergence would surface as a leak.

Everything not a secret is KEPT and escaped rather than dropped: naming a
resource the reader can go and look up is the log line's whole diagnostic
purpose, and a URI rendered with characters missing matches nothing in storage
and can collapse two distinct malformed values onto one string.
"""

# Code points a log record must never carry verbatim out of operator-visible
# data: the C0 controls (CR and LF included), DEL, the C1 range, and the two
# Unicode line separators - the whole set ``str.splitlines`` treats as a record
# boundary, plus ESC, which can drive a terminal. Escaped rather than deleted
# because the record's value is naming a resource someone can look up:
# dropping characters would print a URI that matches nothing in storage and
# could collapse two distinct malformed URIs into one string.
#
# The backslash is escaped too, which is what lets the escapes above be read in
# reverse. Without it, a URI carrying the literal six-character text
# "backslash-u-0-0-0-a" and one carrying a real newline render identically, so
# the rendering collapses two distinct URIs onto one string - the same loss the
# escape-rather-than-delete choice exists to prevent.
_LOG_UNSAFE_ESCAPES = str.maketrans(
    {ord("\\"): "\\\\"} | {code: f"\\u{code:04x}" for code in (*range(0x20), 0x7F, *range(0x80, 0xA0), 0x2028, 0x2029)}
)

# The characters every URI parser DELETES from its input, so the structure a
# consumer acts on can differ from the one the raw text shows.
_PARSER_DELETED = str.maketrans(dict.fromkeys("\t\r\n"))


def _secret_spans(url: str) -> tuple[tuple[int, int], ...]:
    """Return the ``(start, end)`` slices of ``url`` that may hold a secret.

    The single place this module decides WHAT counts as a secret, so the two
    renderings below cannot drift apart about it. Ordered last-span-first, so a
    caller excising by index never has to recompute an offset.

    String surgery rather than ``urlparse``, because the parse cannot be trusted
    on a hostile value in two ways beyond the deletion its caller handles. It
    populates ``netloc`` only when the scheme matches, so one unexpected
    character anywhere in the scheme leaves a ``"@" in netloc`` check inspecting
    an empty string while the credential flows straight through. And it RAISES on
    some malformed authorities, which a redactor called from inside an exception
    handler must never do.

    An ``@`` in a URI with no ``//`` at all is kept as path content: a local path
    may legitimately contain one, and no transport reads credentials from a URI
    that has no authority to hold them. Where a ``//`` IS present the first one is
    taken to open the authority, with no check on what precedes it - erring toward
    over-redaction, because any such check is one more thing a crafted value can
    step around. The cost is that a path with a doubled slash followed by a
    segment containing ``@`` loses that segment's prefix.
    """
    end = len(url)
    for marker in ("?", "#"):
        position = url.find(marker)
        if position != -1:
            end = min(end, position)
    spans = [(end, len(url))] if end < len(url) else []
    head = url[:end]
    start = head.find("//")
    if start == -1:
        return tuple(spans)
    authority_start = start + len("//")
    authority_end = head.find("/", authority_start)
    if authority_end == -1:
        authority_end = len(head)
    authority = head[authority_start:authority_end]
    userinfo_end = authority.rfind("@")
    if userinfo_end == -1:
        return tuple(spans)
    spans.append((authority_start, authority_start + userinfo_end + 1))
    return tuple(spans)


def _strip_secrets(url: str) -> str:
    """Return ``url`` without its query, fragment, or authority userinfo."""
    kept = url
    for start, end in _secret_spans(url):
        kept = kept[:start] + kept[end:]
    return kept


def redact_for_log(url: str) -> str:
    """Return ``url`` with userinfo, query, and fragment stripped, escaped for logging.

    A presigned URL carries its signature in the query string and an authority can
    embed credentials, so logging the raw value would publish a secret to whatever
    aggregator collects the record. Everything else is KEPT and escaped rather
    than dropped, on both counts that matter: the value is untrusted, so it must
    not be able to forge a second log record wherever a caller interpolates it,
    and a URI rendered with characters missing would name nothing in storage -
    which is the record's entire diagnostic value.

    Total by construction: callers invoke it from inside per-item failure paths,
    several from within an ``except`` block, so it never raises.
    """
    # Two readings, because a parser's deletions can CREATE an authority the raw
    # text does not show ("s3:/<tab>/user:pw@host" resolves as "s3://user:pw@host").
    # Where they agree, render the verbatim one and keep every character. Where
    # they disagree, the parser's reading is the one under which a credential
    # would be live, so it wins - and the values it merges are exactly the values
    # the transport itself cannot tell apart.
    verbatim = _strip_secrets(url)
    normalized = _strip_secrets(url.translate(_PARSER_DELETED))
    safest = verbatim if verbatim.translate(_PARSER_DELETED) == normalized else normalized
    return safest.translate(_LOG_UNSAFE_ESCAPES)


def redact_diagnostic_for_log(text: str, url: str) -> str:
    """Return ``text`` with ``url``'s secrets removed, escaped for logging.

    For text a caller did not author and cannot restructure - an exception
    message, a subprocess's stderr - which may quote back a URI the caller
    handed in. ``OSError`` does exactly that: it renders the filename it failed
    on, presigned query and all, so interpolating one into a record publishes
    the signature even when the record's own URI field was redacted.

    Removal is by value, not by position, because the secret's offset in ``text``
    is unknown; a span consisting only of its own delimiter is skipped, since
    deleting every ``?`` or ``@`` in a diagnostic would corrupt text that has
    nothing to do with the URI.

    Args:
        text: The message to render. Not assumed to contain ``url`` at all.
        url: The URI whose secrets must not appear, as handed to the operation.

    Returns:
        ``text`` with each of ``url``'s secret components removed and every
        log-unsafe character escaped.

    """
    for start, end in _secret_spans(url):
        secret = url[start:end]
        if len(secret) > 1:
            text = text.replace(secret, "")
    return text.translate(_LOG_UNSAFE_ESCAPES)
