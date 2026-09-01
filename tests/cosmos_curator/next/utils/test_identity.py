# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared stable-identity hashing."""

from cosmos_curator.next.utils.identity import canonical_digest


def test_digest_is_deterministic_for_equal_values() -> None:
    """The same logical value always hashes to the same digest."""
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})


def test_digest_distinguishes_different_values() -> None:
    """Distinct logical values hash to distinct digests."""
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


def test_digest_is_a_hex_sha256() -> None:
    """The digest is a 64-character lowercase hex SHA-256 string."""
    digest = canonical_digest("example")
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_digest_matches_golden_value() -> None:
    """Anchors the digest to a fixed value.

    A change in serialization (separators, key ordering, Unicode handling) fails
    the suite instead of silently invalidating every stored identity that
    depends on it.
    """
    assert canonical_digest({"a": 1, "b": 2}) == "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
