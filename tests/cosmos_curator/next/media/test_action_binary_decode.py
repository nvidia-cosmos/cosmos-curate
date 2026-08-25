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

"""Round-trip and rejection tests for the ACT2 decoder (inverse of the encoder)."""

import json

import numpy as np
import pytest

from cosmos_curator.next.media.action_binary import (
    _HEADER_SIZE,
    ACTION_BINARY_SPEC_BY_DATASET,
    ACTION_BINARY_SPECS,
    MAGIC,
    decode_action_artifact,
    decode_action_bin,
    encode_action_bin,
    read_action_bin_header,
)


def test_every_registered_dataset_resolves_to_a_known_spec() -> None:
    """The dataset-to-spec mapping never points at a spec that does not exist.

    The one place these tests are allowed to read the live registry: this is a
    structural invariant that survives datasets being added or retired, unlike an
    assertion about which names are present.
    """
    assert set(ACTION_BINARY_SPEC_BY_DATASET.values()) <= set(ACTION_BINARY_SPECS)


def _synthetic(spec_name: str, num_frames: int) -> dict[str, np.ndarray]:
    """Build a deterministic action payload matching one registered spec."""
    spec = ACTION_BINARY_SPECS[spec_name]
    counter = 0.0
    data: dict[str, np.ndarray] = {}
    for field in spec.per_frame_fields:
        size = num_frames * field.value_count
        data[field.name] = (counter + np.arange(size, dtype=np.float32)).reshape(num_frames, *field.shape)
        counter += size
    for field in spec.per_clip_fields:
        data[field.name] = (counter + np.arange(field.value_count, dtype=np.float32)).reshape(*field.shape)
        counter += field.value_count
    return data


def _reheader(encoded: bytes, header: object) -> bytes:
    """Re-serialize *header* over the 1024-byte header region of *encoded*.

    Lets a rejection test corrupt exactly one header field while keeping the
    frame/tail body intact, so the decoder fails on the corruption under test and
    not on an unrelated size mismatch.
    """
    body = encoded[_HEADER_SIZE:]
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    prefix = MAGIC + len(header_bytes).to_bytes(4, "little", signed=False)
    return prefix + header_bytes + bytes(_HEADER_SIZE - len(prefix) - len(header_bytes)) + body


def test_decode_round_trips_mecka_with_per_clip_tail(dexterous_dataset: str) -> None:
    """decode(encode(x)) == x for the mecka spec, exercising the per-clip tail."""
    payload = _synthetic("mecka", num_frames=5)
    decoded = decode_action_bin(encode_action_bin(payload, dexterous_dataset))

    assert set(decoded) == set(payload)
    for name, original in payload.items():
        np.testing.assert_array_equal(decoded[name], original)


def test_decode_round_trips_spec_without_tail(non_dexterous_dataset: str) -> None:
    """decode(encode(x)) == x for a spec with no per-clip tail."""
    payload = _synthetic("libero", num_frames=4)
    decoded = decode_action_bin(encode_action_bin(payload, non_dexterous_dataset))

    assert set(decoded) == set(payload)
    for name, original in payload.items():
        np.testing.assert_array_equal(decoded[name], original)


def test_decode_preserves_field_shapes(dexterous_dataset: str) -> None:
    """Per-frame fields decode to (T, *shape); the per-clip field keeps its shape.

    Shapes come from the spec rather than from literals, so widening a field is a
    one-place change instead of a hunt through assertions.
    """
    frames = 7
    spec = ACTION_BINARY_SPECS["mecka"]
    decoded = decode_action_bin(encode_action_bin(_synthetic("mecka", num_frames=frames), dexterous_dataset))

    for field in spec.per_frame_fields:
        assert decoded[field.name].shape == (frames, *field.shape)
    for field in spec.per_clip_fields:
        assert decoded[field.name].shape == tuple(field.shape)


def test_header_reports_spec_name(dexterous_dataset: str) -> None:
    """The header reader exposes the layout family and the source dataset."""
    header = read_action_bin_header(encode_action_bin(_synthetic("mecka", num_frames=2), dexterous_dataset))

    assert header["spec_name"] == "mecka"
    assert header["source_dataset"] == dexterous_dataset


def test_zero_frame_payload_round_trips(non_dexterous_dataset: str) -> None:
    """An empty (0-frame) payload decodes to empty per-frame arrays."""
    spec = ACTION_BINARY_SPECS["libero"]
    decoded = decode_action_bin(encode_action_bin(_synthetic("libero", num_frames=0), non_dexterous_dataset))

    for field in spec.per_frame_fields:
        assert decoded[field.name].shape == (0, *field.shape)


def test_wrong_magic_is_rejected() -> None:
    """A buffer without the ACT2 magic is not a valid payload."""
    with pytest.raises(ValueError, match="magic"):
        decode_action_bin(b"NOPE" + bytes(2000))


def test_truncated_payload_is_rejected(dexterous_dataset: str) -> None:
    """A payload shorter than the header describes is rejected, not mis-sliced."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=5), dexterous_dataset)
    with pytest.raises(ValueError, match="truncated"):
        decode_action_bin(encoded[:-16])


def test_too_short_for_header_is_rejected() -> None:
    """A buffer shorter than the 1024-byte header cannot be a payload."""
    with pytest.raises(ValueError, match="too short"):
        decode_action_bin(b"ACT2" + bytes(100))


def test_oversized_json_length_is_rejected() -> None:
    """A JSON-length prefix larger than the header capacity is rejected up front.

    Guards the framing before any JSON parse: a corrupt length must not be used
    to slice past the fixed 1024-byte header window.
    """
    oversized_len = (2000).to_bytes(4, "little", signed=False)
    with pytest.raises(ValueError, match="header capacity"):
        decode_action_bin(MAGIC + oversized_len + bytes(2000))


def test_non_json_header_is_rejected() -> None:
    """A header region that is not valid JSON is reported as such (not an opaque crash)."""
    prefix = MAGIC + (5).to_bytes(4, "little", signed=False)
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_action_bin(prefix + b"{bad}" + bytes(2000))


def test_non_object_header_is_rejected(dexterous_dataset: str) -> None:
    """A header whose JSON is a list, not an object, is rejected before field access."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    with pytest.raises(ValueError, match="must be a JSON object"):
        decode_action_bin(_reheader(encoded, ["not", "an", "object"]))


def test_unsupported_version_is_rejected(dexterous_dataset: str) -> None:
    """A header naming a format_version this decoder does not implement is rejected."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        decode_action_bin(_reheader(encoded, header))


def test_missing_required_key_is_rejected(dexterous_dataset: str) -> None:
    """A header missing a field the decoder reads surfaces as ValueError, not KeyError."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    del header["num_frames"]
    with pytest.raises(ValueError, match="missing required key 'num_frames'"):
        decode_action_bin(_reheader(encoded, header))


def test_unknown_dtype_is_rejected(dexterous_dataset: str) -> None:
    """A header dtype numpy cannot resolve is reported as an unknown dtype."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["dtype"] = "not-a-dtype"
    with pytest.raises(ValueError, match="unknown dtype"):
        decode_action_bin(_reheader(encoded, header))


def test_self_inconsistent_offsets_are_rejected(dexterous_dataset: str) -> None:
    """A per_clip_offset that does not equal frame_data_offset + body size is rejected.

    Catches the corruption class that would otherwise decode into silently
    mis-aligned arrays: the tail must begin exactly where the frame body ends.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_clip_offset"] = int(header["per_clip_offset"]) + 4
    with pytest.raises(ValueError, match="self-inconsistent"):
        decode_action_bin(_reheader(encoded, header))


def test_per_frame_field_spilling_past_the_record_is_rejected(dexterous_dataset: str) -> None:
    """A per-frame field whose span exceeds the record stride is rejected.

    Catches a corrupt byte_offset that would slice a per-frame field past the
    end of its fixed record and read into the next frame's bytes.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_frame_fields"][0]["byte_offset"] = int(header["frame_record_size_bytes"])
    with pytest.raises(ValueError, match="past the"):
        decode_action_bin(_reheader(encoded, header))


def test_zero_itemsize_dtype_is_rejected(dexterous_dataset: str) -> None:
    """A zero-itemsize dtype is rejected as ValueError, not a ZeroDivisionError.

    ``decode`` divides ``frame_record_size_bytes`` by the dtype itemsize; a
    ``V0`` void dtype would divide by zero. That ZeroDivisionError escapes the
    reader's per-row ``ValueError`` drop handler, so the header validator must
    reject it up front instead.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["dtype"] = "V0"
    with pytest.raises(ValueError, match="zero itemsize"):
        decode_action_bin(_reheader(encoded, header))


def test_same_width_wrong_dtype_is_rejected(dexterous_dataset: str) -> None:
    """A resolvable dtype the format does not pack is rejected, not honored.

    ``"<i4"`` has the same 4-byte itemsize as the packed float32, so every
    stride, offset and length check passes; honoring it would reinterpret the
    float payload as int32 and decode to silent garbage instead of failing.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["dtype"] = "<i4"
    with pytest.raises(ValueError, match="is not the"):
        decode_action_bin(_reheader(encoded, header))


def test_frame_record_size_not_multiple_of_itemsize_is_rejected(dexterous_dataset: str) -> None:
    """A frame_record_size_bytes that is not a whole number of dtype items is rejected.

    A non-integral record stride would slice the frame body at a fractional
    element boundary; reject it before the decode's integer division silently
    truncates the stride.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["frame_record_size_bytes"] = int(header["frame_record_size_bytes"]) + 1
    with pytest.raises(ValueError, match="whole number"):
        decode_action_bin(_reheader(encoded, header))


def test_misaligned_per_frame_byte_offset_is_rejected(dexterous_dataset: str) -> None:
    """A per-frame byte_offset that is not a multiple of the dtype itemsize is rejected.

    decode divides byte_offset by itemsize to index the frame body; a misaligned
    offset would truncate silently and read the wrong slice.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_frame_fields"][0]["byte_offset"] = 1
    with pytest.raises(ValueError, match="not aligned"):
        decode_action_bin(_reheader(encoded, header))


def test_misaligned_per_clip_byte_offset_is_rejected(dexterous_dataset: str) -> None:
    """A per-clip byte_offset that is not a multiple of the dtype itemsize is rejected."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_clip_fields"][0]["byte_offset"] = 1
    with pytest.raises(ValueError, match="not aligned"):
        decode_action_bin(_reheader(encoded, header))


def test_negative_per_frame_byte_offset_is_rejected(dexterous_dataset: str) -> None:
    """A negative per-frame byte_offset is rejected even when it is dtype-aligned.

    ``-itemsize`` passes the modulo alignment check (``-4 % 4 == 0``) but would
    index the frame body from its tail, so the validator must reject it as
    negative before that check.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    itemsize = int(np.dtype(str(header["dtype"])).itemsize)
    header["per_frame_fields"][0]["byte_offset"] = -itemsize
    with pytest.raises(ValueError, match="negative"):
        decode_action_bin(_reheader(encoded, header))


def test_negative_per_clip_byte_offset_is_rejected(dexterous_dataset: str) -> None:
    """A negative per-clip byte_offset is rejected even when it is dtype-aligned."""
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    itemsize = int(np.dtype(str(header["dtype"])).itemsize)
    header["per_clip_fields"][0]["byte_offset"] = -itemsize
    with pytest.raises(ValueError, match="negative"):
        decode_action_bin(_reheader(encoded, header))


def test_non_mapping_per_frame_field_entry_is_rejected(dexterous_dataset: str) -> None:
    """A per-frame field entry that is not a JSON object is rejected as ValueError.

    A corrupt header could carry a scalar or list where a field object is
    expected; the validator must fail with the documented ValueError rather than
    a TypeError from subscripting a non-mapping.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_frame_fields"][0] = "not-a-mapping"
    with pytest.raises(ValueError, match="not a mapping"):
        decode_action_bin(_reheader(encoded, header))


def test_null_integer_header_field_is_rejected_as_valueerror(dexterous_dataset: str) -> None:
    """A JSON null where an integer header field is required raises ValueError, not TypeError.

    ``int(None)`` raises ``TypeError``, which would escape the module's documented
    ``Raises: ValueError`` contract (and the action leg's per-row ValueError drop
    handler). The header validator must map it to ValueError like every other
    malformed field.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["num_frames"] = None
    with pytest.raises(ValueError, match="num_frames"):
        decode_action_bin(_reheader(encoded, header))


def test_non_iterable_field_shape_is_rejected_as_valueerror(dexterous_dataset: str) -> None:
    """A non-iterable field shape raises ValueError, not a ``prod()`` TypeError.

    ``prod(None)`` raises ``TypeError``; the validator must re-map it to the
    documented ValueError so a corrupt shape stays on the same drop contract as
    every other malformed header field.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_frame_fields"][0]["shape"] = None
    with pytest.raises(ValueError, match="shape"):
        decode_action_bin(_reheader(encoded, header))


@pytest.mark.parametrize(
    "bad_shape",
    [
        [-1],  # negative dimension: prod() would flip the field width's sign
        [1.5],  # non-integer dimension: prod() would yield a fractional width
        [True],  # bool is an int subclass but not a valid dimension
    ],
    ids=["negative", "float", "bool"],
)
def test_invalid_shape_dimension_is_rejected_as_valueerror(dexterous_dataset: str, bad_shape: list[object]) -> None:
    """A shape member that is negative, non-integer, or bool is rejected as ValueError.

    ``prod(shape)`` alone would accept these and hand the decoder a plausible but
    wrong field width (a sign-flipped or fractional byte count) instead of failing
    on the documented malformed-header contract. ``bool`` is an ``int`` subclass,
    so a plain int check would wrongly admit it; it is rejected explicitly.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["per_frame_fields"][0]["shape"] = bad_shape
    with pytest.raises(ValueError, match="invalid shape"):
        decode_action_bin(_reheader(encoded, header))


def test_frame_data_offset_inside_header_is_rejected(dexterous_dataset: str) -> None:
    """A frame_data_offset that points inside the fixed header is rejected.

    decode uses frame_data_offset as the np.frombuffer offset; an offset inside
    the fixed header would read header / JSON bytes as frame records without any
    error, so the validator must reject it up front. The tail and length checks
    do not catch it because a smaller offset only shrinks the expected tail.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=3), dexterous_dataset)
    header = read_action_bin_header(encoded)
    header["frame_data_offset"] = _HEADER_SIZE - 4
    with pytest.raises(ValueError, match="points inside"):
        decode_action_bin(_reheader(encoded, header))


def test_decoded_arrays_are_read_only_views(dexterous_dataset: str) -> None:
    """Decoded arrays are read-only views over the source buffer (no accidental writes).

    They alias the input bytes for zero-copy; a writeable view would let a
    consumer mutate the shared payload buffer under other views of the same span.
    """
    decoded = decode_action_bin(encode_action_bin(_synthetic("mecka", num_frames=4), dexterous_dataset))
    for array in decoded.values():
        assert not array.flags.writeable


def test_decode_action_artifact_exposes_identity_and_arrays(dexterous_dataset: str) -> None:
    """decode_action_artifact returns header identity plus the same arrays as decode_action_bin.

    The artifact is the single-parse path (arrays plus header identity); the thin
    decode_action_bin wrapper must expose exactly its ``arrays``.
    """
    encoded = encode_action_bin(_synthetic("mecka", num_frames=5), dexterous_dataset)
    artifact = decode_action_artifact(encoded)

    assert artifact.spec_name == "mecka"
    assert artifact.source_dataset == dexterous_dataset
    assert artifact.num_frames == 5

    arrays = decode_action_bin(encoded)
    assert set(artifact.arrays) == set(arrays)
    for name, original in arrays.items():
        np.testing.assert_array_equal(artifact.arrays[name], original)
