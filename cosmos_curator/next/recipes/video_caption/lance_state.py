# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption field registration, finite-attempt snapshots, and fragment identity."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import lance
from lance.fragment import FragmentMetadata, LanceFragment

from cosmos_curator.next.recipes.video_caption.contracts import CaptionModelSpec, validate_terminal_value
from cosmos_curator.next.recipes.video_split.lance_sink import validate_clip_table
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA
from cosmos_curator.next.utils.identity import canonical_digest

_OWNED_FIELD_COUNT = 2


class FragmentCaptionState(StrEnum):
    """The only two compatible states; every other shape raises."""

    PENDING = "pending"
    COMPLETE = "complete"


@dataclass(frozen=True)
class CaptionAttempt:
    """The finite snapshot and fragment membership selected by one start."""

    version: int
    pending_fragment_ids: tuple[int, ...]
    complete_fragment_ids: tuple[int, ...]

    @property
    def selected_fragment_ids(self) -> tuple[int, ...]:
        """Return every fragment fixed at the start of the attempt."""
        return self.pending_fragment_ids + self.complete_fragment_ids


def ensure_caption_fields(
    uri: str,
    *,
    storage_options: dict[str, str] | None,
    spec: CaptionModelSpec,
    attempts: int,
) -> lance.LanceDataset:
    """Validate or atomically add both owned fields, retrying disjoint schema races."""
    if attempts < 1:
        msg = f"Schema registration attempts must be positive, got {attempts}"
        raise ValueError(msg)
    last_error: Exception | None = None
    for _ in range(attempts):
        dataset = lance.dataset(uri, storage_options=storage_options)
        validate_clip_table(dataset, uri=uri)
        presence = _field_set_presence(dataset, spec)
        if presence == "present":
            return dataset
        if presence == "partial":
            msg = f"Caption field set {spec.caption_field_name!r} is partially present in {uri}"
            raise ValueError(msg)
        try:
            dataset.add_columns(spec.fields)
        except (OSError, RuntimeError) as exc:
            # A disjoint schema commit may have won. Reopen on the next loop;
            # deterministic storage failures still surface after the bounded retries.
            last_error = exc
            continue
        registered = lance.dataset(uri, storage_options=storage_options)
        _assert_present_fields_match(registered, spec)
        return registered
    latest = lance.dataset(uri, storage_options=storage_options)
    if _field_set_presence(latest, spec) == "present":
        return latest
    msg = f"Could not atomically register caption field set {spec.caption_field_name!r} on {uri}"
    raise RuntimeError(msg) from last_error


def validate_caption_input(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Validate the split schema, required values, and global ``clip_id`` uniqueness."""
    validate_clip_table(dataset, uri=uri)
    query = (
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT clip_id) AS clip_ids, "
        "SUM(CASE WHEN clip_id IS NULL THEN 1 ELSE 0 END) AS null_clip_ids, "
        "SUM(CASE WHEN clip_uri IS NULL THEN 1 ELSE 0 END) AS null_clip_uris, "
        "SUM(CASE WHEN clip_size_bytes IS NULL THEN 1 ELSE 0 END) AS null_clip_sizes FROM dataset"
    )
    batches = dataset.sql(query).build().to_batch_records()
    rows = [row for batch in batches for row in batch.to_pylist()]
    if len(rows) != 1:
        msg = f"Could not validate canonical clip identities in {uri}"
        raise RuntimeError(msg)
    stats = rows[0]
    row_count = int(stats["rows"])
    distinct = int(stats["clip_ids"])
    null_counts = [int(stats[name] or 0) for name in ("null_clip_ids", "null_clip_uris", "null_clip_sizes")]
    if distinct != row_count:
        msg = f"Canonical clip_id values in {uri} are not globally unique ({distinct} distinct for {row_count} rows)"
        raise ValueError(msg)
    if any(null_counts):
        msg = f"Canonical clip_id, clip_uri, and clip_size_bytes values in {uri} must all be non-null"
        raise ValueError(msg)


def capture_attempt(
    dataset: lance.LanceDataset,
    *,
    spec: CaptionModelSpec,
    digest: str,
) -> CaptionAttempt:
    """Classify every fragment of one pinned snapshot."""
    pending: list[int] = []
    complete: list[int] = []
    for fragment in dataset.get_fragments():
        state = classify_fragment(fragment, spec=spec, digest=digest)
        target = pending if state is FragmentCaptionState.PENDING else complete
        target.append(int(fragment.fragment_id))
    return CaptionAttempt(
        version=int(dataset.version),
        pending_fragment_ids=tuple(pending),
        complete_fragment_ids=tuple(complete),
    )


def classify_fragment(
    fragment: LanceFragment,
    *,
    spec: CaptionModelSpec,
    digest: str,
) -> FragmentCaptionState:
    """Return pending/complete or reject mixed, stale, and corrupt values."""
    scanner = fragment.scanner(columns=[spec.caption_field_name, spec.metadata_field_name])
    saw_pending = False
    saw_complete = False
    for batch in scanner.to_batches():
        captions = batch.column(spec.caption_field_name).to_pylist()
        metadata_values = batch.column(spec.metadata_field_name).to_pylist()
        for caption, metadata in zip(captions, metadata_values, strict=True):
            if caption is None and metadata is None:
                saw_pending = True
            else:
                try:
                    validate_terminal_value(caption, metadata, spec=spec, digest=digest)
                except (TypeError, ValueError) as exc:
                    msg = f"Fragment {fragment.fragment_id} has incompatible or corrupt caption state: {exc}"
                    raise ValueError(msg) from exc
                saw_complete = True
            if saw_pending and saw_complete:
                msg = f"Fragment {fragment.fragment_id} mixes pending and terminal caption rows"
                raise ValueError(msg)
    # Lance does not normally retain empty fragments. Treat one as complete: it
    # contains no pending work and update_columns must never be invoked for it.
    return FragmentCaptionState.PENDING if saw_pending else FragmentCaptionState.COMPLETE


def fragment_metadata_json(metadata: FragmentMetadata) -> str:
    """Serialize physical fragment metadata canonically."""
    return json.dumps(metadata.to_json(), allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fragment_fingerprint(metadata: FragmentMetadata) -> str:
    """Fingerprint the complete serialized physical fragment identity."""
    return canonical_digest(metadata.to_json())


def protected_binding_fingerprint(metadata: FragmentMetadata, protected_field_ids: set[int]) -> str:
    """Fingerprint row layout and files binding split/caption-owned leaves."""
    raw: Any = metadata.to_json()
    if not isinstance(raw, dict):
        msg = "FragmentMetadata.to_json() returned a non-object value"
        raise TypeError(msg)
    protected_files = []
    for raw_file in raw.get("files", []):
        if not isinstance(raw_file, dict):
            continue
        fields = {int(value) for value in raw_file.get("fields", [])}
        if fields & protected_field_ids:
            protected_files.append(raw_file)
    identity = {
        "id": raw.get("id"),
        "physical_rows": raw.get("physical_rows"),
        "deletion_file": raw.get("deletion_file"),
        "overlays": raw.get("overlays"),
        "row_id_meta": raw.get("row_id_meta"),
        "files": protected_files,
    }
    return canonical_digest(identity)


def leaf_field_ids(dataset: lance.LanceDataset, field_names: tuple[str, ...]) -> set[int]:
    """Return Lance IDs for primitive leaves beneath the named top-level fields."""
    ids: set[int] = set()
    fields_by_name = {field.name(): field for field in dataset.lance_schema.fields()}
    for name in field_names:
        field = fields_by_name[name]
        ids.update(_leaf_ids(field))
    return ids


def protected_field_ids(dataset: lance.LanceDataset, spec: CaptionModelSpec) -> set[int]:
    """Return every split- or selected-caption-owned physical leaf ID."""
    names = (*CLIP_SCHEMA.names, spec.caption_field_name, spec.metadata_field_name)
    return leaf_field_ids(dataset, names)


def _leaf_ids(field: Any) -> set[int]:  # noqa: ANN401 -- pylance's LanceField is not exposed in its type stubs
    children = field.children()
    if not children:
        return {int(field.id())}
    ids: set[int] = set()
    for child in children:
        ids.update(_leaf_ids(child))
    return ids


def _field_set_presence(dataset: lance.LanceDataset, spec: CaptionModelSpec) -> str:
    names = set(dataset.schema.names)
    found = sum(name in names for name in (spec.caption_field_name, spec.metadata_field_name))
    if found == 0:
        return "absent"
    if found == _OWNED_FIELD_COUNT:
        _assert_present_fields_match(dataset, spec)
        return "present"
    return "partial"


def _assert_present_fields_match(dataset: lance.LanceDataset, spec: CaptionModelSpec) -> None:
    mismatched = [
        expected.name
        for expected in spec.fields
        if expected.name not in dataset.schema.names
        or not dataset.schema.field(expected.name).equals(expected, check_metadata=True)
    ]
    if mismatched:
        msg = f"Existing caption field set {spec.caption_field_name!r} is incompatible: {', '.join(mismatched)}"
        raise ValueError(msg)
