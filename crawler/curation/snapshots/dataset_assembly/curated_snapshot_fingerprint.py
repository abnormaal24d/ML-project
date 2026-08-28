"""Snapshot identity fingerprinting for curated dataset assembly.

Fingerprint semantics: ``build_snapshot_fingerprint_payload`` projects only
settings that affect curated output content. Operational settings (e.g.
``builder.fail_on_empty_snapshot``) must not change the digest, otherwise
an unrelated tweak invalidates every existing curated snapshot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from config.collection.processors import ProcessorSettings
from config.preprocessing.settings import PreprocessingSettings
from config.settings.datasets import CuratedDatasetSettings
from config.source_catalog.catalog_settings import SourceProfileSettings

CURATED_FINGERPRINT_OUTPUT_SETTING_GROUPS = (
    "near_deduper",
    "document_chunker",
    "document_assembler",
)


def build_snapshot_fingerprint_payload(
    *,
    source_profile: SourceProfileSettings,
    preprocessing: PreprocessingSettings,
    curation: CuratedDatasetSettings,
    processors: ProcessorSettings,
) -> dict[str, Any]:
    """Return the crawler-owned content-identity for a snapshot.

    The payload is a content-semantic fingerprint: only settings that change
    curated output content are projected. Storage layout settings
    (``datasets.paths``) are deliberately excluded — the lookup location for
    reusable snapshots already encodes where a snapshot lives, so the digest
    must not change when paths are edited. Dead or operational curation
    groups (removed pair assemblers, snapshot reader, writer, builder) are
    likewise excluded so unrelated tweaks do not invalidate existing
    snapshots.
    """

    return {
        "source_profile": source_profile.model_dump(mode="json"),
        "preprocessing": preprocessing.model_dump(mode="json"),
        "curation": {
            group_name: getattr(curation, group_name).model_dump(mode="json")
            for group_name in CURATED_FINGERPRINT_OUTPUT_SETTING_GROUPS
        },
        "collection_processors": processors.model_dump(mode="json"),
    }


# Bound peak memory for fingerprint serialization. Larger values reduce
# Python call overhead; smaller values tighten the JSON peak.
_FINGERPRINT_RECORD_BATCH_SIZE = 256


def build_curation_input_fingerprint(
    *,
    raw_entries: tuple[Any, ...],
    settings_payload: dict[str, Any],
    relevant_kinds: frozenset[str],
) -> str:
    """Hash the canonical curated input fingerprint.

    Produces the same digest as a full-payload ``json.dumps`` + SHA-256 of:

    ``{"raw_entries":[...],"relevant_kinds":[...],"settings":...}``

    with ``sort_keys=True``, ``separators=(",", ":")``, and ``default=str``.

    Records are sorted by the canonical identity key, then serialized in bounded
    batches and fed incrementally to the hasher so peak memory stays
    O(batch_size) instead of O(n) JSON object graphs and buffers.
    """

    ordered = _ordered_fingerprint_entries(raw_entries)
    hasher = hashlib.sha256()

    # Top-level key order matches json.dumps(..., sort_keys=True):
    # raw_entries, relevant_kinds, settings.
    hasher.update(b'{"raw_entries":[')
    _update_hasher_with_record_batches(hasher=hasher, ordered=ordered)
    hasher.update(b'],"relevant_kinds":')
    hasher.update(
        _canonical_json_bytes(sorted(relevant_kinds)),
    )
    hasher.update(b',"settings":')
    hasher.update(_canonical_json_bytes(settings_payload))
    hasher.update(b"}")
    return hasher.hexdigest()


def _ordered_fingerprint_entries(
    raw_entries: tuple[Any, ...],
) -> list[Any]:
    """Sort by the canonical run, fetch-record, and storage-path identity.

    Sorting least-significant key first preserves Python's stable-sort
    semantics without allocating n temporary 3-tuples.
    """

    ordered = list(raw_entries)
    ordered.sort(key=lambda item: item.record.storage_relative_path)
    ordered.sort(key=lambda item: item.record.fetch_record_id)
    ordered.sort(key=lambda item: item.record.run_id)
    return ordered


def _fingerprint_record_projection(entry: Any) -> dict[str, Any]:
    return {
        "run_id": entry.record.run_id,
        "fetch_record_id": entry.record.fetch_record_id,
        "object_id": entry.record.object_id,
        "kind": entry.record.kind,
        "storage_relative_path": entry.record.storage_relative_path,
        "content_sha256": getattr(entry.record, "content_sha256", None),
        "byte_size": entry.record.byte_size,
    }


def _update_hasher_with_record_batches(
    *,
    hasher: Any,
    ordered: list[Any],
) -> None:
    """Serialize record projections in batches and stream bytes into hasher."""

    batch_size = _FINGERPRINT_RECORD_BATCH_SIZE
    first_record = True
    for start in range(0, len(ordered), batch_size):
        batch = [
            _fingerprint_record_projection(entry)
            for entry in ordered[start : start + batch_size]
        ]
        if not batch:
            continue
        # dumps produces "[{...},{...}]"; strip brackets to append into the
        # outer raw_entries array without a full O(n) payload structure.
        serialized_batch = json.dumps(
            batch,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if not (
            serialized_batch.startswith("[") and serialized_batch.endswith("]")
        ):
            raise RuntimeError(
                "fingerprint batch serialization must produce a JSON array"
            )
        inner = serialized_batch[1:-1]
        if not inner:
            continue
        if not first_record:
            hasher.update(b",")
        hasher.update(inner.encode("utf-8"))
        first_record = False


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
