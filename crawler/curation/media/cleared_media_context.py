"""Resolve safe lineage and governance for cleared preprocessing outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crawler.curation.preprocessing_governance import safe_license_expression
from mmcrawler_datasets.curated.evidence import AssetContextRecord
from preprocessing.privacy.clearance import PrivacyClearance
from preprocessing.privacy.public_provenance import public_source_url

_SAFE_LINEAGE_FIELDS = (
    "fetch_record_id",
    "parent_fetch_record_id",
    "parent_stable_url_id",
    "media_identity",
    "fetch_mode",
    "asset_fetch_mode",
)


@dataclass(frozen=True, slots=True)
class ClearedMediaContext:
    entry: Any
    parent_document: Any | None
    clearance: PrivacyClearance
    license: str | None
    license_url: str | None
    allow_training: bool | None
    governance_note: str | None
    robots_status: str | None
    terms_source: str | None
    usage_rules: str | None

    @property
    def trainable(self) -> bool:
        return bool(
            self.clearance.permits_training
            and self.allow_training is True
            and self.license
        )


def resolve_cleared_media(
    *,
    raw_entries: Iterable[Any],
    documents: Iterable[Any],
    preprocessed: Iterable[Any],
) -> tuple[tuple[Any, ClearedMediaContext], ...]:
    """Match approved preprocessing outputs to non-content raw lineage."""

    entries = _entries_by_source_id(raw_entries)
    parents = _documents_index(documents)
    resolved: list[tuple[Any, ClearedMediaContext]] = []
    for item in preprocessed:
        entry = entries.get(str(item.source_id))
        clearance = item.privacy_clearance
        if (
            entry is None
            or clearance is None
            or not clearance.permits_training
        ):
            continue
        parent = _parent_document(entry=entry, documents=parents)
        resolved.append(
            (
                item,
                _context(
                    entry=entry,
                    parent=parent,
                    item_license=item.license,
                    clearance=clearance,
                ),
            )
        )
    return tuple(resolved)


def safe_asset_context(
    *,
    context: ClearedMediaContext,
    safety_status: str,
) -> AssetContextRecord:
    """Expose privacy evidence and non-content lineage only."""

    record = context.entry.record
    source_page_url = context.clearance.approved_text("source_page_url")
    values = {
        field_name: _optional_text(getattr(record, field_name, None))
        for field_name in _SAFE_LINEAGE_FIELDS
    }
    return AssetContextRecord(
        safety_status=safety_status,
        fetch_record_id=values["fetch_record_id"],
        parent_fetch_record_id=values["parent_fetch_record_id"],
        parent_stable_url_id=values["parent_stable_url_id"],
        media_identity=values["media_identity"],
        fetch_mode=values["fetch_mode"],
        asset_fetch_mode=values["asset_fetch_mode"],
        source_page_url=(
            public_source_url(source_page_url) if source_page_url else None
        ),
        embed_host=_optional_text(
            context.clearance.approved_text("embed_host")
        ),
    )


def project_relative_media_path(
    *,
    media_path: str,
    project_root: Path,
) -> str:
    """Return one existing cleared artifact as a contained project path."""

    root = project_root.resolve(strict=True)
    candidate = Path(media_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError("curated media path escapes project root") from exc
    if not resolved.is_file():
        raise ValueError("curated media path is not a file")
    return relative.as_posix()


def _entries_by_source_id(entries: Iterable[Any]) -> dict[str, Any]:
    return {
        str(entry.record.fetch_record_id): entry
        for entry in entries
        if getattr(entry, "record", None) is not None
    }


def _documents_index(documents: Iterable[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for document in documents:
        source_id = getattr(document, "source_fetch_record_id", None)
        if source_id:
            indexed[f"id:{source_id}"] = document
        for field_name in ("normalized_url", "final_url", "requested_url"):
            key = _url_key(getattr(document, field_name, None))
            if key:
                indexed[f"url:{key}"] = document
    return indexed


def _parent_document(
    *, entry: Any, documents: Mapping[str, Any]
) -> Any | None:
    record = entry.record
    parent_id = getattr(record, "parent_fetch_record_id", None)
    if parent_id:
        parent = documents.get(f"id:{parent_id}")
        if parent is not None:
            return parent
    for value in (
        getattr(record, "parent_url", None),
        getattr(record, "source_page_url", None),
    ):
        key = _url_key(value)
        if key:
            parent = documents.get(f"url:{key}")
            if parent is not None:
                return parent
    return None


def _context(
    *,
    entry: Any,
    parent: Any | None,
    item_license: str | None,
    clearance: PrivacyClearance,
) -> ClearedMediaContext:
    raw = _mapping(getattr(entry.record, "governance", None))
    raw_license = _mapping(raw.get("license"))
    raw_training = _mapping(raw.get("training"))

    allow_training = _optional_bool(raw_training.get("allowed"))
    if allow_training is None and parent is not None:
        allow_training = _optional_bool(parent.allow_training)

    license_value = (
        clearance.approved_text("license")
        or safe_license_expression(item_license)
        or _parent_text(parent, "license")
        or safe_license_expression(raw_license.get("expression"))
    )
    return ClearedMediaContext(
        entry=entry,
        parent_document=parent,
        clearance=clearance,
        license=license_value,
        license_url=clearance.approved_text("license_url")
        or _parent_text(parent, "license_url"),
        allow_training=allow_training,
        governance_note=clearance.approved_text("governance_note")
        or _parent_text(parent, "governance_note"),
        robots_status=clearance.approved_text("robots_status")
        or _parent_text(parent, "robots_status"),
        terms_source=clearance.approved_text("terms_source")
        or _parent_text(parent, "terms_source"),
        usage_rules=clearance.approved_text("usage_rules")
        or _parent_text(parent, "usage_rules"),
    )


def _url_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return public_source_url(value).casefold().rstrip("/")


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _parent_text(parent: Any | None, name: str) -> str | None:
    return _optional_text(getattr(parent, name, None)) if parent else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "allow", "allowed"}:
        return True
    if normalized in {"0", "false", "no", "deny", "denied"}:
        return False
    return None


__all__ = [
    "ClearedMediaContext",
    "project_relative_media_path",
    "resolve_cleared_media",
    "safe_asset_context",
]
