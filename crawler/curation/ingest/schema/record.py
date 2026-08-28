"""Strongly typed view over raw crawl manifest payload rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.path_resolution.project_paths import validate_safe_relative_path
from schemas.versions import (
    RAW_DATASET_SCHEMA_VERSION,
    SUPPORTED_RAW_DATASET_SCHEMA_VERSIONS,
    is_supported_raw_schema_version,
    schema_version_error,
)


@dataclass(frozen=True, slots=True)
class RawManifestRecord:
    """Normalized record parsed from a raw crawl manifest line."""

    schema_version: str
    run_id: str
    fetch_record_id: str
    object_id: str
    requested_url: str
    final_url: str
    normalized_url: str
    parent_url: str | None
    kind: str
    modality: str
    depth: int
    source_type: str
    status_code: int
    content_type: str | None
    mime_type: str | None
    encoding: str | None
    language: str | None
    content_sha256: str
    byte_size: int
    storage_relative_path: str
    domain: str
    path: str
    query: str | None
    extension: str | None
    fetched_at: str
    language_confidence: float | None = None
    language_source: str | None = None
    language_detector_version: str | None = None
    enrichment: dict[str, Any] = field(default_factory=dict)
    fetch_mode: str = "full"
    is_complete_payload: bool = True
    observed_bytes: int = 0
    source_content_length: int | None = None
    parent_fetch_record_id: str | None = None
    parent_stable_url_id: str | None = None
    asset_url: str | None = None
    asset_fetch_mode: str | None = None
    asset_downloaded: bool = False
    asset_metadata_only: bool = False
    context_available: bool = False
    enrichment_available: bool = False
    trainability_reason: str | None = None
    requested_kind: str | None = None
    resolved_kind: str | None = None
    discovery_reason: str | None = None
    selection_reason: str | None = None
    admission_reason: str | None = None
    source_page_url: str | None = None
    embed_url: str | None = None
    embed_host: str | None = None
    parent_title: str | None = None
    parent_text_preview: str | None = None
    media_identity: str | None = None
    asset_rejection_reason: str | None = None
    source_content_type: str | None = None
    fetch_duration_seconds: float | None = None
    payload_sha256: str | None = None
    payload_path: str | None = None
    thumbnail_path: str | None = None
    keyframe_paths: tuple[str, ...] = ()
    transcript_path: str | None = None
    asset_context: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RawManifestRecord:
        """Create a typed record from a manifest row payload."""

        required = _required_manifest_fields(payload=payload)
        kind = _required_text(payload=payload, key="kind")
        modality = _required_text(payload=payload, key="modality")

        return cls(
            schema_version=required.schema_version,
            run_id=required.run_id,
            fetch_record_id=_required_text(
                payload=payload, key="fetch_record_id"
            ),
            object_id=_required_text(payload=payload, key="object_id"),
            requested_url=required.requested_url,
            final_url=required.final_url,
            normalized_url=required.normalized_url,
            parent_url=_as_opt_str(payload.get("parent_url")),
            kind=kind,
            modality=modality,
            depth=_required_int(payload=payload, key="depth"),
            source_type=_required_text(payload=payload, key="source_type"),
            status_code=_required_int(payload=payload, key="status_code"),
            content_type=_as_opt_str(payload.get("content_type")),
            mime_type=_as_opt_str(payload.get("mime_type")),
            encoding=_as_opt_str(payload.get("encoding")),
            language=_as_language_code(payload.get("language")),
            content_sha256=required.content_sha256,
            byte_size=_required_nonnegative_int(
                payload=payload,
                key="byte_size",
            ),
            storage_relative_path=required.storage_relative_path,
            domain=_required_text(payload=payload, key="domain"),
            path=_required_text(payload=payload, key="path"),
            query=_as_opt_str(payload.get("query")),
            extension=_as_opt_str(payload.get("extension")),
            fetched_at=_required_text(payload=payload, key="fetched_at"),
            language_confidence=_as_language_confidence(
                payload.get("language_confidence")
            ),
            language_source=_as_opt_str(payload.get("language_source")),
            language_detector_version=_as_opt_str(
                payload.get("language_detector_version")
            ),
            enrichment=_as_mapping(payload.get("enrichment")),
            fetch_mode=_as_str(payload.get("fetch_mode")) or "full",
            is_complete_payload=_as_bool(
                payload.get("is_complete_payload"),
                default=True,
            ),
            observed_bytes=_required_int(
                payload=payload, key="observed_bytes"
            ),
            source_content_length=_as_optional_int(
                payload.get("source_content_length"),
            ),
            parent_fetch_record_id=_as_opt_str(
                payload.get("parent_fetch_record_id"),
            ),
            parent_stable_url_id=_as_opt_str(
                payload.get("parent_stable_url_id"),
            ),
            asset_url=_as_opt_str(payload.get("asset_url")),
            asset_fetch_mode=_as_opt_str(payload.get("asset_fetch_mode")),
            asset_downloaded=_as_bool(
                payload.get("asset_downloaded"),
                default=False,
            ),
            asset_metadata_only=_as_bool(
                payload.get("asset_metadata_only"),
                default=False,
            ),
            context_available=_as_bool(
                payload.get("context_available"),
                default=False,
            ),
            enrichment_available=_as_bool(
                payload.get("enrichment_available"),
                default=False,
            ),
            trainability_reason=_as_opt_str(
                payload.get("trainability_reason")
            ),
            requested_kind=_as_opt_str(payload.get("requested_kind")),
            resolved_kind=_as_opt_str(payload.get("resolved_kind")),
            discovery_reason=_as_opt_str(payload.get("discovery_reason")),
            selection_reason=_as_opt_str(payload.get("selection_reason")),
            admission_reason=_as_opt_str(payload.get("admission_reason")),
            source_page_url=_as_opt_str(payload.get("source_page_url")),
            embed_url=_as_opt_str(payload.get("embed_url")),
            embed_host=_as_opt_str(payload.get("embed_host")),
            parent_title=_as_opt_str(payload.get("parent_title")),
            parent_text_preview=_as_opt_str(
                payload.get("parent_text_preview")
            ),
            media_identity=_as_opt_str(payload.get("media_identity")),
            asset_rejection_reason=_as_opt_str(
                payload.get("asset_rejection_reason"),
            ),
            source_content_type=_as_opt_str(
                payload.get("source_content_type")
            ),
            fetch_duration_seconds=_as_optional_float(
                payload.get("fetch_duration_seconds"),
            ),
            payload_sha256=_as_opt_str(payload.get("payload_sha256")),
            payload_path=_as_opt_str(payload.get("payload_path")),
            thumbnail_path=_as_opt_str(payload.get("thumbnail_path")),
            keyframe_paths=_as_tuple_text(payload.get("keyframe_paths")),
            transcript_path=_as_opt_str(payload.get("transcript_path")),
            asset_context=_as_mapping(payload.get("asset_context")),
            governance=_as_mapping(payload.get("governance")),
        )


@dataclass(frozen=True, slots=True)
class RequiredManifestFields:
    schema_version: str
    run_id: str
    requested_url: str
    final_url: str
    normalized_url: str
    content_sha256: str
    storage_relative_path: str


def _required_manifest_fields(
    *,
    payload: dict[str, Any],
) -> RequiredManifestFields:
    schema_version = _required_text(payload=payload, key="schema_version")
    if not is_supported_raw_schema_version(schema_version):
        raise ValueError(
            schema_version_error(
                artifact="raw dataset",
                from_version=schema_version,
                supported=SUPPORTED_RAW_DATASET_SCHEMA_VERSIONS,
            )
        )
    if schema_version != RAW_DATASET_SCHEMA_VERSION:
        raise ValueError("raw dataset schema version is not canonical")
    required = RequiredManifestFields(
        schema_version=schema_version,
        run_id=_required_text(payload=payload, key="run_id"),
        requested_url=_required_text(payload=payload, key="requested_url"),
        final_url=_required_text(payload=payload, key="final_url"),
        normalized_url=_required_text(payload=payload, key="normalized_url"),
        content_sha256=_required_sha256(
            payload=payload,
            key="content_sha256",
        ),
        storage_relative_path=validate_storage_relative_path(
            _required_text(payload=payload, key="storage_relative_path")
        ),
    )
    _validate_required_manifest_fields(required=required)
    return required


def _required_text(*, payload: dict[str, Any], key: str) -> str:
    value = _as_str(payload.get(key))
    if not value:
        raise ValueError(f"manifest record missing {key}")
    return value


def _required_int(*, payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"manifest record missing {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest record has invalid {key}") from exc


def _required_nonnegative_int(
    *,
    payload: dict[str, Any],
    key: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"manifest record has invalid {key}; expected non-negative integer"
        )
    return value


def _required_sha256(*, payload: dict[str, Any], key: str) -> str:
    value = _required_text(payload=payload, key=key).lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"manifest record has invalid {key}; expected SHA-256 hex digest"
        )
    return value


def validate_storage_relative_path(value: str) -> str:
    """Return one canonical, platform-independent raw-object path.

    Crawl manifests are portable artifacts and therefore use POSIX path
    separators even when they are consumed on Windows.  Rejecting alternate
    separators here prevents a path from acquiring different semantics on a
    different consumer platform.
    """

    text = str(value).strip()
    if "\\" in text or ":" in text:
        raise ValueError(
            "storage_relative_path must use POSIX separators and contain "
            "no drive or alternate-stream syntax"
        )
    return validate_safe_relative_path(
        text,
        field_name="storage_relative_path",
    )


def _validate_required_manifest_fields(
    *,
    required: RequiredManifestFields,
) -> None:
    missing_fields = [
        field_name
        for field_name, value in (
            ("schema version", required.schema_version),
            ("run id", required.run_id),
            ("requested URL", required.requested_url),
            ("final URL", required.final_url),
            ("normalized URL", required.normalized_url),
            ("content hash", required.content_sha256),
            ("storage path", required.storage_relative_path),
        )
        if not value
    ]
    if missing_fields:
        raise ValueError(f"manifest record missing {missing_fields[0]}")


def _as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _as_str(value: object) -> str:
    return _as_opt_str(value) or ""


def _as_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = _as_int(value, default=-1)
    return parsed if parsed >= 0 else None


def _as_language_code(value: object) -> str | None:
    language = _as_opt_str(value)
    return language.lower() if language is not None else None


def _as_language_confidence(value: object) -> float | None:
    confidence = _as_optional_float(value)
    if confidence is None:
        return None
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("language_confidence must be between 0.0 and 1.0")
    return confidence


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_tuple_text(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))
