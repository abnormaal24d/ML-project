"""Dataset record schema and creation for raw dataset manifest entries.

Strict: extra="forbid", governance required, and no discarded fields.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from crawler.discovery.task_identity import discovered_task_identity_from_parts
from crawler.storage.datasets.records.governance import (
    RecordGovernance,
    create_record_governance,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime
    from pathlib import Path

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.domains.domain_governance_registry import (
        DomainGovernanceRegistry,
    )
    from crawler.governance.processing_activity import (
        ProcessingActivityRegistry,
    )


class DatasetRecord(BaseModel):
    """Serializable record for manifest entries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    item_id: str
    fetch_record_id: str
    stable_url_id: str
    object_id: str
    run_id: str

    requested_url: str
    final_url: str
    normalized_url: str
    domain: str
    path: str
    query: str | None
    extension: str | None
    parent_url: str | None
    referrer_url: str | None
    kind: str
    modality: str
    depth: int
    source_type: str
    fetch_attempt: int
    status_code: int
    content_type: str | None
    mime_type: str | None
    encoding: str | None
    language: str | None
    content_sha256: str
    byte_size: int
    storage_relative_path: str
    http_etag: str | None
    http_last_modified: str | None
    fetched_at: str

    language_confidence: float | None = None
    language_source: str | None = None
    language_detector_version: str | None = None
    category: str | None = None
    relevance_score: float | None = None
    content_signature: str | None = None
    enrichment: dict[str, object] = {}
    record_version: int = 1
    previous_fetch_record_id: str | None = None
    status: str = "fetched"
    metadata: dict[str, object] = {}
    fetch_mode: str = "full"
    is_complete_payload: bool = True
    observed_bytes: int | None = None
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
    asset_context: dict[str, object] = {}
    alignment_group_id: str | None = None

    governance: RecordGovernance


def derive_media_identity(
    *,
    task: CrawlTask,
    normalized_url: str,
    kind: str,
) -> str | None:
    """Return the stable identity used to deduplicate asset records."""

    if kind not in {"image", "audio", "video", "document"}:
        return None
    return discovered_task_identity_from_parts(
        url=normalized_url,
        kind=kind,
        source_type=task.source_type,
    )


def derive_fetch_record_id(
    *,
    run_id: str,
    record_identity: str,
    record_version: int,
) -> str:
    """Return a versioned record id for one logical stored representation."""

    digest = hashlib.sha256(
        (f"{run_id}|{record_identity}|{record_version}").encode("utf-8"),
    ).hexdigest()
    return digest[:24]


def derive_stable_url_id(*, normalized_url: str) -> str:
    return hashlib.sha256(
        normalized_url.encode("utf-8"),
    ).hexdigest()[:24]


class DatasetRecordCreator:
    """Create raw dataset manifest records from persisted fetch results."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        run_id: str,
        now: Callable[[], datetime],
        governance_registry: DomainGovernanceRegistry | None = None,
        processing_activity_registry: ProcessingActivityRegistry | None = None,
        processing_activity_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._run_id = run_id
        self._now = now
        self._governance_registry = governance_registry
        self._processing_activity_registry = processing_activity_registry
        self._processing_activity_id = processing_activity_id

    def create(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        byte_size: int,
        content_sha256: str,
        relative_path: Path,
        normalized_url: str,
        kind: str,
        modality: str,
        previous_record: DatasetRecord | None,
        enrichment: Mapping[str, object] | None,
    ) -> DatasetRecord:
        """Create one manifest record for a persisted crawler payload."""

        record_version = _resolve_record_version(
            previous_record=previous_record,
        )
        normalized_enrichment = _normalize_enrichment(enrichment)
        asset_context = (
            task.context.to_dict() if task.context is not None else {}
        )
        parsed_url = urlparse(result.final_url)

        _validate_asset_parentage(task=task, kind=kind)

        payload = result.payload
        fetch_mode = (
            str(payload.fetch_mode or "full")
            if payload is not None
            else "metadata_only"
        )
        is_complete_payload = (
            bool(payload.is_complete_payload) if payload is not None else False
        )
        observed_bytes = (
            int(payload.observed_bytes)
            if payload is not None and payload.observed_bytes is not None
            else (int(payload.byte_size) if payload is not None else 0)
        )
        source_content_length = (
            int(payload.source_content_length)
            if payload is not None
            and payload.source_content_length is not None
            else None
        )
        fetch_duration_seconds = _payload_duration_seconds(payload=payload)

        source_page_url = (
            _text(asset_context.get("source_page_url")) or task.parent_url
        )
        media_identity = _text(
            asset_context.get("media_identity")
        ) or derive_media_identity(
            task=task,
            normalized_url=normalized_url,
            kind=kind,
        )
        asset_fetch_mode = _resolve_asset_fetch_mode(
            fetch_mode=fetch_mode,
            asset_context=asset_context,
        )
        embed_url = _text(asset_context.get("embed_url"))
        if embed_url is None and asset_fetch_mode == "embed_metadata":
            embed_url = result.final_url
        context_available = _context_available(asset_context=asset_context)
        trainability_evidence = _trainability_evidence(
            asset_context=asset_context,
            enrichment=normalized_enrichment,
        )

        discovery_reason = _text(asset_context.get("discovery_reason"))
        selection_reason = _text(asset_context.get("selection_reason"))
        admission_reason = (
            _text(asset_context.get("admission_reason")) or "accepted"
        )
        embed_host = _text(asset_context.get("embed_host"))
        parent_title = _text(asset_context.get("parent_title"))
        parent_text_preview = _text(asset_context.get("parent_text_preview"))
        asset_rejection_reason = _text(
            asset_context.get("asset_rejection_reason")
        )
        thumbnail_path = _text(asset_context.get("thumbnail_path"))
        keyframe_paths = _text_items(asset_context.get("keyframe_paths"))
        transcript_path = _text(asset_context.get("transcript_path"))

        trainability_reason = _trainability_reason(
            asset_fetch_mode=asset_fetch_mode,
            asset_context=asset_context,
            enrichment=normalized_enrichment,
        )

        governance = create_record_governance(
            registry=self._governance_registry,
            task=task,
            result=result,
            content_hash=content_sha256,
            domain=getattr(parsed_url, "hostname", None) or "",
            asset_context=asset_context,
            enrichment=normalized_enrichment,
            run_id=self._run_id,
            now=self._now(),
            processing_activity_registry=self._processing_activity_registry,
            processing_activity_id=self._processing_activity_id,
        )

        metadata = _build_record_metadata(
            task=task,
            result=result,
            kind=kind,
            source_page_url=source_page_url,
            media_identity=media_identity,
            asset_context=asset_context,
            normalized_enrichment=normalized_enrichment,
            trainability_evidence=trainability_evidence,
            fetch_mode=fetch_mode,
            is_complete_payload=is_complete_payload,
            observed_bytes=observed_bytes,
            source_content_length=source_content_length,
            fetch_duration_seconds=fetch_duration_seconds,
        )

        parent_stable_url_id = (
            derive_stable_url_id(normalized_url=task.parent_url)
            if task.parent_url
            else None
        )

        return DatasetRecord(
            schema_version=self._settings.raw_schema_version,
            item_id=content_sha256,
            fetch_record_id=derive_fetch_record_id(
                run_id=self._run_id,
                record_identity=media_identity or normalized_url,
                record_version=record_version,
            ),
            stable_url_id=derive_stable_url_id(normalized_url=normalized_url),
            object_id=content_sha256,
            run_id=self._run_id,
            requested_url=task.url,
            final_url=result.final_url,
            normalized_url=normalized_url,
            domain=getattr(parsed_url, "hostname", None) or "",
            path=getattr(parsed_url, "path", None) or "/",
            query=getattr(parsed_url, "query", None) or None,
            extension=_path_extension(relative_path=relative_path),
            parent_url=task.parent_url,
            referrer_url=task.parent_url,
            kind=kind,
            modality=modality,
            depth=task.depth,
            source_type=task.source_type,
            fetch_attempt=1,
            status_code=result.status_code,
            content_type=result.content_type,
            mime_type=result.mime_type,
            encoding=result.encoding,
            language=result.language,
            language_confidence=result.language_confidence,
            language_source=result.language_source,
            language_detector_version=result.language_detector_version,
            content_sha256=content_sha256,
            byte_size=byte_size,
            storage_relative_path=relative_path.as_posix(),
            http_etag=_read_response_header(result.headers, "etag"),
            http_last_modified=_read_response_header(
                result.headers,
                "last-modified",
            ),
            fetched_at=result.fetched_at,
            category=result.category,
            relevance_score=result.relevance_score,
            content_signature=result.content_signature,
            enrichment=dict(normalized_enrichment),
            record_version=record_version,
            previous_fetch_record_id=(
                previous_record.fetch_record_id
                if previous_record is not None
                else None
            ),
            status="fetched",
            metadata=metadata,
            fetch_mode=fetch_mode,
            is_complete_payload=is_complete_payload,
            observed_bytes=observed_bytes,
            source_content_length=source_content_length,
            parent_fetch_record_id=None,
            parent_stable_url_id=parent_stable_url_id,
            asset_url=(
                result.final_url
                if kind in {"image", "audio", "video", "document"}
                else None
            ),
            asset_fetch_mode=asset_fetch_mode,
            asset_downloaded=_asset_downloaded(
                payload=payload,
                asset_fetch_mode=asset_fetch_mode,
            ),
            asset_metadata_only=_asset_metadata_only(
                asset_fetch_mode=asset_fetch_mode,
            ),
            context_available=context_available,
            enrichment_available=bool(normalized_enrichment),
            trainability_reason=trainability_reason,
            requested_kind=task.kind,
            resolved_kind=kind,
            discovery_reason=discovery_reason,
            selection_reason=selection_reason,
            admission_reason=admission_reason,
            source_page_url=source_page_url,
            embed_url=embed_url,
            embed_host=embed_host,
            parent_title=parent_title,
            parent_text_preview=parent_text_preview,
            media_identity=media_identity,
            asset_rejection_reason=asset_rejection_reason,
            source_content_type=result.content_type,
            fetch_duration_seconds=fetch_duration_seconds,
            payload_sha256=result.body_sha256 or content_sha256,
            payload_path=relative_path.as_posix(),
            thumbnail_path=thumbnail_path,
            keyframe_paths=keyframe_paths,
            transcript_path=transcript_path,
            asset_context=asset_context,
            alignment_group_id=_resolve_alignment_group_id(
                media_identity=media_identity,
                parent_stable_url_id=parent_stable_url_id,
                kind=kind,
            ),
            governance=governance,
        )


def _resolve_record_version(
    *,
    previous_record: DatasetRecord | None,
) -> int:
    if previous_record is None:
        return 1
    return previous_record.record_version + 1


def _normalize_enrichment(
    enrichment: Mapping[str, object] | None,
) -> dict[str, object]:
    if enrichment is None:
        return {}
    return {str(key): value for key, value in enrichment.items()}


def _validate_asset_parentage(*, task: CrawlTask, kind: str) -> None:
    if kind not in {"image", "audio", "video", "document"}:
        return
    if task.source_type not in {"embedded_asset", "feed_enclosure"}:
        return
    if task.parent_url:
        return
    raise ValueError(
        "asset records from embedded/feed discovery require parent linkage"
    )


def _payload_duration_seconds(*, payload: object) -> float | None:
    if payload is None:
        return None
    value = getattr(payload, "duration_seconds", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_asset_fetch_mode(
    *,
    fetch_mode: str,
    asset_context: Mapping[str, object],
) -> str:
    context_mode = _text(asset_context.get("asset_fetch_mode"))
    if context_mode == "embed_metadata":
        return "embed_metadata"

    normalized = fetch_mode.strip().lower()
    if normalized in {
        "metadata_only",
        "metadata_probe",
        "head_only_oversized",
        "partial_probe_failed_fallback_head_only",
    }:
        return "metadata_only"
    if normalized in {"partial", "fetch_partial"}:
        return "metadata_only"
    return "full_payload"


def _asset_downloaded(
    *,
    payload: object,
    asset_fetch_mode: str,
) -> bool:
    return bool(
        payload is not None
        and asset_fetch_mode == "full_payload"
        and getattr(payload, "byte_size", 0) > 0
    )


def _asset_metadata_only(*, asset_fetch_mode: str) -> bool:
    return asset_fetch_mode in {
        "metadata_only",
        "embed_metadata",
        "skipped_too_large",
        "failed_fetch",
    }


def _context_available(
    *,
    asset_context: Mapping[str, object],
) -> bool:
    return any(
        _text(asset_context.get(key))
        for key in (
            "text_hint",
            "caption_text",
            "surrounding_text",
            "parent_text_preview",
            "parent_title",
        )
    )


def _trainability_evidence(
    *,
    asset_context: Mapping[str, object],
    enrichment: Mapping[str, object],
) -> dict[str, bool]:
    return {
        "has_parent_context": _context_available(asset_context=asset_context),
        "has_enrichment": bool(enrichment),
        "has_transcript": bool(
            _text(asset_context.get("transcript_path"))
            or _text(enrichment.get("transcript_text"))
            or _text(enrichment.get("transcript_preview"))
        ),
        "has_keyframe": bool(
            _text_items(asset_context.get("keyframe_paths"))
            or _text_items(enrichment.get("keyframe_paths"))
            or _text_items(enrichment.get("keyframes"))
        ),
        "has_thumbnail": bool(
            _text(asset_context.get("thumbnail_path"))
            or _text(asset_context.get("poster_url"))
            or _text(enrichment.get("thumbnail_path"))
        ),
    }


def _has_trainable_video_enrichment(
    *,
    enrichment: Mapping[str, object],
) -> bool:
    if any(
        _text(enrichment.get(key))
        for key in (
            "transcript_text",
            "transcript_preview",
            "frame_ocr_text",
            "frame_ocr_preview",
            "video_caption_text",
            "video_summary_text",
            "thumbnail_path",
            "visual_proxy_path",
        )
    ):
        return True
    return bool(
        _text_items(enrichment.get("keyframes"))
        or _text_items(enrichment.get("keyframe_paths"))
        or _text_items(enrichment.get("visual_proxy_paths"))
    )


def _trainability_reason(
    *,
    asset_fetch_mode: str,
    asset_context: Mapping[str, object],
    enrichment: Mapping[str, object],
) -> str | None:
    rejection_reason = _text(asset_context.get("asset_rejection_reason"))
    if rejection_reason:
        return rejection_reason
    if asset_fetch_mode == "embed_metadata":
        if _has_trainable_video_enrichment(enrichment=enrichment):
            return "embed_metadata_enrichment_available"
        if _context_available(asset_context=asset_context):
            return "embed_metadata_context_available"
        if _text(asset_context.get("thumbnail_path")) or _text(
            asset_context.get("poster_url")
        ):
            return "embed_metadata_visual_proxy_available"
        return "embed_metadata_requires_context_or_external_enrichment"
    if asset_fetch_mode == "metadata_only":
        return "metadata_only_requires_context_or_enrichment"
    if enrichment:
        return "enrichment_available"
    if _context_available(asset_context=asset_context):
        return "context_available"
    return None


def _build_record_metadata(
    *,
    task: CrawlTask,
    result: FetchResult,
    kind: str,
    source_page_url: str | None,
    media_identity: str | None,
    asset_context: Mapping[str, object],
    normalized_enrichment: Mapping[str, object],
    trainability_evidence: Mapping[str, bool],
    fetch_mode: str,
    is_complete_payload: bool,
    observed_bytes: int,
    source_content_length: int | None,
    fetch_duration_seconds: float | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "task_id": str(task.task_id),
        "source_type": task.source_type,
        "parent_url": task.parent_url,
        "source_page_url": source_page_url,
        "requested_kind": task.kind,
        "resolved_kind": kind,
        "discovery_reason": _text(asset_context.get("discovery_reason")),
        "selection_reason": asset_context.get("selection_reason"),
        "admission_reason": asset_context.get("admission_reason")
        or "accepted",
        "embed_url": _text(asset_context.get("embed_url")),
        "embed_host": _text(asset_context.get("embed_host")),
        "parent_title": _text(asset_context.get("parent_title")),
        "parent_text_preview": _text(asset_context.get("parent_text_preview")),
        "media_identity": media_identity,
        "trainability_evidence": trainability_evidence,
        "asset_context": asset_context,
        "category": result.category,
        "relevance_score": result.relevance_score,
        "content_signature": result.content_signature,
        "mime_conflict": result.mime_conflict,
        "fetch_mode": fetch_mode,
        "is_complete_payload": is_complete_payload,
        "observed_bytes": observed_bytes,
        "source_content_length": source_content_length,
        "source_content_type": result.content_type,
        "fetch_duration_seconds": fetch_duration_seconds,
        "payload_sha256": result.body_sha256,
        "enrichment": dict(normalized_enrichment),
    }

    return metadata


def _path_extension(*, relative_path: Path) -> str | None:
    extension = relative_path.suffix.strip().lower()
    if not extension:
        return None
    return extension


def _read_response_header(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    direct_value = headers.get(name)
    if direct_value:
        return direct_value

    lowered_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered_name and value:
            return value

    return None


def _resolve_alignment_group_id(
    *,
    media_identity: str | None,
    parent_stable_url_id: str | None,
    kind: str,
) -> str | None:
    if media_identity:
        return media_identity
    if parent_stable_url_id:
        return f"{parent_stable_url_id}:{kind}"
    return None


def _text(value: object) -> str | None:
    """Return stripped text or None for empty/missing values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_items(value: object) -> tuple[str, ...]:
    """Return non-empty text items for list/tuple values."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())
