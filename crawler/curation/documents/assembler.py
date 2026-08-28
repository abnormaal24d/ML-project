"""Build curated document records from raw manifest page entries."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from crawler.curation.preprocessing_governance import (
    safe_license_expression,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord
from mmcrawler_datasets.similarity.text_deduplication import (
    NearTextDeduplicator,
)
from preprocessing.preprocessed_document import PreprocessedDocument
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.settings.datasets import (
        CuratedDocumentAssemblerSettings,
    )
    from crawler.curation.ingest.schema.entry import RawManifestEntry
    from crawler.curation.publishing.curated_artifact_writer import (
        CuratedArtifactWriter,
    )
    from crawler.governance.domains.domain_governance import (
        DomainGovernance,
    )
    from crawler.governance.domains.domain_governance_registry import (
        DomainGovernanceRegistry,
    )


class CuratedDocumentAssembler:
    """Assemble curated document records from already-preprocessed documents.

    Preprocessing owns release normalization, privacy and quality. Snapshot-wide
    URL candidate selection and exact/near deduplication belong here.
    """

    def __init__(
        self,
        *,
        settings: CuratedDocumentAssemblerSettings,
        source_domain_registry: DomainGovernanceRegistry,
        artifact_writer: CuratedArtifactWriter,
        logger: ProjectLogger,
        clock: Clock,
        near_deduper: NearTextDeduplicator,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._source_domain_registry = source_domain_registry
        self._artifact_writer = artifact_writer
        self._logger = logger
        self._near_deduper = near_deduper

    def assemble(
        self,
        *,
        snapshot_id: str,
        documents: tuple[PreprocessedDocument, ...],
        entries_by_source_id: dict[str, RawManifestEntry],
    ) -> tuple[
        tuple[CuratedDocumentRecord, ...],
        dict[str, PreprocessedDocument],
    ]:
        """Curate preprocessed documents into snapshot records."""

        if not documents:
            return (), {}
        provisional = self._build_provisional_documents(
            documents, entries_by_source_id
        )
        selected = self._select_documents(provisional)
        records, preprocessed = self._build_final_records(
            snapshot_id, selected
        )
        return tuple(records), preprocessed

    def _build_provisional_documents(
        self,
        documents: tuple[PreprocessedDocument, ...],
        entries_by_source_id: dict[str, RawManifestEntry],
    ) -> list[_ProvisionalDocument]:
        provisional: list[_ProvisionalDocument] = []
        for document in documents:
            if document.source_id is None:
                self._logger.warning(
                    "curated_document_missing_source_id",
                    extra={"document_id": document.document_id},
                )
                continue
            entry = entries_by_source_id.get(document.source_id)
            if entry is None:
                self._logger.warning(
                    "curated_document_source_not_found",
                    extra={
                        "document_id": document.document_id,
                        "source_id": document.source_id,
                    },
                )
                continue
            if (
                document.quality.score
                < self._settings.min_quality_score_for_inclusion
            ):
                continue
            if (
                document.privacy_clearance is None
                or not document.privacy_clearance.permits_training
            ):
                continue

            domain = document.domain or entry.record.domain
            domain_governance = self._source_domain_registry.get(domain=domain)
            allow_training = _resolve_allow_training(
                document=document,
                entry=entry,
                governance=domain_governance,
            )
            if (
                self._settings.require_allow_training
                and allow_training is not True
            ):
                continue

            provisional.append(
                _ProvisionalDocument(
                    entry=entry,
                    preprocessed_document=document,
                    domain_governance=domain_governance,
                )
            )
        return provisional

    def _select_documents(
        self,
        provisional: list[_ProvisionalDocument],
    ) -> list[_ProvisionalDocument]:
        url_selected = _select_best_by_key(
            provisional=provisional,
            key=lambda item: item.entry.record.normalized_url,
        )
        exact_selected = _select_best_by_key(
            provisional=url_selected,
            key=lambda item: item.preprocessed_document.exact_duplicate_key,
        )
        clustered = self._assign_near_duplicate_clusters(
            provisional=exact_selected
        )
        selected = _select_top_documents_per_domain(
            provisional=clustered,
            max_documents_per_domain=self._settings.max_documents_per_domain,
        )
        self._logger.info(
            "curated_document_selection_completed",
            candidates=len(provisional),
            duplicate_urls_dropped=len(provisional) - len(url_selected),
            exact_duplicates_dropped=len(url_selected) - len(exact_selected),
            near_duplicates=sum(
                item.preprocessed_document.is_near_duplicate
                for item in clustered
            ),
            selected=len(selected),
        )
        return selected

    def _assign_near_duplicate_clusters(
        self,
        *,
        provisional: list[_ProvisionalDocument],
    ) -> list[_ProvisionalDocument]:
        assignments = self._near_deduper.assign_clusters(
            texts_by_document_id={
                item.preprocessed_document.document_id: item.preprocessed_document.text
                for item in provisional
            }
        )
        return [
            replace(
                item,
                preprocessed_document=replace(
                    item.preprocessed_document,
                    near_duplicate_cluster_id=assignments[
                        item.preprocessed_document.document_id
                    ].cluster_id,
                    is_near_duplicate=assignments[
                        item.preprocessed_document.document_id
                    ].is_duplicate,
                ),
            )
            for item in provisional
        ]

    def _build_final_records(
        self,
        snapshot_id: str,
        selected: list[_ProvisionalDocument],
    ) -> tuple[list[CuratedDocumentRecord], dict[str, PreprocessedDocument]]:
        finalized_records: list[CuratedDocumentRecord] = []
        preprocessed_by_id: dict[str, PreprocessedDocument] = {}
        for provisional in selected:
            entry = provisional.entry
            document = provisional.preprocessed_document
            governance = provisional.domain_governance
            text_path = self._artifact_writer.write_text(
                object_id=entry.record.object_id,
                text=document.text,
            )
            markdown_path = None
            if document.markdown.strip():
                markdown_path = self._artifact_writer.write_markdown(
                    object_id=entry.record.object_id,
                    markdown=document.markdown,
                )

            governance_fields = _resolve_governance_fields(
                document=document,
                entry=entry,
                governance=governance,
            )
            quality_signals = document.quality.signals
            record = CuratedDocumentRecord(
                schema_version=self._settings.curated_schema_version,
                snapshot_id=snapshot_id,
                document_id=document.document_id,
                source_run_id=entry.record.run_id,
                source_fetch_record_id=entry.record.fetch_record_id,
                object_id=entry.record.object_id,
                requested_url=document.source_url,
                final_url=document.source_url,
                normalized_url=entry.record.normalized_url,
                domain=document.domain or entry.record.domain,
                path=document.path or entry.record.path,
                modality=_curated_document_modality(entry=entry),
                language=document.language,
                title=document.title or document.metadata.title,
                text_path=text_path.as_posix(),
                markdown_path=(
                    markdown_path.as_posix()
                    if markdown_path is not None
                    else None
                ),
                raw_storage_path=f"restricted://{entry.record.object_id}",
                raw_byte_size=entry.record.byte_size,
                extracted_char_count=document.metadata.char_count,
                extracted_token_count_estimate=(
                    document.metadata.token_count_estimate
                ),
                boilerplate_ratio=document.metadata.boilerplate_ratio,
                code_block_count=document.metadata.code_block_count,
                quality_score=document.quality.score,
                quality_bucket=document.quality.bucket,
                rejection_reason=document.quality.rejection_reason,
                content_role=document.metadata.content_role,
                discovery_useful=_as_bool(
                    entry.record.enrichment.get("discovery_useful"),
                    default=True,
                ),
                exact_duplicate_key=document.exact_duplicate_key,
                near_duplicate_cluster_id=(document.near_duplicate_cluster_id),
                is_near_duplicate=document.is_near_duplicate,
                license=governance_fields.license,
                license_url=governance_fields.license_url,
                allow_training=governance_fields.allow_training,
                created_at=self._clock.now().isoformat(),
                governance_note=governance_fields.governance_note,
                language_confidence=_as_optional_float(
                    quality_signals.get("language_confidence")
                ),
                language_script=_as_optional_text(
                    quality_signals.get("script")
                ),
                robots_status=governance_fields.robots_status,
                terms_source=governance_fields.terms_source,
                usage_rules=governance_fields.usage_rules,
                privacy_clearance=(
                    PrivacyClearanceRecord.model_validate(
                        document.privacy_clearance.to_dict()
                    )
                    if document.privacy_clearance is not None
                    else None
                ),
            )
            finalized_records.append(record)
            preprocessed_by_id[document.document_id] = document
        return finalized_records, preprocessed_by_id


@dataclass(frozen=True, slots=True)
class _ProvisionalDocument:
    """In-memory candidate document before top-N domain selection."""

    entry: RawManifestEntry
    preprocessed_document: PreprocessedDocument
    domain_governance: DomainGovernance | None


@dataclass(frozen=True, slots=True)
class _ResolvedGovernanceFields:
    license: str | None
    license_url: str | None
    allow_training: bool | None
    governance_note: str | None
    robots_status: str | None
    terms_source: str | None
    usage_rules: str | None


def _resolve_governance_fields(
    *,
    document: PreprocessedDocument,
    entry: RawManifestEntry,
    governance: DomainGovernance | None,
) -> _ResolvedGovernanceFields:
    clearance = document.privacy_clearance
    approved = (
        {
            field.name: field.value
            for field in clearance.approved_text_fields
            if field.value.strip()
        }
        if clearance is not None
        else {}
    )
    raw_governance = getattr(entry.record, "governance", {})
    raw_license = _as_mapping(raw_governance.get("license"))
    raw_training = _as_mapping(raw_governance.get("training"))

    allow_training = document.allow_training
    if allow_training is None:
        allow_training = _as_optional_bool(raw_training.get("allowed"))
    if allow_training is None and governance is not None:
        allow_training = governance.allow_training

    return _ResolvedGovernanceFields(
        license=(
            approved.get("license")
            or (governance.license if governance is not None else None)
            or safe_license_expression(raw_license.get("expression"))
        ),
        license_url=(
            approved.get("license_url")
            or (governance.license_url if governance is not None else None)
        ),
        allow_training=allow_training,
        governance_note=(
            approved.get("governance_note")
            or (governance.governance_note if governance is not None else None)
        ),
        robots_status=(
            approved.get("robots_status")
            or (governance.robots_status if governance is not None else None)
        ),
        terms_source=(
            approved.get("terms_source")
            or (governance.terms_source if governance is not None else None)
        ),
        usage_rules=(
            approved.get("usage_rules")
            or (governance.usage_rules if governance is not None else None)
        ),
    )


def _resolve_allow_training(
    *,
    document: PreprocessedDocument,
    entry: RawManifestEntry,
    governance: DomainGovernance | None,
) -> bool | None:
    return _resolve_governance_fields(
        document=document,
        entry=entry,
        governance=governance,
    ).allow_training


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("document metadata text must be a string")
    text = value.strip()
    return text or None


def _as_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError("document metadata boolean must be boolean")
    return value


def _as_bool(value: object, *, default: bool) -> bool:
    parsed = _as_optional_bool(value)
    return default if parsed is None else parsed


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("document metadata number must be numeric")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise ValueError("document metadata number must be finite")
    return candidate


def _candidate_rank(
    item: _ProvisionalDocument,
) -> tuple[float, int, int, str, str]:
    """Return a stable best-candidate rank; larger values are preferred."""

    document = item.preprocessed_document
    return (
        document.quality.score,
        len(document.text),
        -len(document.warnings),
        item.entry.record.fetched_at,
        document.document_id,
    )


def _select_best_by_key(
    *,
    provisional: list[_ProvisionalDocument],
    key: Callable[[_ProvisionalDocument], str],
) -> list[_ProvisionalDocument]:
    """Select the highest-ranked valid candidate for each stable key."""

    selected: dict[str, _ProvisionalDocument] = {}
    for item in provisional:
        candidate_key = str(key(item)).strip()
        if not candidate_key:
            candidate_key = item.preprocessed_document.document_id
        current = selected.get(candidate_key)
        if current is None or _candidate_rank(item) > _candidate_rank(current):
            selected[candidate_key] = item
    return sorted(
        selected.values(),
        key=lambda item: item.preprocessed_document.document_id,
    )


def _select_top_documents_per_domain(
    *,
    provisional: list[_ProvisionalDocument],
    max_documents_per_domain: int,
) -> list[_ProvisionalDocument]:
    grouped: dict[str, list[_ProvisionalDocument]] = {}
    for item in provisional:
        domain = item.preprocessed_document.domain or item.entry.record.domain
        grouped.setdefault(domain, []).append(item)

    selected: list[_ProvisionalDocument] = []
    for domain in sorted(grouped):
        ordered = sorted(grouped[domain], key=_candidate_rank, reverse=True)
        selected.extend(
            ordered[:max_documents_per_domain]
            if max_documents_per_domain > 0
            else ordered
        )
    return selected


def _curated_document_modality(*, entry: RawManifestEntry) -> str:
    modality = str(entry.record.modality or "").strip().lower()
    if modality:
        return modality
    return "document" if entry.record.kind == "document" else "text"


def _rejected_by_reason(
    *,
    records: tuple[CuratedDocumentRecord, ...],
) -> dict[str, int]:
    return dict(
        Counter(
            record.rejection_reason or "quality_bucket_reject"
            for record in records
            if record.quality_bucket == "reject"
        )
    )
