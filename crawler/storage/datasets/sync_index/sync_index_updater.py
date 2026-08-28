"""Append operations for raw dataset synchronization indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crawler.storage.datasets.manifests.dataset_manifest_writer import (
    DatasetManifestWriter,
)
from crawler.storage.datasets.sync_index.sync_index_reader import (
    SyncIndexReader,
)

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult
    from crawler.storage.datasets.records.dataset_record import DatasetRecord


class SyncIndexUpdater:
    """Append rows to auxiliary raw dataset sync indexes."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        reader: SyncIndexReader,
        run_directory: Path,
    ) -> None:
        self._settings = settings
        self._reader = reader
        self._run_directory = run_directory

        self.modality_counts: dict[str, int] = {}
        self.relationships_count = 0
        self.metadata_count = 0
        self.updates_count = 0
        self.errors_count = 0
        self.discovered_assets_count = 0
        self.rejected_assets_count = 0

    def prepare_transaction(self, *, modality: str) -> tuple[Path, ...]:
        return self._reader.prepare_record_handles(modality=modality)

    def flush_transaction(self) -> None:
        self._reader.flush_transaction()

    def snapshot(self) -> dict[str, object]:
        return {
            "modality_counts": dict(self.modality_counts),
            "relationships_count": self.relationships_count,
            "metadata_count": self.metadata_count,
            "updates_count": self.updates_count,
            "errors_count": self.errors_count,
            "discovered_assets_count": self.discovered_assets_count,
            "rejected_assets_count": self.rejected_assets_count,
        }

    def restore(self, state: dict[str, object]) -> None:
        raw_modality_counts = state.get("modality_counts")
        if not isinstance(raw_modality_counts, dict):
            raise ValueError("transaction modality_counts must be a mapping")
        modality_counts: dict[str, int] = {}
        for kind, count in raw_modality_counts.items():
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError("transaction modality kind must be text")
            modality_counts[kind] = _transaction_count(count, field=kind)
        self.modality_counts = modality_counts
        self.relationships_count = _transaction_count(
            state.get("relationships_count"),
            field="relationships_count",
        )
        self.metadata_count = _transaction_count(
            state.get("metadata_count"),
            field="metadata_count",
        )
        self.updates_count = _transaction_count(
            state.get("updates_count"),
            field="updates_count",
        )
        self.errors_count = _transaction_count(
            state.get("errors_count"),
            field="errors_count",
        )
        self.discovered_assets_count = _transaction_count(
            state.get("discovered_assets_count"),
            field="discovered_assets_count",
        )
        self.rejected_assets_count = _transaction_count(
            state.get("rejected_assets_count"),
            field="rejected_assets_count",
        )

    def append_record_indexes(
        self,
        *,
        record: DatasetRecord,
        result: FetchResult,
    ) -> None:
        self.append_modality_manifest(record=record)
        self.append_relationships_manifest(record=record)
        self.append_metadata_manifest(record=record, result=result)

    def append_modality_manifest(self, *, record: DatasetRecord) -> None:
        payload = record.model_dump(mode="json")

        index_handle = self._reader.ensure_modality_manifest_handle(
            modality=record.modality,
        )
        DatasetManifestWriter.write_jsonl_row(index_handle, payload)

        canonical_handle = self._reader.ensure_canonical_modality_handle(
            modality=record.modality,
        )
        if canonical_handle is not None:
            DatasetManifestWriter.write_jsonl_row(canonical_handle, payload)

        self.modality_counts[record.modality] = (
            self.modality_counts.get(record.modality, 0) + 1
        )

    def append_relationships_manifest(self, *, record: DatasetRecord) -> None:
        if record.parent_url is None:
            return

        payload = _build_relationship_payload(
            schema_version=self._settings.raw_schema_version,
            record=record,
        )
        DatasetManifestWriter.write_jsonl_row(
            self._reader.relationships_handle,
            payload,
        )
        self.relationships_count += 1

    def append_metadata_manifest(
        self,
        *,
        record: DatasetRecord,
        result: FetchResult,
    ) -> None:
        payload = _build_metadata_payload(
            schema_version=self._settings.raw_schema_version,
            record=record,
            result=result,
        )
        DatasetManifestWriter.write_jsonl_row(
            self._reader.metadata_handle,
            payload,
        )
        self.metadata_count += 1

    def append_updates_manifest(
        self,
        *,
        previous_record: DatasetRecord,
        updated_record: DatasetRecord,
        reason: str,
    ) -> None:
        payload = _build_update_payload(
            schema_version=self._settings.raw_schema_version,
            previous_record=previous_record,
            updated_record=updated_record,
            reason=reason,
        )
        DatasetManifestWriter.write_jsonl_row(
            self._reader.updates_handle,
            payload,
        )
        DatasetManifestWriter.write_jsonl_row(
            self._reader.superseded_objects_handle,
            previous_record.model_dump(mode="json"),
        )
        self.updates_count += 1

    def append_error(self, *, payload: dict[str, Any]) -> None:
        row = dict(payload)
        row["schema_version"] = self._settings.raw_schema_version

        DatasetManifestWriter.write_jsonl_row(self._reader.errors_handle, row)
        self.errors_count += 1

    def append_discovered_assets(
        self,
        *,
        parent_url: str,
        tasks: tuple[CrawlTask, ...],
    ) -> int:
        count = 0
        for task in tasks:
            if task.source_type != "embedded_asset":
                continue
            DatasetManifestWriter.write_jsonl_row(
                self._reader.discovered_assets_handle,
                _build_discovered_asset_payload(
                    task=task,
                    schema_version=self._settings.raw_schema_version,
                    run_id=self._run_directory.name,
                    parent_url=parent_url,
                ),
            )
            count += 1
        self.discovered_assets_count += count
        return count

    def append_rejected_assets(
        self,
        *,
        parent_url: str,
        rejected: tuple[tuple[CrawlTask, str], ...],
    ) -> int:
        count = 0
        for task, reason in rejected:
            if task.source_type != "embedded_asset":
                continue
            context = _task_context_payload(task=task)
            DatasetManifestWriter.write_jsonl_row(
                self._reader.rejected_assets_handle,
                {
                    "schema_version": self._settings.raw_schema_version,
                    "run_id": self._run_directory.name,
                    "parent_url": parent_url,
                    "asset_url": task.url,
                    "kind": task.kind,
                    "source_type": task.source_type,
                    "reason": reason,
                    "source_page_url": context.get("source_page_url")
                    or parent_url,
                    "discovery_reason": context.get("discovery_reason"),
                    "selection_reason": context.get("selection_reason"),
                    "candidate_strength": context.get("candidate_strength"),
                    "media_identity": context.get("media_identity"),
                    "context": context,
                    "stage": "rejected_before_fetch",
                },
            )
            count += 1
        self.rejected_assets_count += count
        return count


def _build_discovered_asset_payload(
    *,
    task: CrawlTask,
    schema_version: str,
    run_id: str,
    parent_url: str,
) -> dict[str, object]:
    context = _task_context_payload(task=task)
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "parent_url": parent_url,
        "asset_url": task.url,
        "candidate_kind": task.kind,
        "source_type": task.source_type,
        "source_page_url": context.get("source_page_url") or parent_url,
        "discovery_reason": context.get("discovery_reason"),
        "candidate_strength": context.get("candidate_strength"),
        "media_identity": context.get("media_identity"),
        "context": context,
        "stage": "extracted_before_scheduler",
    }


def _task_context_payload(*, task: CrawlTask) -> dict[str, object]:
    return task.context.to_dict() if task.context is not None else {}


def _build_relationship_payload(
    *,
    schema_version: str,
    record: DatasetRecord,
) -> dict[str, Any]:
    relation = (
        "contains_asset"
        if record.source_type == "embedded_asset"
        else "links_to"
    )
    return {
        "schema_version": schema_version,
        "run_id": record.run_id,
        "relation": relation,
        "source_url": record.parent_url,
        "target_url": record.normalized_url,
        "target_kind": record.kind,
        "target_modality": record.modality,
        "parent_url": record.parent_url,
        "child_url": record.final_url,
        "child_kind": record.kind,
        "child_modality": record.modality,
        "child_object_id": record.object_id,
        "child_storage_path": record.storage_relative_path,
        "fetched_at": record.fetched_at,
    }


def _build_metadata_payload(
    *,
    schema_version: str,
    record: DatasetRecord,
    result: FetchResult,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "run_id": record.run_id,
        "fetch_record_id": record.fetch_record_id,
        "object_id": record.object_id,
        "normalized_url": record.normalized_url,
        "domain": record.domain,
        "kind": record.kind,
        "modality": record.modality,
        "status_code": record.status_code,
        "content_type": record.content_type,
        "mime_type": record.mime_type,
        "language": record.language,
        "category": record.category,
        "relevance_score": record.relevance_score,
        "content_signature": record.content_signature,
        "enrichment": dict(record.enrichment),
        "parent_fetch_record_id": record.parent_fetch_record_id,
        "parent_stable_url_id": record.parent_stable_url_id,
        "asset_url": record.asset_url,
        "asset_fetch_mode": record.asset_fetch_mode,
        "asset_downloaded": record.asset_downloaded,
        "asset_metadata_only": record.asset_metadata_only,
        "context_available": record.context_available,
        "enrichment_available": record.enrichment_available,
        "trainability_reason": record.trainability_reason,
        "requested_kind": record.requested_kind,
        "resolved_kind": record.resolved_kind,
        "discovery_reason": record.discovery_reason,
        "selection_reason": record.selection_reason,
        "admission_reason": record.admission_reason,
        "source_page_url": record.source_page_url,
        "embed_url": record.embed_url,
        "embed_host": record.embed_host,
        "parent_title": record.parent_title,
        "parent_text_preview": record.parent_text_preview,
        "media_identity": record.media_identity,
        "trainability_evidence": record.metadata.get(
            "trainability_evidence",
            {},
        ),
        "asset_rejection_reason": record.asset_rejection_reason,
        "source_content_type": record.source_content_type,
        "fetch_duration_seconds": record.fetch_duration_seconds,
        "payload_sha256": record.payload_sha256,
        "payload_path": record.payload_path,
        "thumbnail_path": record.thumbnail_path,
        "keyframe_paths": list(record.keyframe_paths),
        "transcript_path": record.transcript_path,
        "asset_context": dict(record.asset_context),
        "mime_conflict": result.mime_conflict,
        "record_version": record.record_version,
        "fetched_at": record.fetched_at,
    }


def _build_update_payload(
    *,
    schema_version: str,
    previous_record: DatasetRecord,
    updated_record: DatasetRecord,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "run_id": updated_record.run_id,
        "normalized_url": updated_record.normalized_url,
        "reason": reason,
        "previous_fetch_record_id": previous_record.fetch_record_id,
        "updated_fetch_record_id": updated_record.fetch_record_id,
        "previous_object_id": previous_record.object_id,
        "updated_object_id": updated_record.object_id,
        "previous_record_version": previous_record.record_version,
        "updated_record_version": updated_record.record_version,
        "updated_at": updated_record.fetched_at,
    }


def _transaction_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"transaction {field} must be a non-negative integer")
    return value
