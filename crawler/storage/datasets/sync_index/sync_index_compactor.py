"""Compaction and flush helpers for raw dataset sync indexes."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, TextIO

from crawler.storage.datasets.manifests.dataset_manifest_writer import (
    DatasetManifestWriter,
)
from crawler.storage.datasets.sync_index.sync_index_reader import (
    SyncIndexReader,
)
from crawler.storage.datasets.sync_index.sync_index_updater import (
    SyncIndexUpdater,
)

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.storage.datasets.records.dataset_record import DatasetRecord
    from crawler.storage.datasets.records.record_index import (
        DatasetRecordIndex,
    )
    from crawler.storage.datasets.sync_index.sync_index_paths import (
        SyncIndexPaths,
    )


class SyncIndexCompactor:
    """Flush handles and rewrite compact sync-index snapshots."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        paths: SyncIndexPaths,
        reader: SyncIndexReader,
        updater: SyncIndexUpdater,
        manifest_writer: DatasetManifestWriter,
        record_index: DatasetRecordIndex,
        started_at: str,
        run_identity: dict[str, str | None] | None = None,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._reader = reader
        self._updater = updater
        self._manifest_writer = manifest_writer
        self._record_index = record_index
        self._started_at = started_at
        self._run_identity = dict(run_identity or {})
        self._last_summary_write_count = 0

    def should_write_summary(self) -> bool:
        manifest_write_count = self._manifest_writer.write_count
        if manifest_write_count <= 0:
            return False

        interval = max(1, self._settings.raw_sync_summary_every_records)
        return bool(
            manifest_write_count - self._last_summary_write_count >= interval
        )

    def refresh_summary(
        self,
        *,
        completed_at: str | None,
        total_bytes_written: int,
        status: str = "running",
        final: bool = False,
        readiness_report: dict[str, object] | None = None,
        terminal_reason: str | None = None,
        terminal_details: dict[str, object] | None = None,
    ) -> Path:
        """Refresh the latest-record index and raw run summary."""

        current_records = self._record_index.latest_records()

        self.write_current_objects(
            records=current_records,
        )

        current_modality_counts: dict[str, int] = {}

        for record in current_records:
            modality = record.modality.strip().casefold()

            if modality == "feed":
                modality = "page"

            current_modality_counts[modality] = (
                current_modality_counts.get(modality, 0) + 1
            )

        return self.write_summary(
            object_records_total=len(current_records),
            modality_counts=dict(sorted(current_modality_counts.items())),
            completed_at=completed_at,
            total_bytes_written=total_bytes_written,
            status=status,
            final=final,
            readiness_report=readiness_report,
            terminal_reason=terminal_reason,
            terminal_details=terminal_details,
        )

    def write_summary(
        self,
        *,
        object_records_total: int,
        modality_counts: dict[str, int],
        completed_at: str | None,
        total_bytes_written: int,
        relationships_count: int | None = None,
        metadata_count: int | None = None,
        updates_count: int | None = None,
        errors_count: int | None = None,
        status: str = "running",
        final: bool = False,
        readiness_report: dict[str, object] | None = None,
        terminal_reason: str | None = None,
        terminal_details: dict[str, object] | None = None,
    ) -> Path:
        manifest_write_count = self._manifest_writer.write_count
        resolved_relationships_count = (
            self._updater.relationships_count
            if relationships_count is None
            else relationships_count
        )
        resolved_metadata_count = (
            self._updater.metadata_count
            if metadata_count is None
            else metadata_count
        )
        resolved_updates_count = (
            self._updater.updates_count
            if updates_count is None
            else updates_count
        )
        resolved_errors_count = (
            self._updater.errors_count
            if errors_count is None
            else errors_count
        )

        manifest_path = self._manifest_writer.manifest_path
        # Use portable relative paths for release/portability (no absolute Windows paths)
        run_dir_rel = self._paths.relative_to_run(self._paths.run_directory)
        manifest_rel = self._paths.relative_to_run(manifest_path)

        required_records = [
            self._paths.relative_to_run(manifest_path),
            self._paths.relative_to_run(self._paths.errors_path),
            self._paths.relative_to_run(self._paths.discovered_assets_path),
            self._paths.relative_to_run(self._paths.rejected_assets_path),
            self._paths.relative_to_run(self._paths.current_objects_path),
            self._paths.relative_to_run(self._paths.metadata_path),
        ]

        payload = {
            "schema_version": self._settings.raw_schema_version,
            "lifecycle_stage": "raw",
            "status": status,
            "final": final,
            "started_at": self._started_at,
            "completed_at": completed_at,
            "run_directory": run_dir_rel,
            "manifest_path": manifest_rel,
            "manifest_write_count": manifest_write_count,
            "object_records_total": int(object_records_total),
            "failed_url_count": resolved_errors_count,
            "total_bytes_written": total_bytes_written,
            "relationships_count": resolved_relationships_count,
            "metadata_count": resolved_metadata_count,
            "updates_count": resolved_updates_count,
            "errors_count": resolved_errors_count,
            "discovered_assets_count": self._updater.discovered_assets_count,
            "rejected_assets_count": self._updater.rejected_assets_count,
            "modality_counts": dict(modality_counts),
            "required_records": required_records,
            "output_readiness": dict(readiness_report or {}),
            "terminal_reason": terminal_reason,
            "terminal_details": dict(terminal_details or {}),
        }
        payload.update(
            {
                key: value
                for key, value in self._run_identity.items()
                if value is not None
            }
        )

        self._write_json_atomic(
            path=self._paths.summary_path,
            payload=payload,
        )
        self._last_summary_write_count = manifest_write_count
        return self._paths.summary_path

    def write_current_objects(
        self,
        *,
        records: tuple[DatasetRecord, ...],
    ) -> None:
        rows = tuple(record.model_dump(mode="json") for record in records)
        self._write_jsonl_atomic(
            path=self._paths.current_objects_path,
            rows=rows,
        )

    def flush(self) -> None:
        handles = self._reader.open_handles()

        for handle in handles:
            handle.flush()

        if not self._settings.manifest_fsync_enabled:
            return

        fsync_every = max(1, self._settings.manifest_fsync_every_records)
        if self._manifest_writer.write_count % fsync_every != 0:
            return

        for handle in handles:
            os.fsync(handle.fileno())

    def close(self) -> None:
        for handle in self._reader.open_handles():
            self._flush_and_close_handle(handle)

        self._reader.clear_modality_handles()

    def _write_json_atomic(
        self,
        *,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")

        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            if self._settings.manifest_fsync_enabled:
                os.fsync(handle.fileno())

        os.replace(temporary_path, path)

    def _write_jsonl_atomic(
        self,
        *,
        path: Path,
        rows: tuple[dict[str, Any], ...],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")

        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                DatasetManifestWriter.write_jsonl_row(handle, row)

            handle.flush()
            if self._settings.manifest_fsync_enabled:
                os.fsync(handle.fileno())

        os.replace(temporary_path, path)

    def _flush_and_close_handle(self, handle: TextIO) -> None:
        if handle.closed:
            return

        handle.flush()
        if self._settings.manifest_fsync_enabled:
            os.fsync(handle.fileno())
        handle.close()
