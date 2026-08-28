"""Pure snapshot projection for the dataset writer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from schemas.raw_payload_evidence import (
    raw_payload_evidence_matches,
)

if TYPE_CHECKING:
    from crawler.storage.datasets.records.dataset_record import DatasetRecord
    from crawler.storage.datasets.records.record_index import (
        DatasetRecordIndex,
    )
    from crawler.storage.datasets.sync_index.sync_index_updater import (
        SyncIndexUpdater,
    )


@dataclass(frozen=True, slots=True)
class DatasetWriterSnapshot:
    """Current raw dataset writer counters."""

    run_id: str
    object_records_total: int
    crawl_edges_total: int
    object_metadata_total: int
    object_events_total: int
    modality_counts: tuple[tuple[str, int], ...]
    total_bytes_written: int


class DatasetSnapshotBuilder:
    """Build payload-backed run snapshots without owning writer lifecycle."""

    def __init__(
        self,
        *,
        run_id: str,
        run_directory: Path,
        record_index: DatasetRecordIndex,
        sync_updater: SyncIndexUpdater,
    ) -> None:
        self._run_id = run_id
        self._run_directory = run_directory.resolve()
        self._record_index = record_index
        self._sync_updater = sync_updater

    def build(self, *, total_bytes_written: int) -> DatasetWriterSnapshot:
        records = self.valid_current_records(
            self._record_index.latest_records()
        )
        return DatasetWriterSnapshot(
            run_id=self._run_id,
            object_records_total=len(records),
            crawl_edges_total=self._sync_updater.relationships_count,
            object_metadata_total=self._sync_updater.metadata_count,
            object_events_total=self._sync_updater.updates_count,
            modality_counts=tuple(
                sorted(self.valid_modality_counts(records).items())
            ),
            total_bytes_written=total_bytes_written,
        )

    def valid_current_records(
        self,
        records: Iterable[DatasetRecord],
    ) -> tuple[DatasetRecord, ...]:
        """Return the latest payload-backed record for each logical object."""
        latest: dict[str, DatasetRecord] = {}
        for record in records:
            identity = next(
                (
                    value.strip()
                    for value in (
                        record.media_identity,
                        record.stable_url_id,
                        record.normalized_url,
                        record.object_id,
                    )
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
            if identity is None or not self.is_valid_payload_record(record):
                continue
            latest[identity] = record
        return tuple(latest.values())

    def is_valid_payload_record(self, record: DatasetRecord) -> bool:
        relative_path = record.storage_relative_path
        if not isinstance(relative_path, str) or not relative_path.strip():
            return False
        payload_path = (self._run_directory / relative_path).resolve()
        try:
            payload_path.relative_to(self._run_directory)
        except ValueError:
            return False
        if not payload_path.is_file():
            return False
        return self.valid_modality(record, suffix=payload_path.suffix.lower())

    @staticmethod
    def valid_modality(
        record: DatasetRecord,
        *,
        suffix: str,
    ) -> bool:
        return raw_payload_evidence_matches(
            modality=record.modality,
            mime_type=record.mime_type or record.content_type,
            suffix=suffix,
        )

    @staticmethod
    def valid_modality_counts(
        records: Iterable[DatasetRecord],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            modality = record.modality.strip().lower()
            counts[modality] = counts.get(modality, 0) + 1
        return counts
