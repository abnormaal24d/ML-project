"""Canonical outcome schema for one raw dataset write."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.storage.datasets.records.dataset_record import DatasetRecord


class WriteOperation(StrEnum):
    """Mutually exclusive dataset write operations."""

    INSERT = "insert"
    UPDATE = "update"
    DUPLICATE = "duplicate"


def record_is_coverage_eligible(record: DatasetRecord | None) -> bool:
    """Return whether one record owns complete-payload coverage credit."""

    return bool(
        record is not None
        and record.is_complete_payload
        and not record.asset_metadata_only
        and record.byte_size > 0
        and record.payload_sha256
    )


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """Consistent result of an insert, update, or duplicate decision."""

    record: DatasetRecord
    operation: WriteOperation
    previous_kind: str | None
    current_kind: str
    previous_coverage_eligible: bool

    @property
    def inserted(self) -> bool:
        return self.operation is WriteOperation.INSERT

    @property
    def updated(self) -> bool:
        return self.operation is WriteOperation.UPDATE

    @property
    def duplicate(self) -> bool:
        return self.operation is WriteOperation.DUPLICATE

    @property
    def current_coverage_eligible(self) -> bool:
        return record_is_coverage_eligible(self.record)

    @property
    def payload_complete(self) -> bool:
        return self.record.is_complete_payload

    @property
    def record_id(self) -> str:
        return self.record.fetch_record_id
