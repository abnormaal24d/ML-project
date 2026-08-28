"""Track the latest canonical dataset record per normalized URL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.storage.datasets.records.dataset_record import DatasetRecord


class DatasetRecordIndex:
    """Track the latest canonical dataset record per normalized URL."""

    def __init__(self) -> None:
        self._records_by_normalized_url: dict[str, DatasetRecord] = {}
        self._records_by_media_identity: dict[str, DatasetRecord] = {}

    def find_latest(
        self,
        *,
        normalized_url: str,
        media_identity: str | None = None,
    ) -> DatasetRecord | None:
        """Return the latest record registered for this logical object."""

        if media_identity:
            return self._records_by_media_identity.get(media_identity)
        return self._records_by_normalized_url.get(normalized_url)

    def register(self, *, record: DatasetRecord) -> None:
        """Register the latest record for its normalized URL."""

        if record.media_identity:
            self._records_by_media_identity[record.media_identity] = record
            if record.kind in {"image", "audio", "video", "document"}:
                return
        self._records_by_normalized_url[record.normalized_url] = record

    def latest_records(self) -> tuple["DatasetRecord", ...]:
        """Return the compact latest-record view sorted by normalized URL."""

        records_by_fetch_id = {
            record.fetch_record_id: record
            for record in (
                *self._records_by_normalized_url.values(),
                *self._records_by_media_identity.values(),
            )
        }
        return tuple(
            record
            for _, record in sorted(
                records_by_fetch_id.items(),
                key=lambda item: item[1].normalized_url,
            )
        )

    def snapshot(
        self,
    ) -> tuple[dict[str, "DatasetRecord"], dict[str, "DatasetRecord"]]:
        """Return a copy of the index state for rollback recovery."""

        return (
            dict(self._records_by_normalized_url),
            dict(self._records_by_media_identity),
        )

    def restore(
        self,
        state: tuple[
            dict[str, "DatasetRecord"],
            dict[str, "DatasetRecord"],
        ],
    ) -> None:
        """Restore a previously captured index state."""

        normalized, media = state
        self._records_by_normalized_url = dict(normalized)
        self._records_by_media_identity = dict(media)

    def __len__(self) -> int:
        """Return the number of indexed normalized URLs."""

        return len(self.latest_records())
