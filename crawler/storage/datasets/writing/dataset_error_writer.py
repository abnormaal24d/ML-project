"""Persist failed URL records into raw dataset sync error indexes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.storage.datasets.manifests.dataset_manifest_writer import (
        DatasetManifestWriter,
    )
    from crawler.storage.datasets.sync_index.sync_index_compactor import (
        SyncIndexCompactor,
    )
    from crawler.storage.datasets.sync_index.sync_index_updater import (
        SyncIndexUpdater,
    )


class DatasetErrorWriter:
    """Append sync-index error rows and refresh run summaries."""

    def __init__(
        self,
        *,
        run_id: str,
        normalize_url: Callable[[str], str],
        now: Callable[[], datetime],
        sync_updater: SyncIndexUpdater,
        sync_compactor: SyncIndexCompactor,
        manifest_writer: DatasetManifestWriter,
    ) -> None:
        self._run_id = run_id
        self._normalize_url = normalize_url
        self._now = now
        self._sync_updater = sync_updater
        self._sync_compactor = sync_compactor
        self._manifest_writer = manifest_writer

    def write_failure(
        self,
        *,
        task: CrawlTask,
        status: str,
        reason: str,
        stage: str,
        details: Mapping[str, Any] | None,
        completed_at: str | None,
        total_bytes_written: int,
    ) -> None:
        """Append a failed URL row and refresh the run summary."""

        payload_fields = dict(details or {})

        final_url = str(payload_fields.get("final_url") or task.url)
        normalized_url = self._normalize_url(final_url)
        kind = payload_fields.get("kind") or task.kind
        if not isinstance(kind, MediaKind):
            raise TypeError("failed task record kind must be MediaKind")
        kind_value = kind.value

        payload = {
            "run_id": self._run_id,
            "url": task.url,
            "requested_url": task.url,
            "final_url": final_url,
            "normalized_url": normalized_url,
            "parent_url": task.parent_url,
            "kind": kind_value,
            "modality": kind_value,
            "status": status,
            "reason": reason,
            "stage": stage,
            "status_code": payload_fields.get("status_code"),
            "error_type": payload_fields.get("error_type"),
            "error": payload_fields.get("error"),
            "depth": task.depth,
            "source_type": task.source_type,
            "recorded_at": self._now().isoformat(),
            "metadata": payload_fields,
        }

        self._sync_updater.append_error(payload=payload)
        self._sync_compactor.flush()
        self._sync_compactor.refresh_summary(
            completed_at=completed_at,
            total_bytes_written=total_bytes_written,
        )
