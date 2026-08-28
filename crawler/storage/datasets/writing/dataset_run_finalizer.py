"""Finalize a raw dataset run after all writes complete."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from pathlib import Path

    from crawler.storage.datasets.manifests.dataset_manifest_writer import (
        DatasetManifestWriter,
    )
    from crawler.storage.datasets.sync_index.sync_index_compactor import (
        SyncIndexCompactor,
    )


class DatasetRunFinalizer:
    """Close dataset handles, refresh summaries, and emit finalization logs."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        now: Callable[[], datetime],
        run_id: str,
        run_directory: Path,
        manifest_path: Path,
        manifest_writer: DatasetManifestWriter,
        sync_compactor: SyncIndexCompactor,
    ) -> None:
        self._logger = logger
        self._now = now
        self._run_id = run_id
        self._run_directory = run_directory
        self._manifest_path = manifest_path
        self._manifest_writer = manifest_writer
        self._sync_compactor = sync_compactor

    def finalize(
        self,
        *,
        total_bytes_written: int,
        status: str = "completed",
        final: bool = True,
        readiness_report: dict[str, object] | None = None,
        terminal_reason: str | None = None,
        terminal_details: Mapping[str, object] | None = None,
    ) -> str:
        """Flush and close all dataset handles and write the final summary.

        status: "completed", "cancelled", "failed", "incomplete", ...
        final: whether this is a terminal publication.
        """

        completed_at = self._now().isoformat()
        normalized_status = _normalize_status(status)
        resolved_reason = _resolve_terminal_reason(
            status=normalized_status,
            reason=terminal_reason,
        )
        resolved_details = dict(terminal_details or {})

        # Final object rows must be flushed/closed before publishing the
        # canonical terminal run manifest.
        self._manifest_writer.close()

        run_manifest_path = self._sync_compactor.refresh_summary(
            completed_at=completed_at,
            total_bytes_written=total_bytes_written,
            status=normalized_status,
            final=final,
            readiness_report=readiness_report,
            terminal_reason=resolved_reason,
            terminal_details=resolved_details,
        )

        self._sync_compactor.close()

        self._logger.info(
            "raw_dataset_run_finalized",
            run_id=self._run_id,
            run_directory=str(self._run_directory),
            manifest_path=str(self._manifest_path),
            completed_at=completed_at,
            status=normalized_status,
            final=final,
            terminal_reason=resolved_reason,
            terminal_details=resolved_details,
            run_manifest_path=str(run_manifest_path),
            object_records_total=self._manifest_writer.write_count,
            total_bytes_written=total_bytes_written,
        )
        return completed_at


def _normalize_status(status: str) -> str:
    normalized = str(status).strip().lower()
    if not normalized:
        raise ValueError("dataset terminal status must not be empty")
    return normalized


def _resolve_terminal_reason(*, status: str, reason: str | None) -> str | None:
    cleaned = None if reason is None else str(reason).strip()
    if status in {"running", "completed"}:
        if cleaned:
            raise ValueError(
                f"status {status!r} must not define a terminal reason"
            )
        return None
    if cleaned:
        return cleaned
    fallbacks = {
        "failed": "unspecified_failure",
        "incomplete": "unspecified_incomplete",
        "cancelled": "unspecified_cancelled",
    }
    return fallbacks.get(status, "unspecified_terminal_state")
