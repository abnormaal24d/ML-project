"""Dataset writer for raw crawler outputs and synchronization manifests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from crawler.storage.datasets.writing.dataset_snapshot_builder import (
    DatasetSnapshotBuilder,
    DatasetWriterSnapshot,
)
from crawler.storage.datasets.writing.dataset_write_pipeline import (
    DatasetWritePipeline,
)
from crawler.storage.datasets.writing.write_outcome import WriteOutcome
from logger.project_logger import ProjectLogger

TOffloadResult = TypeVar("TOffloadResult")

if TYPE_CHECKING:
    from collections.abc import Mapping

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.coverage.state import CoverageState
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.fetching.response.cache import ConditionalRepresentationCache
    from crawler.fetching.results.result import FetchResult
    from crawler.storage.datasets.manifests.dataset_manifest_writer import (
        DatasetManifestWriter,
    )
    from crawler.storage.datasets.records.dataset_record import (
        DatasetRecord,
        DatasetRecordCreator,
    )
    from crawler.storage.datasets.records.record_index import (
        DatasetRecordIndex,
    )
    from crawler.storage.datasets.sync_index.sync_index_compactor import (
        SyncIndexCompactor,
    )
    from crawler.storage.datasets.sync_index.sync_index_updater import (
        SyncIndexUpdater,
    )
    from crawler.storage.datasets.writing.dataset_error_writer import (
        DatasetErrorWriter,
    )
    from crawler.storage.datasets.writing.dataset_run_finalizer import (
        DatasetRunFinalizer,
    )
    from crawler.storage.datasets.writing.raw_payload_writer import (
        RawPayloadWriter,
    )


class CrawlRunSummary(Protocol):
    """Storage fields required from one terminal crawl summary."""

    output_ready: bool
    unmet_requirements: tuple[str, ...]
    object_records_total: int
    requests_total: int
    successful_requests_total: int
    quality_score: float
    modality_counts: dict[str, int] | None

    @property
    def stop_trigger(self) -> Enum: ...

    @property
    def terminal_outcome(self) -> Enum: ...

    root_seeds_total: int
    root_seeds_succeeded: int
    root_seeds_transient_failed: int
    root_seeds_governance_blocked: int
    required_dependency_failures: int


class DatasetWriter:
    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        run_id: str,
        run_directory: Path,
        manifest_path: Path,
        logger: ProjectLogger,
        url_normalizer: UrlNormalizer,
        record_index: DatasetRecordIndex,
        payload_writer: RawPayloadWriter,
        manifest_writer: DatasetManifestWriter,
        sync_updater: SyncIndexUpdater,
        sync_compactor: SyncIndexCompactor,
        record_creator: DatasetRecordCreator,
        run_finalizer: DatasetRunFinalizer,
        error_writer: DatasetErrorWriter,
        coverage_tracker: CoverageState,
        conditional_representation_cache: ConditionalRepresentationCache,
    ) -> None:
        self._settings = settings
        self._run_id = run_id
        self._completed_at: str | None = None
        self._total_bytes_written = 0
        self._manifest_path = manifest_path
        self._manifest_writer = manifest_writer
        self._sync_updater = sync_updater
        self._sync_compactor = sync_compactor
        self._error_writer = error_writer
        self._write_pipeline = DatasetWritePipeline(
            settings=settings,
            logger=logger,
            url_normalizer=url_normalizer,
            payload_writer=payload_writer,
            manifest_writer=manifest_writer,
            sync_updater=sync_updater,
            record_creator=record_creator,
            record_index=record_index,
            conditional_representation_cache=conditional_representation_cache,
        )
        self._coverage_tracker = coverage_tracker
        self._snapshot_builder = DatasetSnapshotBuilder(
            run_id=run_id,
            run_directory=run_directory,
            record_index=record_index,
            sync_updater=sync_updater,
        )
        self._run_finalizer = run_finalizer
        self._async_write_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._write_sync_summary()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def snapshot(self) -> DatasetWriterSnapshot:
        with self._state_lock:
            return self._build_snapshot()

    def write(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        enrichment: Mapping[str, Any] | None = None,
    ) -> DatasetRecord:
        with self._state_lock:
            self._raise_if_closed()
            outcome: WriteOutcome = self._write_pipeline.execute(
                task=task,
                result=result,
                enrichment=enrichment,
            )
            record = outcome.record

            if not outcome.duplicate:
                self._total_bytes_written += record.byte_size
                self._coverage_tracker.apply_record_transition(
                    previous_kind=outcome.previous_kind,
                    current_kind=outcome.current_kind,
                    previous_eligible=outcome.previous_coverage_eligible,
                    current_eligible=outcome.current_coverage_eligible,
                )
                if self._sync_compactor.should_write_summary():
                    self._write_sync_summary()

            return record

    async def awrite(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        enrichment: Mapping[str, Any] | None = None,
    ) -> DatasetRecord:
        return await self._offload(
            lambda: self.write(
                task=task,
                result=result,
                enrichment=enrichment,
            )
        )

    def write_error(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        reason: str,
        stage: str,
        fields: Mapping[str, Any] | None,
    ) -> None:
        with self._state_lock:
            self._raise_if_closed()
            self._error_writer.write_failure(
                task=task,
                status=outcome,
                reason=reason,
                stage=stage,
                details=fields,
                completed_at=self._completed_at,
                total_bytes_written=self._total_bytes_written,
            )
            if self._sync_compactor.should_write_summary():
                self._write_sync_summary()

    async def awrite_error(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        reason: str,
        stage: str,
        fields: Mapping[str, Any] | None,
    ) -> None:
        await self._offload(
            lambda: self.write_error(
                task=task,
                outcome=outcome,
                reason=reason,
                stage=stage,
                fields=fields,
            )
        )

    def write_discovered_assets(
        self,
        *,
        parent_url: str,
        tasks: tuple[CrawlTask, ...],
    ) -> int:
        with self._state_lock:
            self._raise_if_closed()
            count = self._sync_updater.append_discovered_assets(
                parent_url=parent_url,
                tasks=tasks,
            )
            self._sync_compactor.flush()
            return int(count)

    async def awrite_discovered_assets(
        self,
        *,
        parent_url: str,
        tasks: tuple[CrawlTask, ...],
    ) -> int:
        return await self._offload(
            lambda: self.write_discovered_assets(
                parent_url=parent_url,
                tasks=tasks,
            )
        )

    def write_rejected_assets(
        self,
        *,
        parent_url: str,
        rejected: tuple[tuple[CrawlTask, str], ...],
    ) -> int:
        with self._state_lock:
            self._raise_if_closed()
            count = self._sync_updater.append_rejected_assets(
                parent_url=parent_url,
                rejected=rejected,
            )
            self._sync_compactor.flush()
            return int(count)

    async def awrite_rejected_assets(
        self,
        *,
        parent_url: str,
        rejected: tuple[tuple[CrawlTask, str], ...],
    ) -> int:
        return await self._offload(
            lambda: self.write_rejected_assets(
                parent_url=parent_url,
                rejected=rejected,
            )
        )

    async def aclose(self) -> None:
        """Flush and close resources without assigning a crawl outcome."""
        async with self._async_write_lock:
            self._flush_and_close_resources()

    async def _offload(
        self,
        fn: Callable[[], TOffloadResult],
    ) -> TOffloadResult:
        """Execute one dataset mutation to a definitive outcome.

        Once a filesystem mutation has started, cancellation is deferred
        until the operation has committed or completed its rollback.
        """

        async with self._async_write_lock:
            if not self._settings.raw_persist_offload_to_thread:
                return fn()

            operation = asyncio.create_task(asyncio.to_thread(fn))
            try:
                return await asyncio.shield(operation)
            except asyncio.CancelledError as cancellation:
                try:
                    await asyncio.shield(operation)
                except BaseException as write_error:
                    raise write_error from cancellation
                raise

    def _flush_and_close_resources(self) -> None:
        """Flush and close every owned handle without assigning an outcome."""
        with self._state_lock:
            if self._closed:
                return

            close_error: Exception | None = None
            compactor_closed = False

            if not self._manifest_writer.closed:
                try:
                    self._write_sync_summary()
                except Exception as exc:
                    close_error = exc

                try:
                    self._manifest_writer.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc

            try:
                self._sync_compactor.close()
                compactor_closed = True
            except Exception as exc:
                if close_error is None:
                    close_error = exc

            if self._manifest_writer.closed and compactor_closed:
                self._closed = True

            if close_error is not None:
                raise close_error

    def _build_snapshot(self) -> DatasetWriterSnapshot:
        return self._snapshot_builder.build(
            total_bytes_written=self._total_bytes_written,
        )

    def _raise_if_closed(self) -> None:
        if self._closed or self._manifest_writer.closed:
            raise RuntimeError("dataset writer is closed")

    def _write_sync_summary(self) -> None:
        """Write the non-terminal snapshot used while a crawl is active."""
        self._sync_compactor.refresh_summary(
            completed_at=None,
            total_bytes_written=self._total_bytes_written,
            status="running",
            final=False,
        )

    async def commit_completed(
        self,
        *,
        crawler_result: CrawlRunSummary | None = None,
    ) -> None:
        """Finalize a technically completed raw run.

        A run is ``completed``/``final`` whenever the crawl lifecycle ran to
        completion, whether or not the raw data already satisfies the
        training coverage minimums. Those minimums are recorded as readiness
        evidence, not as a commit admission condition.
        """
        await self._finalize_run(
            status="completed",
            final=True,
            readiness_report=_build_readiness_report(crawler_result),
            reason=None,
            details=None,
        )

    async def mark_cancelled(
        self,
        *,
        reason: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Finalize an explicitly cancelled raw run."""
        await self._finalize_run(
            status="cancelled",
            final=True,
            readiness_report=None,
            reason=reason,
            details=details,
        )

    async def mark_failed(
        self,
        *,
        reason: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        await self._finalize_run(
            status="failed",
            final=True,
            readiness_report=None,
            reason=reason,
            details=details,
        )

    async def mark_incomplete(
        self,
        *,
        reason: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Finalize an incomplete raw run that remains non-promotable."""
        await self._finalize_run(
            status="incomplete",
            final=False,
            readiness_report=None,
            reason=reason,
            details=details,
        )

    async def _finalize_run(
        self,
        *,
        status: str,
        final: bool,
        readiness_report: dict[str, object] | None,
        reason: str | None,
        details: Mapping[str, object] | None,
    ) -> None:
        """Serialize terminal finalization against every pending async write."""
        async with self._async_write_lock:
            with self._state_lock:
                if self._closed:
                    return

                completed_at = self._run_finalizer.finalize(
                    total_bytes_written=self._total_bytes_written,
                    status=status,
                    final=final,
                    readiness_report=readiness_report,
                    terminal_reason=reason,
                    terminal_details=details,
                )
                self._completed_at = completed_at
                self._closed = True


def _build_readiness_report(
    crawler_result: CrawlRunSummary | None,
) -> dict[str, object]:
    """Build the persisted readiness report from one crawl summary."""

    if crawler_result is None:
        return {}

    return {
        "ready": bool(crawler_result.output_ready),
        "unmet_requirements": list(crawler_result.unmet_requirements),
        "object_records_total": crawler_result.object_records_total,
        "requests_total": crawler_result.requests_total,
        "successful_requests_total": crawler_result.successful_requests_total,
        "quality_score": crawler_result.quality_score,
        "modality_counts": dict(crawler_result.modality_counts or {}),
        "stop_trigger": crawler_result.stop_trigger.value,
        "terminal_outcome": crawler_result.terminal_outcome.value,
        "root_seeds_total": crawler_result.root_seeds_total,
        "root_seeds_succeeded": crawler_result.root_seeds_succeeded,
        "root_seeds_transient_failed": crawler_result.root_seeds_transient_failed,
        "root_seeds_governance_blocked": crawler_result.root_seeds_governance_blocked,
        "required_dependency_failures": crawler_result.required_dependency_failures,
    }
