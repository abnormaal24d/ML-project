"""Persisting processor coordinating dataset writes for accepted fetches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

from config.collection.processors import BaseProcessorSettings
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.processing.outcomes.processor_outcome import (
    RESERVED_PROCESSOR_METADATA_KEYS,
    ProcessorOutcome,
)
from crawler.processing.processors.fetched_result_processor import (
    FetchedResultProcessor,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from crawler.runtime.concurrency import TransientLockRaceError

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from logger.project_logger import ProjectLogger

TProcessorSettings = TypeVar(
    "TProcessorSettings",
    bound=BaseProcessorSettings,
)
TAnalysis = TypeVar("TAnalysis")
TStageResult = TypeVar("TStageResult")


class PersistingProcessor(
    FetchedResultProcessor,
    Generic[TProcessorSettings, TAnalysis],
):
    """Base class for processors that optionally persist accepted results."""

    def __init__(
        self,
        *,
        settings: TProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
    ) -> None:
        self._settings = settings
        self._dataset_writer = dataset_writer
        self._logger = logger
        self._failure_handler = failure_handler

    async def process_fetched(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
    ) -> ProcessorOutcome:
        self._logger.debug(
            "processor_started",
            task_id=task.task_id,
            url=task.url,
            requested_kind=task.kind,
            result_kind=result.kind,
            final_url=result.final_url,
            depth=task.depth,
            source=task.source_type,
            parent=task.parent_url,
        )
        analysis_or_outcome = await self._execute_stage(
            task=task,
            result=result,
            stage="analysis",
            operation=lambda: self.analyze_fetched(result=result),
        )
        if isinstance(analysis_or_outcome, ProcessorOutcome):
            return analysis_or_outcome

        return await self.persist_analyzed_result(
            task=task,
            result=result,
            analysis=analysis_or_outcome,
        )

    async def analyze_fetched(
        self,
        *,
        result: FetchResult,
    ) -> TAnalysis | None:
        return await self.prepare_analysis(result=result)

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> TAnalysis | None:
        """Build optional processor-specific analysis state."""
        return None

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> tuple[bool, str | None, Mapping[str, object]]:
        """Return acceptance, reject reason, and quality-related fields."""
        return True, None, {}

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> Mapping[str, object]:
        """Build enrichment fields stored alongside the raw result."""
        return {}

    async def after_persist(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> Mapping[str, object]:
        """Build post-persist outcome fields."""
        return {}

    async def persist_analyzed_result(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> ProcessorOutcome:
        validation_or_outcome = await self._execute_stage(
            task=task,
            result=result,
            stage="validation",
            operation=lambda: self.validate_result(
                result=result,
                analysis=analysis,
            ),
        )
        if isinstance(validation_or_outcome, ProcessorOutcome):
            return validation_or_outcome

        accepted, reject_reason, quality_fields = validation_or_outcome
        if not accepted:
            outcome = self._failure_handler.quality_rejected(
                reject_reason=reject_reason or "quality_rejected",
                quality_fields=quality_fields,
            )
            return await self._finalize_failure_outcome(
                task=task,
                result=result,
                outcome=outcome,
            )

        outcome = await self._execute_stage(
            task=task,
            result=result,
            stage="persistence",
            operation=lambda: self._persist_accepted_analysis(
                task=task,
                result=result,
                analysis=analysis,
                quality_fields=quality_fields,
            ),
        )

        if outcome.status == "success":
            self._logger.debug(
                "processor_finished",
                task_id=task.task_id,
                url=task.url,
                requested_kind=task.kind,
                result_kind=result.kind,
                final_url=result.final_url,
            )
        return outcome

    async def persist_analysis_failure(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        reason: str,
        error_type: str | None = None,
        error: str | None = None,
    ) -> ProcessorOutcome:
        outcome = self._failure_handler.processor_exception(
            stage="analysis",
            reason=reason,
            error_type=error_type,
            error=error,
        )
        return await self._finalize_failure_outcome(
            task=task,
            result=result,
            outcome=outcome,
        )

    async def _persist_accepted_analysis(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
        quality_fields: Mapping[str, object],
    ) -> ProcessorOutcome:
        enrichment_fields = await self._build_enrichment_fields(
            task=task,
            result=result,
            analysis=analysis,
        )
        await self._persist_processor_result(
            task=task,
            result=result,
            enrichment=enrichment_fields,
        )
        outcome_fields = await self._merge_accepted_outcome_fields(
            task=task,
            result=result,
            analysis=analysis,
            quality_fields=quality_fields,
            enrichment_fields=enrichment_fields,
        )
        return self._build_persisted_processor_outcome(
            result=result,
            outcome_fields=outcome_fields,
        )

    async def _build_enrichment_fields(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> dict[str, object]:
        enrichment: object = await self.build_enrichment(
            result=result,
            analysis=analysis,
        )
        if not isinstance(enrichment, Mapping):
            self._logger.warning(
                "processor_extension_fields_invalid",
                task_id=task.task_id,
                url=task.url,
                requested_kind=task.kind,
                result_kind=result.kind,
                final_url=result.final_url,
                stage="enrichment",
                returned_type=type(enrichment).__name__,
            )
            return {}
        return {name: value for name, value in enrichment.items()}

    async def _merge_accepted_outcome_fields(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
        quality_fields: Mapping[str, object],
        enrichment_fields: Mapping[str, object],
    ) -> dict[str, object]:
        after_persist_fields = await self._collect_after_persist_fields(
            task=task,
            result=result,
            analysis=analysis,
        )
        return self._merge_extension_metadata(
            task=task,
            result=result,
            groups=(
                ("quality", quality_fields),
                ("enrichment", enrichment_fields),
                ("after_persist", after_persist_fields),
            ),
        )

    async def _collect_after_persist_fields(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: TAnalysis | None,
    ) -> Mapping[str, object]:
        try:
            return await self.after_persist(
                task=task,
                result=result,
                analysis=analysis,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            self._logger.error(
                "processor_after_persist_failed",
                task_id=task.task_id,
                url=task.url,
                requested_kind=task.kind,
                result_kind=result.kind,
                final_url=result.final_url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return {
                "after_persist_failed": True,
                "after_persist_error_type": type(exc).__name__,
                "after_persist_error": str(exc),
            }

    def _build_persisted_processor_outcome(
        self,
        *,
        result: FetchResult,
        outcome_fields: Mapping[str, object],
    ) -> ProcessorOutcome:
        if outcome_fields.get("after_persist_failed") is True:
            return ProcessorOutcome.deferred(
                stage="persistence",
                reason="after_persist_failed",
                retry_after_seconds=(
                    self._failure_handler.transient_lock_race_wait_seconds
                ),
                metadata=outcome_fields,
            )

        return ProcessorOutcome.success(
            stage="persistence",
            detail=str(result.kind.value),
            metadata=outcome_fields,
        )

    async def _execute_stage(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        stage: str,
        operation: Callable[[], Awaitable[TStageResult]],
    ) -> TStageResult | ProcessorOutcome:
        try:
            return await operation()
        except (
            IgnoredFetchError,
            RetryableFetchError,
            TransientLockRaceError,
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:
            if isinstance(exc, IgnoredFetchError):
                outcome = self._failure_handler.ignored(
                    exc=exc,
                    stage=stage,
                )
            elif isinstance(exc, RetryableFetchError):
                outcome = self._failure_handler.retryable(
                    exc=exc,
                    stage=stage,
                )
            elif isinstance(exc, TransientLockRaceError):
                outcome = self._failure_handler.transient_lock_race(
                    exc=exc,
                    stage=stage,
                )
            else:
                outcome = self._failure_handler.processor_exception(
                    stage=stage,
                    exc=exc,
                )
            return await self._finalize_failure_outcome(
                task=task,
                result=result,
                outcome=outcome,
            )

    async def _finalize_failure_outcome(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        outcome: ProcessorOutcome,
    ) -> ProcessorOutcome:
        """Persist and log one canonical failure outcome at this boundary."""
        persistence_error: Exception | None = None

        if self._settings.persist_raw:
            try:
                await self._dataset_writer.awrite_error(
                    task=task,
                    outcome=outcome.status,
                    reason=outcome.reason,
                    stage=outcome.stage,
                    fields=self._failure_persistence_fields(
                        result=result,
                        outcome=outcome,
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                persistence_error = exc

        if persistence_error is None:
            self._log_failure_outcome(
                task=task,
                result=result,
                outcome=outcome,
            )
        else:
            self._logger.error(
                "processor_error_persist_failed",
                task_id=task.task_id,
                url=task.url,
                requested_kind=task.kind,
                result_kind=result.kind,
                final_url=result.final_url,
                outcome=outcome.status,
                reason=outcome.reason,
                stage=outcome.stage,
                persistence_error_type=type(persistence_error).__name__,
                persistence_error=str(persistence_error),
            )
        return outcome

    @staticmethod
    def _failure_persistence_fields(
        *,
        result: FetchResult,
        outcome: ProcessorOutcome,
    ) -> dict[str, object]:
        record_fields = dict(outcome.metadata)
        record_fields.update(
            {
                "kind": result.kind,
                "final_url": result.final_url,
                "status_code": result.status_code,
                "content_type": result.content_type,
                "mime_type": result.mime_type,
                "category": result.category,
                "relevance_score": result.relevance_score,
                "status": outcome.status,
                "stage": outcome.stage,
                "reason": outcome.reason,
                "detail": outcome.detail,
            }
        )
        if outcome.retry_after_seconds is not None:
            record_fields["retry_after_seconds"] = outcome.retry_after_seconds
        if outcome.retry_class:
            record_fields["retry_class"] = outcome.retry_class
        if outcome.retry_error_kind:
            record_fields["retry_error_kind"] = outcome.retry_error_kind
        if outcome.counts_toward_task_retry_budget:
            record_fields["counts_toward_task_retry_budget"] = True
        if outcome.terminal_eligible:
            record_fields["terminal_eligible"] = True
        if outcome.error_type:
            record_fields["error_type"] = outcome.error_type
        if outcome.error:
            record_fields["error"] = outcome.error
        return record_fields

    def _log_failure_outcome(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        outcome: ProcessorOutcome,
    ) -> None:
        fields: dict[str, object] = dict(outcome.metadata)
        fields.update(
            {
                "task_id": task.task_id,
                "url": task.url,
                "requested_kind": task.kind,
                "result_kind": result.kind,
                "final_url": result.final_url,
                "status": outcome.status,
                "stage": outcome.stage,
                "reason": outcome.reason,
                "detail": outcome.detail,
            }
        )
        if outcome.retry_after_seconds is not None:
            fields["retry_after_seconds"] = outcome.retry_after_seconds
        if outcome.retry_class:
            fields["retry_class"] = outcome.retry_class
        if outcome.retry_error_kind:
            fields["retry_error_kind"] = outcome.retry_error_kind
        if outcome.error_type:
            fields["error_type"] = outcome.error_type
        if outcome.error:
            fields["error"] = outcome.error
        if outcome.status == "deferred":
            self._logger.warning("processor_deferred", **fields)
        elif outcome.reason == "processor_exception" or (
            outcome.status == "failure"
        ):
            self._logger.error("processor_dropped", **fields)
        else:
            self._logger.debug("processor_dropped", **fields)

    def _merge_extension_metadata(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        groups: tuple[tuple[str, object], ...],
    ) -> dict[str, object]:
        merged: dict[str, object] = {}

        for source, candidate in groups:
            if candidate is None:
                continue

            if not isinstance(candidate, Mapping):
                self._logger.warning(
                    "processor_extension_metadata_invalid",
                    task_id=task.task_id,
                    url=task.url,
                    requested_kind=task.kind,
                    result_kind=result.kind,
                    final_url=result.final_url,
                    source=source,
                    returned_type=type(candidate).__name__,
                )
                continue

            for name, value in candidate.items():
                if not isinstance(name, str) or not name:
                    self._logger.warning(
                        "processor_extension_metadata_ignored",
                        task_id=task.task_id,
                        url=task.url,
                        source=source,
                        returned_key_type=type(name).__name__,
                    )
                    continue

                if name in RESERVED_PROCESSOR_METADATA_KEYS:
                    self._logger.warning(
                        "processor_extension_metadata_ignored",
                        task_id=task.task_id,
                        url=task.url,
                        source=source,
                        field=name,
                        reason="reserved_key",
                    )
                    continue

                merged[name] = value

        return merged

    async def _persist_processor_result(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        enrichment: Mapping[str, object],
    ) -> None:
        if not self._settings.persist_raw:
            return

        await self._dataset_writer.awrite(
            task=task,
            result=result,
            enrichment=enrichment,
        )
