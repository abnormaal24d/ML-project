"""Task processor orchestration for fetched crawl results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.runtime.concurrency import TransientLockRaceError

if TYPE_CHECKING:
    from crawler.analysis.enrichment.lanes.analysis_router import (
        AnalysisRouter,
    )
    from crawler.coverage.fetch_admission import CoverageFetchGate
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.fetcher import FetchOrchestrator
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.processing.routing.crawl_result_router import (
        CrawlResultRouter,
    )
    from logger.project_logger import ProjectLogger


class CrawlTaskProcessor:
    """Coordinate coverage admission, fetch, routing, and payload cleanup."""

    def __init__(
        self,
        *,
        fetch_service: FetchOrchestrator,
        coverage_gate: CoverageFetchGate | None,
        failure_handler: ProcessorFailureHandler,
        result_router: CrawlResultRouter,
        logger: ProjectLogger,
    ) -> None:
        self._fetch_service = fetch_service
        self._coverage_gate = coverage_gate
        self._failure_handler = failure_handler
        self._result_router = result_router
        self._logger = logger

    @property
    def analysis_router(self) -> AnalysisRouter | None:
        return self._result_router.analysis_router

    async def process(self, task: CrawlTask) -> ProcessorOutcome:
        """Process one crawl task through fetch and result routing."""
        result: FetchResult | None = None
        cleanup_payload = True

        try:
            gate_outcome = self._coverage_gate_outcome(task=task)

            if gate_outcome is not None:
                return gate_outcome

            try:
                result = await self._fetch_service.fetch(task=task)
            except IgnoredFetchError as exc:
                return self._finalize_failure_outcome(
                    task=task,
                    outcome=self._failure_handler.ignored(
                        exc=exc,
                        stage="fetch",
                    ),
                )
            except RetryableFetchError as exc:
                return self._finalize_failure_outcome(
                    task=task,
                    outcome=self._failure_handler.retryable(
                        exc=exc,
                        stage="fetch",
                    ),
                )
            except TransientLockRaceError as exc:
                return self._finalize_failure_outcome(
                    task=task,
                    outcome=self._failure_handler.transient_lock_race(
                        exc=exc,
                        stage="fetch",
                    ),
                )

            if result is None:
                raise RuntimeError("fetch completed without a result")

            outcome, cleanup_payload = await self._result_router.route(
                task=task,
                result=result,
            )

            return outcome

        finally:
            if (
                result is not None
                and cleanup_payload
                and result.payload is not None
            ):
                try:
                    result.payload.cleanup()
                except OSError as exc:
                    self._logger.warning(
                        "task_processor_payload_cleanup_failed",
                        final_url=result.final_url,
                        payload_path=str(result.payload.temp_path),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

    def _finalize_failure_outcome(
        self,
        *,
        task: CrawlTask,
        outcome: ProcessorOutcome,
    ) -> ProcessorOutcome:
        """Log one task-boundary event for a pure failure outcome."""
        fields: dict[str, object] = dict(outcome.metadata)
        fields.update(
            {
                "task_id": task.task_id,
                "url": task.url,
                "requested_kind": task.kind,
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
        if outcome.counts_toward_task_retry_budget:
            fields["counts_toward_task_retry_budget"] = True
        if outcome.terminal_eligible:
            fields["terminal_eligible"] = True
        if outcome.error_type:
            fields["error_type"] = outcome.error_type
        if outcome.error:
            fields["error"] = outcome.error
        if outcome.status == "deferred":
            if outcome.reason == "transient_lock_race":
                self._logger.warning("task_processor_deferred", **fields)
            elif outcome.reason == "retryable_fetch_error":
                self._logger.info("task_processor_deferred", **fields)
            else:
                self._logger.debug("task_processor_deferred", **fields)
        else:
            self._logger.debug("task_processor_dropped", **fields)
        return outcome

    def _coverage_gate_outcome(
        self,
        *,
        task: CrawlTask,
    ) -> ProcessorOutcome | None:
        gate = self._coverage_gate

        if gate is None:
            return None

        outcome = gate.outcome_for(task=task)

        if outcome is None:
            return None

        self._logger.debug(
            "task_processor_dropped",
            task_id=task.task_id,
            url=task.url,
            requested_kind=task.kind,
            stage=outcome.stage,
            reason=outcome.reason,
            counts_toward_task_retry_budget=(
                outcome.counts_toward_task_retry_budget
            ),
            terminal_eligible=outcome.terminal_eligible,
        )

        return outcome
