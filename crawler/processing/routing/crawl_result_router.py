"""Route a classified fetch result to the matching processor handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.runtime.concurrency import TransientLockRaceError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crawler.analysis.enrichment.lanes.analysis_router import (
        AnalysisRouter,
    )
    from crawler.classification.media_kind import MediaKind
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.persisting_processor import (
        PersistingProcessor,
    )
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from logger.project_logger import ProjectLogger

type RouteResult = tuple[ProcessorOutcome, bool]


class CrawlResultRouter:
    """Select handlers, hand off to analysis, and normalize handler multimodal."""

    def __init__(
        self,
        *,
        handlers_by_result_kind: Mapping[
            MediaKind,
            PersistingProcessor[Any, Any],
        ],
        analysis_router: AnalysisRouter | None,
        failure_handler: ProcessorFailureHandler,
        drop_unknown_tasks: bool,
        logger: ProjectLogger,
    ) -> None:
        self._handlers_by_result_kind = handlers_by_result_kind
        self._analysis_router = analysis_router
        self._failure_handler = failure_handler
        self._drop_unknown_tasks = drop_unknown_tasks
        self._logger = logger

    @property
    def analysis_router(self) -> AnalysisRouter | None:
        return self._analysis_router

    async def route(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
    ) -> RouteResult:
        """Route one fetch result; return outcome and whether payload cleanup is owed."""
        handler = self._handlers_by_result_kind.get(result.kind)

        if handler is None:
            return (
                self._handle_unknown_result_kind(
                    task=task,
                    result_kind=result.kind,
                ),
                True,
            )

        self._logger.debug(
            "task_routed_by_result_kind",
            task_id=task.task_id,
            url=task.url,
            requested_kind=task.kind,
            result_kind=result.kind,
            depth=task.depth,
            source_name=task.source_name,
            source_type=task.source_type,
            handler=type(handler).__name__,
        )

        return await self._process_with_handler(
            task=task,
            result=result,
            handler=handler,
        )

    async def _process_with_handler(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        handler: PersistingProcessor[Any, Any],
    ) -> RouteResult:
        try:
            router = self._analysis_router

            if router is not None and router.owns(kind=result.kind):
                outcome = await router.submit(
                    task=task,
                    fetch_result=result,
                    kind=result.kind,
                    processor=handler,
                )

                self._logger.debug(
                    "task_processor_payload_cleanup_delegated",
                    task_id=task.task_id,
                    url=task.url,
                    result_kind=result.kind,
                    final_url=result.final_url,
                )

                return (
                    self._coerce_handler_outcome(
                        task=task,
                        result_kind=result.kind,
                        outcome=outcome,
                    ),
                    False,
                )

            outcome = await handler.process_fetched(
                task=task,
                result=result,
            )

            return (
                self._coerce_handler_outcome(
                    task=task,
                    result_kind=result.kind,
                    outcome=outcome,
                ),
                True,
            )

        except IgnoredFetchError as exc:
            return (
                self._finalize_handler_failure(
                    task=task,
                    result_kind=result.kind,
                    outcome=self._failure_handler.ignored(
                        exc=exc,
                        stage="handler",
                    ),
                ),
                True,
            )

        except RetryableFetchError as exc:
            return (
                self._finalize_handler_failure(
                    task=task,
                    result_kind=result.kind,
                    outcome=self._failure_handler.retryable(
                        exc=exc,
                        stage="handler",
                    ),
                ),
                True,
            )

        except TransientLockRaceError as exc:
            return (
                self._finalize_handler_failure(
                    task=task,
                    result_kind=result.kind,
                    outcome=self._failure_handler.transient_lock_race(
                        exc=exc,
                        stage="handler",
                    ),
                ),
                True,
            )

        except (RuntimeError, OSError, ValueError) as exc:
            return (
                self._finalize_handler_failure(
                    task=task,
                    result_kind=result.kind,
                    outcome=self._failure_handler.handler_exception(
                        exc=exc,
                    ),
                ),
                True,
            )

    def _finalize_handler_failure(
        self,
        *,
        task: CrawlTask,
        result_kind: MediaKind,
        outcome: ProcessorOutcome,
    ) -> ProcessorOutcome:
        """Log one handler-boundary event for a pure failure outcome."""
        fields: dict[str, object] = dict(outcome.metadata)
        fields.update(
            {
                "task_id": task.task_id,
                "url": task.url,
                "requested_kind": task.kind,
                "result_kind": result_kind,
                "status": outcome.status,
                "stage": outcome.stage,
                "reason": outcome.reason,
                "detail": outcome.detail,
            }
        )
        if outcome.retry_after_seconds is not None:
            fields["retry_after_seconds"] = outcome.retry_after_seconds
        if outcome.error_type:
            fields["error_type"] = outcome.error_type
        if outcome.error:
            fields["error"] = outcome.error
        if outcome.status == "deferred":
            self._logger.warning("task_processor_deferred", **fields)
        elif outcome.reason == "handler_exception":
            self._logger.error("task_processor_handler_failed", **fields)
        else:
            self._logger.debug("task_processor_dropped", **fields)
        return outcome

    def _coerce_handler_outcome(
        self,
        *,
        task: CrawlTask,
        result_kind: MediaKind,
        outcome: object,
    ) -> ProcessorOutcome:
        if isinstance(outcome, ProcessorOutcome):
            return outcome

        self._logger.warning(
            "task_processor_dropped",
            task_id=task.task_id,
            url=task.url,
            requested_kind=task.kind,
            result_kind=result_kind,
            stage="handler",
            reason="missing_processor_outcome",
            returned_type=type(outcome).__name__,
        )

        return ProcessorOutcome.dropped(
            stage="handler",
            reason="missing_processor_outcome",
            metadata={"returned_type": type(outcome).__name__},
        )

    def _handle_unknown_result_kind(
        self,
        *,
        task: CrawlTask,
        result_kind: MediaKind,
    ) -> ProcessorOutcome:
        if not self._drop_unknown_tasks:
            raise ValueError(f"unknown fetch result kind: {result_kind}")

        self._logger.debug(
            "task_processor_dropped",
            task_id=task.task_id,
            url=task.url,
            requested_kind=task.kind,
            result_kind=result_kind,
            stage="routing",
            reason="unknown_result_kind",
        )

        return ProcessorOutcome.dropped(
            stage="routing",
            reason="unknown_result_kind",
        )
