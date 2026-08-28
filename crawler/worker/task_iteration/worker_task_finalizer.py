"""Finalize worker task completion, callbacks, session state, and logs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import WorkerPoolSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
    from crawler.worker.task_iteration.worker_task_result_persister import (
        WorkerTaskResultPersister,
    )
    from crawler.worker.worker_loop.worker_session_tracker import (
        WorkerSessionTracker,
    )
    from crawler.worker.worker_loop.worker_state import WorkerState


class WorkerTaskFinalizer:
    """Build completion evidence and close one worker task session."""

    def __init__(
        self,
        *,
        settings: WorkerPoolSettings,
        logger: ProjectLogger,
        persister: WorkerTaskResultPersister,
        session_tracker: WorkerSessionTracker,
        register_failure: Callable[..., bool],
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._persister = persister
        self._session_tracker = session_tracker
        self._register_failure = register_failure

    async def finalize(
        self,
        *,
        task: CrawlTask,
        outcome: ProcessorOutcome | None,
        runtime_outcome: str | None,
        error: BaseException | None,
        timeout_origin: str | None,
        timeout_elapsed_seconds: float | None,
        wait_seconds: float | None,
        worker_id: int,
        state: WorkerState,
    ) -> None:
        completion_outcome, completion_fields = self._build_completion(
            task=task,
            outcome=outcome,
            runtime_outcome=runtime_outcome,
            error=error,
            timeout_origin=timeout_origin,
            timeout_elapsed_seconds=timeout_elapsed_seconds,
            wait_seconds=wait_seconds,
        )

        completion_error: BaseException | None = None
        callback_error: BaseException | None = None
        duration_seconds = 0.0

        try:
            (
                completion_error,
                callback_error,
            ) = await self._persister.complete_and_emit(
                task=task,
                completion_outcome=completion_outcome,
                completion_fields=completion_fields,
                outcome=outcome,
                skip_callback=runtime_outcome
                in {
                    "cancelled",
                    "interrupted",
                },
            )
        finally:
            duration_seconds = self._session_tracker.mark_task_finished(
                state=state,
                outcome=completion_outcome,
                task=task,
            )

        self._log_result(
            task=task,
            outcome=outcome,
            runtime_outcome=runtime_outcome,
            timeout_origin=timeout_origin,
            timeout_elapsed_seconds=timeout_elapsed_seconds,
            worker_id=worker_id,
            duration_seconds=duration_seconds,
        )

        if callback_error is not None:
            self._register_failure(
                worker_id=worker_id,
                task=task,
                cause=callback_error,
                fatal=False,
            )

        if completion_error is not None:
            self._register_failure(
                worker_id=worker_id,
                task=task,
                cause=completion_error,
                fatal=False,
            )

    def _build_completion(
        self,
        *,
        task: CrawlTask,
        outcome: ProcessorOutcome | None,
        runtime_outcome: str | None,
        error: BaseException | None,
        timeout_origin: str | None,
        timeout_elapsed_seconds: float | None,
        wait_seconds: float | None,
    ) -> tuple[str, dict[str, object]]:
        completion_outcome = (
            outcome.status
            if outcome is not None
            else runtime_outcome or "failed"
        )
        completion_fields: dict[str, object]
        if outcome is not None:
            completion_fields = dict(outcome.metadata)
            completion_fields.update(
                {
                    "status": outcome.status,
                    "stage": outcome.stage,
                    "reason": outcome.reason,
                    "detail": outcome.detail,
                }
            )
            if outcome.retry_after_seconds is not None:
                completion_fields["retry_after_seconds"] = (
                    outcome.retry_after_seconds
                )
            if outcome.retry_class:
                completion_fields["retry_class"] = outcome.retry_class
            if outcome.retry_error_kind:
                completion_fields["retry_error_kind"] = (
                    outcome.retry_error_kind
                )
            if outcome.counts_toward_task_retry_budget:
                completion_fields["counts_toward_task_retry_budget"] = True
            if outcome.terminal_eligible:
                completion_fields["terminal_eligible"] = True
            if outcome.error_type:
                completion_fields["error_type"] = outcome.error_type
            if outcome.error:
                completion_fields["error"] = outcome.error
        else:
            completion_fields = {
                "kind": task.kind,
                "stage": "worker_runtime",
                "reason": completion_outcome,
            }

        if timeout_origin is not None:
            completion_fields.update(
                {
                    "timeout": True,
                    "timeout_origin": timeout_origin,
                    "elapsed_seconds": timeout_elapsed_seconds,
                    "configured_worker_timeout_seconds": float(
                        self._settings.processing_timeout_seconds,
                    ),
                }
            )

        if error is not None:
            completion_fields["error_type"] = type(error).__name__
            completion_fields["error"] = str(error)

            observed_bytes = getattr(error, "observed_bytes", None)
            if observed_bytes is not None:
                completion_fields["observed_bytes"] = observed_bytes

            partial_path = getattr(error, "partial_path", None)
            if partial_path is not None:
                completion_fields["partial_path"] = str(partial_path)

        return completion_outcome, completion_fields

    def _log_result(
        self,
        *,
        task: CrawlTask,
        outcome: ProcessorOutcome | None,
        runtime_outcome: str | None,
        timeout_origin: str | None,
        timeout_elapsed_seconds: float | None,
        worker_id: int,
        duration_seconds: float,
    ) -> None:
        if runtime_outcome == "cancelled":
            self._logger.warning(
                "worker_task_cancelled",
                worker_id=worker_id,
                task_id=task.task_id,
                url=task.url,
                kind=task.kind,
            )

        if runtime_outcome == "timeout":
            processing_timeout_seconds = float(
                self._settings.processing_timeout_seconds,
            )
            self._logger.warning(
                "worker_task_timed_out",
                worker_id=worker_id,
                task_id=task.task_id,
                url=task.url,
                kind=task.kind,
                seconds=duration_seconds,
                timeout_seconds=processing_timeout_seconds,
                configured_worker_timeout_seconds=processing_timeout_seconds,
                elapsed_seconds=(timeout_elapsed_seconds or duration_seconds),
                timeout_origin=timeout_origin,
            )

        if outcome is not None and self._logger.is_debug_enabled():
            self._logger.debug(
                "url_finished",
                worker_id=worker_id,
                task_id=task.task_id,
                url=task.url,
                status=outcome.status,
                kind=task.kind,
                seconds=duration_seconds,
            )
