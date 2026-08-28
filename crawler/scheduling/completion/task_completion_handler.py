"""Task completion handling for the URL scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.runtime.concurrency import condition_notify_all
from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)

from ..queueing.delayed_task_queue import DelayedTaskQueue
from .dead_letter_writer import DeadLetterRecord, DeadLetterStatus

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.scheduling.completion.dead_letter_writer import (
        DeadLetterWriter,
    )
    from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from logger.project_logger import ProjectLogger

    from ..progress.active_task_registry import (
        ActiveTaskRecord,
        ActiveTaskRegistry,
    )
    from ..progress.scheduler_progress_state import SchedulerProgressState
    from ..queueing.host_eligibility_queue import HostEligibilityQueue
    from ..queueing.host_task_queue import HostTaskQueue
    from .scheduler_retry_budget import SchedulerRetryBudget


@dataclass(frozen=True, slots=True)
class _CompletionDisposition:
    requeued: bool
    delayed_requeue: bool
    forget_identity: bool = False
    dead_letter_record: DeadLetterRecord | None = None


@dataclass(frozen=True, slots=True)
class _CompletionCounts:
    total_queued: int
    ready_queued: int
    delayed_queued: int
    inflight: int


class SchedulerCompletionHandler:
    """Complete active tasks and update scheduler accounting."""

    def __init__(
        self,
        *,
        condition: asyncio.Condition,
        active_registry: ActiveTaskRegistry,
        retry_rules: SchedulerRetryBudget,
        seen_urls: SeenUrlRegistry,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        progress_state: SchedulerProgressState,
        run_url_feedback: RunUrlFeedback,
        logger: ProjectLogger,
        is_closed: Callable[[], bool],
        dead_letter_writer: DeadLetterWriter | None = None,
        host_eligibility_queue: HostEligibilityQueue | None = None,
    ) -> None:
        self._condition = condition
        self._active_registry = active_registry
        self._retry_rules = retry_rules
        self._seen_urls = seen_urls
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._progress_state = progress_state
        self._run_url_feedback = run_url_feedback
        self._logger = logger
        self._is_closed = is_closed
        self._dead_letter_writer = dead_letter_writer
        self._host_eligibility_queue = host_eligibility_queue

    async def complete(
        self,
        task: CrawlTask,
        *,
        outcome: str = "completed",
        fields: dict[str, object] | None = None,
    ) -> None:
        pending_dead_letter = False
        async with self._condition:
            if self._active_registry.count == 0:
                self._logger.warning(
                    "scheduler_completion_without_active_tasks",
                    task_id=task.task_id,
                    url=task.url,
                    outcome=outcome,
                )
                return

            record = self._active_registry.remove(task=task)
            if record is None:
                self._logger.warning(
                    "scheduler_completion_unknown_active_task",
                    task_id=task.task_id,
                    url=task.url,
                    outcome=outcome,
                )
                return

            if self._host_eligibility_queue is not None:
                self._host_eligibility_queue.release_blocked(
                    host=record.host,
                )

            disposition = self._requeue_active_record_locked(
                record=record,
                outcome=outcome,
                fields=fields,
            )

            pending_dead_letter = (
                disposition.dead_letter_record is not None
                and self._dead_letter_writer is not None
            )
            if pending_dead_letter:
                self._active_registry.stage_dead_letter(record=record)
                condition_notify_all(self._condition)
                counts = None
            else:
                counts = self._finalize_completion_locked(
                    record=record,
                    outcome=outcome,
                    fields=fields,
                    disposition=disposition,
                )

        if pending_dead_letter:
            dead_letter_record = disposition.dead_letter_record
            dead_letter_writer = self._dead_letter_writer
            if dead_letter_record is None or dead_letter_writer is None:
                raise RuntimeError(
                    "dead-letter persistence was not configured"
                )

            persistence_task = asyncio.create_task(
                self._persist_and_finalize_dead_letter(
                    record=record,
                    dead_letter_record=dead_letter_record,
                    dead_letter_writer=dead_letter_writer,
                    outcome=outcome,
                    fields=fields,
                    disposition=disposition,
                ),
                name="scheduler-dead-letter-persistence",
            )
            try:
                counts = await asyncio.shield(persistence_task)
            except asyncio.CancelledError as cancellation:
                await self._wait_for_critical_task(persistence_task)
                raise cancellation

        if counts is None:
            return

        self._logger.debug(
            "task_completed",
            task_id=task.task_id,
            url=task.url,
            outcome=outcome,
            queued=counts.total_queued,
            ready_queued=counts.ready_queued,
            delayed_queued=counts.delayed_queued,
            inflight=counts.inflight,
            requeued=disposition.requeued,
            delayed_requeue=disposition.delayed_requeue,
        )

    def _requeue_active_record_locked(
        self,
        *,
        record: ActiveTaskRecord,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> _CompletionDisposition:
        # Persist terminal outcomes before shutdown prevents requeueing.
        if outcome in {"failed", "failure"}:
            return _CompletionDisposition(
                requeued=False,
                delayed_requeue=False,
                forget_identity=True,
                dead_letter_record=self._dead_letter_record(
                    record=record,
                    status="failed",
                    outcome=outcome,
                    detail=self._completion_detail(
                        outcome=outcome,
                        fields=fields,
                    ),
                    fields=fields,
                ),
            )

        if outcome == "interrupted":
            return _CompletionDisposition(
                requeued=False,
                delayed_requeue=False,
                forget_identity=True,
                dead_letter_record=self._dead_letter_record(
                    record=record,
                    status="cancelled",
                    outcome=outcome,
                    detail=self._completion_detail(
                        outcome=outcome,
                        fields=fields,
                    ),
                    fields=fields,
                ),
            )

        if outcome == "cancelled" and self._is_closed():
            return _CompletionDisposition(
                requeued=False,
                delayed_requeue=False,
                forget_identity=True,
                dead_letter_record=self._dead_letter_record(
                    record=record,
                    status="cancelled",
                    outcome=outcome,
                    detail=self._completion_detail(
                        outcome=outcome,
                        fields=fields,
                    ),
                    fields=fields,
                ),
            )

        # A graceful early-success shutdown keeps active tasks alive so their
        # writes can finish, but it must never repopulate the frontier once
        # output readiness has stopped discovery.
        if self._is_closed():
            return _CompletionDisposition(
                requeued=False,
                delayed_requeue=False,
                forget_identity=True,
            )

        if outcome not in {"cancelled", "deferred", "timeout"}:
            return _CompletionDisposition(False, False)

        retry_decision = self._retry_rules.evaluate(
            task=record.task,
            outcome=outcome,
            fields=fields,
        )

        if retry_decision.terminal:
            self._logger.warning(
                "scheduler_task_retry_exhausted",
                task_id=record.task.task_id,
                url=record.task.url,
                kind=record.task.kind,
                outcome=outcome,
                reason=retry_decision.reason,
                fields=fields or {},
                action=(
                    "dead_letter"
                    if self._dead_letter_writer is not None
                    else "drop_from_frontier"
                ),
            )
            return _CompletionDisposition(
                requeued=False,
                delayed_requeue=False,
                forget_identity=True,
                dead_letter_record=self._dead_letter_record(
                    record=record,
                    status="retry_exhausted",
                    outcome=outcome,
                    detail=(retry_decision.reason or "retry_exhausted"),
                    fields=fields,
                ),
            )

        delayed_wait_seconds = DelayedTaskQueue.coerce_requeue_wait_seconds(
            outcome=outcome,
            fields=fields,
        )

        if outcome == "timeout" and delayed_wait_seconds is None:
            delayed_wait_seconds = (
                self._retry_rules.timeout_retry_wait_seconds()
            )

        if delayed_wait_seconds is not None:
            delayed_requeue = self._delayed_queue.push(
                host=record.host,
                priority=record.priority,
                sequence=record.sequence,
                task=record.task,
                wait_seconds=delayed_wait_seconds,
            )
            if delayed_requeue:
                condition_notify_all(self._condition)
                return _CompletionDisposition(True, True)

        self._host_queue.push(
            host=record.host,
            priority=record.priority,
            sequence=record.sequence,
            task=record.task,
        )
        condition_notify_all(self._condition)
        return _CompletionDisposition(True, False)

    def _finalize_completion_locked(
        self,
        *,
        record: ActiveTaskRecord,
        outcome: str,
        fields: dict[str, object] | None,
        disposition: _CompletionDisposition,
    ) -> _CompletionCounts:
        if is_not_modified_completion(fields=fields):
            self._run_url_feedback.remember_not_modified(
                task=record.task,
                url=record.task.url,
            )

        if fields is not None and fields.get("status_code") == 403:
            final_url = fields.get("final_url")
            self._run_url_feedback.remember_forbidden_endpoint(
                url=(str(final_url) if final_url else record.task.url),
            )

        if not disposition.requeued:
            self._retry_rules.forget(task=record.task)

        if disposition.forget_identity:
            identity = scheduler_task_identity_key(task=record.task)
            self._seen_urls.forget(identity)

        ready_queued = self._host_queue.queue_size
        delayed_queued = self._delayed_queue.queue_size
        self._progress_state.record_completed_outcome(outcome=outcome)
        if outcome == "deferred" and isinstance(fields, dict):
            reason = fields.get("reason")
            self._progress_state.record_task_deferral(
                reason=str(reason) if reason is not None else None,
            )
        condition_notify_all(self._condition)
        return _CompletionCounts(
            total_queued=ready_queued + delayed_queued,
            ready_queued=ready_queued,
            delayed_queued=delayed_queued,
            inflight=self._active_registry.count,
        )

    async def _requeue_after_dead_letter_write_failure(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        error: BaseException,
    ) -> None:
        async with self._condition:
            record = self._active_registry.remove_dead_letter_pending(
                task=task,
            )
            if record is None:
                return
            self._host_queue.push(
                host=record.host,
                priority=record.priority,
                sequence=record.sequence,
                task=record.task,
            )
            condition_notify_all(self._condition)

        self._logger.error(
            "scheduler_dead_letter_write_failed_requeued",
            task_id=task.task_id,
            url=task.url,
            outcome=outcome,
            error_type=type(error).__name__,
            error=str(error),
            action="requeue",
        )

    async def _persist_and_finalize_dead_letter(
        self,
        *,
        record: ActiveTaskRecord,
        dead_letter_record: DeadLetterRecord,
        dead_letter_writer: DeadLetterWriter,
        outcome: str,
        fields: dict[str, object] | None,
        disposition: _CompletionDisposition,
    ) -> _CompletionCounts | None:
        try:
            await dead_letter_writer.append(dead_letter_record)
        except BaseException as exc:
            recovery_task = asyncio.create_task(
                self._requeue_after_dead_letter_write_failure(
                    task=record.task,
                    outcome=outcome,
                    error=exc,
                ),
                name="scheduler-dead-letter-write-recovery",
            )
            await self._wait_for_critical_task(recovery_task)
            raise

        async with self._condition:
            persisted_record = (
                self._active_registry.remove_dead_letter_pending(
                    task=record.task,
                )
            )
            if persisted_record is None:
                self._logger.warning(
                    "scheduler_dead_letter_pending_task_missing",
                    task_id=record.task.task_id,
                    url=record.task.url,
                    outcome=outcome,
                )
                return None
            return self._finalize_completion_locked(
                record=persisted_record,
                outcome=outcome,
                fields=fields,
                disposition=disposition,
            )

    @staticmethod
    async def _wait_for_critical_task(task: asyncio.Task[object]) -> object:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    @staticmethod
    def _dead_letter_record(
        *,
        record: ActiveTaskRecord,
        status: DeadLetterStatus,
        outcome: str,
        detail: str,
        fields: dict[str, object] | None,
    ) -> DeadLetterRecord:
        return DeadLetterRecord(
            task=record.task,
            status=status,
            original_outcome=outcome,
            detail=detail,
            fields=dict(fields or {}),
        )

    @staticmethod
    def _completion_detail(
        *,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> str:
        if fields is not None:
            for key in ("detail", "reason", "error"):
                value = fields.get(key)
                if value is None:
                    continue
                detail = str(value).strip()
                if detail:
                    return detail
        return outcome


def is_not_modified_completion(
    *,
    fields: dict[str, object] | None,
) -> bool:
    if not isinstance(fields, dict):
        return False
    status_code = fields.get("status_code")
    reason = str(fields.get("reason", "") or "").strip().lower()
    retry_kind = str(fields.get("retry_error_kind", "") or "").strip().lower()
    # Only treat as terminal not_modified if it is not the "force one unconditional" recovery case.
    if retry_kind == "not_modified_force_unconditional":
        return False
    return status_code == 304 or reason in {
        "not_modified",
        "not_modified_this_run",
    }
