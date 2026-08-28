"""Scheduler admission for selected page-discovery tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.exceptions.crawler_error import CrawlerRuntimeError
from crawler.scheduling.admission.admission_pressure_rules import (
    is_coverage_recovery_target_task,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
)
from logger.project_logger import ProjectLogger
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.scheduling.url_scheduler import UrlScheduler


class SchedulerAdmissionContractError(CrawlerRuntimeError):
    """Raised when scheduler batch results violate the admission contract."""


class PageDiscoverySchedulerAdmission:
    def __init__(
        self,
        *,
        scheduler: UrlScheduler,
        logger: ProjectLogger,
        id_generator: IdGenerator,
    ) -> None:
        if id_generator is None:
            raise ValueError("id_generator is required")

        self._scheduler = scheduler
        self._logger = logger
        self._id_generator = id_generator

    async def admit_selected_tasks(
        self,
        *,
        parent_url: str,
        selected_tasks: tuple[CrawlTask, ...],
    ) -> tuple[tuple[ScheduleDecision, ...], tuple[CrawlTask, ...]]:
        if not selected_tasks:
            return (), ()

        try:
            drain_budget = await self._scheduler.discovery_drain_budget(
                configured_cap=len(selected_tasks),
                force=False,
            )
        except Exception as exc:  # Fail closed when capacity is unknown.
            self._logger.exception(
                "discovery_drain_budget_failed",
                url=parent_url,
                selected=len(selected_tasks),
                error_type=type(exc).__name__,
            )
            return (
                self.rejection_decisions_for_unsubmitted_tasks(
                    selected_tasks=selected_tasks,
                    reason=ScheduleDecisionReason.SCHEDULER_UNAVAILABLE,
                ),
                (),
            )

        try:
            identified_tasks = self.tasks_with_stable_ids(
                selected_tasks=selected_tasks,
            )
        except SchedulerAdmissionContractError as exc:
            self._log_contract_violation(
                parent_url=parent_url,
                selected_count=len(selected_tasks),
                submitted_count=0,
                decision_count=0,
                error=exc,
            )
            raise

        submitted_tasks = self.tasks_allowed_under_admission_pressure(
            selected_tasks=identified_tasks,
            drain_budget=drain_budget,
        )

        if not submitted_tasks:
            self._logger.info(
                "asset_admission_skipped",
                url=parent_url,
                selected=len(selected_tasks),
                reason="scheduler_backpressure",
            )
            return (
                self.rejection_decisions_for_unsubmitted_tasks(
                    selected_tasks=identified_tasks,
                    reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                ),
                (),
            )

        scheduler_decisions = await self._scheduler.enqueue_many(
            submitted_tasks
        )

        try:
            decisions = self.align_admission_decisions(
                selected_tasks=identified_tasks,
                submitted_tasks=submitted_tasks,
                scheduler_decisions=scheduler_decisions,
            )
        except SchedulerAdmissionContractError as exc:
            self._log_contract_violation(
                parent_url=parent_url,
                selected_count=len(identified_tasks),
                submitted_count=len(submitted_tasks),
                decision_count=len(scheduler_decisions),
                error=exc,
            )
            raise
        return decisions, submitted_tasks

    @staticmethod
    def tasks_allowed_under_admission_pressure(
        *,
        selected_tasks: tuple[CrawlTask, ...],
        drain_budget: int,
    ) -> tuple[CrawlTask, ...]:
        submitted: list[CrawlTask] = []
        remaining_non_target_budget = max(0, int(drain_budget))

        for task in selected_tasks:
            if is_coverage_recovery_target_task(task):
                submitted.append(task)
                continue

            if remaining_non_target_budget <= 0:
                continue

            submitted.append(task)
            remaining_non_target_budget -= 1

        return tuple(submitted)

    def tasks_with_stable_ids(
        self,
        *,
        selected_tasks: tuple[CrawlTask, ...],
    ) -> tuple[CrawlTask, ...]:
        """Return selected tasks with unique, process-independent IDs."""

        identified_tasks = tuple(
            task.ensure_id(id_generator=self._id_generator)
            for task in selected_tasks
        )
        PageDiscoverySchedulerAdmission._task_index_by_id(
            tasks=identified_tasks,
            collection_name="selected_tasks",
        )
        return identified_tasks

    @staticmethod
    def align_admission_decisions(
        *,
        selected_tasks: tuple[CrawlTask, ...],
        submitted_tasks: tuple[CrawlTask, ...],
        scheduler_decisions: tuple[ScheduleDecision, ...],
    ) -> tuple[ScheduleDecision, ...]:
        """Align scheduler results by stable task ID under a strict contract."""

        submitted_count = len(submitted_tasks)
        decision_count = len(scheduler_decisions)
        if decision_count != submitted_count:
            raise SchedulerAdmissionContractError(
                "scheduler returned an invalid decision count: "
                f"submitted={submitted_count}, returned={decision_count}"
            )

        selected_by_task_id = (
            PageDiscoverySchedulerAdmission._task_index_by_id(
                tasks=selected_tasks,
                collection_name="selected_tasks",
            )
        )
        submitted_by_task_id = (
            PageDiscoverySchedulerAdmission._task_index_by_id(
                tasks=submitted_tasks,
                collection_name="submitted_tasks",
            )
        )
        unknown_submitted_ids = set(submitted_by_task_id).difference(
            selected_by_task_id
        )
        if unknown_submitted_ids:
            raise SchedulerAdmissionContractError(
                "submitted tasks are not present in selected tasks: "
                f"task_ids={sorted(unknown_submitted_ids)!r}"
            )

        decision_by_task_id: dict[str, ScheduleDecision] = {}
        for task, decision in zip(
            submitted_tasks,
            scheduler_decisions,
            strict=True,
        ):
            if not isinstance(decision, ScheduleDecision):
                raise SchedulerAdmissionContractError(
                    "scheduler returned an invalid decision type: "
                    f"{type(decision).__name__}"
                )

            task_id = PageDiscoverySchedulerAdmission._required_task_id(task)
            if decision.accepted:
                if decision.task is None:
                    raise SchedulerAdmissionContractError(
                        "accepted scheduler decision has no task: "
                        f"task_id={task_id!r}"
                    )
                decision_task_id = (
                    PageDiscoverySchedulerAdmission._required_task_id(
                        decision.task
                    )
                )
                if decision_task_id != task_id:
                    raise SchedulerAdmissionContractError(
                        "accepted scheduler decision task ID mismatch: "
                        f"submitted={task_id!r}, "
                        f"returned={decision_task_id!r}"
                    )

            decision_by_task_id[task_id] = decision

        aligned: list[ScheduleDecision] = []
        for task in selected_tasks:
            task_id = PageDiscoverySchedulerAdmission._required_task_id(task)
            if task_id in decision_by_task_id:
                aligned.append(decision_by_task_id[task_id])
                continue

            aligned.append(
                ScheduleDecision.reject(
                    reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                    normalized_url=task.url,
                )
            )

        return tuple(aligned)

    @staticmethod
    def _task_index_by_id(
        *,
        tasks: tuple[CrawlTask, ...],
        collection_name: str,
    ) -> dict[str, CrawlTask]:
        indexed: dict[str, CrawlTask] = {}
        for task in tasks:
            task_id = PageDiscoverySchedulerAdmission._required_task_id(task)
            if task_id in indexed:
                raise SchedulerAdmissionContractError(
                    f"{collection_name} contains duplicate task ID {task_id!r}"
                )
            indexed[task_id] = task
        return indexed

    @staticmethod
    def _required_task_id(task: CrawlTask) -> str:
        task_id = str(task.task_id or "").strip()
        if not task_id:
            raise SchedulerAdmissionContractError(
                f"task has no stable identifier: url={task.url!r}"
            )
        return task_id

    def _log_contract_violation(
        self,
        *,
        parent_url: str,
        selected_count: int,
        submitted_count: int,
        decision_count: int,
        error: SchedulerAdmissionContractError,
    ) -> None:
        self._logger.error(
            "discovery_scheduler_contract_violation",
            url=parent_url,
            selected=selected_count,
            submitted=submitted_count,
            returned_decisions=decision_count,
            error_type=type(error).__name__,
            error=str(error),
        )

    @staticmethod
    def rejection_decisions_for_unsubmitted_tasks(
        *,
        selected_tasks: tuple[CrawlTask, ...],
        reason: ScheduleDecisionReason,
    ) -> tuple[ScheduleDecision, ...]:
        return tuple(
            ScheduleDecision.reject(
                reason=reason,
                normalized_url=task.url,
            )
            for task in selected_tasks
        )

    @staticmethod
    def backpressure_decisions_for_unsubmitted_tasks(
        *,
        selected_tasks: tuple[CrawlTask, ...],
    ) -> tuple[ScheduleDecision, ...]:
        return PageDiscoverySchedulerAdmission.rejection_decisions_for_unsubmitted_tasks(
            selected_tasks=selected_tasks,
            reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
        )
