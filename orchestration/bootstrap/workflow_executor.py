"""Coordinator for autonomous workflow phase execution.

Owns workflow-loop execution, phase routing, and the blocking executor.
Concrete runners are injected as an action-to-callable mapping, so this
module never depends on a specific runner.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from functools import partial
from typing import Any, TypeVar

from datachecker.data_checker import DataCheckerTimeoutError
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from logger.project_logger import ProjectLogger
from orchestration.workflow.phase import (
    PhaseOutcome,
    PhaseRunner,
    PhaseStatus,
)

_T = TypeVar("_T")

EXIT_SUCCESS = 0
EXIT_PARTIAL_DOWNSTREAM_INVALID = 2


def _workflow_progress_signature(plan: WorkflowExecutionPlan) -> str:
    """Return a stable signature for the complete semantic workflow plan."""

    payload = repr(
        (
            plan.action.value,
            plan.reason.value,
            tuple(sorted(plan.coverage_gaps.items())),
            str(plan.raw_run_directory) if plan.raw_run_directory else None,
            (
                str(plan.raw_records_manifest_path)
                if plan.raw_records_manifest_path
                else None
            ),
            plan.training_snapshot_id,
            str(plan.training_root) if plan.training_root else None,
            plan.dataset_manifest_hash,
            tuple(plan.details),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def run_blocking(
    semaphore: asyncio.Semaphore,
    func: Callable[..., _T],
    /,
    *args: Any,
    timeout_seconds: float | None = None,
    cancel: Callable[[], None] | None = None,
    **kwargs: Any,
) -> _T:
    """Run a blocking function on asyncio's shared executor.

    Capacity remains reserved until the underlying worker has really exited.
    Timeout and task cancellation both request cooperative cancellation when
    the caller supplied a ``cancel`` callback; neither abandons capacity while
    the worker can still be mutating workflow state.
    """
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    if timeout_seconds is None:
        await semaphore.acquire()
        worker_timeout = None
    else:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=timeout_seconds,
        )
        worker_timeout = max(
            0.0,
            timeout_seconds - (loop.time() - started_at),
        )
    released = False

    def release_once() -> None:
        nonlocal released
        if released:
            return
        semaphore.release()
        released = True

    def complete_worker(task: asyncio.Task[_T]) -> None:
        # Consume any terminal exception so a timed-out orphaned task cannot
        # emit "Task exception was never retrieved" later.
        with suppress(BaseException):
            task.result()
        release_once()

    call = partial(func, *args, **kwargs)
    worker_task = asyncio.create_task(asyncio.to_thread(call))

    try:
        protected_task = asyncio.shield(worker_task)
        if worker_timeout is None:
            result = await protected_task
        else:
            result = await asyncio.wait_for(
                protected_task,
                timeout=worker_timeout,
            )
    except (TimeoutError, asyncio.CancelledError):
        if cancel is not None:
            with suppress(Exception):
                cancel()
        if worker_task.done():
            complete_worker(worker_task)
        else:
            worker_task.add_done_callback(complete_worker)
        raise
    except BaseException:
        release_once()
        raise

    release_once()
    return result


class WorkflowPhaseExecutor:
    """Execute workflow phases selected by the DataChecker.

    Owns the complete execution lifecycle: optional fresh-run cleanup and
    crawl-state reconciliation, the checker-driven iteration loop, phase
    routing, and termination. The checker pass, cleanup, and reconciliation
    arrive as already-bound callables from the composition root, so this
    executor only knows check -> run phase -> interpret outcome -> iterate.
    """

    _CONTINUING_STATUSES = frozenset(
        {
            PhaseStatus.SUCCEEDED,
            PhaseStatus.RECRAWL_REQUESTED,
        }
    )

    def __init__(
        self,
        *,
        check: Callable[[], WorkflowExecutionPlan],
        runners: Mapping[WorkflowAction, PhaseRunner],
        cleanup: Callable[[], Awaitable[list[str]]] | None = None,
        reconcile_crawl: Callable[[], Awaitable[bool]] | None = None,
        max_iterations: int,
        iteration_pause_seconds: float,
        logger: ProjectLogger,
    ) -> None:
        self._check_workflow = check
        self._runners = runners
        self._cleanup = cleanup
        self._reconcile_crawl = reconcile_crawl
        self._max_iterations = max_iterations
        self._iteration_pause_seconds = iteration_pause_seconds
        self._logger = logger
        self._logger.debug("workflow_phase_executor_initialized")

    @property
    def logger(self) -> ProjectLogger:
        return self._logger

    async def execute(self) -> int:
        """Run the complete autonomous workflow.

        Owns the complete lifecycle: optional fresh-run cleanup, optional
        crawl-state reconciliation, and the checker-driven iteration loop
        with progress-stall and status-based termination.
        """

        self._logger.info(
            "data_workflow_started",
        )

        if self._cleanup is not None:
            removed_paths = await self._cleanup()

            self._logger.info(
                "workflow_fresh_run_cleanup_completed",
                removed_paths=len(removed_paths),
            )

        if self._reconcile_crawl is not None:
            self._logger.info(
                "crawl_state_reconciliation_started",
            )

            promoted = await self._reconcile_crawl()

            self._logger.info(
                "crawl_state_reconciliation_completed",
                promoted=bool(promoted),
            )
        else:
            self._logger.info(
                "crawl_state_reconciliation_skipped",
                reason="resume_disabled",
            )

        try:
            previous_progress_signature: str | None = None
            pending_plan: WorkflowExecutionPlan | None = None

            for iteration in range(1, self._max_iterations + 1):
                self._logger.debug(
                    "workflow_iteration_started",
                    iteration=iteration,
                )

                if pending_plan is None:
                    try:
                        plan = await self._check()
                    except DataCheckerTimeoutError as error:
                        self._logger.error(
                            "workflow_checker_timed_out",
                            iteration=iteration,
                            timeout_stage=error.stage,
                            timeout_seconds=error.timeout_seconds,
                            elapsed_seconds=round(error.elapsed_seconds, 3),
                        )
                        raise
                else:
                    plan = pending_plan
                    pending_plan = None
                    self._logger.info(
                        "workflow_pending_plan_selected",
                        iteration=iteration,
                        action=plan.action.value,
                        reason=plan.reason.value,
                        coverage_gaps=dict(plan.coverage_gaps),
                    )
                action = plan.action
                phase_name = (
                    action.name.lower()
                    if isinstance(action, WorkflowAction)
                    else str(action)
                )

                if action is WorkflowAction.NOOP:
                    self._logger.info(
                        "data_workflow_completed_successfully",
                        iterations_taken=iteration,
                        final_action=WorkflowAction.NOOP.value,
                    )
                    return EXIT_SUCCESS

                if action is WorkflowAction.BLOCKED:
                    self._logger.error(
                        "data_workflow_halted",
                        iteration=iteration,
                        phase_type=phase_name,
                        action=phase_name,
                        reason=plan.reason.value,
                    )
                    return EXIT_PARTIAL_DOWNSTREAM_INVALID

                progress_signature = _workflow_progress_signature(plan)
                if progress_signature == previous_progress_signature:
                    self._logger.error(
                        "workflow_progress_stalled",
                        iteration=iteration,
                        phase_type=phase_name,
                        action=phase_name,
                        reason=(
                            WorkflowDecisionReason.WORKFLOW_STATE_INCONSISTENT.value
                        ),
                        original_reason=plan.reason.value,
                        decision_signature=progress_signature,
                    )
                    return EXIT_PARTIAL_DOWNSTREAM_INVALID

                previous_progress_signature = progress_signature

                outcome = await self._run_phase(
                    iteration=iteration,
                    plan=plan,
                )

                if outcome.status not in self._CONTINUING_STATUSES:
                    self._logger.error(
                        "data_workflow_halted",
                        iteration=iteration,
                        phase_type=phase_name,
                        action=phase_name,
                        reason=plan.reason.value,
                    )
                    return EXIT_PARTIAL_DOWNSTREAM_INVALID

                if outcome.next_plan is not None:
                    pending_plan = outcome.next_plan
                    self._logger.info(
                        "workflow_recrawl_plan_scheduled",
                        iteration=iteration,
                        action=pending_plan.action.value,
                        reason=pending_plan.reason.value,
                        coverage_gaps=dict(pending_plan.coverage_gaps),
                    )

                if self._iteration_pause_seconds > 0:
                    await asyncio.sleep(self._iteration_pause_seconds)

            self._logger.critical(
                "data_workflow_failed",
                reason="exceeded_max_iterations",
                max_iterations=self._max_iterations,
            )
            raise RuntimeError(
                f"Workflow exceeded {self._max_iterations} iterations without "
                "completing."
            )
        except Exception as exc:
            self._logger.exception(
                "workflow_executor_uncaught_error",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            raise

    async def _check(self) -> WorkflowExecutionPlan:
        """Run one checker pass without abandoning its worker on cancellation."""

        checker_task = asyncio.create_task(
            asyncio.to_thread(self._check_workflow)
        )

        try:
            report = await asyncio.shield(checker_task)
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(checker_task)
            except BaseException as checker_error:
                raise checker_error from cancellation
            raise
        if not isinstance(report, WorkflowExecutionPlan):
            raise TypeError(
                "Workflow checker returned an unsupported report type: "
                f"{type(report).__name__}. Expected WorkflowExecutionPlan."
            )

        return report

    async def _run_phase(
        self,
        *,
        iteration: int,
        plan: WorkflowExecutionPlan,
    ) -> PhaseOutcome:
        """Run exactly the workflow phase selected by the DataChecker."""

        action = plan.action
        phase_name = (
            action.name.lower()
            if isinstance(action, WorkflowAction)
            else str(action)
        )

        self._logger.info(
            "workflow_phase_started",
            iteration=iteration,
            phase_type=phase_name,
            action=phase_name,
            reason=plan.reason.value,
        )

        phase_started_at = time.monotonic()

        try:
            runner = self._runners.get(action)
            if runner is None:
                raise TypeError(f"Unsupported workflow action: {action!r}")
            outcome = await runner(plan)
        except Exception as exc:
            self._logger.exception(
                "workflow_phase_failed",
                iteration=iteration,
                phase_type=phase_name,
                action=phase_name,
                reason=plan.reason.value,
                duration_seconds=round(
                    time.monotonic() - phase_started_at,
                    3,
                ),
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            raise

        self._logger.info(
            "workflow_phase_completed",
            iteration=iteration,
            phase_type=phase_name,
            action=phase_name,
            reason=plan.reason.value,
            duration_seconds=round(
                time.monotonic() - phase_started_at,
                3,
            ),
            phase_status=outcome.status.value,
            recrawl_requested=(outcome.next_plan is not None),
        )

        return outcome
