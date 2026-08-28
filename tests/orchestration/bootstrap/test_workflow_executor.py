from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from functools import partial

import pytest

from datachecker.data_checker import DataCheckerTimeoutError
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from orchestration.bootstrap.workflow_executor import (
    EXIT_PARTIAL_DOWNSTREAM_INVALID,
    WorkflowPhaseExecutor,
)
from orchestration.workflow.phase import (
    PhaseOutcome,
    PhaseStatus,
)


class _Logger:
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, object]]] = []
        self.infos: list[tuple[str, dict[str, object]]] = []

    def error(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))

    def debug(self, _event: str, **_fields: object) -> None:
        return None

    def info(self, event: str, **fields: object) -> None:
        self.infos.append((event, fields))

    def exception(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))


def _report() -> WorkflowExecutionPlan:
    return WorkflowExecutionPlan(
        action=WorkflowAction.NOOP,
        reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        details=(),
    )


class _ReturningChecker:
    def __init__(self) -> None:
        self.timeout_seconds: float | None = None

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        self.timeout_seconds = timeout_seconds
        return _report()


@pytest.mark.asyncio
async def test_run_data_checker_passes_the_configured_deadline() -> None:
    checker = _ReturningChecker()
    executor = _executor(
        check=partial(checker.check, timeout_seconds=2.5),
        logger=_Logger(),
    )

    report = await executor._check()

    assert report is not None
    assert checker.timeout_seconds == 2.5


class _InvalidChecker:
    def check(self, *, timeout_seconds: float) -> object:
        del timeout_seconds
        return object()


@pytest.mark.asyncio
async def test_run_data_checker_rejects_an_invalid_report() -> None:
    executor = _executor(
        check=partial(_InvalidChecker().check, timeout_seconds=1.0),
        logger=_Logger(),
    )

    with pytest.raises(TypeError, match="unsupported report type"):
        await executor._check()


class _TimeoutChecker:
    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        raise DataCheckerTimeoutError(
            stage="validation",
            timeout_seconds=timeout_seconds,
            elapsed_seconds=0.125,
        )


def _executor(
    *,
    check: Callable[[], WorkflowExecutionPlan],
    logger: _Logger,
    max_iterations: int = 1,
    cleanup: Callable[[], Awaitable[list[str]]] | None = None,
) -> WorkflowPhaseExecutor:
    return WorkflowPhaseExecutor(
        check=check,
        runners={},
        cleanup=cleanup,
        max_iterations=max_iterations,
        iteration_pause_seconds=0.0,
        logger=logger,
    )


@pytest.mark.asyncio
async def test_run_data_checker_logs_and_reraises_timeout() -> None:
    logger = _Logger()

    with pytest.raises(DataCheckerTimeoutError):
        await _executor(
            check=partial(_TimeoutChecker().check, timeout_seconds=1.5),
            logger=logger,
        ).execute()

    assert logger.errors == [
        (
            "workflow_checker_timed_out",
            {
                "iteration": 1,
                "timeout_stage": "validation",
                "timeout_seconds": 1.5,
                "elapsed_seconds": 0.125,
            },
        ),
        (
            "workflow_executor_uncaught_error",
            {
                "error_type": "DataCheckerTimeoutError",
                "error_message": (
                    "DataChecker exceeded its execution budget during "
                    "'validation': 0.125s elapsed, 1.500s allowed."
                ),
            },
        ),
    ]


class _BlockingChecker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        del timeout_seconds
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test checker was not released")
        self.completed.set()
        return _report()


@pytest.mark.asyncio
async def test_run_data_checker_waits_for_worker_before_cancellation() -> None:
    checker = _BlockingChecker()
    executor = _executor(
        check=partial(checker.check, timeout_seconds=1.0),
        logger=_Logger(),
    )
    task = asyncio.create_task(executor._check())

    try:
        await asyncio.wait_for(asyncio.to_thread(checker.started.wait), 1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        checker.release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1.0)
    finally:
        checker.release.set()

    assert checker.completed.is_set()


class _FailingChecker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        del timeout_seconds
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test checker was not released")
        raise RuntimeError("checker worker failed")


@pytest.mark.asyncio
async def test_cancellation_preserves_a_worker_failure() -> None:
    checker = _FailingChecker()
    executor = _executor(
        check=partial(checker.check, timeout_seconds=1.0),
        logger=_Logger(),
    )
    task = asyncio.create_task(executor._check())

    try:
        await asyncio.wait_for(asyncio.to_thread(checker.started.wait), 1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        checker.release.set()

        with pytest.raises(
            RuntimeError, match="checker worker failed"
        ) as raised:
            await asyncio.wait_for(task, 1.0)
    finally:
        checker.release.set()

    assert isinstance(raised.value.__cause__, asyncio.CancelledError)


class _RepeatingChecker:
    def __init__(self, report: WorkflowExecutionPlan) -> None:
        self._report = report
        self.calls = 0

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        del timeout_seconds
        self.calls += 1
        return self._report


def _crawl_required_report() -> WorkflowExecutionPlan:
    return WorkflowExecutionPlan(
        action=WorkflowAction.CRAWL,
        reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
        details=("raw crawl run is not a completed final run",),
    )


@pytest.mark.asyncio
async def test_workflow_halts_when_checker_decision_does_not_change() -> None:
    logger = _Logger()
    checker = _RepeatingChecker(_crawl_required_report())
    phase_calls = 0

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        nonlocal phase_calls
        phase_calls += 1
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    executor = WorkflowPhaseExecutor(
        check=partial(checker.check, timeout_seconds=1.0),
        runners={WorkflowAction.CRAWL: runner},
        max_iterations=50,
        iteration_pause_seconds=0.0,
        logger=logger,
    )

    result = await executor.execute()

    assert result == EXIT_PARTIAL_DOWNSTREAM_INVALID
    assert checker.calls == 2
    assert phase_calls == 1
    assert logger.errors[-1][0] == "workflow_progress_stalled"
    assert logger.errors[-1][1]["reason"] == (
        WorkflowDecisionReason.WORKFLOW_STATE_INCONSISTENT.value
    )
    assert logger.errors[-1][1]["original_reason"] == (
        WorkflowDecisionReason.CRAWL_OUTPUT_INVALID.value
    )


@pytest.mark.asyncio
async def test_fresh_run_executes_the_injected_synchronous_cleanup() -> None:
    logger = _Logger()
    cleanup_calls: list[str] = []

    async def cleanup() -> list[str]:
        cleanup_calls.append("fresh")
        return ["runtime/cache", "data/raw"]

    executor = WorkflowPhaseExecutor(
        check=partial(_ReturningChecker().check, timeout_seconds=1.0),
        runners={},
        cleanup=cleanup,
        max_iterations=1,
        iteration_pause_seconds=0.0,
        logger=logger,
    )

    result = await executor.execute()

    assert result == 0
    assert cleanup_calls == ["fresh"]
    assert (
        "workflow_fresh_run_cleanup_completed",
        {"removed_paths": 2},
    ) in logger.infos
