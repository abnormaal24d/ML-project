"""Guarded phase transitions of the DataChecker-driven workflow loop.

The workflow is decision-driven rather than an explicit enum state machine,
so this file pins the real routing and termination transitions exposed by
``WorkflowPhaseExecutor``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.load import load_settings
from config.settings.fingerprint_sections import build_settings_payloads
from datachecker.fingerprints import (
    SettingsFingerprintCalculator,
    SourceFingerprintCalculator,
)
from datachecker.manifests.artifact_manifest import RunArtifactIdentity
from datachecker.manifests.crawl_attempt_artifacts import (
    CrawlAttemptArtifacts,
)
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from orchestration.bootstrap.crawl_state_reconciler import (
    reconcile_crawl_state,
)
from orchestration.bootstrap.workflow_executor import (
    EXIT_PARTIAL_DOWNSTREAM_INVALID,
    EXIT_SUCCESS,
    WorkflowPhaseExecutor,
)
from orchestration.workflow.phase import (
    PhaseOutcome,
    PhaseStatus,
)

_ACTION_REASON_BY_PHASE: dict[WorkflowAction, WorkflowDecisionReason] = {
    WorkflowAction.CRAWL: WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
    WorkflowAction.PREPROCESS: (
        WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID
    ),
    WorkflowAction.AUGMENT: WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
    WorkflowAction.TRAIN: WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
}

_Runner = Callable[[WorkflowExecutionPlan], Awaitable[PhaseOutcome]]


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.errors: list[tuple[str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))
        self.events.append((event, fields))

    def critical(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))
        self.events.append((event, fields))

    def exception(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))
        self.events.append((event, fields))


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.last_plan: WorkflowExecutionPlan | None = None

    async def __call__(self, plan: WorkflowExecutionPlan) -> PhaseOutcome:
        self.calls += 1
        self.last_plan = plan
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)


def _build_runners() -> dict[WorkflowAction, _Runner]:
    runners = {
        WorkflowAction.CRAWL: _RecordingRunner(),
        WorkflowAction.PREPROCESS: _RecordingRunner(),
        WorkflowAction.AUGMENT: _RecordingRunner(),
        WorkflowAction.TRAIN: _RecordingRunner(),
    }
    return runners


def _plan(
    action: WorkflowAction,
    *,
    plan_details: tuple[str, ...] = (),
) -> WorkflowExecutionPlan:
    reason = _ACTION_REASON_BY_PHASE.get(
        action,
        WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
    )
    return WorkflowExecutionPlan(
        action=action,
        reason=reason,
        details=plan_details,
    )


def _report(
    plan: WorkflowExecutionPlan,
    *,
    details: tuple[str, ...] = (),
) -> WorkflowExecutionPlan:
    # Now plan is already the execution plan, we just need to inject details
    # But dataclass is frozen. We can use object.__setattr__ or just reconstruct.
    return WorkflowExecutionPlan(
        action=plan.action,
        reason=plan.reason,
        details=details if details else plan.details,
    )


class _Checker:
    def __init__(
        self,
        factory,
    ) -> None:
        self._factory = factory
        self.calls = 0

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        del timeout_seconds
        self.calls += 1
        return self._factory(self.calls)


def _build_executor(
    *,
    logger: _Logger,
    checker: _Checker,
    runners: Mapping[WorkflowAction, _Runner],
    max_iterations: int = 50,
) -> WorkflowPhaseExecutor:
    return WorkflowPhaseExecutor(
        check=partial(checker.check, timeout_seconds=1.0),
        runners=runners,
        max_iterations=max_iterations,
        iteration_pause_seconds=0.0,
        logger=logger,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        WorkflowAction.CRAWL,
        WorkflowAction.PREPROCESS,
        WorkflowAction.AUGMENT,
        WorkflowAction.TRAIN,
    ),
)
async def test_route_dispatches_each_phase_to_its_runner(
    action: WorkflowAction,
) -> None:
    runners = _build_runners()
    executor = _build_executor(
        logger=_Logger(),
        checker=_Checker(lambda _call: _report(_plan(WorkflowAction.NOOP))),
        runners=runners,
    )

    outcome = await executor._run_phase(iteration=1, plan=_plan(action))

    assert outcome.status is PhaseStatus.SUCCEEDED
    assert runners[action].calls == 1
    for name, runner in runners.items():
        if name != action:
            assert runner.calls == 0


@pytest.mark.asyncio
async def test_route_refuses_non_runnable_plan_without_invoking_runners() -> (
    None
):
    runners = _build_runners()
    executor = _build_executor(
        logger=_Logger(),
        checker=_Checker(
            lambda _call: _report(
                _plan(WorkflowAction.BLOCKED, plan_details=("broken",))
            ),
        ),
        runners=runners,
    )

    result = await executor.execute()

    assert result == EXIT_PARTIAL_DOWNSTREAM_INVALID
    for runner in runners.values():
        assert runner.calls == 0


@pytest.mark.asyncio
async def test_route_noop_is_terminal_without_invoking_runners() -> None:
    runners = _build_runners()
    executor = _build_executor(
        logger=_Logger(),
        checker=_Checker(lambda _call: _report(_plan(WorkflowAction.NOOP))),
        runners=runners,
    )

    result = await executor.execute()

    assert result == EXIT_SUCCESS
    for runner in runners.values():
        assert runner.calls == 0


@pytest.mark.asyncio
async def test_route_rejects_unsupported_workflow_action() -> None:
    unsupported_plan = WorkflowExecutionPlan(
        action="custom",
        reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
    )
    executor = _build_executor(
        logger=_Logger(),
        checker=_Checker(lambda _call: _report(_plan(WorkflowAction.NOOP))),
        runners=_build_runners(),
    )

    with pytest.raises(TypeError, match="Unsupported workflow action"):
        await executor._run_phase(iteration=1, plan=unsupported_plan)


@pytest.mark.asyncio
async def test_loop_returns_success_on_noop_without_running_a_phase() -> None:
    logger = _Logger()
    checker = _Checker(lambda _call: _report(_plan(WorkflowAction.NOOP)))
    phase_calls = 0

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        nonlocal phase_calls
        phase_calls += 1
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={WorkflowAction.CRAWL: runner},
    )

    result = await executor.execute()

    assert result == EXIT_SUCCESS
    assert checker.calls == 1
    assert phase_calls == 0


@pytest.mark.asyncio
async def test_loop_halts_on_non_runnable_plan_without_running_a_phase() -> (
    None
):
    logger = _Logger()
    checker = _Checker(lambda _call: _report(_plan(WorkflowAction.BLOCKED)))
    phase_calls = 0

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        nonlocal phase_calls
        phase_calls += 1
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={WorkflowAction.CRAWL: runner},
    )

    result = await executor.execute()

    assert result == EXIT_PARTIAL_DOWNSTREAM_INVALID
    assert phase_calls == 0
    assert logger.errors[-1][0] == "data_workflow_halted"


@pytest.mark.asyncio
async def test_loop_halts_when_progress_is_stalled() -> None:
    logger = _Logger()
    checker = _Checker(
        lambda _call: _report(_plan(WorkflowAction.CRAWL)),
    )
    phase_calls = 0

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        nonlocal phase_calls
        phase_calls += 1
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={WorkflowAction.CRAWL: runner},
    )

    result = await executor.execute()

    assert result == EXIT_PARTIAL_DOWNSTREAM_INVALID
    assert checker.calls == 2
    assert phase_calls == 1
    assert logger.errors[-1][0] == "workflow_progress_stalled"
    assert logger.errors[-1][1]["reason"] == (
        WorkflowDecisionReason.WORKFLOW_STATE_INCONSISTENT.value
    )


@pytest.mark.asyncio
async def test_loop_halts_when_a_phase_does_not_continue() -> None:
    logger = _Logger()

    def crawl_report(call: int) -> WorkflowExecutionPlan:
        return _report(
            _plan(
                WorkflowAction.CRAWL,
                plan_details=(f"cycle={call}",),
            ),
        )

    checker = _Checker(crawl_report)

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        return PhaseOutcome(status=PhaseStatus.BLOCKED)

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={WorkflowAction.CRAWL: runner},
    )

    result = await executor.execute()

    assert result == EXIT_PARTIAL_DOWNSTREAM_INVALID
    assert checker.calls == 1
    assert logger.errors[-1][0] == "data_workflow_halted"


@pytest.mark.asyncio
async def test_loop_raises_when_max_iterations_are_exceeded(
    monkeypatch,
) -> None:
    from orchestration.bootstrap import workflow_executor

    monkeypatch.setattr(
        workflow_executor, "_workflow_progress_signature", lambda p: str(id(p))
    )
    logger = _Logger()

    def distinct_report(call: int) -> WorkflowExecutionPlan:
        return _report(
            _plan(
                WorkflowAction.CRAWL,
                plan_details=(f"cycle={call}",),
            ),
        )

    checker = _Checker(distinct_report)

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={WorkflowAction.CRAWL: runner},
        max_iterations=3,
    )

    with pytest.raises(RuntimeError, match="exceeded 3 iterations"):
        await executor.execute()

    assert checker.calls == 3
    failed_events = [
        fields
        for event, fields in logger.errors
        if event == "data_workflow_failed"
    ]
    assert failed_events[-1]["reason"] == "exceeded_max_iterations"


@pytest.mark.asyncio
async def test_loop_continues_with_recrawl_plan_from_phase_outcome() -> None:
    logger = _Logger()
    expected_plan = WorkflowExecutionPlan(
        action=WorkflowAction.CRAWL,
        reason=WorkflowDecisionReason.COVERAGE_TARGETS_NOT_MET,
        details=("recrawl",),
    )

    def report(call: int) -> WorkflowExecutionPlan:
        if call == 1:
            return _report(
                _plan(
                    WorkflowAction.PREPROCESS,
                    plan_details=("cycle=1",),
                ),
            )
        return _report(_plan(WorkflowAction.NOOP))

    checker = _Checker(report)
    executed_plans: list[WorkflowExecutionPlan] = []
    outcomes: list[PhaseOutcome] = []

    async def runner(plan: WorkflowExecutionPlan) -> PhaseOutcome:
        assert isinstance(plan, WorkflowExecutionPlan)
        executed_plans.append(plan)
        if plan.details == ("cycle=1",):
            outcome = PhaseOutcome(
                status=PhaseStatus.RECRAWL_REQUESTED,
                next_plan=expected_plan,
            )
        else:
            outcome = PhaseOutcome(status=PhaseStatus.SUCCEEDED)
        outcomes.append(outcome)
        return outcome

    executor = _build_executor(
        logger=logger,
        checker=checker,
        runners={
            WorkflowAction.PREPROCESS: runner,
            WorkflowAction.CRAWL: runner,
        },
    )

    result = await executor.execute()

    assert result == EXIT_SUCCESS
    first_outcome = outcomes[0]
    assert first_outcome.status is PhaseStatus.RECRAWL_REQUESTED
    assert first_outcome.next_plan is expected_plan
    assert executed_plans[1] is expected_plan
    assert checker.calls == 2


@pytest.mark.asyncio
async def test_resume_closes_interrupted_running_attempt_before_new_work(
    tmp_path: Path,
) -> None:
    settings = load_settings(
        "test",
        project_root=tmp_path,
        config_root=None,
        environment="test",
        fingerprint=False,
    )
    fingerprint_calculator = SettingsFingerprintCalculator()
    source_fingerprint = SourceFingerprintCalculator(
        settings_fingerprint_calculator=fingerprint_calculator,
    ).calculate(
        seed_urls=tuple(settings.sources.active.seed_urls),
        source_profile=settings.sources.active,
    )
    crawl_fingerprint = fingerprint_calculator.calculate(
        payload=build_settings_payloads(
            settings=settings,
            checker_settings=settings.collection.datachecker,
        ).crawl,
    )

    raw_run_directory = tmp_path / "data" / "raw" / "interrupted"
    raw_run_directory.mkdir(parents=True)
    summary_path = raw_run_directory / "run_manifest.json"
    running_summary = json.dumps(
        {"status": "running", "final": False},
        sort_keys=True,
    )
    summary_path.write_text(running_summary, encoding="utf-8")

    identity = RunArtifactIdentity(
        generation_id="gen_interrupted",
        workflow_id="wf_interrupted",
        project_fingerprint="project",
        config_fingerprint="config",
        environment_name="test",
        environment_fingerprint="environment",
        python_version="3.12",
        dependency_lock_fingerprint="lock",
    )
    interrupted = CrawlStateManifest(
        **identity.manifest_fields(),
        status=WorkflowLifecycleStatus.RUNNING,
        attempt_id="crawl_attempt_interrupted",
        started_at="2026-08-20T10:00:00+00:00",
        updated_at="2026-08-20T10:01:00+00:00",
        completed_at=None,
        raw_run_directory=raw_run_directory,
        run_summary_path=summary_path,
        previous_status=None,
        previous_raw_run_directory=None,
        last_successful_completed_at=None,
        last_successful_manifest_path=None,
        error_type=None,
        error_message=None,
        source_registry_hash=source_fingerprint,
        crawl_settings_hash=crawl_fingerprint,
        raw_run_id="raw_interrupted",
        crawl_session_id="crawl_interrupted",
    )
    events: list[str] = []
    state_writer = _RecoveryStateWriter(
        state=interrupted,
        events=events,
    )
    crawl_writer = _NonFinalCrawlWriter(
        manifest_path=tmp_path / "crawl_manifest.json",
        events=events,
    )
    manifest_writers = SimpleNamespace(
        crawl_state_manifest_writer=state_writer,
        crawl=crawl_writer,
        crawl_promotion=_UnexpectedPromotion(),
    )

    async def run_blocking(
        function: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("timeout_seconds", None)
        return function(**kwargs)

    promoted = await reconcile_crawl_state(
        settings=settings,
        logger=_Logger(),
        manifest_writers=manifest_writers,
        run_blocking=run_blocking,
        io_timeout_seconds=1.0,
    )

    assert promoted is False
    assert events == [
        "state_read",
        "state_recovering",
        "attempt_resolved",
        "finalization_checked",
        "state_abandoned",
    ]
    assert state_writer.state.status is WorkflowLifecycleStatus.FAILED
    assert state_writer.state.previous_status is (
        WorkflowLifecycleStatus.RECOVERING
    )
    assert state_writer.state.error_type == "abandoned_interrupted_run"
    assert state_writer.state.completed_at is not None
    assert summary_path.read_text(encoding="utf-8") == running_summary


class _RecoveryStateWriter:
    def __init__(
        self,
        *,
        state: CrawlStateManifest,
        events: list[str],
    ) -> None:
        self.state = state
        self._events = events

    def read_current_state(self) -> CrawlStateManifest:
        self._events.append("state_read")
        return self.state

    def write_crawl_state_recovering(
        self,
        *,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
        error_type: str | None,
        error_message: str | None,
    ) -> CrawlStateManifest:
        self._events.append("state_recovering")
        self.state = replace(
            self.state,
            status=WorkflowLifecycleStatus.RECOVERING,
            previous_status=self.state.status,
            completed_at="2026-08-20T10:02:00+00:00",
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            error_type=error_type,
            error_message=error_message,
        )
        return self.state

    def resolve_latest_crawl_attempt(self) -> CrawlAttemptArtifacts:
        self._events.append("attempt_resolved")
        return CrawlAttemptArtifacts(
            raw_run_directory=self.state.raw_run_directory,
            run_summary_path=self.state.run_summary_path,
        )

    def write_crawl_state_abandoned(
        self,
        *,
        reason: str,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
    ) -> CrawlStateManifest:
        self._events.append("state_abandoned")
        self.state = replace(
            self.state,
            status=WorkflowLifecycleStatus.FAILED,
            previous_status=self.state.status,
            completed_at="2026-08-20T10:03:00+00:00",
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            error_type="abandoned_interrupted_run",
            error_message=reason,
        )
        return self.state


class _NonFinalCrawlWriter:
    def __init__(self, *, manifest_path: Path, events: list[str]) -> None:
        self._manifest_path = manifest_path
        self._events = events

    def crawl_manifest_path(self) -> Path:
        return self._manifest_path

    def has_finalized_raw_output(self, **_kwargs: object) -> bool:
        self._events.append("finalization_checked")
        return False


class _UnexpectedPromotion:
    def resume_existing(self, **_kwargs: object) -> None:
        raise AssertionError("no canonical manifest should be resumed")

    def commit(self, **_kwargs: object) -> None:
        raise AssertionError("a non-final raw attempt must not be promoted")
