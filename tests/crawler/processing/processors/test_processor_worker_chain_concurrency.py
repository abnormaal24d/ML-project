from __future__ import annotations

import asyncio
from typing import Any

import pytest

from config.collection.discovery import WorkerPoolSettings
from config.collection.processors import BaseProcessorSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.errors.exceptions import RetryableFetchError
from crawler.fetching.results.result import FetchResult
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from crawler.worker.task_iteration.worker_task_finalizer import (
    WorkerTaskFinalizer,
)
from crawler.worker.task_iteration.worker_task_result_persister import (
    WorkerTaskResultPersister,
)
from crawler.worker.worker_loop.worker_state import WorkerState


class _CapturingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append(("debug", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append(("warning", event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append(("error", event, fields))

    def is_debug_enabled(self) -> bool:
        return False


class _DummyDatasetWriter:
    async def awrite(self, **payload: object) -> None:
        del payload

    async def awrite_error(self, **payload: object) -> None:
        del payload


class _RejectedProcessor(PersistingProcessor[BaseProcessorSettings, None]):
    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: None,
    ) -> tuple[bool, str | None, dict[str, object]]:
        del result, analysis
        return False, "quality_too_low", {"quality_score": 0.1}


class _RetryableProcessor(PersistingProcessor[BaseProcessorSettings, None]):
    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> None:
        del result
        raise RetryableFetchError(
            "temporary transport failure",
            retry_class="transport",
            retry_error_kind="connection_reset",
        )


class _ExplodingProcessor(PersistingProcessor[BaseProcessorSettings, None]):
    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> None:
        del result
        raise RuntimeError("analysis exploded")


class _SchedulerFake:
    def __init__(self) -> None:
        self.completions: list[
            tuple[CrawlTask, str, dict[str, object] | None]
        ] = []

    async def complete(
        self,
        task: CrawlTask,
        *,
        outcome: str = "completed",
        fields: dict[str, object] | None = None,
    ) -> None:
        self.completions.append((task, outcome, fields))


class _CallbackFake:
    def __init__(self) -> None:
        self.calls: list[
            tuple[CrawlTask, str, dict[str, object] | None, object]
        ] = []

    def __call__(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
        result: object,
    ) -> None:
        self.calls.append((task, outcome, fields, result))


class _SessionTrackerFake:
    def __init__(self) -> None:
        self.finished: list[tuple[str | None, CrawlTask]] = []
        self._duration = 1.0

    def mark_task_finished(
        self,
        *,
        state: WorkerState,
        outcome: str | None,
        task: CrawlTask | None = None,
    ) -> float:
        del state
        self.finished.append((outcome, task))
        return self._duration


class _FailureRegistrar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **fields: object) -> bool:
        self.calls.append(fields)
        return False


def _task(*, task_id: str = "task-1") -> CrawlTask:
    return CrawlTask(
        url="https://example.test/input",
        source_name="test",
        task_id=task_id,
        kind=MediaKind.PAGE,
    )


def _result() -> FetchResult:
    return FetchResult(
        url="https://example.test/input",
        final_url="https://example.test/final",
        status_code=200,
        headers={},
        fetched_at="2026-07-26T00:00:00Z",
        content_type="text/html",
        mime_type="text/html",
        encoding="utf-8",
        language="en",
        kind=MediaKind.PAGE,
    )


def _finalizer(
    *,
    scheduler: _SchedulerFake | None = None,
    callback: _CallbackFake | None = None,
    failing_scheduler: bool = False,
    failing_callback: bool = False,
) -> tuple[WorkerTaskFinalizer, _FailureRegistrar, object]:
    scheduler = scheduler or _SchedulerFake()
    registrar = _FailureRegistrar()
    settings = WorkerPoolSettings()

    if failing_scheduler:

        async def complete(task: object, **kwargs: object) -> None:
            del task, kwargs
            raise RuntimeError("completion exploded")

        scheduler.complete = complete  # type: ignore[method-assign]

    if failing_callback:
        callback = _FailingCallback()
    else:
        callback = callback or _CallbackFake()

    persister = WorkerTaskResultPersister(
        settings=settings,
        scheduler=scheduler,
        task_result_callback=callback,
    )
    finalizer = WorkerTaskFinalizer(
        settings=settings,
        logger=_CapturingLogger(),
        persister=persister,
        session_tracker=_SessionTrackerFake(),
        register_failure=registrar,
    )
    return finalizer, registrar, callback


class _FailingCallback:
    def __call__(self, **kwargs: object) -> Any:
        del kwargs
        raise RuntimeError("callback boom")


async def _run_finalize(
    finalizer: WorkerTaskFinalizer,
    *,
    task: CrawlTask,
    outcome: ProcessorOutcome | None = None,
    runtime_outcome: str | None = None,
    error: BaseException | None = None,
    timeout_origin: str | None = None,
    timeout_elapsed_seconds: float | None = None,
    wait_seconds: float | None = None,
) -> None:
    await finalizer.finalize(
        task=task,
        outcome=outcome,
        runtime_outcome=runtime_outcome,
        error=error,
        timeout_origin=timeout_origin,
        timeout_elapsed_seconds=timeout_elapsed_seconds,
        wait_seconds=wait_seconds,
        worker_id=7,
        state=WorkerState(worker_id=7),
    )


# --- processor failure outcomes flow into completion payloads --------------


@pytest.mark.asyncio
async def test_dropped_quality_outcome_flows_into_completion_payload() -> None:
    processor = _RejectedProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=_DummyDatasetWriter(),  # type: ignore[arg-type]
        logger=_CapturingLogger(),  # type: ignore[arg-type]
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
    )
    outcome = await processor.process_fetched(task=_task(), result=_result())
    assert outcome.status == "dropped"

    completion_outcome, fields = WorkerTaskFinalizer._build_completion(
        None,
        task=_task(),
        outcome=outcome,
        runtime_outcome=None,
        error=None,
        timeout_origin=None,
        timeout_elapsed_seconds=None,
        wait_seconds=None,
    )
    assert completion_outcome == "dropped"
    assert fields["status"] == "dropped"
    assert fields["stage"] == "quality"
    assert fields["reason"] == "quality_too_low"
    assert fields["detail"] == "quality_too_low"
    assert fields["quality_score"] == 0.1


@pytest.mark.asyncio
async def test_deferred_retry_outcome_flows_into_completion_payload() -> None:
    processor = _RetryableProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=_DummyDatasetWriter(),  # type: ignore[arg-type]
        logger=_CapturingLogger(),  # type: ignore[arg-type]
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
    )
    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "deferred"
    completion_outcome, fields = WorkerTaskFinalizer._build_completion(
        None,
        task=_task(),
        outcome=outcome,
        runtime_outcome=None,
        error=None,
        timeout_origin=None,
        timeout_elapsed_seconds=None,
        wait_seconds=None,
    )
    assert completion_outcome == "deferred"
    assert fields["retry_class"] == "transport"
    assert fields["retry_error_kind"] == "connection_reset"
    assert fields["counts_toward_task_retry_budget"] is True
    assert fields["terminal_eligible"] is True
    assert fields["retry_after_seconds"] == 5.0


@pytest.mark.asyncio
async def test_success_outcome_flows_into_completion_payload() -> None:
    outcome = ProcessorOutcome.success(
        stage="persistence",
        detail="page",
        metadata={"quality_score": 0.5},
    )
    completion_outcome, fields = WorkerTaskFinalizer._build_completion(
        None,
        task=_task(),
        outcome=outcome,
        runtime_outcome=None,
        error=None,
        timeout_origin=None,
        timeout_elapsed_seconds=None,
        wait_seconds=None,
    )
    assert completion_outcome == "success"
    assert fields["status"] == "success"
    assert fields["stage"] == "persistence"
    assert fields["detail"] == "page"
    assert fields["quality_score"] == 0.5


@pytest.mark.asyncio
async def test_processor_exception_outcome_flows_into_completion_payload() -> (
    None
):
    processor = _ExplodingProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=_DummyDatasetWriter(),  # type: ignore[arg-type]
        logger=_CapturingLogger(),  # type: ignore[arg-type]
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
    )
    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "dropped"
    assert outcome.reason == "processor_exception"
    completion_outcome, fields = WorkerTaskFinalizer._build_completion(
        None,
        task=_task(),
        outcome=outcome,
        runtime_outcome=None,
        error=None,
        timeout_origin=None,
        timeout_elapsed_seconds=None,
        wait_seconds=None,
    )
    assert completion_outcome == "dropped"
    assert fields["error_type"] == "RuntimeError"
    assert fields["error"] == "analysis exploded"


@pytest.mark.asyncio
async def test_after_persist_failure_contract_surfaces_deferred_outcome() -> (
    None
):
    outcome = ProcessorOutcome.deferred(
        stage="persistence",
        reason="after_persist_failed",
        retry_after_seconds=1.0,
        error_type="OSError",
        counts_toward_task_retry_budget=True,
        terminal_eligible=True,
    )
    completion_outcome, fields = WorkerTaskFinalizer._build_completion(
        None,
        task=_task(),
        outcome=outcome,
        runtime_outcome=None,
        error=None,
        timeout_origin=None,
        timeout_elapsed_seconds=None,
        wait_seconds=None,
    )
    assert completion_outcome == "deferred"
    assert fields["reason"] == "after_persist_failed"
    assert fields["retry_after_seconds"] == 1.0
    assert fields["error_type"] == "OSError"
    assert fields["counts_toward_task_retry_budget"] is True
    assert fields["terminal_eligible"] is True


@pytest.mark.asyncio
async def test_reserved_metadata_keys_rejected_at_outcome_construction() -> (
    None
):
    with pytest.raises(ValueError, match="reserved processor key"):
        ProcessorOutcome.success(
            stage="persistence",
            metadata={"reason": "forged", "value": 2},
        )


# --- worker finalizer completion payload interop ---------------------------


@pytest.mark.asyncio
async def test_finalize_success_completes_scheduler_and_callback() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, callback = _finalizer(scheduler=scheduler)
    task = _task()
    outcome = ProcessorOutcome.success(
        stage="persistence",
        detail="page",
        metadata={"quality_score": 0.5},
    )

    await _run_finalize(finalizer, task=task, outcome=outcome)

    assert len(scheduler.completions) == 1
    completed_task, completed_outcome, fields = scheduler.completions[0]
    assert completed_task is task
    assert completed_outcome == "success"
    assert fields["quality_score"] == 0.5
    assert len(callback.calls) == 1
    assert callback.calls[0][1] == "success"
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_finalize_deferred_runtime_injects_retry_semantics() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, _callback = _finalizer(scheduler=scheduler)
    task = _task()
    outcome = ProcessorOutcome.deferred(
        stage="fetch",
        reason="retryable_fetch_error",
        retry_after_seconds=2.5,
        retry_class="fetch_retryable",
        retry_error_kind="connect_error",
        counts_toward_task_retry_budget=True,
        terminal_eligible=True,
        error_type="RetryableFetchError",
        error="nested transport failure",
        metadata={
            "observed_bytes": None,
            "partial_path": None,
            "unconditional_retry_performed": False,
            "retry_budget_seconds_remaining": None,
        },
    )

    await _run_finalize(
        finalizer,
        task=task,
        outcome=outcome,
    )

    assert scheduler.completions[0][1] == "deferred"
    fields = scheduler.completions[0][2]
    assert fields["reason"] == "retryable_fetch_error"
    assert fields["retry_class"] == "fetch_retryable"
    assert fields["retry_error_kind"] == "connect_error"
    assert fields["counts_toward_task_retry_budget"] is True
    assert fields["terminal_eligible"] is True
    assert fields["retry_after_seconds"] == 2.5
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_finalize_cancellation_skips_callback_but_completes() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, callback = _finalizer(scheduler=scheduler)

    await _run_finalize(
        finalizer,
        task=_task(),
        runtime_outcome="cancelled",
    )

    assert scheduler.completions[0][1] == "cancelled"
    assert callback.calls == []
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_finalize_without_outcome_falls_back_to_failed() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, _callback = _finalizer(scheduler=scheduler)

    await _run_finalize(finalizer, task=_task())

    assert scheduler.completions[0][1] == "failed"
    fields = scheduler.completions[0][2]
    assert fields["stage"] == "worker_runtime"
    assert fields["reason"] == "failed"
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_finalize_timeout_carries_worker_timeout_fields() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, _callback = _finalizer(scheduler=scheduler)

    await _run_finalize(
        finalizer,
        task=_task(),
        runtime_outcome="timeout",
        error=TimeoutError("too slow"),
        timeout_origin="processing",
        timeout_elapsed_seconds=2.0,
    )

    assert scheduler.completions[0][1] == "timeout"
    fields = scheduler.completions[0][2]
    assert fields["timeout"] is True
    assert fields["timeout_origin"] == "processing"
    assert fields["elapsed_seconds"] == 2.0
    assert fields["configured_worker_timeout_seconds"] == 300.0
    assert fields["error_type"] == "TimeoutError"
    assert registrar.calls == []


@pytest.mark.asyncio
async def test_finalize_registers_callback_and_completion_failures_nonfatal() -> (
    None
):
    finalizer, registrar, _callback = _finalizer(
        failing_scheduler=True,
        failing_callback=True,
    )

    await _run_finalize(finalizer, task=_task())

    assert len(registrar.calls) == 2
    assert all(call.get("fatal") is False for call in registrar.calls)


@pytest.mark.asyncio
async def test_finalize_concurrent_tasks_keep_isolated_completions() -> None:
    scheduler = _SchedulerFake()
    finalizer, registrar, _callback = _finalizer(scheduler=scheduler)
    first = _task(task_id="first")
    second = _task(task_id="second")
    first_outcome = ProcessorOutcome.success(stage="persistence")
    second_outcome = ProcessorOutcome.failure(
        stage="analysis",
        reason="analysis_failed",
    )

    await asyncio.gather(
        _run_finalize(
            finalizer,
            task=first,
            outcome=first_outcome,
        ),
        _run_finalize(
            finalizer,
            task=second,
            outcome=second_outcome,
        ),
    )

    completed = sorted(
        (
            (task.task_id, outcome)
            for task, outcome, _ in scheduler.completions
        ),
    )
    assert completed == [("first", "success"), ("second", "failure")]
    assert registrar.calls == []
