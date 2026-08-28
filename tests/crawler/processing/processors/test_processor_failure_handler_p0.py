from __future__ import annotations

from typing import Any

import pytest

from config.collection.processors import BaseProcessorSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.results.result import FetchResult
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)


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


class _CapturingDatasetWriter:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.accepted: list[dict[str, Any]] = []

    async def awrite_error(self, **payload: Any) -> None:
        self.errors.append(payload)

    async def awrite(self, **payload: Any) -> None:
        self.accepted.append(payload)


class _FailingDatasetWriter(_CapturingDatasetWriter):
    async def awrite_error(self, **payload: Any) -> None:
        del payload
        raise RuntimeError("storage unavailable")


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


class _RejectedProcessor(PersistingProcessor[BaseProcessorSettings, None]):
    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: None,
    ) -> tuple[bool, str | None, dict[str, object]]:
        del result, analysis
        return False, "quality_too_low", {"quality_score": 0.1}


class _ExplodingProcessor(PersistingProcessor[BaseProcessorSettings, None]):
    async def _persist_accepted_analysis(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: None,
        quality_fields: dict[str, object],
    ) -> ProcessorOutcome:
        raise OSError("disk unavailable")


def _failure_handler(
    *,
    default_retry_wait_seconds: float = 5.0,
) -> ProcessorFailureHandler:
    return ProcessorFailureHandler(
        default_retry_wait_seconds=default_retry_wait_seconds,
    )


def _task() -> CrawlTask:
    return CrawlTask(
        url="https://example.test/input",
        source_name="test",
        task_id="task-1",
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


def test_retry_fallback_and_stage_are_canonical() -> None:
    handler = _failure_handler(default_retry_wait_seconds=7.5)
    exc = RetryableFetchError(
        "temporary transport failure",
        retry_class="transport",
        retry_error_kind="connection_reset",
    )

    task_outcome = handler.retryable(exc=exc, stage="analysis")
    processor_outcome = handler.retryable(
        exc=exc,
        stage="analysis",
    )

    assert task_outcome.retry_after_seconds == 7.5
    assert processor_outcome.retry_after_seconds == 7.5
    assert task_outcome.stage == "analysis"
    assert processor_outcome.stage == "analysis"
    assert task_outcome.reason == "retryable_fetch_error"
    assert task_outcome.counts_toward_task_retry_budget is True
    assert task_outcome.terminal_eligible is True
    assert task_outcome.retry_class == "transport"
    assert task_outcome.retry_error_kind == "connection_reset"


def test_explicit_retry_after_overrides_configured_fallback() -> None:
    handler = _failure_handler(default_retry_wait_seconds=7.5)

    outcome = handler.retryable(
        exc=RetryableFetchError(
            "temporary transport failure",
            retry_class="transport",
            retry_error_kind="connection_reset",
            retry_after_seconds=2.25,
        ),
        stage="analysis",
    )

    assert outcome.retry_after_seconds == 2.25


def test_ignored_fetch_preserves_stage_and_reason() -> None:
    handler = _failure_handler()

    outcome = handler.ignored(
        exc=IgnoredFetchError(reason="not_modified"),
        stage="analysis",
    )

    assert outcome.status == "dropped"
    assert outcome.detail == ""
    assert outcome.reason == "not_modified"
    assert outcome.stage == "analysis"
    assert "final_url" not in outcome.metadata
    assert outcome.error_type == "IgnoredFetchError"


def test_quality_fields_with_reserved_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved processor key"):
        _failure_handler().quality_rejected(
            reject_reason="quality_too_low",
            quality_fields={
                "reason": "forged_reason",
                "stage": "forged_stage",
                "quality_score": 0.1,
            },
        )


def test_quality_rejection_carries_only_extension_metadata() -> None:
    outcome = _failure_handler().quality_rejected(
        reject_reason="quality_too_low",
        quality_fields={"quality_score": 0.1},
    )

    assert outcome.status == "dropped"
    assert outcome.stage == "quality"
    assert outcome.reason == "quality_too_low"
    assert outcome.detail == "quality_too_low"
    assert outcome.metadata == {"quality_score": 0.1}


@pytest.mark.asyncio
async def test_one_exception_creates_one_error_write_and_one_log_event() -> (
    None
):
    writer = _CapturingDatasetWriter()
    logger = _CapturingLogger()
    processor = _RetryableProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=writer,  # type: ignore[arg-type]
        logger=logger,  # type: ignore[arg-type]
        failure_handler=_failure_handler(),
    )

    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "deferred"
    assert outcome.retry_after_seconds == 5.0
    assert len(writer.errors) == 1
    assert writer.errors[0]["stage"] == "analysis"
    failure_events = [
        event
        for _level, event, _fields in logger.events
        if event in {"processor_deferred", "processor_dropped"}
    ]
    assert failure_events == ["processor_deferred"]


@pytest.mark.asyncio
async def test_persistence_failure_does_not_change_retry_decision() -> None:
    writer = _FailingDatasetWriter()
    logger = _CapturingLogger()
    processor = _RetryableProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=writer,  # type: ignore[arg-type]
        logger=logger,  # type: ignore[arg-type]
        failure_handler=_failure_handler(),
    )

    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "deferred"
    assert outcome.reason == "retryable_fetch_error"
    assert outcome.retry_after_seconds == 5.0
    assert [event for _level, event, _fields in logger.events].count(
        "processor_error_persist_failed"
    ) == 1
    assert not any(
        event == "processor_deferred"
        for _level, event, _fields in logger.events
    )


@pytest.mark.asyncio
async def test_quality_rejection_uses_the_same_finalizer() -> None:
    writer = _CapturingDatasetWriter()
    logger = _CapturingLogger()
    processor = _RejectedProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=writer,  # type: ignore[arg-type]
        logger=logger,  # type: ignore[arg-type]
        failure_handler=_failure_handler(),
    )

    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "dropped"
    assert outcome.reason == "quality_too_low"
    assert outcome.stage == "quality"
    assert outcome.metadata["quality_score"] == 0.1
    assert len(writer.errors) == 1
    assert writer.errors[0]["reason"] == "quality_too_low"
    assert writer.errors[0]["stage"] == "quality"
    assert writer.errors[0]["fields"]["kind"] == MediaKind.PAGE
    assert writer.errors[0]["fields"]["final_url"] == (
        "https://example.test/final"
    )
    assert writer.errors[0]["fields"]["quality_score"] == 0.1
    assert not writer.accepted


@pytest.mark.asyncio
async def test_persistence_exception_finalized_exactly_once() -> None:
    writer = _CapturingDatasetWriter()
    logger = _CapturingLogger()
    processor = _ExplodingProcessor(
        settings=BaseProcessorSettings(persist_raw=True),
        dataset_writer=writer,  # type: ignore[arg-type]
        logger=logger,  # type: ignore[arg-type]
        failure_handler=_failure_handler(),
    )

    outcome = await processor.process_fetched(task=_task(), result=_result())

    assert outcome.status == "dropped"
    assert outcome.stage == "persistence"
    assert outcome.error_type == "OSError"
