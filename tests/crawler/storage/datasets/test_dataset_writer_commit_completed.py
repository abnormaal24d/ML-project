"""Commit-completed contracts for the raw dataset writer."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from crawler.runtime.loop.crawl_run_summary import (
    CrawlRunResult,
    CrawlStopTrigger,
    CrawlTerminalOutcome,
)
from crawler.storage.datasets.writing.dataset_writer import (
    DatasetWriter,
    _build_readiness_report,
)


def _result(*, output_ready: bool) -> CrawlRunResult:
    return CrawlRunResult(
        stop_trigger=CrawlStopTrigger.FRONTIER_DRAINED,
        terminal_outcome=CrawlTerminalOutcome.SUCCESS,
        completed_tasks=5,
        worker_failures=0,
        task_failures_total=0,
        non_fatal_timeouts_total=0,
        retry_exhausted_total=0,
        average_processing_seconds=0.5,
        output_ready=output_ready,
        unmet_requirements=() if output_ready else ("audio<5", "video<5"),
        object_records_total=17,
        requests_total=22,
        successful_requests_total=20,
        quality_score=0.8,
        modality_counts={"page": 2, "document": 10, "audio": 3, "video": 2},
    )


class _LifecycleRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def finalize(self, **kwargs: Any) -> str:
        # Map the actual parameter names used by DatasetRunFinalizer.finalize
        mapped = dict(kwargs)
        if "terminal_reason" in mapped:
            mapped["reason"] = mapped.pop("terminal_reason")
        if "terminal_details" in mapped:
            mapped["details"] = mapped.pop("terminal_details")
        self.calls.append(mapped)
        return "2026-08-10T00:00:00+00:00"


def _commit_writer() -> tuple[DatasetWriter, _LifecycleRecorder]:
    writer = DatasetWriter.__new__(DatasetWriter)
    writer._state_lock = threading.Lock()
    writer._async_write_lock = asyncio.Lock()
    writer._closed = False
    writer._total_bytes_written = 1234
    lifecycle = _LifecycleRecorder()
    writer._run_finalizer = lifecycle
    return writer, lifecycle


@pytest.mark.asyncio
async def test_commit_completed_finalizes_undercovered_run() -> None:
    writer, lifecycle = _commit_writer()

    await writer.commit_completed(crawler_result=_result(output_ready=False))

    assert len(lifecycle.calls) == 1
    finalize = lifecycle.calls[0]
    assert finalize["status"] == "completed"
    assert finalize["final"] is True
    assert finalize["reason"] is None
    assert finalize["total_bytes_written"] == 1234
    readiness = finalize["readiness_report"]
    assert isinstance(readiness, dict)
    assert readiness["ready"] is False
    assert set(readiness["unmet_requirements"]) == {"audio<5", "video<5"}
    assert writer._closed is True
    assert writer._completed_at == "2026-08-10T00:00:00+00:00"


@pytest.mark.asyncio
async def test_commit_completed_finalizes_ready_run() -> None:
    writer, lifecycle = _commit_writer()

    await writer.commit_completed(crawler_result=_result(output_ready=True))

    finalize = lifecycle.calls[0]
    assert finalize["status"] == "completed"
    assert finalize["final"] is True
    readiness = finalize["readiness_report"]
    assert readiness["ready"] is True
    assert readiness["unmet_requirements"] == []


@pytest.mark.asyncio
async def test_commit_completed_without_result_writes_empty_readiness() -> (
    None
):
    writer, lifecycle = _commit_writer()

    await writer.commit_completed()

    finalize = lifecycle.calls[0]
    assert finalize["status"] == "completed"
    assert finalize["final"] is True
    assert finalize["readiness_report"] == {}


@pytest.mark.asyncio
async def test_commit_completed_is_idempotent_after_close() -> None:
    writer, lifecycle = _commit_writer()
    await writer.commit_completed(crawler_result=_result(output_ready=False))

    await writer.commit_completed(crawler_result=_result(output_ready=False))

    assert len(lifecycle.calls) == 1


def test_build_readiness_report_maps_output_ready() -> None:
    report = _build_readiness_report(_result(output_ready=False))
    assert report["ready"] is False
    assert list(report["unmet_requirements"]) == ["audio<5", "video<5"]
    assert report["stop_trigger"] == CrawlStopTrigger.FRONTIER_DRAINED.value
    assert report["terminal_outcome"] == CrawlTerminalOutcome.SUCCESS.value


def test_build_readiness_report_ready_success_run() -> None:
    report = _build_readiness_report(_result(output_ready=True))
    assert report["ready"] is True
    assert report["unmet_requirements"] == []
