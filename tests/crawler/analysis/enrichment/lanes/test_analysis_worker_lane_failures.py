"""Regression tests for analysis lane failure isolation and fail-fast."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawler.analysis.enrichment.lanes.analysis_result_writer import (
    AnalysisResultWriter,
)
from crawler.analysis.enrichment.lanes.analysis_router import AnalysisRouter
from crawler.analysis.enrichment.lanes.analysis_worker_lane import (
    AnalysisJob,
    AnalysisJobResult,
    AnalysisWorkerLane,
)
from crawler.classification.media_kind import MediaKind
from crawler.exceptions.crawler_error import (
    AnalysisLaneFailedError,
    CrawlerDrainStalledError,
)
from crawler.runtime.loop.crawl_run_supervisor import CrawlRunSupervisor


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))


class _Payload:
    def __init__(self) -> None:
        self.cleanup_called = 0
        self.temp_path = "payload.tmp"

    def cleanup(self) -> None:
        self.cleanup_called += 1


class _Processor:
    def __init__(
        self,
        *,
        analyze_side_effects: list[Any] | None = None,
        persist_side_effect: Exception | None = None,
    ) -> None:
        self._analyze_side_effects = list(analyze_side_effects or [])
        self.analyze_calls = 0
        self.persist_calls: list[AnalysisJobResult | dict[str, Any]] = []
        self._persist_side_effect = persist_side_effect

    async def analyze_fetched(self, *, result: Any) -> Any:
        del result
        self.analyze_calls += 1
        if self._analyze_side_effects:
            effect = self._analyze_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            if callable(effect):
                return await effect()
            return effect
        return {"ok": True}

    async def persist_analyzed_result(
        self,
        *,
        task: Any,
        result: Any,
        analysis: Any,
    ) -> Any:
        if self._persist_side_effect is not None:
            raise self._persist_side_effect
        outcome = SimpleNamespace(status="persisted")
        self.persist_calls.append(
            {"task_id": task.task_id, "analysis": analysis}
        )
        return outcome

    async def persist_analysis_failure(
        self,
        *,
        task: Any,
        result: Any,
        reason: str,
        error_type: str | None,
        error: str | None,
    ) -> Any:
        if self._persist_side_effect is not None:
            raise self._persist_side_effect
        outcome = SimpleNamespace(status="failure_persisted")
        self.persist_calls.append(
            {
                "task_id": task.task_id,
                "reason": reason,
                "error_type": error_type,
                "error": error,
            }
        )
        return outcome


def _job(
    *,
    task_id: str,
    processor: _Processor,
    payload: _Payload | None = None,
) -> AnalysisJob:
    return AnalysisJob(
        task=SimpleNamespace(
            task_id=task_id, url=f"https://example.test/{task_id}"
        ),
        fetch_result=SimpleNamespace(
            final_url=f"https://example.test/{task_id}",
            payload=payload,
        ),
        kind=MediaKind.IMAGE,
        processor=processor,
    )


def _lane(
    *,
    result_sink: Any | None = None,
    worker_count: int = 1,
    queue_size: int = 8,
    timeout_seconds: float = 5.0,
    logger: _RecordingLogger | None = None,
) -> tuple[AnalysisWorkerLane, list[AnalysisJobResult], _RecordingLogger]:
    results: list[AnalysisJobResult] = []
    log = logger or _RecordingLogger()

    async def sink(result: AnalysisJobResult) -> None:
        if result_sink is not None:
            await result_sink(result)
            return
        results.append(result)

    lane = AnalysisWorkerLane(
        name="image",
        worker_count=worker_count,
        queue_size=queue_size,
        timeout_seconds=timeout_seconds,
        result_sink=sink,
        logger=log,
    )
    return lane, results, log


@pytest.mark.asyncio
async def test_processor_typeerror_does_not_kill_worker() -> None:
    processor = _Processor(
        analyze_side_effects=[TypeError("boom"), {"ok": True}],
    )
    lane, results, _log = _lane(worker_count=1)

    await lane.submit_job(_job(task_id="a", processor=processor))
    await lane.submit_job(_job(task_id="b", processor=processor))
    await lane.drain()

    assert len(results) == 2
    assert results[0].reason == "analysis_failed"
    assert results[0].error_type == "TypeError"
    assert results[1].reason is None
    assert results[1].analysis == {"ok": True}
    assert lane.snapshot()["alive_workers"] == 1
    assert lane.snapshot()["failed"] is False
    assert lane.snapshot()["queued"] == 0

    await lane.stop()


@pytest.mark.asyncio
async def test_lane_owns_full_job_timeout_and_worker_survives() -> None:
    async def block_forever() -> Any:
        await asyncio.Event().wait()
        return {"unreachable": True}

    processor = _Processor(
        analyze_side_effects=[block_forever, {"ok": True}],
    )
    lane, results, _log = _lane(
        worker_count=1,
        timeout_seconds=0.01,
    )

    await lane.submit_job(_job(task_id="timeout", processor=processor))
    await lane.submit_job(_job(task_id="after-timeout", processor=processor))
    await asyncio.wait_for(lane.drain(), timeout=1.0)

    assert len(results) == 2
    assert results[0].reason == "analysis_timeout"
    assert results[1].reason is None
    assert results[1].analysis == {"ok": True}
    assert lane.snapshot()["failed"] is False
    assert lane.snapshot()["alive_workers"] == 1

    await lane.stop()


@pytest.mark.asyncio
async def test_cancelled_error_is_not_converted_to_job_failure() -> None:
    started = asyncio.Event()

    async def block_forever() -> Any:
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    processor = _Processor(analyze_side_effects=[block_forever])
    results: list[AnalysisJobResult] = []

    async def sink(result: AnalysisJobResult) -> None:
        results.append(result)

    lane = AnalysisWorkerLane(
        name="image",
        worker_count=1,
        queue_size=2,
        timeout_seconds=30.0,
        result_sink=sink,
        logger=_RecordingLogger(),
    )

    await lane.submit_job(_job(task_id="blocked", processor=processor))
    await started.wait()
    await lane.stop()

    assert results == []
    assert lane.snapshot()["failed"] is False
    # Normal stop must not register a fatal lane failure.
    assert lane._fatal_error is None


@pytest.mark.asyncio
async def test_sink_error_marks_lane_failed_and_drain_raises() -> None:
    processor = _Processor()

    async def failing_sink(_result: AnalysisJobResult) -> None:
        raise OSError("disk full")

    lane, _results, _log = _lane(result_sink=failing_sink, worker_count=1)

    await lane.submit_job(_job(task_id="a", processor=processor))

    with pytest.raises(AnalysisLaneFailedError) as raised:
        await asyncio.wait_for(lane.drain(), timeout=2.0)

    assert lane.failed is True
    assert isinstance(raised.value.__cause__, OSError)
    assert lane.snapshot()["failed"] is True
    assert lane.snapshot()["failure_type"] == "OSError"

    await lane.stop()


@pytest.mark.asyncio
async def test_submit_fails_fast_after_lane_failure_with_full_queue() -> None:
    processor = _Processor()
    gate = asyncio.Event()

    async def failing_sink(_result: AnalysisJobResult) -> None:
        raise OSError("sink boom")

    lane, _results, _log = _lane(
        result_sink=failing_sink,
        worker_count=1,
        queue_size=1,
    )

    # First job kills the lane via sink failure.
    await lane.submit_job(_job(task_id="fatal", processor=processor))
    with pytest.raises(AnalysisLaneFailedError):
        await asyncio.wait_for(lane.drain(), timeout=2.0)

    # Fill queue capacity accounting is less important than fail-fast submit.
    with pytest.raises(AnalysisLaneFailedError):
        await asyncio.wait_for(
            lane.submit_job(_job(task_id="next", processor=processor)),
            timeout=1.0,
        )

    gate.set()
    await lane.stop()


@pytest.mark.asyncio
async def test_start_does_not_treat_done_task_as_live_worker() -> None:
    processor = _Processor()
    results: list[AnalysisJobResult] = []

    async def sink(result: AnalysisJobResult) -> None:
        results.append(result)

    lane = AnalysisWorkerLane(
        name="image",
        worker_count=1,
        queue_size=2,
        timeout_seconds=5.0,
        result_sink=sink,
        logger=_RecordingLogger(),
    )

    # Simulate a dead worker still present in the list (pre-fix truthy trap).
    dead = asyncio.create_task(asyncio.sleep(0))
    await dead
    lane._workers = [dead]

    # Without a recorded failure, start() should replace the dead task.
    await lane.start()
    assert len(lane._workers) == 1
    assert not lane._workers[0].done()

    await lane.submit_job(_job(task_id="ok", processor=processor))
    await lane.drain()
    assert len(results) == 1

    # With a recorded failure, start must raise.
    lane._fatal_error = RuntimeError("already failed")
    lane._failure_event.set()
    with pytest.raises(AnalysisLaneFailedError):
        await lane.start()

    await lane.stop()


@pytest.mark.asyncio
async def test_persistence_failure_does_not_cleanup_payload() -> None:
    payload = _Payload()
    processor = _Processor(persist_side_effect=OSError("write failed"))
    writer = AnalysisResultWriter(logger=_RecordingLogger())
    job = _job(task_id="p1", processor=processor, payload=payload)
    result = AnalysisJobResult(
        job=job,
        analysis={"ok": True},
        reason=None,
        error_type=None,
        error=None,
    )

    with pytest.raises(OSError, match="write failed"):
        await writer.write_result(result)

    assert payload.cleanup_called == 0


@pytest.mark.asyncio
async def test_successful_persistence_cleans_payload_once() -> None:
    payload = _Payload()
    processor = _Processor()
    writer = AnalysisResultWriter(logger=_RecordingLogger())
    job = _job(task_id="p2", processor=processor, payload=payload)
    result = AnalysisJobResult(
        job=job,
        analysis={"ok": True},
        reason=None,
        error_type=None,
        error=None,
    )

    await writer.write_result(result)
    assert payload.cleanup_called == 1


@pytest.mark.asyncio
async def test_successful_failure_persist_also_cleans_payload_once() -> None:
    payload = _Payload()
    processor = _Processor()
    writer = AnalysisResultWriter(logger=_RecordingLogger())
    job = _job(task_id="p3", processor=processor, payload=payload)
    result = AnalysisJobResult(
        job=job,
        analysis=None,
        reason="analysis_failed",
        error_type="TypeError",
        error="boom",
    )

    await writer.write_result(result)
    assert payload.cleanup_called == 1


@pytest.mark.asyncio
async def test_router_drain_propagates_lane_failure() -> None:
    processor = _Processor()
    logger = _RecordingLogger()

    async def failing_sink(_result: AnalysisJobResult) -> None:
        raise OSError("router sink fail")

    lane = AnalysisWorkerLane(
        name="image",
        worker_count=1,
        queue_size=2,
        timeout_seconds=5.0,
        result_sink=failing_sink,
        logger=logger,
    )
    writer = AnalysisResultWriter(logger=logger)
    router = AnalysisRouter(
        lanes={MediaKind.IMAGE: lane},
        record_writer=writer,
        logger=logger,
    )

    await router.submit(
        task=SimpleNamespace(task_id="r1", url="https://example.test/r1"),
        fetch_result=SimpleNamespace(
            final_url="https://example.test/r1",
            payload=None,
        ),
        kind=MediaKind.IMAGE,
        processor=processor,
    )

    with pytest.raises(AnalysisLaneFailedError):
        await asyncio.wait_for(router.drain(), timeout=2.0)

    await router.stop(force=True)


@pytest.mark.asyncio
async def test_metadata_only_job_uses_lane_timeout_and_persistence_flow() -> (
    None
):
    async def block_forever() -> Any:
        await asyncio.Event().wait()
        return {"unreachable": True}

    payload = _Payload()
    payload.fetch_mode = "metadata_only"
    processor = _Processor(analyze_side_effects=[block_forever])
    logger = _RecordingLogger()
    writer = AnalysisResultWriter(logger=logger)
    lane = AnalysisWorkerLane(
        name="image",
        worker_count=1,
        queue_size=2,
        timeout_seconds=0.01,
        result_sink=writer.write_result,
        logger=logger,
    )
    router = AnalysisRouter(
        lanes={MediaKind.IMAGE: lane},
        record_writer=writer,
        logger=logger,
    )

    outcome = await router.submit(
        task=SimpleNamespace(
            task_id="metadata-only",
            url="https://example.test/metadata-only",
        ),
        fetch_result=SimpleNamespace(
            final_url="https://example.test/metadata-only",
            payload=payload,
        ),
        kind=MediaKind.IMAGE,
        processor=processor,
    )
    await asyncio.wait_for(router.drain(), timeout=1.0)

    assert outcome.detail == "analysis_handoff"
    assert processor.analyze_calls == 1
    assert processor.persist_calls == [
        {
            "task_id": "metadata-only",
            "reason": "analysis_timeout",
            "error_type": "TimeoutError",
            "error": "",
        }
    ]
    assert payload.cleanup_called == 1

    await router.stop(force=True)


@pytest.mark.asyncio
async def test_supervisor_marks_lane_failure_as_run_failure() -> None:
    logger = _RecordingLogger()
    analysis_router = MagicMock()
    analysis_router.snapshot.return_value = {"failed": True}
    analysis_router.drain = AsyncMock(
        side_effect=AnalysisLaneFailedError("lane dead")
    )
    analysis_router.stop = AsyncMock()

    runtime_session = SimpleNamespace(analysis_router=analysis_router)

    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=MagicMock(),
        worker_pool=MagicMock(),
        worker_scaler=MagicMock(),
        logger=logger,  # type: ignore[arg-type]
        seed_enqueuer=MagicMock(),
        min_workers=1,
    )

    with pytest.raises(AnalysisLaneFailedError, match="lane dead"):
        await supervisor._drain_media_analysis(runtime_session=runtime_session)

    analysis_router.stop.assert_awaited_once_with(force=True)
    assert any(
        event == "crawler_media_analysis_failed" for event, _ in logger.events
    )


@pytest.mark.asyncio
async def test_supervisor_marks_stalled_analysis_drain_as_fatal() -> None:
    logger = _RecordingLogger()

    async def drain_forever() -> None:
        await asyncio.Event().wait()

    analysis_router = MagicMock()
    analysis_router.snapshot.return_value = {"queued": 1, "active": 1}
    analysis_router.drain = drain_forever
    analysis_router.stop = AsyncMock()
    runtime_session = SimpleNamespace(analysis_router=analysis_router)
    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=0.01,
        drain_watch_interval_seconds=5.0,
        scheduler=MagicMock(),
        worker_pool=MagicMock(),
        worker_scaler=MagicMock(),
        logger=logger,  # type: ignore[arg-type]
        seed_enqueuer=MagicMock(),
        min_workers=1,
    )

    with pytest.raises(CrawlerDrainStalledError) as raised:
        await supervisor._drain_media_analysis(runtime_session=runtime_session)

    assert isinstance(raised.value.__cause__, TimeoutError)
    analysis_router.stop.assert_awaited_once_with(force=True)
    stalled_logs = [
        fields
        for event, fields in logger.events
        if event == "crawler_media_analysis_drain_stalled"
    ]
    assert stalled_logs == [
        {
            "timeout_seconds": 0.01,
            "media_analysis_snapshot": {"queued": 1, "active": 1},
            "error_type": "TimeoutError",
            "error": "",
        }
    ]
