"""Bounded asynchronous analysis lane for one content modality."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from crawler.analysis.enrichment.audio.audio_analyzer import (
    AudioAnalysisResult,
)
from crawler.analysis.enrichment.documents.document_analyzer import (
    DocumentAnalysisResult,
)
from crawler.analysis.enrichment.image.image_analyzer import ImageAnalysis
from crawler.analysis.enrichment.video.video_analysis_result import (
    VideoAnalysisResult,
)
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.exceptions.crawler_error import AnalysisLaneFailedError
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.results.result import FetchResult
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger

T = TypeVar("T")
type AnalysisResult = (
    AudioAnalysisResult
    | ImageAnalysis
    | DocumentAnalysisResult
    | VideoAnalysisResult
)


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    task: CrawlTask
    fetch_result: FetchResult
    kind: MediaKind
    processor: PersistingProcessor[Any, AnalysisResult]


@dataclass(frozen=True, slots=True)
class AnalysisJobResult:
    job: AnalysisJob
    analysis: AnalysisResult | None
    reason: str | None
    error_type: str | None
    error: str | None


def failed_analysis_result(
    *,
    job: AnalysisJob,
    exc: BaseException,
    timeout: bool = False,
) -> AnalysisJobResult:
    """Map an analyzer exception to a job-level failure result."""

    if timeout:
        reason = "analysis_timeout"
    elif isinstance(exc, IgnoredFetchError):
        reason = "analysis_rejected"
    else:
        reason = "analysis_failed"

    return AnalysisJobResult(
        job=job,
        analysis=None,
        reason=reason,
        error_type=type(exc).__name__,
        error=str(exc),
    )


async def execute_analysis_job(*, job: AnalysisJob) -> AnalysisJobResult:
    """Run analyzer code and convert ordinary failures into job results."""

    try:
        analysis = await job.processor.analyze_fetched(
            result=job.fetch_result,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return failed_analysis_result(job=job, exc=exc)

    return AnalysisJobResult(
        job=job,
        analysis=analysis,
        reason=None,
        error_type=None,
        error=None,
    )


class AnalysisWorkerLane:
    """Run one bounded analysis lane for one modality.

    Failure model:
    - Processor/analyzer exceptions become job-level failure results; the
      worker stays alive.
    - Result-sink / persistence exceptions are lane-level fatal failures.
    - Unexpected worker exits are lane-level fatal failures.
    - ``submit_job()`` and ``drain()`` fail fast when the lane is failed.
    """

    def __init__(
        self,
        *,
        name: str,
        worker_count: int,
        queue_size: int,
        timeout_seconds: float,
        result_sink: Callable[
            [AnalysisJobResult],
            Awaitable[ProcessorOutcome],
        ],
        logger: ProjectLogger,
    ) -> None:
        self._name = str(name).strip().lower()
        self._worker_count = max(1, int(worker_count))
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._queue: asyncio.Queue[AnalysisJob] = asyncio.Queue(
            maxsize=max(1, int(queue_size)),
        )
        self._result_sink = result_sink
        self._logger = logger
        self._workers: list[asyncio.Task[None]] = []
        self._active_items: dict[int, AnalysisJob] = {}
        self._failure_event = asyncio.Event()
        self._fatal_error: BaseException | None = None
        self._stopping = False
        self._lifecycle_lock = asyncio.Lock()

    @property
    def failed(self) -> bool:
        return self._fatal_error is not None

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            self._raise_if_failed()

            if self._stopping:
                raise RuntimeError(f"analysis lane {self._name!r} is stopping")

            alive = [worker for worker in self._workers if not worker.done()]
            self._workers = alive

            for index in range(len(alive), self._worker_count):
                task = asyncio.create_task(
                    self._run_worker_loop(index=index),
                    name=f"{self._name}-analysis-lane-{index}",
                )

                def on_worker_done(
                    done_task: asyncio.Task[None],
                    worker_index: int = index,
                ) -> None:
                    self._on_worker_done(
                        task=done_task,
                        worker_index=worker_index,
                    )

                task.add_done_callback(on_worker_done)
                self._workers.append(task)

    async def submit_job(self, job: AnalysisJob) -> None:
        await self.start()
        await self._wait_or_failure(self._queue.put(job))

    async def drain(self) -> None:
        await self._wait_or_failure(self._queue.join())

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            self._stopping = True

        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def snapshot(self) -> dict[str, object]:
        """Return lightweight diagnostics for stalled drain logging."""

        active_items = []
        for worker_index, item in sorted(self._active_items.items()):
            active_items.append(
                {
                    "worker": worker_index,
                    "task_id": item.task.task_id,
                    "url": item.task.url,
                    "result_kind": item.kind,
                    "final_url": item.fetch_result.final_url,
                }
            )

        fatal = self._fatal_error
        return {
            "lane": self._name,
            "queued": self._queue.qsize(),
            "active": len(active_items),
            "active_items": active_items,
            "workers": len(self._workers),
            "alive_workers": sum(
                1 for worker in self._workers if not worker.done()
            ),
            "dead_workers": sum(
                1 for worker in self._workers if worker.done()
            ),
            "failed": fatal is not None,
            "failure_type": (
                type(fatal).__name__ if fatal is not None else None
            ),
            "failure": str(fatal) if fatal is not None else None,
            "stopping": self._stopping,
        }

    async def _run_worker_loop(self, *, index: int) -> None:
        while True:
            if self._stopping or self.failed:
                return

            try:
                job = await self._wait_or_failure(self._queue.get())
            except AnalysisLaneFailedError:
                return
            except asyncio.CancelledError:
                raise

            self._active_items[index] = job
            try:
                try:
                    self._logger.debug(
                        "media_analysis_started",
                        lane=self._name,
                        lane_worker=index,
                        **self._safe_job_context(job),
                    )
                    if self._queue.qsize() > self._queue.maxsize * 0.8:
                        self._logger.warning(
                            "media_analysis_lane_high_watermark",
                            lane=self._name,
                            queue_size=self._queue.qsize(),
                        )
                except Exception:
                    pass

                # Analyzer exceptions stay job-level; only the sink call is a
                # lane-level fatal boundary.
                try:
                    result = await asyncio.wait_for(
                        execute_analysis_job(job=job),
                        timeout=self._timeout_seconds,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    result = failed_analysis_result(
                        job=job,
                        exc=exc,
                        timeout=True,
                    )
                if result.reason is not None:
                    try:
                        self._logger.error(
                            "media_analysis_job_failed",
                            lane=self._name,
                            lane_worker=index,
                            reason=result.reason,
                            error_type=result.error_type,
                            error=result.error,
                            **self._safe_job_context(job),
                        )
                    except Exception:
                        pass

                try:
                    await self._result_sink(result)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._record_fatal_failure(
                        worker_index=index,
                        job=job,
                        cause=exc,
                        event="media_analysis_persistence_failed",
                    )
                    raise
            except asyncio.CancelledError:
                raise
            finally:
                self._active_items.pop(index, None)
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._fatal_error is None:
            return
        raise AnalysisLaneFailedError(
            f"analysis lane {self._name!r} failed"
        ) from self._fatal_error

    def _record_fatal_failure(
        self,
        *,
        cause: BaseException,
        worker_index: int,
        job: AnalysisJob | None,
        event: str = "media_analysis_lane_failed",
    ) -> None:
        if self._stopping:
            return

        first_failure = self._fatal_error is None
        if first_failure:
            # Set failure state before logging so a logging fault still
            # leaves the lane marked defective.
            self._fatal_error = cause
            self._failure_event.set()

        try:
            self._logger.error(
                event,
                lane=self._name,
                lane_worker=worker_index,
                error_type=type(cause).__name__,
                error=str(cause),
                first_failure=first_failure,
                **self._safe_job_context(job),
            )
        except Exception:
            # Never let logging mask the original fatal cause.
            return

        if event != "media_analysis_lane_failed":
            try:
                self._logger.error(
                    "media_analysis_lane_failed",
                    lane=self._name,
                    lane_worker=worker_index,
                    error_type=type(cause).__name__,
                    error=str(cause),
                    first_failure=first_failure,
                    **self._safe_job_context(job),
                )
            except Exception:
                return

    def _on_worker_done(
        self,
        *,
        task: asyncio.Task[None],
        worker_index: int,
    ) -> None:
        # Always retrieve the task outcome so asyncio does not warn about
        # "Task exception was never retrieved".
        retrieved_exc: BaseException | None = None
        cancelled = task.cancelled()
        if not cancelled:
            try:
                retrieved_exc = task.exception()
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                retrieved_exc = None

        if self._stopping or self.failed:
            return

        if cancelled:
            self._record_fatal_failure(
                worker_index=worker_index,
                job=self._active_items.get(worker_index),
                cause=RuntimeError(
                    f"analysis worker {worker_index} cancelled unexpectedly"
                ),
                event="media_analysis_worker_exited",
            )
            return

        if retrieved_exc is not None:
            self._record_fatal_failure(
                worker_index=worker_index,
                job=self._active_items.get(worker_index),
                cause=retrieved_exc,
                event="media_analysis_worker_exited",
            )
            return

        # Infinite worker loop should not exit cleanly while the lane is open.
        self._record_fatal_failure(
            worker_index=worker_index,
            job=self._active_items.get(worker_index),
            cause=RuntimeError(
                f"analysis worker {worker_index} exited unexpectedly"
            ),
            event="media_analysis_worker_exited",
        )

    async def _wait_or_failure[T](self, operation: Awaitable[T]) -> T:
        self._raise_if_failed()

        operation_task = asyncio.ensure_future(operation)
        failure_task = asyncio.create_task(self._failure_event.wait())

        try:
            done, _pending = await asyncio.wait(
                {operation_task, failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if failure_task in done:
                self._raise_if_failed()

            result = await operation_task
            # Catch the race where both complete nearly together.
            self._raise_if_failed()
            return result
        finally:
            for task in (operation_task, failure_task):
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                operation_task,
                failure_task,
                return_exceptions=True,
            )

    @staticmethod
    def _safe_job_context(
        job: AnalysisJob | None,
    ) -> dict[str, str | MediaKind | None]:
        if job is None:
            return {
                "task_id": None,
                "url": None,
                "result_kind": None,
                "final_url": None,
            }

        return {
            "task_id": job.task.task_id,
            "url": job.task.url,
            "result_kind": job.kind,
            "final_url": job.fetch_result.final_url,
        }

    # cleanup is owned by AnalysisResultWriter only
    # lane does not cleanup on normal path
