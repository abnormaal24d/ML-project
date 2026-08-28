"""Crawl run lifecycle supervision."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeAlias

from crawler.exceptions.crawler_error import (
    AnalysisLaneFailedError,
    CrawlerDrainStalledError,
)
from crawler.runtime.loop.crawl_run_summary import (
    CrawlRunResult,
    CrawlStopTrigger,
    CrawlTerminalOutcome,
    active_worker_task_summaries,
    build_run_result,
)

if TYPE_CHECKING:
    from crawler.runtime.actions.crawl_runtime_actions import (
        CrawlOutputReadinessReport,
        CrawlRuntimeActions,
    )
    from crawler.runtime.crawler_runtime_session import CrawlerRuntimeSession
    from crawler.runtime.loop.crawl_seed_enqueuer import CrawlerSeedEnqueuer
    from crawler.scheduling.url_scheduler import UrlScheduler
    from crawler.worker.pool.worker_pool import WorkerPool
    from crawler.worker.worker_scaler import WorkerScaler
    from logger.project_logger import ProjectLogger

LoopBreak: TypeAlias = tuple[CrawlStopTrigger, bool] | None


def _classify_terminal_outcome(
    *,
    stop_trigger: CrawlStopTrigger,
) -> CrawlTerminalOutcome:
    """Derive dataset-level outcome from the crawl stop trigger.

    Frontier exhaustion means the crawl lifecycle executed to completion,
    regardless of whether the collected raw data already satisfies the
    training coverage minimums. Coverage shortfalls are dataset coverage
    concerns handled by DataChecker recrawls, not crawl lifecycle failures.
    """

    if stop_trigger is CrawlStopTrigger.CANCELLED:
        return CrawlTerminalOutcome.CANCELLED

    if stop_trigger is CrawlStopTrigger.FAILED:
        return CrawlTerminalOutcome.FAILED

    if stop_trigger in {
        CrawlStopTrigger.FRONTIER_DRAINED,
        CrawlStopTrigger.OUTPUT_READY,
    }:
        return CrawlTerminalOutcome.SUCCESS

    if stop_trigger in {
        CrawlStopTrigger.STOP_REQUESTED,
        CrawlStopTrigger.INTERRUPTED,
        CrawlStopTrigger.DELAYED_BACKLOG_DEFERRED,
        CrawlStopTrigger.NO_ACCEPTED_SEEDS,
    }:
        return CrawlTerminalOutcome.INCOMPLETE

    raise ValueError(f"unsupported crawl stop trigger: {stop_trigger.value}")


class CrawlRunSupervisor:
    """Supervise one complete crawler run lifecycle."""

    def __init__(
        self,
        *,
        drain_delayed_backlog_before_finish: bool,
        max_idle_delay_wait_seconds: float,
        drain_stall_timeout_seconds: float,
        drain_watch_interval_seconds: float,
        scheduler: UrlScheduler,
        worker_pool: WorkerPool,
        worker_scaler: WorkerScaler,
        logger: ProjectLogger,
        seed_enqueuer: CrawlerSeedEnqueuer,
        min_workers: int,
    ) -> None:
        self._drain_delayed_backlog_before_finish = (
            drain_delayed_backlog_before_finish
        )
        self._max_idle_delay_wait_seconds = max_idle_delay_wait_seconds
        self._drain_stall_timeout_seconds = drain_stall_timeout_seconds
        self._drain_watch_interval_seconds = drain_watch_interval_seconds
        self._scheduler = scheduler
        self._worker_pool = worker_pool
        self._worker_scaler = worker_scaler
        self._logger = logger
        self._seed_enqueuer = seed_enqueuer
        self._min_workers = max(0, min_workers)

        self._scheduler_join_task: asyncio.Task[Any] | None = None
        self._worker_failure_task: asyncio.Task[Any] | None = None
        self._drain_watch_task: asyncio.Task[Any] | None = None
        self._worker_scaler_failure_task: asyncio.Task[Any] | None = None

    async def run(
        self,
        *,
        runtime_session: CrawlerRuntimeSession,
    ) -> CrawlRunResult:
        """Run one supervised crawler lifecycle."""

        stop_trigger = CrawlStopTrigger.FAILED
        force_worker_pool_shutdown = False

        try:
            (
                stop_trigger,
                finished_by_frontier_drain,
            ) = await self._run_session(runtime_session=runtime_session)
            readiness = self._collect_readiness(
                finished_by_frontier_drain=finished_by_frontier_drain,
                runtime_session=runtime_session,
            )
            terminal_outcome = _classify_terminal_outcome(
                stop_trigger=stop_trigger,
            )
            if terminal_outcome is CrawlTerminalOutcome.INCOMPLETE:
                self._logger.warning(
                    "crawler_training_data_not_ready",
                    stop_trigger=stop_trigger.value,
                    unmet_requirements=readiness["unmet_requirements"],
                    object_records_total=readiness["object_records_total"],
                    requests_total=readiness["requests_total"],
                    successful_requests_total=(
                        readiness["successful_requests_total"]
                    ),
                    quality_score=readiness["quality_score"],
                    modality_counts=readiness["modality_counts"],
                )
            elif (
                stop_trigger is CrawlStopTrigger.FRONTIER_DRAINED
                and not readiness["ready"]
            ):
                self._logger.warning(
                    "crawler_dataset_coverage_insufficient",
                    unmet_requirements=readiness["unmet_requirements"],
                    object_records_total=readiness["object_records_total"],
                    requests_total=readiness["requests_total"],
                    successful_requests_total=(
                        readiness["successful_requests_total"]
                    ),
                    quality_score=readiness["quality_score"],
                    modality_counts=readiness["modality_counts"],
                )

            worker_snapshot = self._worker_pool.snapshot(
                now=asyncio.get_running_loop().time()
            )
            return build_run_result(
                worker_snapshot=worker_snapshot,
                stop_trigger=stop_trigger,
                terminal_outcome=terminal_outcome,
                requests_total=readiness["requests_total"],
                successful_requests_total=readiness[
                    "successful_requests_total"
                ],
                object_records_total=readiness["object_records_total"],
                root_seeds_total=worker_snapshot.root_seeds_total,
                root_seeds_succeeded=worker_snapshot.root_seeds_succeeded,
                root_seeds_transient_failed=worker_snapshot.root_seeds_transient_failed,
                root_seeds_governance_blocked=worker_snapshot.root_seeds_governance_blocked,
                unmet_requirements=readiness["unmet_requirements"],
                quality_score=readiness["quality_score"],
                modality_counts=dict(readiness["modality_counts"]) or None,
                output_ready=readiness["ready"],
            )

        except asyncio.CancelledError:
            stop_trigger = CrawlStopTrigger.CANCELLED
            force_worker_pool_shutdown = True
            raise
        except KeyboardInterrupt:
            stop_trigger = CrawlStopTrigger.INTERRUPTED
            force_worker_pool_shutdown = True
            raise
        except (
            RuntimeError,
            OSError,
            ValueError,
            CrawlerDrainStalledError,
            AnalysisLaneFailedError,
        ):
            stop_trigger = CrawlStopTrigger.FAILED
            force_worker_pool_shutdown = True
            raise
        finally:
            await self._cleanup(
                runtime_session=runtime_session,
                stop_trigger=stop_trigger,
                force_worker_pool_shutdown=force_worker_pool_shutdown,
            )

    async def _run_session(
        self,
        *,
        runtime_session: CrawlerRuntimeSession,
    ) -> tuple[CrawlStopTrigger, bool]:
        seed_result = await self._seed_enqueuer.prepare()

        if (
            seed_result.accepted_seeds == 0
            and seed_result.restored_tasks == 0
            and seed_result.requeued_dead_letters == 0
        ):
            self._logger.info(
                "crawler_start_skipped_no_work",
                total_seeds=seed_result.total_seeds,
                accepted_seeds=seed_result.accepted_seeds,
                rejected_seeds=seed_result.rejected_seeds,
                restored_tasks=seed_result.restored_tasks,
                requeued_dead_letters=seed_result.requeued_dead_letters,
                seed_source=seed_result.seed_source,
            )
            return CrawlStopTrigger.NO_ACCEPTED_SEEDS, False

        self._logger.info(
            "crawler_started",
            total_seeds=seed_result.total_seeds,
            accepted_seeds=seed_result.accepted_seeds,
            rejected_seeds=seed_result.rejected_seeds,
            seed_source=seed_result.seed_source,
            restored_tasks=seed_result.restored_tasks,
            requeued_dead_letters=seed_result.requeued_dead_letters,
            min_workers=self._min_workers,
        )

        await self._worker_pool.scale_to(self._min_workers)
        self._worker_scaler.start()
        self._start_run_watch_tasks()

        loop = asyncio.get_running_loop()
        stop_trigger = CrawlStopTrigger.FAILED
        finished_by_frontier_drain = False

        while True:
            loop_break = await self._run_one_iteration(
                runtime_session=runtime_session,
                loop=loop,
            )
            if loop_break is None:
                continue

            stop_trigger, finished_by_frontier_drain = loop_break
            break

        await self._cancel_scheduler_join_if_running()
        if stop_trigger is CrawlStopTrigger.OUTPUT_READY:
            await self._stop_discovery_after_output_ready()
        self._worker_pool.raise_if_failed()
        await self._drain_media_analysis(runtime_session=runtime_session)

        return stop_trigger, finished_by_frontier_drain

    def _start_run_watch_tasks(self) -> None:
        self._scheduler_join_task = asyncio.create_task(
            self._scheduler.join(),
            name="crawler-scheduler-join",
        )

        self._drain_watch_task = asyncio.create_task(
            self._watch_for_stalled_drain(),
            name="crawler-drain-watchdog",
        )

        self._worker_scaler_failure_task = (
            asyncio.create_task(
                self._worker_scaler.wait_for_failure(),
                name="crawler-worker-scaler-failure-watch",
            )
            if self._worker_scaler.is_enabled
            else None
        )

        if self._worker_pool.fail_fast_on_processing_error:
            self._worker_failure_task = asyncio.create_task(
                self._worker_pool.wait_for_failure(),
                name="crawler-worker-failure-watch",
            )
            return

        self._worker_failure_task = None

    def _run_watch_tasks(self) -> set[asyncio.Task[Any]]:
        tasks = {
            task
            for task in (
                self._scheduler_join_task,
                self._worker_failure_task,
                self._drain_watch_task,
                self._worker_scaler_failure_task,
            )
            if task is not None
        }

        if not tasks:
            raise RuntimeError("crawler run watch tasks were not started")

        return tasks

    async def _handle_completed_run_watch_tasks(
        self,
        done: set[asyncio.Task[Any]],
    ) -> bool:
        if not done:
            return False

        scheduler_join = self._scheduler_join_task
        worker_failure = self._worker_failure_task
        drain_watch = self._drain_watch_task
        worker_scaler_failure = self._worker_scaler_failure_task

        if scheduler_join is not None and scheduler_join in done:
            await scheduler_join

            self._logger.info("crawler_frontier_drained")

            await self._cancel_task_if_running(drain_watch)
            await self._cancel_task_if_running(worker_scaler_failure)
            return True

        if worker_failure is not None and worker_failure in done:
            await self._cancel_task_if_running(scheduler_join)
            await self._cancel_task_if_running(drain_watch)
            await self._cancel_task_if_running(worker_scaler_failure)

            self._logger.error("crawler_worker_failure_detected")

            self._worker_pool.raise_if_failed()

        if worker_scaler_failure is not None and worker_scaler_failure in done:
            await self._cancel_task_if_running(scheduler_join)
            await self._cancel_task_if_running(drain_watch)

            self._logger.error("crawler_worker_scaler_failure_detected")

            await worker_scaler_failure
            return False

        if drain_watch is not None and drain_watch in done:
            await self._cancel_task_if_running(scheduler_join)
            await self._cancel_task_if_running(worker_scaler_failure)

            self._logger.error("crawler_drain_watchdog_completed")

            await drain_watch

        return False

    async def _run_one_iteration(
        self,
        *,
        runtime_session: CrawlerRuntimeSession,
        loop: asyncio.AbstractEventLoop,
    ) -> LoopBreak:
        runtime_actions = runtime_session.runtime_actions

        control_break, paused = await self._control_break_if_requested(
            runtime_actions=runtime_actions
        )
        if control_break is not None:
            return control_break
        if paused:
            return None

        readiness = runtime_actions.crawl_output_readiness_report()
        if readiness["ready"]:
            self._logger.info(
                "crawler_output_ready",
                stop_trigger=CrawlStopTrigger.OUTPUT_READY.value,
                object_records_total=readiness["object_records_total"],
                requests_total=readiness["requests_total"],
                successful_requests_total=(
                    readiness["successful_requests_total"]
                ),
                quality_score=readiness["quality_score"],
                modality_counts=readiness["modality_counts"],
            )
            return CrawlStopTrigger.OUTPUT_READY, True

        if await self._should_defer_delayed_backlog():
            return CrawlStopTrigger.DELAYED_BACKLOG_DEFERRED, False

        timeout = min(
            max(
                0.0,
                runtime_actions.wakeup_deadline() - loop.time(),
            ),
            runtime_actions.control_poll_interval_seconds(),
        )

        done, _ = await asyncio.wait(
            self._run_watch_tasks(),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        control_break, paused = await self._control_break_if_requested(
            runtime_actions=runtime_actions
        )
        if control_break is not None:
            return control_break
        if paused:
            return None

        if await self._handle_completed_run_watch_tasks(done):
            return CrawlStopTrigger.FRONTIER_DRAINED, True

        await runtime_actions.run_due_actions(current_time=loop.time())

        if await self._should_defer_delayed_backlog():
            return CrawlStopTrigger.DELAYED_BACKLOG_DEFERRED, False

        return None

    async def _control_break_if_requested(
        self,
        *,
        runtime_actions: CrawlRuntimeActions,
    ) -> tuple[LoopBreak, bool]:
        """Apply stop and resume controls at an iteration boundary."""

        if runtime_actions.should_stop():
            self._logger.warning("crawler_stop_requested")
            return (CrawlStopTrigger.STOP_REQUESTED, False), False

        if await runtime_actions.handle_pause_if_requested():
            return None, True

        await runtime_actions.resume_if_needed()
        return None, False

    async def _stop_discovery_after_output_ready(self) -> None:
        """Discard queued work while allowing currently active writes to finish."""

        snapshot = await self._scheduler.snapshot()
        await self._scheduler.close(discard_pending=True)

        self._logger.info(
            "crawler_output_ready_frontier_discarded",
            queued=snapshot.queued,
            delayed_queued=snapshot.delayed_queued,
            inflight=snapshot.inflight,
        )

        # The scheduler has stopped accepting discovery and has no queued
        # frontier left. Joining now waits only for work already handed to a
        # worker, so their dataset writes and completion callbacks can finish
        # before raw-run finalization begins.
        await self._scheduler.join()

    async def _should_defer_delayed_backlog(self) -> bool:
        if self._drain_delayed_backlog_before_finish:
            return False

        max_idle_delay_wait_seconds = self._max_idle_delay_wait_seconds
        snapshot = await self._scheduler.snapshot()
        worker_snapshot = self._worker_pool.snapshot(
            now=asyncio.get_running_loop().time()
        )

        if (
            snapshot.queued != 0
            or snapshot.delayed_queued <= 0
            or snapshot.inflight != 0
            or worker_snapshot.busy_worker_count != 0
            or worker_snapshot.completed_task_count <= 0
        ):
            return False

        next_wait_seconds = snapshot.next_delayed_ready_in_seconds
        if next_wait_seconds is None:
            return False

        if float(next_wait_seconds) <= max_idle_delay_wait_seconds:
            return False

        self._logger.info(
            "crawler_delayed_backlog_deferred",
            stop_trigger=CrawlStopTrigger.DELAYED_BACKLOG_DEFERRED.value,
            delayed_queued=snapshot.delayed_queued,
            next_delayed_ready_in_seconds=round(
                float(next_wait_seconds),
                4,
            ),
            max_idle_delay_wait_seconds=round(
                max_idle_delay_wait_seconds,
                4,
            ),
            completed_tasks=worker_snapshot.completed_task_count,
        )
        return True

    def _collect_readiness(
        self,
        *,
        finished_by_frontier_drain: bool,
        runtime_session: CrawlerRuntimeSession,
    ) -> CrawlOutputReadinessReport:
        if not finished_by_frontier_drain:
            return {
                "ready": False,
                "unmet_requirements": (),
                "object_records_total": 0,
                "requests_total": 0,
                "successful_requests_total": 0,
                "quality_score": 0.0,
                "modality_counts": {},
            }
        return runtime_session.runtime_actions.crawl_output_readiness_report()

    async def _drain_media_analysis(
        self,
        *,
        runtime_session: CrawlerRuntimeSession,
    ) -> None:
        analysis_router = runtime_session.analysis_router

        if analysis_router is None:
            return

        try:
            await asyncio.wait_for(
                analysis_router.drain(),
                timeout=self._drain_stall_timeout_seconds,
            )

            self._logger.info("crawler_media_analysis_drained")

        except AnalysisLaneFailedError as exc:
            self._logger.error(
                "crawler_media_analysis_failed",
                media_analysis_snapshot=analysis_router.snapshot(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await analysis_router.stop(force=True)
            raise

        except asyncio.TimeoutError as exc:
            self._logger.error(
                "crawler_media_analysis_drain_stalled",
                timeout_seconds=self._drain_stall_timeout_seconds,
                media_analysis_snapshot=analysis_router.snapshot(),
                error_type=type(exc).__name__,
                error=str(exc),
            )

            await analysis_router.stop(force=True)

            self._logger.warning("crawler_media_analysis_force_stopped")
            raise CrawlerDrainStalledError(
                "crawler media analysis drain stalled"
            ) from exc

    async def _watch_for_stalled_drain(self) -> None:
        loop = asyncio.get_running_loop()
        stall_started_at: float | None = None
        last_completed_task_count: int | None = None

        while True:
            await asyncio.sleep(self._drain_watch_interval_seconds)

            scheduler_snapshot = await self._scheduler.snapshot()
            worker_snapshot = self._worker_pool.snapshot(now=loop.time())
            completed_task_count = worker_snapshot.completed_task_count

            if last_completed_task_count is None:
                last_completed_task_count = completed_task_count

            if completed_task_count != last_completed_task_count:
                stall_started_at = None
                last_completed_task_count = completed_task_count
                continue

            queued = int(scheduler_snapshot.queued or 0)
            delayed = int(scheduler_snapshot.delayed_queued or 0)
            inflight = int(scheduler_snapshot.inflight or 0)
            busy = worker_snapshot.busy_worker_count

            drain_is_waiting_on_workers = (
                queued == 0 and delayed == 0 and inflight > 0 and busy > 0
            )

            if not drain_is_waiting_on_workers:
                stall_started_at = None
                continue

            now = loop.time()

            if stall_started_at is None:
                stall_started_at = now
                self._logger.warning(
                    "crawler_drain_waiting_on_busy_workers",
                    queued=queued,
                    delayed=delayed,
                    inflight=inflight,
                    busy_workers=busy,
                    completed_tasks=completed_task_count,
                    active_worker_tasks=(
                        active_worker_task_summaries(worker_snapshot)
                    ),
                )
                continue

            stalled_for_seconds = now - stall_started_at
            if stalled_for_seconds < self._drain_stall_timeout_seconds:
                continue

            self._logger.error(
                "crawler_drain_stalled",
                queued=queued,
                delayed=delayed,
                inflight=inflight,
                busy_workers=busy,
                completed_tasks=completed_task_count,
                stalled_for_seconds=round(stalled_for_seconds, 3),
                stall_timeout_seconds=self._drain_stall_timeout_seconds,
                active_worker_tasks=(
                    active_worker_task_summaries(worker_snapshot)
                ),
            )

            raise CrawlerDrainStalledError(
                "crawler drain stalled with no queued work and busy workers"
            )

    async def _cancel_task_if_running(
        self,
        task: asyncio.Task[Any] | None,
    ) -> None:
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _cancel_scheduler_join_if_running(self) -> None:
        await self._cancel_task_if_running(self._scheduler_join_task)

    async def _cancel_run_watch_tasks(self) -> None:
        await self._cancel_task_if_running(self._scheduler_join_task)
        await self._cancel_task_if_running(self._worker_failure_task)
        await self._cancel_task_if_running(self._drain_watch_task)
        await self._cancel_task_if_running(self._worker_scaler_failure_task)

    async def _cleanup(
        self,
        *,
        runtime_session: CrawlerRuntimeSession,
        stop_trigger: CrawlStopTrigger,
        force_worker_pool_shutdown: bool,
    ) -> None:
        cleanup_error_count = 0

        try:
            await self._cancel_run_watch_tasks()
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            self._logger.error(
                "crawler_watch_tasks_cancel_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        try:
            await self._worker_scaler.stop()
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            self._logger.error(
                "crawler_worker_scaler_stop_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        try:
            await self._scheduler.close()
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            self._logger.error(
                "crawler_scheduler_close_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        try:
            await self._worker_pool.aclose()
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            force_worker_pool_shutdown = True
            self._logger.error(
                "crawler_worker_pool_close_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        analysis_router = runtime_session.analysis_router

        if analysis_router is not None:
            try:
                await analysis_router.stop(
                    force=force_worker_pool_shutdown,
                )
            except (RuntimeError, OSError, ValueError) as exc:
                cleanup_error_count += 1
                self._logger.error(
                    "crawler_media_analysis_stop_failed",
                    forced=force_worker_pool_shutdown,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        try:
            await runtime_session.state_writer.persist_on_completion()
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            self._logger.error(
                "crawler_completion_state_persist_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        try:
            self._worker_pool.set_task_result_callback(None)
        except (RuntimeError, OSError, ValueError) as exc:
            cleanup_error_count += 1
            self._logger.error(
                "crawler_result_callback_unbind_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        worker_snapshot = self._worker_pool.snapshot(
            now=asyncio.get_running_loop().time()
        )

        if cleanup_error_count:
            self._logger.warning(
                "crawler_cleanup_completed_with_errors",
                stop_trigger=stop_trigger.value,
                error_count=cleanup_error_count,
            )
        else:
            self._logger.debug(
                "crawler_cleanup_completed",
                stop_trigger=stop_trigger.value,
            )

        self._logger.info(
            "crawler_finished",
            stop_trigger=stop_trigger.value,
            size=worker_snapshot.size,
            effective_worker_count=worker_snapshot.effective_worker_count,
            busy_worker_count=worker_snapshot.busy_worker_count,
            completed_task_count=worker_snapshot.completed_task_count,
            failure_count=worker_snapshot.failure_count,
            non_fatal_timeout_count=worker_snapshot.non_fatal_timeout_count,
            retry_exhausted_count=worker_snapshot.retry_exhausted_count,
            average_processing_seconds=round(
                worker_snapshot.average_processing_seconds,
                3,
            ),
        )
