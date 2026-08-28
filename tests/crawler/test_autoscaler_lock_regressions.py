"""Regression tests for worker scaler supervision and lock ownership."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.collection.autoscaling import AutoscalerSettings
from config.settings.crawler import CrawlerSettings
from crawler.runtime.actions.crawl_runtime_actions import (
    CrawlRuntimeActions,
)
from crawler.runtime.concurrency import (
    TransientLockRaceError,
    condition_notify_all,
)
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.runtime.crawler import Crawler
from crawler.runtime.loop.crawl_run_summary import CrawlStopTrigger
from crawler.runtime.loop.crawl_run_supervisor import CrawlRunSupervisor
from crawler.worker.pool.worker_pool import WorkerPool
from crawler.worker.worker_loop.worker_state import WorkerState
from crawler.worker.worker_scaler import WorkerScaler


class _Logger:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


class _WorkerPool:
    def __init__(self):
        self.effective_worker_count = 1
        self.busy_worker_count = 1
        self.failure_count = 0
        self.longest_busy_seconds = 0.0

    def snapshot(self, *, now: float):
        return self

    async def scale_to(self, target: int):
        self.effective_worker_count = target


class _Scheduler:
    def __init__(self):
        self.queued = 0
        self.delayed_queued = 0
        self.inflight = 0
        self.runnable_slots = 0
        self.pending_hosts = 0
        self.delayed_pending_hosts = 0
        self.total_pending_hosts = 0

    async def snapshot(self):
        return self


@pytest.mark.asyncio
async def test_worker_scaler_failure_is_observed_once() -> None:
    settings = AutoscalerSettings(
        enabled=True,
        cancel_timeout_seconds=0.1,
        check_interval_seconds=0.01,
        min_workers=1,
        max_workers=8,
        scale_up_delay_seconds=0.9,
        scale_down_delay_seconds=1.2,
        scale_up_cooldown_seconds=0.3,
        scale_down_cooldown_seconds=2.0,
        max_scale_up_step=2,
        max_scale_down_step=1,
        failure_burst_threshold=2,
        slow_task_seconds_threshold=3.0,
    )
    worker_pool = _WorkerPool()
    scheduler = _Scheduler()
    logger = _Logger()

    scaler = WorkerScaler(
        settings=settings,
        worker_pool=worker_pool,
        scheduler=scheduler,
        logger=logger,
    )

    async def fail_tick() -> None:
        raise RuntimeError("worker scaler boom")

    scaler._scale_once = fail_tick  # type: ignore[method-assign]
    scaler.start()
    with pytest.raises(RuntimeError, match="worker scaler boom"):
        await scaler.wait_for_failure()
    await scaler.stop()


class _SynchronousWorkerScaler:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.is_enabled = False

    def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


class _ScaleRecordingPool:
    def __init__(self) -> None:
        self.targets: list[int] = []

    async def scale_to(self, target: int) -> None:
        self.targets.append(target)


def test_stop_signal_is_consumed_once_and_status_does_not_consume_it(
    tmp_path: Path,
) -> None:
    control_directory = CrawlerControlDirectory(
        settings=CrawlerSettings(control_directory="runtime/control"),
        project_root=tmp_path,
    )
    control_directory.request_stop()

    status = control_directory.status(
        crawl_state_status=None,
        crawl_state_path=tmp_path / "crawl-state.json",
    )

    assert status["stop_requested"] is True
    assert control_directory.consume_stop() is True
    assert control_directory.should_stop() is False
    assert control_directory.consume_stop() is False


def test_runtime_actions_consumes_stop_signal_once(tmp_path: Path) -> None:
    control_directory = CrawlerControlDirectory(
        settings=CrawlerSettings(control_directory="runtime/control"),
        project_root=tmp_path,
    )
    control_directory.request_stop()
    actions = CrawlRuntimeActions(
        shutdown_poll_interval_seconds=0.01,
        crawl_output_gate=SimpleNamespace(),
        worker_scaler=SimpleNamespace(),
        worker_pool=SimpleNamespace(),
        control_directory=control_directory,
        logger=_Logger(),
        min_workers=1,
        now=0.0,
        progress_interval=1.0,
        checkpoint_interval=None,
        metrics_interval=None,
        emit_progress=lambda: None,
        emit_metrics=lambda: None,
        state_writer=SimpleNamespace(),
        dataset_snapshot=None,
        metrics=None,
    )

    assert actions.should_stop() is True
    assert actions.should_stop() is False


@pytest.mark.asyncio
async def test_runtime_actions_resume_with_synchronous_scaler_start() -> None:
    worker_scaler = _SynchronousWorkerScaler()
    worker_pool = _ScaleRecordingPool()
    actions = CrawlRuntimeActions(
        shutdown_poll_interval_seconds=0.01,
        crawl_output_gate=SimpleNamespace(),
        worker_scaler=worker_scaler,
        worker_pool=worker_pool,
        control_directory=SimpleNamespace(),
        logger=_Logger(),
        min_workers=3,
        now=0.0,
        progress_interval=1.0,
        checkpoint_interval=None,
        metrics_interval=None,
        emit_progress=lambda: None,
        emit_metrics=lambda: None,
        state_writer=SimpleNamespace(),
        dataset_snapshot=None,
        metrics=None,
    )
    actions._pause_active = True

    await actions.resume_if_needed()

    assert worker_pool.targets == [3]
    assert worker_scaler.start_calls == 1


@pytest.mark.asyncio
async def test_supervisor_starts_scaler_without_awaiting_none() -> None:
    worker_scaler = _SynchronousWorkerScaler()
    worker_pool = _ScaleRecordingPool()
    worker_pool.raise_if_failed = lambda: None  # type: ignore[attr-defined]
    seed_result = SimpleNamespace(
        total_seeds=1,
        accepted_seeds=1,
        rejected_seeds=0,
        restored_tasks=0,
        requeued_dead_letters=0,
        seed_source="test",
    )

    async def prepare() -> SimpleNamespace:
        return seed_result

    runtime_session = SimpleNamespace(
        runtime_actions=SimpleNamespace(),
        analysis_router=None,
    )
    received_sessions: list[object] = []

    async def stop_after_first_iteration(
        *,
        runtime_session: object,
        **_kwargs: object,
    ):
        received_sessions.append(runtime_session)
        return CrawlStopTrigger.STOP_REQUESTED, False

    async def no_op(**_kwargs: object) -> None:
        return None

    async def drain(*, runtime_session: object) -> None:
        received_sessions.append(runtime_session)

    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(),
        worker_pool=worker_pool,
        worker_scaler=worker_scaler,
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(prepare=prepare),
        min_workers=2,
    )
    supervisor._start_run_watch_tasks = lambda: None  # type: ignore[method-assign]
    supervisor._run_one_iteration = stop_after_first_iteration  # type: ignore[method-assign]
    supervisor._cancel_scheduler_join_if_running = no_op  # type: ignore[method-assign]
    supervisor._drain_media_analysis = drain  # type: ignore[method-assign]

    stop_trigger, finished_by_frontier_drain = await supervisor._run_session(
        runtime_session=runtime_session,
    )

    assert stop_trigger is CrawlStopTrigger.STOP_REQUESTED
    assert finished_by_frontier_drain is False
    assert worker_pool.targets == [2]
    assert worker_scaler.start_calls == 1
    assert received_sessions == [runtime_session, runtime_session]


@pytest.mark.asyncio
async def test_supervisor_requires_only_current_runtime_actions_api() -> None:
    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(),
        worker_pool=SimpleNamespace(),
        worker_scaler=SimpleNamespace(),
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )
    runtime_actions = SimpleNamespace(should_stop=lambda: True)

    loop_break = await supervisor._run_one_iteration(
        runtime_session=SimpleNamespace(runtime_actions=runtime_actions),
        loop=asyncio.get_running_loop(),
    )

    assert loop_break == (CrawlStopTrigger.STOP_REQUESTED, False)


@pytest.mark.asyncio
async def test_supervisor_bounds_wait_and_observes_stop_before_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RuntimeActions:
        def __init__(self) -> None:
            self._stop_checks = 0
            self.resume_calls = 0

        def should_stop(self) -> bool:
            self._stop_checks += 1
            return self._stop_checks == 2

        async def handle_pause_if_requested(self) -> bool:
            return False

        async def resume_if_needed(self) -> None:
            self.resume_calls += 1

        def crawl_output_readiness_report(self) -> dict[str, object]:
            return {"ready": False}

        def wakeup_deadline(self) -> float:
            return asyncio.get_running_loop().time() + 60.0

        def control_poll_interval_seconds(self) -> float:
            return 0.02

    actions = _RuntimeActions()
    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(),
        worker_pool=SimpleNamespace(),
        worker_scaler=SimpleNamespace(),
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )
    observed_timeouts: list[float | None] = []

    async def controlled_wait(
        *_tasks: object,
        timeout: float | None = None,
        **_kwargs: object,
    ) -> tuple[set[object], set[object]]:
        observed_timeouts.append(timeout)
        return set(), set()

    async def no_deferred_backlog() -> bool:
        return False

    async def drained_before_control(_: object) -> bool:
        raise AssertionError("frontier drain was handled before stop control")

    monkeypatch.setattr(
        "crawler.runtime.loop.crawl_run_supervisor.asyncio.wait",
        controlled_wait,
    )
    supervisor._run_watch_tasks = lambda: set()  # type: ignore[method-assign]
    supervisor._should_defer_delayed_backlog = no_deferred_backlog  # type: ignore[method-assign]
    supervisor._handle_completed_run_watch_tasks = drained_before_control  # type: ignore[method-assign]

    loop_break = await supervisor._run_one_iteration(
        runtime_session=SimpleNamespace(runtime_actions=actions),
        loop=asyncio.get_running_loop(),
    )

    assert loop_break == (CrawlStopTrigger.STOP_REQUESTED, False)
    assert observed_timeouts == [0.02]
    assert actions.resume_calls == 1


@pytest.mark.asyncio
async def test_supervisor_observes_pause_before_drain_after_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RuntimeActions:
        def __init__(self) -> None:
            self._pause_checks = 0

        def should_stop(self) -> bool:
            return False

        async def handle_pause_if_requested(self) -> bool:
            self._pause_checks += 1
            return self._pause_checks == 2

        async def resume_if_needed(self) -> None:
            return None

        def crawl_output_readiness_report(self) -> dict[str, object]:
            return {"ready": False}

        def wakeup_deadline(self) -> float:
            return asyncio.get_running_loop().time() + 60.0

        def control_poll_interval_seconds(self) -> float:
            return 0.02

    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(),
        worker_pool=SimpleNamespace(),
        worker_scaler=SimpleNamespace(),
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )

    async def controlled_wait(
        *_tasks: object,
        **_kwargs: object,
    ) -> tuple[set[object], set[object]]:
        return set(), set()

    async def no_deferred_backlog() -> bool:
        return False

    async def drained_before_control(_: object) -> bool:
        raise AssertionError("frontier drain was handled before pause control")

    monkeypatch.setattr(
        "crawler.runtime.loop.crawl_run_supervisor.asyncio.wait",
        controlled_wait,
    )
    supervisor._run_watch_tasks = lambda: set()  # type: ignore[method-assign]
    supervisor._should_defer_delayed_backlog = no_deferred_backlog  # type: ignore[method-assign]
    supervisor._handle_completed_run_watch_tasks = drained_before_control  # type: ignore[method-assign]

    loop_break = await supervisor._run_one_iteration(
        runtime_session=SimpleNamespace(runtime_actions=_RuntimeActions()),
        loop=asyncio.get_running_loop(),
    )

    assert loop_break is None


@pytest.mark.asyncio
async def test_supervisor_cleanup_does_not_call_removed_pause_api() -> None:
    class _CleanupPool(_ScaleRecordingPool):
        async def aclose(self) -> None:
            return None

        def set_task_result_callback(self, _callback: object | None) -> None:
            return None

        def snapshot(self, *, now: float) -> SimpleNamespace:
            del now
            return SimpleNamespace(
                size=0,
                effective_worker_count=0,
                busy_worker_count=0,
                completed_task_count=0,
                failure_count=0,
                non_fatal_timeout_count=0,
                retry_exhausted_count=0,
                average_processing_seconds=0.0,
            )

    async def close() -> None:
        return None

    async def persist_on_completion() -> None:
        return None

    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(close=close),
        worker_pool=_CleanupPool(),
        worker_scaler=_SynchronousWorkerScaler(),
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )

    await supervisor._cleanup(
        runtime_session=SimpleNamespace(
            runtime_actions=SimpleNamespace(),
            state_writer=SimpleNamespace(
                persist_on_completion=persist_on_completion,
            ),
            analysis_router=None,
        ),
        stop_trigger=CrawlStopTrigger.STOP_REQUESTED,
        force_worker_pool_shutdown=False,
    )


@pytest.mark.asyncio
async def test_crawler_builds_one_runtime_session_after_enabled_check() -> (
    None
):
    built_sessions: list[object] = []
    received_sessions: list[object] = []
    runtime_session = object()

    class _RecordingSupervisor:
        async def run(self, *, runtime_session: object) -> str:
            received_sessions.append(runtime_session)
            return "completed"

    def build_runtime_session() -> object:
        built_sessions.append(runtime_session)
        return runtime_session

    crawler = Crawler(
        enabled=True,
        worker_pool=SimpleNamespace(
            set_task_result_callback=lambda _callback: None,
        ),
        logger=_Logger(),
        control_directory=SimpleNamespace(ensure_exists=lambda: None),
        task_feedback=SimpleNamespace(),
        run_supervisor=_RecordingSupervisor(),
        build_runtime_session=build_runtime_session,
    )

    result = await crawler.crawl()

    assert result == "completed"
    assert built_sessions == [runtime_session]
    assert received_sessions == [runtime_session]


@pytest.mark.asyncio
async def test_disabled_crawler_does_not_build_runtime_session() -> None:
    class _RecordingSupervisor:
        async def run(self, *, runtime_session: object) -> str:
            raise AssertionError("disabled crawler called supervisor.run")

    def build_runtime_session() -> object:
        raise AssertionError("disabled crawler built a runtime session")

    worker_snapshot = SimpleNamespace(
        completed_task_count=0,
        failure_count=0,
        non_fatal_timeout_count=0,
        retry_exhausted_count=0,
        average_processing_seconds=0.0,
    )
    crawler = Crawler(
        enabled=False,
        worker_pool=SimpleNamespace(
            snapshot=lambda *, now: worker_snapshot,
        ),
        logger=_Logger(),
        control_directory=SimpleNamespace(),
        task_feedback=SimpleNamespace(),
        run_supervisor=_RecordingSupervisor(),
        build_runtime_session=build_runtime_session,
    )

    result = await crawler.crawl()

    assert result.stop_trigger is CrawlStopTrigger.FAILED


@pytest.mark.asyncio
async def test_supervisor_uses_one_runtime_session_throughout_run() -> None:
    runtime_session = SimpleNamespace(
        runtime_actions=SimpleNamespace(),
        state_writer=SimpleNamespace(),
        analysis_router=SimpleNamespace(),
    )
    received: list[tuple[str, object]] = []
    worker_snapshot = SimpleNamespace(
        completed_task_count=0,
        failure_count=0,
        non_fatal_timeout_count=0,
        retry_exhausted_count=0,
        average_processing_seconds=0.0,
        root_seeds_total=0,
        root_seeds_succeeded=0,
        root_seeds_transient_failed=0,
        root_seeds_governance_blocked=0,
    )
    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=SimpleNamespace(),
        worker_pool=SimpleNamespace(
            snapshot=lambda *, now: worker_snapshot,
        ),
        worker_scaler=SimpleNamespace(),
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )

    async def run_session(*, runtime_session: object):
        received.append(("run", runtime_session))
        return CrawlStopTrigger.STOP_REQUESTED, False

    def collect_readiness(
        *,
        finished_by_frontier_drain: bool,
        runtime_session: object,
    ) -> dict[str, object]:
        assert finished_by_frontier_drain is False
        received.append(("readiness", runtime_session))
        return {
            "ready": False,
            "unmet_requirements": (),
            "object_records_total": 0,
            "requests_total": 0,
            "successful_requests_total": 0,
            "quality_score": 0.0,
            "modality_counts": {},
        }

    async def cleanup(*, runtime_session: object, **_kwargs: object) -> None:
        received.append(("cleanup", runtime_session))

    supervisor._run_session = run_session  # type: ignore[method-assign]
    supervisor._collect_readiness = collect_readiness  # type: ignore[method-assign]
    supervisor._cleanup = cleanup  # type: ignore[method-assign]

    await supervisor.run(runtime_session=runtime_session)

    assert received == [
        ("run", runtime_session),
        ("readiness", runtime_session),
        ("cleanup", runtime_session),
    ]


@pytest.mark.asyncio
async def test_disabled_worker_scaler_does_not_create_failure_watcher() -> (
    None
):
    async def wait_forever() -> None:
        await asyncio.Event().wait()

    worker_scaler = SimpleNamespace(
        is_enabled=False,
        wait_for_failure=wait_forever,
    )
    scheduler = SimpleNamespace(join=wait_forever)
    worker_pool = SimpleNamespace(fail_fast_on_processing_error=False)
    supervisor = CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=True,
        max_idle_delay_wait_seconds=0.5,
        drain_stall_timeout_seconds=180.0,
        drain_watch_interval_seconds=5.0,
        scheduler=scheduler,
        worker_pool=worker_pool,
        worker_scaler=worker_scaler,
        logger=_Logger(),
        seed_enqueuer=SimpleNamespace(),
        min_workers=1,
    )
    supervisor._watch_for_stalled_drain = wait_forever  # type: ignore[method-assign]

    supervisor._start_run_watch_tasks()
    try:
        assert supervisor._worker_scaler_failure_task is None
    finally:
        await supervisor._cancel_run_watch_tasks()


def test_condition_notify_uses_typed_precondition() -> None:
    condition = asyncio.Condition()
    with pytest.raises(TransientLockRaceError):
        condition_notify_all(condition)


@pytest.mark.asyncio
async def test_enqueue_many_wakes_waiting_dispatcher() -> None:
    from crawler.scheduling.admission.scheduler_frontier import (
        SchedulerFrontier,
    )

    condition = asyncio.Condition()
    awakened = asyncio.Event()
    waiting = asyncio.Event()

    async def waiter() -> None:
        async with condition:
            waiting.set()
            await condition.wait()
            awakened.set()

    waiter_task = asyncio.create_task(waiter())
    await waiting.wait()

    frontier = object.__new__(SchedulerFrontier)
    frontier._condition = condition
    frontier._logger = _Logger()
    frontier._max_rejection_samples = 0
    accepted = SimpleNamespace(accepted=True)
    frontier.enqueue_locked = lambda **_kwargs: (accepted, None, None)  # type: ignore[method-assign]

    async with condition:
        decisions = frontier.enqueue_many_locked(tasks=[SimpleNamespace()])

    await asyncio.wait_for(awakened.wait(), timeout=1.0)
    await waiter_task
    assert len(decisions) == 1


def test_compressed_byte_adapter_is_instance_based() -> None:
    from crawler.fetching.network.body.stream_writer import (
        _compressed_byte_count,
    )

    assert _compressed_byte_count(SimpleNamespace(total_raw_bytes=12)) == 12
    assert (
        _compressed_byte_count(SimpleNamespace(total_compressed_bytes=9)) == 9
    )
    assert _compressed_byte_count(SimpleNamespace()) is None


class _WorkerSnapshot:
    def __init__(self, *, busy: int = 0):
        self.busy_worker_count = busy
        self.failure_count = 0
        self.longest_busy_seconds = 0.0


def _desired(
    *,
    min_workers: int = 1,
    max_workers: int = 8,
    inflight: int = 0,
    runnable_slots: int = 0,
    busy: int = 0,
) -> int:
    settings = AutoscalerSettings(
        min_workers=min_workers,
        max_workers=max_workers,
        max_scale_up_step=4,
        max_scale_down_step=2,
    )
    scheduler = SimpleNamespace(
        inflight=inflight, runnable_slots=runnable_slots
    )
    scaler = object.__new__(WorkerScaler)
    scaler._settings = settings
    return scaler._desired_workers(
        worker=_WorkerSnapshot(busy=busy),
        scheduler=scheduler,
    )


def test_desired_workers_follows_runnable_slots() -> None:
    assert _desired(inflight=0, runnable_slots=5) == 5


def test_desired_workers_ignores_pure_queued_backlog() -> None:
    queued_backlog = SimpleNamespace(
        queued=100,
        inflight=1,
        runnable_slots=0,
    )
    settings = AutoscalerSettings(
        min_workers=1,
        max_workers=8,
        max_scale_up_step=4,
        max_scale_down_step=2,
    )
    scaler = object.__new__(WorkerScaler)
    scaler._settings = settings
    assert (
        scaler._desired_workers(
            worker=_WorkerSnapshot(),
            scheduler=queued_backlog,
        )
        == 1
    )


def test_desired_workers_tracks_inflight_plus_slots() -> None:
    assert _desired(inflight=2, runnable_slots=3) == 5


def test_desired_workers_respects_min_and_max() -> None:
    assert _desired(min_workers=3, inflight=0, runnable_slots=0) == 3
    assert _desired(max_workers=4, inflight=3, runnable_slots=3) == 4


def test_desired_workers_never_below_busy_count() -> None:
    assert _desired(busy=6, inflight=1, runnable_slots=1) == 6


def test_compute_runnable_slots_counts_only_available_capacity() -> None:
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        compute_runnable_slots,
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host_a": 20, "host_b": 10, "host_c": 5},
            inflight_count_by_host={"host_a": 1, "host_c": 1},
            max_inflight_per_host=1,
        )
        == 1
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host_a": 20, "host_b": 10, "host_c": 5},
            inflight_count_by_host={},
            max_inflight_per_host=3,
        )
        == 9
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host_a": 20},
            inflight_count_by_host={"host_a": 2},
            max_inflight_per_host=1,
        )
        == 0
    )

    assert (
        compute_runnable_slots(
            ready_pending={},
            inflight_count_by_host={"host_a": 2},
            max_inflight_per_host=3,
        )
        == 0
    )


def test_compute_runnable_slots_partial_capacity() -> None:
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        compute_runnable_slots,
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host": 1},
            inflight_count_by_host={"host": 1},
            max_inflight_per_host=3,
        )
        == 1
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host": 2},
            inflight_count_by_host={"host": 2},
            max_inflight_per_host=3,
        )
        == 1
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host": 2},
            inflight_count_by_host={"host": 1},
            max_inflight_per_host=3,
        )
        == 2
    )

    assert (
        compute_runnable_slots(
            ready_pending={"host": 10},
            inflight_count_by_host={"host": 3},
            max_inflight_per_host=3,
        )
        == 0
    )


def test_compute_runnable_slots_excludes_paced_hosts() -> None:
    from time import monotonic

    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        compute_runnable_slots,
    )
    from crawler.scheduling.queueing.host_eligibility_queue import (
        HostEligibilityQueue,
    )

    eligibility_queue = HostEligibilityQueue()
    now = monotonic()

    # host_a is paced (future eligibility), host_b is due, host_c is blocked.
    eligibility_queue.upsert(
        host="host_a",
        next_eligible_at=now + 10.0,
        inflight=0,
    )
    eligibility_queue.upsert(
        host="host_b",
        next_eligible_at=0.0,
        inflight=0,
    )
    eligibility_queue.upsert(
        host="host_c",
        next_eligible_at=float("inf"),
        inflight=1,
    )

    # Only host_b counts: min(10 pending, 3 free slots).
    assert (
        compute_runnable_slots(
            ready_pending={"host_a": 20, "host_b": 10, "host_c": 5},
            inflight_count_by_host={},
            max_inflight_per_host=3,
            host_eligibility_queue=eligibility_queue,
            now=now,
        )
        == 3
    )


def test_compute_runnable_slots_unknown_host_counts_as_runnable() -> None:
    from time import monotonic

    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        compute_runnable_slots,
    )
    from crawler.scheduling.queueing.host_eligibility_queue import (
        HostEligibilityQueue,
    )

    eligibility_queue = HostEligibilityQueue()
    eligibility_queue.upsert(
        host="paced.example",
        next_eligible_at=monotonic() + 10.0,
        inflight=0,
    )

    # unknown.example has no cached entry yet: the dispatcher will compute it
    # immediately, so it counts as runnable (min(4 pending, 3 free slots)).
    assert (
        compute_runnable_slots(
            ready_pending={"paced.example": 5, "unknown.example": 4},
            inflight_count_by_host={},
            max_inflight_per_host=3,
            host_eligibility_queue=eligibility_queue,
            now=monotonic(),
        )
        == 3
    )


class _FakeWorker:
    def __init__(
        self,
        worker_id: int,
        *,
        busy: bool = False,
        retiring: bool = False,
    ) -> None:
        self.worker_id = worker_id
        self.state = WorkerState(worker_id=worker_id)
        self.state.busy = busy
        self.state.retire_when_idle = retiring

    def retire(self) -> None:
        self.state.retire_when_idle = True


def _pool_with_workers(
    *,
    workers: list[tuple[int, bool]],
    retiring_ids: set[int] | None = None,
) -> tuple[WorkerPool, dict[int, _FakeWorker]]:
    retiring = set() if retiring_ids is None else retiring_ids
    pool = object.__new__(WorkerPool)
    pool.workers = {
        worker_id: _FakeWorker(
            worker_id,
            busy=busy,
            retiring=(worker_id in retiring),
        )
        for worker_id, busy in workers
    }
    return pool, pool.workers


def test_scale_down_selects_idle_worker_before_busy() -> None:
    pool, _workers = _pool_with_workers(
        workers=[(0, True), (1, True), (2, False), (3, False)]
    )
    assert pool._select_worker_for_retirement() == 2


def test_scale_down_falls_back_to_busy_when_no_idle() -> None:
    pool, _workers = _pool_with_workers(workers=[(0, True), (1, True)])
    assert pool._select_worker_for_retirement() == 0


def test_scale_down_skips_workers_already_retiring() -> None:
    pool, _workers = _pool_with_workers(
        workers=[(0, False), (1, False)],
        retiring_ids={0},
    )
    assert pool._select_worker_for_retirement() == 1


def test_scale_up_reuses_busy_retiring_worker() -> None:
    pool, workers = _pool_with_workers(
        workers=[(0, True), (1, False)],
        retiring_ids={0},
    )
    assert pool._select_retiring_worker_for_reuse() == 0

    pool._scale_up_locked(target=2)
    assert workers[0].state.retire_when_idle is False


def test_scale_up_does_not_reuse_idle_retiring_worker() -> None:
    pool, _workers = _pool_with_workers(
        workers=[(0, False), (1, False)],
        retiring_ids={0},
    )
    assert pool._select_retiring_worker_for_reuse() is None
