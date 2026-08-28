"""URL scheduler coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.exceptions.crawler_error import CrawlerRuntimeError
from crawler.runtime.concurrency import condition_notify_all
from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
    scheduler_task_identity_key_for_url,
)
from crawler.scheduling.reason_key import reason_key
from logger.project_logger import ProjectLogger


class SchedulerClosedError(CrawlerRuntimeError):
    """Raised when work is requested from a closed scheduler."""


if TYPE_CHECKING:
    import asyncio

    from crawler.classification.media_kind import MediaKind
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.host_suppression import HostSuppressionStore
    from crawler.scheduling.completion.task_completion_handler import (
        SchedulerCompletionHandler,
    )
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.dispatch.host_dispatch_wait_reader import (
        HostDispatchWaitReader,
    )
    from crawler.scheduling.dispatch.scheduler_dispatcher import (
        SchedulerDispatcher,
    )
    from crawler.scheduling.host_control.host_advice import HostAdvice
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )
    from crawler.scheduling.progress.active_task_registry import (
        ActiveTaskRegistry,
    )
    from crawler.scheduling.progress.scheduler_backlog_reader import (
        SchedulerBacklogReader,
    )
    from crawler.scheduling.progress.scheduler_progress_state import (
        SchedulerProgressState,
    )
    from crawler.scheduling.progress.scheduler_runtime_state import (
        SchedulerRuntimeState,
    )
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        DiscoveryCapacitySnapshot as SchedulerSnapshotDiscoveryCapacity,
    )
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        SchedulerSnapshot,
        SchedulerSnapshotReader,
    )

    from .admission.schedule_decision import (
        ScheduleDecision,
        ScopeEligibilityDecision,
    )
    from .admission.scheduler_frontier import SchedulerFrontier
    from .checkpointing.scheduler_state_exporter import SchedulerStateExporter
    from .checkpointing.scheduler_state_restorer import SchedulerStateRestorer
    from .checkpointing.scheduler_task_deserializer import (
        SchedulerTaskDeserializer,
    )
    from .completion.scheduler_retry_budget import SchedulerRetryBudget
    from .queueing.delayed_task_queue import DelayedTaskQueue
    from .queueing.host_eligibility_queue import HostEligibilityQueue
    from .queueing.host_task_queue import HostTaskQueue


class UrlScheduler:
    """Coordinate ready, delayed, active and persisted scheduler state."""

    MAX_REJECTION_SAMPLES = 12

    def __init__(
        self,
        *,
        url_normalizer: UrlNormalizer,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
        state: SchedulerRuntimeState,
        condition: asyncio.Condition,
        seen_urls: SeenUrlRegistry,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        host_eligibility_queue: HostEligibilityQueue,
        active_registry: ActiveTaskRegistry,
        host_advice_tracker: HostAdviceTracker,
        dispatch_wait_reader: HostDispatchWaitReader,
        snapshot_reader: SchedulerSnapshotReader,
        completion_handler: SchedulerCompletionHandler,
        progress_state: SchedulerProgressState,
        state_exporter: SchedulerStateExporter,
        state_restorer: SchedulerStateRestorer,
        checkpoint_task_deserializer: SchedulerTaskDeserializer,
        backlog_reader: SchedulerBacklogReader,
        retry_rules: SchedulerRetryBudget,
        frontier_service: SchedulerFrontier,
        dispatch_service: SchedulerDispatcher,
        logger: ProjectLogger,
    ) -> None:
        self._state = state
        self._condition = condition
        self._dispatch_wait_reader = dispatch_wait_reader
        self._dispatch_service = dispatch_service
        self._checkpoint_task_deserializer = checkpoint_task_deserializer
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._host_eligibility_queue = host_eligibility_queue
        self._active_registry = active_registry
        self._host_advice_tracker = host_advice_tracker
        self._snapshot_reader = snapshot_reader
        self._completion_handler = completion_handler
        self._frontier_service = frontier_service
        self._logger = logger
        self._seen_urls = seen_urls
        self._url_normalizer = url_normalizer
        self._host_extractor = host_extractor
        self._host_normalizer = host_normalizer
        self._progress_state = progress_state
        self._state_exporter = state_exporter
        self._state_restorer = state_restorer
        self._backlog_reader = backlog_reader
        self._retry_rules = retry_rules

    def create_checkpoint_task_deserializer(self) -> SchedulerTaskDeserializer:
        return self._checkpoint_task_deserializer

    def is_closed(self) -> bool:
        return self._state.is_closed()

    async def enqueue(self, task: CrawlTask) -> ScheduleDecision:
        async with self._condition:
            self._host_advice_tracker.prune()
            decision, scheduled_task, sequence = (
                self._frontier_service.enqueue_locked(task=task)
            )
            if decision.accepted:
                condition_notify_all(self._condition)

        if not decision.accepted:
            self._logger.debug(
                "task_rejected",
                task_id=task.task_id,
                url=decision.normalized_url,
                reason=reason_key(decision.reason),
            )
            return decision

        if scheduled_task is None or sequence is None:
            raise CrawlerRuntimeError(
                "accepted schedule decision without scheduled task"
            )

        self._logger.debug(
            "task_scheduled",
            task_id=scheduled_task.task_id,
            url=scheduled_task.url,
            kind=scheduled_task.kind,
            depth=scheduled_task.depth,
            source=scheduled_task.source_type,
            priority=scheduled_task.priority,
            parent=scheduled_task.parent_url,
            sequence=sequence,
        )
        return decision

    async def enqueue_many(
        self,
        tasks: tuple[CrawlTask, ...] | list[CrawlTask],
    ) -> tuple[ScheduleDecision, ...]:
        async with self._condition:
            return self._frontier_service.enqueue_many_locked(
                tasks=tasks,
            )

    def already_seen_for_discovery(self, identity_key: str) -> bool:
        return self._seen_urls.is_seen(identity_key)

    async def register_host_rules_advice(
        self,
        *,
        url: str,
        advice: HostAdvice,
    ) -> None:
        canonical_host = self._host_normalizer.normalize(
            self._host_extractor.extract(url)
        )
        if canonical_host is None:
            return

        async with self._condition:
            self._host_advice_tracker.remember(
                host=canonical_host,
                advice=advice,
            )
            condition_notify_all(self._condition)

        self._logger.debug(
            "scheduler_registered_host_rules_advice",
            url=url,
            host=canonical_host,
            discovery_factor=advice.discovery_factor,
            priority_penalty=advice.priority_penalty,
            hostility_score=advice.hostility_score,
            crawl_delay_seconds=(
                self._host_advice_tracker.extract_crawl_delay_seconds(advice)
            ),
        )

    async def register_final_url(
        self,
        *,
        task: CrawlTask | None = None,
        requested_url: str,
        final_url: str,
    ) -> None:
        if task is None:
            return

        normalized_requested = self._url_normalizer.normalize(requested_url)
        normalized_final = self._url_normalizer.normalize(final_url)

        if not normalized_requested or not normalized_final:
            return

        if normalized_requested == normalized_final:
            return

        async with self._condition:
            self._seen_urls.remember_equivalent_urls(
                scheduler_task_identity_key_for_url(
                    task=task,
                    url=normalized_requested,
                ),
                scheduler_task_identity_key_for_url(
                    task=task,
                    url=normalized_final,
                ),
            )

        self._logger.debug(
            "scheduler_registered_final_url",
            requested_url=normalized_requested,
            final_url=normalized_final,
        )

    def set_host_suppression_reader(
        self,
        reader: HostSuppressionStore | None,
    ) -> None:
        self._dispatch_wait_reader.set_host_suppression_reader(reader)

    async def get(self) -> CrawlTask:
        return await self._dispatch_service.get()

    async def complete(
        self,
        task: CrawlTask,
        *,
        outcome: str = "completed",
        fields: dict[str, object] | None = None,
    ) -> None:
        await self._completion_handler.complete(
            task=task,
            outcome=outcome,
            fields=fields,
        )

    def _is_idle_locked(self) -> bool:
        return (
            self._host_queue.queue_size == 0
            and self._delayed_queue.queue_size == 0
            and self._active_registry.total_tracked_count == 0
        )

    async def join(self) -> None:
        async with self._condition:
            while not self._is_idle_locked():
                await self._condition.wait()

    async def is_idle(self) -> bool:
        async with self._condition:
            return self._is_idle_locked()

    async def snapshot(self) -> SchedulerSnapshot:
        async with self._condition:
            return self._snapshot_reader.snapshot()

    async def discovery_drain_budget(
        self,
        *,
        configured_cap: int,
        force: bool = False,
    ) -> int:
        return await self._snapshot_reader.discovery_drain_budget(
            configured_cap=configured_cap,
            force=force,
        )

    async def discovery_capacity_snapshot(
        self,
    ) -> SchedulerSnapshotDiscoveryCapacity:
        async with self._condition:
            return self._snapshot_reader.discovery_capacity_snapshot()

    async def discovery_scope_decisions(
        self,
        tasks: tuple[CrawlTask, ...] | list[CrawlTask],
    ) -> tuple[ScopeEligibilityDecision, ...]:
        """Return read-only crawl-scope verdicts for discovery candidates.

        Batch preflight for page-discovery selection. Final admission re-checks
        crawl scope under the same policy and remains the authority.
        """

        async with self._condition:
            return self._frontier_service.discovery_scope_decisions_locked(
                tasks=tasks,
            )

    async def queue_size(self) -> int:
        async with self._condition:
            return self._snapshot_reader.queue_size()

    async def pending_host_count(self) -> int:
        async with self._condition:
            return self._snapshot_reader.pending_host_count()

    async def max_pending_per_host(self) -> int:
        async with self._condition:
            return self._snapshot_reader.max_pending_per_host()

    async def inflight_count(self) -> int:
        async with self._condition:
            return self._snapshot_reader.inflight_count()

    async def close(
        self,
        *,
        abort: bool = False,
        discard_pending: bool = False,
    ) -> None:
        """Close the scheduler, optionally dropping work not yet dispatched.

        ``discard_pending`` is the graceful early-success path: queued and
        delayed frontier entries are removed, while active tasks stay tracked
        so their completion callbacks can finish normally. ``abort`` also
        forgets active tasks and is reserved for forced shutdown.
        """

        async with self._condition:
            self._state.closed = True
            if abort or discard_pending:
                self._host_queue.clear()
                self._delayed_queue.clear()
                self._host_eligibility_queue.clear()
            if abort:
                self._active_registry.clear()
            condition_notify_all(self._condition)

    async def export_state(
        self,
        *,
        max_queued_tasks: int,
        include_seen_urls: bool,
    ) -> dict[str, object]:
        async with self._condition:
            self._host_advice_tracker.prune()

            return self._state_exporter.export_state(
                max_queued_tasks=max_queued_tasks,
                include_seen_urls=include_seen_urls,
                host_queue=self._host_queue,
                delayed_queue=self._delayed_queue,
                active_registry=self._active_registry,
                seen_urls=self._seen_urls,
                next_sequence_value=self._state.next_sequence_value,
                total_pending_by_host=(
                    self._backlog_reader.combined_pending_count_by_host()
                ),
                progress_counters=self._progress_state.export_state(),
                retry_budget_state=self._retry_rules.export_state(),
            )

    async def restore_state(
        self,
        *,
        payload: dict[str, object],
        clear_existing: bool = True,
    ) -> int:
        async with self._condition:
            # Eligibility entries are a cache derivable from queued work and
            # authoritative governance state; rebuild them on restore.
            self._host_eligibility_queue.clear()

            retry_budget_payload = payload.get("retry_budget")
            if not isinstance(retry_budget_payload, dict):
                raise ValueError(
                    "scheduler checkpoint missing retry_budget payload"
                )
            retry_budget_state = self._retry_rules.parse_restore_state(
                retry_budget_payload
            )
            current_ready_items = self._host_queue.snapshot_items()
            current_delayed_items = self._delayed_queue.snapshot_items()
            current_active_records = tuple(self._active_registry.values())
            current_dispatching_records = tuple(
                self._active_registry.dispatching_values()
            )
            current_dead_letter_records = tuple(
                self._active_registry.dead_letter_pending_values()
            )
            current_sequences = {
                sequence
                for _host, _priority, sequence, _task in current_ready_items
            }
            current_sequences.update(
                entry.sequence for entry in current_delayed_items
            )
            current_sequences.update(
                record.sequence
                for record in (
                    *current_active_records,
                    *current_dispatching_records,
                    *current_dead_letter_records,
                )
            )
            current_identity_keys = set(self._seen_urls.snapshot_urls())
            current_identity_keys.update(
                scheduler_task_identity_key(task=task)
                for _host, _priority, _sequence, task in current_ready_items
            )
            current_identity_keys.update(
                scheduler_task_identity_key(task=entry.task)
                for entry in current_delayed_items
            )
            current_identity_keys.update(
                scheduler_task_identity_key(task=record.task)
                for record in (
                    *current_active_records,
                    *current_dispatching_records,
                    *current_dead_letter_records,
                )
            )
            current_ready_kind_pending: dict[
                tuple[str | None, MediaKind], int
            ] = {}
            for host, _priority, _sequence, task in current_ready_items:
                key = (host, task.kind)
                current_ready_kind_pending[key] = (
                    current_ready_kind_pending.get(key, 0) + 1
                )

            plan = self._state_restorer.parse_restore_plan(
                payload=payload,
                clear_existing=clear_existing,
                current_next_sequence=self._state.next_sequence_value,
                current_queue_size=(
                    self._host_queue.queue_size
                    + self._delayed_queue.queue_size
                ),
                current_ready_pending_by_host=(
                    self._host_queue.pending_count_by_host()
                ),
                current_ready_kind_pending_by_host=current_ready_kind_pending,
                current_seen_identity_keys=current_identity_keys,
                current_sequences=current_sequences,
                prepare_restored_envelope=lambda **kwargs: (
                    self._frontier_service.prepare_restored_envelope(
                        canonical_host=self._host_normalizer.normalize,
                        **kwargs,
                    )
                ),
                parse_progress_state=(
                    self._progress_state.parse_restore_state
                ),
            )
            result = self._state_restorer.commit_restore_plan(
                plan=plan,
                clear_existing=clear_existing,
                host_queue=self._host_queue,
                delayed_queue=self._delayed_queue,
                active_registry=self._active_registry,
                seen_urls=self._seen_urls,
                host_advice_tracker=self._host_advice_tracker,
                progress_state=self._progress_state,
                commit_restored_envelope=(
                    self._frontier_service.commit_restored_envelope
                ),
            )
            self._state.next_sequence_value = result.next_sequence_value
            self._retry_rules.apply_restore_state(retry_budget_state)
            condition_notify_all(self._condition)

        self._logger.info(
            "scheduler_state_restored",
            queued=result.queued,
            pending_hosts=result.pending_hosts,
            restored_tasks=result.restored_count,
            restored_queued=result.restored_queued_count,
            restored_delayed=result.restored_delayed_count,
            requeued_inflight=result.restored_requeued_inflight_count,
            restored_dispatching=result.restored_dispatching_count,
            skipped_tasks=result.skipped_tasks,
        )
        return result.restored_count
