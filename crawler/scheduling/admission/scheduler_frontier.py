"""Evaluate admission and mutate ready/delayed frontier queues for the URL scheduler."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.runtime.concurrency import condition_notify_all
from crawler.scheduling.admission.admission_context import AdmissionContext
from crawler.scheduling.admission.admission_pressure_rules import (
    is_coverage_recovery_target_task,
)
from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
    ScopeEligibilityDecision,
)
from crawler.scheduling.checkpointing.scheduler_task_envelope import (
    SchedulerTaskEnvelope,
)
from crawler.scheduling.reason_key import reason_key
from logger.project_logger import ProjectLogger
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping
    from typing import AbstractSet

    from crawler.classification.media_kind import MediaKind
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.priority.crawl_task_priority_calculator import (
        CrawlTaskPriorityCalculator,
    )

    from ..progress.scheduler_backlog_reader import SchedulerBacklogReader
    from ..progress.scheduler_progress_state import SchedulerProgressState
    from ..queueing.delayed_task_queue import DelayedTaskQueue
    from ..queueing.host_task_queue import HostTaskQueue
    from .scheduler_task_admitter import SchedulerTaskAdmitter

CanonicalHostFromUrl = Callable[[str], str | None]
CanonicalHost = Callable[[str | None], str | None]
IsClosed = Callable[[], bool]
AllocateSequence = Callable[[], int]


class SchedulerFrontier:
    """Evaluate admission and mutate ready/delayed frontier queues."""

    MAX_REJECTION_SAMPLES = 12

    def __init__(
        self,
        *,
        id_generator: IdGenerator,
        url_normalizer: UrlNormalizer,
        priority_resolver: CrawlTaskPriorityCalculator,
        task_admitter: SchedulerTaskAdmitter,
        progress_state: SchedulerProgressState,
        backlog_reader: SchedulerBacklogReader,
        seen_urls: SeenUrlRegistry,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        condition: asyncio.Condition,
        canonical_host_from_url: CanonicalHostFromUrl,
        is_closed: IsClosed,
        allocate_sequence: AllocateSequence,
        logger: ProjectLogger,
        max_rejection_samples: int = MAX_REJECTION_SAMPLES,
    ) -> None:
        if id_generator is None:
            raise ValueError("id_generator is required")

        self._id_generator = id_generator
        self._url_normalizer = url_normalizer
        self._priority_resolver = priority_resolver
        self._task_admitter = task_admitter
        self._progress_state = progress_state
        self._backlog_reader = backlog_reader
        self._seen_urls = seen_urls
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._condition = condition
        self._canonical_host_from_url = canonical_host_from_url
        self._is_closed = is_closed
        self._allocate_sequence = allocate_sequence
        self._logger = logger
        self._max_rejection_samples = max(
            0,
            int(max_rejection_samples),
        )

    def enqueue_locked(
        self,
        *,
        task: CrawlTask,
    ) -> tuple[ScheduleDecision, CrawlTask | None, int | None]:
        task_with_identity = task.ensure_id(
            id_generator=self._id_generator,
        )
        normalized_url = self._url_normalizer.normalize(task_with_identity.url)

        if not normalized_url:
            decision = ScheduleDecision.reject(
                normalized_url=task_with_identity.url,
                reason=ScheduleDecisionReason.INVALID_URL,
            )
            self._progress_state.record_rejected_decision(decision=decision)
            return decision, None, None

        canonical_task = task_with_identity.clone(url=normalized_url)
        canonical_host = self._canonical_host_from_url(canonical_task.url)
        decision = self.evaluate_enqueue_decision(
            task=canonical_task,
            host=canonical_host,
        )
        if not decision.accepted:
            self._progress_state.record_rejected_decision(decision=decision)
            return decision, None, None

        drain_reservation = self._reserve_high_pressure_drain_slot(
            task=canonical_task,
            queue_size=self.queue_size_total(),
        )
        if drain_reservation is None:
            decision = ScheduleDecision.reject(
                reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                normalized_url=canonical_task.url,
            )
            self._progress_state.record_rejected_decision(decision=decision)
            return decision, None, None

        scheduled_task = CrawlTask.prepare_for_enqueue(
            task=canonical_task,
            priority_resolver=self._priority_resolver,
        )
        sequence = self._allocate_sequence()

        try:
            self.record_accepted_task_locked(
                task=scheduled_task,
                host=canonical_host,
                sequence=sequence,
                count_acceptance=True,
            )
        finally:
            if drain_reservation > 0:
                self._progress_state.release_drain_budget_reservation(
                    reserved=drain_reservation,
                )

        return (
            ScheduleDecision.accept(
                normalized_url=scheduled_task.url,
                task=scheduled_task,
            ),
            scheduled_task,
            sequence,
        )

    def enqueue_many_locked(
        self,
        *,
        tasks: tuple[CrawlTask, ...] | list[CrawlTask],
    ) -> tuple[ScheduleDecision, ...]:
        if not tasks:
            return ()

        decisions: list[ScheduleDecision] = []
        rejection_reasons: Counter[str] = Counter()
        rejection_samples: list[dict[str, object]] = []
        accepted_count = 0
        filtered_count = 0
        rejected_count = 0

        for task in tasks:
            decision, _, _ = self.enqueue_locked(task=task)
            decisions.append(decision)

            if decision.accepted:
                accepted_count += 1
                continue

            serialized_reason = reason_key(decision.reason)
            rejection_reasons[serialized_reason] += 1

            if decision.reason == ScheduleDecisionReason.URL_FILTERED:
                filtered_count += 1
            else:
                rejected_count += 1

            if len(rejection_samples) < self._max_rejection_samples:
                rejection_samples.append(
                    {
                        "task_id": task.task_id,
                        "url": decision.normalized_url or task.url,
                        "reason": serialized_reason,
                    }
                )

        if accepted_count > 0:
            condition_notify_all(self._condition)

        self._logger.debug(
            "task_batch_scheduled",
            requested=len(tasks),
            accepted=accepted_count,
            filtered=filtered_count,
            rejected=rejected_count,
            rejection_reasons=dict(sorted(rejection_reasons.items())),
            rejection_samples=rejection_samples,
        )

        return tuple(decisions)

    def discovery_scope_decisions_locked(
        self,
        *,
        tasks: tuple[CrawlTask, ...] | list[CrawlTask],
    ) -> tuple[ScopeEligibilityDecision, ...]:
        """Return read-only crawl-scope verdicts for discovery candidates.

        This is a selection-time optimization for page discovery. It uses the
        same normalizer and the same AdmissionScopeChecker-backed admitter view
        as final admission, but it never mutates frontier state. Final
        admission re-checks scope and remains the authority.
        """

        decisions: list[ScopeEligibilityDecision] = []
        for task in tasks:
            normalized_url = self._url_normalizer.normalize(task.url)
            if not normalized_url:
                continue
            canonical_task = task.clone(url=normalized_url)
            canonical_host = self._canonical_host_from_url(canonical_task.url)
            rejection_reason = self._task_admitter.scope_rejection_reason(
                task=canonical_task,
                host=canonical_host,
            )
            decisions.append(
                ScopeEligibilityDecision(
                    normalized_url=canonical_task.url,
                    allowed=rejection_reason is None,
                )
            )
        return tuple(decisions)

    def prepare_restored_envelope(
        self,
        *,
        envelope: SchedulerTaskEnvelope,
        canonical_host: CanonicalHost,
        queue_size: int,
        ready_pending_by_host: Mapping[str | None, int],
        ready_kind_pending_by_host: Mapping[tuple[str | None, MediaKind], int],
        seen_identity_keys: AbstractSet[str],
        use_host_advice: bool,
    ) -> SchedulerTaskEnvelope | None:
        """Validate and normalize one restore item without live mutation."""

        restored_host = canonical_host(envelope.host)
        normalized_url = self._url_normalizer.normalize(envelope.task.url)
        if not normalized_url:
            return None

        restored_task = CrawlTask.with_url_and_preserved_priority(
            task=envelope.task,
            url=normalized_url,
            priority=envelope.priority,
        )
        url_host = self._canonical_host_from_url(restored_task.url)
        if envelope.host is not None and restored_host != url_host:
            return None
        restored_host = url_host

        decision = self._task_admitter.evaluate(
            ctx=AdmissionContext(
                task=restored_task,
                host=restored_host,
                source=restored_task.source_type,
                now=monotonic(),
                queue_size=queue_size,
                host_pending=ready_pending_by_host.get(restored_host, 0),
                kind_host_pending=ready_kind_pending_by_host.get(
                    (restored_host, restored_task.kind),
                    0,
                ),
                closed=self._is_closed(),
            ),
            seen_identity_keys=seen_identity_keys,
            use_host_advice=use_host_advice,
            record_metrics=False,
        )
        if not decision.accepted:
            return None

        return SchedulerTaskEnvelope(
            task=restored_task,
            host=restored_host,
            priority=envelope.priority,
            sequence=envelope.sequence,
        )

    def commit_restored_envelope(
        self,
        *,
        envelope: SchedulerTaskEnvelope,
        delayed_wait_seconds: float | None,
    ) -> None:
        """Insert a prevalidated restore item into the live frontier."""

        self.record_accepted_task_locked(
            task=envelope.task,
            host=envelope.host,
            sequence=envelope.sequence,
            delayed_wait_seconds=delayed_wait_seconds,
            count_acceptance=False,
            notify=False,
        )

    def evaluate_enqueue_decision(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> ScheduleDecision:
        return self._task_admitter.evaluate(
            ctx=AdmissionContext(
                task=task,
                host=host,
                source=task.source_type,
                now=monotonic(),
                queue_size=self.queue_size_total(),
                host_pending=self._backlog_reader.host_pending(host=host),
                kind_host_pending=(
                    self._backlog_reader.kind_host_pending_if_needed(
                        task=task,
                        host=host,
                    )
                ),
                closed=self._is_closed(),
            ),
        )

    def record_accepted_task_locked(
        self,
        *,
        task: CrawlTask,
        host: str | None,
        sequence: int,
        delayed_wait_seconds: float | None = None,
        count_acceptance: bool = True,
        notify: bool = True,
    ) -> None:
        if count_acceptance:
            self._progress_state.record_accepted_task()

        self._seen_urls.remember(scheduler_task_identity_key(task=task))

        if delayed_wait_seconds is not None and delayed_wait_seconds > 0:
            delayed = self._delayed_queue.push(
                host=host,
                priority=task.priority,
                sequence=sequence,
                task=task,
                wait_seconds=delayed_wait_seconds,
            )
            if delayed:
                if notify:
                    condition_notify_all(self._condition)
                return

        self._host_queue.push(
            host=host,
            priority=task.priority,
            sequence=sequence,
            task=task,
        )
        if notify:
            condition_notify_all(self._condition)

    def queue_size_total(self) -> int:
        return self._host_queue.queue_size + self._delayed_queue.queue_size

    def _reserve_high_pressure_drain_slot(
        self,
        *,
        task: CrawlTask,
        queue_size: int,
    ) -> int | None:
        pressure_state = self._task_admitter.queue_pressure_state(
            queue_size=queue_size,
        )

        if pressure_state == "high" and task.source_type != "seed":
            if is_coverage_recovery_target_task(task):
                return 0

            reserved = self._progress_state.reserve_drain_budget(
                configured_cap=1,
            )
            if reserved <= 0:
                return None
            return reserved

        if pressure_state != "high":
            self._progress_state.reset_drain_budget_window()

        return 0
