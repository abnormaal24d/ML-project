"""Coordinates the admission process for crawl tasks into the scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.scheduling.admission.admission_context import AdmissionContext
from crawler.scheduling.admission.admission_host_limit_resolver import (
    AdmissionHostLimitResolver,
)
from crawler.scheduling.admission.admission_prerequisite_checker import (
    AdmissionPrerequisiteChecker,
)
from crawler.scheduling.admission.admission_pressure_rules import (
    AdmissionPressureRules,
)
from crawler.scheduling.admission.admission_scope_checker import (
    AdmissionScopeChecker,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
)

if TYPE_CHECKING:
    from typing import AbstractSet

    from config.collection.discovery import SchedulingSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )

    from ..host_control.host_budget_tracker import HostBudgetTracker
    from ..host_control.host_media_byte_budget import HostMediaByteBudget


class SchedulerTaskAdmitter:
    """Coordinate crawl-task admission into the scheduler frontier."""

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        seen_urls: SeenUrlRegistry,
        host_advice_tracker: HostAdviceTracker,
        url_filter: UrlAdmissionFilter | None = None,
        blacklist_repository: BlacklistRepository | None = None,
        host_budget_tracker: HostBudgetTracker | None = None,
        host_media_byte_budget: HostMediaByteBudget | None = None,
        run_url_feedback: RunUrlFeedback | None = None,
        source_scope_registry: SourceScopeRegistry,
        metrics: CollectionMetrics | None = None,
    ) -> None:
        self._settings = settings
        self._seen_urls = seen_urls
        self._host_advice_tracker = host_advice_tracker
        self._url_filter = url_filter
        self._blacklist_repository = blacklist_repository
        self._host_budget_tracker = host_budget_tracker
        self._host_media_byte_budget = host_media_byte_budget
        self._run_url_feedback = run_url_feedback
        self._metrics = metrics

        self._prerequisite_checker = AdmissionPrerequisiteChecker(
            settings=settings,
            url_filter=url_filter,
            blacklist_repository=blacklist_repository,
            seen_urls=seen_urls,
            run_url_feedback=run_url_feedback,
        )
        self._scope_checker = AdmissionScopeChecker(
            url_filter=url_filter,
            host_budget_tracker=host_budget_tracker,
            source_scope_registry=source_scope_registry,
        )
        self._pressure_rules = AdmissionPressureRules(
            settings=settings,
            host_advice_tracker=host_advice_tracker,
        )
        self._host_limit_resolver = AdmissionHostLimitResolver(
            settings=settings,
            host_advice_tracker=host_advice_tracker,
            host_budget_tracker=host_budget_tracker,
            pressure_rules=self._pressure_rules,
        )

    def evaluate(
        self,
        *,
        ctx: AdmissionContext,
        seen_identity_keys: AbstractSet[str] | None = None,
        use_host_advice: bool = True,
        record_metrics: bool = True,
    ) -> ScheduleDecision:
        """Evaluate if a task should be admitted based on system and host state."""

        prerequisite_decision = self._prerequisite_checker.evaluate(
            task=ctx.task,
            closed=ctx.closed,
            seen_identity_keys=seen_identity_keys,
        )
        if prerequisite_decision is not None:
            if (
                prerequisite_decision.reason
                == ScheduleDecisionReason.BLACKLISTED
                and record_metrics
            ):
                metrics = self._metrics
                if metrics is not None:
                    metrics.record_blacklist_block(
                        url=ctx.task.url,
                        host=ctx.host,
                        stage="scheduler",
                        reason="blacklisted",
                    )
            return prerequisite_decision

        scope_rejection_reason = self._scope_checker.rejection_reason(
            task=ctx.task,
            host=ctx.host,
        )
        if scope_rejection_reason is not None:
            return ScheduleDecision.reject(
                reason=scope_rejection_reason,
                normalized_url=ctx.task.url,
            )

        media_budget = self._host_media_byte_budget
        if ctx.host is not None and media_budget is not None:
            exhausted, _ = media_budget.host_budget_exhausted(
                task=ctx.task,
                host=ctx.host,
            )
            if exhausted:
                return ScheduleDecision.reject(
                    reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                    normalized_url=ctx.task.url,
                )

        if (
            ctx.queue_size >= self._settings.queue_critical_watermark
            and ctx.task.source_type != "seed"
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                normalized_url=ctx.task.url,
            )

        if (
            ctx.task.kind is MediaKind.FEED
            and ctx.task.source_type != "seed"
            and ctx.task.depth > self._settings.max_feed_depth
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
                normalized_url=ctx.task.url,
            )

        max_feeds_per_host = self._settings.max_feeds_per_host
        if (
            ctx.task.kind is MediaKind.FEED
            and ctx.task.source_type != "seed"
            and ctx.kind_host_pending >= max_feeds_per_host > 0
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
                normalized_url=ctx.task.url,
            )

        pressure_rejection_reason = self._pressure_rules.rejection_reason(
            task=ctx.task,
            queue_size=ctx.queue_size,
        )
        if pressure_rejection_reason is not None:
            return ScheduleDecision.reject(
                reason=pressure_rejection_reason,
                normalized_url=ctx.task.url,
            )

        hostility_rejection_reason = (
            self._pressure_rules.hostility_rejection_reason(
                task=ctx.task,
                host=ctx.host,
                use_host_advice=use_host_advice,
            )
        )
        if hostility_rejection_reason is not None:
            return ScheduleDecision.reject(
                reason=hostility_rejection_reason,
                normalized_url=ctx.task.url,
            )

        effective_max_pending_per_host = (
            self._host_limit_resolver.effective_max_pending_per_host(
                task=ctx.task,
                host=ctx.host,
                queue_size=ctx.queue_size,
                use_host_advice=use_host_advice,
            )
        )
        pending_backlog = self._host_limit_resolver.pending_for_limit(ctx=ctx)
        if (
            effective_max_pending_per_host is not None
            and pending_backlog >= effective_max_pending_per_host
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
                normalized_url=ctx.task.url,
            )

        crawl_budget_rejection_reason = (
            self._host_limit_resolver.crawl_budget_rejection_reason(
                task=ctx.task,
                host=ctx.host,
            )
        )
        if crawl_budget_rejection_reason is not None:
            return ScheduleDecision.reject(
                reason=crawl_budget_rejection_reason,
                normalized_url=ctx.task.url,
            )

        return ScheduleDecision.accept(
            normalized_url=ctx.task.url,
            task=ctx.task,
        )

    def host_pending_limit(
        self,
        *,
        kind: MediaKind,
        host: str | None,
        queue_size: int,
    ) -> int | None:
        """Return the scheduler-owned pending limit for one host and kind.

        This is the read-only view used by page discovery to reserve frontier
        capacity before final selection. It resolves the same pressure, kind,
        and host-advice policy that admission enforces, without duplicating it.
        """

        return self._host_limit_resolver.effective_max_pending_for_kind(
            kind=kind,
            host=host,
            queue_size=queue_size,
        )

    def scope_rejection_reason(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> ScheduleDecisionReason | None:
        """Return the read-only crawl-scope verdict for one task.

        This is the discovery preflight view backed by the same
        AdmissionScopeChecker that final admission uses. The scope checker
        remains the single policy owner; this call mutates nothing.
        """

        return self._scope_checker.rejection_reason(
            task=task,
            host=host,
        )

    def queue_pressure_state(self, *, queue_size: int) -> str:
        """Return the current scheduler queue-pressure state."""

        return self._pressure_rules.queue_pressure_state(queue_size=queue_size)
