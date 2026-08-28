"""Crawl-scope rejection checks for scheduler task admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.scheduling.host_control.host_budget_tracker import (
        HostBudgetTracker,
    )


class AdmissionScopeChecker:
    """Reject tasks outside the source scope unless dynamically expanded."""

    def __init__(
        self,
        *,
        url_filter: UrlAdmissionFilter | None,
        host_budget_tracker: HostBudgetTracker | None,
        source_scope_registry: SourceScopeRegistry,
    ) -> None:
        self._url_filter = url_filter
        self._host_budget_tracker = host_budget_tracker
        self._source_scope_registry = source_scope_registry

    def rejection_reason(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> ScheduleDecisionReason | None:
        if task.source_type == "seed":
            return None

        if host is None:
            return ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED

        source_name = task.source_name or ""
        if not self._is_source_authorized(
            source_name=source_name,
            host=host,
            source_type=task.source_type,
        ):
            return ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED

        if task.source_type == "embedded_asset":
            return None

        if not self._restrict_to_seed_hosts_enabled():
            return None

        if self._host_budget_tracker is None:
            return ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED

        if self._host_budget_tracker.is_seed_host(host):
            return None

        if self._host_budget_tracker.is_expanded_host(source_name, host):
            return None

        return ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED

    def _restrict_to_seed_hosts_enabled(self) -> bool:
        if self._url_filter is None:
            return True
        return self._url_filter.restrict_to_seed_hosts_enabled

    def _is_source_authorized(
        self,
        *,
        source_name: str,
        host: str,
        source_type: str,
    ) -> bool:
        try:
            scope = self._source_scope_registry.require(source_name)
        except ValueError:
            return False

        if source_type == "embedded_asset":
            return scope.allows_page_host(host) or scope.allows_asset_host(
                host,
            )

        return scope.allows_page_host(host)
