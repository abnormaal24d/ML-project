"""Describe the result of a scheduler enqueue attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


class ScheduleDecisionReason(StrEnum):
    """Enumerated reasons for scheduler enqueue decisions."""

    ACCEPTED = "accepted"
    SCHEDULER_CLOSED = "scheduler_closed"
    SCHEDULER_UNAVAILABLE = "scheduler_unavailable"
    MAX_DEPTH_EXCEEDED = "max_depth_exceeded"
    URL_FILTERED = "url_filtered"
    CRAWL_SCOPE_BLOCKED = "crawl_scope_blocked"
    INVALID_URL = "invalid_url"
    BLACKLISTED = "blacklisted"
    DUPLICATE_URL = "duplicate_url"
    NOT_MODIFIED_THIS_RUN = "not_modified_this_run"
    FORBIDDEN_ENDPOINT_THIS_RUN = "forbidden_endpoint_this_run"
    SCHEDULER_BACKPRESSURE = "scheduler_backpressure"
    HOSTILITY_BACKPRESSURE = "hostility_backpressure"
    MAX_PENDING_PER_HOST_REACHED = "max_pending_per_host_reached"
    CRAWL_BUDGET_EXHAUSTED = "crawl_budget_exhausted"


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """Describe the result of a scheduler enqueue attempt."""

    accepted: bool
    reason: ScheduleDecisionReason
    normalized_url: str
    task: CrawlTask | None = None

    def __bool__(self) -> bool:
        """Return the acceptance status of the decision."""
        return self.accepted

    @classmethod
    def accept(
        cls,
        *,
        normalized_url: str,
        task: CrawlTask,
    ) -> ScheduleDecision:
        """Create a successful schedule decision for a given task."""
        return cls(
            accepted=True,
            reason=ScheduleDecisionReason.ACCEPTED,
            normalized_url=normalized_url,
            task=task,
        )

    @classmethod
    def reject(
        cls,
        *,
        reason: ScheduleDecisionReason,
        normalized_url: str,
    ) -> ScheduleDecision:
        """Create a rejection schedule decision with a specific reason."""
        return cls(
            accepted=False,
            reason=reason,
            normalized_url=normalized_url,
            task=None,
        )


@dataclass(frozen=True, slots=True)
class ScopeEligibilityDecision:
    """Read-only scope preflight verdict for one discovery candidate.

    This is a selection-time optimization only. The final admission pass
    re-checks crawl scope and remains the authority.
    """

    normalized_url: str
    allowed: bool
