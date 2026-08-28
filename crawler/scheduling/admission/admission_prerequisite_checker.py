"""Prerequisite rejection checks for scheduler task admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.scheduling.admission.admission_task_identity import (
    max_depth_for_task,
    scheduler_task_identity_key,
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
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry


class AdmissionPrerequisiteChecker:
    """Reject tasks that fail basic admission prerequisites."""

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        url_filter: UrlAdmissionFilter | None,
        blacklist_repository: BlacklistRepository | None,
        seen_urls: SeenUrlRegistry,
        run_url_feedback: RunUrlFeedback | None,
    ) -> None:
        self._settings = settings
        self._url_filter = url_filter
        self._blacklist_repository = blacklist_repository
        self._seen_urls = seen_urls
        self._run_url_feedback = run_url_feedback

    def evaluate(
        self,
        *,
        task: CrawlTask,
        closed: bool,
        seen_identity_keys: AbstractSet[str] | None = None,
    ) -> ScheduleDecision | None:
        if closed:
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.SCHEDULER_CLOSED,
                normalized_url=task.url,
            )

        max_allowed_depth = max_depth_for_task(
            task=task,
            settings=self._settings,
        )
        if task.depth > max_allowed_depth:
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.MAX_DEPTH_EXCEEDED,
                normalized_url=task.url,
            )

        if self._url_filter is not None:
            if not self._url_filter.evaluate_task(task).allowed:
                return ScheduleDecision.reject(
                    reason=ScheduleDecisionReason.URL_FILTERED,
                    normalized_url=task.url,
                )

        if self._blacklist_repository is not None:
            if self._blacklist_repository.contains(url=task.url):
                return ScheduleDecision.reject(
                    reason=ScheduleDecisionReason.BLACKLISTED,
                    normalized_url=task.url,
                )

        if (
            self._run_url_feedback is not None
            and self._run_url_feedback.was_not_modified(
                task=task,
                url=task.url,
            )
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.NOT_MODIFIED_THIS_RUN,
                normalized_url=task.url,
            )

        if (
            self._run_url_feedback is not None
            and self._run_url_feedback.is_forbidden_endpoint(
                url=task.url,
            )
        ):
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.FORBIDDEN_ENDPOINT_THIS_RUN,
                normalized_url=task.url,
            )

        task_identity_key = scheduler_task_identity_key(task=task)
        identity_seen = (
            self._seen_urls.is_seen(task_identity_key)
            if seen_identity_keys is None
            else task_identity_key in seen_identity_keys
        )
        if identity_seen:
            return ScheduleDecision.reject(
                reason=ScheduleDecisionReason.DUPLICATE_URL,
                normalized_url=task.url,
            )

        return None
