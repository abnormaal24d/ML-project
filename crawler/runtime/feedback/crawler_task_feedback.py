"""Route completed crawler task feedback to runtime collaborators."""

from __future__ import annotations

from typing import Any

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.runtime.feedback.crawler_host_feedback import CrawlerHostFeedback
from crawler.scheduling.host_control.host_budget_tracker import (
    HostBudgetTracker,
)
from crawler.scheduling.host_control.host_media_byte_budget import (
    HostMediaByteBudget,
)
from crawler.scheduling.url_scheduler import UrlScheduler


class CrawlerTaskFeedback:
    """
    Route completed task feedback to scheduler, host, and budget collaborators.
    """

    def __init__(
        self,
        *,
        scheduler: UrlScheduler,
        host_feedback: CrawlerHostFeedback,
        host_budget_tracker: HostBudgetTracker,
        host_media_byte_budget: HostMediaByteBudget,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._scheduler = scheduler
        self._host_feedback = host_feedback
        self._host_budget_tracker = host_budget_tracker
        self._host_media_byte_budget = host_media_byte_budget
        self._host_normalizer = host_normalizer

    async def on_task_processed(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None = None,
        result: Any | None = None,
    ) -> None:
        payload = dict(fields or {})
        final_url = self._extract_final_url(
            task=task,
            fields=payload,
            result=result,
        )
        await self._register_final_url_redirect(
            task=task,
            requested_url=task.url,
            final_url=final_url,
        )
        await self._host_feedback.register(
            url=final_url or task.url,
            advice=self._extract_host_advice(fields=payload, result=result),
        )
        if self._host_budget_tracker is not None:
            self._host_budget_tracker.register_task_outcome(
                task=task,
                outcome=outcome,
                fields=payload,
            )
        if self._host_media_byte_budget is not None:
            byte_count = payload.get("bytes")
            if isinstance(byte_count, (int, float)) and int(byte_count) > 0:
                host = self._host_normalizer.normalize(
                    getattr(task, "host", None) or _host_from_url(task.url)
                )
                if host is not None:
                    self._host_media_byte_budget.record_download(
                        host=host,
                        kind=task.kind,
                        byte_count=int(byte_count),
                    )

    @staticmethod
    def _extract_final_url(
        *,
        task: CrawlTask,
        fields: dict[str, object],
        result: Any | None,
    ) -> str | None:
        for candidate in (
            getattr(result, "final_url", None),
            getattr(result, "response_url", None),
            getattr(result, "url", None),
            fields.get("final_url"),
            fields.get("response_url"),
            fields.get("resolved_url"),
        ):
            if not isinstance(candidate, str):
                continue
            value = candidate.strip()
            if value and value != task.url:
                return value
        return None

    @staticmethod
    def _extract_host_advice(
        *,
        fields: dict[str, object],
        result: Any | None,
    ) -> Any | None:
        for candidate in (
            getattr(result, "host_rules_advice", None),
            getattr(result, "robots_host_rules_advice", None),
            fields.get("host_rules_advice"),
            fields.get("robots_host_rules_advice"),
            fields.get("robots_advice"),
        ):
            if candidate is not None:
                return candidate
        return None

    async def _register_final_url_redirect(
        self,
        *,
        task: CrawlTask,
        requested_url: str,
        final_url: str | None,
    ) -> None:
        if not isinstance(final_url, str):
            return
        normalized_final_url = final_url.strip()
        if not normalized_final_url:
            return
        if normalized_final_url == requested_url.strip():
            return
        await self._scheduler.register_final_url(
            task=task,
            requested_url=requested_url,
            final_url=normalized_final_url,
        )


def _host_from_url(url: str) -> str | None:
    from urllib.parse import urlparse

    return urlparse(url).hostname
