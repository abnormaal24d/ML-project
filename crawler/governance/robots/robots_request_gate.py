"""Per-request robots.txt access enforcement for the fetch pipeline.

The gate sits between redirect validation and the rate limiter inside the
request runner: every URL hop (initial URL and each redirect target) must
pass the robots decision before a request may be issued.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Protocol
from urllib.parse import urlsplit

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.governance.robots.robots_check_result import (
    RobotsAccessAction,
    RobotsCheckResult,
    RobotsDecision,
)
from crawler.scheduling.host_control.host_advice import HostAdvice

if TYPE_CHECKING:
    from config.collection.governance import RobotsSettings
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.rate_limit.rate_limiter import RateLimiter
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from logger.project_logger import ProjectLogger


class SchedulerAdviceRegistrar(Protocol):
    """Accept host feedback with the scheduler's keyword-only API."""

    def __call__(
        self,
        *,
        url: str,
        advice: HostAdvice,
    ) -> Awaitable[None]: ...


class RobotsCheckerProtocol(Protocol):
    """The subset of RobotsChecker the gate depends on."""

    async def check(self, url: str) -> RobotsCheckResult: ...

    def host_rules_snapshot(self, *, url: str) -> Any: ...


class RobotsRequestGate:
    """Enforce robots decisions for single fetch requests."""

    def __init__(
        self,
        *,
        checker: RobotsCheckerProtocol,
        settings: RobotsSettings,
        rate_limiter: RateLimiter,
        host_normalizer: HostNormalizer,
        metrics: CollectionMetrics | None = None,
        logger: ProjectLogger,
    ) -> None:
        self._checker = checker
        self._settings = settings
        self._rate_limiter = rate_limiter
        self._host_normalizer = host_normalizer
        self._metrics = metrics
        self._logger = logger
        self._advice_registrar: SchedulerAdviceRegistrar | None = None

    def set_scheduler_advice_registrar(
        self,
        registrar: SchedulerAdviceRegistrar,
    ) -> None:
        """Bind the scheduler advice registrar after composition completes."""

        self._advice_registrar = registrar

    async def authorize(
        self,
        *,
        url: str,
    ) -> None:
        """Raise when robots rules prevent fetching ``url`` right now."""

        if self._settings.mode == "disabled":
            return

        result = await self._checker.check(url)

        await self._register_host_advice(url=url, result=result)
        await self._apply_crawl_delay(url=url, result=result)

        action = result.to_access_action(
            on_weak_unknown=self._settings.on_weak_unknown,
            on_transient_unknown=self._settings.on_transient_unknown,
            on_hostile_unknown=self._settings.on_hostile_unknown,
        )

        if (
            self._settings.mode == "observe"
            and action == RobotsAccessAction.BLOCK
        ):
            self._record_blacklist(
                url=url,
                reason="robots_would_block",
            )
            return

        if action == RobotsAccessAction.BLOCK:
            self._record_blacklist(
                url=url,
                reason="robots_disallowed",
            )
            raise IgnoredFetchError(
                reason="robots_disallowed",
                status_code=result.http_status,
                final_url=url,
                metrics_recorded=True,
            )

        if action == RobotsAccessAction.DEFER:
            raise RetryableFetchError(
                f"robots.txt decision deferred for {url}: {result.reason}",
                retry_class="robots",
                retry_error_kind=result.reason,
                status_code=result.http_status,
                retry_after_seconds=result.retry_after_seconds,
            )

    async def _apply_crawl_delay(
        self,
        *,
        url: str,
        result: RobotsCheckResult,
    ) -> None:
        if result.decision != RobotsDecision.ALLOWED:
            return
        if result.crawl_delay_seconds is None:
            return
        host = self._host_from_url(url)
        await self._rate_limiter.set_host_crawl_delay(
            host=host,
            crawl_delay_seconds=result.crawl_delay_seconds,
        )

    async def _register_host_advice(
        self,
        *,
        url: str,
        result: RobotsCheckResult,
    ) -> None:
        registrar = self._advice_registrar
        if registrar is None:
            return
        snapshot = self._checker.host_rules_snapshot(url=url)
        await registrar(
            url=url,
            advice=HostAdvice(
                discovery_factor=snapshot.discovery_factor,
                priority_penalty=snapshot.priority_penalty,
                hostility_score=snapshot.hostility_score,
                crawl_delay_seconds=result.crawl_delay_seconds,
            ),
        )

    def _record_blacklist(
        self,
        *,
        url: str,
        reason: str,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.record_blacklist_block(
            url=url,
            host=self._host_from_url(url),
            stage="robots",
            reason=reason,
        )

    def _host_from_url(self, url: str) -> str | None:
        try:
            hostname = urlsplit(url).hostname
        except ValueError:
            return None
        if not hostname:
            return None
        return self._host_normalizer.normalize(hostname)
