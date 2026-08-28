"""Robots.txt rules checker."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Protocol
from urllib.robotparser import RobotFileParser

from crawler.governance.robots.robots_check_result import (
    RobotsAccessAction,
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)
from crawler.governance.robots.robots_fetch_errors import RobotsLoaderError
from crawler.scheduling.host_control.host_advice import HostAdvice
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.governance import RobotsSettings
    from config.collection.http_rules import TimeoutRulesSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.robots.robots_decision_evaluator import (
        RobotsDecisionEvaluator,
    )
    from crawler.governance.robots.robots_error_resolver import (
        RobotsErrorResolver,
    )
    from crawler.governance.robots.robots_host_rules_store import (
        RobotsHostRulesStore,
    )
    from crawler.governance.robots.robots_parser_cache import RobotsParserCache
    from crawler.governance.robots.robots_unknown_result_suppressor import (
        RobotsUnknownResultSuppressor,
    )
    from crawler.governance.robots.robots_url_resolver import RobotsUrlResolver
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics


class SchedulerAdviceRegistrar(Protocol):
    async def register_host_rules_advice(
        self,
        *,
        url: str,
        advice: HostAdvice,
    ) -> None: ...


class RobotsChecker:
    """Check URLs and crawl tasks against robots.txt rules."""

    def __init__(
        self,
        *,
        settings: "RobotsSettings",
        timeout_rules: "TimeoutRulesSettings",
        decision_evaluator: "RobotsDecisionEvaluator",
        error_rules: "RobotsErrorResolver",
        robots_url_resolver: "RobotsUrlResolver",
        parser_cache: "RobotsParserCache",
        host_rules_store: "RobotsHostRulesStore",
        user_agent: str,
        host_normalizer: "HostNormalizer",
        blacklist_repository: "BlacklistRepository | None" = None,
        duplicate_result_tracker: "RobotsUnknownResultSuppressor",
        logger: ProjectLogger,
        metrics: "CollectionMetrics | None" = None,
    ) -> None:
        self._settings = settings
        self._timeout_rules = timeout_rules
        self._logger = logger
        self._decision_evaluator = decision_evaluator
        self._error_rules = error_rules
        self._robots_url_resolver = robots_url_resolver
        self._parser_cache = parser_cache
        self._blacklist_repository = blacklist_repository
        self._host_rules_store = host_rules_store
        self._duplicate_result_tracker = duplicate_result_tracker
        self._user_agent = user_agent
        self._host_normalizer = host_normalizer
        self._metrics = metrics

        self._on_weak_unknown = settings.on_weak_unknown
        self._on_transient_unknown = settings.on_transient_unknown
        self._on_hostile_unknown = settings.on_hostile_unknown

    async def aclose(self) -> None:
        """Close parser-cache work before its shared transport is closed."""

        await self._parser_cache.aclose()

    def _host_from_url(self, url: str) -> str:
        try:
            from urllib.parse import urlparse

            return (
                self._host_normalizer.normalize(urlparse(url).hostname)
                or "unknown"
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):  # exception-rules: best-effort-cleanup
            return "unknown"

    # ------------------------------------------------------------------
    # Core check API
    # ------------------------------------------------------------------
    async def is_allowed(self, url: str) -> bool:
        """Return whether crawling the URL is effectively allowed."""

        result = await self.check(url)
        return (
            result.to_access_action(
                on_weak_unknown=self._on_weak_unknown,
                on_transient_unknown=self._on_transient_unknown,
                on_hostile_unknown=self._on_hostile_unknown,
            )
            == RobotsAccessAction.ALLOW
        )

    async def check_task(
        self,
        task: CrawlTask,
        *,
        scheduler: SchedulerAdviceRegistrar | None = None,
    ) -> RobotsCheckResult:
        """Return the effective robots decision for a crawl task."""

        result = await self.check(task.url)
        await self._register_scheduler_advice(
            url=task.url,
            scheduler=scheduler,
        )
        return result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------
    async def check(self, url: str) -> RobotsCheckResult:
        """Return the structured robots decision for a URL."""

        if (
            self._blacklist_repository is not None
            and self._blacklist_repository.contains(url=url)
        ):
            result = RobotsCheckResult(
                robots_url="",
                decision=RobotsDecision.DISALLOWED,
                confidence=RobotsConfidence.AUTHORITATIVE_DENY,
                reason="blacklisted",
                source="blacklist",
                is_authoritative=True,
            )
            self._host_rules_store.record(url=url, result=result)
            metrics = self._metrics
            if metrics is not None:
                metrics.record_blacklist_block(
                    url=url,
                    host=self._host_from_url(url),
                    stage="robots",
                    reason="blacklisted",
                )
            self._logger.debug(
                "robots_skipped_blacklisted",
                extra={
                    "url_host": self._host_from_url(url),
                    "reason": "blacklisted",
                },
            )
            return result

        robots_url = self._robots_url_resolver.build(url)

        try:
            parser = await self._parser_cache.get(
                robots_url,
                timeout=self._timeout_rules.robots_timeout_seconds,
            )
        except self._expected_loader_exceptions() as exc:
            result = self._error_rules.resolve(
                robots_url=robots_url,
                exc=exc,
                target_url=url,
                allowed_host_suffixes=(),
            )
            duplicate_tracker = self._duplicate_result_tracker
            duplicate_unknown = (
                duplicate_tracker.should_suppress_duplicate_unknown(
                    url=url,
                    result=result,
                )
            )
            if not duplicate_unknown:
                self._host_rules_store.record(url=url, result=result)

            self._log_result(
                url=url,
                result=result,
                duplicate_unknown=duplicate_unknown,
            )
            return result

        fetch_result = self._parser_cache.last_fetch_result(robots_url)
        result = self._decision_evaluator.evaluate(
            parser=parser,
            url=url,
            robots_url=(
                fetch_result.final_url if fetch_result else robots_url
            ),
            http_status=(fetch_result.status_code if fetch_result else None),
            user_agent=self._user_agent,
        )
        result = replace(
            result,
            sitemap_urls=self._extract_sitemap_urls(parser=parser),
        )

        duplicate_unknown = (
            self._duplicate_result_tracker.should_suppress_duplicate_unknown(
                url=url,
                result=result,
            )
        )
        if not duplicate_unknown:
            self._host_rules_store.record(url=url, result=result)

        self._log_result(
            url=url,
            result=result,
            duplicate_unknown=duplicate_unknown,
        )
        return result

    def host_rules_snapshot(self, *, url: str) -> Any:
        """Return a defensive snapshot of current host-level robots rules."""

        snapshot = self._host_rules_store.snapshot(url=url)
        try:
            return deepcopy(snapshot)
        except (RecursionError, RuntimeError, TypeError, ValueError):
            return snapshot

    async def _register_scheduler_advice(
        self,
        *,
        url: str,
        scheduler: SchedulerAdviceRegistrar | None,
    ) -> None:
        """
        Push robots-derived host advice into the scheduler when available.
        """

        if scheduler is None:
            return

        source_advice = self._host_rules_store.advise(url=url)
        await scheduler.register_host_rules_advice(
            url=url,
            advice=HostAdvice(
                discovery_factor=source_advice.discovery_factor,
                priority_penalty=source_advice.priority_penalty,
                hostility_score=source_advice.hostility_score,
                crawl_delay_seconds=source_advice.crawl_delay_seconds,
            ),
        )

    # ------------------------------------------------------------------
    # Logging and result helpers
    # ------------------------------------------------------------------
    def _log_result(
        self,
        *,
        url: str,
        result: RobotsCheckResult,
        duplicate_unknown: bool = False,
    ) -> None:
        confidence = result.confidence or RobotsConfidence.WEAK_UNKNOWN
        if duplicate_unknown:
            self._logger.debug(
                "robots_check_duplicate_suppressed",
                extra={
                    "url_host": self._host_from_url(url),
                    "decision": getattr(
                        result.decision, "value", result.decision
                    ),
                    "confidence": getattr(confidence, "value", confidence),
                    "http_status": result.http_status,
                    "reason": result.reason,
                    "source": result.source,
                },
            )
            return

        log_method = self._robots_result_log_method(result=result)
        log_method(
            "robots_check_completed",
            extra={
                "url_host": self._host_from_url(url),
                "robots_url": result.robots_url,
                "decision": getattr(result.decision, "value", result.decision),
                "confidence": getattr(confidence, "value", confidence),
                "action": result.to_access_action(
                    on_weak_unknown=self._on_weak_unknown,
                    on_transient_unknown=self._on_transient_unknown,
                    on_hostile_unknown=self._on_hostile_unknown,
                ).value,
                "authoritative": result.is_authoritative,
                "http_status": result.http_status,
                "error": result.error_type,
                "reason": result.reason,
                "source": result.source,
                "crawl_delay_seconds": result.crawl_delay_seconds,
                "retry_after_seconds": result.retry_after_seconds,
                "host_penalty": result.host_penalty,
                "suggested_discovery_factor": result.suggested_discovery_factor,
                "sitemap_urls": list(result.sitemap_urls),
            },
        )

    def _robots_result_log_method(
        self,
        *,
        result: RobotsCheckResult,
    ) -> Callable[..., None]:
        logger = self._logger
        confidence = result.confidence
        if result.decision == RobotsDecision.DISALLOWED:
            return logger.warning
        if confidence == RobotsConfidence.HOSTILE_UNKNOWN:
            return logger.info
        if result.decision == RobotsDecision.UNKNOWN:
            return logger.info
        if confidence in {
            RobotsConfidence.AUTHORITATIVE_ALLOW,
            RobotsConfidence.AUTHORITATIVE_ABSENT,
        }:
            return logger.debug
        return logger.info

    @staticmethod
    def _extract_sitemap_urls(
        *,
        parser: RobotFileParser,
    ) -> tuple[str, ...]:
        """Return sitemap URLs advertised by robots.txt, if available."""

        try:
            values = parser.site_maps() or ()
        except (TypeError, ValueError):
            return ()

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            url = str(value).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            normalized.append(url)
        return tuple(normalized)

    @staticmethod
    def _expected_loader_exceptions() -> tuple[type[Exception], ...]:
        """
        Return exceptions that represent expected robots loading failures.
        """

        return (RobotsLoaderError,)
