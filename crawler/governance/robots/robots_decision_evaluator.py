"""Evaluate RobotFileParser decisions into structured RobotsCheckResult."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from crawler.governance.robots.robots_check_result import (
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)
from crawler.governance.robots.robots_parser_loader import (
    crawl_delay_seconds_for_agent,
)

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger


class RobotsDecisionEvaluator:
    """Evaluate robots parser decisions against runtime rules."""

    def __init__(
        self,
        *,
        respect_crawl_delay: bool,
        max_crawl_delay_s: float,
        user_agent: str,
        logger: ProjectLogger,
    ) -> None:
        self._respect_crawl_delay = respect_crawl_delay
        self._max_crawl_delay_s = float(max_crawl_delay_s)
        self._user_agent = user_agent.strip()
        self._logger = logger

    # ------------------------------------------------------------------
    # Decision API
    # ------------------------------------------------------------------
    def is_allowed(
        self,
        parser: RobotFileParser,
        url: str,
        *,
        user_agent: str | None = None,
    ) -> bool:
        """Return whether the parser allows crawling the URL."""

        return (
            self.evaluate(
                parser=parser,
                url=url,
                user_agent=user_agent,
            ).decision
            == RobotsDecision.ALLOWED
        )

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        parser: RobotFileParser,
        url: str,
        robots_url: str = "",
        http_status: int | None = None,
        user_agent: str | None = None,
    ) -> RobotsCheckResult:
        """Return the structured robots decision for the parser and URL."""

        effective_user_agent = self._resolve_user_agent(user_agent)

        crawl_delay_seconds: float | None = None
        if self._respect_crawl_delay:
            crawl_delay_seconds = self._crawl_delay_seconds(
                parser=parser,
                user_agent=effective_user_agent,
            )

            if crawl_delay_seconds is not None and (
                crawl_delay_seconds > self._max_crawl_delay_s
            ):
                self._logger.debug(
                    "robots_crawl_delay_rejected",
                    extra={
                        "url_host": self._host_from_url(url),
                        "robots_url": robots_url,
                        "user_agent": effective_user_agent,
                        "crawl_delay": crawl_delay_seconds,
                        "max_crawl_delay_s": self._max_crawl_delay_s,
                    },
                )
                return RobotsCheckResult(
                    robots_url=robots_url,
                    decision=RobotsDecision.UNKNOWN,
                    confidence=RobotsConfidence.WEAK_UNKNOWN,
                    reason="robots_crawl_delay_operationally_unsupported",
                    source="robots_rules",
                    is_authoritative=False,
                    http_status=http_status,
                    crawl_delay_seconds=crawl_delay_seconds,
                )

        allowed = bool(parser.can_fetch(effective_user_agent, url))
        self._logger.debug(
            "robots_decision_evaluated",
            extra={
                "url_host": self._host_from_url(url),
                "robots_url": robots_url,
                "allowed": allowed,
                "user_agent": effective_user_agent,
                "crawl_delay": crawl_delay_seconds,
            },
        )

        if not allowed:
            return RobotsCheckResult(
                robots_url=robots_url,
                decision=RobotsDecision.DISALLOWED,
                confidence=RobotsConfidence.AUTHORITATIVE_DENY,
                reason="robots_rules_disallow",
                source="robots_rules",
                is_authoritative=True,
                http_status=http_status,
                crawl_delay_seconds=crawl_delay_seconds,
            )

        return RobotsCheckResult(
            robots_url=robots_url,
            decision=RobotsDecision.ALLOWED,
            confidence=RobotsConfidence.AUTHORITATIVE_ALLOW,
            reason="robots_rules_allow",
            source="robots_rules",
            is_authoritative=True,
            http_status=http_status,
            crawl_delay_seconds=crawl_delay_seconds,
        )

    def _resolve_user_agent(self, user_agent: str | None) -> str:
        """Return the runtime user-agent override or the configured default."""

        if user_agent is None:
            return self._user_agent

        normalized = user_agent.strip()
        if normalized:
            return normalized

        return self._user_agent

    def _crawl_delay_seconds(
        self,
        *,
        parser: RobotFileParser,
        user_agent: str,
    ) -> float | None:
        """Return the crawl-delay matching the agent from the parsed table.

        urllib's own ``crawl_delay`` accessor is unavailable for in-process
        parses (it is gated behind a file read), so the loader attaches the
        extracted delay table to the parser instance instead.
        """

        table = getattr(parser, "crawl_delay_seconds_table", None)
        if not table:
            return None

        try:
            return crawl_delay_seconds_for_agent(
                table=table,
                user_agent=user_agent,
            )
        except (TypeError, ValueError):  # exception-rules: defensive-parse
            return None

    @staticmethod
    def _host_from_url(url: str) -> str:
        try:
            return urlsplit(url).hostname or "unknown"
        except (TypeError, ValueError):
            return "unknown"
