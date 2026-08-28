"""Store host-level robots multimodal for adaptive scheduling advice."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from urllib.parse import urlsplit

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_check_result import (
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)
from logger.project_logger import ProjectLogger


@dataclass(frozen=True, slots=True)
class RobotsHostRulesAdvice:
    """Read-only downstream advice derived from host rules state."""

    host: str
    should_reduce_discovery: bool
    discovery_factor: float
    priority_penalty: float
    hostility_score: float
    crawl_delay_seconds: float | None
    consecutive_unknown_count: int
    consecutive_hostile_unknown_count: int
    last_http_status: int | None
    last_decision: str | None
    last_confidence: str | None


@dataclass(slots=True)
class _RobotsHostRulesState:
    """Mutable host-level robots rules memory."""

    last_http_status: int | None = None
    last_decision: str | None = None
    last_confidence: str | None = None
    crawl_delay_seconds: float | None = None
    last_updated_monotonic: float = 0.0
    last_authoritative_success_monotonic: float = 0.0
    last_unknown_monotonic: float = 0.0
    consecutive_unknown_count: int = 0
    consecutive_hostile_unknown_count: int = 0
    hostility_score: float = 0.0
    discovery_factor: float = 1.0
    priority_penalty: float = 0.0


class RobotsHostRulesStore:
    """Track host-level robots multimodal for adaptive downstream behavior."""

    def __init__(
        self,
        *,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
    ) -> None:
        self._host_normalizer = host_normalizer
        self._logger = logger
        self._states: dict[str, _RobotsHostRulesState] = {}

    def record(self, *, url: str, result: RobotsCheckResult) -> None:
        """Update host-level rules state from a robots check result."""

        host = self._host_from_url(url)
        if host is None:
            return

        state = self._states.setdefault(host, _RobotsHostRulesState())
        now = monotonic()
        confidence = result.confidence or RobotsConfidence.WEAK_UNKNOWN
        state.last_http_status = result.http_status
        state.last_decision = result.decision.value
        state.last_confidence = confidence.value
        state.last_updated_monotonic = now
        if result.crawl_delay_seconds is not None:
            state.crawl_delay_seconds = max(
                0.0,
                float(result.crawl_delay_seconds),
            )
        elif result.is_authoritative:
            state.crawl_delay_seconds = None
        state.discovery_factor = min(
            state.discovery_factor,
            max(0.05, float(result.suggested_discovery_factor)),
        )
        state.priority_penalty = max(
            state.priority_penalty,
            0.0,
            float(result.host_penalty),
        )

        if result.decision == RobotsDecision.UNKNOWN:
            state.consecutive_unknown_count += 1
            state.last_unknown_monotonic = now
        else:
            state.consecutive_unknown_count = 0

        if confidence == RobotsConfidence.HOSTILE_UNKNOWN:
            state.consecutive_hostile_unknown_count += 1
            state.hostility_score = min(1.0, state.hostility_score + 0.25)
        elif (
            result.decision == RobotsDecision.ALLOWED
            and result.is_authoritative
        ):
            state.consecutive_hostile_unknown_count = 0
            state.hostility_score = max(0.0, state.hostility_score - 0.20)
            state.last_authoritative_success_monotonic = now
            state.discovery_factor = 1.0
            state.priority_penalty = 0.0
        else:
            state.consecutive_hostile_unknown_count = 0
            state.hostility_score = max(0.0, state.hostility_score - 0.05)

        self._logger.debug(
            "robots_host_rules_updated",
            extra={
                "host": host,
                "last_decision": state.last_decision,
                "last_confidence": state.last_confidence,
                "hostility_score": round(state.hostility_score, 3),
                "consecutive_unknown_count": (state.consecutive_unknown_count),
                "consecutive_hostile_unknown_count": (
                    state.consecutive_hostile_unknown_count
                ),
                "crawl_delay_seconds": (
                    round(state.crawl_delay_seconds, 6)
                    if state.crawl_delay_seconds is not None
                    else None
                ),
                "discovery_factor": round(state.discovery_factor, 3),
                "priority_penalty": round(state.priority_penalty, 3),
            },
        )

    def snapshot(self, *, url: str) -> _RobotsHostRulesState:
        """Return the current host-level robots rules snapshot."""

        host = self._host_from_url(url)
        if host is None:
            return _RobotsHostRulesState()
        existing = self._states.get(host)
        if existing is not None:
            return existing
        return _RobotsHostRulesState()

    def advise(self, *, url: str) -> RobotsHostRulesAdvice:
        """Return downstream scheduling and discovery advice for the host."""

        host = self._host_from_url(url) or ""
        state = self._states.get(host, _RobotsHostRulesState())
        return RobotsHostRulesAdvice(
            host=host,
            should_reduce_discovery=state.discovery_factor < 0.999,
            discovery_factor=state.discovery_factor,
            priority_penalty=state.priority_penalty,
            hostility_score=state.hostility_score,
            crawl_delay_seconds=state.crawl_delay_seconds,
            consecutive_unknown_count=state.consecutive_unknown_count,
            consecutive_hostile_unknown_count=(
                state.consecutive_hostile_unknown_count
            ),
            last_http_status=state.last_http_status,
            last_decision=state.last_decision,
            last_confidence=state.last_confidence,
        )

    def get_discovery_factor(self, *, url: str) -> float:
        """Return the current discovery multiplier for the host."""

        return self.advise(url=url).discovery_factor

    def get_priority_penalty(self, *, url: str) -> float:
        """Return the current priority penalty for the host."""

        return self.advise(url=url).priority_penalty

    def should_reduce_discovery(self, *, url: str) -> bool:
        """
        Return whether newly discovered URLs should be reduced for the host.
        """

        return self.advise(url=url).should_reduce_discovery

    def _host_from_url(self, url: str) -> str | None:
        return self._host_normalizer.normalize(urlsplit(url).hostname)
