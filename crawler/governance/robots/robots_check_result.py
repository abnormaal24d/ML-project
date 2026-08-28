"""Robots check outcome schema and access-action model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RobotsDecision(StrEnum):
    """Possible multimodal of a robots access check."""

    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    UNKNOWN = "unknown"


class RobotsConfidence(StrEnum):
    """Confidence and semantics of a robots access decision."""

    AUTHORITATIVE_ALLOW = "authoritative_allow"
    AUTHORITATIVE_DENY = "authoritative_deny"
    AUTHORITATIVE_ABSENT = "authoritative_absent"
    WEAK_UNKNOWN = "weak_unknown"
    TRANSIENT_UNKNOWN = "transient_unknown"
    HOSTILE_UNKNOWN = "hostile_unknown"


class RobotsAccessAction(StrEnum):
    """Final access action applied to a request by the robots gate."""

    ALLOW = "allow"
    BLOCK = "block"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class RobotsCheckResult:
    """Structured result of a robots access check."""

    robots_url: str
    decision: RobotsDecision
    reason: str
    source: str
    is_authoritative: bool
    error_type: str | None = None
    confidence: RobotsConfidence | None = None
    http_status: int | None = None
    crawl_delay_seconds: float | None = None
    retry_after_seconds: float | None = None
    host_penalty: float = 0.0
    suggested_discovery_factor: float = 1.0
    sitemap_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence is None:
            object.__setattr__(self, "confidence", self._default_confidence())

    @classmethod
    def allowed_absent(
        cls,
        *,
        robots_url: str,
        error_type: str,
        http_status: int,
    ) -> RobotsCheckResult:
        """Return the authoritative result for absent robots.txt."""

        return cls(
            robots_url=robots_url,
            decision=RobotsDecision.ALLOWED,
            confidence=RobotsConfidence.AUTHORITATIVE_ABSENT,
            reason="missing_robots_allows",
            source="robots_absence",
            is_authoritative=True,
            error_type=error_type,
            http_status=http_status,
        )

    @classmethod
    def unknown(
        cls,
        *,
        robots_url: str,
        confidence: RobotsConfidence,
        reason: str,
        error_type: str,
        host_penalty: float,
        suggested_discovery_factor: float,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
        source: str = "robots_error_resolver",
    ) -> RobotsCheckResult:
        """Return a non-authoritative UNKNOWN result."""

        return cls(
            robots_url=robots_url,
            decision=RobotsDecision.UNKNOWN,
            confidence=confidence,
            reason=reason,
            source=source,
            is_authoritative=False,
            error_type=error_type,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
            host_penalty=host_penalty,
            suggested_discovery_factor=suggested_discovery_factor,
        )

    def to_access_action(
        self,
        *,
        on_weak_unknown: str,
        on_transient_unknown: str,
        on_hostile_unknown: str,
    ) -> RobotsAccessAction:
        """Project the tri-state decision to an access action.

        Resolution rules:
        - ALLOWED -> ALLOW
        - DISALLOWED -> BLOCK
        - UNKNOWN -> confidence-specific policy setting (defaults block for
          weak/hostile unknowns and defer for transient unknowns)
        """

        if self.decision == RobotsDecision.ALLOWED:
            return RobotsAccessAction.ALLOW

        if self.decision == RobotsDecision.DISALLOWED:
            return RobotsAccessAction.BLOCK

        confidence = self.confidence or RobotsConfidence.WEAK_UNKNOWN
        if confidence == RobotsConfidence.HOSTILE_UNKNOWN:
            return RobotsAccessAction(on_hostile_unknown)
        if confidence == RobotsConfidence.TRANSIENT_UNKNOWN:
            return RobotsAccessAction(on_transient_unknown)
        return RobotsAccessAction(on_weak_unknown)

    @property
    def should_reduce_discovery(self) -> bool:
        """Return whether downstream discovery should be reduced."""

        return self.suggested_discovery_factor < 0.999

    def _default_confidence(self) -> RobotsConfidence:
        if self.decision == RobotsDecision.ALLOWED and self.is_authoritative:
            return RobotsConfidence.AUTHORITATIVE_ALLOW

        if (
            self.decision == RobotsDecision.DISALLOWED
            and self.is_authoritative
        ):
            return RobotsConfidence.AUTHORITATIVE_DENY

        if (
            self.decision == RobotsDecision.ALLOWED
            and self.reason == "missing_robots_allows"
        ):
            return RobotsConfidence.AUTHORITATIVE_ABSENT

        return RobotsConfidence.WEAK_UNKNOWN
