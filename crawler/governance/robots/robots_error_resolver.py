"""
Translate robots loading failures into structured tri-state results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.governance.robots.robots_check_result import (
    RobotsCheckResult,
    RobotsConfidence,
)
from crawler.governance.robots.robots_error_classifier import (
    RobotsErrorClassifier,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crawler.governance.robots.robots_fallback_rules import (
        RobotsFallbackRules,
    )


@dataclass(frozen=True, slots=True)
class RobotsErrorResolverRules:
    """Rules values used when robots loading fails."""

    default_disallow_on_unknown: bool = True
    hostile_unknown_penalty: float = 0.35
    hostile_unknown_discovery_factor: float = 0.25
    transient_unknown_penalty: float = 0.10
    transient_unknown_discovery_factor: float = 0.70
    weak_unknown_penalty: float = 0.05
    weak_unknown_discovery_factor: float = 0.85
    rate_limited_unknown_penalty: float = 0.10
    rate_limited_unknown_discovery_factor: float = 0.70


@dataclass(frozen=True, slots=True)
class RobotsErrorResolverDependencies:
    """Classifiers and fallback rules used by robots error resolution."""

    classifier: RobotsErrorClassifier
    fallback_rules: RobotsFallbackRules


@dataclass(frozen=True, slots=True)
class RobotsUnknownResultRequest:
    """Fields used to build a structured UNKNOWN robots result."""

    robots_url: str
    confidence: RobotsConfidence
    reason: str
    error_type: str
    host_penalty: float
    suggested_discovery_factor: float
    http_status: int | None = None
    retry_after_seconds: float | None = None
    source: str = "robots_error_resolver"


class RobotsErrorResolver:
    """Translate robots loading failures into structured tri-state results.

    Important rules rule:
    - only missing robots (404/410) becomes authoritative ALLOWED
    - uncertain or hostile conditions remain UNKNOWN
    """

    _HOSTILE_HTTP_CODES = frozenset({401, 403})
    _TRANSIENT_HTTP_CODES = frozenset({408, 500, 502, 503, 504})
    _RATE_LIMIT_HTTP_CODES = frozenset({429})

    def __init__(
        self,
        *,
        rules: RobotsErrorResolverRules,
        dependencies: RobotsErrorResolverDependencies,
    ) -> None:
        self._default_disallow_on_unknown = bool(
            rules.default_disallow_on_unknown
        )
        self._classifier = dependencies.classifier
        self._fallback_rules = dependencies.fallback_rules

        self._hostile_unknown_penalty = float(rules.hostile_unknown_penalty)
        self._hostile_unknown_discovery_factor = float(
            rules.hostile_unknown_discovery_factor
        )

        self._transient_unknown_penalty = float(
            rules.transient_unknown_penalty
        )
        self._transient_unknown_discovery_factor = float(
            rules.transient_unknown_discovery_factor
        )

        self._weak_unknown_penalty = float(rules.weak_unknown_penalty)
        self._weak_unknown_discovery_factor = float(
            rules.weak_unknown_discovery_factor
        )

        self._rate_limited_unknown_penalty = float(
            rules.rate_limited_unknown_penalty
        )
        self._rate_limited_unknown_discovery_factor = float(
            rules.rate_limited_unknown_discovery_factor
        )

    @property
    def default_disallow_on_unknown(self) -> bool:
        """Return whether UNKNOWN should map to disallow."""

        return self._default_disallow_on_unknown

    # ------------------------------------------------------------------
    # Error resolution logic
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Public resolution
    # ------------------------------------------------------------------
    def resolve(
        self,
        *,
        robots_url: str,
        exc: Exception,
        target_url: str | None = None,
        allowed_host_suffixes: Sequence[str] | None = None,
    ) -> RobotsCheckResult:
        """Translate a robots loading failure into a tri-state decision."""

        classification = self._classifier.classify(exc)

        if classification.kind == "http" and (
            classification.http_status is not None
        ):
            return self._resolve_http_error(
                robots_url=robots_url,
                status_code=classification.http_status,
                error_type=classification.error_type,
                retry_after_seconds=classification.retry_after_seconds,
                target_url=target_url,
                allowed_host_suffixes=allowed_host_suffixes,
            )

        if classification.kind == "timeout":
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.TRANSIENT_UNKNOWN,
                    reason="loader_timeout_unknown",
                    error_type=classification.error_type,
                    host_penalty=self._transient_unknown_penalty,
                    suggested_discovery_factor=(
                        self._transient_unknown_discovery_factor
                    ),
                ),
            )

        if classification.kind == "url_error":
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.TRANSIENT_UNKNOWN,
                    reason="loader_network_unknown",
                    error_type=classification.error_type,
                    host_penalty=self._transient_unknown_penalty,
                    suggested_discovery_factor=(
                        self._transient_unknown_discovery_factor
                    ),
                ),
            )

        if classification.kind == "client_error":
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.WEAK_UNKNOWN,
                    reason="loader_client_error_unknown",
                    error_type=classification.error_type,
                    host_penalty=self._weak_unknown_penalty,
                    suggested_discovery_factor=(
                        self._weak_unknown_discovery_factor
                    ),
                ),
            )

        return self._unknown_result(
            request=RobotsUnknownResultRequest(
                robots_url=robots_url,
                confidence=RobotsConfidence.WEAK_UNKNOWN,
                reason="loader_error_unknown",
                error_type=classification.error_type,
                host_penalty=self._weak_unknown_penalty,
                suggested_discovery_factor=self._weak_unknown_discovery_factor,
            ),
        )

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------
    def _resolve_http_error(
        self,
        *,
        robots_url: str,
        status_code: int,
        error_type: str,
        retry_after_seconds: float | None,
        target_url: str | None,
        allowed_host_suffixes: Sequence[str] | None,
    ) -> RobotsCheckResult:
        if status_code in {404, 410}:
            return RobotsCheckResult.allowed_absent(
                robots_url=robots_url,
                error_type=error_type,
                http_status=status_code,
            )

        if status_code == 403 and self._fallback_rules.is_allowed_host_suffix(
            target_url=target_url,
            allowed_host_suffixes=allowed_host_suffixes,
        ):
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.WEAK_UNKNOWN,
                    reason="http_403_allowed_host_suffix_override_unknown",
                    error_type=error_type,
                    http_status=status_code,
                    retry_after_seconds=retry_after_seconds,
                    host_penalty=self._weak_unknown_penalty,
                    suggested_discovery_factor=(
                        self._weak_unknown_discovery_factor
                    ),
                    source="robots_error_resolver_override",
                ),
            )

        if (
            status_code == 403
            and self._fallback_rules.is_storage_backed_media_asset(
                target_url=target_url
            )
        ):
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.WEAK_UNKNOWN,
                    reason="http_403_storage_media_unknown",
                    error_type=error_type,
                    http_status=status_code,
                    retry_after_seconds=retry_after_seconds,
                    host_penalty=self._weak_unknown_penalty,
                    suggested_discovery_factor=(
                        self._weak_unknown_discovery_factor
                    ),
                    source="robots_error_resolver_storage_media_override",
                ),
            )

        if status_code in self._RATE_LIMIT_HTTP_CODES:
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.TRANSIENT_UNKNOWN,
                    reason=f"http_{status_code}_rate_limited_unknown",
                    error_type=error_type,
                    http_status=status_code,
                    retry_after_seconds=retry_after_seconds,
                    host_penalty=self._rate_limited_unknown_penalty,
                    suggested_discovery_factor=(
                        self._rate_limited_unknown_discovery_factor
                    ),
                ),
            )

        if status_code in self._HOSTILE_HTTP_CODES:
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.HOSTILE_UNKNOWN,
                    reason=f"http_{status_code}_unknown",
                    error_type=error_type,
                    http_status=status_code,
                    retry_after_seconds=retry_after_seconds,
                    host_penalty=self._hostile_unknown_penalty,
                    suggested_discovery_factor=(
                        self._hostile_unknown_discovery_factor
                    ),
                ),
            )

        if status_code in self._TRANSIENT_HTTP_CODES:
            return self._unknown_result(
                request=RobotsUnknownResultRequest(
                    robots_url=robots_url,
                    confidence=RobotsConfidence.TRANSIENT_UNKNOWN,
                    reason=f"http_{status_code}_unknown",
                    error_type=error_type,
                    http_status=status_code,
                    retry_after_seconds=retry_after_seconds,
                    host_penalty=self._transient_unknown_penalty,
                    suggested_discovery_factor=(
                        self._transient_unknown_discovery_factor
                    ),
                ),
            )

        return self._unknown_result(
            request=RobotsUnknownResultRequest(
                robots_url=robots_url,
                confidence=RobotsConfidence.WEAK_UNKNOWN,
                reason=f"http_{status_code}_unknown",
                error_type=error_type,
                http_status=status_code,
                retry_after_seconds=retry_after_seconds,
                host_penalty=self._weak_unknown_penalty,
                suggested_discovery_factor=self._weak_unknown_discovery_factor,
            ),
        )

    # ------------------------------------------------------------------
    # Unknown result construction
    # ------------------------------------------------------------------
    def _unknown_result(
        self,
        *,
        request: RobotsUnknownResultRequest,
    ) -> RobotsCheckResult:
        return RobotsCheckResult.unknown(
            robots_url=request.robots_url,
            confidence=request.confidence,
            reason=request.reason,
            error_type=request.error_type,
            http_status=request.http_status,
            retry_after_seconds=request.retry_after_seconds,
            host_penalty=request.host_penalty,
            suggested_discovery_factor=request.suggested_discovery_factor,
            source=request.source,
        )
