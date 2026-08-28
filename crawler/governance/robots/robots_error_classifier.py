"""Exception classification for robots loading failures."""

from __future__ import annotations

from dataclasses import dataclass

from crawler.governance.robots.robots_fetch_errors import (
    RobotsHttpStatusError,
    RobotsNetworkError,
    RobotsRedirectRejectedError,
    RobotsTimeoutError,
)


@dataclass(frozen=True, slots=True)
class RobotsErrorClassification:
    """Normalized robots loading error classification."""

    kind: str
    error_type: str
    http_status: int | None = None
    retry_after_seconds: float | None = None


class RobotsErrorClassifier:
    """Classify loader exceptions before rules decisions are made."""

    def classify(self, exc: Exception) -> RobotsErrorClassification:
        """Classify a robots loader exception."""
        if isinstance(exc, RobotsHttpStatusError):
            return RobotsErrorClassification(
                kind="http",
                error_type=exc.error_type,
                http_status=exc.status_code,
                retry_after_seconds=exc.retry_after_seconds,
            )

        if isinstance(exc, RobotsTimeoutError):
            return RobotsErrorClassification(
                kind="timeout",
                error_type=exc.error_type,
            )

        if isinstance(exc, RobotsNetworkError):
            return RobotsErrorClassification(
                kind="url_error",
                error_type=exc.error_type,
            )

        if isinstance(exc, RobotsRedirectRejectedError):
            return RobotsErrorClassification(
                kind="client_error",
                error_type=exc.reason,
            )

        return RobotsErrorClassification(
            kind="unknown",
            error_type=type(exc).__name__,
        )
