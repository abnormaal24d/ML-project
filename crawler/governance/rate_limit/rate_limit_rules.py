"""
Public models and helpers for
crawler.governance.rate_limit.rate_limit_rules domain.

Exports: RateLimitRules.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.governance.rate_limit.rate_limit_host_state import (
        RateLimitHostState,
    )


class RateLimitRules:
    """Apply adaptive rate, cooldown, and crawl-delay rules to host state."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        min_requests_per_second_value: float,
        max_requests_per_second_value: float,
        backoff_factor: float,
        ramp_up_factor: float,
        error_cooldown_seconds: float,
        feedback_status_codes: frozenset[int],
        default_crawl_delay_seconds: float | None,
    ) -> None:
        self._logger = logger
        self._min_requests_per_second_value = min_requests_per_second_value
        self._max_requests_per_second_value = max_requests_per_second_value
        self._backoff_factor = backoff_factor
        self._ramp_up_factor = ramp_up_factor
        self._error_cooldown_seconds = error_cooldown_seconds
        self._feedback_status_codes = feedback_status_codes
        self._default_crawl_delay_seconds = default_crawl_delay_seconds

    # ------------------------------------------------------------------
    # Result reporting and adaptation
    # ------------------------------------------------------------------
    def report_result(
        self,
        *,
        host: str,
        state: RateLimitHostState,
        status_code: int,
        latency_seconds: float,
        now: float,
    ) -> None:
        """Adapt host request rate based on response multimodal."""

        safe_latency_seconds = max(0.0, float(latency_seconds))
        crawl_delay_seconds = self.effective_crawl_delay_seconds(state)

        previous_adaptive_rps = state.adaptive_requests_per_second
        previous_effective_rps = state.effective_requests_per_second
        previous_cooldown_until = state.cooldown_until
        next_adaptive_rps = previous_adaptive_rps

        if status_code in self._feedback_status_codes:
            if previous_adaptive_rps > 0:
                next_adaptive_rps = max(
                    self._min_requests_per_second_value,
                    previous_adaptive_rps * self._backoff_factor,
                )
            state.cooldown_until = max(
                state.cooldown_until,
                now + self._error_cooldown_seconds,
            )
        elif (
            previous_adaptive_rps > 0
            and now >= state.cooldown_until
            and 200 <= status_code < 400
        ):
            if safe_latency_seconds <= 1.0:
                next_adaptive_rps = min(
                    self._max_requests_per_second_value,
                    previous_adaptive_rps * self._ramp_up_factor,
                )
            elif safe_latency_seconds >= 3.0:
                next_adaptive_rps = max(
                    self._min_requests_per_second_value,
                    previous_adaptive_rps * 0.95,
                )

        next_adaptive_rps = self.clamp_requests_per_second(next_adaptive_rps)
        next_effective_rps = self.effective_requests_per_second_from_values(
            adaptive_requests_per_second=next_adaptive_rps,
            crawl_delay_seconds=crawl_delay_seconds,
        )

        adaptive_changed = (
            abs(next_adaptive_rps - previous_adaptive_rps) >= 1e-9
        )
        effective_changed = (
            abs(next_effective_rps - previous_effective_rps) >= 1e-9
        )
        cooldown_changed = (
            abs(state.cooldown_until - previous_cooldown_until) >= 1e-9
        )
        if (
            not adaptive_changed
            and not effective_changed
            and not cooldown_changed
        ):
            return

        state.adaptive_requests_per_second = next_adaptive_rps
        state.effective_requests_per_second = next_effective_rps

        should_log_update = (
            status_code in self._feedback_status_codes
            or cooldown_changed
            or self._rate_bucket(previous_effective_rps)
            != self._rate_bucket(next_effective_rps)
            or self._rate_bucket(previous_adaptive_rps)
            != self._rate_bucket(next_adaptive_rps)
        )
        if not should_log_update:
            return

        self._logger.debug(
            "rate_limiter_host_rate_updated",
            extra={
                "host": host,
                "status_code": status_code,
                "latency_seconds": round(safe_latency_seconds, 4),
                "crawl_delay_seconds": (
                    round(crawl_delay_seconds, 6)
                    if crawl_delay_seconds and crawl_delay_seconds > 0
                    else None
                ),
                "previous_rps": round(previous_effective_rps, 4),
                "next_rps": round(next_effective_rps, 4),
                "adaptive_previous_rps": round(previous_adaptive_rps, 4),
                "adaptive_next_rps": round(next_adaptive_rps, 4),
                "previous_cooldown_until": round(previous_cooldown_until, 6),
                "next_cooldown_until": round(state.cooldown_until, 6),
            },
        )

    # ------------------------------------------------------------------
    # Rules setters
    # ------------------------------------------------------------------
    def set_host_crawl_delay(
        self,
        *,
        host: str,
        state: RateLimitHostState,
        crawl_delay_seconds: float | None,
    ) -> None:
        """Set or clear a hard crawl-delay override for a host."""

        previous_crawl_delay = state.crawl_delay_override_seconds
        previous_effective_rps = state.effective_requests_per_second
        previous_next_request_not_before = state.next_request_not_before
        last_reserved_at = state.timestamps[-1] if state.timestamps else None

        state.crawl_delay_override_seconds = (
            self.coerce_optional_non_negative_float(crawl_delay_seconds)
        )
        applied_crawl_delay = state.crawl_delay_override_seconds

        if (
            applied_crawl_delay is not None
            and applied_crawl_delay > 0
            and last_reserved_at is not None
        ):
            state.next_request_not_before = max(
                state.next_request_not_before,
                last_reserved_at + applied_crawl_delay,
            )

        state.effective_requests_per_second = (
            self.effective_requests_per_second_from_values(
                adaptive_requests_per_second=(
                    state.adaptive_requests_per_second
                ),
                crawl_delay_seconds=self.effective_crawl_delay_seconds(state),
            )
        )
        effective_rps = state.effective_requests_per_second

        if (
            previous_crawl_delay == applied_crawl_delay
            and abs(effective_rps - previous_effective_rps) < 1e-9
            and abs(
                state.next_request_not_before
                - previous_next_request_not_before
            )
            < 1e-9
        ):
            return

        self._logger.debug(
            "rate_limiter_host_crawl_delay_updated",
            extra={
                "host": host,
                "crawl_delay_seconds": applied_crawl_delay,
                "previous_next_request_not_before": round(
                    previous_next_request_not_before,
                    6,
                ),
                "next_request_not_before": round(
                    state.next_request_not_before,
                    6,
                ),
                "effective_requests_per_second": round(effective_rps, 4),
            },
        )

    def set_host_requests_per_second(
        self,
        *,
        host: str,
        state: RateLimitHostState,
        requests_per_second: float,
    ) -> None:
        """Set the current adaptive host request-rate budget explicitly."""

        state.adaptive_requests_per_second = self.clamp_requests_per_second(
            requests_per_second
        )
        state.effective_requests_per_second = (
            self.effective_requests_per_second_from_values(
                adaptive_requests_per_second=(
                    state.adaptive_requests_per_second
                ),
                crawl_delay_seconds=self.effective_crawl_delay_seconds(state),
            )
        )

        self._logger.debug(
            "rate_limiter_host_rps_set",
            extra={
                "host": host,
                "adaptive_requests_per_second": round(
                    state.adaptive_requests_per_second,
                    4,
                ),
                "effective_requests_per_second": round(
                    state.effective_requests_per_second,
                    4,
                ),
            },
        )

    def effective_crawl_delay_seconds(
        self,
        state: RateLimitHostState,
    ) -> float:
        """Return the hard crawl delay currently in force for the host."""

        if state.crawl_delay_override_seconds is not None:
            return state.crawl_delay_override_seconds
        return self._default_crawl_delay_seconds or 0.0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def clamp_requests_per_second(self, value: float) -> float:
        """Clamp a request rate into the configured allowed range."""

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not isfinite(numeric_value) or numeric_value <= 0:
            return 0.0

        return max(
            self._min_requests_per_second_value,
            min(self._max_requests_per_second_value, numeric_value),
        )

    @classmethod
    def coerce_non_negative_finite_float_or_default(
        cls,
        value: object,
        *,
        default: float,
    ) -> float:
        """Return a non-negative finite float or the provided default."""

        numeric_value = cls._coerce_float(value)
        if numeric_value is None:
            return default
        if not isfinite(numeric_value) or numeric_value < 0:
            return default
        return numeric_value

    @classmethod
    def coerce_optional_non_negative_float(
        cls,
        value: object,
    ) -> float | None:
        """Return a non-negative finite float or None."""

        if value is None:
            return None
        numeric_value = cls._coerce_float(value)
        if numeric_value is None:
            raise ValueError("expected a non-negative float or None")
        if not isfinite(numeric_value) or numeric_value < 0:
            raise ValueError("expected a non-negative finite float or None")
        return numeric_value

    @staticmethod
    # ------------------------------------------------------------------
    # Effective rate calculations
    # ------------------------------------------------------------------
    def effective_requests_per_second_from_values(
        *,
        adaptive_requests_per_second: float,
        crawl_delay_seconds: float,
    ) -> float:
        """Return the effective request rate after crawl-delay capping."""

        effective_rps = adaptive_requests_per_second
        if crawl_delay_seconds > 0:
            crawl_delay_cap_rps = 1.0 / crawl_delay_seconds
            if effective_rps <= 0:
                effective_rps = crawl_delay_cap_rps
            else:
                effective_rps = min(effective_rps, crawl_delay_cap_rps)
        return max(0.0, effective_rps)

    @staticmethod
    def rate_bucket(value: float) -> int:
        """Return a coarse bucket for material request-rate changes."""

        return int(max(0.0, value) * 10.0)

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, str, bytes, bytearray)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _rate_bucket(value: float) -> int:
        return RateLimitRules.rate_bucket(value)
