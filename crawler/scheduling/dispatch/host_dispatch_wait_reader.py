"""Read host dispatch wait times from host profile state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.environment.default_values import (
    DEFAULT_INFLIGHT_HOST_WAIT_SECONDS,
)

if TYPE_CHECKING:
    from crawler.governance.host_suppression import HostSuppressionStore
    from crawler.governance.rate_limit.rate_limiter import RateLimiter


class HostDispatchWaitReader:
    """Compute dispatch wait windows from suppression and rate limiting."""

    def __init__(
        self,
        *,
        rate_limiter: RateLimiter | None,
        max_inflight_per_host: int | None = None,
        inflight_host_wait_seconds: float = (
            DEFAULT_INFLIGHT_HOST_WAIT_SECONDS
        ),
    ) -> None:
        """Initialize dispatch wait rules.

        ``inflight_host_wait_seconds`` is measured in seconds and defaults to
        the scheduler host-wait rules.
        """

        self._rate_limiter = rate_limiter
        self._host_suppression_store: HostSuppressionStore | None = None
        self._max_inflight_per_host = max_inflight_per_host
        self._inflight_host_wait_seconds = max(
            0.001,
            float(inflight_host_wait_seconds),
        )

    def set_host_suppression_reader(
        self,
        reader: HostSuppressionStore | None,
    ) -> None:
        self._host_suppression_store = reader

    async def wait_seconds(
        self,
        *,
        host: str | None,
        inflight_count_by_host: dict[str, int],
    ) -> float | None:
        inflight_seconds = self._inflight_wait_seconds(
            host=host,
            inflight_count_by_host=inflight_count_by_host,
        )
        suppression_seconds = self._suppression_remaining_seconds(host=host)
        rate_limit_seconds = await self._rate_limit_wait_seconds(host=host)

        wait_candidates = [
            candidate
            for candidate in (
                inflight_seconds,
                suppression_seconds,
                rate_limit_seconds,
            )
            if candidate is not None and candidate > 0
        ]

        if not wait_candidates:
            return None

        return max(wait_candidates)

    async def governance_wait_seconds(
        self,
        *,
        host: str | None,
    ) -> float | None:
        """Return the host governance delay excluding the inflight cap.

        The dispatcher applies the inflight cap separately (event-driven
        instead of polling), so this only covers suppression and rate limit
        waits.
        """
        suppression_seconds = self._suppression_remaining_seconds(host=host)
        rate_limit_seconds = await self._rate_limit_wait_seconds(host=host)

        wait_candidates = [
            candidate
            for candidate in (suppression_seconds, rate_limit_seconds)
            if candidate is not None and candidate > 0
        ]

        if not wait_candidates:
            return None

        return max(wait_candidates)

    def _inflight_wait_seconds(
        self,
        *,
        host: str | None,
        inflight_count_by_host: dict[str, int],
    ) -> float | None:
        if host is None or self._max_inflight_per_host is None:
            return None

        if inflight_count_by_host.get(host, 0) < self._max_inflight_per_host:
            return None

        return self._inflight_host_wait_seconds

    def suppression_remaining_seconds(
        self,
        *,
        host: str | None,
    ) -> float | None:
        """Expose host suppression wait for scheduler pruning decisions."""

        return self._suppression_remaining_seconds(host=host)

    def _suppression_remaining_seconds(
        self,
        *,
        host: str | None,
    ) -> float | None:
        if host is None or self._host_suppression_store is None:
            return None

        remaining_seconds = (
            self._host_suppression_store.get_suppression_remaining_seconds(
                host,
            )
        )

        if remaining_seconds is None or remaining_seconds <= 0:
            return None

        return float(remaining_seconds)

    async def _rate_limit_wait_seconds(
        self,
        *,
        host: str | None,
    ) -> float | None:
        if host is None or self._rate_limiter is None:
            return None

        wait_seconds = await self._rate_limiter.wait_seconds_until_ready(host)

        if wait_seconds is None or wait_seconds <= 0:
            return None

        return float(wait_seconds)
