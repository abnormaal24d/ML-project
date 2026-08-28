"""
Public models and helpers for
crawler.governance.rate_limit.rate_limit_slot_scheduler.

Exports: RateLimitSlotScheduler.
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.governance.rate_limit.rate_limit_host_state import (
        RateLimitHostState,
    )


class RateLimitSlotScheduler:
    """Preview, reserve, and commit host slots for paced dispatch."""

    def __init__(
        self,
        *,
        burst_size: int,
        logger: ProjectLogger,
        reservation_log_threshold_seconds: float,
        random_delay_min_seconds: float = 0.0,
        random_delay_max_seconds: float = 0.0,
    ) -> None:
        self._burst_size = burst_size
        self._logger = logger
        self._reservation_log_threshold_seconds = (
            reservation_log_threshold_seconds
        )
        self._random_delay_min_seconds = max(
            0.0,
            float(random_delay_min_seconds),
        )
        self._random_delay_max_seconds = max(
            self._random_delay_min_seconds,
            float(random_delay_max_seconds),
        )

    # ------------------------------------------------------------------
    # Slot preview and reservation
    # ------------------------------------------------------------------
    def preview_slot(
        self,
        *,
        state: RateLimitHostState,
        now: float,
    ) -> float:
        """Return the earliest allowed request time without mutating state."""

        reserved_at = max(
            now,
            state.cooldown_until,
            state.next_request_not_before,
        )
        effective_requests_per_second = state.effective_requests_per_second
        if effective_requests_per_second <= 0:
            return reserved_at

        interval = 1.0 / effective_requests_per_second
        bucket = deque(state.timestamps)

        while True:
            prune_before = reserved_at - interval
            while bucket and bucket[0] <= prune_before:
                bucket.popleft()

            if len(bucket) < self._burst_size:
                return reserved_at

            reserved_at = max(reserved_at, bucket[0] + interval)

    def reserve_slot(
        self,
        *,
        host: str,
        state: RateLimitHostState,
        now: float,
        crawl_delay_seconds: float,
    ) -> float:
        """Reserve and persist the earliest allowed slot for a host."""

        reserved_at = self.preview_slot(state=state, now=now)
        self.commit_slot(
            host=host,
            state=state,
            reserved_at=reserved_at,
            now=now,
            crawl_delay_seconds=crawl_delay_seconds,
        )
        return reserved_at

    # ------------------------------------------------------------------
    # Commit and cancel
    # ------------------------------------------------------------------
    def commit_slot(
        self,
        *,
        host: str,
        state: RateLimitHostState,
        reserved_at: float,
        now: float,
        crawl_delay_seconds: float,
    ) -> None:
        """Persist a previously computed host slot reservation."""

        effective_requests_per_second = state.effective_requests_per_second
        bucket = state.timestamps

        if effective_requests_per_second > 0:
            interval = 1.0 / effective_requests_per_second
            prune_before = reserved_at - interval
            while bucket and bucket[0] <= prune_before:
                bucket.popleft()
            bucket.append(reserved_at)

        random_delay_seconds = self._random_delay_seconds()
        next_request_delay_seconds = max(
            crawl_delay_seconds,
            random_delay_seconds,
        )
        if next_request_delay_seconds > 0:
            state.next_request_not_before = (
                reserved_at + next_request_delay_seconds
            )

        throttle_delay = max(0.0, reserved_at - now)
        if (
            throttle_delay >= self._reservation_log_threshold_seconds
            or state.cooldown_until > now
            or crawl_delay_seconds >= self._reservation_log_threshold_seconds
        ):
            self._logger.debug(
                "rate_limiter_slot_reserved",
                extra={
                    "host": host,
                    "reserved_at": round(reserved_at, 6),
                    "cooldown_until": round(state.cooldown_until, 6),
                    "crawl_delay_seconds": (
                        round(crawl_delay_seconds, 6)
                        if crawl_delay_seconds and crawl_delay_seconds > 0
                        else None
                    ),
                    "random_delay_seconds": (
                        round(random_delay_seconds, 6)
                        if random_delay_seconds and random_delay_seconds > 0
                        else None
                    ),
                    "adaptive_requests_per_second": round(
                        state.adaptive_requests_per_second,
                        6,
                    ),
                    "effective_requests_per_second": round(
                        effective_requests_per_second,
                        6,
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Cancel logic
    # ------------------------------------------------------------------
    def cancel_slot(
        self,
        *,
        state: RateLimitHostState,
        reserved_at: float,
        previous_next_request_not_before: float,
        applied_next_request_not_before: float | None,
    ) -> None:
        """Remove a pending reservation that became stale before use."""

        bucket = state.timestamps
        for index, timestamp in enumerate(bucket):
            if abs(timestamp - reserved_at) < 1e-9:
                del bucket[index]
                break

        if (
            applied_next_request_not_before is not None
            and abs(
                state.next_request_not_before - applied_next_request_not_before
            )
            < 1e-9
        ):
            state.next_request_not_before = previous_next_request_not_before

    def _random_delay_seconds(self) -> float:
        if self._random_delay_max_seconds <= 0.0:
            return 0.0
        if self._random_delay_max_seconds == self._random_delay_min_seconds:
            return self._random_delay_max_seconds
        return random.uniform(  # nosec B311
            self._random_delay_min_seconds,
            self._random_delay_max_seconds,
        )
