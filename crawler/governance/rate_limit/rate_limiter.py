"""Public models and helpers for crawler.governance.rate_limit.rate_limiter.

Exports: RateLimiter.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from crawler.runtime.concurrency import owned_lock
from crawler.worker.activity.worker_activity import waiting_phase
from logger.project_logger import ProjectLogger


class RateLimiterNotReadyError(RuntimeError):
    """Signal that host pacing requires deferred dispatch instead of waiting."""

    def __init__(self, *, host: str, wait_seconds: float) -> None:
        self.host = host
        self.wait_seconds = max(0.0, float(wait_seconds))
        super().__init__(
            f"host {host} is not ready for dispatch for "
            f"{self.wait_seconds:.4f} seconds"
        )


if TYPE_CHECKING:
    from crawler.governance.rate_limit.rate_limit_host_state import (
        RateLimitHostState,
    )
    from crawler.governance.rate_limit.rate_limit_rules import RateLimitRules
    from crawler.governance.rate_limit.rate_limit_slot_scheduler import (
        RateLimitSlotScheduler,
    )
    from crawler.governance.rate_limit.rate_limit_state_registry import (
        RateLimitStateRegistry,
    )


class RateLimiter:
    """Apply per-host request pacing without cross-host lock contention."""

    def __init__(
        self,
        *,
        state_registry: RateLimitStateRegistry,
        slot_scheduler: RateLimitSlotScheduler,
        adaptive_rules: RateLimitRules,
        default_effective_requests_per_second: float,
        honor_retry_after: bool,
        max_retry_after_seconds: float,
        sleep_log_threshold_seconds: float = 1.0,
        logger: ProjectLogger,
    ) -> None:
        self._logger = logger
        self._sleep_log_threshold_seconds = sleep_log_threshold_seconds
        self._state_registry = state_registry
        self._slot_scheduler = slot_scheduler
        self._adaptive_rules = adaptive_rules
        self._default_effective_requests_per_second = (
            default_effective_requests_per_second
        )
        self._honor_retry_after = bool(honor_retry_after)
        self._max_retry_after_seconds = (
            self._require_nonnegative_finite_seconds(
                max_retry_after_seconds,
                field_name="max_retry_after_seconds",
            )
        )
        self._pacing_wait_count = 0
        self._pacing_wait_seconds_total = 0.0

    # ------------------------------------------------------------------
    # Public acquisition API
    # ------------------------------------------------------------------
    @property
    def pacing_wait_count(self) -> int:
        """Return how many acquisitions had to wait for a host slot."""
        return self._pacing_wait_count

    @property
    def pacing_wait_seconds_total(self) -> float:
        """Return total seconds spent waiting for host pacing slots."""
        return self._pacing_wait_seconds_total

    async def acquire(self, host: str | None) -> None:
        """Wait until the next request for the host is allowed."""

        host_key, state = self._state_registry.state_for_host(host)
        reserved_at: float | None = None
        reserved_rules_revision: int | None = None
        previous_next_request_not_before = 0.0
        applied_next_request_not_before: float | None = None

        while True:
            async with owned_lock(state.lock):
                now = self._now()
                crawl_delay_seconds = (
                    self._adaptive_rules.effective_crawl_delay_seconds(state)
                )

                if (
                    reserved_at is not None
                    and reserved_rules_revision != state.rules_revision
                ):
                    self._slot_scheduler.cancel_slot(
                        state=state,
                        reserved_at=reserved_at,
                        previous_next_request_not_before=(
                            previous_next_request_not_before
                        ),
                        applied_next_request_not_before=(
                            applied_next_request_not_before
                        ),
                    )
                    reserved_at = None
                    reserved_rules_revision = None
                    applied_next_request_not_before = None

                if reserved_at is None:
                    previous_next_request_not_before = (
                        state.next_request_not_before
                    )
                    reserved_at = self._slot_scheduler.reserve_slot(
                        host=host_key,
                        state=state,
                        now=now,
                        crawl_delay_seconds=crawl_delay_seconds,
                    )
                    reserved_rules_revision = state.rules_revision
                    applied_next_request_not_before = (
                        state.next_request_not_before
                        if (
                            abs(
                                state.next_request_not_before
                                - previous_next_request_not_before
                            )
                            >= 1e-9
                        )
                        else None
                    )

                wait_until = max(reserved_at, state.cooldown_until)
                sleep_for = wait_until - now
                if sleep_for <= 0:
                    return
                should_log_sleep = (
                    sleep_for >= self._sleep_log_threshold_seconds
                    or state.cooldown_until > now
                    or crawl_delay_seconds >= self._sleep_log_threshold_seconds
                )

            if should_log_sleep and self._logger is not None:
                self._logger.debug(
                    "rate_limiter_sleep",
                    extra={
                        "host": host_key,
                        "seconds": round(sleep_for, 4),
                    },
                )
            self._pacing_wait_count += 1
            self._pacing_wait_seconds_total += sleep_for
            with waiting_phase():
                await asyncio.sleep(sleep_for)

    async def wait_seconds_until_ready(
        self,
        host: str | None,
    ) -> float | None:
        """Return remaining host pacing delay without consuming a slot."""

        state = self._state_registry.get(host)
        if state is None:
            return None

        async with owned_lock(state.lock):
            now = self._now()
            reserved_at = self._slot_scheduler.preview_slot(
                state=state,
                now=now,
            )
            wait_seconds = max(0.0, reserved_at - now)

        return wait_seconds if wait_seconds > 0 else None

    async def acquire_or_raise_not_ready(self, host: str | None) -> None:
        """Consume a ready slot or ask the caller to defer dispatch."""

        host_key, state = self._state_registry.state_for_host(host)

        async with owned_lock(state.lock):
            now = self._now()
            reserved_at = self._slot_scheduler.preview_slot(
                state=state,
                now=now,
            )
            wait_seconds = max(0.0, reserved_at - now)
            if wait_seconds > 0:
                raise RateLimiterNotReadyError(
                    host=host_key,
                    wait_seconds=wait_seconds,
                )

            self._slot_scheduler.commit_slot(
                host=host_key,
                state=state,
                reserved_at=reserved_at,
                now=now,
                crawl_delay_seconds=(
                    self._adaptive_rules.effective_crawl_delay_seconds(state)
                ),
            )

    async def acquire_for_fetch(
        self,
        *,
        host: str | None,
        defer_if_rate_limited: bool,
    ) -> None:
        """Acquire a rate-limit slot for fetch execution."""

        if defer_if_rate_limited:
            await self.acquire_or_raise_not_ready(host)
            return

        await self.acquire(host)

    # ------------------------------------------------------------------
    # Result feedback
    # ------------------------------------------------------------------
    async def report_result(
        self,
        *,
        host: str | None,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Adapt host request rate based on response multimodal."""

        host_key, state = self._state_registry.state_for_host(host)
        async with owned_lock(state.lock):
            before = self._rules_snapshot(state)
            self._adaptive_rules.report_result(
                host=host_key,
                state=state,
                status_code=status_code,
                latency_seconds=latency_seconds,
                now=self._now(),
            )
            self._bump_rules_revision_if_changed(state=state, before=before)

    # ------------------------------------------------------------------
    # Rules and state updates
    # ------------------------------------------------------------------
    async def set_host_crawl_delay(
        self,
        *,
        host: str | None,
        crawl_delay_seconds: float | None,
    ) -> None:
        """Set or clear a hard crawl-delay override for a host."""

        host_key, state = self._state_registry.state_for_host(host)
        async with owned_lock(state.lock):
            before = self._rules_snapshot(state)
            self._adaptive_rules.set_host_crawl_delay(
                host=host_key,
                state=state,
                crawl_delay_seconds=crawl_delay_seconds,
            )
            self._bump_rules_revision_if_changed(state=state, before=before)

    # ------------------------------------------------------------------
    # Response hint handling
    # ------------------------------------------------------------------
    async def apply_response_rate_limit_hints(
        self,
        *,
        host: str | None,
        retry_after_seconds: float | None = None,
        rate_limit_remaining: int | None = None,
        rate_limit_reset_seconds: float | None = None,
    ) -> float | None:
        """Apply server supplied pacing hints such as Retry-After.

        Parameters:
            host: Host whose request budget should be cooled down.
            retry_after_seconds: Delay derived from the Retry-After header.
            rate_limit_remaining: Remaining request count from rate-limit
                headers. A zero value activates the reset delay when present.
            rate_limit_reset_seconds: Delay until the rate-limit window resets.
        """

        cooldown_seconds = self._cooldown_seconds_from_hints(
            retry_after_seconds=retry_after_seconds,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset_seconds=rate_limit_reset_seconds,
        )
        if cooldown_seconds is None:
            return None

        host_key, state = self._state_registry.state_for_host(host)
        async with owned_lock(state.lock):
            now = self._now()
            before = self._rules_snapshot(state)
            state.cooldown_until = max(
                state.cooldown_until,
                now + cooldown_seconds,
            )
            self._bump_rules_revision_if_changed(state=state, before=before)
            self._logger.info(
                "rate_limiter_response_hint_applied",
                extra={
                    "host": host_key,
                    "cooldown_seconds": round(cooldown_seconds, 4),
                    "retry_after_seconds": retry_after_seconds,
                    "rate_limit_remaining": rate_limit_remaining,
                    "rate_limit_reset_seconds": rate_limit_reset_seconds,
                },
            )
        return cooldown_seconds

    async def set_host_requests_per_second(
        self,
        *,
        host: str | None,
        requests_per_second: float,
    ) -> None:
        """Set the current adaptive host request rate budget explicitly."""

        host_key, state = self._state_registry.state_for_host(host)
        async with owned_lock(state.lock):
            before = self._rules_snapshot(state)
            self._adaptive_rules.set_host_requests_per_second(
                host=host_key,
                state=state,
                requests_per_second=requests_per_second,
            )
            self._bump_rules_revision_if_changed(state=state, before=before)

    def host_requests_per_second(self, host: str | None) -> float:
        """Return the effective requests/sec budget for the host."""

        state = self._state_registry.get(host)
        if state is None:
            return self._default_effective_requests_per_second
        return state.effective_requests_per_second

    def host_crawl_delay_seconds(self, host: str | None) -> float | None:
        """Return the effective hard crawl delay for the host, if any."""

        state = self._state_registry.get(host)
        if state is None:
            crawl_delay_seconds = 0.0
        else:
            crawl_delay_seconds = (
                self._adaptive_rules.effective_crawl_delay_seconds(state)
            )
        return crawl_delay_seconds if crawl_delay_seconds > 0 else None

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    @staticmethod
    def _rules_snapshot(
        state: RateLimitHostState,
    ) -> tuple[float, float, float | None, float, float]:
        return (
            state.adaptive_requests_per_second,
            state.effective_requests_per_second,
            state.crawl_delay_override_seconds,
            state.cooldown_until,
            state.next_request_not_before,
        )

    @staticmethod
    def _bump_rules_revision_if_changed(
        *,
        state: RateLimitHostState,
        before: tuple[float, float, float | None, float, float],
    ) -> None:
        after = RateLimiter._rules_snapshot(state)
        if after != before:
            state.rules_revision += 1

    @staticmethod
    def clamp_initial_requests_per_second(
        *,
        value: float,
        min_requests_per_second_value: float,
        max_requests_per_second_value: float,
    ) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if numeric_value <= 0:
            return 0.0

        return max(
            min_requests_per_second_value,
            min(max_requests_per_second_value, numeric_value),
        )

    def _cooldown_seconds_from_hints(
        self,
        *,
        retry_after_seconds: float | None,
        rate_limit_remaining: int | None,
        rate_limit_reset_seconds: float | None,
    ) -> float | None:
        """Return the cooldown delay requested by rate-limit headers."""

        candidates: list[float] = []
        if self._honor_retry_after:
            retry_after_delay = self._bounded_server_delay_seconds(
                retry_after_seconds
            )
            if retry_after_delay is not None:
                candidates.append(retry_after_delay)

        if rate_limit_remaining == 0:
            reset_delay = self._bounded_server_delay_seconds(
                rate_limit_reset_seconds
            )
            if reset_delay is not None:
                candidates.append(reset_delay)

        if not candidates:
            return None
        return max(candidates)

    def _bounded_server_delay_seconds(
        self,
        value: float | None,
    ) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            delay_seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(delay_seconds) or delay_seconds <= 0:
            return None
        bounded_delay = min(delay_seconds, self._max_retry_after_seconds)
        return bounded_delay if bounded_delay > 0 else None

    @staticmethod
    def _require_nonnegative_finite_seconds(
        value: float,
        *,
        field_name: str,
    ) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be finite and nonnegative"
            ) from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"{field_name} must be finite and nonnegative")
        return seconds
