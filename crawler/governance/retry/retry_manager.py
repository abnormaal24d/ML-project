"""Retry management for crawler fetch operations."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, TypeVar

from crawler.fetching.errors.exceptions import RetryableFetchError
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from config.collection.http_rules import (
        HttpStatusRulesSettings,
        RetryRulesSettings,
    )

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RetryDelay:
    """Computed delay metadata for one retry attempt."""

    seconds: float
    deterministic_seconds: float
    multiplier: float
    jitter_strategy: str


class RetryManager:
    """Run async fetch operations with bounded retry/backoff behavior."""

    def __init__(
        self,
        *,
        settings: RetryRulesSettings,
        status_rules: HttpStatusRulesSettings,
        logger: ProjectLogger,
        random_generator: random.Random,
        total_budget_seconds: float,
        minimum_attempt_seconds: float,
    ) -> None:
        self._settings = settings
        self._status_rules = status_rules
        self._logger = logger
        self._random = random_generator
        self._total_budget_seconds = self._positive_float_or_default(
            total_budget_seconds,
            default=1.0,
        )
        self._minimum_attempt_seconds = self._positive_float_or_default(
            minimum_attempt_seconds,
            default=0.1,
        )

        # Validate jitter strategy early
        valid_strategies = {
            "none",
            "additive",
            "full",
            "equal",
            "decorrelated",
        }
        if self._jitter_strategy not in valid_strategies:
            raise ValueError(
                f"invalid jitter_strategy {self._jitter_strategy!r}; "
                f"expected one of {sorted(valid_strategies)}"
            )

    def is_retryable_status(self, status_code: int) -> bool:
        """Return whether an HTTP status code should be retried."""

        return int(status_code) in self._status_rules.retryable

    # ------------------------------------------------------------------
    # Public retry API
    # ------------------------------------------------------------------
    async def run_with_retry_rules(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        url: str,
        cancel_event: asyncio.Event | None = None,
    ) -> _T:
        """Execute once and reserve an immediate retry only for invalid 304."""

        started_at = asyncio.get_running_loop().time()
        unconditional_exc: RetryableFetchError | None = None
        self._raise_if_cancelled(cancel_event=cancel_event)
        try:
            return await operation()
        except asyncio.CancelledError:
            self._log_cancelled(url=url, fetch_attempt=1)
            raise
        except RetryableFetchError as exc:
            if (
                exc.retry_error_kind != "not_modified_force_unconditional"
                or self._settings.invalid_not_modified_retries < 1
            ):
                self._prepare_scheduler_deferral(
                    exc=exc,
                    attempt_index=0,
                    started_at=started_at,
                )
                raise
            unconditional_exc = exc

        remaining = self._remaining_budget(started_at=started_at)
        if unconditional_exc is None:
            raise RuntimeError("unconditional retry state was not established")
        if remaining < self._minimum_attempt_seconds:
            unconditional_exc.retry_budget_seconds_remaining = remaining
            self._prepare_scheduler_deferral(
                exc=unconditional_exc,
                attempt_index=0,
                started_at=started_at,
            )
            raise unconditional_exc

        self._logger.info(
            "fetch_unconditional_retry_started",
            extra={
                "url_host": self._host_from_url(url),
                "fetch_attempt": 2,
                "retry_error_kind": unconditional_exc.retry_error_kind,
                "remaining_budget_seconds": round(remaining, 4),
            },
        )

        self._raise_if_cancelled(cancel_event=cancel_event)
        try:
            return await operation()
        except asyncio.CancelledError:
            self._log_cancelled(url=url, fetch_attempt=2)
            raise
        except RetryableFetchError as retry_exc:
            retry_exc.unconditional_retry_performed = True
            self._prepare_scheduler_deferral(
                exc=retry_exc,
                attempt_index=1,
                started_at=started_at,
            )
            raise

    # ------------------------------------------------------------------
    # Helper API for tests and delay calculation
    # ------------------------------------------------------------------
    def delay_for_attempt(
        self,
        *,
        attempt_index: int,
        retry_class: str | None = None,
        retry_error_kind: str | None = None,
        previous_delay_seconds: float | None = None,
    ) -> float:
        """Return the retry delay for a zero-based fetch failure attempt."""

        return self._retry_delay_for_attempt(
            attempt_index=attempt_index,
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            previous_delay_seconds=previous_delay_seconds,
        ).seconds

    # ------------------------------------------------------------------
    # Internal delay logic
    # ------------------------------------------------------------------
    def _retry_delay_for_attempt(
        self,
        *,
        attempt_index: int,
        retry_class: str | None = None,
        retry_error_kind: str | None = None,
        previous_delay_seconds: float | None = None,
    ) -> RetryDelay:
        base_delay = self._non_negative_float(
            self._settings.base_delay_seconds
        )
        backoff_multiplier = max(
            1.0,
            self._non_negative_float(self._settings.backoff_multiplier),
        )
        max_delay_seconds = self._non_negative_float(
            self._settings.max_delay_seconds,
        )

        exponential_delay = base_delay * (
            backoff_multiplier ** max(0, int(attempt_index))
        )

        delay_multiplier = self._delay_multiplier(
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
        )

        deterministic_delay = min(
            exponential_delay * delay_multiplier,
            max_delay_seconds,
        )

        jittered_delay = self._apply_jitter(
            deterministic_delay=deterministic_delay,
            delay_multiplier=delay_multiplier,
            previous_delay_seconds=previous_delay_seconds,
        )

        return RetryDelay(
            seconds=jittered_delay,
            deterministic_seconds=deterministic_delay,
            multiplier=delay_multiplier,
            jitter_strategy=self._jitter_strategy,
        )

    def _delay_multiplier(
        self,
        *,
        retry_class: str | None,
        retry_error_kind: str | None,
    ) -> float:
        retry_error_kind_multipliers = (
            self._settings.retry_error_kind_delay_multipliers
        )
        normalized_retry_error_kind = self._normalize_retry_key(
            retry_error_kind
        )

        if normalized_retry_error_kind in retry_error_kind_multipliers:
            return self._positive_float_or_default(
                retry_error_kind_multipliers[normalized_retry_error_kind],
                default=1.0,
            )

        retry_class_multipliers = self._settings.retry_class_delay_multipliers
        normalized_retry_class = self._normalize_retry_key(retry_class)

        if normalized_retry_class in retry_class_multipliers:
            return self._positive_float_or_default(
                retry_class_multipliers[normalized_retry_class],
                default=1.0,
            )

        return 1.0

    def _apply_jitter(
        self,
        *,
        deterministic_delay: float,
        delay_multiplier: float,
        previous_delay_seconds: float | None,
    ) -> float:
        max_delay_seconds = self._non_negative_float(
            self._settings.max_delay_seconds,
        )
        deterministic_delay = self._non_negative_float(deterministic_delay)

        if deterministic_delay <= 0.0 or max_delay_seconds <= 0.0:
            return 0.0

        strategy = self._jitter_strategy

        if strategy == "none":
            return min(deterministic_delay, max_delay_seconds)

        if strategy == "additive":
            jitter_ratio = self._non_negative_float(
                self._settings.jitter_ratio
            )
            if jitter_ratio <= 0.0:
                return min(deterministic_delay, max_delay_seconds)

            jitter = deterministic_delay * jitter_ratio
            return min(
                deterministic_delay + self._random.uniform(0.0, jitter),
                max_delay_seconds,
            )

        if strategy == "full":
            return self._random.uniform(0.0, deterministic_delay)

        if strategy == "equal":
            half_delay = deterministic_delay / 2.0
            return half_delay + self._random.uniform(0.0, half_delay)

        if strategy == "decorrelated":
            floor_delay = min(
                self._non_negative_float(self._settings.base_delay_seconds)
                * max(1.0, delay_multiplier),
                max_delay_seconds,
            )
            previous_delay = (
                previous_delay_seconds
                if previous_delay_seconds is not None
                else floor_delay
            )
            previous_delay = max(
                floor_delay, self._non_negative_float(previous_delay)
            )
            upper_bound = max(floor_delay, previous_delay * 3.0)

            return min(
                self._random.uniform(floor_delay, upper_bound),
                max_delay_seconds,
            )

        return min(deterministic_delay, max_delay_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_scheduler_deferral(
        self,
        *,
        exc: RetryableFetchError,
        attempt_index: int,
        started_at: float,
    ) -> None:
        delay = self._retry_delay_for_attempt(
            attempt_index=attempt_index,
            retry_class=exc.retry_class,
            retry_error_kind=exc.retry_error_kind,
        )
        server_delay = self._non_negative_float(exc.retry_after_seconds)
        requested_delay = min(
            max(delay.seconds, server_delay),
            self._non_negative_float(self._settings.max_delay_seconds),
        )
        remaining = self._remaining_budget(started_at=started_at)
        exc.retry_after_seconds = min(requested_delay, remaining)
        exc.retry_budget_seconds_remaining = remaining

    def _remaining_budget(self, *, started_at: float) -> float:
        elapsed = max(0.0, asyncio.get_running_loop().time() - started_at)
        return max(0.0, self._total_budget_seconds - elapsed)

    def _log_cancelled(self, *, url: str, fetch_attempt: int) -> None:
        self._logger.debug(
            "fetch_retry_cancelled",
            extra={
                "url_host": self._host_from_url(url),
                "fetch_attempt": fetch_attempt,
                "retry_in_progress": False,
            },
        )

    @staticmethod
    def _raise_if_cancelled(
        *,
        cancel_event: asyncio.Event | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("retry cancelled")

    @property
    def _jitter_strategy(self) -> str:
        return str(self._settings.jitter_strategy).strip().lower()

    @staticmethod
    def _normalize_retry_key(value: object) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip().lower()
        return normalized or None

    @staticmethod
    def _host_from_url(url: str) -> str:
        """Extract host for logging without exposing full URL or query."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc or parsed.hostname or "unknown"
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):  # exception-rules: best-effort-cleanup
            return "unknown"

    @staticmethod
    def _non_negative_float(value: object) -> float:
        if isinstance(value, bool):
            return 0.0
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not isfinite(number):
            return 0.0

        return max(0.0, number)

    @staticmethod
    def _positive_float_or_default(value: object, *, default: float) -> float:
        if isinstance(value, bool):
            return default
        if not isinstance(value, (str, bytes, bytearray, int, float)):
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        if not isfinite(number) or number <= 0.0:
            return default

        return number
