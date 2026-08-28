"""Cumulative worker task counters and duration aggregates."""

from __future__ import annotations

import asyncio

from crawler.fetching.errors.exceptions import RetryableFetchError
from crawler.numeric import coerce_finite_float


class WorkerTaskCounters:
    """Record cumulative task multimodal without tracking live worker state."""

    def __init__(self) -> None:
        self._completed_tasks = 0
        self._timed_tasks = 0
        self._total_processing_seconds = 0.0
        self._failure_count = 0
        self._non_fatal_timeout_count = 0
        self._retry_exhausted_count = 0
        self._root_seeds_total = 0
        self._root_seeds_succeeded = 0
        self._root_seeds_transient_failed = 0
        self._root_seeds_governance_blocked = 0

    @property
    def completed_task_count(self) -> int:
        return self._completed_tasks

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def non_fatal_timeout_count(self) -> int:
        return self._non_fatal_timeout_count

    @property
    def retry_exhausted_count(self) -> int:
        return self._retry_exhausted_count

    @property
    def root_seeds_total(self) -> int:
        return self._root_seeds_total

    @property
    def root_seeds_succeeded(self) -> int:
        return self._root_seeds_succeeded

    @property
    def root_seeds_transient_failed(self) -> int:
        return self._root_seeds_transient_failed

    @property
    def root_seeds_governance_blocked(self) -> int:
        return self._root_seeds_governance_blocked

    @property
    def average_processing_seconds(self) -> float:
        if self._timed_tasks == 0:
            return 0.0
        return self._total_processing_seconds / self._timed_tasks

    def record_task_completed(
        self,
        *,
        processing_seconds: float,
        outcome: str | None = None,
    ) -> float:
        """Record one completed attempt and return its rounded duration."""

        normalized_duration = coerce_finite_float(
            processing_seconds,
            default=0.0,
            minimum=0.0,
        )
        self._completed_tasks += 1
        if self.should_include_in_processing_average(outcome=outcome):
            self._timed_tasks += 1
            self._total_processing_seconds += normalized_duration
        return round(normalized_duration, 3)

    def register_failure(
        self,
        *,
        cause: BaseException,
        fatal: bool,
    ) -> int:
        """Record one failure and return the cumulative failure count."""

        self._failure_count += 1
        if not fatal and isinstance(
            cause, (asyncio.TimeoutError, TimeoutError)
        ):
            self._non_fatal_timeout_count += 1
        if not fatal and isinstance(cause, RetryableFetchError):
            self._retry_exhausted_count += 1
        return self._failure_count

    @staticmethod
    def should_include_in_processing_average(
        *,
        outcome: str | None,
    ) -> bool:
        """Return whether an outcome represents meaningful processing work."""

        return outcome not in {"cancelled", "deferred", "interrupted"}

    def record_root_seed(
        self,
        *,
        outcome: str,
    ) -> None:
        """Record a root-seed task outcome for final statistics."""

        self._root_seeds_total += 1
        if outcome == "success":
            self._root_seeds_succeeded += 1
        elif outcome in {"failure", "timeout"}:
            self._root_seeds_transient_failed += 1
        elif outcome in {"dropped", "deferred"}:
            self._root_seeds_governance_blocked += 1
