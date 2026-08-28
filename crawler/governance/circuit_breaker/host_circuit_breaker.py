"""Bounded per-host circuit breaker for repeated network failures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(frozen=True, slots=True)
class CircuitDecision:
    """Admission result for one host request."""

    allowed: bool
    retry_after_seconds: float | None = None
    half_open_trial: bool = False


@dataclass(slots=True)
class _HostCircuitState:
    failures: Counter[str] = field(default_factory=Counter)
    opened_until: float = 0.0
    half_open_in_flight: bool = False


class HostCircuitBreaker:
    """Open a host circuit after repeated classified network failures."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        monotonic_seconds: Callable[[], float],
        host_normalizer: HostNormalizer,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least one")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        self._failure_threshold = int(failure_threshold)
        self._cooldown_seconds = float(cooldown_seconds)
        self._monotonic_seconds = monotonic_seconds
        self._host_normalizer = host_normalizer
        self._states: dict[str, _HostCircuitState] = {}
        self._lock = RLock()

    def before_request(self, *, host: str) -> CircuitDecision:
        """Admit closed circuits and one trial after an open cooldown."""

        key = self._host_normalizer.require(host)
        with self._lock:
            state = self._states.get(key)
            if state is None or state.opened_until <= 0:
                return CircuitDecision(allowed=True)
            now = self._monotonic_seconds()
            if now < state.opened_until:
                return CircuitDecision(
                    allowed=False,
                    retry_after_seconds=state.opened_until - now,
                )
            if state.half_open_in_flight:
                return CircuitDecision(
                    allowed=False,
                    retry_after_seconds=self._cooldown_seconds,
                )
            state.half_open_in_flight = True
            return CircuitDecision(allowed=True, half_open_trial=True)

    def record_success(self, *, host: str) -> None:
        """Close and clear the circuit after a successful response."""

        with self._lock:
            self._states.pop(self._host_normalizer.require(host), None)

    def record_failure(self, *, host: str, category: str) -> None:
        """Count a classified failure and open the circuit when required."""

        key = self._host_normalizer.require(host)
        failure_category = category.strip().lower()
        if not failure_category:
            raise ValueError("circuit failure category must not be empty")
        with self._lock:
            state = self._states.setdefault(key, _HostCircuitState())
            state.failures[failure_category] += 1
            should_open = (
                state.half_open_in_flight
                or state.failures[failure_category] >= self._failure_threshold
            )
            state.half_open_in_flight = False
            if should_open:
                state.opened_until = (
                    self._monotonic_seconds() + self._cooldown_seconds
                )

    def snapshot(self, *, host: str) -> dict[str, object]:
        """Return metrics-safe state for observability and tests."""

        with self._lock:
            state = self._states.get(self._host_normalizer.require(host))
            if state is None:
                return {
                    "failures": {},
                    "opened_until": 0.0,
                    "half_open_in_flight": False,
                }
            return {
                "failures": dict(state.failures),
                "opened_until": state.opened_until,
                "half_open_in_flight": state.half_open_in_flight,
            }
