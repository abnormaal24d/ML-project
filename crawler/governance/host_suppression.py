"""Host suppression tracking based on consecutive forbidden responses."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Callable

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(slots=True)
class _HostSuppressionState:
    """Mutable suppression state for one host."""

    consecutive_forbidden_count: int = 0
    suppressed_until_monotonic: float = 0.0
    expires_at_monotonic: float = 0.0


class HostSuppressionStore:
    """Bounded TTL/LRU store for host-level suppression after repeated 403s."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_size: int,
        suppress_after_forbidden_responses: int = 3,
        forbidden_host_cooldown_seconds: float = 300.0,
        host_normalizer: HostNormalizer,
        monotonic_seconds: Callable[[], float],
        logger: ProjectLogger,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if suppress_after_forbidden_responses <= 0:
            raise ValueError("suppress_after_forbidden_responses must be > 0")
        if forbidden_host_cooldown_seconds < 0:
            raise ValueError("forbidden_host_cooldown_seconds must be >= 0")

        self._ttl_seconds = float(ttl_seconds)
        self._max_size = int(max_size)
        self._suppress_after_forbidden_responses = int(
            suppress_after_forbidden_responses
        )
        self._forbidden_host_cooldown_seconds = float(
            forbidden_host_cooldown_seconds
        )
        self._logger = logger
        self._host_normalizer = host_normalizer
        self._monotonic_seconds = monotonic_seconds
        self._entries: OrderedDict[str, _HostSuppressionState] = OrderedDict()
        self._lock = RLock()

    def get_suppression_remaining_seconds(self, host: str) -> float | None:
        normalized_host = self._normalize_host(host)
        if not normalized_host:
            return None

        with self._lock:
            self._purge_expired()

            entry = self._entries.get(normalized_host)
            if entry is None:
                return None

            now_monotonic = self._monotonic_seconds()
            self._expire_suppression_if_needed(
                entry=entry,
                now_monotonic=now_monotonic,
            )

            if entry.suppressed_until_monotonic <= now_monotonic:
                self._touch(normalized_host, entry)
                return None

            remaining_seconds = (
                entry.suppressed_until_monotonic - now_monotonic
            )
            self._touch(normalized_host, entry)
            return remaining_seconds

    def record_response_status(
        self,
        *,
        host: str,
        status_code: int,
    ) -> None:
        normalized_host = self._normalize_host(host)
        if not normalized_host:
            return

        with self._lock:
            entry = self._get_or_create(normalized_host)
            now_monotonic = self._monotonic_seconds()
            self._expire_suppression_if_needed(
                entry=entry,
                now_monotonic=now_monotonic,
            )

            if status_code == 403:
                entry.consecutive_forbidden_count += 1
                self._suppress_host_if_threshold_reached(
                    host=normalized_host,
                    entry=entry,
                    now_monotonic=now_monotonic,
                )
                self._touch(normalized_host, entry)
                return

            if 200 <= status_code < 300:
                entry.consecutive_forbidden_count = 0
                self._clear_suppression(entry=entry)
                self._touch(normalized_host, entry)
                return

            self._touch(normalized_host, entry)

    def _suppress_host_if_threshold_reached(
        self,
        *,
        host: str,
        entry: _HostSuppressionState,
        now_monotonic: float,
    ) -> None:
        if self._forbidden_host_cooldown_seconds <= 0:
            return
        if (
            entry.consecutive_forbidden_count
            < self._suppress_after_forbidden_responses
        ):
            return

        was_suppressed = entry.suppressed_until_monotonic > now_monotonic
        entry.suppressed_until_monotonic = (
            now_monotonic + self._forbidden_host_cooldown_seconds
        )
        if was_suppressed:
            return

        self._logger.warning(
            "host_temporarily_suppressed",
            host=host,
            cooldown_seconds=self._forbidden_host_cooldown_seconds,
            consecutive_forbidden_count=entry.consecutive_forbidden_count,
            suppression_threshold=self._suppress_after_forbidden_responses,
        )

    def _clear_suppression(self, *, entry: _HostSuppressionState) -> None:
        if entry.suppressed_until_monotonic <= 0.0:
            return
        entry.suppressed_until_monotonic = 0.0

    def _expire_suppression_if_needed(
        self,
        *,
        entry: _HostSuppressionState,
        now_monotonic: float,
    ) -> None:
        if entry.suppressed_until_monotonic <= 0.0:
            return
        if entry.suppressed_until_monotonic > now_monotonic:
            return
        entry.suppressed_until_monotonic = 0.0
        entry.consecutive_forbidden_count = 0

    def _get_or_create(self, host: str) -> _HostSuppressionState:
        self._purge_expired()
        entry = self._entries.get(host)
        if entry is not None:
            self._touch(host, entry)
            return entry
        entry = _HostSuppressionState()
        self._entries[host] = entry
        self._touch(host, entry)
        self._evict_if_needed()
        return entry

    def _touch(self, host: str, entry: _HostSuppressionState) -> None:
        entry.expires_at_monotonic = (
            self._monotonic_seconds() + self._ttl_seconds
        )
        self._entries[host] = entry
        self._entries.move_to_end(host)

    def _purge_expired(self) -> None:
        now = self._monotonic_seconds()
        expired = [
            host
            for host, entry in self._entries.items()
            if entry.expires_at_monotonic <= now
        ]
        for host in expired:
            self._entries.pop(host, None)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def _normalize_host(self, host: str) -> str:
        return self._host_normalizer.normalize(host) or ""
