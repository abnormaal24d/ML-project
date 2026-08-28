"""Store bounded host advice used by scheduler admission decisions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING

from .host_advice import HostAdvice

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(frozen=True, slots=True)
class HostAdviceCacheEntry:
    """Cached host scheduling advice with expiration metadata."""

    advice: HostAdvice
    touched_at: float


class HostAdviceTracker:
    """Store bounded host advice used by scheduler admission decisions."""

    def __init__(
        self,
        *,
        ttl_seconds: float | None,
        max_hosts: int,
        host_normalizer: HostNormalizer,
    ) -> None:
        """Initialize the tracker with a TTL and maximum capacity."""
        self._ttl_seconds = ttl_seconds
        self._max_hosts = max_hosts
        self._host_normalizer = host_normalizer
        self._entries: OrderedDict[str, HostAdviceCacheEntry] = OrderedDict()

    def clear(self) -> None:
        """Remove all entries from the tracker."""
        self._entries.clear()

    def get(self, host: str) -> HostAdviceCacheEntry | None:
        """Retrieve advice for a specific host if it exists."""
        normalized = self._host_normalizer.normalize(host)
        return (
            self._entries.get(normalized) if normalized is not None else None
        )

    def remember(self, *, host: str, advice: HostAdvice) -> None:
        """Add or update host advice in the cache, maintaining size bounds."""
        if self._max_hosts == 0:
            return

        host = self._host_normalizer.require(host)

        now = monotonic()
        self.prune(now=now)

        # Verwijder bestaande host om deze opnieuw achteraan te voegen
        # (LRU/FIFO hybride)
        self._entries.pop(host, None)
        self._entries[host] = HostAdviceCacheEntry(
            advice=advice,
            touched_at=now,
        )

        while len(self._entries) > self._max_hosts:
            self._entries.popitem(last=False)

    def prune(self, *, now: float | None = None) -> int:
        """Remove expired entries and enforce the maximum host limit."""
        if not self._entries:
            return 0

        removed = 0
        current = monotonic() if now is None else now

        if self._ttl_seconds is not None:
            expire_before = current - self._ttl_seconds

            # Omdat we een OrderedDict gebruiken, zijn de oudste entries
            # vooraan.
            # We kunnen stoppen zodra we een entry vinden die nog geldig is.
            while self._entries:
                host, entry = next(iter(self._entries.items()))
                if entry.touched_at > expire_before:
                    break
                self._entries.pop(host)
                removed += 1

        while len(self._entries) > self._max_hosts:
            self._entries.popitem(last=False)
            removed += 1

        return removed

    @staticmethod
    def extract_crawl_delay_seconds(advice: HostAdvice) -> float | None:
        """Parse and validate the crawl delay from raw host advice."""
        raw_value = advice.crawl_delay_seconds
        if raw_value is None:
            return None

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        if not isfinite(value) or value < 0:
            return None

        return value
