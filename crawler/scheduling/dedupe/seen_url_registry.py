"""Track normalized URLs already admitted or completed by the scheduler."""

from __future__ import annotations

from collections import OrderedDict
from math import isfinite
from time import time
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable

SeenUrlEntry: TypeAlias = tuple[str, float]


class SeenUrlRegistry:
    """Track requested and canonical URLs seen by the crawler.

    Semantics:
    - The registry has a fixed maximum capacity.
    - When capacity is reached, inserting a new unseen URL evicts the oldest
      tracked URL.
    - When TTL is enabled, entries older than the configured TTL are purged
      lazily before reads and writes.
    - Checkpoint/restore can either preserve original timestamps
      (export_entries/replace_entries) or intentionally reset timestamps
      (export_urls/replace_urls).

    Notes:
    - This is an eviction-based registry, not a hard-cap rejection registry.
      Therefore reaching max capacity does not make a valid new URL
      untrackable; it causes eviction of the oldest entry.
    - Ordering is insertion/refresh order. Re-remembering an existing URL
      moves it to the newest position.
    """

    def __init__(
        self, *, max_seen: int, ttl_seconds: float | None = None
    ) -> None:
        if max_seen <= 0:
            raise ValueError("max_seen must be greater than 0")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than 0 when provided"
            )

        self._max_seen = max_seen
        self._ttl_seconds = ttl_seconds
        self._seen: OrderedDict[str, float] = OrderedDict()

    @property
    def max_seen(self) -> int:
        """Return the configured maximum number of tracked URLs."""
        return self._max_seen

    @property
    def ttl_seconds(self) -> float | None:
        """Return the configured TTL in seconds, if any."""
        return self._ttl_seconds

    @property
    def size(self) -> int:
        """Return the number of currently tracked URLs."""
        self._purge_expired()
        return len(self._seen)

    def is_seen(self, url: str) -> bool:
        """Return whether the given URL is currently tracked."""
        self._purge_expired()
        normalized = self._normalize_url(url)
        return normalized in self._seen if normalized is not None else False

    def remember(self, url: str) -> bool:
        """Track a URL, evicting the oldest entry when capacity is reached.

        Returns:
            True when the URL was newly inserted.
            False when the URL was already present and was only refreshed, or
            when the supplied URL is blank after normalization.
        """
        self._purge_expired()
        normalized = self._normalize_url(url)
        if normalized is None:
            return False

        if normalized in self._seen:
            self._put(normalized, time())
            return False

        self._evict_oldest_if_needed()
        self._put(normalized, time())
        return True

    def remember_equivalent_urls(self, *urls: str) -> None:
        """Track equivalent canonical URLs, evicting oldest entries when needed."""
        self._purge_expired()
        now = time()

        for url in urls:
            normalized = self._normalize_url(url)
            if normalized is None:
                continue

            if normalized not in self._seen:
                self._evict_oldest_if_needed()

            self._put(normalized, now)

    def forget(self, url: str) -> bool:
        """Remove a tracked URL so a later retry/discovery can be admitted."""
        self._purge_expired()
        normalized = self._normalize_url(url)
        if normalized is None:
            return False
        return self._seen.pop(normalized, None) is not None

    def export_entries(self) -> tuple[SeenUrlEntry, ...]:
        """Return a stable timestamp-preserving snapshot.

        This snapshot preserves TTL semantics across restore because each entry
        keeps its original timestamp.
        """
        self._purge_expired()

        entries: list[SeenUrlEntry] = []
        for url, timestamp in self._seen.items():
            entries.append((url, timestamp))

        return tuple(entries)

    def replace_entries(self, entries: Iterable[SeenUrlEntry]) -> int:
        """Replace registry content from a timestamp-preserving snapshot.

        Behavior:
        - Existing state is cleared first.
        - Blank URLs are ignored.
        - Non-finite timestamps are ignored.
        - Duplicate URLs are deduplicated by keeping the last occurrence from
          the input and placing it at the newest position among duplicates.
        - At most ``max_seen`` entries are loaded.
        - After load, expired entries are purged so TTL semantics remain
          correct immediately after restore.

        Returns:
            The number of loaded entries remaining after purge.
        """
        restored: OrderedDict[str, float] = OrderedDict()

        for url, ts in entries:
            normalized = self._normalize_url(url)
            if normalized is None:
                continue

            timestamp = self._coerce_timestamp(ts)
            if timestamp is None:
                continue

            restored.pop(normalized, None)
            restored[normalized] = timestamp

        while len(restored) > self._max_seen:
            restored.popitem(last=False)

        self._seen = restored
        self._purge_expired()
        return len(self._seen)

    def export_urls(self) -> tuple[str, ...]:
        """Return a URL-only snapshot.

        This snapshot intentionally omits timestamps and therefore cannot
        preserve TTL age across restore.
        """
        self._purge_expired()
        return tuple(self._seen.keys())

    def snapshot_urls(self) -> tuple[str, ...]:
        """Return current identities without expiry pruning or mutation.

        Restore validation uses this view so a rejected checkpoint cannot
        change live deduplication state as a side effect of preflight.
        """

        return tuple(self._seen.keys())

    def replace_urls(self, urls: Iterable[str]) -> int:
        """Replace registry content from a URL-only snapshot.

        Behavior:
        - Existing state is cleared first.
        - Restored URLs receive fresh timestamps at restore time.
        - Duplicate URLs are deduplicated by keeping the last occurrence from
          the input and placing it at the newest position among duplicates.
        - At most ``max_seen`` entries are loaded.

        This method should only be used when TTL persistence is intentionally
        not required.

        Returns:
            The number of loaded entries.
        """
        restored: OrderedDict[str, float] = OrderedDict()
        now = time()

        for url in urls:
            normalized = self._normalize_url(url)
            if normalized is None:
                continue

            restored.pop(normalized, None)
            restored[normalized] = now

        while len(restored) > self._max_seen:
            restored.popitem(last=False)

        self._seen = restored
        return len(self._seen)

    def _purge_expired(self) -> None:
        """Remove every expired entry from the registry."""
        if self._ttl_seconds is None:
            return

        expire_before = time() - self._ttl_seconds
        expired = tuple(
            url
            for url, stored_at in self._seen.items()
            if stored_at <= expire_before
        )
        for url in expired:
            self._seen.pop(url, None)

    def _evict_oldest_if_needed(self) -> None:
        """Evict the oldest entry when the registry is at capacity."""
        if len(self._seen) >= self._max_seen:
            self._seen.popitem(last=False)

    def _put(self, url: str, timestamp: float) -> None:
        """Insert or refresh an entry and place it at the newest position."""
        self._seen.pop(url, None)
        self._seen[url] = timestamp

    @staticmethod
    def _normalize_url(url: str) -> str | None:
        """Normalize a URL-like value into a non-empty string."""
        normalized = str(url).strip()
        return normalized or None

    @staticmethod
    def _coerce_timestamp(value: float) -> float | None:
        """Convert a timestamp-like value to a finite float."""
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        return timestamp if isfinite(timestamp) else None
