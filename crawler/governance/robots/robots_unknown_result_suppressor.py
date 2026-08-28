"""Suppress duplicate UNKNOWN robots multimodal within a TTL window."""

from __future__ import annotations

from time import monotonic
from urllib.parse import urlsplit

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_check_result import (
    RobotsCheckResult,
    RobotsConfidence,
    RobotsDecision,
)


class RobotsUnknownResultSuppressor:
    """Suppress duplicate UNKNOWN robots multimodal within a TTL window."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        prune_every: int,
        max_entries: int,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._prune_every = max(1, int(prune_every))
        self._max_entries = max(1, int(max_entries))
        self._host_normalizer = host_normalizer
        self._recent_result_insertions = 0
        self._recent_result_keys: dict[str, float] = {}

    def should_suppress_duplicate_unknown(
        self,
        *,
        url: str,
        result: RobotsCheckResult,
    ) -> bool:
        """Return whether an UNKNOWN result is a duplicate within the TTL."""

        if result.decision != RobotsDecision.UNKNOWN:
            return False
        return self._mark_duplicate_result(url=url, result=result)

    def _mark_duplicate_result(
        self,
        *,
        url: str,
        result: RobotsCheckResult,
    ) -> bool:
        host = self._host_normalizer.normalize(urlsplit(url).hostname)
        if host is None:
            return False

        confidence = result.confidence or RobotsConfidence.WEAK_UNKNOWN
        key = (
            f"{host}|{result.decision.value}|{confidence.value}|"
            f"{result.http_status}|{result.reason}|{result.source}"
        )
        now = monotonic()
        expires_at = self._recent_result_keys.get(key, 0.0)
        if expires_at > now:
            return True

        self._recent_result_keys[key] = now + self._ttl_seconds
        self._recent_result_insertions += 1

        if self._recent_result_insertions % self._prune_every == 0:
            self._prune(now)

        self._enforce_bounds(now)
        return False

    def _prune(self, now: float) -> None:
        expired_keys = [
            key
            for key, expires_at in self._recent_result_keys.items()
            if expires_at <= now
        ]
        for key in expired_keys:
            self._recent_result_keys.pop(key, None)

    def _enforce_bounds(self, now: float) -> None:
        if len(self._recent_result_keys) <= self._max_entries:
            return

        self._prune(now)
        while len(self._recent_result_keys) > self._max_entries:
            oldest_key = next(iter(self._recent_result_keys))
            self._recent_result_keys.pop(oldest_key, None)
