"""Bounded cache for durable conditional HTTP representations."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from crawler.fetching.results.result import FetchResult


@dataclass(frozen=True, slots=True)
class CachedRepresentation:
    """Conditional validators bound to a complete local representation."""

    validators: dict[str, str]
    result: FetchResult
    expires_at: float


class ConditionalRepresentationCache:
    """Cache durable representations and their HTTP validators."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_entries: int,
        ttl_seconds: float | None,
        clock: Callable[[], float],
    ) -> None:
        self._enabled = enabled
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, CachedRepresentation] = OrderedDict()
        self._lock = RLock()

    def get_representation(self, url: str) -> CachedRepresentation | None:
        """Return a complete, hash-valid local representation for a URL."""

        if self._disabled:
            return None

        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(url)
            if entry is None:
                return None

            if not self._representation_is_valid(entry.result):
                self._invalidate_result_locked(result=entry.result)
                return None

            self._entries.move_to_end(url)
            return entry

    def invalidate(self, url: str) -> None:
        """Remove any cached validators for the given URL (requested or final)."""

        if self._disabled:
            return
        with self._lock:
            self._entries.pop(url, None)

    def enrich_headers(
        self,
        *,
        url: str,
        base_headers: dict[str, str],
    ) -> dict[str, str]:
        """
        Return request headers enriched with ETag/Last-Modified validators.
        """

        representation = self.get_representation(url)
        if representation is None:
            return base_headers

        headers = dict(base_headers)
        validators = representation.validators
        if etag := validators.get("etag"):
            headers.setdefault("If-None-Match", etag)
        if last_modified := validators.get("last_modified"):
            headers.setdefault("If-Modified-Since", last_modified)
        return headers

    def commit_response(
        self,
        *,
        requested_url: str,
        final_url: str,
        headers: Mapping[str, str],
        result: FetchResult,
    ) -> dict[str, str]:
        """Commit validators only with a complete local representation."""

        validators = self._extract_validators(headers)
        if self._disabled or not validators:
            return {}
        if not self._representation_is_valid(result):
            raise ValueError(
                "conditional validators require a complete hash-valid payload"
            )

        normalized_validators = {
            key: value for key, value in validators.items() if value
        }
        if not normalized_validators:
            return {}

        ttl_seconds = cast(float, self._ttl_seconds)

        with self._lock:
            self._purge_expired_locked()
            for url in self._unique_urls(requested_url, final_url):
                self._entries[url] = CachedRepresentation(
                    validators=dict(normalized_validators),
                    result=result,
                    expires_at=self._clock() + ttl_seconds,
                )
                self._entries.move_to_end(url)
            self._evict_locked()
        return validators

    @property
    def size(self) -> int:
        """Return the current number of cached URL entries."""

        if self._disabled:
            return 0

        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    @property
    def _disabled(self) -> bool:
        return not self._enabled

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        expired_urls = [
            url
            for url, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for url in expired_urls:
            self._entries.pop(url, None)

    def _evict_locked(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _invalidate_result_locked(self, *, result: FetchResult) -> None:
        for url in tuple(self._entries):
            if self._entries[url].result is result:
                self._entries.pop(url, None)

    @staticmethod
    def _extract_validators(headers: Mapping[str, str]) -> dict[str, str]:
        normalized_headers = {
            key.lower(): value for key, value in headers.items()
        }
        validators: dict[str, str] = {}
        if etag := normalized_headers.get("etag"):
            validators["etag"] = etag
        if last_modified := normalized_headers.get("last-modified"):
            validators["last_modified"] = last_modified
        return validators

    @staticmethod
    def _representation_is_valid(result: FetchResult) -> bool:
        payload = result.payload
        if (
            payload is None
            or not payload.is_complete_payload
            or payload.truncated
            or not result.body_sha256
            or payload.sha256_hex != result.body_sha256
        ):
            return False
        path = Path(payload.temp_path)
        try:
            if not path.is_file() or path.stat().st_size != payload.byte_size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65_536), b""):
                    digest.update(chunk)
        except OSError:
            return False
        return digest.hexdigest() == result.body_sha256

    @staticmethod
    def _unique_urls(*urls: str) -> tuple[str, ...]:
        unique: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(url)
        return tuple(unique)
