"""Resolve alternate feed URLs after retryable primary-feed failures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask, CrawlTaskBuildRequest
from crawler.governance.domains.host_normalizer import HostNormalizer
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )

RETRYABLE_FEED_STATUS_CODES = frozenset({503, 504})
RETRYABLE_FEED_ERROR_TYPES = frozenset(
    {
        "TimeoutError",
        "ClientConnectorError",
        "transport_timeout",
        "body_timeout",
        "header_timeout",
        "connect_timeout",
        "sock_read_timeout",
        "transport_error",
        "connection_closed",
    }
)


def expand_seed_tasks_with_feed_alternates(
    *,
    seed_tasks: Iterable[CrawlTask],
    alternates_by_primary: Mapping[str, tuple[str, ...] | list[str]] | None,
    url_normalizer: UrlNormalizer,
    host_normalizer: HostNormalizer,
    source_scope_registry: SourceScopeRegistry,
    id_generator: IdGenerator,
) -> tuple[CrawlTask, ...]:
    """Expand configured feed primaries into source-bound seed tasks."""

    if id_generator is None:
        raise ValueError("id_generator is required")

    normalized_alternates = _normalize_alternates_by_primary(
        alternates_by_primary=alternates_by_primary,
        url_normalizer=url_normalizer,
    )
    expanded: list[CrawlTask] = []
    seen: set[tuple[str, str]] = set()

    for task in seed_tasks:
        primary_marker = _canonical_feed_url(
            task.url,
            url_normalizer=url_normalizer,
        )
        if not _is_absolute_http_url(primary_marker):
            raise ValueError(f"invalid feed seed URL: {task.url!r}")
        is_configured_primary = primary_marker in normalized_alternates
        configured_alternates = normalized_alternates.get(
            primary_marker,
            (),
        )

        primary_task = replace(task, url=primary_marker)
        if is_configured_primary and task.kind is not MediaKind.FEED:
            primary_task = replace(
                primary_task,
                kind=MediaKind.FEED,
            )

        primary_key = (
            primary_task.source_name,
            primary_marker,
        )
        if primary_key not in seen:
            seen.add(primary_key)
            expanded.append(primary_task)

        if not configured_alternates:
            continue

        source_scope = source_scope_registry.require(task.source_name)

        for alternate_url in configured_alternates:
            alternate_host = _normalized_url_host(
                alternate_url,
                host_normalizer=host_normalizer,
            )

            if alternate_host is None or not source_scope.allows_page_host(
                alternate_host
            ):
                raise ValueError(
                    "feed alternate is outside the configured "
                    "source page scope: "
                    f"source={task.source_name!r}, "
                    f"primary={task.url!r}, "
                    f"alternate={alternate_url!r}"
                )

            alternate_key = (
                primary_task.source_name,
                alternate_url,
            )
            if alternate_key in seen:
                continue

            seen.add(alternate_key)
            expanded.append(
                CrawlTask.build(
                    request=_alternate_seed_request(
                        primary_task=primary_task,
                        alternate_url=alternate_url,
                    ),
                    id_generator=id_generator,
                )
            )

    return tuple(expanded)


def _normalize_alternates_by_primary(
    *,
    alternates_by_primary: Mapping[str, tuple[str, ...] | list[str]] | None,
    url_normalizer: UrlNormalizer,
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, list[str]] = {}
    seen_by_primary: dict[str, set[str]] = {}

    for raw_primary, raw_alternates in (alternates_by_primary or {}).items():
        primary_marker = _canonical_feed_url(
            raw_primary,
            url_normalizer=url_normalizer,
        )

        if not _is_absolute_http_url(primary_marker):
            raise ValueError(f"invalid primary feed URL: {raw_primary!r}")

        values = normalized.setdefault(primary_marker, [])
        seen = seen_by_primary.setdefault(
            primary_marker,
            {primary_marker, *values},
        )

        for raw_alternate in raw_alternates:
            alternate_marker = _canonical_feed_url(
                raw_alternate,
                url_normalizer=url_normalizer,
            )
            if not _is_absolute_http_url(alternate_marker):
                raise ValueError(
                    f"invalid alternate feed URL: {raw_alternate!r}"
                )
            if alternate_marker in seen:
                continue

            seen.add(alternate_marker)
            values.append(alternate_marker)

    return {primary: tuple(values) for primary, values in normalized.items()}


def _canonical_feed_url(
    value: object,
    *,
    url_normalizer: UrlNormalizer,
) -> str:
    """Return the shared canonical URL projected to feed identity."""

    normalized = url_normalizer.normalize(value)
    if not normalized:
        return ""
    try:
        parsed = urlsplit(normalized)
    except (UnicodeError, ValueError):
        return ""

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.query,
            "",
        )
    )


def _normalized_url_host(
    url: str,
    *,
    host_normalizer: HostNormalizer,
) -> str | None:
    try:
        return host_normalizer.normalize(urlsplit(url).hostname)
    except ValueError:
        return None


def _is_absolute_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except (UnicodeError, ValueError):
        return False

    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
        and bool(parsed.hostname)
    )


def _alternate_seed_request(
    *,
    primary_task: CrawlTask,
    alternate_url: str,
) -> CrawlTaskBuildRequest:
    return CrawlTaskBuildRequest(
        url=alternate_url,
        source_name=primary_task.source_name,
        kind=MediaKind.FEED,
        depth=primary_task.depth,
        source_type=primary_task.source_type,
        priority=primary_task.priority,
        parent_url=primary_task.url,
        context={
            "discovery_reason": "feed_alternate_seed",
            "source_page_url": primary_task.url,
            "source_page_depth": primary_task.depth,
        },
        default_kind=MediaKind.FEED,
        default_depth=primary_task.depth,
        default_source_type=primary_task.source_type,
        default_priority=primary_task.priority,
    )


class FeedAlternateResolver:
    """Map primary feed URLs to registry-provided alternates."""

    def __init__(
        self,
        *,
        alternates_by_primary: Mapping[str, tuple[str, ...] | list[str]]
        | None = None,
        url_normalizer: UrlNormalizer,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._url_normalizer = url_normalizer
        self._host_normalizer = host_normalizer
        self._alternates_by_primary = _normalize_alternates_by_primary(
            alternates_by_primary=alternates_by_primary,
            url_normalizer=url_normalizer,
        )

    def alternates_for(
        self,
        *,
        url: str,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> tuple[str, ...]:
        """Return all alternates for one retryable primary-feed failure."""

        if not _is_retryable_failure(
            status_code=status_code,
            error_type=error_type,
        ):
            return ()
        key = _canonical_feed_url(
            url,
            url_normalizer=self._url_normalizer,
        )
        if not _is_absolute_http_url(key):
            return ()
        return self._alternates_by_primary.get(key, ())

    def alternate_for(
        self,
        *,
        url: str,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> str | None:
        """Return the first alternate for one independent fetch operation."""

        alternates = self.alternates_for(
            url=url,
            status_code=status_code,
            error_type=error_type,
        )
        return alternates[0] if alternates else None

    def host_matches(
        self,
        *,
        url: str,
        allowed_hosts: tuple[str, ...],
    ) -> bool:
        canonical_url = _canonical_feed_url(
            url,
            url_normalizer=self._url_normalizer,
        )
        host = _normalized_url_host(
            canonical_url,
            host_normalizer=self._host_normalizer,
        )
        if host is None:
            return False
        return host in {
            normalized
            for item in allowed_hosts
            if (normalized := self._host_normalizer.normalize(item))
            is not None
        }


def _is_retryable_failure(
    *,
    status_code: int | None,
    error_type: str | None,
) -> bool:
    if status_code in RETRYABLE_FEED_STATUS_CODES:
        return True
    normalized_error = str(error_type or "").strip()
    return normalized_error in RETRYABLE_FEED_ERROR_TYPES
