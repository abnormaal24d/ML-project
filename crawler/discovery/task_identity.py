"""Canonical identities for discovered crawl tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


def discovered_task_identity(*, task: CrawlTask, normalized_url: str) -> str:
    """Return the canonical duplicate identity for one discovered task."""

    return discovered_task_identity_from_parts(
        url=normalized_url,
        kind=task.kind,
        source_type=task.source_type,
    )


def discovered_task_identity_from_parts(
    *,
    url: str,
    kind: str | None,
    source_type: str | None,
) -> str:
    """Return a stable task identity from already-known task fields.

    The URL is expected to already be canonical (produced by UrlNormalizer).
    Identity performs no second URL normalization: UrlNormalizer is the only
    owner of URL canonicalization, so no query parameters are interpreted or
    removed here.
    """

    task_kind = _normalize_token(kind, default="unknown")
    task_source = _normalize_token(source_type, default="unknown")
    return f"{str(url).strip()}|{task_kind}|{task_source}"


def _normalize_token(value: str | None, *, default: str) -> str:
    token = str(value or "").strip().lower()
    return token or default
