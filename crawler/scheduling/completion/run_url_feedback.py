"""Run-local URL feedback used by scheduler admission and dispatch."""

from __future__ import annotations

import posixpath
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, unquote, urlsplit

from crawler.discovery.task_identity import discovered_task_identity

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


class RunUrlFeedback:
    """Track transient URL feedback that must not survive a crawler run."""

    def __init__(self, *, normalize_url: Callable[[str], str]) -> None:
        self._normalize_url = normalize_url
        self._not_modified_identities: set[str] = set()
        self._forbidden_endpoint_identities: set[str] = set()

    def remember_not_modified(
        self,
        *,
        task: CrawlTask,
        url: str | None = None,
    ) -> bool:
        """Remember that a task identity returned HTTP 304 in this run."""

        identity = self._identity(task=task, url=url)
        if not identity:
            return False
        already_present = identity in self._not_modified_identities
        self._not_modified_identities.add(identity)
        return not already_present

    def was_not_modified(
        self,
        *,
        task: CrawlTask,
        url: str | None = None,
    ) -> bool:
        """Return whether this task identity already produced HTTP 304."""

        identity = self._identity(task=task, url=url)
        return identity in self._not_modified_identities if identity else False

    @property
    def not_modified_count(self) -> int:
        """Return the number of run-local HTTP 304 identities."""

        return len(self._not_modified_identities)

    def remember_forbidden_endpoint(
        self,
        *,
        url: str,
    ) -> bool:
        """Remember an endpoint that answered HTTP 403 in this run."""

        identity = forbidden_endpoint_identity(url)
        if not identity:
            return False
        already_present = identity in self._forbidden_endpoint_identities
        self._forbidden_endpoint_identities.add(identity)
        return not already_present

    def is_forbidden_endpoint(
        self,
        *,
        url: str,
    ) -> bool:
        """Return whether this endpoint already produced HTTP 403."""

        identity = forbidden_endpoint_identity(url)
        return (
            identity in self._forbidden_endpoint_identities
            if identity
            else False
        )

    @property
    def forbidden_endpoint_count(self) -> int:
        """Return the number of run-local forbidden endpoints."""

        return len(self._forbidden_endpoint_identities)

    def _identity(self, *, task: CrawlTask, url: str | None) -> str:
        raw_url = task.url if url is None else url
        normalized_url = self._normalize_url(raw_url)
        if not normalized_url:
            return ""
        return discovered_task_identity(
            task=task, normalized_url=normalized_url
        )


def forbidden_endpoint_identity(url: str) -> str:
    """Return an endpoint identity from scheme, host, path, and query keys.

    Query values are deliberately excluded so that
    ``.../gsearch?q=water&page=2`` and ``.../gsearch?q=ocean&page=7``
    resolve to the same endpoint while distinct paths stay distinct.
    """

    try:
        parsed = urlsplit(str(url))
    except ValueError:
        return ""

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return ""

    path = unquote(parsed.path or "/").replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")

    normalized_path = posixpath.normpath(path)
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    normalized_path = normalized_path.rstrip("/") or "/"

    query_keys = sorted(
        {
            key
            for key, _ in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
        }
    )

    query = "?" + "&".join(query_keys) if query_keys else ""

    return f"{host}{normalized_path}{query}"
