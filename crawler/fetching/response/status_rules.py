"""Apply fetch-response status rules and retry multimodal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from config.collection.http_rules import HttpStatusRulesSettings


class FetchResponseStatusRules:
    """Apply settings-driven status handling for one fetch response."""

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        status_rules: HttpStatusRulesSettings,
        logger: ProjectLogger,
    ) -> None:
        self._logger = logger
        self._raise_for_non_success_status = (
            settings.raise_for_non_success_status
        )
        self._accepted_non_success_statuses = frozenset(
            status_rules.accepted_non_success
        )
        self._retryable_statuses = frozenset(status_rules.retryable)

    def handle(
        self,
        *,
        status_code: int,
        url: str,
        host: str,
        final_url: str | None = None,
    ) -> None:
        """Handle one HTTP response status. No identity/profile switching."""
        if status_code == 403:
            # P1.3: 403 is authoritative deny (not like 503 temp). Do not waste frontier retrying.
            # Expect caller/host profile to: cool host, apply host rules, suppress pattern, use alt endpoints.
            # Limit per-status retry budget handled by governance/fetch rules.
            self._logger.info(
                "http_403_host_denied",
                host=host,
                url=url,
                advice="cool_host;suppress_pattern;prefer_official_alt",
            )
            raise IgnoredFetchError(
                reason="server_denied_403",
                observed_bytes=0,
                status_code=status_code,
                final_url=final_url or url,
            )

        if status_code in self._retryable_statuses:
            raise RetryableFetchError(
                f"retryable status code received: {status_code}",
                retry_class="status_retry",
                retry_error_kind=f"http_{status_code}",
                status_code=status_code,
            )

        if self._should_ignore_non_success_status(status_code=status_code):
            raise IgnoredFetchError(
                reason=f"non_success_status_{status_code}",
                observed_bytes=0,
                status_code=status_code,
                final_url=final_url or url,
            )

    def _should_ignore_non_success_status(self, *, status_code: int) -> bool:
        return (
            self._raise_for_non_success_status
            and not (200 <= status_code < 300)
            and status_code not in self._accepted_non_success_statuses
        )
