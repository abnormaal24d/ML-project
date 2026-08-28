"""Validate redirect rules for safe redirect handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn
from urllib.parse import urlsplit

from crawler.fetching.errors.exceptions import IgnoredFetchError
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.http_rules import RedirectRulesSettings
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )
    from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics


class RedirectRulesValidator:
    """Validate every redirect hop before the next request is issued."""

    def __init__(
        self,
        *,
        settings: RedirectRulesSettings,
        host_extractor: HostExtractor,
        url_validator: UrlSchemeRules,
        blacklist_repository: BlacklistRepository,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
        network_access_guard: NetworkAddressGuard,
        metrics: CollectionMetrics | None = None,
        source_scope_registry: SourceScopeRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._host_extractor = host_extractor
        self._url_validator = url_validator
        self._blacklist_repository = blacklist_repository
        self._host_normalizer = host_normalizer
        self._logger = logger
        self._network_access_guard = network_access_guard
        self._metrics = metrics
        self._source_scope_registry = source_scope_registry

    @property
    def max_redirects(self) -> int:
        """Return the configured redirect limit."""
        return self._settings.max_redirects

    @property
    def max_location_length(self) -> int:
        """Return the maximum accepted raw Location-header length."""
        return self._settings.max_location_length

    def validate_hop(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
        source_name: str | None,
    ) -> None:
        """Raise when one redirect hop violates transport governance."""

        self._validate_shared_hop(
            current_url=current_url,
            target_url=target_url,
            redirect_count=redirect_count,
            max_redirects=self._settings.max_redirects,
        )

        current_host = self._normalized_host_from_url(current_url)
        target_host = self._normalized_host_from_url(target_url)

        if current_host is None or target_host is None:
            self._logger.debug(
                "redirect_host_unresolved",
                extra={
                    "current_host": current_host,
                    "target_host": target_host,
                },
            )
            raise IgnoredFetchError(
                reason="redirect_host_unresolved",
                observed_bytes=0,
            )

        if current_host == target_host:
            return

        if self._settings.cross_host_mode == "deny":
            self._reject_cross_host(
                current_host=current_host,
                target_host=target_host,
                reason="redirect_cross_host_blocked",
            )

        self._ensure_target_in_source_scope(
            source_name=source_name,
            current_host=current_host,
            target_host=target_host,
        )

    def validate_robots_hop(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
    ) -> None:
        """Raise when one robots.txt redirect hop violates transport governance.

        Robots documents may redirect across authorities (RFC 9309); source
        scopes constrain crawl targets, not the robots resource itself, so
        no source-scope approval is required here.
        """

        self._validate_shared_hop(
            current_url=current_url,
            target_url=target_url,
            redirect_count=redirect_count,
            max_redirects=self._settings.robots_max_redirects,
        )

    def _validate_shared_hop(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
        max_redirects: int,
    ) -> None:
        if redirect_count > max_redirects:
            self._logger.debug(
                "redirect_too_many_redirects",
                extra={
                    "redirect_count": redirect_count,
                    "max_redirects": max_redirects,
                },
            )
            raise IgnoredFetchError(
                reason="redirect_too_many_redirects",
                observed_bytes=0,
            )

        if not self._url_validator.is_allowed(target_url):
            self._logger.debug(
                "redirect_unsupported_scheme_rejected",
                extra={
                    "current_host": self._host_from_url(current_url),
                    "target_host": self._host_from_url(target_url),
                },
            )
            raise IgnoredFetchError(
                reason="redirect_unsupported_scheme",
                observed_bytes=0,
            )

        self._reject_https_downgrade(
            current_url=current_url,
            target_url=target_url,
        )

        network_reason = self._network_access_guard.rejection_reason_for_url(
            target_url
        )
        if network_reason is not None:
            self._logger.debug(
                "redirect_to_blocked_network_target",
                extra={
                    "target_host": self._host_from_url(target_url),
                    "reason": network_reason,
                },
            )
            raise IgnoredFetchError(
                reason=f"redirect_{network_reason}",
                observed_bytes=0,
            )

        if self._blacklist_repository.contains(url=target_url):
            blocked_target_host = self._host_from_url(target_url)
            if self._metrics is not None:
                self._metrics.record_blacklist_block(
                    url=target_url,
                    host=blocked_target_host,
                    stage="redirect",
                    reason="redirect_blacklisted",
                )
            self._logger.debug(
                "redirect_to_blacklisted_url",
                extra={"target_host": blocked_target_host},
            )
            raise IgnoredFetchError(
                reason="redirect_blacklisted",
                observed_bytes=0,
                metrics_recorded=True,
            )

    def _reject_https_downgrade(
        self,
        *,
        current_url: str,
        target_url: str,
    ) -> None:
        current_scheme = urlsplit(current_url).scheme.lower()
        target_scheme = urlsplit(target_url).scheme.lower()

        if (
            self._settings.block_https_downgrade
            and current_scheme == "https"
            and target_scheme != "https"
        ):
            self._logger.debug(
                "redirect_https_downgrade_rejected",
                extra={
                    "current_host": self._host_from_url(current_url),
                    "target_host": self._host_from_url(target_url),
                },
            )
            raise IgnoredFetchError(
                reason="redirect_https_downgrade_blocked",
                observed_bytes=0,
            )

    def _ensure_target_in_source_scope(
        self,
        *,
        source_name: str | None,
        current_host: str,
        target_host: str,
    ) -> None:
        registry = self._source_scope_registry

        if registry is None or not source_name:
            self._reject_cross_host(
                current_host=current_host,
                target_host=target_host,
                reason="redirect_source_scope_unavailable",
            )

        try:
            scope = registry.require(source_name)
        except ValueError as exc:
            self._logger.debug(
                "redirect_source_scope_unknown",
                extra={
                    "source_name": source_name,
                    "current_host": current_host,
                    "target_host": target_host,
                },
            )
            raise IgnoredFetchError(
                reason="redirect_source_scope_unknown",
                observed_bytes=0,
            ) from exc

        if scope.allows_redirect_host(target_host):
            return

        self._reject_cross_host(
            current_host=current_host,
            target_host=target_host,
            reason="redirect_target_not_source_approved",
        )

    def _reject_cross_host(
        self,
        *,
        current_host: str,
        target_host: str,
        reason: str,
    ) -> NoReturn:
        self._logger.debug(
            "redirect_cross_host_rejected",
            extra={
                "current_host": current_host,
                "target_host": target_host,
                "reason": reason,
            },
        )
        raise IgnoredFetchError(
            reason=reason,
            observed_bytes=0,
        )

    def _normalized_host_from_url(self, url: str) -> str | None:
        try:
            host = self._host_extractor.extract(url)
            return self._host_normalizer.normalize(host) or None
        except (ValueError, TypeError, AttributeError):
            return None

    def _host_from_url(self, url: str) -> str:
        return self._normalized_host_from_url(url) or "unknown"
