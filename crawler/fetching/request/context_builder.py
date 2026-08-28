"""Fetch request context construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from crawler.classification.media_kind import MediaKind
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.request.context import FetchRequestContext
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.fetching.acceptance.resolver import (
        FetchAcceptanceResolver,
    )
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics

AcceptanceMode = Literal["strict", "exploratory"]


class FetchRequestContextBuilder:
    """Build an immutable request context for a logical fetch operation."""

    _EXPLORATORY_KINDS = frozenset({MediaKind.PAGE, MediaKind.FEED})

    def __init__(
        self,
        *,
        url_validator: UrlSchemeRules,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
        acceptance_resolver: FetchAcceptanceResolver,
        logger: ProjectLogger,
        blacklist_repository: BlacklistRepository | None = None,
        network_access_guard: NetworkAddressGuard,
        metrics: CollectionMetrics | None = None,
    ) -> None:
        self._url_validator = url_validator
        self._host_extractor = host_extractor
        self._host_normalizer = host_normalizer
        self._acceptance_resolver = acceptance_resolver
        self._logger = logger
        self._blacklist_repository = blacklist_repository
        self._network_access_guard = network_access_guard
        self._metrics = metrics

    def build(
        self,
        *,
        task: CrawlTask,
    ) -> FetchRequestContext:
        """Validate request input and build the fetch request context."""
        url = task.url
        self._validate_url(url=url)
        self._ensure_network_access_allowed(url=url)

        normalized_host = self._resolve_host(url=url)
        self._ensure_not_blacklisted(
            url=url,
            host=normalized_host,
        )

        requested_kind = task.kind
        acceptance_mode = self._resolve_acceptance_mode(
            requested_kind=requested_kind,
        )
        acceptance = self._acceptance_resolver.resolve(
            kind=requested_kind,
            acceptance_mode=acceptance_mode,
        )

        return FetchRequestContext(
            url=url,
            host=normalized_host,
            source_name=task.source_name,
            requested_kind=requested_kind,
            acceptance_mode=acceptance_mode,
            acceptance=acceptance,
            task_context=self._task_context_payload(task=task),
        )

    @staticmethod
    def _task_context_payload(*, task: CrawlTask) -> dict[str, object]:
        context = task.context
        if context is None:
            return {}
        return context.to_dict()

    def _validate_url(self, *, url: str) -> None:
        if not self._url_validator.is_allowed(url):
            raise IgnoredFetchError(
                reason="unsupported_url_scheme",
                observed_bytes=0,
            )

    def _resolve_host(self, *, url: str) -> str:
        host = self._host_extractor.extract(url)
        try:
            return self._host_normalizer.require(host)
        except ValueError as exc:
            raise IgnoredFetchError(
                reason="missing_url_host",
                observed_bytes=0,
            ) from exc

    def _ensure_network_access_allowed(self, *, url: str) -> None:
        reason = self._network_access_guard.rejection_reason_for_url(url)
        if reason is None:
            return

        self._logger.debug(
            "fetch_skipped_network_target",
            url=url,
            reason=reason,
        )
        raise IgnoredFetchError(
            reason=f"blocked_network_target:{reason}",
            observed_bytes=0,
        )

    def _ensure_not_blacklisted(
        self,
        *,
        url: str,
        host: str,
    ) -> None:
        if self._blacklist_repository is None:
            return

        if not self._blacklist_repository.contains(url=url):
            return

        self._logger.debug(
            "fetch_skipped_blacklisted",
            url=url,
            host=host,
        )
        metrics = self._metrics
        if metrics is not None:
            metrics.record_blacklist_block(
                url=url,
                host=host,
                stage="fetch_request",
                reason="url_blacklisted",
            )
        raise IgnoredFetchError(
            reason="url_blacklisted",
            observed_bytes=0,
            metrics_recorded=True,
        )

    def _resolve_acceptance_mode(
        self,
        *,
        requested_kind: MediaKind,
    ) -> AcceptanceMode:
        if requested_kind in self._EXPLORATORY_KINDS:
            return "exploratory"

        return "strict"
