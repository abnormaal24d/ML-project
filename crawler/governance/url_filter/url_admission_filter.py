"""URL admission filter for rules-based crawl decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.governance.url_filter.embedded_asset_rules import (
    EmbeddedAssetRules,
)
from crawler.governance.url_filter.host_denylist_rules import (
    HostDenylistRules,
)
from crawler.governance.url_filter.ip_literal_rules import IpLiteralRules
from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
from crawler.governance.url_filter.url_syntax_rules import UrlSyntaxRules
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(frozen=True, slots=True)
class UrlAdmissionDecision:
    """Structured URL-filter outcome with a stable rejection reason."""

    allowed: bool
    reason: str | None
    kind: str
    source_type: str
    normalized_url: str | None = None
    raw_url: str | None = None


class UrlAdmissionFilter:
    """Compose specialized URL validators for crawl admission decisions."""

    def __init__(
        self,
        *,
        url_scheme_validator: UrlSchemeRules,
        syntax_validator: UrlSyntaxRules,
        host_denylist_validator: HostDenylistRules,
        ip_literal_validator: IpLiteralRules,
        embedded_asset_validator: EmbeddedAssetRules,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
    ) -> None:
        self._url_scheme_validator = url_scheme_validator
        self._syntax_validator = syntax_validator
        self._host_denylist_validator = host_denylist_validator
        self._ip_literal_validator = ip_literal_validator
        self._embedded_asset_validator = embedded_asset_validator
        self._host_extractor = host_extractor
        self._host_normalizer = host_normalizer
        self._logger = logger

    def evaluate_task(self, task: CrawlTask) -> UrlAdmissionDecision:
        return self.evaluate(
            url=task.url,
            source_type=task.source_type,
            kind=task.kind,
            context=task.context,
            source_name=task.source_name,
        )

    # ------------------------------------------------------------------
    # Main evaluation pipeline
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Main evaluation pipeline
    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        url: str,
        source_type: str,
        kind: str | None = None,
        context: CrawlTaskContext | None = None,
        source_name: str | None = None,
    ) -> UrlAdmissionDecision:
        task_kind = str(kind or "unknown").strip().lower() or "unknown"
        decision_url = url

        def reject(reason: str) -> UrlAdmissionDecision:
            return UrlAdmissionDecision(
                allowed=False,
                reason=reason,
                kind=task_kind,
                source_type=source_type,
                normalized_url=decision_url,
                raw_url=url,
            )

        def allow() -> UrlAdmissionDecision:
            return UrlAdmissionDecision(
                allowed=True,
                reason=None,
                kind=task_kind,
                source_type=source_type,
                normalized_url=decision_url,
                raw_url=url,
            )

        if not self._url_scheme_validator.is_allowed(url):
            return reject("url_validator_rejected")

        parsed = self._syntax_validator.parse_url(url)
        if parsed is None:
            return reject("invalid_url")

        decision_url = parsed.geturl()
        path = parsed.path or ""

        for reason in (
            self._syntax_validator.discovery_noise_rejection_reason(
                url=decision_url,
                path=path,
                query=parsed.query,
                kind=task_kind,
                source_type=source_type,
            ),
            self._embedded_asset_validator.static_asset_rejection_reason(
                url=decision_url,
                parsed=parsed,
                kind=task_kind,
                source_type=source_type,
            ),
        ):
            if reason is not None:
                return reject(reason)

        host = self._syntax_validator.extract_normalized_host(
            url=decision_url,
            parsed=parsed,
            host_extractor=self._host_extractor,
            host_normalizer=self._host_normalizer,
        )

        if (
            reason := self._host_denylist_validator.rejection_reason(host)
        ) is not None:
            self._logger.debug(
                "url_filter_rejected",
                extra={
                    "url_host": host,
                    "source_type": source_type,
                    "kind": task_kind,
                    "reason": reason,
                    "stage": "host_denylist",
                },
            )
            return reject(reason)

        if (
            reason := self._ip_literal_validator.rejection_reason(host)
        ) is not None:
            return reject(reason)

        if (
            reason := self._syntax_validator.url_shape_rejection_reason(
                url=decision_url,
                path=path,
                query=parsed.query,
                source_type=source_type,
            )
        ) is not None:
            self._logger.debug(
                "url_filter_rejected",
                extra={
                    "url_host": host,
                    "source_type": source_type,
                    "kind": task_kind,
                    "reason": reason,
                    "stage": "shape",
                },
            )
            return reject(reason)

        if (
            source_type == "embedded_asset"
            and (
                reason := self._embedded_asset_validator.rejection_reason(
                    host=host,
                    source_name=source_name or "",
                )
            )
            is not None
        ):
            return reject(reason)

        return allow()

    @property
    def restrict_to_seed_hosts_enabled(self) -> bool:
        return self._embedded_asset_validator.restrict_to_seed_hosts_enabled
