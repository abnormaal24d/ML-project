"""Apply embedded-asset scope, CDN-host, and static-noise rules."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import ParseResult

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.governance import UrlFilterSettings
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )


class EmbeddedAssetRules:
    _EXTERNAL_EMBEDDED_ALLOWED_KINDS: frozenset[str] = frozenset(
        {
            "image",
            "audio",
            "video",
            "document",
        }
    )

    def __init__(
        self,
        *,
        settings: UrlFilterSettings,
        host_normalizer: HostNormalizer,
        source_scope_registry: SourceScopeRegistry,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._host_normalizer = host_normalizer
        self._source_scope_registry = source_scope_registry
        self._logger = logger
        self._blocked_static_asset_extensions = tuple(
            ext.lower()
            for ext in self._settings.static_assets.blocked_extensions
        )

    @property
    def restrict_to_seed_hosts_enabled(self) -> bool:
        return bool(self._settings.restrict_to_seed_hosts)

    def static_asset_rejection_reason(
        self,
        *,
        url: str,
        parsed: ParseResult,
        kind: str | None,
        source_type: str,
    ) -> str | None:
        static_settings = self._settings.static_assets
        if (
            not static_settings.enabled
            or not static_settings.apply_to_embedded_assets
        ):
            return None
        if source_type != "embedded_asset":
            return None
        blockable_kinds = (
            static_settings.blockable_static_asset_kinds
            or static_settings.document_kinds
        )
        if kind is not None and kind not in blockable_kinds:
            return None

        path = (parsed.path or "").lower()
        if self._blocked_static_asset_extensions and path.endswith(
            self._blocked_static_asset_extensions
        ):
            self._logger.debug(
                "url_filter_rejected",
                extra={
                    "url_host": parsed.hostname or "",
                    "reason": "static_asset_extension_blocked",
                },
            )
            return "static_asset_extension_blocked"
        return None

    def rejection_reason(
        self,
        *,
        host: str,
        source_name: str,
    ) -> str | None:
        normalized_host = self._host_normalizer.normalize(host) or ""

        if not normalized_host:
            return "embedded_asset_external_host_not_allowlisted"

        try:
            scope = self._source_scope_registry.require(source_name)
        except ValueError:
            return "embedded_asset_external_host_blocked"

        if scope.allows_page_host(normalized_host):
            return None

        if scope.allows_asset_host(normalized_host):
            return None

        self._logger.debug(
            "url_filter_rejected",
            extra={
                "url_host": normalized_host,
                "reason": "embedded_asset_external_host_blocked",
            },
        )
        return "embedded_asset_external_host_blocked"
