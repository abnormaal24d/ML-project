"""Fetch response validation.

Uses FetchResponseSnapshot to avoid direct dependency on aiohttp.ClientResponse
in the response layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
)
from crawler.fetching.response.snapshot import (
    FetchResponseSnapshot,
)

if TYPE_CHECKING:
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from logger.project_logger import ProjectLogger


def read_content_type_header(
    headers: Mapping[str, str],
) -> str | None:
    """Return the raw Content-Type header using case-insensitive lookup."""

    for key, value in headers.items():
        if key.casefold() != "content-type":
            continue

        normalized = value.strip()

        return normalized or None

    return None


class FetchResponseValidator:
    """Validate redirect target and transport-level response acceptance."""

    def __init__(
        self,
        *,
        redirector: RedirectRulesValidator,
        logger: ProjectLogger,
        metrics: CollectionMetrics | None = None,
    ) -> None:
        self._redirector = redirector
        self._logger = logger
        self._metrics = metrics

    def validate(
        self,
        *,
        context: FetchRequestContext,
        response: FetchResponseSnapshot,
    ) -> str:
        """Validate response transport constraints and return the final URL.

        Accepts only FetchResponseSnapshot (created in the transport layer).
        """
        final_url = response.url

        self._validate_redirects(
            context=context,
            response=response,
        )

        content_type = normalize_mime_type(
            read_content_type_header(response.headers)
        )
        content_length = response.content_length

        self._validate_transport_acceptance(
            context=context,
            final_url=final_url,
            content_type=content_type,
            content_length=content_length,
        )

        return final_url

    @staticmethod
    def blocks_indexing(headers: Mapping[str, str]) -> bool:
        """Return whether X-Robots-Tag forbids indexing this response."""

        raw_value = headers.get("X-Robots-Tag") or headers.get("x-robots-tag")
        if raw_value is None:
            return False

        tokens = {
            token.strip().lower()
            for part in str(raw_value).split(",")
            for token in part.split(":")[-1].split()
            if token.strip()
        }
        return bool(tokens.intersection({"none", "noindex"}))

    def _validate_redirects(
        self,
        *,
        context: FetchRequestContext,
        response: FetchResponseSnapshot,
    ) -> None:
        redirect_chain = response.redirect_chain or ()

        if not redirect_chain:
            self._redirector.validate_hop(
                current_url=context.url,
                target_url=response.url,
                redirect_count=0,
                source_name=context.source_name,
            )
            return

        current_url = context.url
        for redirect_count, target_url in enumerate(redirect_chain, start=1):
            self._redirector.validate_hop(
                current_url=current_url,
                target_url=target_url,
                redirect_count=redirect_count,
                source_name=context.source_name,
            )
            current_url = target_url

    def _validate_transport_acceptance(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
        content_type: str | None,
        content_length: int | None,
    ) -> None:
        """Apply coarse transport-level acceptance checks."""
        if (
            content_type is not None
            and not context.acceptance.allows_content_type(content_type)
        ):
            self._record_skipped_metric(
                host=context.host,
                reason="transport_content_type_not_allowed",
                observed_bytes=0,
            )
            self._logger.warning(
                "fetch_skipped",
                url=context.url,
                final_url=final_url,
                reason="transport_content_type_not_allowed",
                content_type=content_type,
                acceptance_mode=context.acceptance_mode,
            )
            raise IgnoredFetchError(
                reason="transport_content_type_not_allowed",
                observed_bytes=0,
                metrics_recorded=True,
            )

        max_bytes = context.acceptance.max_bytes_for_content_type(content_type)

        if content_length is not None and content_length > max_bytes:
            if self._can_defer_oversized_media_decision(context=context):
                self._logger.warning(
                    "fetch_oversized_media_deferred_to_body_plan",
                    url=context.url,
                    final_url=final_url,
                    max_bytes=max_bytes,
                    observed_bytes=content_length,
                    acceptance_mode=context.acceptance_mode,
                )
                return
            self._record_skipped_metric(
                host=context.host,
                reason="transport_content_length_exceeded",
                observed_bytes=content_length,
            )
            self._logger.warning(
                "fetch_skipped",
                url=context.url,
                final_url=final_url,
                reason="transport_content_length_exceeded",
                max_bytes=max_bytes,
                observed_bytes=content_length,
                acceptance_mode=context.acceptance_mode,
            )
            raise IgnoredFetchError(
                reason="transport_content_length_exceeded",
                observed_bytes=content_length,
                metrics_recorded=True,
            )

    @staticmethod
    def _can_defer_oversized_media_decision(
        *,
        context: FetchRequestContext,
    ) -> bool:
        return context.requested_kind in {MediaKind.AUDIO, MediaKind.VIDEO}

    def _record_skipped_metric(
        self,
        *,
        host: str | None,
        reason: str,
        observed_bytes: int,
    ) -> None:
        metrics = self._metrics
        if metrics is None or not metrics.enabled:
            return

        metrics.record_fetch_skipped(
            host=host,
            reason=reason,
            bytes_downloaded=observed_bytes,
        )
