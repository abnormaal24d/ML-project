"""HEAD preflight response validation and transport acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.media.strategy import (
    HeadPreflightResult,
    MediaFetchStrategyResolver,
)
from crawler.fetching.response.validator import (
    read_content_type_header,
)

if TYPE_CHECKING:
    from aiohttp import ClientResponse

    from config.collection.fetching import FetcherSettings
    from config.collection.http_rules import HttpStatusRulesSettings
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.governance.retry.retry_manager import RetryManager
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from logger.project_logger import ProjectLogger


@dataclass(frozen=True, slots=True)
class PreflightStatusDetails:
    """Normalized response fields used by preflight status rules."""

    url: str
    host: str | None
    status_code: int
    final_url: str
    content_type: str | None
    content_length: int | None


class HeadPreflightResponseEvaluator:
    """Validate HEAD responses and return selected preflight actions.

    Redirect governance is owned by AiohttpRequestRunner before each hop.
    This evaluator only interprets the final HEAD response.
    """

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        logger: ProjectLogger,
        metrics: CollectionMetrics | None,
        media_strategy_resolver: MediaFetchStrategyResolver,
        status_settings: HttpStatusRulesSettings,
        retry_manager: RetryManager,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._metrics = metrics
        self._media_strategy_resolver = media_strategy_resolver
        self._status_settings = status_settings
        self._retry_manager = retry_manager

    def handle_head_result(
        self,
        *,
        context: FetchRequestContext,
        url: str,
        host: str | None,
        response: ClientResponse,
    ) -> HeadPreflightResult:
        """Validate a HEAD response and return the selected preflight action."""

        status_code = int(response.status)
        final_url = str(response.url)
        content_type = normalize_mime_type(
            read_content_type_header(response.headers)
        )
        content_length = response.content_length

        details = PreflightStatusDetails(
            url=url,
            host=host,
            status_code=status_code,
            final_url=final_url,
            content_type=content_type,
            content_length=content_length,
        )
        status_result = self._evaluate_status(
            context=context,
            details=details,
            enforce_status_rules=self._settings.drop_if_head_disallowed,
        )
        if status_result is not None:
            return status_result

        if not self._settings.drop_if_head_disallowed:
            return HeadPreflightResult(
                attempted=True,
                status_code=status_code,
                final_url=final_url,
                content_type=content_type,
                content_length=content_length,
                record_kind=getattr(context, "requested_kind", None),
            )

        transport_decision = self.validate_transport_acceptance(
            context=context,
            url=url,
            final_url=final_url,
            content_type=content_type,
            content_length=content_length,
            status_code=status_code,
        )
        if transport_decision is not None:
            return transport_decision

        return HeadPreflightResult(
            attempted=True,
            status_code=status_code,
            final_url=final_url,
            content_type=content_type,
            content_length=content_length,
            record_kind=getattr(context, "requested_kind", None),
        )

    def _evaluate_status(
        self,
        *,
        context: FetchRequestContext,
        details: PreflightStatusDetails,
        enforce_status_rules: bool,
    ) -> HeadPreflightResult | None:
        if self._retry_manager.is_retryable_status(details.status_code):
            return self._soft_status_rejection(
                details=details,
                reason="head_status_retryable",
            )
        if not enforce_status_rules:
            return None
        if self._status_settings.is_head_method_not_supported(
            details.status_code
        ):
            return self._soft_status_rejection(
                details=details,
                reason="head_status_method_not_supported",
            )
        if not self._is_rejected_status(details.status_code):
            return None

        if (
            details.status_code
            in self._status_settings.head_preflight_hard_drop
        ):
            self._logger.debug(
                "head_preflight_hard_reject",
                url=details.url,
                host=details.host,
                reason="head_status_not_allowed",
                status_code=details.status_code,
                acceptance_mode=context.acceptance_mode,
            )
            self._record_skipped_metric(
                host=details.host,
                reason="head_status_not_allowed",
                observed_bytes=0,
            )
            raise IgnoredFetchError(
                reason="head_status_not_allowed",
                observed_bytes=0,
                metrics_recorded=True,
            )

        return self._soft_status_rejection(
            details=details,
            reason="head_status_get_fallback",
        )

    def _is_rejected_status(self, status_code: int) -> bool:
        if not self._settings.raise_for_non_success_status:
            return False
        if 200 <= status_code < 300:
            return False
        return status_code not in self._status_settings.accepted_non_success

    def _soft_status_rejection(
        self,
        *,
        details: PreflightStatusDetails,
        reason: str,
    ) -> HeadPreflightResult:
        result = HeadPreflightResult(
            attempted=True,
            status_code=details.status_code,
            final_url=details.final_url,
            content_type=details.content_type,
            content_length=details.content_length,
            soft_rejected=True,
            rejection_reason=reason,
            record_kind=None,
        )
        self._logger.debug(
            "head_preflight_soft_reject",
            url=details.url,
            host=details.host,
            reason=reason,
            status_code=details.status_code,
        )
        return result

    def validate_transport_acceptance(
        self,
        *,
        context: FetchRequestContext,
        url: str,
        final_url: str,
        content_type: str | None,
        content_length: int | None,
        status_code: int | None,
    ) -> HeadPreflightResult | None:
        """Apply coarse transport-level acceptance checks."""
        if (
            content_type is not None
            and not context.acceptance.allows_content_type(content_type)
        ):
            self._record_skipped_metric(
                host=context.host,
                reason="head_transport_content_type_not_allowed",
                observed_bytes=0,
            )
            self._logger.warning(
                "fetch_skipped",
                url=url,
                final_url=final_url,
                reason="head_transport_content_type_not_allowed",
                content_type=content_type,
                acceptance_mode=context.acceptance_mode,
            )
            raise IgnoredFetchError(
                reason="head_transport_content_type_not_allowed",
                observed_bytes=0,
                metrics_recorded=True,
            )

        max_bytes = context.acceptance.max_bytes_for_content_type(content_type)

        if content_length is not None and content_length > max_bytes:
            media_decision = self.oversized_media_decision(
                context=context,
                url=url,
                final_url=final_url,
                content_type=content_type,
                content_length=content_length,
                status_code=status_code,
                max_bytes=max_bytes,
            )
            if media_decision is not None:
                return media_decision

            self._record_skipped_metric(
                host=context.host,
                reason="head_transport_content_length_exceeded",
                observed_bytes=content_length,
            )
            self._logger.info(
                "fetch_skipped",
                url=url,
                final_url=final_url,
                reason="head_transport_content_length_exceeded",
                max_bytes=max_bytes,
                observed_bytes=content_length,
                acceptance_mode=context.acceptance_mode,
                rules_skip=True,
                suggested_action="increase_fetch_max_bytes_for_full_media",
            )
            raise IgnoredFetchError(
                reason="head_transport_content_length_exceeded",
                observed_bytes=content_length,
                metrics_recorded=True,
            )

        return None

    def oversized_media_decision(
        self,
        *,
        context: FetchRequestContext,
        url: str,
        final_url: str,
        content_type: str | None,
        content_length: int,
        status_code: int | None,
        max_bytes: int,
    ) -> HeadPreflightResult | None:
        strategy = self._media_strategy_resolver.resolve(
            context=context,
            content_type=content_type,
        )
        if strategy is None:
            return None

        requested_kind = context.requested_kind
        self._logger.info(
            "head_preflight_oversized_media_accepted",
            url=url,
            final_url=final_url,
            reason=strategy.reason,
            action=strategy.action.value,
            content_type=content_type,
            acceptance_mode=context.acceptance_mode,
            rules_skip=False,
            record_kind=requested_kind,
            oversized_reason=strategy.reason,
            observed_bytes=content_length,
            max_bytes=max_bytes,
            acceptance_mode_resolved=strategy.action.value,
        )
        return HeadPreflightResult(
            attempted=True,
            allowed=True,
            action=strategy.action,
            status_code=status_code,
            final_url=final_url,
            content_type=content_type,
            content_length=content_length,
            action_reason=strategy.reason,
            record_kind=requested_kind,
        )

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
