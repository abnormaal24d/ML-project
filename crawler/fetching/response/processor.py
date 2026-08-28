"""Process HTTP responses for fetch attempts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from aiohttp import ClientResponse

from crawler.fetching.errors.exceptions import RetryableFetchError
from crawler.fetching.response.snapshot import (
    FetchResponseSnapshot,
    safe_response_headers,
)
from crawler.fetching.response.validator import (
    read_content_type_header,
)
from crawler.fetching.results.result import FetchResult

if TYPE_CHECKING:
    from crawler.classification.content_classifier import ContentClassifier
    from crawler.fetching.acceptance.body_strategy import (
        MediaBodyReadStrategy,
    )
    from crawler.fetching.request.body_plan import BodyReadPlan
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.fetching.response.body_reader import (
        FetchResponseBodyReader,
    )
    from crawler.fetching.response.cache import (
        ConditionalRepresentationCache,
    )
    from crawler.fetching.response.status_rules import (
        FetchResponseStatusRules,
    )
    from crawler.fetching.response.validator import (
        FetchResponseValidator,
    )
    from crawler.governance.host_suppression import HostSuppressionStore
    from logger.project_logger import ProjectLogger


@dataclass(frozen=True, slots=True)
class FetchResponseOutcome:
    """Outcome of processing one HTTP response."""

    result: FetchResult | None
    status_code: int
    bytes_downloaded: int
    quality_score: float | None
    should_record_feedback: bool
    final_url: str


class FetchResponseProcessor:
    """Process HTTP responses produced by fetch attempts."""

    def __init__(
        self,
        *,
        fetch_response_body_reader: FetchResponseBodyReader,
        response_validator: FetchResponseValidator,
        response_status_rules: FetchResponseStatusRules,
        content_classifier: ContentClassifier,
        host_suppression_store: HostSuppressionStore,
        media_body_read_strategy: MediaBodyReadStrategy,
        conditional_representation_cache: ConditionalRepresentationCache,
        media_semaphore: asyncio.Semaphore,
        logger: ProjectLogger,
        now_utc: Callable[[], datetime],
    ) -> None:
        self._fetch_response_body_reader = fetch_response_body_reader
        self._response_validator = response_validator
        self._response_status_rules = response_status_rules
        self._content_classifier = content_classifier
        self._host_suppression_store = host_suppression_store
        self._now_utc = now_utc
        self._media_body_read_strategy = media_body_read_strategy
        self._conditional_representation_cache = (
            conditional_representation_cache
        )
        self._media_semaphore = media_semaphore
        self._logger = logger

    async def handle_response(
        self,
        *,
        context: FetchRequestContext,
        response: ClientResponse,
        read_plan: BodyReadPlan,
    ) -> FetchResponseOutcome:
        status_code = int(response.status)
        final_url_for_log = str(response.url)

        self._logger.debug(
            "fetch_response_received",
            url=context.url,
            final_url=str(response.url),
            host=context.host,
            status_code=int(response.status),
            response_headers=safe_response_headers(response.headers),
            proxy=None,
            body_read_mode=read_plan.mode,
            range_header=read_plan.headers.get("Range"),
            max_bytes=read_plan.max_bytes,
            allow_partial=read_plan.allow_partial,
        )
        early_outcome = self._early_rules_outcome(
            context=context,
            response=response,
            status_code=status_code,
            final_url=final_url_for_log,
        )
        if early_outcome is not None:
            return early_outcome

        snapshot = FetchResponseSnapshot.from_response(response)
        final_url = self._response_validator.validate(
            context=context,
            response=snapshot,
        )
        self._media_body_read_strategy.ensure_size_acceptance_allows_streaming(
            context=context,
            response=response,
            final_url=final_url,
            read_plan=read_plan,
        )

        read_result = (
            await self._media_body_read_strategy.read_with_optional_throttle(
                requested_kind=context.requested_kind,
                media_semaphore=self._media_semaphore,
                read_body=lambda: self._fetch_response_body_reader.read(
                    context=context,
                    response=response,
                    final_url=final_url,
                    read_plan=read_plan,
                ),
            )
        )

        classified = self._content_classifier.classify(
            url=final_url,
            content_type_header=(read_content_type_header(response.headers)),
            sniff_bytes=read_result.payload.sniff_bytes,
            payload_byte_size=read_result.byte_size,
            requested_kind=context.requested_kind,
        )
        result = FetchResult.build(
            requested_url=context.url,
            final_url=final_url,
            status_code=status_code,
            response_headers=response.headers,
            payload=read_result.payload,
            body_sha256=read_result.sha256,
            classified=classified,
            fetched_at=self._now_utc().isoformat(),
        )
        self._host_suppression_store.record_response_status(
            host=context.host,
            status_code=status_code,
        )
        self._logger.debug(
            "fetch_completed",
            url=context.url,
            final_url=final_url,
            status_code=status_code,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            result_kind=result.kind,
            bytes=read_result.byte_size,
            chunk_count=read_result.chunk_count,
            body_read_mode=read_plan.mode,
            range_header=read_plan.headers.get("Range"),
            allow_partial=read_plan.allow_partial,
            max_bytes=read_plan.max_bytes,
            body_truncated=read_result.truncated,
            source_content_length=read_result.source_content_length,
        )
        return FetchResponseOutcome(
            result=result,
            status_code=status_code,
            bytes_downloaded=read_result.byte_size,
            quality_score=classified.relevance_score,
            should_record_feedback=True,
            final_url=final_url,
        )

    def _early_rules_outcome(
        self,
        *,
        context: FetchRequestContext,
        response: ClientResponse,
        status_code: int,
        final_url: str,
    ) -> FetchResponseOutcome | None:
        if status_code == 304:
            return self._handle_not_modified(
                context=context,
                final_url=final_url,
            )

        self._response_status_rules.handle(
            status_code=status_code,
            url=context.url,
            host=context.host,
            final_url=final_url,
        )

        if not self._response_validator.blocks_indexing(response.headers):
            return None

        self._logger.info(
            "fetch_skipped",
            url=context.url,
            final_url=final_url,
            reason="x_robots_tag_blocked",
            status_code=status_code,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
        )
        return FetchResponseOutcome(
            result=None,
            status_code=status_code,
            bytes_downloaded=0,
            quality_score=None,
            should_record_feedback=True,
            final_url=final_url,
        )

    def _handle_not_modified(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
    ) -> FetchResponseOutcome:
        representation = (
            self._conditional_representation_cache.get_representation(
                context.url
            )
            or self._conditional_representation_cache.get_representation(
                final_url
            )
        )
        if representation is not None:
            self._logger.info(
                "fetch_cache_hit",
                url=context.url,
                final_url=final_url,
                status_code=304,
                requested_kind=context.requested_kind,
                acceptance_mode=context.acceptance_mode,
            )
            return FetchResponseOutcome(
                result=representation.result,
                status_code=304,
                bytes_downloaded=0,
                quality_score=representation.result.relevance_score,
                should_record_feedback=True,
                final_url=final_url,
            )

        self._conditional_representation_cache.invalidate(context.url)
        self._conditional_representation_cache.invalidate(final_url)
        self._logger.info(
            "fetch_not_modified",
            url=context.url,
            final_url=final_url,
            status_code=304,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            action="force_unconditional_retry",
        )
        raise RetryableFetchError(
            "not_modified_without_local_payload",
            retry_class="fetch_retryable",
            retry_error_kind="not_modified_force_unconditional",
            status_code=304,
            observed_bytes=0,
        )
