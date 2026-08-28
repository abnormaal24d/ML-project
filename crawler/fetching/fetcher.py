"""Logical fetch orchestration."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from aiohttp import ClientSession

from crawler.classification.media_kind import MediaKind
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.media.strategy import (
    HeadPreflightResult,
    MediaFetchStrategyResolver,
)

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from crawler.discovery.feed_alternate_resolver import FeedAlternateResolver
    from crawler.fetching.execution.attempt import FetchAttemptExecutor
    from crawler.fetching.media.metadata_builder import (
        MediaMetadataResultBuilder,
    )
    from crawler.fetching.network.preflight.executor import (
        HeadPreflightExecutor,
    )
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.fetching.request.context_builder import (
        FetchRequestContextBuilder,
    )
    from crawler.fetching.request.headers.builder import (
        RequestHeaderBuilder,
    )
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.retry.retry_manager import RetryManager
    from crawler.runtime.runtime_dependencies import (
        HttpClientSessionProvider,
    )
    from logger.project_logger import ProjectLogger

from crawler.crawl_tasks.crawl_task import CrawlTask


class FetchOrchestrator:
    """Orchestrate one logical fetch operation."""

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        session_provider: HttpClientSessionProvider,
        head_preflight_executor: HeadPreflightExecutor,
        request_header_builder: RequestHeaderBuilder,
        request_context_builder: FetchRequestContextBuilder,
        attempt_executor: FetchAttemptExecutor,
        media_strategy_resolver: MediaFetchStrategyResolver,
        media_metadata_result_builder: MediaMetadataResultBuilder,
        retry_manager: RetryManager,
        logger: ProjectLogger,
        feed_alternate_resolver: FeedAlternateResolver | None = None,
    ) -> None:
        self._settings = settings
        self._session_provider = session_provider
        self._head_preflight_executor = head_preflight_executor
        self._request_header_builder = request_header_builder
        self._request_context_builder = request_context_builder
        self._attempt_executor = attempt_executor
        self._media_strategy_resolver = media_strategy_resolver
        self._media_metadata_result_builder = media_metadata_result_builder
        self._retry_manager = retry_manager
        self._logger = logger
        self._feed_alternate_resolver = feed_alternate_resolver

    async def fetch(
        self,
        task: CrawlTask,
        *,
        defer_if_rate_limited: bool = False,
    ) -> FetchResult:
        """Fetch the resource and return the normalized classified result."""
        context = self._request_context_builder.build(task=task)

        try:
            return await self._retry_manager.run_with_retry_rules(
                lambda: self._fetch_once(
                    context=context,
                    defer_if_rate_limited=defer_if_rate_limited,
                ),
                url=context.url,
            )
        except RetryableFetchError as exc:
            alternates = self._resolve_feed_alternates(task=task, exc=exc)
            if not alternates:
                raise

            last_error = exc
            for alternate_url in alternates:
                alternate_context = self._request_context_builder.build(
                    task=replace(task, url=alternate_url),
                )
                self._logger.info(
                    "feed_alternate_fallback",
                    original_url=task.url,
                    alternate_url=alternate_url,
                    error_type=self._retry_error_type(last_error),
                )
                try:
                    return await self._retry_manager.run_with_retry_rules(
                        partial(
                            self._fetch_once,
                            context=alternate_context,
                            defer_if_rate_limited=defer_if_rate_limited,
                        ),
                        url=alternate_context.url,
                    )
                except RetryableFetchError as alternate_error:
                    last_error = alternate_error

            raise last_error from exc

    async def _fetch_once(
        self,
        *,
        context: FetchRequestContext,
        defer_if_rate_limited: bool,
    ) -> FetchResult:
        self._logger.debug(
            "fetch_started",
            url=context.url,
            host=context.host,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
        )

        if MediaFetchStrategyResolver.is_embed_metadata_context(
            context=context
        ):
            self._logger.debug(
                "fetch_embed_metadata_short_circuit",
                url=context.url,
                host=context.host,
                requested_kind=context.requested_kind,
                asset_fetch_mode="embed_metadata",
            )
            return self._media_metadata_result_builder.build_embed_metadata(
                context=context,
            )

        session = await self._session_provider.get_session()
        request_headers = self._request_header_builder.build(
            url=context.url,
            host=context.host,
        )

        head_preflight_result = await self._run_head_preflight_if_enabled(
            context=context,
            session=session,
            request_headers=request_headers,
        )

        head_only_result = self._try_head_only_metadata(
            context=context,
            head_preflight_result=head_preflight_result,
        )
        if head_only_result is not None:
            return head_only_result

        result = await self._attempt_executor.execute(
            session=session,
            context=context,
            request_headers=request_headers,
            defer_if_rate_limited=(
                defer_if_rate_limited
                and not self._head_preflight_attempted(
                    head_preflight_result,
                )
            ),
            head_preflight_result=head_preflight_result,
        )

        if result is not None:
            return result

        raise IgnoredFetchError(
            reason="fetch_returns_none",
            observed_bytes=0,
        )

    async def _run_head_preflight_if_enabled(
        self,
        *,
        context: FetchRequestContext,
        session: ClientSession,
        request_headers: dict[str, str],
    ) -> HeadPreflightResult | None:
        if not self._settings.head_preflight_enabled:
            return None

        return await self._head_preflight_executor.run(
            context=context,
            session=session,
            request_headers=request_headers,
        )

    def _try_head_only_metadata(
        self,
        *,
        context: FetchRequestContext,
        head_preflight_result: HeadPreflightResult | None,
    ) -> FetchResult | None:
        if not self._media_strategy_resolver.should_build_head_only_result(
            settings=self._settings,
            context=context,
            head_preflight_result=head_preflight_result,
        ):
            return None
        if head_preflight_result is None:
            raise RuntimeError(
                "head-only oversized video requires HEAD preflight result"
            )
        return self._media_metadata_result_builder.build_head_only_metadata(
            context=context,
            head_preflight_result=head_preflight_result,
        )

    @staticmethod
    def _head_preflight_attempted(
        result: HeadPreflightResult | None,
    ) -> bool:
        if result is None:
            return False

        return result.attempted

    def _resolve_feed_alternates(
        self,
        *,
        task: CrawlTask,
        exc: RetryableFetchError,
    ) -> tuple[str, ...]:
        if (
            task.kind is not MediaKind.FEED
            or self._feed_alternate_resolver is None
        ):
            return ()
        return self._feed_alternate_resolver.alternates_for(
            url=task.url,
            status_code=exc.status_code,
            error_type=self._retry_error_type(exc),
        )

    @staticmethod
    def _retry_error_type(exc: RetryableFetchError) -> str:
        return str(
            exc.retry_error_kind or exc.retry_class or type(exc).__name__
        )


__all__ = ["FetchOrchestrator"]
