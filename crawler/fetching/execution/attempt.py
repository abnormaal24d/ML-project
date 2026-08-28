"""Execute one governed HTTP fetch attempt."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout

from config.collection.http_rules import (
    HttpStatusRulesSettings,
    TimeoutRulesSettings,
)
from crawler.classification.media_kind import MediaKind
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.errors.translator import (
    transport_client_error,
    transport_timeout_error,
)
from crawler.fetching.response.rate_limit_hints import ResponseRateLimitHints

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crawler.fetching.feedback.attempt_recorder import (
        FetchAttemptOutcomeRecorder,
    )
    from crawler.fetching.media.strategy import HeadPreflightResult
    from crawler.fetching.network.request import AiohttpRequestRunner
    from crawler.fetching.request.body_plan import BodyReadPlan
    from crawler.fetching.request.body_plan_resolver import (
        BodyReadPlanResolver,
    )
    from crawler.fetching.request.context import FetchRequestContext
    from crawler.fetching.response.cache import ConditionalRepresentationCache
    from crawler.fetching.response.processor import (
        FetchResponseProcessor,
    )
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.circuit_breaker.host_circuit_breaker import (
        HostCircuitBreaker,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.rate_limit.rate_limiter import RateLimiter


@dataclass(slots=True)
class FetchAttemptState:
    """Mutable outcome facts collected during one governed fetch attempt."""

    status_code: int
    bytes_downloaded: int
    quality_score: float | None
    should_record_feedback: bool
    final_url: str
    started_at: float
    retry_after_seconds: float | None = None


class FetchAttemptExecutor:
    """Execute one paced request attempt and record its host feedback."""

    _LONG_REQUEST_KINDS = frozenset(
        {MediaKind.AUDIO, MediaKind.VIDEO, MediaKind.DOCUMENT}
    )

    _NON_HOST_FAILURE_KINDS = frozenset(
        {
            "circuit_open",
            "peer_address_unavailable",
            "not_modified_force_unconditional",
        }
    )

    def __init__(
        self,
        *,
        status_rules: HttpStatusRulesSettings,
        rate_limiter: RateLimiter,
        now_utc: Callable[[], datetime],
        attempt_feedback: FetchAttemptOutcomeRecorder,
        response_processor: FetchResponseProcessor,
        conditional_representation_cache: ConditionalRepresentationCache,
        timeout_rules: TimeoutRulesSettings,
        large_body_threshold_bytes: int,
        body_read_plan_resolver: BodyReadPlanResolver,
        circuit_breaker: HostCircuitBreaker,
        host_normalizer: HostNormalizer,
        request_runner: AiohttpRequestRunner,
    ) -> None:
        self._status_rules = status_rules
        self._rate_limiter = rate_limiter
        self._now_utc = now_utc
        self._attempt_feedback = attempt_feedback
        self._response_processor = response_processor
        self._conditional_representation_cache = (
            conditional_representation_cache
        )
        self._timeout_rules = timeout_rules
        self._large_body_threshold_bytes = self._require_positive_int(
            large_body_threshold_bytes,
            field_name="large_body_threshold_bytes",
        )
        self._body_read_plan_resolver = body_read_plan_resolver
        self._circuit_breaker = circuit_breaker
        self._host_normalizer = host_normalizer
        self._request_runner = request_runner

    async def execute(
        self,
        *,
        session: ClientSession,
        context: FetchRequestContext,
        request_headers: Mapping[str, str],
        defer_if_rate_limited: bool,
        head_preflight_result: HeadPreflightResult | None,
    ) -> FetchResult | None:
        state = FetchAttemptState(
            status_code=0,
            bytes_downloaded=0,
            quality_score=None,
            should_record_feedback=False,
            final_url=context.url,
            started_at=asyncio.get_running_loop().time(),
        )

        try:
            return await self._execute_request(
                session=session,
                context=context,
                request_headers=request_headers,
                defer_if_rate_limited=defer_if_rate_limited,
                head_preflight_result=head_preflight_result,
                state=state,
            )
        except RetryableFetchError as exc:
            self._handle_retryable_error(context=context, state=state, exc=exc)
            raise
        except IgnoredFetchError as exc:
            await self._handle_ignored_error(
                context=context,
                state=state,
                exc=exc,
            )
            raise
        except asyncio.CancelledError as exc:
            await self._handle_cancellation(
                context=context,
                state=state,
                exc=exc,
            )
            raise
        except TimeoutError as exc:
            self._handle_transport_timeout(context=context, state=state)
            raise transport_timeout_error(exc) from exc
        except ClientError as exc:
            self._handle_transport_failure(context=context, state=state)
            raise transport_client_error(exc) from exc
        finally:
            await self._record_feedback(
                context=context,
                state=state,
            )

    async def _execute_request(
        self,
        *,
        session: ClientSession,
        context: FetchRequestContext,
        request_headers: Mapping[str, str],
        defer_if_rate_limited: bool,
        head_preflight_result: HeadPreflightResult | None,
        state: FetchAttemptState,
    ) -> FetchResult | None:
        circuit_decision = self._circuit_breaker.before_request(
            host=context.host
        )
        if not circuit_decision.allowed:
            raise RetryableFetchError(
                "host circuit is open",
                retry_class="circuit_breaker",
                retry_error_kind="circuit_open",
                retry_after_seconds=circuit_decision.retry_after_seconds,
            )

        read_plan = self._body_read_plan_resolver.build(
            context=context,
            request_headers=request_headers,
            head_preflight_result=head_preflight_result,
        )
        async with self._perform_request(
            session=session,
            context=context,
            read_plan=read_plan,
            defer_if_rate_limited=defer_if_rate_limited,
        ) as response:
            response_host = self._host_normalizer.require(response.url.host)
            state.retry_after_seconds = await self._apply_rate_limit_hints(
                context=context,
                response=response,
                response_host=response_host,
            )
            outcome = await self._response_processor.handle_response(
                context=context,
                response=response,
                read_plan=read_plan,
            )
            state.status_code = outcome.status_code
            state.bytes_downloaded = outcome.bytes_downloaded
            state.quality_score = outcome.quality_score
            state.should_record_feedback = outcome.should_record_feedback
            state.final_url = outcome.final_url
            if 200 <= outcome.status_code < 400:
                self._circuit_breaker.record_success(
                    host=response_host or context.host
                )
            return outcome.result

    def _handle_retryable_error(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
        exc: RetryableFetchError,
    ) -> None:
        if state.retry_after_seconds is not None:
            exc.retry_after_seconds = state.retry_after_seconds
        if self._should_record_circuit_failure(exc):
            self._circuit_breaker.record_failure(
                host=context.host,
                category=self._circuit_failure_category(exc),
            )
        if exc.status_code is not None:
            state.status_code = int(exc.status_code)
            state.should_record_feedback = (
                state.status_code in self._status_rules.rate_limiter_feedback
            )

    async def _handle_ignored_error(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
        exc: IgnoredFetchError,
    ) -> None:
        if exc.status_code is not None:
            state.status_code = int(exc.status_code)
            state.should_record_feedback = True
        if exc.final_url:
            state.final_url = str(exc.final_url)

        final_host = (
            self._host_normalizer.normalize(urlsplit(state.final_url).hostname)
            if state.final_url
            else None
        )
        await self._attempt_feedback.record_ignored_fetch(
            context=context,
            status_code=state.status_code,
            started_at=state.started_at,
            exc=exc,
            final_host=final_host,
        )

    async def _handle_cancellation(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
        exc: asyncio.CancelledError,
    ) -> None:
        await self._attempt_feedback.record_cancelled_fetch(
            context=context,
            status_code=state.status_code,
            final_url=state.final_url,
            started_at=state.started_at,
            bytes_downloaded=state.bytes_downloaded,
            exc=exc,
        )

    def _handle_transport_timeout(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
    ) -> None:
        state.status_code = 504
        state.should_record_feedback = True
        self._circuit_breaker.record_failure(
            host=context.host,
            category="transport_timeout",
        )

    def _handle_transport_failure(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
    ) -> None:
        state.status_code = 503
        state.should_record_feedback = True
        self._circuit_breaker.record_failure(
            host=context.host,
            category="transport_failure",
        )

    async def _apply_rate_limit_hints(
        self,
        *,
        context: FetchRequestContext,
        response: ClientResponse,
        response_host: str | None,
    ) -> float | None:
        hints = ResponseRateLimitHints.from_headers(
            response.headers,
            now=self._now_utc(),
        )
        if not hints.has_delay:
            return None

        return await self._rate_limiter.apply_response_rate_limit_hints(
            host=response_host or context.host,
            retry_after_seconds=hints.retry_after_seconds,
            rate_limit_remaining=hints.rate_limit_remaining,
            rate_limit_reset_seconds=hints.rate_limit_reset_seconds,
        )

    async def _record_feedback(
        self,
        *,
        context: FetchRequestContext,
        state: FetchAttemptState,
    ) -> None:
        if not state.should_record_feedback or state.status_code <= 0:
            return
        final_host = (
            self._host_normalizer.normalize(urlsplit(state.final_url).hostname)
            if state.final_url
            else None
        )
        await self._attempt_feedback.record_attempt_feedback(
            context=context,
            status_code=state.status_code,
            final_url=state.final_url,
            started_at=state.started_at,
            bytes_downloaded=state.bytes_downloaded,
            quality_score=state.quality_score,
            final_host=final_host,
        )

    @classmethod
    def _should_record_circuit_failure(
        cls,
        exc: RetryableFetchError,
    ) -> bool:
        return cls._retry_error_kind(exc) not in cls._NON_HOST_FAILURE_KINDS

    @classmethod
    def _circuit_failure_category(cls, exc: RetryableFetchError) -> str:
        if exc.status_code is not None:
            return f"http_{exc.status_code}"
        return cls._retry_error_kind(exc)

    @staticmethod
    def _retry_error_kind(exc: RetryableFetchError) -> str:
        return str(exc.retry_error_kind or exc.retry_class).strip().lower()

    def _build_client_timeout(
        self,
        *,
        requested_kind: MediaKind,
        planned_max_bytes: int,
    ) -> ClientTimeout:
        if not isinstance(requested_kind, MediaKind):
            raise TypeError("requested_kind must be a MediaKind")

        max_bytes = self._require_positive_int(
            planned_max_bytes,
            field_name="planned_max_bytes",
        )
        total_seconds = float(self._timeout_rules.request_timeout_seconds)
        if (
            requested_kind in self._LONG_REQUEST_KINDS
            or max_bytes >= self._large_body_threshold_bytes
        ):
            total_seconds = float(
                self._timeout_rules.large_media_request_timeout_seconds
            )

        connect_seconds = float(self._timeout_rules.connect_timeout_seconds)
        return ClientTimeout(
            total=total_seconds,
            connect=connect_seconds,
            sock_connect=connect_seconds,
            sock_read=None,
        )

    @staticmethod
    def _require_positive_int(
        value: object,
        *,
        field_name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")
        if value < 1:
            raise ValueError(f"{field_name} must be at least 1")
        return value

    @asynccontextmanager
    async def _perform_request(
        self,
        *,
        session: ClientSession,
        context: FetchRequestContext,
        read_plan: BodyReadPlan,
        defer_if_rate_limited: bool,
    ) -> AsyncIterator[ClientResponse]:
        """Delegate transport and redirect handling to the shared runner."""

        timeout = self._build_client_timeout(
            requested_kind=context.requested_kind,
            planned_max_bytes=read_plan.max_bytes,
        )
        async with self._request_runner.perform(
            session=session,
            method="GET",
            url=context.url,
            source_name=context.source_name,
            base_headers=dict(read_plan.headers),
            timeout=timeout,
            defer_if_rate_limited=defer_if_rate_limited,
            enrich_headers=self._conditional_representation_cache.enrich_headers,
        ) as response:
            yield response
