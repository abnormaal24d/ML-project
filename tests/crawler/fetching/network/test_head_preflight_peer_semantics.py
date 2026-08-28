"""Regression coverage for HEAD peer-metadata fallback semantics."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from config.collection.fetching import FetcherSettings
from config.collection.http_rules import (
    NetworkAccessSettings,
    TimeoutRulesSettings,
)
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.fetcher import FetchOrchestrator
from crawler.fetching.media.strategy import HeadPreflightResult
from crawler.fetching.network.preflight.executor import HeadPreflightExecutor
from crawler.fetching.network.request import AiohttpRequestRunner
from crawler.fetching.request.context import FetchRequestContext
from crawler.governance.domains.host_normalizer import HostNormalizer


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _ResponseEvaluator:
    def handle_head_result(self, **kwargs: object) -> HeadPreflightResult:
        del kwargs
        raise AssertionError(
            "an untrusted HEAD response must not be evaluated"
        )


class _PeerFailureRequestRunner:
    def __init__(self, *, head_error: BaseException) -> None:
        self._head_error = head_error
        self.methods: list[str] = []

    @asynccontextmanager
    async def perform(self, *, method: str, **kwargs: object):  # type: ignore[no-untyped-def]
        del kwargs
        self.methods.append(method)
        if method == "HEAD":
            raise self._head_error
        yield SimpleNamespace()


class _FakeResponse:
    def __init__(
        self,
        *,
        connected_peer: object,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = SimpleNamespace(host="example.test")
        self.connected_peer = connected_peer
        self.release_calls = 0
        self.close_calls = 0

    def release(self) -> None:
        self.release_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _SequentialSession:
    def __init__(self, *, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeResponse:
        del kwargs
        self.calls.append((method, url))
        if not self._responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        return self._responses.pop(0)


class _Redirector:
    max_location_length = 4096

    def validate_hop(self, **kwargs: object) -> None:
        del kwargs


class _RateLimiter:
    async def acquire_for_fetch(self, **kwargs: object) -> None:
        del kwargs


class _RobotsGate:
    async def authorize(self, **kwargs: object) -> None:
        del kwargs


class _NetworkAddressGuard:
    def __init__(self) -> None:
        self.settings = NetworkAccessSettings()

    def rejection_reason_for_address(self, address: str) -> str | None:
        del address
        return None


def _request_runner() -> AiohttpRequestRunner:
    return AiohttpRequestRunner(
        redirector=_Redirector(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rate_limiter=_RateLimiter(),  # type: ignore[arg-type]
        robots_gate=_RobotsGate(),  # type: ignore[arg-type]
        network_address_guard=_NetworkAddressGuard(),  # type: ignore[arg-type]
    )


class _SessionProvider:
    def __init__(self, *, session: object) -> None:
        self._session = session

    async def get_session(self) -> object:
        return self._session


class _RequestHeaderBuilder:
    def build(self, **kwargs: object) -> dict[str, str]:
        del kwargs
        return {"User-Agent": "test-agent"}


class _RequestContextBuilder:
    def __init__(self, *, context: FetchRequestContext) -> None:
        self._context = context

    def build(self, **kwargs: object) -> FetchRequestContext:
        del kwargs
        return self._context


class _RetryManager:
    async def run_with_retry_rules(
        self,
        operation: object,
        *,
        url: str,
    ) -> object:
        del url
        return await operation()  # type: ignore[operator]


class _MediaStrategyResolver:
    def should_build_head_only_result(self, **kwargs: object) -> bool:
        del kwargs
        return False


class _GetAttemptExecutor:
    def __init__(
        self,
        *,
        request_runner: Any,
        result: object,
    ) -> None:
        self._request_runner = request_runner
        self._result = result
        self.head_results: list[HeadPreflightResult | None] = []

    async def execute(
        self,
        *,
        session: object,
        context: FetchRequestContext,
        request_headers: dict[str, str],
        defer_if_rate_limited: bool,
        head_preflight_result: HeadPreflightResult | None,
    ) -> object:
        self.head_results.append(head_preflight_result)
        async with self._request_runner.perform(
            session=session,
            method="GET",
            url=context.url,
            source_name=context.source_name,
            base_headers=request_headers,
            timeout=object(),
            defer_if_rate_limited=defer_if_rate_limited,
        ):
            pass
        return self._result


def _context() -> FetchRequestContext:
    return FetchRequestContext(
        url="https://example.test/resource",
        host="example.test",
        source_name="test-source",
        requested_kind=MediaKind.IMAGE,
        acceptance_mode="strict",
        acceptance=SimpleNamespace(),
    )


def _head_executor(
    *,
    request_runner: _PeerFailureRequestRunner,
) -> HeadPreflightExecutor:
    return HeadPreflightExecutor(
        settings=FetcherSettings(),
        timeout_rules=TimeoutRulesSettings(),
        request_runner=request_runner,  # type: ignore[arg-type]
        response_evaluator=_ResponseEvaluator(),  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
        feedback_reporter=None,
        host_normalizer=HostNormalizer(),
        host_allowlist=frozenset(),
        monotonic_seconds=lambda: 0.0,
    )


@pytest.mark.asyncio
async def test_peer_address_unavailable_is_a_soft_head_failure() -> None:
    error = RetryableFetchError(
        "connected peer address is unavailable",
        retry_class="local_transport_metadata",
        retry_error_kind="peer_address_unavailable",
    )
    request_runner = _PeerFailureRequestRunner(head_error=error)

    result = await _head_executor(request_runner=request_runner).run(
        context=_context(),
        session=object(),  # type: ignore[arg-type]
        request_headers={"User-Agent": "test-agent"},
    )

    assert request_runner.methods == ["HEAD"]
    assert result.attempted is True
    assert result.allowed is True
    assert result.status_code is None
    assert result.failure_type == "RetryableFetchError"
    assert result.soft_rejected is True
    assert result.rejection_reason == "peer_address_unavailable"
    assert result.record_kind is MediaKind.IMAGE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        RetryableFetchError(
            "robots state requires a retry",
            retry_class="robots",
            retry_error_kind="http_503_unknown",
        ),
        RetryableFetchError(
            "connected peer address is unavailable",
            retry_class="fetch_retryable",
            retry_error_kind="peer_address_unavailable",
        ),
    ),
)
async def test_other_retryable_head_failures_propagate(
    error: RetryableFetchError,
) -> None:
    request_runner = _PeerFailureRequestRunner(head_error=error)

    with pytest.raises(RetryableFetchError) as raised:
        await _head_executor(request_runner=request_runner).run(
            context=_context(),
            session=object(),  # type: ignore[arg-type]
            request_headers={"User-Agent": "test-agent"},
        )

    assert raised.value is error
    assert request_runner.methods == ["HEAD"]


@pytest.mark.asyncio
async def test_blocked_peer_head_failure_remains_hard_rejection() -> None:
    error = IgnoredFetchError(
        reason="blocked_connected_peer:loopback",
        observed_bytes=0,
    )
    request_runner = _PeerFailureRequestRunner(head_error=error)

    with pytest.raises(IgnoredFetchError) as raised:
        await _head_executor(request_runner=request_runner).run(
            context=_context(),
            session=object(),  # type: ignore[arg-type]
            request_headers={"User-Agent": "test-agent"},
        )

    assert raised.value is error
    assert request_runner.methods == ["HEAD"]


@pytest.mark.asyncio
async def test_fetch_orchestrator_runs_get_after_soft_head_peer_failure() -> (
    None
):
    context = _context()
    head_response = _FakeResponse(connected_peer=None)
    get_response = _FakeResponse(
        connected_peer=("93.184.216.34", 443),
    )
    session = _SequentialSession(
        responses=[head_response, get_response],
    )
    request_runner = _request_runner()
    expected_result = object()
    attempt_executor = _GetAttemptExecutor(
        request_runner=request_runner,
        result=expected_result,
    )
    fetcher = FetchOrchestrator(
        settings=FetcherSettings(),
        session_provider=_SessionProvider(session=session),  # type: ignore[arg-type]
        head_preflight_executor=_head_executor(
            request_runner=request_runner,
        ),
        request_header_builder=_RequestHeaderBuilder(),  # type: ignore[arg-type]
        request_context_builder=_RequestContextBuilder(
            context=context,
        ),  # type: ignore[arg-type]
        attempt_executor=attempt_executor,  # type: ignore[arg-type]
        media_strategy_resolver=_MediaStrategyResolver(),  # type: ignore[arg-type]
        media_metadata_result_builder=object(),  # type: ignore[arg-type]
        retry_manager=_RetryManager(),  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
    )

    result = await fetcher.fetch(
        CrawlTask(url=context.url, source_name=context.source_name),
    )

    assert result is expected_result
    assert session.calls == [
        ("HEAD", context.url),
        ("GET", context.url),
    ]
    assert head_response.close_calls == 1
    assert get_response.release_calls == 1
    assert len(attempt_executor.head_results) == 1
    head_result = attempt_executor.head_results[0]
    assert head_result is not None
    assert head_result.soft_rejected is True
    assert head_result.rejection_reason == "peer_address_unavailable"


@pytest.mark.asyncio
async def test_fetch_orchestrator_restarts_get_after_redirect_hop_peer_failure() -> (
    None
):
    context = _context()
    redirect_url = "https://cdn.example.test/resource"
    head_redirect_response = _FakeResponse(
        status=302,
        headers={"Location": redirect_url},
        connected_peer=("93.184.216.34", 443),
    )
    head_unavailable_response = _FakeResponse(connected_peer=None)
    get_response = _FakeResponse(
        connected_peer=("93.184.216.34", 443),
    )
    session = _SequentialSession(
        responses=[
            head_redirect_response,
            head_unavailable_response,
            get_response,
        ],
    )
    request_runner = _request_runner()
    expected_result = object()
    attempt_executor = _GetAttemptExecutor(
        request_runner=request_runner,
        result=expected_result,
    )
    fetcher = FetchOrchestrator(
        settings=FetcherSettings(),
        session_provider=_SessionProvider(session=session),  # type: ignore[arg-type]
        head_preflight_executor=_head_executor(
            request_runner=request_runner,
        ),
        request_header_builder=_RequestHeaderBuilder(),  # type: ignore[arg-type]
        request_context_builder=_RequestContextBuilder(
            context=context,
        ),  # type: ignore[arg-type]
        attempt_executor=attempt_executor,  # type: ignore[arg-type]
        media_strategy_resolver=_MediaStrategyResolver(),  # type: ignore[arg-type]
        media_metadata_result_builder=object(),  # type: ignore[arg-type]
        retry_manager=_RetryManager(),  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
    )

    result = await fetcher.fetch(
        CrawlTask(url=context.url, source_name=context.source_name),
    )

    assert result is expected_result
    assert session.calls == [
        ("HEAD", context.url),
        ("HEAD", redirect_url),
        ("GET", context.url),
    ]
    assert head_redirect_response.release_calls == 1
    assert head_unavailable_response.close_calls == 1
    assert get_response.release_calls == 1


@pytest.mark.asyncio
async def test_fetch_orchestrator_does_not_fallback_after_blocked_head_peer() -> (
    None
):
    context = _context()
    request_runner = _PeerFailureRequestRunner(
        head_error=IgnoredFetchError(
            reason="blocked_connected_peer:loopback",
            observed_bytes=0,
        )
    )
    attempt_executor = _GetAttemptExecutor(
        request_runner=request_runner,
        result=object(),
    )
    fetcher = FetchOrchestrator(
        settings=FetcherSettings(),
        session_provider=_SessionProvider(session=object()),  # type: ignore[arg-type]
        head_preflight_executor=_head_executor(
            request_runner=request_runner,
        ),
        request_header_builder=_RequestHeaderBuilder(),  # type: ignore[arg-type]
        request_context_builder=_RequestContextBuilder(
            context=context,
        ),  # type: ignore[arg-type]
        attempt_executor=attempt_executor,  # type: ignore[arg-type]
        media_strategy_resolver=_MediaStrategyResolver(),  # type: ignore[arg-type]
        media_metadata_result_builder=object(),  # type: ignore[arg-type]
        retry_manager=_RetryManager(),  # type: ignore[arg-type]
        logger=_Logger(),  # type: ignore[arg-type]
    )

    with pytest.raises(IgnoredFetchError, match="blocked_connected_peer"):
        await fetcher.fetch(
            CrawlTask(url=context.url, source_name=context.source_name),
        )

    assert request_runner.methods == ["HEAD"]
    assert attempt_executor.head_results == []
