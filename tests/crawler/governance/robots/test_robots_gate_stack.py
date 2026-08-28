"""End-to-end robots enforcement through checker, cache, gate, and runner.

The stack mirrors production composition: the request runner authorizes each
URL through the robots gate, which loads robots.txt once per host through the
parser cache, applies the allow/block/defer policy, and only then lets the
page request reach the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from config.collection.governance import RobotsSettings
from config.collection.http_rules import (
    NetworkAccessSettings,
    TimeoutRulesSettings,
)
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.network.request import AiohttpRequestRunner
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_check_result import (
    RobotsDecision,
)
from crawler.governance.robots.robots_checker import RobotsChecker
from crawler.governance.robots.robots_decision_evaluator import (
    RobotsDecisionEvaluator,
)
from crawler.governance.robots.robots_error_classifier import (
    RobotsErrorClassifier,
)
from crawler.governance.robots.robots_error_resolver import (
    RobotsErrorResolver,
    RobotsErrorResolverDependencies,
    RobotsErrorResolverRules,
)
from crawler.governance.robots.robots_fallback_rules import (
    RobotsFallbackRules,
)
from crawler.governance.robots.robots_fetch_errors import (
    RobotsHttpStatusError,
)
from crawler.governance.robots.robots_fetcher import RobotsFetchResult
from crawler.governance.robots.robots_host_rules_store import (
    RobotsHostRulesStore,
)
from crawler.governance.robots.robots_parser_cache import RobotsParserCache
from crawler.governance.robots.robots_parser_loader import (
    RobotsParserLoader,
)
from crawler.governance.robots.robots_request_gate import RobotsRequestGate
from crawler.governance.robots.robots_unknown_result_suppressor import (
    RobotsUnknownResultSuppressor,
)
from crawler.governance.robots.robots_url_resolver import RobotsUrlResolver
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from tests.support.logging import TEST_LOGGER

USER_AGENT = "MultimodalCrawler/1.0"

DISALLOW_PRIVATE_BODY = b"User-agent: *\nDisallow: /private\nAllow: /\n"


class _CountingFetcher:
    """Robots transport double that counts fetches and serves scripted data."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = DISALLOW_PRIVATE_BODY,
        retry_after_seconds: float | None = None,
        error: Exception | None = None,
    ) -> None:
        self._status_code = status_code
        self._body = body
        self._retry_after_seconds = retry_after_seconds
        self._error = error
        self.calls: list[str] = []

    async def fetch(
        self,
        *,
        robots_url: str,
        headers: dict[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> RobotsFetchResult:
        del headers, timeout_seconds, max_body_bytes
        self.calls.append(robots_url)
        if self._error is not None:
            raise self._error
        if not 200 <= self._status_code < 300:
            raise RobotsHttpStatusError(
                status_code=self._status_code,
                error_type="HTTPError",
                headers={},
                final_url=robots_url,
                requested_url=robots_url,
                retry_after_seconds=self._retry_after_seconds,
            )
        return RobotsFetchResult(
            requested_url=robots_url,
            final_url=robots_url,
            status_code=self._status_code,
            headers={},
            body=self._body,
            latency_seconds=0.01,
            retry_after_seconds=self._retry_after_seconds,
        )


class _StackRateLimiter:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, bool]] = []
        self.crawl_delay_calls: list[tuple[str | None, float | None]] = []

    async def acquire_for_fetch(
        self,
        *,
        host: str | None,
        defer_if_rate_limited: bool,
    ) -> None:
        self.acquired.append((str(host), bool(defer_if_rate_limited)))

    async def set_host_crawl_delay(
        self,
        *,
        host: str | None,
        crawl_delay_seconds: float | None,
    ) -> None:
        self.crawl_delay_calls.append((host, crawl_delay_seconds))


class _FakeRedirector:
    max_location_length = 64

    def validate_hop(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
        source_name: str | None,
    ) -> None:
        del current_url, target_url, redirect_count, source_name


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        url: Any = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.connected_peer = ("127.0.0.1", 443)

    def release(self) -> None:
        pass


class _FakeSession:
    """Counts page requests; any request in block tests is a failure."""

    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> _FakeResponse:
        del kwargs
        self.calls.append((method, url))
        if not self._responses:
            raise AssertionError(f"unexpected page request for {method} {url}")
        return self._responses.pop(0)


def _build_stack(
    *,
    fetcher: _CountingFetcher,
    respect_crawl_delay: bool = True,
    max_crawl_delay_s: float = 30.0,
    mode: str = "enforce",
) -> tuple[
    _CountingFetcher,
    _StackRateLimiter,
    AiohttpRequestRunner,
    RobotsChecker,
]:
    host_normalizer = HostNormalizer()
    loader = RobotsParserLoader(
        user_agent=USER_AGENT,
        accept_language_header=None,
        accept_compressed=False,
        fetcher=fetcher,  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    cache = RobotsParserCache(
        cache_ttl_s=3600,
        error_cache_ttl_s=60,
        parser_loader=loader,
        host_normalizer=host_normalizer,
        logger=TEST_LOGGER,
    )
    url_resolver = RobotsUrlResolver(
        robots_path="/robots.txt",
        host_normalizer=host_normalizer,
        logger=TEST_LOGGER,
    )
    evaluator = RobotsDecisionEvaluator(
        respect_crawl_delay=respect_crawl_delay,
        max_crawl_delay_s=max_crawl_delay_s,
        user_agent=USER_AGENT,
        logger=TEST_LOGGER,
    )
    fallback_rules = RobotsFallbackRules(
        http_403_allow_host_suffixes=None,
        host_normalizer=host_normalizer,
    )
    error_resolver = RobotsErrorResolver(
        rules=RobotsErrorResolverRules(),
        dependencies=RobotsErrorResolverDependencies(
            classifier=RobotsErrorClassifier(),
            fallback_rules=fallback_rules,
        ),
    )
    host_rules_store = RobotsHostRulesStore(
        host_normalizer=host_normalizer,
        logger=TEST_LOGGER,
    )
    suppressor = RobotsUnknownResultSuppressor(
        ttl_seconds=60.0,
        prune_every=1000,
        max_entries=1024,
        host_normalizer=host_normalizer,
    )
    metrics = CollectionMetrics(
        enabled=True,
        logger=TEST_LOGGER,
        host_normalizer=host_normalizer,
    )
    settings = RobotsSettings(mode=mode)
    checker = RobotsChecker(
        settings=settings,
        timeout_rules=TimeoutRulesSettings(),
        decision_evaluator=evaluator,
        error_rules=error_resolver,
        robots_url_resolver=url_resolver,
        parser_cache=cache,
        host_rules_store=host_rules_store,
        user_agent=USER_AGENT,
        host_normalizer=host_normalizer,
        duplicate_result_tracker=suppressor,
        logger=TEST_LOGGER,
        metrics=metrics,
    )
    limiter = _StackRateLimiter()
    gate = RobotsRequestGate(
        checker=checker,
        settings=settings,
        rate_limiter=limiter,  # type: ignore[arg-type]
        host_normalizer=host_normalizer,
        metrics=metrics,
        logger=TEST_LOGGER,
    )

    class _FakeNetworkAddressGuard:
        def __init__(self) -> None:
            self.settings = NetworkAccessSettings()

        def rejection_reason_for_address(self, address: str) -> str | None:
            return None

    runner = AiohttpRequestRunner(
        redirector=_FakeRedirector(),  # type: ignore[arg-type]
        host_normalizer=host_normalizer,
        rate_limiter=limiter,  # type: ignore[arg-type]
        robots_gate=gate,
        network_address_guard=_FakeNetworkAddressGuard(),
    )
    return fetcher, limiter, runner, checker


async def _perform(
    *,
    runner: AiohttpRequestRunner,
    session: _FakeSession,
    url: str,
    method: str = "GET",
) -> Any:
    async with runner.perform(
        session=session,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        url=url,
        source_name="test-source",
        base_headers={"User-Agent": USER_AGENT},
        timeout=aiohttp.ClientTimeout(total=5.0),
        defer_if_rate_limited=False,
    ) as response:
        return response


@pytest.mark.asyncio
async def test_disallowed_url_fetches_robots_once_and_never_the_target() -> (
    None
):
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(body=DISALLOW_PRIVATE_BODY)
    )
    session = _FakeSession([])

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/private",
        )

    assert raised.value.reason == "robots_disallowed"
    assert fetcher.calls == ["https://example.test/robots.txt"]
    assert session.calls == []


@pytest.mark.asyncio
async def test_allowed_url_proceeds_to_network_after_one_robots_fetch() -> (
    None
):
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(body=DISALLOW_PRIVATE_BODY)
    )
    session = _FakeSession(
        [_FakeResponse(status=200, url=SimpleNamespace(host="example.test"))]
    )

    response = await _perform(
        runner=runner,
        session=session,
        url="https://example.test/public",
    )

    assert response.status == 200
    assert fetcher.calls == ["https://example.test/robots.txt"]
    assert session.calls == [("GET", "https://example.test/public")]


@pytest.mark.asyncio
async def test_absent_robots_404_allows_crawling() -> None:
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(status_code=404)
    )
    session = _FakeSession(
        [_FakeResponse(status=200, url=SimpleNamespace(host="example.test"))]
    )

    response = await _perform(
        runner=runner,
        session=session,
        url="https://example.test/page",
    )

    assert response.status == 200
    assert session.calls == [("GET", "https://example.test/page")]


@pytest.mark.asyncio
async def test_rate_limited_robots_defers_with_retry_after() -> None:
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(
            status_code=429,
            retry_after_seconds=12.0,
        )
    )
    session = _FakeSession([])

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/page",
        )

    assert raised.value.retry_class == "robots"
    assert raised.value.retry_after_seconds == 12.0
    assert session.calls == []


@pytest.mark.asyncio
async def test_server_error_robots_defers_fail_closed() -> None:
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(status_code=503)
    )
    session = _FakeSession([])

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/page",
        )

    assert raised.value.retry_class == "robots"
    assert session.calls == []


@pytest.mark.asyncio
async def test_network_error_robots_defers_fail_closed() -> None:
    from crawler.governance.robots.robots_fetch_errors import (
        RobotsNetworkError,
    )

    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(
            error=RobotsNetworkError(error_type="ClientConnectorError")
        )
    )
    session = _FakeSession([])

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/page",
        )

    assert raised.value.retry_class == "robots"
    assert session.calls == []


@pytest.mark.asyncio
async def test_forbidden_robots_blocks_fail_closed() -> None:
    fetcher, _limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(status_code=403)
    )
    session = _FakeSession([])

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/page",
        )

    assert raised.value.reason == "robots_disallowed"
    assert session.calls == []


@pytest.mark.asyncio
async def test_crawl_delay_is_applied_to_pacing() -> None:
    body = b"User-agent: *\nCrawl-delay: 2.5\nDisallow:\n"
    fetcher, limiter, runner, _checker = _build_stack(
        fetcher=_CountingFetcher(body=body)
    )
    session = _FakeSession(
        [_FakeResponse(status=200, url=SimpleNamespace(host="example.test"))]
    )

    await _perform(
        runner=runner,
        session=session,
        url="https://example.test/page",
    )

    assert limiter.crawl_delay_calls == [("example.test", 2.5)]
    assert session.calls == [("GET", "https://example.test/page")]


@pytest.mark.asyncio
async def test_crawl_delay_over_max_is_operational_block() -> None:
    body = b"User-agent: *\nCrawl-delay: 60\nDisallow:\n"
    fetcher, _limiter, runner, checker = _build_stack(
        fetcher=_CountingFetcher(body=body),
        max_crawl_delay_s=30.0,
    )
    session = _FakeSession([])

    result = await checker.check("https://example.test/page")
    assert result.decision == RobotsDecision.UNKNOWN
    assert result.reason == "robots_crawl_delay_operationally_unsupported"

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/page",
        )

    assert raised.value.reason == "robots_disallowed"
    assert session.calls == []
