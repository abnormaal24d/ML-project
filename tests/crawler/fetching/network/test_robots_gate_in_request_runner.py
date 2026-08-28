"""Robots gate integration tests for the shared AiohttpRequestRunner.

The runner must authorize every hop (initial URL, HEAD preflights, and each
redirect target) through the robots gate before touching the rate limiter or
the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from config.collection.http_rules import NetworkAccessSettings
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.network.request import AiohttpRequestRunner
from crawler.governance.domains.host_normalizer import HostNormalizer


class _FakeGate:
    def __init__(
        self,
        *,
        events: list[str],
        block: bool = False,
        defer: bool = False,
    ) -> None:
        self._events = events
        self.calls: list[str] = []
        self._block = block
        self._defer = defer

    async def authorize(
        self,
        *,
        url: str,
    ) -> None:
        self._events.append(f"gate:{url}")
        self.calls.append(url)
        if self._block:
            raise IgnoredFetchError(
                reason="robots_disallowed",
                observed_bytes=0,
                metrics_recorded=True,
            )
        if self._defer:
            raise RetryableFetchError(
                "robots check defers crawling",
                retry_class="robots",
                retry_error_kind="http_503_unknown",
                retry_after_seconds=10.0,
            )


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


class _FakeRateLimiter:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.acquired: list[tuple[str, bool]] = []

    async def acquire_for_fetch(
        self,
        *,
        host: str | None,
        defer_if_rate_limited: bool,
    ) -> None:
        self._events.append(f"acquire:{host}")
        self.acquired.append((str(host), bool(defer_if_rate_limited)))


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
    def __init__(
        self,
        responses: list[_FakeResponse],
        events: list[str],
    ) -> None:
        self._responses = list(responses)
        self._events = events
        self.calls: list[tuple[str, str]] = []

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> _FakeResponse:
        del kwargs
        self._events.append(f"request:{url}")
        self.calls.append((method, url))
        if not self._responses:
            raise AssertionError(
                f"unexpected session.request for {method} {url}"
            )
        return self._responses.pop(0)


def _runner(
    *,
    gate: _FakeGate,
    events: list[str],
) -> AiohttpRequestRunner:
    return AiohttpRequestRunner(
        redirector=_FakeRedirector(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rate_limiter=_FakeRateLimiter(events),  # type: ignore[arg-type]
        robots_gate=gate,  # type: ignore[arg-type]
        network_address_guard=_FakeNetworkAddressGuard(),
    )


class _FakeNetworkAddressGuard:
    def __init__(self) -> None:
        self.settings = NetworkAccessSettings()

    def rejection_reason_for_address(self, address: str) -> str | None:
        return None


async def _perform(
    *,
    runner: AiohttpRequestRunner,
    session: _FakeSession,
    method: str = "GET",
    url: str,
) -> _FakeResponse | None:
    yielded: _FakeResponse | None = None
    async with runner.perform(
        session=session,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        url=url,
        source_name="test-source",
        base_headers={"User-Agent": "test-agent"},
        timeout=aiohttp.ClientTimeout(total=5.0),
        defer_if_rate_limited=False,
    ) as response:
        yielded = response
    return yielded


def test_runner_cannot_be_constructed_without_robots_gate() -> None:
    with pytest.raises(TypeError):
        AiohttpRequestRunner(
            redirector=_FakeRedirector(),  # type: ignore[arg-type]
            host_normalizer=HostNormalizer(),
            rate_limiter=_FakeRateLimiter([]),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_gates_initial_url_before_rate_limit_and_request() -> None:
    events: list[str] = []
    gate = _FakeGate(events=events)
    runner = _runner(gate=gate, events=events)
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url=SimpleNamespace(host="example.test"),
            )
        ],
        events,
    )

    await _perform(
        runner=runner,
        session=session,
        url="https://example.test/resource",
    )

    assert events == [
        "gate:https://example.test/resource",
        "acquire:example.test",
        "request:https://example.test/resource",
    ]
    assert gate.calls == ["https://example.test/resource"]


@pytest.mark.asyncio
async def test_gates_every_redirect_hop() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    events: list[str] = []
    gate = _FakeGate(events=events)
    runner = _runner(gate=gate, events=events)
    session = _FakeSession(
        [
            _FakeResponse(status=302, headers={"Location": final}),
            _FakeResponse(status=200),
        ],
        events,
    )

    await _perform(runner=runner, session=session, url=start)

    assert events == [
        "gate:https://example.test/start",
        "acquire:example.test",
        "request:https://example.test/start",
        "gate:https://cdn.example.test/final",
        "acquire:cdn.example.test",
        "request:https://cdn.example.test/final",
    ]
    assert gate.calls == [start, final]


@pytest.mark.asyncio
async def test_gate_block_prevents_rate_acquire_and_request() -> None:
    events: list[str] = []
    gate = _FakeGate(events=events, block=True)
    runner = _runner(gate=gate, events=events)
    session = _FakeSession([], events)

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/private",
        )

    assert raised.value.reason == "robots_disallowed"
    assert events == ["gate:https://example.test/private"]
    assert session.calls == []


@pytest.mark.asyncio
async def test_gate_defer_propagates_retryable_before_network() -> None:
    events: list[str] = []
    gate = _FakeGate(events=events, defer=True)
    runner = _runner(gate=gate, events=events)
    session = _FakeSession([], events)

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/private",
        )

    assert raised.value.retry_class == "robots"
    assert raised.value.retry_after_seconds == 10.0
    assert events == ["gate:https://example.test/private"]
    assert session.calls == []


@pytest.mark.asyncio
async def test_head_preflight_is_gated() -> None:
    events: list[str] = []
    gate = _FakeGate(events=events)
    runner = _runner(gate=gate, events=events)
    session = _FakeSession(
        [_FakeResponse(status=200)],
        events,
    )

    await _perform(
        runner=runner,
        session=session,
        method="HEAD",
        url="https://example.test/head-check",
    )

    assert gate.calls == ["https://example.test/head-check"]
    assert session.calls == [("HEAD", "https://example.test/head-check")]
