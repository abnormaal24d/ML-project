"""Guard and redirect-policy tests for AiohttpRobotsFetcher.

The robots transport must validate every target (initial URL, each redirect
hop, and the connected peer) with the same network security primitives as the
page transport, while allowing cross-authority redirects without source
scopes (RFC 9309).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from config.collection.http_rules import (
    NetworkAccessSettings,
    RedirectRulesSettings,
)
from crawler.fetching.network.robots.fetcher import AiohttpRobotsFetcher
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from crawler.governance.robots.robots_fetch_errors import (
    RobotsHttpStatusError,
    RobotsNetworkError,
    RobotsRedirectRejectedError,
)
from crawler.governance.robots.robots_fetcher import RobotsFetchResult
from tests.support.logging import TEST_LOGGER


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        url: str = "",
        connected_peer: tuple[str, int] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.connected_peer = connected_peer
        self._body = body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def close(self) -> None:
        pass

    @property
    def content(self) -> _FakeResponse:
        return self

    async def readexactly(self, n: int) -> bytes:
        if len(self._body) >= n:
            return self._body[:n]
        raise asyncio.IncompleteReadError(self._body, n)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: Any = None,
        allow_redirects: bool = False,
    ) -> _FakeResponse:
        del headers, timeout, allow_redirects
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected session.get for {url}")
        return self._responses.pop(0)


class _FakeSessionProvider:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def get_session(self) -> _FakeSession:
        return self._session


class _FakeLimiter:
    def __init__(self) -> None:
        self.acquired: list[str | None] = []
        self.reported: list[tuple[str | None, int, float]] = []
        self.response_hints: list[
            tuple[str | None, float | None, int | None, float | None]
        ] = []

    async def acquire(
        self,
        host: str | None,
    ) -> None:
        self.acquired.append(host)

    async def report_result(
        self,
        *,
        host: str | None,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        self.reported.append((host, status_code, latency_seconds))

    async def apply_response_rate_limit_hints(
        self,
        *,
        host: str | None,
        retry_after_seconds: float | None = None,
        rate_limit_remaining: int | None = None,
        rate_limit_reset_seconds: float | None = None,
    ) -> float | None:
        self.response_hints.append(
            (
                host,
                retry_after_seconds,
                rate_limit_remaining,
                rate_limit_reset_seconds,
            )
        )
        candidates = tuple(
            value
            for value in (
                retry_after_seconds,
                rate_limit_reset_seconds
                if rate_limit_remaining == 0
                else None,
            )
            if value is not None and value > 0
        )
        return max(candidates) if candidates else None


class _FakeClock:
    def now(self) -> datetime:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)


class _HostExtractor:
    def extract(self, url: str) -> str:
        return url.split("/")[2]


class _SchemeRules:
    def is_allowed(self, url: str) -> bool:
        return url.startswith(("http://", "https://"))


class _Blacklist:
    def contains(self, *, url: str) -> bool:
        del url
        return False


def _real_redirector(
    *,
    guard: NetworkAddressGuard,
) -> RedirectRulesValidator:
    return RedirectRulesValidator(
        settings=RedirectRulesSettings(),
        host_extractor=_HostExtractor(),  # type: ignore[arg-type]
        url_validator=_SchemeRules(),  # type: ignore[arg-type]
        blacklist_repository=_Blacklist(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        logger=TEST_LOGGER,
        network_access_guard=guard,
    )


def _real_guard() -> NetworkAddressGuard:
    return NetworkAddressGuard(
        settings=NetworkAccessSettings(),
        host_normalizer=HostNormalizer(),
        logger=TEST_LOGGER,
    )


def _fetcher(
    *,
    session: _FakeSession,
    limiter: _FakeLimiter | None = None,
    redirector: RedirectRulesValidator | None = None,
    guard: NetworkAddressGuard | None = None,
) -> AiohttpRobotsFetcher:
    network_guard = guard or _real_guard()
    return AiohttpRobotsFetcher(
        session_provider=_FakeSessionProvider(session),  # type: ignore[arg-type]
        rate_limiter=limiter or _FakeLimiter(),  # type: ignore[arg-type]
        redirector=redirector or _real_redirector(guard=network_guard),
        network_address_guard=network_guard,
        host_normalizer=HostNormalizer(),
        clock=_FakeClock(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )


async def _fetch(
    fetcher: AiohttpRobotsFetcher,
    *,
    robots_url: str,
) -> RobotsFetchResult:
    return await fetcher.fetch(
        robots_url=robots_url,
        headers={"User-Agent": "test-agent"},
        timeout_seconds=5.0,
        max_body_bytes=4096,
    )


@pytest.mark.asyncio
async def test_rejects_unsafe_initial_url_before_any_request() -> None:
    session = _FakeSession([])
    fetcher = _fetcher(session=session)

    with pytest.raises(RobotsRedirectRejectedError) as raised:
        await _fetch(
            fetcher,
            robots_url="http://127.0.0.1/robots.txt",
        )

    assert "blocked_initial_url" in raised.value.reason
    assert session.calls == []


@pytest.mark.asyncio
async def test_allows_cross_authority_robots_redirect_without_source_scope() -> (
    None
):
    """RFC 9309: robots redirects may cross authorities.

    The page redirect rules reject cross-authority hops when no source scope
    is available; robots fetching must not depend on source scopes.
    """

    cdn_url = "https://cdn.example.test/robots.txt"
    session = _FakeSession(
        [
            _FakeResponse(
                status=302,
                headers={"Location": cdn_url},
                url="https://example.test/robots.txt",
                connected_peer=("93.184.216.34", 443),
            ),
            _FakeResponse(
                status=200,
                url=cdn_url,
                connected_peer=("93.184.216.34", 443),
                body=b"User-agent: *\nDisallow:\n",
            ),
        ]
    )
    fetcher = _fetcher(session=session)

    result = await _fetch(
        fetcher,
        robots_url="https://example.test/robots.txt",
    )

    assert session.calls == [
        "https://example.test/robots.txt",
        cdn_url,
    ]
    assert result.final_url == cdn_url
    assert result.is_success


@pytest.mark.asyncio
async def test_rejects_unsafe_redirect_target_via_shared_rules() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status=302,
                headers={"Location": "https://127.0.0.1/robots.txt"},
                url="https://example.test/robots.txt",
                connected_peer=("93.184.216.34", 443),
            ),
        ]
    )
    fetcher = _fetcher(session=session)

    with pytest.raises(RobotsRedirectRejectedError) as raised:
        await _fetch(
            fetcher,
            robots_url="https://example.test/robots.txt",
        )

    assert raised.value.reason == "redirect_loopback_ip_blocked"
    assert session.calls == ["https://example.test/robots.txt"]


@pytest.mark.asyncio
async def test_robots_fetch_waits_for_rate_limit_slot() -> None:
    """The robots transport waits for the host pacing slot before fetching."""
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url="https://example.test/robots.txt",
                connected_peer=("93.184.216.34", 443),
                body=b"User-agent: *\nDisallow:\n",
            )
        ]
    )
    limiter = _FakeLimiter()
    fetcher = _fetcher(session=session, limiter=limiter)

    result = await _fetch(
        fetcher,
        robots_url="https://example.test/robots.txt",
    )

    assert limiter.acquired == ["example.test"]
    assert session.calls == ["https://example.test/robots.txt"]
    assert result.is_success


@pytest.mark.asyncio
async def test_response_rate_limit_hints_are_applied_before_error_return() -> (
    None
):
    limiter = _FakeLimiter()
    session = _FakeSession(
        [
            _FakeResponse(
                status=429,
                headers={"Retry-After": "12"},
                url="https://example.test/robots.txt",
                connected_peer=("93.184.216.34", 443),
            )
        ]
    )
    fetcher = _fetcher(session=session, limiter=limiter)

    with pytest.raises(RobotsHttpStatusError) as raised:
        await _fetch(
            fetcher,
            robots_url="https://example.test/robots.txt",
        )

    assert raised.value.retry_after_seconds == 12.0
    assert limiter.response_hints == [
        ("example.test", 12.0, None, None),
    ]


@pytest.mark.asyncio
async def test_blocks_connected_peer_after_dns_resolution() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url="https://example.test/robots.txt",
                connected_peer=("127.0.0.1", 443),
                body=b"User-agent: *\nDisallow:\n",
            ),
        ]
    )
    fetcher = _fetcher(session=session)

    with pytest.raises(RobotsRedirectRejectedError) as raised:
        await _fetch(
            fetcher,
            robots_url="https://example.test/robots.txt",
        )

    assert "blocked_connected_peer" in raised.value.reason


@pytest.mark.asyncio
async def test_accepts_public_connected_peer() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url="https://example.test/robots.txt",
                connected_peer=("93.184.216.34", 443),
                body=b"User-agent: *\nDisallow: /private\n",
            ),
        ]
    )
    fetcher = _fetcher(session=session)

    result = await _fetch(
        fetcher,
        robots_url="https://example.test/robots.txt",
    )

    assert result.is_success
    assert result.body == b"User-agent: *\nDisallow: /private\n"


@pytest.mark.asyncio
async def test_rejects_response_when_connected_peer_is_unavailable() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url="https://example.test/robots.txt",
                connected_peer=None,
                body=b"User-agent: *\nDisallow:\n",
            ),
        ]
    )

    fetcher = _fetcher(session=session)

    with pytest.raises(RobotsNetworkError) as raised:
        await _fetch(
            fetcher,
            robots_url="https://example.test/robots.txt",
        )

    assert raised.value.error_type == "peer_address_unavailable"


@pytest.mark.asyncio
async def test_rejects_invalid_connected_peer_metadata() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status=200,
                url="https://example.test/robots.txt",
                connected_peer=(),
                body=b"User-agent: *\nDisallow:\n",
            ),
        ]
    )

    fetcher = _fetcher(session=session)

    with pytest.raises(RobotsNetworkError) as raised:
        await _fetch(
            fetcher,
            robots_url="https://example.test/robots.txt",
        )

    assert raised.value.error_type == "peer_address_unavailable"
