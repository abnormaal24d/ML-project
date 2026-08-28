"""Behaviour tests for shared AiohttpRequestRunner GET/HEAD transport."""

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
from crawler.fetching.network.request import (
    AiohttpRequestRunner,
)
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)


class _FakeRedirector:
    max_location_length = 64

    def __init__(self, *, reject: bool = False) -> None:
        self.hops: list[dict[str, Any]] = []
        self._reject = reject

    def validate_hop(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
        source_name: str | None,
    ) -> None:
        self.hops.append(
            {
                "current_url": current_url,
                "target_url": target_url,
                "redirect_count": redirect_count,
                "source_name": source_name,
            }
        )
        if self._reject:
            raise IgnoredFetchError(
                reason="redirect_disallowed",
                observed_bytes=0,
            )


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, bool]] = []

    async def acquire_for_fetch(
        self,
        *,
        host: str | None,
        defer_if_rate_limited: bool,
    ) -> None:
        self.acquired.append((str(host), bool(defer_if_rate_limited)))


_UNSET = object()


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        url: Any = None,
        connected_peer: object = _UNSET,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.connected_peer = (
            ("93.184.216.34", 443)
            if connected_peer is _UNSET
            else connected_peer
        )
        self.release_calls = 0
        self.close_calls = 0

    def release(self) -> None:
        self.release_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.headers_by_call: list[dict[str, str]] = []

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> _FakeResponse:
        self.calls.append((method, url))
        self.headers_by_call.append(dict(kwargs.get("headers") or {}))
        if not self._responses:
            raise AssertionError(
                f"unexpected session.request for {method} {url}"
            )
        return self._responses.pop(0)


class _FakeRobotsGate:
    def __init__(self) -> None:
        self.authorized: list[str] = []

    async def authorize(
        self,
        *,
        url: str,
    ) -> None:
        self.authorized.append(url)


def _runner(
    *,
    rate_limiter: _FakeRateLimiter | None = None,
    redirector: _FakeRedirector | None = None,
    network_guard: NetworkAddressGuard | None = None,
) -> tuple[AiohttpRequestRunner, _FakeRateLimiter, _FakeRedirector]:
    limiter = rate_limiter or _FakeRateLimiter()
    hops = redirector or _FakeRedirector()
    runner = AiohttpRequestRunner(
        redirector=hops,  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rate_limiter=limiter,  # type: ignore[arg-type]
        robots_gate=_FakeRobotsGate(),  # type: ignore[arg-type]
        network_address_guard=network_guard or _FakeNetworkAddressGuard(),
    )
    return runner, limiter, hops


class _FakeNetworkAddressGuard:
    def __init__(
        self,
        *,
        blocked_address: str | None = None,
    ) -> None:
        self.settings = NetworkAccessSettings()
        self._blocked_address = blocked_address

    def rejection_reason_for_address(self, address: str) -> str | None:
        if address == self._blocked_address:
            return "loopback"
        return None


async def _perform(
    *,
    runner: AiohttpRequestRunner,
    session: _FakeSession,
    method: str = "GET",
    url: str,
    base_headers: dict[str, str] | None = None,
    defer_if_rate_limited: bool = False,
    enrich_headers: Any | None = None,
) -> _FakeResponse | None:
    yielded: _FakeResponse | None = None
    async with runner.perform(
        session=session,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        url=url,
        source_name="test-source",
        base_headers=base_headers or {"User-Agent": "test-agent"},
        timeout=aiohttp.ClientTimeout(total=5.0),
        defer_if_rate_limited=defer_if_rate_limited,
        enrich_headers=enrich_headers,
    ) as response:
        yielded = response
    return yielded


def _etag_enricher(
    *, url: str, base_headers: dict[str, str]
) -> dict[str, str]:
    """Mimic ConditionalRepresentationCache header enrichment behaviour."""

    headers = dict(base_headers)
    if url.endswith("/start"):
        headers.setdefault("If-None-Match", '"start"')
    elif url.endswith("/final"):
        headers.setdefault("If-None-Match", '"final"')
    return headers


@pytest.mark.asyncio
async def test_normal_get_acquires_canonical_host_before_network() -> None:
    response = _FakeResponse(
        status=200,
        url=SimpleNamespace(host="example.test"),
    )
    session = _FakeSession([response])
    runner, limiter, _redirector = _runner()

    yielded = await _perform(
        runner=runner,
        session=session,
        url="https://example.test/resource",
    )

    assert session.calls == [("GET", "https://example.test/resource")]
    assert limiter.acquired == [("example.test", False)]
    assert yielded is response
    assert response.release_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("connected_peer", (None, ()))
async def test_missing_connected_peer_is_retryable_local_metadata_failure(
    connected_peer: object,
) -> None:
    response = _FakeResponse(status=200, connected_peer=connected_peer)
    session = _FakeSession([response])
    runner, _limiter, _redirector = _runner()

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/resource",
        )

    assert raised.value.retry_class == "local_transport_metadata"
    assert raised.value.retry_error_kind == "peer_address_unavailable"
    assert session.calls == [("GET", "https://example.test/resource")]
    assert response.close_calls == 1
    assert response.release_calls == 1


@pytest.mark.asyncio
async def test_get_known_blocked_peer_remains_non_retryable() -> None:
    response = _FakeResponse(
        status=200,
        connected_peer=("127.0.0.1", 443),
    )
    session = _FakeSession([response])
    runner, _limiter, _redirector = _runner(
        network_guard=_FakeNetworkAddressGuard(
            blocked_address="127.0.0.1",
        )
    )

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/resource",
        )

    assert raised.value.reason == "blocked_connected_peer:loopback"
    assert session.calls == [("GET", "https://example.test/resource")]
    assert response.close_calls == 1
    assert response.release_calls == 1


@pytest.mark.asyncio
async def test_get_redirect_hop_with_missing_peer_is_retryable() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
        connected_peer=("93.184.216.34", 443),
    )
    unavailable_response = _FakeResponse(
        status=200,
        connected_peer=None,
    )
    session = _FakeSession([redirect_response, unavailable_response])
    runner, _limiter, _redirector = _runner()

    with pytest.raises(RetryableFetchError) as raised:
        await _perform(runner=runner, session=session, url=start)

    assert raised.value.retry_error_kind == "peer_address_unavailable"
    assert session.calls == [("GET", start), ("GET", final)]
    assert redirect_response.release_calls == 1
    assert unavailable_response.close_calls == 1
    assert unavailable_response.release_calls == 1


@pytest.mark.asyncio
async def test_redirect_uses_host_budget_of_each_hop() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
        url=SimpleNamespace(host="example.test"),
    )
    final_response = _FakeResponse(
        status=200,
        url=SimpleNamespace(host="cdn.example.test"),
    )
    session = _FakeSession([redirect_response, final_response])
    runner, limiter, redirector = _runner()

    yielded = await _perform(runner=runner, session=session, url=start)

    assert limiter.acquired == [
        ("example.test", False),
        ("cdn.example.test", False),
    ]
    assert session.calls == [
        ("GET", start),
        ("GET", final),
    ]
    assert redirector.hops == [
        {
            "current_url": start,
            "target_url": final,
            "redirect_count": 1,
            "source_name": "test-source",
        }
    ]
    assert yielded is final_response
    assert redirect_response.release_calls == 1
    assert final_response.release_calls == 1


@pytest.mark.asyncio
async def test_perform_canonicalizes_host_via_normalizer() -> None:
    response = _FakeResponse(
        status=200, url=SimpleNamespace(host="example.test")
    )
    session = _FakeSession([response])
    runner, limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url="https://EXAMPLE.test./resource",
    )

    assert limiter.acquired == [("example.test", False)]
    assert session.calls == [("GET", "https://EXAMPLE.test./resource")]


@pytest.mark.asyncio
async def test_invalid_request_host_fails_before_network() -> None:
    session = _FakeSession([])
    runner, limiter, _ = _runner()

    with pytest.raises(ValueError, match="valid host"):
        await _perform(
            runner=runner,
            session=session,
            url="https:///missing-host",
        )

    assert session.calls == []
    assert limiter.acquired == []


@pytest.mark.asyncio
async def test_head_method_uses_same_redirect_and_rate_limit_path() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
        url=SimpleNamespace(host="example.test"),
    )
    final_response = _FakeResponse(
        status=200,
        url=SimpleNamespace(host="cdn.example.test"),
    )
    session = _FakeSession([redirect_response, final_response])
    runner, limiter, redirector = _runner()

    yielded = await _perform(
        runner=runner,
        session=session,
        method="HEAD",
        url=start,
        defer_if_rate_limited=True,
    )

    assert session.calls == [
        ("HEAD", start),
        ("HEAD", final),
    ]
    assert limiter.acquired == [
        ("example.test", True),
        ("cdn.example.test", True),
    ]
    assert len(redirector.hops) == 1
    assert yielded is final_response
    assert redirect_response.release_calls == 1
    assert final_response.release_calls == 1


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_sensitive_headers() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        method="HEAD",
        url=start,
        base_headers={
            "Authorization": "Bearer secret",
            "Cookie": "sid=1",
            "User-Agent": "test-agent",
            "Accept": "image/*",
        },
        defer_if_rate_limited=True,
    )

    assert session.headers_by_call[0]["Authorization"] == "Bearer secret"
    assert session.headers_by_call[0]["Cookie"] == "sid=1"
    assert "Authorization" not in session.headers_by_call[1]
    assert "Cookie" not in session.headers_by_call[1]
    assert session.headers_by_call[1]["User-Agent"] == "test-agent"
    assert session.headers_by_call[1]["Accept"] == "image/*"


@pytest.mark.asyncio
async def test_same_origin_redirect_keeps_authorization_drops_cookie() -> None:
    start = "https://example.test/start"
    final = "https://example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={
            "Authorization": "Bearer secret",
            "Cookie": "sid=1",
            "User-Agent": "test-agent",
        },
    )

    # Authorization is origin-scoped; explicit Cookie is recomputed per hop
    # so the cookie jar can attach the correct cookies for the next URL.
    assert session.headers_by_call[1]["Authorization"] == "Bearer secret"
    assert "Cookie" not in session.headers_by_call[1]


@pytest.mark.asyncio
async def test_default_port_is_same_origin_for_sensitive_headers() -> None:
    start = "https://example.test/start"
    final = "https://example.test:443/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={
            "Authorization": "Bearer secret",
            "Proxy-Authorization": "Basic abc",
            "Range": "bytes=0-1",
        },
    )

    assert session.headers_by_call[1]["Authorization"] == "Bearer secret"
    assert session.headers_by_call[1]["Proxy-Authorization"] == "Basic abc"
    assert session.headers_by_call[1]["Range"] == "bytes=0-1"


@pytest.mark.asyncio
async def test_non_default_port_is_cross_origin_for_sensitive_headers() -> (
    None
):
    start = "https://example.test/start"
    final = "https://example.test:8443/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={
            "Authorization": "Bearer secret",
            "Cookie": "sid=1",
            "User-Agent": "test-agent",
        },
    )

    assert "Authorization" not in session.headers_by_call[1]
    assert "Cookie" not in session.headers_by_call[1]
    assert session.headers_by_call[1]["User-Agent"] == "test-agent"


@pytest.mark.asyncio
async def test_scheme_change_is_cross_origin_for_sensitive_headers() -> None:
    start = "https://example.test/start"
    final = "http://example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={
            "Authorization": "Bearer secret",
            "Cookie": "sid=1",
        },
    )

    assert "Authorization" not in session.headers_by_call[1]
    assert "Cookie" not in session.headers_by_call[1]


@pytest.mark.asyncio
async def test_same_origin_redirect_recomputes_target_bound_validators() -> (
    None
):
    start = "https://example.test/start"
    final = "https://example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={"User-Agent": "test-agent"},
        enrich_headers=_etag_enricher,
    )

    assert session.headers_by_call[0]["If-None-Match"] == '"start"'
    assert session.headers_by_call[1]["If-None-Match"] == '"final"'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_name,header_value",
    [
        ("Proxy-Authorization", "Basic abc"),
        ("Range", "bytes=0-99"),
        ("If-Match", '"v1"'),
        ("If-Modified-Since", "Mon, 01 Jan 2024 00:00:00 GMT"),
        ("If-Unmodified-Since", "Mon, 01 Jan 2024 00:00:00 GMT"),
        ("If-Range", '"v1"'),
        ("Cookie", "sid=1"),
    ],
)
async def test_redirect_header_categories(
    header_name: str,
    header_value: str,
) -> None:
    """Cross-origin drops credentials; all redirects drop recomputed headers."""

    start = "https://example.test/start"
    same_origin = "https://example.test/final"
    cross_origin = "https://cdn.example.test/final"

    # Same-origin hop
    same_session = _FakeSession(
        [
            _FakeResponse(status=302, headers={"Location": same_origin}),
            _FakeResponse(status=200),
        ]
    )
    same_runner, _, _ = _runner()
    await _perform(
        runner=same_runner,
        session=same_session,
        url=start,
        base_headers={header_name: header_value, "User-Agent": "ua"},
    )

    recomputed = {
        "cookie",
        "if-match",
        "if-none-match",
        "if-modified-since",
        "if-unmodified-since",
        "if-range",
        "host",
    }
    if header_name.lower() in recomputed:
        assert header_name not in same_session.headers_by_call[1]
    else:
        assert same_session.headers_by_call[1][header_name] == header_value

    # Cross-origin hop drops credentials and recomputed headers.
    cross_session = _FakeSession(
        [
            _FakeResponse(status=302, headers={"Location": cross_origin}),
            _FakeResponse(status=200),
        ]
    )
    cross_runner, _, _ = _runner()
    await _perform(
        runner=cross_runner,
        session=cross_session,
        url=start,
        base_headers={header_name: header_value, "User-Agent": "ua"},
    )
    # Every parametrized header is either recomputed every hop or
    # cross-origin sensitive, so none survive a host change.
    assert header_name not in cross_session.headers_by_call[1]


@pytest.mark.asyncio
async def test_redirect_does_not_reuse_host_header() -> None:
    start = "https://example.test/start"
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    final_response = _FakeResponse(status=200)
    session = _FakeSession([redirect_response, final_response])
    runner, _limiter, _ = _runner()

    await _perform(
        runner=runner,
        session=session,
        url=start,
        base_headers={
            "Host": "example.test",
            "User-Agent": "test-agent",
        },
    )

    assert session.headers_by_call[0]["Host"] == "example.test"
    assert "Host" not in session.headers_by_call[1]


@pytest.mark.asyncio
async def test_rejected_redirect_releases_response_exactly_once() -> None:
    final = "https://cdn.example.test/final"
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": final},
    )
    session = _FakeSession([redirect_response])
    runner, _limiter, _ = _runner(redirector=_FakeRedirector(reject=True))

    with pytest.raises(IgnoredFetchError, match="redirect_disallowed"):
        await _perform(
            runner=runner,
            session=session,
            url="https://example.test/start",
        )

    assert redirect_response.release_calls == 1


@pytest.mark.asyncio
async def test_redirect_location_too_large_raises_same_error() -> None:
    oversized = "https://cdn.example.test/" + ("x" * 200)
    redirect_response = _FakeResponse(
        status=302,
        headers={"Location": oversized},
    )
    session = _FakeSession([redirect_response])
    runner, _limiter, redirector = _runner()

    with pytest.raises(IgnoredFetchError) as raised:
        await _perform(
            runner=runner,
            session=session,
            method="HEAD",
            url="https://example.test/start",
            defer_if_rate_limited=True,
        )

    assert raised.value.reason == "redirect_location_too_large"
    assert session.calls == [("HEAD", "https://example.test/start")]
    assert redirector.hops == []
    assert redirect_response.release_calls == 1
