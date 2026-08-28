"""Regression tests for scheme-aware robots parser cache identity."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.robotparser import RobotFileParser

import pytest

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_fetcher import RobotsFetchResult
from crawler.governance.robots.robots_parser_cache import RobotsParserCache
from crawler.governance.robots.robots_parser_loader import (
    RobotsParserLoadResult,
)
from tests.support.logging import TEST_LOGGER

HTTP_ROBOTS = "http://example.test/robots.txt"
HTTPS_ROBOTS = "https://example.test/robots.txt"
HTTPS_8443_ROBOTS = "https://example.test:8443/robots.txt"


def _parser(lines: list[str]) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(lines)
    return parser


def _fetch_result(
    *,
    requested_url: str,
    final_url: str | None = None,
    status_code: int = 200,
) -> RobotsFetchResult:
    return RobotsFetchResult(
        requested_url=requested_url,
        final_url=final_url or requested_url,
        status_code=status_code,
        headers={},
        body=b"",
        latency_seconds=0.01,
    )


class _ScriptedLoader:
    """Test double that returns configured results per robots URL."""

    def __init__(self) -> None:
        self._results: dict[str, Any] = {}
        self._gates: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []

    def set_result(
        self,
        robots_url: str,
        *,
        parser: RobotFileParser | None = None,
        fetch_result: RobotsFetchResult | None = None,
        error: BaseException | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self._results[robots_url] = {
            "parser": parser,
            "fetch_result": fetch_result,
            "error": error,
        }
        if gate is not None:
            self._gates[robots_url] = gate

    async def load(
        self,
        robots_url: str,
        timeout: float,
    ) -> RobotsParserLoadResult:
        del timeout
        self.calls.append(robots_url)
        gate = self._gates.get(robots_url)
        if gate is not None:
            await gate.wait()

        configured = self._results[robots_url]
        if configured["error"] is not None:
            raise configured["error"]
        return RobotsParserLoadResult(
            parser=configured["parser"],
            fetch_result=configured["fetch_result"],
        )


def _cache(loader: _ScriptedLoader) -> RobotsParserCache:
    return RobotsParserCache(
        cache_ttl_s=3600,
        error_cache_ttl_s=60,
        parser_loader=loader,  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        logger=TEST_LOGGER,
    )


def test_http_and_https_have_different_cache_keys() -> None:
    cache = _cache(_ScriptedLoader())
    assert cache._cache_key(HTTP_ROBOTS) != cache._cache_key(HTTPS_ROBOTS)


def test_default_ports_are_normalized() -> None:
    cache = _cache(_ScriptedLoader())
    assert cache._cache_key("http://example.test:80/robots.txt") == (
        cache._cache_key("http://example.test/robots.txt")
    )
    assert cache._cache_key("https://example.test:443/robots.txt") == (
        cache._cache_key("https://example.test/robots.txt")
    )
    assert cache._cache_key("HTTP://EXAMPLE.TEST./robots.txt") == (
        cache._cache_key("http://example.test/robots.txt")
    )


def test_non_default_ports_remain_distinct() -> None:
    cache = _cache(_ScriptedLoader())
    assert cache._cache_key(HTTPS_ROBOTS) != cache._cache_key(
        HTTPS_8443_ROBOTS
    )


def test_path_case_is_preserved() -> None:
    cache = _cache(_ScriptedLoader())
    assert cache._cache_key("https://example.test/robots.txt") != (
        cache._cache_key("https://example.test/Robots.txt")
    )


@pytest.mark.asyncio
async def test_http_and_https_are_loaded_separately() -> None:
    loader = _ScriptedLoader()
    http_parser = _parser(["User-agent: *", "Disallow: /private", ""])
    https_parser = _parser(["User-agent: *", "Allow: /", ""])
    loader.set_result(
        HTTP_ROBOTS,
        parser=http_parser,
        fetch_result=_fetch_result(requested_url=HTTP_ROBOTS),
    )
    loader.set_result(
        HTTPS_ROBOTS,
        parser=https_parser,
        fetch_result=_fetch_result(requested_url=HTTPS_ROBOTS),
    )
    cache = _cache(loader)

    loaded_http = await cache.get(HTTP_ROBOTS, timeout=1.0)
    loaded_https = await cache.get(HTTPS_ROBOTS, timeout=1.0)

    assert loader.calls.count(HTTP_ROBOTS) == 1
    assert loader.calls.count(HTTPS_ROBOTS) == 1
    assert loaded_http.can_fetch("*", "http://example.test/private") is False
    assert loaded_https.can_fetch("*", "https://example.test/private") is True

    # Second lookups must hit cache without extra loads.
    await cache.get(HTTP_ROBOTS, timeout=1.0)
    await cache.get(HTTPS_ROBOTS, timeout=1.0)
    assert loader.calls.count(HTTP_ROBOTS) == 1
    assert loader.calls.count(HTTPS_ROBOTS) == 1


@pytest.mark.asyncio
async def test_success_does_not_mask_error_of_other_scheme() -> None:
    loader = _ScriptedLoader()
    loader.set_result(
        HTTP_ROBOTS,
        parser=_parser(["User-agent: *", "Allow: /", ""]),
        fetch_result=_fetch_result(requested_url=HTTP_ROBOTS),
    )
    loader.set_result(
        HTTPS_ROBOTS,
        error=OSError("https robots unreachable"),
    )
    cache = _cache(loader)

    await cache.get(HTTP_ROBOTS, timeout=1.0)

    with pytest.raises(OSError, match="https robots unreachable"):
        await cache.get(HTTPS_ROBOTS, timeout=1.0)

    # HTTPS error is cached independently; HTTP success remains available.
    with pytest.raises(OSError, match="https robots unreachable"):
        await cache.get(HTTPS_ROBOTS, timeout=1.0)
    http_parser = await cache.get(HTTP_ROBOTS, timeout=1.0)
    assert http_parser.can_fetch("*", "http://example.test/") is True
    assert loader.calls.count(HTTP_ROBOTS) == 1
    assert loader.calls.count(HTTPS_ROBOTS) == 1


@pytest.mark.asyncio
async def test_concurrent_loads_do_not_overwrite_each_other() -> None:
    loader = _ScriptedLoader()
    http_gate = asyncio.Event()
    https_gate = asyncio.Event()
    http_parser = _parser(["User-agent: *", "Disallow: /private", ""])
    https_parser = _parser(["User-agent: *", "Allow: /", ""])
    loader.set_result(
        HTTP_ROBOTS,
        parser=http_parser,
        fetch_result=_fetch_result(requested_url=HTTP_ROBOTS),
        gate=http_gate,
    )
    loader.set_result(
        HTTPS_ROBOTS,
        parser=https_parser,
        fetch_result=_fetch_result(requested_url=HTTPS_ROBOTS),
        gate=https_gate,
    )
    cache = _cache(loader)

    http_task = asyncio.create_task(cache.get(HTTP_ROBOTS, timeout=2.0))
    https_task = asyncio.create_task(cache.get(HTTPS_ROBOTS, timeout=2.0))

    # Let both enter loader, then finish HTTPS first (would have been
    # last-writer-wins on the old shared success key).
    await asyncio.sleep(0)
    https_gate.set()
    loaded_https = await https_task
    http_gate.set()
    loaded_http = await http_task

    assert loaded_http is http_parser
    assert loaded_https is https_parser
    assert cache.last_fetch_result(HTTP_ROBOTS) is not None
    assert cache.last_fetch_result(HTTP_ROBOTS).requested_url == HTTP_ROBOTS
    assert cache.last_fetch_result(HTTPS_ROBOTS) is not None
    assert cache.last_fetch_result(HTTPS_ROBOTS).requested_url == HTTPS_ROBOTS
    assert loaded_http.can_fetch("*", "http://example.test/private") is False
    assert loaded_https.can_fetch("*", "https://example.test/private") is True


@pytest.mark.asyncio
async def test_close_cancels_owned_load_after_waiter_cancellation() -> None:
    loader = _ScriptedLoader()
    gate = asyncio.Event()
    loader.set_result(
        HTTPS_ROBOTS,
        parser=_parser(["User-agent: *", "Allow: /", ""]),
        fetch_result=_fetch_result(requested_url=HTTPS_ROBOTS),
        gate=gate,
    )
    cache = _cache(loader)
    cancelled_waiter = asyncio.create_task(
        cache.get(HTTPS_ROBOTS, timeout=1.0)
    )
    joined_waiter = asyncio.create_task(cache.get(HTTPS_ROBOTS, timeout=1.0))

    for _ in range(10):
        if loader.calls:
            break
        await asyncio.sleep(0)
    assert loader.calls == [HTTPS_ROBOTS]

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    assert not joined_waiter.done()

    await cache.aclose()

    with pytest.raises(asyncio.CancelledError):
        await joined_waiter
    assert cache._inflight == {}

    await cache.aclose()
    with pytest.raises(RuntimeError, match="cache is closed"):
        await cache.get(HTTPS_ROBOTS, timeout=1.0)


@pytest.mark.asyncio
async def test_fetch_metadata_stays_bound_to_parser() -> None:
    loader = _ScriptedLoader()
    loader.set_result(
        HTTP_ROBOTS,
        parser=_parser(["User-agent: *", "Disallow: /", ""]),
        fetch_result=_fetch_result(
            requested_url=HTTP_ROBOTS,
            final_url=HTTP_ROBOTS,
            status_code=200,
        ),
    )
    loader.set_result(
        HTTPS_ROBOTS,
        parser=_parser(["User-agent: *", "Allow: /", ""]),
        fetch_result=_fetch_result(
            requested_url=HTTPS_ROBOTS,
            final_url=HTTPS_ROBOTS,
            status_code=200,
        ),
    )
    cache = _cache(loader)

    await cache.get(HTTP_ROBOTS, timeout=1.0)
    await cache.get(HTTPS_ROBOTS, timeout=1.0)

    http_meta = cache.last_fetch_result(HTTP_ROBOTS)
    https_meta = cache.last_fetch_result(HTTPS_ROBOTS)
    assert http_meta is not None
    assert https_meta is not None
    assert http_meta.requested_url == HTTP_ROBOTS
    assert https_meta.requested_url == HTTPS_ROBOTS


@pytest.mark.asyncio
async def test_redirects_key_on_initial_authority_not_final_url() -> None:
    """Two initial robots URLs redirecting to one final URL stay distinct."""

    shared_final = "https://cdn.example.test/shared.txt"
    loader = _ScriptedLoader()
    http_parser = _parser(["User-agent: *", "Disallow: /a", ""])
    https_parser = _parser(["User-agent: *", "Disallow: /b", ""])
    loader.set_result(
        HTTP_ROBOTS,
        parser=http_parser,
        fetch_result=_fetch_result(
            requested_url=HTTP_ROBOTS,
            final_url=shared_final,
        ),
    )
    loader.set_result(
        HTTPS_ROBOTS,
        parser=https_parser,
        fetch_result=_fetch_result(
            requested_url=HTTPS_ROBOTS,
            final_url=shared_final,
        ),
    )
    cache = _cache(loader)

    loaded_http = await cache.get(HTTP_ROBOTS, timeout=1.0)
    loaded_https = await cache.get(HTTPS_ROBOTS, timeout=1.0)

    assert loader.calls.count(HTTP_ROBOTS) == 1
    assert loader.calls.count(HTTPS_ROBOTS) == 1
    assert loaded_http is http_parser
    assert loaded_https is https_parser
    assert cache.last_fetch_result(HTTP_ROBOTS).final_url == shared_final
    assert cache.last_fetch_result(HTTPS_ROBOTS).final_url == shared_final
    assert cache._cache_key(HTTP_ROBOTS) != cache._cache_key(HTTPS_ROBOTS)
