"""Regression tests for robots parser cache stale-serving and error caching.

A known-good robots parser must be served stale when a fresh reload fails
within the stale window (RFC 9309 section 3.3.1 semantics), while local
pacing deferrals must never pollute the long error cache.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.robotparser import RobotFileParser

import pytest

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_fetch_errors import (
    RobotsFetchDeferredError,
    RobotsHttpStatusError,
)
from crawler.governance.robots.robots_fetcher import RobotsFetchResult
from crawler.governance.robots.robots_parser_cache import RobotsParserCache
from crawler.governance.robots.robots_parser_loader import (
    RobotsParserLoadResult,
)
from tests.support.logging import TEST_LOGGER

ROBOTS_URL = "https://example.test/robots.txt"


def _parser(lines: list[str]) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(lines)
    return parser


class _ScriptedLoader:
    def __init__(self) -> None:
        self._script: list[Any] = []
        self.calls: list[str] = []

    def add_result(
        self,
        *,
        parser: RobotFileParser | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._script.append({"parser": parser, "error": error})

    async def load(
        self,
        robots_url: str,
        timeout: float,
    ) -> RobotsParserLoadResult:
        del timeout
        self.calls.append(robots_url)
        configured = self._script[
            min(len(self.calls) - 1, len(self._script) - 1)
        ]
        if configured["error"] is not None:
            raise configured["error"]
        return RobotsParserLoadResult(
            parser=configured["parser"],
            fetch_result=RobotsFetchResult(
                requested_url=robots_url,
                final_url=robots_url,
                status_code=200,
                headers={},
                body=b"",
                latency_seconds=0.01,
            ),
        )


def _cache(
    loader: _ScriptedLoader,
    *,
    cache_ttl_s: float = 3600.0,
    error_cache_ttl_s: float = 60.0,
    stale_ttl_s: float | None = None,
) -> RobotsParserCache:
    return RobotsParserCache(
        cache_ttl_s=cache_ttl_s,
        error_cache_ttl_s=error_cache_ttl_s,
        stale_ttl_s=stale_ttl_s,
        parser_loader=loader,  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        logger=TEST_LOGGER,
    )


@pytest.mark.asyncio
async def test_stale_parser_served_when_reload_fails() -> None:
    loader = _ScriptedLoader()
    known_good = _parser(["User-agent: *", "Disallow: /private", ""])
    loader.add_result(parser=known_good)
    loader.add_result(
        error=RobotsHttpStatusError(
            status_code=503,
            error_type="HTTPError",
            headers={},
            final_url=ROBOTS_URL,
            requested_url=ROBOTS_URL,
        )
    )
    cache = _cache(loader, cache_ttl_s=0.05, stale_ttl_s=3600.0)

    loaded = await cache.get(ROBOTS_URL, timeout=1.0)
    assert loaded is known_good

    await asyncio.sleep(0.06)

    # Reload fails (503): the last-known-good parser must still be served.
    stale = await cache.get(ROBOTS_URL, timeout=1.0)
    assert stale is known_good
    assert stale.can_fetch("*", "https://example.test/private") is False
    assert loader.calls.count(ROBOTS_URL) == 2


@pytest.mark.asyncio
async def test_error_cached_without_stale_parser_keeps_raising() -> None:
    loader = _ScriptedLoader()
    loader.add_result(
        error=RobotsHttpStatusError(
            status_code=503,
            error_type="HTTPError",
            headers={},
            final_url=ROBOTS_URL,
            requested_url=ROBOTS_URL,
        )
    )
    cache = _cache(loader, cache_ttl_s=0.05, stale_ttl_s=3600.0)

    with pytest.raises(RobotsHttpStatusError):
        await cache.get(ROBOTS_URL, timeout=1.0)

    # The transient loader error is cached: no second fetch.
    with pytest.raises(RobotsHttpStatusError):
        await cache.get(ROBOTS_URL, timeout=1.0)

    assert loader.calls.count(ROBOTS_URL) == 1


@pytest.mark.asyncio
async def test_local_deferred_error_is_never_error_cached() -> None:
    loader = _ScriptedLoader()
    loader.add_result(
        error=RobotsFetchDeferredError(
            reason="robots_fetch_rate_limited_locally",
            retry_after_seconds=2.5,
        )
    )
    cache = _cache(loader, error_cache_ttl_s=60.0, stale_ttl_s=None)

    with pytest.raises(RobotsFetchDeferredError):
        await cache.get(ROBOTS_URL, timeout=1.0)
    with pytest.raises(RobotsFetchDeferredError):
        await cache.get(ROBOTS_URL, timeout=1.0)

    assert loader.calls.count(ROBOTS_URL) == 2


@pytest.mark.asyncio
async def test_stale_parser_not_served_beyond_stale_window() -> None:
    loader = _ScriptedLoader()
    known_good = _parser(["User-agent: *", "Disallow: /private", ""])
    loader.add_result(parser=known_good)
    loader.add_result(
        error=RobotsHttpStatusError(
            status_code=503,
            error_type="HTTPError",
            headers={},
            final_url=ROBOTS_URL,
            requested_url=ROBOTS_URL,
        )
    )
    cache = _cache(loader, cache_ttl_s=0.02, stale_ttl_s=0.03)

    loaded = await cache.get(ROBOTS_URL, timeout=1.0)
    assert loaded is known_good

    await asyncio.sleep(0.08)

    with pytest.raises(RobotsHttpStatusError):
        await cache.get(ROBOTS_URL, timeout=1.0)
