"""Cache and coalesce robots.txt parser loads."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from crawler.governance.robots.robots_fetch_errors import (
    RobotsFetchDeferredError,
    RobotsLoaderError,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.robots.robots_fetcher import RobotsFetchResult
    from crawler.governance.robots.robots_parser_loader import (
        RobotsParserLoader,
    )


class RobotsParserCache:
    """Cache robots.txt parsers for a limited time.

    All cache layers (success, error, singleflight, fetch metadata) share one
    canonical key: ``scheme + canonical authority + path`` derived from the
    originally requested robots URL (RFC 9309 initial authority).

    A known-good parser may be served stale (RFC 9309 section 3.3.1) while a
    fresh reload keeps failing, up to ``stale_ttl_s`` past the normal TTL.
    """

    def __init__(
        self,
        *,
        cache_ttl_s: int,
        error_cache_ttl_s: int,
        parser_loader: RobotsParserLoader,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
        stale_ttl_s: float | None = None,
    ) -> None:
        self._cache_ttl_s = cache_ttl_s
        self._error_cache_ttl_s = error_cache_ttl_s
        self._stale_ttl_s = (
            max(0.0, float(stale_ttl_s)) if stale_ttl_s is not None else 0.0
        )
        self._parser_loader = parser_loader
        self._host_normalizer = host_normalizer
        self._logger = logger
        self._cache: dict[
            str, tuple[float, float, RobotFileParser, RobotsFetchResult | None]
        ] = {}
        self._error_cache: dict[str, tuple[float, Exception]] = {}
        self._inflight: dict[str, asyncio.Task[RobotFileParser]] = {}
        self._inflight_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public cache API
    # ------------------------------------------------------------------
    async def get(
        self,
        robots_url: str,
        timeout: float,
    ) -> RobotFileParser:
        """Return a cached, stale, or freshly loaded robots parser."""

        if self._closed:
            raise RuntimeError("robots parser cache is closed")
        parser, error = self._cached_resolution(robots_url)
        if parser is not None:
            return parser
        if error is not None:
            raise error

        self._logger.debug(
            "robots_cache_miss",
            extra={
                "robots_url": robots_url,
                "timeout_seconds": timeout,
            },
        )
        return await self._get_or_load_singleflight(
            robots_url=robots_url,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Cancel and await every cache-owned parser load."""

        async with self._inflight_lock:
            self._closed = True
            tasks = tuple(self._inflight.values())
            self._cache.clear()
            self._error_cache.clear()

        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        async with self._inflight_lock:
            for load_key, task in tuple(self._inflight.items()):
                if task.done():
                    self._inflight.pop(load_key, None)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------
    def last_fetch_result(
        self,
        robots_url: str,
    ) -> RobotsFetchResult | None:
        """Return cached fetch metadata for the robots URL, if available."""

        cached = self._cache.get(self._cache_key(robots_url))
        if cached is None:
            return None
        return cached[3]

    async def _get_or_load_singleflight(
        self,
        *,
        robots_url: str,
        timeout: float,
    ) -> RobotFileParser:
        load_key = self._cache_key(robots_url)

        async with self._inflight_lock:
            if self._closed:
                raise RuntimeError("robots parser cache is closed")
            parser, error = self._cached_resolution(robots_url)
            if parser is not None:
                return parser
            if error is not None:
                raise error

            task = self._inflight.get(load_key)
            if task is None:
                task = asyncio.create_task(
                    self._load_and_cache(
                        robots_url=robots_url,
                        timeout=timeout,
                    )
                )
                self._inflight[load_key] = task

                def _on_task_done(
                    completed_task: asyncio.Task[RobotFileParser],
                    *,
                    key: str = load_key,
                ) -> None:
                    self._finalize_inflight_task(
                        load_key=key,
                        task=completed_task,
                    )

                task.add_done_callback(_on_task_done)
                self._logger.debug(
                    "robots_singleflight_load_started",
                    extra={
                        "robots_url": robots_url,
                    },
                )
            else:
                self._logger.debug(
                    "robots_singleflight_joined",
                    extra={
                        "robots_url": robots_url,
                    },
                )

        return await asyncio.shield(task)

    async def _load_and_cache(
        self,
        *,
        robots_url: str,
        timeout: float,
    ) -> RobotFileParser:
        # One key for success, error, singleflight, and fetch metadata.
        cache_key = self._cache_key(robots_url)

        try:
            loaded = await self._parser_loader.load(robots_url, timeout)
        except RobotsFetchDeferredError:
            # Local pacing deferrals are volatile; never long-cache them.
            raise
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            self._cache_error(cache_key=cache_key, exc=exc)
            raise
        except RobotsLoaderError as exc:
            self._cache_error(cache_key=cache_key, exc=exc)
            stale = self._stale_parser(robots_url)
            if stale is not None:
                self._log_stale_served(robots_url)
                return stale
            raise

        parser = loaded.parser
        fetch_result = loaded.fetch_result
        now = monotonic()
        self._cache[cache_key] = (
            now + self._cache_ttl_s,
            now + self._cache_ttl_s + self._stale_ttl_s,
            parser,
            fetch_result,
        )
        self._logger.debug(
            "robots_cache_stored",
            extra={
                "robots_url": robots_url,
                "cache_ttl_seconds": self._cache_ttl_s,
                "stale_ttl_seconds": self._stale_ttl_s,
                "has_fetch_metadata": fetch_result is not None,
            },
        )
        return parser

    def _cache_error(
        self,
        *,
        cache_key: str,
        exc: Exception,
    ) -> None:
        now = monotonic()
        self._error_cache[cache_key] = (
            now + self._error_cache_ttl_s,
            exc,
        )
        self._logger.debug(
            "robots_error_cache_stored",
            extra={
                "robots_url": cache_key,
                "cache_ttl_seconds": self._error_cache_ttl_s,
                "error_type": type(exc).__name__,
            },
        )

    def _finalize_inflight_task(
        self,
        *,
        load_key: str,
        task: asyncio.Task[RobotFileParser],
    ) -> None:
        """Detach completed singleflight tasks without leaking exceptions."""

        try:
            error = task.exception()
        except asyncio.CancelledError:
            error = None

        if error is not None and not isinstance(
            error,
            (
                OSError,
                RuntimeError,
                TimeoutError,
                ValueError,
                RobotsLoaderError,
            ),
        ):
            self._logger.error(
                "robots_singleflight_unexpected_failure",
                extra={
                    "load_key": load_key,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        if self._inflight.get(load_key) is task:
            self._inflight.pop(load_key, None)

    def _cached_parser(self, robots_url: str) -> RobotFileParser | None:
        now = monotonic()
        cache_key = self._cache_key(robots_url)
        cached = self._cache.get(cache_key)
        if cached is None:
            return None
        expires_at, stale_until, parser, _fetch_result = cached
        if expires_at <= now and stale_until <= now:
            self._cache.pop(cache_key, None)
            return None
        if expires_at <= now:
            return None
        self._logger.debug(
            "robots_cache_hit",
            extra={
                "robots_url": robots_url,
                "ttl_remaining_seconds": round(expires_at - now, 3),
                "has_fetch_metadata": _fetch_result is not None,
            },
        )
        return parser

    def _stale_parser(self, robots_url: str) -> RobotFileParser | None:
        """Return the expired-but-valid last-known-good parser, if any."""

        cached = self._cache.get(self._cache_key(robots_url))
        if cached is None:
            return None
        expires_at, stale_until, parser, _fetch_result = cached
        if expires_at <= monotonic() < stale_until:
            return parser
        return None

    def _cached_resolution(
        self,
        robots_url: str,
    ) -> tuple[RobotFileParser | None, Exception | None]:
        """Return (fresh-or-stale parser, cached error) for one robots URL.

        A stale parser wins over a cached loader error while it remains
        within the stale window.
        """

        fresh = self._cached_parser(robots_url)
        if fresh is not None:
            return fresh, None

        cached_error = self._cached_error(robots_url)
        if cached_error is not None:
            stale = self._stale_parser(robots_url)
            if stale is not None:
                self._log_stale_served(robots_url)
                return stale, None
            return None, cached_error
        return None, None

    def _log_stale_served(self, robots_url: str) -> None:
        self._logger.debug(
            "robots_stale_parser_served",
            extra={
                "robots_url": robots_url,
                "stale_ttl_seconds": self._stale_ttl_s,
            },
        )

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------
    def _cached_error(self, robots_url: str) -> Exception | None:
        now = monotonic()
        cache_key = self._cache_key(robots_url)
        cached_error = self._error_cache.get(cache_key)
        if cached_error is None:
            return None
        if cached_error[0] <= now:
            self._error_cache.pop(cache_key, None)
            return None
        self._logger.debug(
            "robots_error_cache_hit",
            extra={
                "robots_url": robots_url,
                "ttl_remaining_seconds": round(cached_error[0] - now, 3),
            },
        )
        return cached_error[1]

    def _cache_key(self, robots_url: str) -> str:
        """Return the canonical cache identity for a robots resource.

        Keyed on the originally requested robots URL:
        ``scheme + canonical authority + path`` (RFC 9309 / RFC 3986).
        HTTP and HTTPS are distinct resources even when host and path match.
        """

        parsed = urlsplit(robots_url)
        scheme = (parsed.scheme or "").lower()
        port = self._canonical_port(scheme=scheme, port=parsed.port)
        netloc = self._canonical_netloc(parsed.hostname, port)
        return urlunsplit(
            (
                scheme,
                netloc,
                parsed.path or "/",
                "",
                "",
            )
        )

    @staticmethod
    def _canonical_port(
        *,
        scheme: str,
        port: int | None,
    ) -> int | None:
        """Drop explicit default ports so equivalent URLs share a key."""

        if (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        ):
            return None
        return port

    def _canonical_netloc(self, host: str | None, port: int | None) -> str:
        canonical_host = self._host_normalizer.require(host)
        rendered_host = (
            f"[{canonical_host}]" if ":" in canonical_host else canonical_host
        )
        return rendered_host if port is None else f"{rendered_host}:{port}"
