"""Shared aiohttp session lifecycle and guarded transport adapters."""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult

from config.collection.http_rules import (
    ConnectionPoolSettings,
    TimeoutRulesSettings,
)
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.runtime.runtime_dependencies import HttpClientSessionProvider

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger


def _peer_from_transport(transport: object | None) -> object | None:
    """Return the raw peer metadata exposed by an aiohttp transport."""

    get_extra_info = getattr(transport, "get_extra_info", None)
    if not callable(get_extra_info):
        return None

    return cast(object | None, get_extra_info("peername"))


def connected_peer_address(response: aiohttp.ClientResponse) -> str | None:
    """Return the captured connected-peer address, if available."""

    peer = getattr(response, "connected_peer", None)
    if not isinstance(peer, tuple) or not peer:
        return None
    return str(peer[0])


class _PeerAwareClientResponse(aiohttp.ClientResponse):
    """Preserve the connected peer before aiohttp releases its connection."""

    connected_peer: object | None = None

    async def start(self, connection: Any) -> aiohttp.ClientResponse:
        self.connected_peer = _peer_from_transport(
            getattr(connection, "transport", None)
        )
        response = await super().start(connection)
        if self.connected_peer is None:
            protocol = getattr(connection, "protocol", None) or getattr(
                self, "_protocol", None
            )
            self.connected_peer = _peer_from_transport(
                getattr(protocol, "transport", None)
            )
        return response


class _GuardedResolver(AbstractResolver):
    """Filter aiohttp DNS records through the network access guard."""

    def __init__(
        self,
        *,
        resolver: AbstractResolver,
        network_access_guard: NetworkAddressGuard,
    ) -> None:
        self._resolver = resolver
        self._network_access_guard = network_access_guard

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        results = await self._resolver.resolve(host, port=port, family=family)
        allowed = self._network_access_guard.filter_resolved_addresses(
            host=host,
            addresses=list(results),
        )
        if not allowed:
            raise OSError(
                f"DNS resolution for {host!r} returned only blocked addresses"
            )
        return allowed

    async def close(self) -> None:
        await self._resolver.close()


class AiohttpClientSessionProvider(HttpClientSessionProvider):
    """Create and manage the shared guarded aiohttp client session."""

    def __init__(
        self,
        *,
        timeout_rules: TimeoutRulesSettings,
        connection_pool: ConnectionPoolSettings,
        logger: ProjectLogger,
        network_access_guard: NetworkAddressGuard | None = None,
    ) -> None:
        self._timeout_rules = timeout_rules
        self._connection_pool = connection_pool
        self._logger = logger
        self._network_access_guard = network_access_guard
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Return the lazily created shared aiohttp client session."""

        existing_session = self._session
        if existing_session is not None and not existing_session.closed:
            return existing_session

        async with self._session_lock:
            existing_session = self._session
            if existing_session is not None and not existing_session.closed:
                return existing_session

            timeout = aiohttp.ClientTimeout(
                total=self._timeout_rules.request_timeout_seconds,
                connect=self._timeout_rules.connect_timeout_seconds,
                sock_connect=self._timeout_rules.connect_timeout_seconds,
                sock_read=self._timeout_rules.request_timeout_seconds,
            )
            connector = aiohttp.TCPConnector(
                limit=self._connection_pool.max_connections,
                limit_per_host=self._connection_pool.max_connections_per_host,
                enable_cleanup_closed=True,
                ttl_dns_cache=self._connection_pool.ttl_dns_cache_seconds,
                resolver=self._build_resolver(),
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                cookie_jar=aiohttp.DummyCookieJar(),
                auto_decompress=self._connection_pool.auto_decompress,
                response_class=_PeerAwareClientResponse,
            )
            self._logger.debug("http_session_created")
            return self._session

    async def aclose(self) -> None:
        """Close the session if it is open."""

        async with self._session_lock:
            session = self._session
            self._session = None

            if session is not None and not session.closed:
                await session.close()
                self._logger.debug("http_session_closed")

    def _build_resolver(self) -> AbstractResolver:
        base_resolver = self._build_base_resolver()
        guard = self._network_access_guard
        if guard is None or not guard.dns_resolution_enabled:
            return base_resolver

        return _GuardedResolver(
            resolver=base_resolver,
            network_access_guard=guard,
        )

    def _build_base_resolver(self) -> AbstractResolver:
        if not self._connection_pool.async_dns_enabled:
            return aiohttp.DefaultResolver()

        try:
            return aiohttp.AsyncResolver()
        except (ImportError, RuntimeError, OSError) as exc:
            self._logger.info(
                "async_dns_resolver_unavailable",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return aiohttp.DefaultResolver()
