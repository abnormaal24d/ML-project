"""Shared HTTP transport for GET and HEAD with governed redirects.

Owns request execution, per-hop rate limiting, redirect rules validation,
redirect header sanitization, and response release for both methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal
from urllib.parse import urljoin, urlsplit

import aiohttp

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.fetching.network.session import connected_peer_address

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.rate_limit.rate_limiter import RateLimiter
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )
    from crawler.governance.robots.robots_request_gate import (
        RobotsRequestGate,
    )

HeaderEnricher = Callable[..., dict[str, str]]

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class AiohttpRequestRunner:
    """Execute GET/HEAD requests and authorize every redirect hop."""

    # Target-bound / revalidated headers: always drop after a redirect so the
    # next hop can recompute them (enrichers, cookie jar, Host, conditionals).
    _REDIRECT_RECOMPUTED_HEADERS = frozenset(
        {
            "host",
            "cookie",
            "if-match",
            "if-none-match",
            "if-modified-since",
            "if-unmodified-since",
            "if-range",
        }
    )

    # Credential-like headers: keep on same origin, strip on origin change.
    _CROSS_ORIGIN_SENSITIVE_HEADERS = frozenset(
        {
            "authorization",
            "proxy-authorization",
            "range",
        }
    )

    def __init__(
        self,
        *,
        redirector: RedirectRulesValidator,
        host_normalizer: HostNormalizer,
        rate_limiter: RateLimiter,
        robots_gate: RobotsRequestGate,
        network_address_guard: NetworkAddressGuard,
    ) -> None:
        self._redirector = redirector
        self._host_normalizer = host_normalizer
        self._rate_limiter = rate_limiter
        self._robots_gate = robots_gate
        self._network_address_guard = network_address_guard

    @asynccontextmanager
    async def perform(
        self,
        *,
        session: aiohttp.ClientSession,
        method: Literal["GET", "HEAD"],
        url: str,
        source_name: str | None,
        base_headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        defer_if_rate_limited: bool,
        enrich_headers: HeaderEnricher | None = None,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Perform one request flow with manually governed redirects.

        Persistent headers flow:

        base/persistent headers
        → optional per-URL enrichment
        → request
        → on redirect: sanitize persistent headers
        → enrichment for the next URL
        """

        response: aiohttp.ClientResponse | None = None
        current_url = url
        redirect_count = 0
        hop_headers = dict(base_headers)

        try:
            while True:
                if enrich_headers is not None:
                    headers = enrich_headers(
                        url=current_url,
                        base_headers=hop_headers,
                    )
                else:
                    headers = dict(hop_headers)

                request_host = self._host_normalizer.require(
                    urlsplit(current_url).hostname,
                )
                await self._robots_gate.authorize(
                    url=current_url,
                )
                await self._rate_limiter.acquire_for_fetch(
                    host=request_host,
                    defer_if_rate_limited=defer_if_rate_limited,
                )

                response = await session.request(
                    method,
                    current_url,
                    headers=headers,
                    allow_redirects=False,
                    timeout=timeout,
                )

                self._validate_connected_peer(response=response)

                next_url = self._redirect_target(
                    response=response,
                    current_url=current_url,
                )
                if next_url is None:
                    yield response
                    return

                redirect_count += 1
                try:
                    self._redirector.validate_hop(
                        current_url=current_url,
                        target_url=next_url,
                        redirect_count=redirect_count,
                        source_name=source_name,
                    )
                    # Sanitize persistent hop headers, not the enriched copy,
                    # so target-bound validators are recomputed next hop.
                    hop_headers = self._headers_for_redirect(
                        headers=hop_headers,
                        previous_url=current_url,
                        target_url=next_url,
                    )
                finally:
                    response.release()
                    # Prevent the outer finally from releasing twice when a
                    # hop validation error propagates out of this block.
                    response = None

                current_url = next_url
        finally:
            if response is not None:
                response.release()

    def _redirect_target(
        self,
        *,
        response: aiohttp.ClientResponse,
        current_url: str,
    ) -> str | None:
        if int(response.status) not in _REDIRECT_STATUSES:
            return None

        location = response.headers.get("Location")
        if not location:
            return None

        location_size = len(location.encode("utf-8", errors="surrogatepass"))
        if location_size > self._redirector.max_location_length:
            raise IgnoredFetchError(
                reason="redirect_location_too_large",
                observed_bytes=0,
            )

        return urljoin(current_url, location)

    def _headers_for_redirect(
        self,
        *,
        headers: dict[str, str],
        previous_url: str,
        target_url: str,
    ) -> dict[str, str]:
        same_origin = self._origin(previous_url) == self._origin(target_url)
        drop = set(self._REDIRECT_RECOMPUTED_HEADERS)
        if not same_origin:
            drop |= self._CROSS_ORIGIN_SENSITIVE_HEADERS

        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in drop
        }

    def _origin(self, url: str) -> tuple[str, str, int | None]:
        parsed = urlsplit(url)
        scheme = (parsed.scheme or "").lower()
        host = self._host_normalizer.require(parsed.hostname)

        port = parsed.port
        if port is None:
            if scheme == "http":
                port = 80
            elif scheme == "https":
                port = 443

        return scheme, host, port

    def _validate_connected_peer(
        self,
        *,
        response: aiohttp.ClientResponse,
    ) -> None:
        """Validate the actual peer selected after DNS resolution."""

        peer_address = connected_peer_address(response)
        if peer_address is None:
            response.close()
            raise RetryableFetchError(
                "connected peer address is unavailable",
                retry_class="local_transport_metadata",
                retry_error_kind="peer_address_unavailable",
            )

        reason = self._network_address_guard.rejection_reason_for_address(
            peer_address,
        )
        if reason is not None:
            response.close()
            raise IgnoredFetchError(
                reason=f"blocked_connected_peer:{reason}",
                observed_bytes=0,
            )
