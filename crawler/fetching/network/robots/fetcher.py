"""Aiohttp-backed robots.txt transport adapter."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import TYPE_CHECKING, Final, override
from urllib.parse import urljoin, urlsplit

import aiohttp

from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.network.session import connected_peer_address
from crawler.fetching.response.rate_limit_hints import ResponseRateLimitHints
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.robots.robots_fetch_errors import (
    RobotsHttpStatusError,
    RobotsNetworkError,
    RobotsRedirectRejectedError,
    RobotsTimeoutError,
)
from crawler.governance.robots.robots_fetcher import (
    RobotsFetcher,
    RobotsFetchResult,
)
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.rate_limit.rate_limiter import RateLimiter
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )
    from crawler.runtime.runtime_dependencies import (
        HttpClientSessionProvider,
    )
    from logger.project_logger import ProjectLogger


_REDIRECT_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {
        301,
        302,
        303,
        307,
        308,
    }
)


class AiohttpRobotsFetcher(RobotsFetcher):
    """Fetch robots.txt through shared crawler transport rules."""

    def __init__(
        self,
        *,
        session_provider: HttpClientSessionProvider,
        rate_limiter: RateLimiter,
        redirector: RedirectRulesValidator,
        host_normalizer: HostNormalizer,
        network_address_guard: NetworkAddressGuard,
        clock: Clock,
        logger: ProjectLogger,
    ) -> None:
        self._session_provider = session_provider
        self._rate_limiter = rate_limiter
        self._redirector = redirector
        self._host_normalizer = host_normalizer
        self._network_address_guard = network_address_guard
        self._clock = clock
        self._logger = logger

    @override
    async def fetch(
        self,
        *,
        robots_url: str,
        headers: dict[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> RobotsFetchResult:
        """Fetch one robots document and return its transport result."""

        self._validate_limits(
            timeout_seconds=timeout_seconds,
            max_body_bytes=max_body_bytes,
        )

        session = await self._session_provider.get_session()

        fetch_started_at = monotonic()
        current_url = robots_url
        redirect_count = 0

        while True:
            request_host = self._host_from_url(current_url)

            initial_guard_reason = (
                self._network_address_guard.rejection_reason_for_url(
                    current_url,
                )
            )
            if initial_guard_reason is not None:
                raise RobotsRedirectRejectedError(
                    reason=f"blocked_initial_url:{initial_guard_reason}",
                    status_code=None,
                    final_url=current_url,
                )

            await self._rate_limiter.acquire(request_host)

            request_started_at = monotonic()
            status_code: int | None = None

            try:
                async with session.get(
                    current_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=timeout_seconds,
                    ),
                    allow_redirects=False,
                ) as response:
                    status_code = int(response.status)

                    self._validate_connected_peer(
                        response=response,
                        status_code=status_code,
                    )

                    redirect_target = self._redirect_target(
                        response=response,
                        current_url=current_url,
                    )

                    if redirect_target is not None:
                        redirect_count += 1

                        self._validate_redirect(
                            current_url=current_url,
                            target_url=redirect_target,
                            redirect_count=redirect_count,
                            status_code=status_code,
                        )

                        current_url = redirect_target
                        continue

                    body = await self._read_bounded_body(
                        response=response,
                        max_body_bytes=max_body_bytes,
                        robots_url=robots_url,
                        status_code=status_code,
                    )

                    response_headers = {
                        str(key): str(value)
                        for key, value in response.headers.items()
                    }

                    latency_seconds = monotonic() - fetch_started_at

                    rate_limit_hints = ResponseRateLimitHints.from_headers(
                        response.headers,
                        now=self._clock.now(),
                    )
                    applied_delay_seconds = await self._rate_limiter.apply_response_rate_limit_hints(
                        host=request_host,
                        retry_after_seconds=(
                            rate_limit_hints.retry_after_seconds
                        ),
                        rate_limit_remaining=(
                            rate_limit_hints.rate_limit_remaining
                        ),
                        rate_limit_reset_seconds=(
                            rate_limit_hints.rate_limit_reset_seconds
                        ),
                    )

                    final_url = str(response.url)

                    if not 200 <= status_code < 300:
                        raise RobotsHttpStatusError(
                            status_code=status_code,
                            error_type="HTTPError",
                            headers=response_headers,
                            final_url=final_url,
                            requested_url=robots_url,
                            body=body,
                            latency_seconds=latency_seconds,
                            retry_after_seconds=applied_delay_seconds,
                        )

                    return RobotsFetchResult(
                        requested_url=robots_url,
                        final_url=final_url,
                        status_code=status_code,
                        headers=response_headers,
                        body=body,
                        latency_seconds=latency_seconds,
                        retry_after_seconds=applied_delay_seconds,
                    )

            except aiohttp.ClientResponseError as exc:
                status_code = int(exc.status)

                response_headers = (
                    {
                        str(key): str(value)
                        for key, value in exc.headers.items()
                    }
                    if exc.headers is not None
                    else {}
                )

                rate_limit_hints = ResponseRateLimitHints.from_headers(
                    response_headers,
                    now=self._clock.now(),
                )
                applied_delay_seconds = await self._rate_limiter.apply_response_rate_limit_hints(
                    host=request_host,
                    retry_after_seconds=rate_limit_hints.retry_after_seconds,
                    rate_limit_remaining=rate_limit_hints.rate_limit_remaining,
                    rate_limit_reset_seconds=(
                        rate_limit_hints.rate_limit_reset_seconds
                    ),
                )

                raise RobotsHttpStatusError(
                    status_code=status_code,
                    error_type=type(exc).__name__,
                    headers=response_headers,
                    final_url=str(exc.request_info.real_url),
                    requested_url=robots_url,
                    latency_seconds=(monotonic() - fetch_started_at),
                    retry_after_seconds=applied_delay_seconds,
                ) from exc

            except (
                asyncio.TimeoutError,
                TimeoutError,
            ) as exc:
                raise RobotsTimeoutError(
                    error_type=type(exc).__name__,
                ) from exc

            except aiohttp.ClientError as exc:
                raise RobotsNetworkError(
                    error_type=type(exc).__name__,
                ) from exc

            finally:
                if status_code is not None:
                    await self._rate_limiter.report_result(
                        host=request_host,
                        status_code=status_code,
                        latency_seconds=(monotonic() - request_started_at),
                    )

    def _validate_redirect(
        self,
        *,
        current_url: str,
        target_url: str,
        redirect_count: int,
        status_code: int,
    ) -> None:
        """Validate one redirect and map rules rejection."""

        try:
            self._redirector.validate_robots_hop(
                current_url=current_url,
                target_url=target_url,
                redirect_count=redirect_count,
            )

        except IgnoredFetchError as exc:
            raise RobotsRedirectRejectedError(
                reason=exc.reason,
                status_code=status_code,
                final_url=target_url,
            ) from exc

    def _validate_connected_peer(
        self,
        *,
        response: aiohttp.ClientResponse,
        status_code: int,
    ) -> None:
        """Reject responses whose connected peer violates address rules."""

        peer_address = connected_peer_address(response)
        if peer_address is None:
            response.close()
            raise RobotsNetworkError(
                error_type="peer_address_unavailable",
            )

        peer_reason = self._network_address_guard.rejection_reason_for_address(
            peer_address,
        )
        if peer_reason is not None:
            raise RobotsRedirectRejectedError(
                reason=f"blocked_connected_peer:{peer_reason}",
                status_code=status_code,
                final_url=str(response.url),
            )

    def _redirect_target(
        self,
        *,
        response: aiohttp.ClientResponse,
        current_url: str,
    ) -> str | None:
        """Resolve a valid redirect target."""

        if int(response.status) not in _REDIRECT_STATUS_CODES:
            return None

        location = response.headers.get("Location")

        if not location:
            return None

        location_size = len(
            location.encode(
                "utf-8",
                errors="surrogatepass",
            )
        )

        if location_size > self._redirector.max_location_length:
            raise RobotsRedirectRejectedError(
                reason="redirect_location_too_large",
                status_code=int(response.status),
                final_url=current_url,
            )

        return urljoin(
            current_url,
            location,
        )

    async def _read_bounded_body(
        self,
        *,
        response: aiohttp.ClientResponse,
        max_body_bytes: int,
        robots_url: str,
        status_code: int,
    ) -> bytes:
        """Read the robots body up to its configured limit."""

        try:
            body = await response.content.readexactly(max_body_bytes + 1)

        except asyncio.IncompleteReadError as exc:
            body = exc.partial

        if len(body) <= max_body_bytes:
            return body

        self._logger.warning(
            "robots_body_truncated",
            robots_url=robots_url,
            status_code=status_code,
            max_bytes=max_body_bytes,
        )

        return body[:max_body_bytes]

    def _host_from_url(
        self,
        url: str,
    ) -> str | None:
        """Return the normalized host from a URL."""

        try:
            hostname = urlsplit(url).hostname
        except ValueError:
            return None

        return self._host_normalizer.normalize(hostname)

    @staticmethod
    def _validate_limits(
        *,
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> None:
        """Reject invalid request limits."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
