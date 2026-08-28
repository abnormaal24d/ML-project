"""Synchronous HTTP transport with per-connection DNS pinning."""

from __future__ import annotations

import http.client
import ipaddress
import math
import socket
import ssl
import urllib.request
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import SplitResult, urljoin, urlsplit

from crawler.fetching.errors.exceptions import IgnoredFetchError

if TYPE_CHECKING:
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

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

_CROSS_ORIGIN_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "range",
    }
)

_ENTITY_HEADERS = frozenset(
    {
        "content-length",
        "content-type",
        "transfer-encoding",
    }
)


class SafeUrlOpener:
    """Open requests through DNS-pinned, peer-verified connections."""

    def __init__(
        self,
        *,
        network_access_guard: NetworkAddressGuard,
        redirector: RedirectRulesValidator | None = None,
        follow_redirects: bool = True,
        max_redirects: int = 10,
        source_name: str | None = None,
    ) -> None:
        self._network_access_guard = network_access_guard
        self._redirector = redirector
        self._follow_redirects = bool(follow_redirects)
        self._max_redirects = _validate_max_redirects(max_redirects)
        self._source_name = source_name

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        source_name: str | None = None,
    ) -> http.client.HTTPResponse:
        """Open a request without modifying process-global resolver state."""

        current_url = request.full_url
        method = request.get_method().upper()
        headers = dict(request.header_items())
        body = request.data
        redirect_count = 0

        effective_source_name = (
            source_name if source_name is not None else self._source_name
        )
        validated_timeout = _validate_timeout(timeout)

        while True:
            response = self._open_once(
                url=current_url,
                method=method,
                headers=headers,
                body=body,
                timeout=validated_timeout,
            )

            status_code = int(response.status)
            location = response.headers.get("Location")

            if (
                not self._follow_redirects
                or status_code not in _REDIRECT_STATUSES
                or not location
            ):
                return response

            try:
                redirect_url = self._redirect_url(
                    current_url=current_url,
                    location=location,
                )
                redirect_count += 1

                if self._redirector is not None:
                    try:
                        self._redirector.validate_hop(
                            current_url=current_url,
                            target_url=redirect_url,
                            redirect_count=redirect_count,
                            source_name=effective_source_name,
                        )
                    except IgnoredFetchError as exc:
                        raise IgnoredFetchError(
                            reason=exc.reason,
                            observed_bytes=exc.observed_bytes,
                            metrics_recorded=exc.metrics_recorded,
                            status_code=status_code,
                            final_url=redirect_url,
                        ) from exc

                if redirect_count > self._max_redirects:
                    raise RuntimeError("redirect_limit_exceeded")

                next_method, next_body = _redirect_method_and_body(
                    status_code=status_code,
                    method=method,
                    body=body,
                )

                headers = _headers_for_redirect(
                    headers=headers,
                    previous_url=current_url,
                    target_url=redirect_url,
                    drops_entity_body=(
                        next_method != method and next_body is None
                    ),
                )
            finally:
                response.close()

            current_url = redirect_url
            method = next_method
            body = next_body

    def validate_url(self, *, url: str) -> None:
        """Resolve and validate all addresses without opening a connection."""

        self._resolve_pinned_address(url=url)

    def _redirect_url(
        self,
        *,
        current_url: str,
        location: str,
    ) -> str:
        """Resolve a relative or absolute redirect Location header."""

        return urljoin(current_url, location)

    def _open_once(
        self,
        *,
        url: str,
        method: str,
        headers: dict[str, str],
        body: Any,
        timeout: float,
    ) -> http.client.HTTPResponse:
        parsed = _parse_http_url(url)

        host = parsed.hostname
        if host is None:
            raise RuntimeError("invalid_network_url")

        port = _effective_port(parsed)
        pinned_address = self._resolve_pinned_address(url=url)

        if parsed.scheme.lower() == "https":
            connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
                host=host,
                port=port,
                pinned_address=pinned_address,
                timeout=timeout,
                network_access_guard=self._network_access_guard,
            )
        else:
            connection = _PinnedHTTPConnection(
                host=host,
                port=port,
                pinned_address=pinned_address,
                timeout=timeout,
                network_access_guard=self._network_access_guard,
            )

        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"

        try:
            connection.request(
                method,
                target,
                body=body,
                headers=headers,
            )
            return connection.getresponse()
        except BaseException:
            connection.close()
            raise

    def _resolve_pinned_address(self, *, url: str) -> str:
        parsed = _parse_http_url(url)

        reason = self._network_access_guard.rejection_reason_for_url(url)
        if reason is not None:
            raise RuntimeError(f"blocked_network_target:{reason}")

        host = parsed.hostname
        if host is None:
            raise RuntimeError("invalid_network_url")

        try:
            answers = socket.getaddrinfo(
                host,
                _effective_port(parsed),
                type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("dns_resolution_failed") from exc

        addresses = tuple(
            dict.fromkeys(str(answer[4][0]) for answer in answers)
        )

        if not addresses:
            raise RuntimeError("dns_resolution_empty")

        # Reject the complete DNS answer set when any address is unsafe.
        # This prevents selecting an allowed address from a mixed answer set
        # that also contains private, loopback or metadata addresses.
        for address in addresses:
            reason = self._network_access_guard.rejection_reason_for_address(
                address
            )
            if reason is not None:
                raise RuntimeError(f"blocked_network_target:{reason}")

        return addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects only to a prevalidated IP address."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        pinned_address: str,
        timeout: float,
        network_access_guard: NetworkAddressGuard,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
        )
        self._pinned_address = pinned_address
        self._network_access_guard = network_access_guard

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
        )

        _verify_connected_peer(
            sock=self.sock,
            expected_address=self._pinned_address,
            network_access_guard=self._network_access_guard,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection using pinned TCP routing and hostname TLS checks."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        pinned_address: str,
        timeout: float,
        network_access_guard: NetworkAddressGuard,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
        )
        self._pinned_address = pinned_address
        self._network_access_guard = network_access_guard
        self._context: ssl.SSLContext = cast(
            ssl.SSLContext,
            object.__getattribute__(self, "_context"),
        )

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
        )

        try:
            _verify_connected_peer(
                sock=raw_socket,
                expected_address=self._pinned_address,
                network_access_guard=self._network_access_guard,
            )

            tls_socket = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
            self.sock = tls_socket

            _verify_connected_peer(
                sock=tls_socket,
                expected_address=self._pinned_address,
                network_access_guard=self._network_access_guard,
            )
        except BaseException:
            raw_socket.close()
            raise


def _parse_http_url(url: str) -> SplitResult:
    """Parse and validate one supported network URL."""

    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname

        # Accessing port performs urllib's port validation.
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError("invalid_network_url") from exc

    if scheme not in {"http", "https"} or not hostname:
        raise RuntimeError("unsupported_network_url")

    return parsed


def _effective_port(parsed: SplitResult) -> int:
    """Return the explicit or scheme-default network port."""

    if parsed.port is not None:
        return parsed.port

    return 443 if parsed.scheme.lower() == "https" else 80


def _headers_for_redirect(
    *,
    headers: dict[str, str],
    previous_url: str,
    target_url: str,
    drops_entity_body: bool,
) -> dict[str, str]:
    """Remove target-bound and cross-origin-sensitive redirect headers."""

    drop = set(_REDIRECT_RECOMPUTED_HEADERS)

    if _origin(previous_url) != _origin(target_url):
        drop.update(_CROSS_ORIGIN_SENSITIVE_HEADERS)

    if drops_entity_body:
        drop.update(_ENTITY_HEADERS)

    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in drop
    }


def _redirect_method_and_body(
    *,
    status_code: int,
    method: str,
    body: Any,
) -> tuple[str, Any]:
    """Apply standard redirect method and request-body semantics."""

    if status_code == 303 and method != "HEAD":
        return "GET", None

    if status_code in {301, 302} and method == "POST":
        return "GET", None

    return method, body


def _origin(url: str) -> tuple[str, str, int]:
    """Return a normalized scheme, hostname and effective port tuple."""

    parsed = _parse_http_url(url)
    hostname = parsed.hostname

    if hostname is None:
        raise RuntimeError("invalid_network_url")

    return (
        parsed.scheme.lower(),
        hostname.rstrip(".").casefold(),
        _effective_port(parsed),
    )


def _verify_connected_peer(
    *,
    sock: socket.socket,
    expected_address: str,
    network_access_guard: NetworkAddressGuard,
) -> None:
    """Verify the connected peer against the pinned and allowed address."""

    peer_address = str(sock.getpeername()[0])

    try:
        peer_ip = ipaddress.ip_address(peer_address)
        expected_ip = ipaddress.ip_address(expected_address)
    except ValueError as exc:
        sock.close()
        raise RuntimeError("connected_peer_address_invalid") from exc

    if peer_ip != expected_ip:
        sock.close()
        raise RuntimeError("connected_peer_does_not_match_dns_pin")

    reason = network_access_guard.rejection_reason_for_address(peer_address)
    if reason is not None:
        sock.close()
        raise RuntimeError(f"blocked_connected_peer:{reason}")


def _validate_max_redirects(value: int) -> int:
    """Validate the configured redirect limit."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_redirects must be an integer")

    if not 0 <= value <= 10:
        raise ValueError("max_redirects must be between 0 and 10")

    return value


def _validate_timeout(value: float) -> float:
    """Validate and normalize a socket timeout."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError("timeout must be a number")

    timeout = float(value)

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")

    return timeout
