"""Immutable response metadata passed above the HTTP transport layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from aiohttp import ClientResponse

_ALLOWED_RESPONSE_HEADERS = frozenset(
    {
        "accept-ranges",
        "cache-control",
        "content-encoding",
        "content-language",
        "content-length",
        "content-range",
        "content-type",
        "etag",
        "expires",
        "last-modified",
        "location",
        "retry-after",
        "x-robots-tag",
    }
)


def safe_response_headers(
    headers: Mapping[str, object],
) -> dict[str, str]:
    """Return response metadata approved for snapshots and persistence."""

    return {
        str(name): str(value)[:4096]
        for name, value in headers.items()
        if str(name).strip().lower() in _ALLOWED_RESPONSE_HEADERS
    }


@dataclass(frozen=True, slots=True)
class FetchResponseSnapshot:
    """Transport-neutral response metadata used by validators."""

    status: int
    url: str
    headers: Mapping[str, str]
    content_length: int | None = None
    redirect_chain: tuple[str, ...] = ()

    @classmethod
    def from_response(cls, response: ClientResponse) -> FetchResponseSnapshot:
        """Create a snapshot from an aiohttp-compatible response."""

        history = getattr(response, "history", ()) or ()
        return cls(
            status=int(getattr(response, "status", 0)),
            url=str(getattr(response, "url", "")),
            headers=safe_response_headers(getattr(response, "headers", {})),
            content_length=getattr(response, "content_length", None),
            redirect_chain=tuple(
                str(getattr(item, "url", "")) for item in history
            ),
        )
