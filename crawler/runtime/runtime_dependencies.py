"""Crawler contract for access to the composed HTTP client session."""

from __future__ import annotations

from typing import Protocol

import aiohttp

__all__ = ["HttpClientSessionProvider"]


class HttpClientSessionProvider(Protocol):
    """Provides a shared (or per-call) aiohttp ClientSession.

    Domain code must never construct ClientSession directly.
    """

    async def get_session(self) -> aiohttp.ClientSession: ...

    async def aclose(self) -> None: ...
