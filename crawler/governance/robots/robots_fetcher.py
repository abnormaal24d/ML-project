"""Robots.txt fetch schema without exposing concrete HTTP backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RobotsFetchResult:
    """Transport-level result of fetching robots.txt."""

    requested_url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    latency_seconds: float
    retry_after_seconds: float | None = None

    @property
    def is_success(self) -> bool:
        """Return whether the fetch completed with a 2xx status."""

        return self.status_code >= 200 and self.status_code < 300


class RobotsFetcher(Protocol):
    """Consumer-owned schema implemented by robots transport adapters."""

    async def fetch(
        self,
        *,
        robots_url: str,
        headers: dict[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> RobotsFetchResult: ...
