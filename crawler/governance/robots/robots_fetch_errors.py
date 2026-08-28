"""Project-owned robots loader exceptions.

Transport adapters map concrete HTTP library errors to these types.
Governance code must only catch and classify these abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass


class RobotsLoaderError(Exception):
    """Base class for expected robots loading failures."""


@dataclass
class RobotsHttpStatusError(RobotsLoaderError):
    """HTTP response with a non-success status for robots.txt."""

    status_code: int
    error_type: str
    headers: dict[str, str]
    final_url: str
    requested_url: str
    body: bytes = b""
    latency_seconds: float = 0.0
    retry_after_seconds: float | None = None


@dataclass
class RobotsNetworkError(RobotsLoaderError):
    """Transport-level network failure while fetching robots.txt."""

    error_type: str


@dataclass
class RobotsTimeoutError(RobotsLoaderError):
    """Timeout while fetching robots.txt."""

    error_type: str = "timeout"


@dataclass
class RobotsRedirectRejectedError(RobotsLoaderError):
    """Redirect chain rejected by crawl redirect rules."""

    reason: str
    status_code: int | None = None
    final_url: str | None = None


@dataclass
class RobotsFetchDeferredError(RobotsLoaderError):
    """Local pacing required deferring the robots fetch itself.

    This is a local operational condition, never a remote robots document
    state; it must not be cached like remote loader errors.
    """

    reason: str
    retry_after_seconds: float | None = None
