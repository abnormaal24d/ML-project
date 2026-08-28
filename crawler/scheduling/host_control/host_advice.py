"""Host-specific scheduling hints for intake and dispatch decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostAdvice:
    """Host-specific scheduling hints for intake and dispatch decisions."""

    discovery_factor: float | None = None
    priority_penalty: float | None = None
    hostility_score: float | None = None
    crawl_delay_seconds: float | None = None
