"""Mutable per-host rate-limit state."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class RateLimitHostState:
    """Mutable rate-control state maintained for one normalized host."""

    adaptive_requests_per_second: float
    effective_requests_per_second: float
    crawl_delay_override_seconds: float | None = None
    timestamps: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    next_request_not_before: float = 0.0
    rules_revision: int = 0
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False,
    )
