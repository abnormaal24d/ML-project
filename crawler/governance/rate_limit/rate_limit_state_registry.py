"""
Public models and helpers for
crawler.governance.rate_limit.rate_limit_state_registry.

Exports: RateLimitStateRegistry.
"""

from __future__ import annotations

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.rate_limit.rate_limit_host_state import (
    RateLimitHostState,
)

_DEFAULT_HOST_KEY = "unknown"


class RateLimitStateRegistry:
    """Resolve canonical host keys and own per-host rate state creation."""

    def __init__(
        self,
        *,
        host_normalizer: HostNormalizer,
        default_adaptive_requests_per_second: float,
        default_effective_requests_per_second: float,
    ) -> None:
        self._host_normalizer = host_normalizer
        self._default_adaptive_requests_per_second = (
            default_adaptive_requests_per_second
        )
        self._default_effective_requests_per_second = (
            default_effective_requests_per_second
        )
        self._states: dict[str, RateLimitHostState] = {}

    def resolve_host_key(self, host: str | None) -> str:
        """Return the canonical rate-limiter host key with fallback rules."""

        host_key = self._host_normalizer.normalize(host)
        return host_key or _DEFAULT_HOST_KEY

    def get(self, host: str | None) -> RateLimitHostState | None:
        """Return the state for a host when it exists."""

        return self._states.get(self.resolve_host_key(host))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def state_for_host(
        self, host: str | None
    ) -> tuple[str, RateLimitHostState]:
        """Return the canonical host key and create state on demand."""

        host_key = self.resolve_host_key(host)
        existing = self._states.get(host_key)
        if existing is not None:
            return host_key, existing

        created = RateLimitHostState(
            adaptive_requests_per_second=(
                self._default_adaptive_requests_per_second
            ),
            effective_requests_per_second=(
                self._default_effective_requests_per_second
            ),
        )
        return host_key, self._states.setdefault(host_key, created)
