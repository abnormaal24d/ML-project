"""Apply configured host-deny and social/share rules."""

from __future__ import annotations

from crawler.governance.domains.host_normalizer import HostNormalizer


class HostDenylistRules:
    _SOCIAL_OR_SHARE_ROOTS = frozenset(
        {
            "facebook.com",
            "twitter.com",
            "x.com",
            "instagram.com",
            "linkedin.com",
            "tiktok.com",
            "pinterest.com",
            "addtoany.com",
        }
    )

    def __init__(
        self,
        *,
        blocked_hosts: set[str],
        host_normalizer: HostNormalizer,
    ) -> None:
        self._host_normalizer = host_normalizer
        self._blocked_hosts = {
            normalized
            for host in blocked_hosts
            if (normalized := host_normalizer.normalize(host)) is not None
        }

    def rejection_reason(self, host: str) -> str | None:
        normalized = self._host_normalizer.normalize(host)
        if normalized is None:
            return None
        if self._is_blocked_by_config(normalized):
            return "configured_blocked_host"
        if self._is_social_or_share_host(normalized):
            return "blocked_social_or_share_host"
        return None

    def _is_blocked_by_config(self, host: str) -> bool:
        if not host:
            return False
        if host in self._blocked_hosts:
            return True
        return any(
            host.endswith(f".{blocked}") for blocked in self._blocked_hosts
        )

    @classmethod
    def _is_social_or_share_host(cls, host: str) -> bool:
        return any(
            host == root or host.endswith(f".{root}")
            for root in cls._SOCIAL_OR_SHARE_ROOTS
        )
