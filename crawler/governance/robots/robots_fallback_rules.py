"""Fallback and override rules for robots loading errors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from crawler.classification.media_kind_registry import known_extensions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crawler.governance.domains.host_normalizer import HostNormalizer


class RobotsFallbackRules:
    """Evaluate robots error fallback rules that depend on target hosts."""

    def __init__(
        self,
        *,
        http_403_allow_host_suffixes: Sequence[str] | None,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._host_normalizer = host_normalizer
        self._http_403_allow_host_suffixes = self._normalize_host_suffixes(
            http_403_allow_host_suffixes
        )

    def is_allowed_host_suffix(
        self,
        *,
        target_url: str | None,
        allowed_host_suffixes: Sequence[str] | None,
    ) -> bool:
        """Return whether a target host is covered by the 403 override."""
        host = self._host_from_url(target_url)
        if not host:
            return False

        suffixes = (
            self._normalize_host_suffixes(allowed_host_suffixes)
            if allowed_host_suffixes is not None
            else self._http_403_allow_host_suffixes
        )
        return any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in suffixes
        )

    def is_storage_backed_media_asset(
        self,
        *,
        target_url: str | None,
    ) -> bool:
        """
        Return whether a URL looks like a media asset on storage/CDN infra.
        """
        parsed = urlsplit(target_url or "")
        host = self._host_normalizer.normalize(parsed.hostname)
        path = parsed.path.lower()
        if not host or not path:
            return False

        if not any(path.endswith(ext) for ext in known_extensions()):
            return False

        storage_suffixes = (
            "amazonaws.com",
            "blob.core.windows.net",
            "cloudfront.net",
            "azureedge.net",
            "googleusercontent.com",
        )
        if any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in storage_suffixes
        ):
            return True

        storage_markers = ("s3", "cdn", "media", "assets", "static", "blob")
        labels = tuple(label for label in host.split(".") if label)
        return any(
            marker in label for label in labels for marker in storage_markers
        )

    def _normalize_host_suffixes(
        self,
        suffixes: Sequence[str] | None,
    ) -> tuple[str, ...]:
        """Return normalized host suffixes once and consistently."""
        if not suffixes:
            return ()

        normalized: list[str] = []
        seen: set[str] = set()

        for suffix in suffixes:
            if not suffix:
                continue
            value = self._host_normalizer.normalize(suffix.lstrip("."))
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)

        return tuple(normalized)

    def _host_from_url(self, url: str | None) -> str | None:
        try:
            return self._host_normalizer.normalize(
                urlsplit(url or "").hostname
            )
        except ValueError:
            return None
