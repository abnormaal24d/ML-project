"""URL-scheme rules for crawler governance admission.

Exports: UrlSchemeRules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.fetching import UrlSchemeValidatorSettings


class UrlSchemeRules:
    """Validate URL schemes against the configured allow list."""

    def __init__(
        self,
        *,
        settings: UrlSchemeValidatorSettings,
        logger: ProjectLogger,
    ) -> None:
        self._allowed_schemes = frozenset(
            scheme.strip().lower()
            for scheme in settings.allowed_schemes
            if scheme and scheme.strip()
        )
        self._logger = logger

    def is_allowed(self, url: str) -> bool:
        """Return whether the URL scheme is allowed."""

        try:
            scheme = urlparse(url.strip()).scheme.strip().lower()
        except ValueError:
            scheme = ""
        allowed = scheme in self._allowed_schemes

        if not allowed:
            self._logger.debug(
                "url_scheme_rejected",
                extra={
                    "url_host": self._host_from_url(url),
                    "scheme": scheme,
                    "reason": "scheme_not_allowed",
                },
            )

        return allowed

    @staticmethod
    def _host_from_url(url: str) -> str:
        try:
            return urlparse(url).netloc or urlparse(url).hostname or "unknown"
        except Exception:  # exception-rules: best-effort-cleanup
            return "unknown"
