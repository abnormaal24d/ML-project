"""Public models and helpers for crawler.extraction.hosts_extractor.

Exports: HostExtractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger


class HostExtractor:
    """Extract hostnames from URLs."""

    def __init__(
        self,
        logger: ProjectLogger,
    ) -> None:
        self._logger = logger

    def extract(self, url: str) -> str | None:
        """Return the hostname for the URL."""

        try:
            hostname = urlparse(url).hostname
        except ValueError:
            hostname = None

        if hostname is None:
            self._logger.debug(
                "host_extractor_missing_host",
                url=url,
            )
            return None

        self._logger.debug(
            "host_extractor_resolved",
            url=url,
            host=hostname,
        )
        return hostname
