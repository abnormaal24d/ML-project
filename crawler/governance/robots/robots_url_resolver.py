"""Build canonical robots.txt URLs from arbitrary resource URLs."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from crawler.governance.domains.host_normalizer import HostNormalizer
from logger.project_logger import ProjectLogger


class RobotsUrlResolver:
    """Build canonical robots.txt URLs from arbitrary resource URLs."""

    def __init__(
        self,
        robots_path: str,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
    ) -> None:
        self._robots_path = (
            robots_path if robots_path.startswith("/") else f"/{robots_path}"
        )
        self._logger = logger
        self._host_normalizer = host_normalizer

    # ------------------------------------------------------------------
    # URL building logic
    # ------------------------------------------------------------------
    def build(self, url: str) -> str:
        """Return the canonical robots.txt URL for the resource URL."""

        parsed = urlsplit(url)

        hostname = self._host_normalizer.require(parsed.hostname)

        scheme = parsed.scheme.lower()

        netloc = f"[{hostname}]" if ":" in hostname else hostname

        if parsed.port is not None:
            is_default_port = (scheme == "http" and parsed.port == 80) or (
                scheme == "https" and parsed.port == 443
            )
            if not is_default_port:
                netloc = f"{netloc}:{parsed.port}"

        robots_url = urlunsplit(
            (
                scheme,
                netloc,
                self._robots_path,
                "",
                "",
            )
        )

        self._logger.debug(
            "robots_url_built",
            extra={
                "url_host": self._host_from_url(url),
                "robots_url": robots_url,
            },
        )
        return robots_url

    @staticmethod
    def _host_from_url(url: str) -> str:
        try:
            return urlsplit(url).netloc or urlsplit(url).hostname or "unknown"
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):  # exception-rules: best-effort-cleanup
            return "unknown"
