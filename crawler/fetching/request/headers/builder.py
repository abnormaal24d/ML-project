"""Request header construction."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import definition_for
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from config.collection.identity import IdentitySettings

BROTLI_AVAILABLE = find_spec("brotli") is not None


class RequestHeaderBuilder:
    """Build request headers for one outbound fetch request."""

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        identity: IdentitySettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._identity = identity
        self._derived_accept_header = self._build_accept_header()
        self._logger = logger

    def build(
        self,
        *,
        url: str | None = None,
        host: str | None = None,
    ) -> dict[str, str]:
        """Build headers using fixed identity only."""

        headers: dict[str, str] = {
            "User-Agent": self._identity.user_agent,
        }

        if self._derived_accept_header:
            headers["Accept"] = self._derived_accept_header

        if self._settings.accept_language_header:
            headers["Accept-Language"] = self._settings.accept_language_header

        if self._settings.accept_compressed:
            headers["Accept-Encoding"] = self._build_accept_encoding_header()

        self._logger.debug(
            "request_headers_built",
            url=url,
            host=host,
            has_user_agent=True,
            has_accept="Accept" in headers,
            has_accept_language="Accept-Language" in headers,
            has_accept_encoding="Accept-Encoding" in headers,
            header_names=sorted(headers.keys()),
        )

        return headers

    def _build_accept_encoding_header(self) -> str:
        """Build the Accept-Encoding header value."""

        encodings = ["gzip", "deflate"]
        if BROTLI_AVAILABLE:
            encodings.append("br")

        header_value = ", ".join(encodings)
        self._logger.debug(
            "request_accept_encoding_built",
            brotli_available=BROTLI_AVAILABLE,
            accept_encoding=header_value,
        )
        return header_value

    @staticmethod
    def _build_accept_header() -> str:
        values: list[str] = []
        seen: set[str] = set()
        for kind in MediaKind:
            for mime_type in definition_for(kind).mime_types:
                normalized = str(mime_type).strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                values.append(normalized)
        values.append("*/*;q=0.1")
        return ",".join(values)
