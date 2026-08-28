"""Resolve raw URL candidates into normalized, fetchable URLs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import SplitResult, urljoin, urlsplit

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
    from crawler.extraction.assets.candidate.asset_extraction_records import (
        AssetCandidate,
    )
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from logger.project_logger import ProjectLogger


_DEFAULT_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_MAX_URL_LENGTH = 8192
_FRAGMENT_ONLY_PREFIX = "#"
_DATA_SCHEME = "data"


@dataclass(frozen=True, slots=True)
class ExtractionCandidate:
    """Normalized fetch candidate with its source and crawl context."""

    url: str
    kind: str | None = None
    context: CrawlTaskContext | None = None
    asset: AssetCandidate | None = None
    source_type: str = "unknown"


class UrlCandidateResolution:
    """Resolve one raw URL candidate through a narrow public API."""

    def __init__(
        self,
        *,
        url_normalizer: UrlNormalizer,
        logger: ProjectLogger,
        include_data_urls: bool = False,
        allowed_schemes: tuple[str, ...] = _DEFAULT_ALLOWED_SCHEMES,
        max_url_length: int = _DEFAULT_MAX_URL_LENGTH,
    ) -> None:
        if url_normalizer is None:
            raise ValueError("url_normalizer is required")

        if max_url_length < 1:
            raise ValueError("max_url_length must be positive")

        normalized_schemes = frozenset(
            scheme.strip().casefold()
            for scheme in allowed_schemes
            if isinstance(scheme, str) and scheme.strip()
        )

        if not normalized_schemes:
            raise ValueError(
                "allowed_schemes must contain at least one scheme"
            )

        if _DATA_SCHEME in normalized_schemes:
            raise ValueError(
                "configure data URLs through include_data_urls, "
                "not allowed_schemes"
            )

        self._url_normalizer = url_normalizer
        self._logger = logger
        self._include_data_urls = bool(include_data_urls)
        self._allowed_schemes = normalized_schemes
        self._max_url_length = int(max_url_length)
        self._logger.debug("url_candidate_resolution_initialized")

    def resolve(
        self,
        *,
        base_url: str,
        candidate: str,
    ) -> str | None:
        """Return one normalized absolute URL or ``None``."""

        candidate_value = self._prepare_value(candidate)
        if candidate_value is None:
            return None

        parsed_candidate = self._safe_split(candidate_value)
        if parsed_candidate is None:
            return None

        if not self._has_valid_port(parsed_candidate):
            return None

        candidate_scheme = parsed_candidate.scheme.casefold()

        if candidate_scheme == _DATA_SCHEME:
            return self._resolve_data_url(candidate_value)

        if candidate_scheme and candidate_scheme not in self._allowed_schemes:
            return None

        absolute_url = self._resolve_absolute_url(
            base_url=base_url,
            candidate=candidate_value,
            candidate_scheme=candidate_scheme,
        )
        if absolute_url is None:
            return None

        normalized_url = self._normalize(absolute_url)
        if normalized_url is None:
            return None

        parsed_normalized = self._safe_split(normalized_url)
        if parsed_normalized is None:
            return None

        if not self._is_fetchable_url(parsed_normalized):
            return None

        return normalized_url

    def _prepare_value(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None

        prepared = value.strip()
        if not prepared:
            return None

        if len(prepared) > self._max_url_length:
            return None

        if prepared.startswith(_FRAGMENT_ONLY_PREFIX):
            return None

        if self._contains_control_character(prepared):
            return None

        if "\\" in prepared:
            return None

        return prepared

    def _resolve_data_url(self, candidate: str) -> str | None:
        if not self._include_data_urls:
            return None

        if not self._is_valid_data_url(candidate):
            return None

        return candidate

    def _resolve_absolute_url(
        self,
        *,
        base_url: str,
        candidate: str,
        candidate_scheme: str,
    ) -> str | None:
        if candidate_scheme:
            return candidate

        base_value = self._prepare_value(base_url)
        if base_value is None:
            return None

        parsed_base = self._safe_split(base_value)
        if parsed_base is None:
            return None

        if not self._is_fetchable_url(parsed_base):
            return None

        try:
            return urljoin(base_value, candidate)
        except (TypeError, ValueError, UnicodeError):
            return None

    def _normalize(self, absolute_url: str) -> str | None:
        try:
            normalized_url = self._url_normalizer.normalize(absolute_url)
        except (TypeError, ValueError, UnicodeError):
            return None

        return self._prepare_value(normalized_url)

    def _is_fetchable_url(self, parsed: SplitResult) -> bool:
        if parsed.scheme.casefold() not in self._allowed_schemes:
            return False

        if parsed.username is not None or parsed.password is not None:
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        if any(
            character.isspace()
            or self._is_control_character(character)
            or character in {"\\", "/"}
            for character in hostname
        ):
            return False

        return self._has_valid_port(parsed)

    @staticmethod
    def _has_valid_port(parsed: SplitResult) -> bool:
        try:
            _ = parsed.port
        except ValueError:
            return False

        return True

    @classmethod
    def _is_valid_data_url(cls, value: str) -> bool:
        if cls._contains_control_character(value):
            return False

        scheme, separator, payload = value.partition(":")
        if separator != ":" or scheme.casefold() != _DATA_SCHEME:
            return False

        metadata, comma, _data = payload.partition(",")
        if comma != ",":
            return False

        return not cls._contains_control_character(metadata)

    @staticmethod
    def _safe_split(value: str) -> SplitResult | None:
        try:
            return urlsplit(value)
        except (TypeError, ValueError, UnicodeError):
            return None

    @classmethod
    def _contains_control_character(cls, value: str) -> bool:
        return any(cls._is_control_character(character) for character in value)

    @staticmethod
    def _is_control_character(character: str) -> bool:
        codepoint = ord(character)
        return codepoint < 32 or codepoint == 127


__all__ = ["UrlCandidateResolution"]
