"""Apply URL syntax, path/query shape, and discovery-noise rules."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, parse_qsl, unquote, urlparse

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.domains.host_normalizer import HostNormalizer


class UrlSyntaxRules:
    """URL syntax, shape, discovery noise, and low-value asset rules.

    Hard rules lists are injected from UrlFilterSettings (no class constants).
    """

    # These are internal and not (yet) exposed in settings
    _IMAGE_NOISE_EXTENSIONS = frozenset(
        {".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
    )
    _LOW_VALUE_EXTENSIONS = frozenset(
        {
            ".7z",
            ".bz2",
            ".gz",
            ".ico",
            ".ics",
            ".iso",
            ".m3u8",
            ".svg",
            ".rar",
            ".tar",
            ".tar.bz2",
            ".tar.gz",
            ".tar.xz",
            ".tgz",
            ".ts",
            ".xz",
            ".zip",
        }
    )
    _LOW_VALUE_SEGMENTS = frozenset(
        {
            "account",
            "calendar",
            "cart",
            "checkout",
            "ical",
            "login",
            "logout",
            "print",
            "register",
            "search",
            "sign-in",
            "sign-out",
            "signin",
            "signout",
            "signup",
        }
    )
    _LOW_VALUE_FRAGMENTS = ("/oembed/", "/wp-json/oembed", "/1.0/embed")

    def __init__(
        self,
        *,
        max_page_number: int,
        pagination_query_keys: tuple[str, ...],
        blocked_path_fragments: tuple[str, ...],
        blocked_query_keys: tuple[str, ...],
        blocked_query_value_patterns: dict[str, tuple[str, ...]],
        tracking_query_tokens: tuple[str, ...],
        low_value_image_path_fragments: tuple[str, ...],
        low_value_image_filenames: tuple[str, ...],
        social_icon_tokens: tuple[str, ...],
        logger: ProjectLogger,
    ) -> None:
        self._max_page_number = max_page_number
        self._pagination_query_keys = tuple(
            key.strip().casefold()
            for key in pagination_query_keys
            if key.strip()
        )
        self._blocked_path_fragments = tuple(
            fragment.casefold() for fragment in blocked_path_fragments
        )
        self._blocked_query_keys = frozenset(
            key.strip().casefold() for key in blocked_query_keys if key.strip()
        )
        blocked_value_patterns: dict[str, set[str]] = {}
        for raw_key, values in blocked_query_value_patterns.items():
            key = raw_key.strip().casefold()
            if not key:
                continue
            blocked_value_patterns.setdefault(key, set()).update(
                value.casefold() for value in values
            )
        self._blocked_query_value_patterns = MappingProxyType(
            {
                key: frozenset(values)
                for key, values in blocked_value_patterns.items()
            }
        )
        self._tracking_tokens = tuple(
            token.casefold() for token in tracking_query_tokens
        )
        self._low_value_image_fragments = tuple(
            fragment.casefold() for fragment in low_value_image_path_fragments
        )
        self._low_value_image_filenames = frozenset(
            filename.casefold() for filename in low_value_image_filenames
        )
        self._social_icon_tokens = tuple(
            token.casefold() for token in social_icon_tokens
        )
        self._logger = logger

    # ------------------------------------------------------------------
    # URL parsing
    # ------------------------------------------------------------------
    def parse_url(self, url: str) -> ParseResult | None:
        try:
            return urlparse(url)
        except ValueError as exc:
            self._logger.debug(
                "url_filter_rejected",
                extra={
                    "url_host": self._extract_host_for_log(url),
                    "reason": "invalid_url",
                    "error_type": type(exc).__name__,
                },
            )
            return None

    # ------------------------------------------------------------------
    # Host parsing
    # ------------------------------------------------------------------
    def extract_normalized_host(
        self,
        *,
        url: str,
        parsed: ParseResult,
        host_extractor: HostExtractor,
        host_normalizer: HostNormalizer,
    ) -> str:
        host = (
            host_normalizer.normalize(self.safe_hostname(parsed) or "") or ""
        )
        if host:
            return host
        extracted = host_extractor.extract(url)
        return host_normalizer.normalize(extracted or "") or ""

    # ------------------------------------------------------------------
    # Discovery noise
    # ------------------------------------------------------------------
    def discovery_noise_rejection_reason(
        self,
        *,
        url: str,
        path: str,
        query: str,
        kind: str | None,
        source_type: str,
    ) -> str | None:
        if source_type == "seed":
            return None

        policy_path, query_values = self._build_policy_view(
            path=path,
            query=query,
        )

        if kind not in {"image", "audio", "video", "document"}:
            discovery_reason = self._low_value_discovery_rejection_reason(
                path=policy_path
            )
            if discovery_reason is not None:
                self._logger.debug(
                    "url_filter_rejected",
                    extra={
                        "url_host": self._extract_host_for_log(url),
                        "reason": discovery_reason,
                        "stage": "discovery_noise",
                    },
                )
                return discovery_reason

        shape_discovery_reason = self._discovery_noise_reason(
            policy_path=policy_path,
            query_values=query_values,
            kind=kind,
        )
        if shape_discovery_reason is not None:
            self._logger.debug(
                "url_filter_rejected",
                extra={
                    "url_host": self._extract_host_for_log(url),
                    "reason": shape_discovery_reason,
                    "stage": "shape",
                },
            )
            return shape_discovery_reason

        return None

    # ------------------------------------------------------------------
    # URL shape / pagination
    # ------------------------------------------------------------------
    def url_shape_rejection_reason(
        self,
        *,
        url: str,
        path: str,
        query: str,
        source_type: str,
    ) -> str | None:
        policy_path, query_values = self._build_policy_view(
            path=path,
            query=query,
        )
        if self._is_email_artifact(
            url=url,
            policy_path=policy_path,
            query_values=query_values,
        ):
            return "email_artifact_url"
        if self._has_blocked_path(policy_path=policy_path):
            return "blocked_path_pattern"
        if source_type != "seed" and self._is_deep_pagination_policy(
            policy_path=policy_path,
            query_values=query_values,
        ):
            return "pagination_loop_candidate"
        return None

    def discovery_noise_reason(
        self,
        *,
        path: str,
        query: str,
        kind: str | None,
    ) -> str | None:
        policy_path, query_values = self._build_policy_view(
            path=path,
            query=query,
        )
        return self._discovery_noise_reason(
            policy_path=policy_path,
            query_values=query_values,
            kind=kind,
        )

    def _discovery_noise_reason(
        self,
        *,
        policy_path: str,
        query_values: dict[str, list[str]],
        kind: str | None,
    ) -> str | None:
        if self._is_tracking_or_social_asset(
            policy_path=policy_path,
            query_values=query_values,
        ):
            return "tracking_or_social_asset"
        if self._has_blocked_query(
            query_values,
            allow_feed_query=kind == "feed",
        ):
            return "blocked_query_pattern"
        return None

    # ------------------------------------------------------------------
    # Pagination rules
    # ------------------------------------------------------------------
    def _is_deep_pagination_policy(
        self,
        *,
        policy_path: str,
        query_values: dict[str, list[str]],
    ) -> bool:
        page_numbers = (
            *self._page_numbers_from_query(query_values=query_values),
            *self._page_numbers_from_path(policy_path=policy_path),
        )
        return any(number > self._max_page_number for number in page_numbers)

    @staticmethod
    def safe_hostname(parsed: ParseResult) -> str | None:
        try:
            return parsed.hostname
        except ValueError:
            return None

    @staticmethod
    def _extract_host_for_log(url: str) -> str:
        try:
            p = urlparse(url)
            return p.netloc or p.hostname or "unknown"
        except (
            ValueError,
            TypeError,
            AttributeError,
        ):  # exception-rules: best-effort-cleanup
            return "unknown"

    @staticmethod
    def _build_policy_view(
        *,
        path: str,
        query: str,
    ) -> tuple[str, dict[str, list[str]]]:
        policy_path = unquote(path or "").casefold()
        query_values: dict[str, list[str]] = {}
        for raw_key, value in parse_qsl(query, keep_blank_values=True):
            key = raw_key.strip().casefold()
            query_values.setdefault(key, []).append(value)
        return policy_path, query_values

    def _page_numbers_from_query(
        self,
        *,
        query_values: dict[str, list[str]],
    ) -> tuple[int, ...]:
        numbers: list[int] = []
        for key in self._pagination_query_keys:
            for value in query_values.get(key, ()):
                number = self._to_positive_int(value)
                if number is not None:
                    numbers.append(number)
        return tuple(numbers)

    @staticmethod
    def _page_numbers_from_path(*, policy_path: str) -> tuple[int, ...]:
        numbers: list[int] = []
        for match in re.finditer(
            r"(?:^|/)(?:page|p|paged)/(?P<number>\d+)(?:/|$)",
            policy_path,
        ):
            number = UrlSyntaxRules._to_positive_int(match.group("number"))
            if number is not None:
                numbers.append(number)
        return tuple(numbers)

    @staticmethod
    def _to_positive_int(value: object) -> int | None:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    # ------------------------------------------------------------------
    # Path filtering
    # ------------------------------------------------------------------
    def _has_blocked_path(self, *, policy_path: str) -> bool:
        return any(
            fragment in policy_path
            for fragment in self._blocked_path_fragments
        )

    # ------------------------------------------------------------------
    # Query + path filtering helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Query filtering
    # ------------------------------------------------------------------
    def _has_blocked_query(
        self,
        query_values: dict[str, list[str]],
        *,
        allow_feed_query: bool = False,
    ) -> bool:
        if not query_values:
            return False
        if any(key in self._blocked_query_keys for key in query_values):
            return True
        return any(
            self._contains_blocked_value(
                values=values,
                blocked_values=blocked_values,
                key=key,
                allow_feed_query=allow_feed_query,
            )
            for key, blocked_values in self._blocked_query_value_patterns.items()
            for values in (query_values.get(key),)
            if values
        )

    @staticmethod
    def _contains_blocked_value(
        *,
        values: list[str],
        blocked_values: frozenset[str],
        key: str,
        allow_feed_query: bool,
    ) -> bool:
        if allow_feed_query and key in {"feed", "format", "output"}:
            blocked_values = blocked_values.difference({"rss", "atom", "xml"})
        normalized_tokens: set[str] = set()
        for value in values:
            for token in re.split(r"[\s,]+", value.strip().casefold()):
                if token:
                    normalized_tokens.add(token)
        return any(
            blocked == token or blocked in token
            for blocked in blocked_values
            for token in normalized_tokens
        )

    @staticmethod
    def _is_email_artifact(
        *,
        url: str,
        policy_path: str,
        query_values: dict[str, list[str]],
    ) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme.casefold() == "mailto":
            return True
        query_parts = [
            part.casefold()
            for key, values in query_values.items()
            for part in (key, *values)
        ]
        haystack = " ".join(
            (parsed.netloc.casefold(), policy_path, *query_parts)
        )
        return bool(
            re.search(
                r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])",
                haystack,
            )
        )

    # ------------------------------------------------------------------
    # Embedded / static asset noise
    # ------------------------------------------------------------------
    def _is_tracking_or_social_asset(
        self,
        *,
        policy_path: str,
        query_values: dict[str, list[str]],
    ) -> bool:
        suffix = PurePosixPath(policy_path).suffix.casefold()
        if suffix not in self._IMAGE_NOISE_EXTENSIONS:
            return False
        if any(
            fragment in policy_path
            for fragment in self._low_value_image_fragments
        ):
            return True
        filename = PurePosixPath(policy_path).name.casefold()
        if filename in self._low_value_image_filenames:
            return True
        query_parts = (
            part.casefold()
            for key, values in query_values.items()
            for part in (key, *values)
        )
        policy_parts = (policy_path, *query_parts)
        return any(
            token in part
            for token in self._tracking_tokens + self._social_icon_tokens
            for part in policy_parts
        )

    def _low_value_discovery_rejection_reason(
        self,
        *,
        path: str,
    ) -> str | None:
        if self._has_low_value_discovery_extension(path):
            return "low_value_discovery_extension"
        if self._is_low_value_discovery_path(path):
            return "low_value_discovery_path"
        return None

    def _has_low_value_discovery_extension(self, path: str) -> bool:
        suffixes = [
            suffix.lower() for suffix in PurePosixPath(path or "").suffixes
        ]
        return any(
            "".join(suffixes[start_index:]) in self._LOW_VALUE_EXTENSIONS
            for start_index in range(len(suffixes))
        )

    def _is_low_value_discovery_path(self, path: str) -> bool:
        normalized_path = (path or "").lower()
        if any(
            fragment in normalized_path
            for fragment in self._LOW_VALUE_FRAGMENTS
        ):
            return True
        segments = {
            segment
            for segment in normalized_path.strip("/").split("/")
            if segment
        }
        return bool(segments.intersection(self._LOW_VALUE_SEGMENTS))
