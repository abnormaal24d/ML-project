"""Semantic content category detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from crawler.classification.media_kind import MediaKind
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.classification import ContentCategoryDetectorSettings


class ContentCategory(StrEnum):
    """Supported semantic content categories."""

    DOCUMENTATION = "documentation"
    ACADEMIC = "academic"
    NEWS = "news"
    MEDIA = "media"
    BOILERPLATE = "boilerplate"


@dataclass(frozen=True, slots=True)
class CategoryScore:
    """Score assigned to one semantic content category."""

    category: ContentCategory
    score: float
    reasons: tuple[str, ...] = ()


_CATEGORY_PRIORITY: dict[ContentCategory, int] = {
    ContentCategory.BOILERPLATE: 0,
    ContentCategory.DOCUMENTATION: 1,
    ContentCategory.ACADEMIC: 2,
    ContentCategory.NEWS: 3,
    ContentCategory.MEDIA: 4,
}


class ContentCategoryDetector:
    """Detect one semantic category from URL, kind, MIME, and decoded text."""

    def __init__(
        self,
        *,
        settings: ContentCategoryDetectorSettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._media_kinds = frozenset(
            MediaKind.parse(kind) for kind in (settings.media_kinds or ())
        )

    def detect(
        self,
        *,
        url: str,
        kind: MediaKind,
        content_type: str | None,
        text_sample: str | None,
    ) -> ContentCategory | None:
        """Detect the most likely semantic category."""

        if not self._settings.enabled or not self._should_classify_kind(kind):
            return None

        normalized_url = url.strip().casefold()
        normalized_content_type = normalize_mime_type(content_type)

        try:
            parsed = urlsplit(normalized_url)
        except ValueError as exc:
            self._logger.debug(
                "content_category_url_parse_failed",
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            parsed = urlsplit("")

        scores = (
            self._score_documentation(
                host=(parsed.hostname or "").casefold(),
                path=parsed.path.casefold(),
                query=parsed.query.casefold(),
            ),
            self._score_academic(normalized_url),
            self._score_news(normalized_url),
            self._score_media(
                kind,
                normalized_url,
            ),
            self._score_boilerplate(
                kind,
                normalized_content_type,
                text_sample or "",
            ),
        )

        best = min(
            scores,
            key=lambda item: (
                -item.score,
                _CATEGORY_PRIORITY[item.category],
            ),
        )

        if best.score < self._settings.minimum_confidence:
            return _parse_default_category(self._settings.default_category)

        return best.category

    def _should_classify_kind(
        self,
        kind: MediaKind,
    ) -> bool:
        """Return whether the supplied media kind should be classified."""

        return not (
            (
                kind is MediaKind.DOCUMENT
                and not self._settings.classify_documents
            )
            or (kind is MediaKind.IMAGE and not self._settings.classify_images)
            or (kind is MediaKind.AUDIO and not self._settings.classify_audio)
            or (kind is MediaKind.VIDEO and not self._settings.classify_video)
        )

    def _score_documentation(
        self,
        *,
        host: str,
        path: str,
        query: str,
    ) -> CategoryScore:
        """Score documentation-specific URL characteristics."""

        reasons: list[str] = []

        if self._contains_any(
            host,
            self._settings.documentation_host_markers,
        ):
            reasons.append("documentation_host")

        if self._contains_any(
            path,
            self._settings.documentation_url_markers,
        ):
            reasons.append("documentation_path")

        if self._contains_any(
            query,
            self._settings.documentation_url_markers,
        ):
            reasons.append("documentation_query")

        return CategoryScore(
            category=ContentCategory.DOCUMENTATION,
            score=(self._settings.documentation_score if reasons else 0.0),
            reasons=tuple(reasons),
        )

    def _score_academic(
        self,
        url: str,
    ) -> CategoryScore:
        """Score academic URL characteristics."""

        matched = self._contains_any(
            url,
            self._settings.academic_url_markers,
        )

        return CategoryScore(
            category=ContentCategory.ACADEMIC,
            score=(self._settings.academic_score if matched else 0.0),
            reasons=("academic_url",) if matched else (),
        )

    def _score_news(
        self,
        url: str,
    ) -> CategoryScore:
        """Score news URL characteristics."""

        matched = self._contains_any(
            url,
            self._settings.news_url_markers,
        )

        return CategoryScore(
            category=ContentCategory.NEWS,
            score=(self._settings.news_score if matched else 0.0),
            reasons=("news_url",) if matched else (),
        )

    def _score_media(
        self,
        kind: MediaKind,
        url: str,
    ) -> CategoryScore:
        """Score media-oriented characteristics."""

        reasons: list[str] = []

        if kind in self._media_kinds:
            reasons.append("media_kind")

        if kind is MediaKind.PAGE and self._contains_any(
            url,
            self._settings.media_page_url_markers,
        ):
            reasons.append("media_page_url")

        return CategoryScore(
            category=ContentCategory.MEDIA,
            score=(self._settings.media_score if reasons else 0.0),
            reasons=tuple(reasons),
        )

    def _score_boilerplate(
        self,
        kind: MediaKind,
        content_type: str | None,
        text: str,
    ) -> CategoryScore:
        """Score boilerplate characteristics."""

        matched = bool(
            self._settings.classify_boilerplate_text
            and kind is MediaKind.PAGE
            and self._is_text_like_mime_type(content_type)
            and self._contains_any(
                text[: self._settings.snippet_bytes].casefold(),
                self._settings.boilerplate_snippet_markers,
            )
        )

        return CategoryScore(
            category=ContentCategory.BOILERPLATE,
            score=(self._settings.boilerplate_score if matched else 0.0),
            reasons=("boilerplate_text",) if matched else (),
        )

    def _is_text_like_mime_type(
        self,
        content_type: str | None,
    ) -> bool:
        """Return whether a MIME type represents text-like content."""

        return bool(
            content_type
            and (
                content_type.startswith("text/")
                or content_type in self._settings.text_like_mime_types
            )
        )

    @staticmethod
    def _contains_any(
        value: str,
        markers: tuple[str, ...],
    ) -> bool:
        """Return whether the value contains any configured marker."""

        return any(
            normalized_marker in value
            for marker in markers
            if (normalized_marker := marker.strip().casefold())
        )


def _parse_default_category(
    value: object,
) -> ContentCategory | None:
    """Parse a configured default content category."""

    if value is None:
        return None

    if isinstance(value, ContentCategory):
        return value

    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()

    if not normalized:
        return None

    try:
        return ContentCategory(normalized)
    except ValueError:
        return None
