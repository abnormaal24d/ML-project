"""Orchestrate response classification from bounded evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.classification.media_kind import (
    MediaKind,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.classification.content_category_detector import (
        ContentCategoryDetector,
    )
    from crawler.classification.content_kind_resolver import (
        ContentKindResolver,
    )
    from crawler.classification.content_relevance import (
        ContentRelevanceScorer,
    )
    from crawler.classification.encoding_detector import (
        EncodingDetector,
    )
    from crawler.classification.language_detector import (
        LanguageDetector,
    )
    from crawler.classification.mime_type_resolver import (
        MimeTypeResolver,
    )


@dataclass(frozen=True, slots=True)
class ClassifiedContent:
    """Normalized classification metadata for one fetched resource."""

    raw_content_type_header: str | None
    normalized_mime_type: str | None

    encoding: str | None
    encoding_confidence: float | None

    language: str | None
    language_confidence: float | None

    kind: MediaKind

    category: str | None = None
    relevance_score: float | None = None

    mime_conflict: bool = False
    mime_source: str | None = None
    major_mime_conflict: bool = False

    encoding_source: str | None = None

    language_source: str | None = None
    language_detector_version: str | None = None

    kind_source: str | None = None
    content_role: str | None = None


@dataclass(frozen=True, slots=True)
class ContentClassifierConfig:
    """Runtime configuration required by ContentClassifier."""

    default_language: str
    text_metadata_sample_bytes: int


@dataclass(frozen=True, slots=True)
class _DetectedTextMetadata:
    """Internal decoded text metadata for one classification pass."""

    encoding: str | None
    encoding_confidence: float | None

    language: str | None
    language_confidence: float | None

    encoding_source: str | None = None
    language_source: str | None = None

    decoded_text: str | None = None


class ContentClassifier:
    """Coordinate MIME, kind, text, category and relevance classification."""

    def __init__(
        self,
        *,
        mime_type_resolver: MimeTypeResolver,
        encoding_detector: EncodingDetector,
        language_detector: LanguageDetector,
        kind_resolver: ContentKindResolver,
        category_detector: ContentCategoryDetector,
        relevance_scorer: ContentRelevanceScorer,
        config: ContentClassifierConfig,
        logger: ProjectLogger,
    ) -> None:
        self._mime_type_resolver = mime_type_resolver
        self._encoding_detector = encoding_detector
        self._language_detector = language_detector
        self._kind_resolver = kind_resolver
        self._category_detector = category_detector
        self._relevance_scorer = relevance_scorer

        self._default_language = config.default_language

        self._text_metadata_sample_bytes = max(
            1,
            int(config.text_metadata_sample_bytes),
        )

        self._logger = logger

    def classify(
        self,
        *,
        url: str,
        content_type_header: str | None,
        sniff_bytes: bytes,
        payload_byte_size: int | None = None,
        requested_kind: (str | MediaKind | None) = None,
    ) -> ClassifiedContent:
        """Classify a bounded sniff sample and real payload size."""

        sample = sniff_bytes

        actual_size = (
            len(sample)
            if payload_byte_size is None
            else max(
                0,
                int(payload_byte_size),
            )
        )

        mime = self._mime_type_resolver.detect_resolution(
            url=url,
            content_type_header=content_type_header,
            sample=sample,
        )

        kind = self._kind_resolver.resolve_with_metadata(
            content_type=mime.mime_type,
            body=sample,
            url=url,
            requested_kind=requested_kind,
        )

        text = self._detect_text_metadata(
            content_type_header=content_type_header,
            sample=sample,
            mime_type=mime.mime_type,
        )

        category_enum = self._category_detector.detect(
            url=url,
            kind=kind.kind,
            content_type=mime.mime_type,
            text_sample=text.decoded_text,
        )

        category = category_enum.value if category_enum is not None else None

        relevance = self._relevance_scorer.score(
            kind=kind.kind,
            category=category_enum,
            byte_size=actual_size,
        )

        result = ClassifiedContent(
            raw_content_type_header=(mime.raw_header),
            normalized_mime_type=(mime.mime_type),
            encoding=text.encoding,
            encoding_confidence=(text.encoding_confidence),
            language=text.language,
            language_confidence=(text.language_confidence),
            kind=kind.kind,
            category=category,
            relevance_score=relevance,
            mime_conflict=(mime.exact_conflict),
            mime_source=mime.source,
            major_mime_conflict=(mime.major_conflict),
            encoding_source=(text.encoding_source),
            language_source=(text.language_source),
            language_detector_version=(self._language_detector.version),
            kind_source=kind.source,
            content_role=kind.role,
        )

        self._logger.debug(
            "content_classified",
            url=url,
            meta={
                "kind": result.kind.value,
                "kind_source": (result.kind_source),
                "content_role": (result.content_role),
                "category": (result.category),
                "raw_content_type_header": (result.raw_content_type_header),
                "normalized_mime_type": (result.normalized_mime_type),
                "mime_source": (result.mime_source),
                "mime_conflict": (result.mime_conflict),
                "major_mime_conflict": (result.major_mime_conflict),
                "encoding": (result.encoding),
                "encoding_source": (result.encoding_source),
                "language": (result.language),
                "language_source": (result.language_source),
                "relevance_score": (result.relevance_score),
                "sniff_sample_bytes": (len(sample)),
                "payload_byte_size": (actual_size),
            },
        )

        return result

    def _detect_text_metadata(
        self,
        *,
        content_type_header: str | None,
        sample: bytes,
        mime_type: str | None,
    ) -> _DetectedTextMetadata:
        if not (self._kind_resolver.should_detect_text_metadata(mime_type)):
            return _DetectedTextMetadata(
                None,
                None,
                None,
                None,
            )

        text_sample = sample[: self._text_metadata_sample_bytes]

        encoding = self._encoding_detector.detect(
            content_type_header=content_type_header,
            body=text_sample,
        )

        decoded_text = _decode_sample(
            text_sample,
            encoding.value,
        )

        language = (
            self._language_detector.detect_text(markup_text=decoded_text)
            if decoded_text
            else None
        )

        language_value = language.value if language is not None else None

        language_confidence = (
            language.confidence if language is not None else None
        )

        language_source = language.source if language is not None else None

        if not language_value and self._default_language:
            language_value = self._default_language
            language_confidence = 0.0
            language_source = "default"

        return _DetectedTextMetadata(
            encoding=encoding.value,
            encoding_confidence=(encoding.confidence),
            language=language_value,
            language_confidence=(language_confidence),
            encoding_source=(encoding.source),
            language_source=(language_source),
            decoded_text=decoded_text,
        )


def _decode_sample(
    sample: bytes,
    encoding: str | None,
) -> str:
    """Decode a bounded sample using the detected codec."""

    if not sample:
        return ""

    try:
        return sample.decode(
            encoding or "utf-8",
            errors="replace",
        )
    except LookupError:
        return sample.decode(
            "utf-8",
            errors="replace",
        )
