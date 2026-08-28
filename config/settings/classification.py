"""Classification detector settings: full parity with the legacy models.

Full port of ``config/classification/`` (MIME detectors, encoding/language
detection, content categories, relevance scoring and MIME-to-kind
resolution).
"""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel
from config.environment.default_values import (
    DEFAULT_CONTENT_RELEVANCE_ROUNDING_DIGITS,
    DEFAULT_TEXT_SAMPLE_BYTES,
)


def _default_stopwords_by_language() -> dict[str, tuple[str, ...]]:
    return {
        "en": (
            "the",
            "and",
            "of",
            "to",
            "in",
            "for",
            "with",
            "on",
        ),
        "nl": (
            "de",
            "het",
            "een",
            "en",
            "van",
            "voor",
            "met",
            "op",
        ),
        "fr": (
            "le",
            "la",
            "les",
            "et",
            "de",
            "des",
            "pour",
            "avec",
        ),
    }


class ContentCategoryDetectorSettings(SettingsModel):
    enabled: bool = True

    classify_documents: bool = True
    classify_images: bool = True
    classify_audio: bool = True
    classify_video: bool = True

    default_category: str = "other"
    academic_url_markers: tuple[str, ...] = (
        "arxiv.org",
        "doi.org",
        "scholar",
        "research",
        "paper",
    )
    news_url_markers: tuple[str, ...] = (
        "/news/",
        "news.",
        "press-release",
    )
    media_kinds: tuple[str, ...] = ("image", "audio", "video")
    media_page_url_markers: tuple[str, ...] = (
        "/gallery",
        "/media",
        "/video",
        "/audio",
    )
    classify_boilerplate_text: bool = True
    boilerplate_snippet_markers: tuple[str, ...] = (
        "cookie rules",
        "privacy rules",
        "subscribe",
        "all rights reserved",
    )
    documentation_host_markers: tuple[str, ...] = (
        "docs.",
        "documentation.",
        "developer.",
    )
    documentation_url_markers: tuple[str, ...] = (
        "/docs",
        "/documentation",
        "/manual",
        "/guide",
        "/reference",
    )
    text_like_mime_types: tuple[str, ...] = (
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
    )
    snippet_bytes: int = Field(default=DEFAULT_TEXT_SAMPLE_BYTES, ge=1)
    documentation_score: float = Field(default=0.9, ge=0.0, le=1.0)
    academic_score: float = Field(default=0.85, ge=0.0, le=1.0)
    news_score: float = Field(default=0.8, ge=0.0, le=1.0)
    media_score: float = Field(default=0.75, ge=0.0, le=1.0)
    boilerplate_score: float = Field(default=0.7, ge=0.0, le=1.0)

    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ContentRelevanceScorerSettings(SettingsModel):
    enabled: bool = True

    minimum_score: float = Field(default=0.25, ge=0.0, le=1.0)

    normalize_scores: bool = True
    base_score: float = Field(default=0.45, ge=0.0, le=1.0)
    page_bonus: float = 0.1
    documentation_bonus: float = 0.25
    academic_bonus: float = 0.2
    media_bonus: float = 0.1
    boilerplate_penalty: float = 0.3
    small_payload_threshold_bytes: int = Field(default=512, ge=0)
    small_payload_penalty: float = 0.15
    large_payload_threshold_bytes: int = Field(default=50_000, ge=0)
    large_payload_bonus: float = 0.1
    rounding_digits: int = Field(
        default=DEFAULT_CONTENT_RELEVANCE_ROUNDING_DIGITS,
        ge=0,
    )


class ContentTypeDetectorSettings(SettingsModel):
    enabled: bool = True

    detect_from_headers: bool = True
    detect_from_extension: bool = True
    detect_from_signature: bool = True

    fallback_content_type: str = "application/octet-stream"
    generic_header_mime_types: tuple[str, ...] = (
        "application/octet-stream",
        "binary/octet-stream",
    )
    signature_validation_mime_types: tuple[str, ...] = (
        "application/octet-stream",
        "application/download",
    )
    signature_validation_major_types: tuple[str, ...] = (
        "image",
        "audio",
        "video",
        "application",
    )
    signature_exempt_mime_types: tuple[str, ...] = (
        "text/html",
        "text/plain",
        "application/json",
        "application/xml",
    )


class EncodingDetectorSettings(SettingsModel):
    enabled: bool = True

    default_encoding: str = "utf-8"

    detect_bom: bool = True
    fallback_to_utf8: bool = True

    allowed_encodings: tuple[str, ...] = (
        "utf-8",
        "utf-16",
        "latin-1",
    )
    charset_pattern: str = r"charset=([a-zA-Z0-9_\-]+)"
    sample_size_bytes: int = Field(default=DEFAULT_TEXT_SAMPLE_BYTES, ge=1)
    fallback_encoding: str = "utf-8"


class LanguageHeuristicSettings(SettingsModel):
    enabled: bool = True

    use_stopword_matching: bool = True
    use_unicode_ranges: bool = True

    min_text_length: int = Field(default=20, ge=1)
    min_best_score: int = Field(default=2, ge=1)
    require_clear_winner: bool = True
    stopwords_by_language: dict[str, tuple[str, ...]] = Field(
        default_factory=_default_stopwords_by_language
    )


class LanguageDetectorSettings(SettingsModel):
    enabled: bool = True

    minimum_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    fallback_language: str = "unknown"

    max_input_characters: int = Field(default=10000, ge=1)
    sample_size_bytes: int = Field(default=DEFAULT_TEXT_SAMPLE_BYTES, ge=1)
    language_attribute_pattern: str = (
        r"(?:lang|xml:lang)\s*=\s*[\"']([a-zA-Z-]+)[\"']"
    )
    meta_language_patterns: tuple[str, ...] = (
        r"<meta[^>]+(?:http-equiv=[\"']content-language[\"'][^>]+content|"
        r"content[^>]+http-equiv=[\"']content-language[\"'])\s*=\s*"
        r"[\"']([a-zA-Z-]+)[\"']",
        r"<meta[^>]+name=[\"']language[\"'][^>]+content=[\"']([a-zA-Z-]+)[\"']",
    )
    default_language: str = "unknown"
    use_fasttext: bool = False
    fasttext_model_path: str | None = None
    heuristic: LanguageHeuristicSettings = LanguageHeuristicSettings()


class MimeDetectorSettings(SettingsModel):
    enabled: bool = True

    trust_http_headers: bool = True
    trust_file_extensions: bool = True
    trust_file_signatures: bool = True

    fallback_mime_type: str = "application/octet-stream"
    prefer_python_magic: bool = False
    use_python_magic: bool = True


class MimeSignatureDetectorSettings(SettingsModel):
    enabled: bool = True

    scan_binary_headers: bool = True

    maximum_signature_size: int = Field(default=8192, ge=1)
    sample_size_bytes: int = Field(default=DEFAULT_TEXT_SAMPLE_BYTES, ge=1)
    use_filetype: bool = True


def _default_exact_kind_map() -> dict[str, str]:
    return {
        "text/html": "page",
        "application/rss+xml": "feed",
        "application/atom+xml": "feed",
        "application/pdf": "document",
        "application/msword": "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
        "image/webp": "image",
        "image/avif": "image",
        "image/heic": "image",
        "image/tiff": "image",
        "image/svg+xml": "image",
        "audio/mp4": "audio",
        "audio/m4a": "audio",
        "audio/aac": "audio",
        "audio/webm": "audio",
        "audio/ogg": "audio",
        "audio/wav": "audio",
        "audio/flac": "audio",
        "video/mp4": "video",
        "video/webm": "video",
        "video/quicktime": "video",
        "video/x-matroska": "video",
        "video/ogg": "video",
    }


def _default_prefix_kind_map() -> dict[str, str]:
    return {
        "image/": "image",
        "audio/": "audio",
        "video/": "video",
    }


class ContentKindResolverSettings(SettingsModel):
    fallback_kind: str = Field(default="page", min_length=1)
    exact_kind_map: dict[str, str] = Field(
        default_factory=_default_exact_kind_map,
    )
    prefix_kind_map: dict[str, str] = Field(
        default_factory=_default_prefix_kind_map,
    )
    document_mime_types: tuple[str, ...] = (
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/rtf",
        "text/rtf",
        "application/epub+zip",
        "application/x-parquet",
    )
    feed_markers: tuple[str, ...] = (
        "<rss",
        "<feed",
        "<rdf:rdf",
        "<channel",
        "<entry",
        "xmlns:atom",
        "<atom:",
    )
    feed_snippet_bytes: int = Field(default=2048, ge=1)


class ClassificationSettings(SettingsModel):
    mime_detector: MimeDetectorSettings = MimeDetectorSettings()
    mime_signature_detector: MimeSignatureDetectorSettings = (
        MimeSignatureDetectorSettings()
    )
    mime_type_resolver: ContentTypeDetectorSettings = (
        ContentTypeDetectorSettings()
    )
    encoding_detector: EncodingDetectorSettings = EncodingDetectorSettings()
    language_detector: LanguageDetectorSettings = LanguageDetectorSettings()
    content_category_detector: ContentCategoryDetectorSettings = (
        ContentCategoryDetectorSettings()
    )
    content_relevance_scorer: ContentRelevanceScorerSettings = (
        ContentRelevanceScorerSettings()
    )
    kind_resolver: ContentKindResolverSettings = ContentKindResolverSettings()
