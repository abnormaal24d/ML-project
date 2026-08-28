"""Settings for crawler media and document processors."""

from __future__ import annotations

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel
from config.environment.default_values import (
    DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    DEFAULT_FEED_PROCESSOR_MAX_ENTRIES,
    DEFAULT_PROCESSOR_RETRIES,
    DEFAULT_TEXT_SAMPLE_BYTES,
)


class BaseProcessorSettings(SettingsModel):
    """Shared processor defaults.

    Timeout defaults are measured in seconds; retry defaults count additional
    attempts after the first execution attempt.
    """

    enabled: bool = Field(default=True)
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    max_retries: int = Field(default=DEFAULT_PROCESSOR_RETRIES, ge=0, le=32)
    max_concurrent_tasks: int = Field(default=8, ge=0, le=1024)
    persist_raw: bool = Field(default=True)


class AudioProcessorSettings(BaseProcessorSettings):
    analysis_workers: int = Field(default=2, ge=1, le=64)
    analysis_queue_size: int = Field(default=80, ge=1, le=10000)
    analysis_timeout_seconds: float = Field(default=180.0, gt=0.0, le=3600.0)
    sample_rate: int = Field(
        default=DEFAULT_AUDIO_SAMPLE_RATE_HZ,
        ge=8000,
        le=192000,
    )
    normalize_audio: bool = True
    run_transcription: bool = False
    transcription_language: str | None = Field(
        default=None, min_length=2, max_length=32
    )
    extract_metadata: bool = True
    max_transcription_bytes: int = Field(default=20 * 1024 * 1024, ge=0)
    min_bytes: int = Field(default=512, ge=0)
    min_sample_rate: int = Field(default=8_000, ge=1)
    max_channels: int = Field(default=8, ge=1)
    min_duration_seconds: float = Field(default=0.2, ge=0.0)
    max_duration_seconds: float = Field(default=3600.0, gt=0.0)
    require_metadata_for_acceptance: bool = False

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> AudioProcessorSettings:
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError(
                "max_duration_seconds must be greater than or equal to "
                "min_duration_seconds"
            )
        return self


class AverageHashSettings(SettingsModel):
    size: int = Field(default=8, ge=2, le=64)


class PdfTextExtractionSettings(SettingsModel):
    max_pages: int = Field(default=8, ge=1, le=256)


class DocumentNativeTextSettings(SettingsModel):
    """Bounded native document text retained during collection."""

    max_characters: int = Field(
        default=DEFAULT_TEXT_SAMPLE_BYTES,
        ge=0,
        le=1_000_000,
    )


class HtmlTextPreviewSettings(SettingsModel):
    max_characters: int = Field(default=32768, ge=0, le=2_000_000)


class PageTextExtractionSettings(SettingsModel):
    """Crawler-owned structural HTML text extraction settings."""

    max_text_chars: int = Field(default=200_000, ge=1)
    preview_max_chars: int = Field(default=32_768, ge=0)
    drop_tags: tuple[str, ...] = (
        "script",
        "style",
        "noscript",
        "svg",
    )
    drop_selectors: tuple[str, ...] = (
        "nav",
        "header",
        "footer",
        "aside",
        ".sidebar",
        ".sphinxsidebar",
        ".related",
        ".toctree-wrapper",
        ".breadcrumbs",
        ".wy-menu",
        ".wy-side-nav-search",
        ".wy-nav-side",
        ".wy-nav-top",
        "#searchbox",
    )
    main_selectors: tuple[str, ...] = (
        "main",
        "article",
        "[role='main']",
        ".document",
        ".documentwrapper",
        ".body",
        ".section",
    )


class VideoKeyframeSettings(SettingsModel):
    enabled: bool = True
    max_keyframes: int = Field(default=8, ge=0, le=512)


class ImageProcessorSettings(BaseProcessorSettings):
    analysis_workers: int = Field(default=6, ge=1, le=64)
    analysis_queue_size: int = Field(default=200, ge=1, le=10000)
    analysis_timeout_seconds: float = Field(default=45.0, gt=0.0, le=3600.0)
    average_hash: AverageHashSettings = Field(
        default_factory=AverageHashSettings
    )
    resize_images: bool = True
    max_image_width: int = Field(default=2048, ge=1)
    max_image_height: int = Field(default=2048, ge=1)
    extract_metadata: bool = True
    run_ocr: bool = True
    max_ocr_bytes: int = Field(default=20 * 1024 * 1024, ge=0)
    min_bytes: int = Field(default=1, ge=0)
    min_width: int = Field(default=1, ge=1)
    min_height: int = Field(default=1, ge=1)
    max_aspect_ratio: float = Field(default=6.0, gt=0.0)
    reject_unknown_dimensions: bool = False
    detect_blur: bool = False
    blur_variance_threshold: float = Field(default=0.0, ge=0.0)


class VideoProcessorSettings(BaseProcessorSettings):
    analysis_workers: int = Field(default=1, ge=1, le=64)
    analysis_queue_size: int = Field(default=40, ge=1, le=10000)
    keyframes: VideoKeyframeSettings = Field(
        default_factory=VideoKeyframeSettings
    )
    fallback_fps: float = Field(default=25.0, gt=0.0, le=240.0)
    probe_user_agent: str = (
        "MultimodalCrawler/1.0 "
        "(+https://github.com/multimodal-crawler/multimodal-crawler)"
    )
    tail_probe_chunk_size: int = 256000
    temp_suffix: str = ".tmp"
    frame_token_length: int = 24
    analysis_timeout_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)
    max_full_analysis_bytes: int = Field(default=25_000_000, ge=0)
    max_transcription_bytes: int = Field(default=16_000_000, ge=0)
    max_frame_analysis_bytes: int = Field(default=12_000_000, ge=0)
    generate_transcriptions: bool = False
    extract_audio_track: bool = True
    extract_metadata: bool = True
    normalize_video: bool = True
    run_transcription: bool = False
    run_ocr: bool = True
    min_bytes: int = Field(default=2048, ge=0)
    min_duration_seconds: float = Field(default=0.5, ge=0.0)
    max_duration_seconds: float = Field(default=3600.0, gt=0.0)
    min_fps: float = Field(default=1.0, ge=0.0)
    min_width: int = Field(default=96, ge=1)
    min_height: int = Field(default=96, ge=1)
    max_aspect_ratio: float = Field(default=10.0, gt=0.0)
    require_metadata_for_acceptance: bool = False

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> VideoProcessorSettings:
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError(
                "max_duration_seconds must be greater than or equal to "
                "min_duration_seconds"
            )
        return self


class DocumentProcessorSettings(BaseProcessorSettings):
    analysis_workers: int = Field(default=3, ge=1, le=64)
    analysis_queue_size: int = Field(default=120, ge=1, le=10000)
    analysis_timeout_seconds: float = Field(default=90.0, gt=0.0, le=3600.0)
    pdf_text_extraction: PdfTextExtractionSettings = Field(
        default_factory=PdfTextExtractionSettings
    )
    native_text: DocumentNativeTextSettings = Field(
        default_factory=DocumentNativeTextSettings
    )
    ocr_first_page: int = Field(default=1, ge=1)
    ocr_last_page: int = Field(default=2, ge=1)
    extract_text: bool = True
    extract_metadata: bool = True
    run_ocr: bool = True
    max_ocr_bytes: int = Field(default=25 * 1024 * 1024, ge=0)
    min_bytes: int = Field(default=64, ge=0)
    min_text_preview_chars: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def validate_ocr_page_bounds(self) -> DocumentProcessorSettings:
        if self.ocr_last_page < self.ocr_first_page:
            raise ValueError(
                "ocr_last_page must be greater than or equal to ocr_first_page"
            )
        return self


class PageDiscoveryRankingSettings(SettingsModel):
    enabled: bool = True
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    prioritize_internal_links: bool = True
    prioritize_fresh_content: bool = True
    prioritize_multimedia_content: bool = True
    default_kind_weight: float = Field(default=0.6, ge=0.0)
    discovered_link_bonus: float = Field(default=0.12)
    non_link_source_penalty: float = Field(default=0.0)
    embedded_media_asset_penalty: float = Field(default=0.0)
    embedded_asset_penalty: float = Field(default=10.0)
    page_bonus: float = Field(default=0.05)
    page_bonus_tokens: tuple[str, ...] = (
        "article",
        "blog",
        "gallery",
        "media",
        "multimedia",
        "news",
        "photo",
        "podcast",
        "story",
        "video",
    )
    media_page_bonus: float = Field(default=0.35)
    page_penalty: float = Field(default=0.25)
    page_penalty_tokens: tuple[str, ...] = (
        "account",
        "cart",
        "login",
        "privacy",
        "search",
        "signup",
        "terms",
    )
    asset_path_penalty: float = Field(default=0.15, ge=0.0)
    asset_penalty_tokens: tuple[str, ...] = (
        "avatar",
        "badge",
        "button",
        "favicon",
        "icon",
        "logo",
        "sprite",
        "thumbnail",
    )
    kind_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "page": 1.0,
            "image": 0.7,
            "video": 0.9,
            "audio": 0.8,
            "document": 0.75,
            "feed": 0.5,
        }
    )
    query_penalty: float = Field(default=0.02, ge=0.0)


class PageProcessorSettings(BaseProcessorSettings):
    text_preview: HtmlTextPreviewSettings = Field(
        default_factory=HtmlTextPreviewSettings
    )
    text_extraction: PageTextExtractionSettings = Field(
        default_factory=PageTextExtractionSettings
    )
    html_charset_scan_bytes: int = DEFAULT_TEXT_SAMPLE_BYTES
    extract_links: bool = True
    extract_assets: bool = True
    extract_metadata: bool = True
    follow_redirects: bool = True
    parse_structured_data: bool = True
    min_html_chars: int = Field(default=80, ge=0)
    max_html_chars: int = Field(default=2_000_000, ge=1)
    max_links_per_page: int = Field(default=160, ge=0)
    max_discovered_tasks_per_page: int = Field(default=20, ge=0)
    max_discovered_tasks_per_page_under_pressure: int = Field(default=2, ge=0)
    max_discovered_tasks_per_page_critical: int = Field(default=0, ge=0)
    max_pages_per_page: int = Field(default=6, ge=0)
    max_pages_per_page_under_pressure: int = Field(default=1, ge=0)
    max_pages_per_page_critical: int = Field(default=0, ge=0)
    max_embedded_assets_per_page: int = Field(default=8, ge=0)
    max_embedded_assets_per_page_under_pressure: int = Field(default=1, ge=0)
    max_embedded_assets_per_page_critical: int = Field(default=0, ge=0)
    max_media_assets_per_page: int = Field(default=24, ge=0)
    min_media_assets_per_crawl_batch: int = Field(default=3, ge=0)
    max_non_page_media_per_page: int = Field(default=8, ge=0)
    max_non_page_media_per_page_under_pressure: int = Field(default=1, ge=0)
    max_non_page_media_per_page_critical: int = Field(default=0, ge=0)
    multimodal_reserved_slots_by_kind: dict[str, int] = Field(
        default_factory=lambda: {
            "image": 3,
            "document": 3,
            "audio": 2,
            "video": 2,
            "feed": 1,
        }
    )
    multimodal_reserved_media_page_slots: int = Field(default=12, ge=0)
    discovery_queue_high_watermark: int = Field(default=96, ge=1)
    discovery_queue_critical_watermark: int = Field(default=160, ge=1)
    discovery_ranking: PageDiscoveryRankingSettings = Field(
        default_factory=PageDiscoveryRankingSettings
    )

    @model_validator(mode="after")
    def validate_pressure_thresholds(self) -> PageProcessorSettings:
        _validate_descending_limit(
            "max_discovered_tasks_per_page",
            self.max_discovered_tasks_per_page,
            self.max_discovered_tasks_per_page_under_pressure,
            self.max_discovered_tasks_per_page_critical,
        )
        _validate_descending_limit(
            "max_pages_per_page",
            self.max_pages_per_page,
            self.max_pages_per_page_under_pressure,
            self.max_pages_per_page_critical,
        )
        _validate_descending_limit(
            "max_embedded_assets_per_page",
            self.max_embedded_assets_per_page,
            self.max_embedded_assets_per_page_under_pressure,
            self.max_embedded_assets_per_page_critical,
        )
        _validate_descending_limit(
            "max_non_page_media_per_page",
            self.max_non_page_media_per_page,
            self.max_non_page_media_per_page_under_pressure,
            self.max_non_page_media_per_page_critical,
        )

        if (
            self.discovery_queue_critical_watermark
            < self.discovery_queue_high_watermark
        ):
            raise ValueError(
                "discovery_queue_critical_watermark must be greater than or "
                "equal to discovery_queue_high_watermark"
            )

        return self


class FeedProcessorSettings(BaseProcessorSettings):
    """Feed processor defaults.

    ``max_feed_entries`` caps parsed feed entries; it is a positive configured
    limit, while lower-level feed analysis uses ``None`` to mean unbounded.
    """

    parse_rss: bool = True
    parse_atom: bool = True
    deduplicate_entries: bool = True
    max_feed_entries: int = Field(
        default=DEFAULT_FEED_PROCESSOR_MAX_ENTRIES,
        ge=1,
    )
    schedule_entry_links: bool = True
    max_feed_items_discovered: int = Field(default=16, ge=0)
    max_feed_items_discovered_under_pressure: int = Field(default=2, ge=0)
    max_feed_items_discovered_critical: int = Field(default=0, ge=0)
    max_discovered_links_per_host: int = Field(default=6, ge=0)
    max_audio_links: int = Field(default=3, ge=0)
    discovery_queue_high_watermark: int = Field(default=96, ge=1)
    discovery_queue_critical_watermark: int = Field(default=160, ge=1)

    @model_validator(mode="after")
    def validate_pressure_thresholds(self) -> FeedProcessorSettings:
        _validate_descending_limit(
            "max_feed_items_discovered",
            self.max_feed_items_discovered,
            self.max_feed_items_discovered_under_pressure,
            self.max_feed_items_discovered_critical,
        )
        if (
            self.discovery_queue_critical_watermark
            < self.discovery_queue_high_watermark
        ):
            raise ValueError(
                "discovery_queue_critical_watermark must be greater than or "
                "equal to discovery_queue_high_watermark"
            )
        return self


class TaskProcessorSettings(BaseProcessorSettings):
    batch_size: int = Field(default=32, ge=1)
    enable_priority_scheduling: bool = True
    fail_fast: bool = False
    persist_task_results: bool = True
    drop_unknown_tasks: bool = True


class ProcessorSettings(SettingsModel):
    audio: AudioProcessorSettings = Field(
        default_factory=AudioProcessorSettings
    )
    image: ImageProcessorSettings = Field(
        default_factory=ImageProcessorSettings
    )
    video: VideoProcessorSettings = Field(
        default_factory=VideoProcessorSettings
    )
    document: DocumentProcessorSettings = Field(
        default_factory=DocumentProcessorSettings
    )
    page: PageProcessorSettings = Field(default_factory=PageProcessorSettings)
    feed: FeedProcessorSettings = Field(default_factory=FeedProcessorSettings)
    task: TaskProcessorSettings = Field(default_factory=TaskProcessorSettings)


def _validate_descending_limit(
    name: str,
    normal: int,
    under_pressure: int,
    critical: int,
) -> None:
    if under_pressure > normal:
        raise ValueError(
            f"{name}_under_pressure must be less than or equal to {name}"
        )

    if critical > under_pressure:
        raise ValueError(
            f"{name}_critical must be less than or equal to "
            f"{name}_under_pressure"
        )
