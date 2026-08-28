"""Public models for config.collection.discovery.

Exports: ExtensionDetectorSettings, DiscoveryFeedbackSettings,
    SchedulingSettings, HtmlParserSettings, UrlPrioritySettings,
    WorkerPoolSettings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import normalize_string_tuple
from config.environment.default_values import (
    DEFAULT_INFLIGHT_HOST_WAIT_SECONDS,
)


class ExtensionDetectorSettings(SettingsModel):
    """Heuristics for URL extension detection."""

    ignored_unknown_suffixes: tuple[str, ...] = (
        ".html",
        ".htm",
        ".php",
        ".asp",
        ".aspx",
        ".jsp",
        ".jspx",
        ".cfm",
        ".cgi",
        ".do",
        ".action",
        ".shtml",
        ".dhtml",
        ".xhtml",
        ".xht",
        ".xhtm",
    )
    domain_like_tlds: tuple[str, ...] = (
        "be",
        "co",
        "com",
        "de",
        "edu",
        "eu",
        "fr",
        "gov",
        "int",
        "io",
        "mil",
        "net",
        "org",
        "tv",
        "uk",
    )

    @field_validator("ignored_unknown_suffixes", mode="before")
    @classmethod
    def normalize_ignored_unknown_suffixes(
        cls, value: object
    ) -> tuple[str, ...]:
        """Normalize ignored suffixes to lowercase dotted values."""

        return normalize_string_tuple(
            value, lowercase=True, require_prefix="."
        )

    @field_validator("domain_like_tlds", mode="before")
    @classmethod
    def normalize_domain_like_tlds(cls, value: object) -> tuple[str, ...]:
        """Normalize pseudo-TLD values without leading dots."""

        normalized = normalize_string_tuple(value, lowercase=True)
        return tuple(item.lstrip(".") for item in normalized)


class DiscoveryFeedbackSettings(SettingsModel):
    """Settings for converting discovered child multimodal into info-gain."""

    ewma_alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    default_info_gain: float = Field(default=0.5, ge=0.0, le=1.0)
    default_host_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    seed_host_quality: float = Field(default=0.65, ge=0.0, le=1.0)

    success_weight_novelty: float = Field(default=0.4, ge=0.0, le=1.0)
    success_weight_acceptance: float = Field(default=0.35, ge=0.0, le=1.0)
    success_weight_retained_after_truncation: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
    )
    success_weight_low_rejection: float = Field(default=0.1, ge=0.0, le=1.0)

    filtered_rejection_penalty_scale: float = Field(
        default=0.5, ge=0.0, le=1.0
    )
    quality_hint_blend_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    cancelled_info_gain: float = Field(default=0.3, ge=0.0, le=1.0)
    dropped_info_gain: float = Field(default=0.2, ge=0.0, le=1.0)
    failed_info_gain: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_success_weights(self) -> DiscoveryFeedbackSettings:
        """Validate feedback success weights for positive contribution."""

        total_success_weight = (
            self.success_weight_novelty
            + self.success_weight_acceptance
            + self.success_weight_retained_after_truncation
            + self.success_weight_low_rejection
        )

        if total_success_weight <= 0.0:
            raise ValueError(
                "at least one success weight must be greater than zero"
            )

        return self


class SchedulingSettings(SettingsModel):
    """Settings for crawl scheduling, fairness and queue pressure."""

    max_depth: int = Field(default=2, ge=0)
    max_document_candidate_depth: int | None = Field(default=None, ge=0)
    max_audio_candidate_depth: int | None = Field(default=None, ge=0)
    max_video_candidate_depth: int | None = Field(default=None, ge=0)
    max_seen: int = Field(default=10_000, ge=1)
    seen_url_ttl_seconds: float | None = Field(default=None, ge=0.0)

    max_pending_per_host: int = Field(default=8, ge=1)
    max_pending_per_host_under_pressure: int | None = Field(default=2, ge=1)
    max_pending_per_host_critical: int | None = Field(default=1, ge=1)
    max_pending_per_host_by_kind: dict[str, int] = Field(
        default_factory=lambda: {
            "page": 12,
            "image": 16,
            "audio": 8,
            "document": 8,
            "video": 4,
        },
    )
    max_pending_per_host_by_kind_under_pressure: dict[str, int] = Field(
        default_factory=dict,
    )
    max_pending_per_host_by_kind_critical: dict[str, int] = Field(
        default_factory=dict,
    )
    max_media_bytes_per_host: int = Field(default=512_000_000, ge=1)
    max_media_bytes_per_host_by_kind: dict[str, int] = Field(
        default_factory=lambda: {
            "image": 256_000_000,
            "audio": 128_000_000,
            "document": 128_000_000,
            "video": 96_000_000,
        }
    )
    max_inflight_per_host: int = Field(default=1, ge=1)
    inflight_host_wait_seconds: float = Field(
        default=DEFAULT_INFLIGHT_HOST_WAIT_SECONDS,
        gt=0.0,
    )

    max_total_attempts: int = Field(default=4, ge=0)
    max_deferrals: int = Field(default=3, ge=0)
    max_timeouts: int = Field(default=1, ge=0)

    feed_max_total_attempts: int = Field(default=3, ge=0)
    feed_max_deferrals: int = Field(default=2, ge=0)
    feed_max_timeouts: int = Field(default=1, ge=0)
    max_total_attempts_by_kind: dict[str, int] = Field(default_factory=dict)
    max_deferrals_by_kind: dict[str, int] = Field(default_factory=dict)
    max_timeouts_by_kind: dict[str, int] = Field(default_factory=dict)

    default_retry_wait_seconds: float = Field(default=5.0, ge=0.0)
    timeout_retry_wait_seconds: float = Field(default=5.0, ge=0.0)
    dead_letter_on_drain: bool = True

    allow_scheduler_feedback: bool = True
    record_host_quality: bool = True
    host_feedback_max_hosts: int | None = Field(default=2048, ge=1)

    dynamic_crawl_budget_enabled: bool = True
    crawl_budget_window: int = Field(default=200, ge=1)
    crawl_budget_low_info_threshold: float = Field(
        default=0.12, ge=0.0, le=1.0
    )
    max_feeds_per_host: int = Field(default=2, ge=0)
    max_feed_depth: int = Field(default=1, ge=0)

    discovery_feedback: DiscoveryFeedbackSettings = Field(
        default_factory=DiscoveryFeedbackSettings,
    )

    robots_host_rules_advice_ttl_seconds: float | None = Field(
        default=1800.0, gt=0.0
    )
    robots_host_rules_advice_max_hosts: int = Field(default=2048, ge=0)
    hostility_reject_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    queue_high_watermark: int = Field(default=96, ge=1)
    queue_critical_watermark: int = Field(default=160, ge=1)

    abandon_suppressed_host_threshold_seconds: float | None = Field(
        default=1200.0,
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_scheduler_thresholds(self) -> SchedulingSettings:
        """Validate scheduler pressure and queue threshold ordering."""

        if self.queue_critical_watermark < self.queue_high_watermark:
            raise ValueError(
                "queue_critical_watermark must be greater than or equal to "
                "queue_high_watermark"
            )

        if (
            self.max_pending_per_host_under_pressure is not None
            and self.max_pending_per_host_under_pressure
            > self.max_pending_per_host
        ):
            raise ValueError(
                "max_pending_per_host_under_pressure must be less than or "
                "equal to max_pending_per_host"
            )

        if (
            self.max_pending_per_host_critical is not None
            and self.max_pending_per_host_under_pressure is not None
            and self.max_pending_per_host_critical
            > self.max_pending_per_host_under_pressure
        ):
            raise ValueError(
                "max_pending_per_host_critical must be less than or equal to "
                "max_pending_per_host_under_pressure"
            )

        allowed_kinds = {"page", "image", "audio", "document", "video"}
        for field_name in (
            "max_pending_per_host_by_kind",
            "max_pending_per_host_by_kind_under_pressure",
            "max_pending_per_host_by_kind_critical",
        ):
            for kind, limit in getattr(self, field_name).items():
                if kind not in allowed_kinds:
                    raise ValueError(
                        f"{field_name} only supports: "
                        "page, image, audio, document, video"
                    )
                if limit < 1:
                    raise ValueError(f"{field_name} values must be >= 1")

        media_kinds = {"image", "audio", "document", "video"}
        for kind, limit in self.max_media_bytes_per_host_by_kind.items():
            if kind not in media_kinds:
                raise ValueError(
                    "max_media_bytes_per_host_by_kind only supports: "
                    "image, audio, document, video"
                )
            if limit < 0:
                raise ValueError(
                    "max_media_bytes_per_host_by_kind values must be >= 0"
                )
            if limit > self.max_media_bytes_per_host:
                raise ValueError(
                    "max_media_bytes_per_host_by_kind values must not exceed "
                    "max_media_bytes_per_host"
                )

        return self

    def effective_max_document_candidate_depth(self) -> int:
        if self.max_document_candidate_depth is not None:
            return int(self.max_document_candidate_depth)
        return int(self.max_depth) + 1

    def effective_max_audio_candidate_depth(self) -> int:
        if self.max_audio_candidate_depth is not None:
            return int(self.max_audio_candidate_depth)
        return int(self.max_depth) + 1

    def effective_max_video_candidate_depth(self) -> int:
        if self.max_video_candidate_depth is not None:
            return int(self.max_video_candidate_depth)
        return int(self.max_depth) + 1


class HtmlParserSettings(SettingsModel):
    """Settings for HTML parsing."""

    parser: str = Field(default="html.parser", min_length=1)
    parser_candidates: tuple[str, ...] = ("lxml", "html5lib", "html.parser")
    recover: bool = True
    prefer_beautiful_soup: bool = True
    allow_stdlib_fallback: bool = True

    @field_validator("parser_candidates", mode="before")
    @classmethod
    def normalize_parser_candidates(cls, value: object) -> tuple[str, ...]:
        """Normalize parser candidate names to lowercase strings."""

        return normalize_string_tuple(value, lowercase=True)

    @model_validator(mode="after")
    def validate_parser_settings(self) -> HtmlParserSettings:
        """Validate parser selection and stdlib fallback availability."""

        if not self.parser_candidates:
            raise ValueError("parser_candidates must not be empty")

        if self.parser not in self.parser_candidates:
            raise ValueError("parser must be included in parser_candidates")

        if (
            self.allow_stdlib_fallback
            and "html.parser" not in self.parser_candidates
        ):
            raise ValueError(
                "html.parser must be included in parser_candidates when "
                "allow_stdlib_fallback is true"
            )

        return self


class UrlPrioritySettings(SettingsModel):
    """Settings for crawl-task priority resolution."""

    seed_priority: int = Field(default=0)
    discovered_priority: int = Field(default=10)
    feed_priority: Literal["high", "medium", "low"] = "medium"
    depth_penalty: int = Field(default=1, ge=0)
    min_priority: int = Field(default=-100)
    max_priority: int = Field(default=10_000)
    host_quality_boost_scale: float = Field(default=12.0, ge=0.0)
    host_noise_penalty_scale: float = Field(default=12.0, ge=0.0)
    info_gain_boost_scale: float = Field(default=6.0, ge=0.0)
    external_host_exploration_penalty: int = Field(default=8, ge=0)
    low_info_gain_penalty_enabled: bool = True
    low_info_gain_penalty_threshold: float = Field(
        default=0.12, ge=0.0, le=1.0
    )
    low_info_gain_penalty: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def validate_priority_bounds(self) -> UrlPrioritySettings:
        """Validate crawl priority bounds and default priorities."""

        if self.max_priority < self.min_priority:
            raise ValueError(
                "max_priority must be greater than or equal to min_priority"
            )

        if not (
            self.min_priority <= self.seed_priority
            and self.seed_priority <= self.max_priority
        ):
            raise ValueError(
                "seed_priority must be between min_priority and max_priority"
            )

        if not (
            self.min_priority <= self.discovered_priority
            and self.discovered_priority <= self.max_priority
        ):
            raise ValueError(
                "discovered_priority must be between min_priority and "
                "max_priority"
            )

        return self


class WorkerPoolSettings(SettingsModel):
    """Settings for crawl worker lifecycle behavior."""

    queue_poll_timeout_seconds: float = Field(default=0.25, gt=0.0)
    empty_backoff_seconds: float = Field(default=0.05, ge=0.0)
    fail_fast_on_processing_error: bool = False
    processing_timeout_seconds: float = Field(default=300.0, gt=0.0)
    completion_timeout_seconds: float = Field(default=30.0, gt=0.0)
    callback_timeout_seconds: float = Field(default=30.0, gt=0.0)
    stop_timeout_seconds: float = Field(default=30.0, gt=0.0)
    finalizer_drain_timeout_seconds: float = Field(default=30.0, gt=0.0)
