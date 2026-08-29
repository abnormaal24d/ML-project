"""Public models and helpers for config.collection.fetching.

Exports: FetcherSettings, ResponseBodyReaderSettings,
    UrlSchemeValidatorSettings.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.collection.value_normalizers import (
    as_candidates,
)


class FetcherSettings(SettingsModel):
    """Fetcher transport settings for headers, preflight, and media limits."""

    large_media_timeout_threshold_bytes: int = Field(default=8_000_000, ge=1)

    media_metadata_probe_enabled: bool = True
    audio_metadata_probe_bytes: int = Field(default=1_048_576, ge=1024)
    video_metadata_probe_bytes: int = Field(default=2_097_152, ge=1024)
    oversized_video_metadata_mode: str = "head_only"

    accept_compressed: bool = True
    accept_language_header: str = "en-US,en;q=0.9"

    raise_for_non_success_status: bool = True

    record_rate_limiter_feedback: bool = True
    record_fetch_metrics: bool = True
    host_profile_forbidden_host_threshold: int = Field(default=1, ge=1)
    host_profile_forbidden_host_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
    )

    head_preflight_enabled: bool = True
    head_preflight_for_all_hosts: bool = True
    head_preflight_host_allowlist: tuple[str, ...] = ()
    head_preflight_task_kinds: tuple[str, ...] = (
        "feed",
        "image",
        "audio",
        "video",
        "document",
    )
    head_preflight_adaptive_skip_enabled: bool = False
    head_preflight_useless_host_threshold: int = Field(default=3, ge=1)
    head_preflight_useless_host_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
    )
    drop_if_head_disallowed: bool = True
    head_preflight_counts_toward_rate_feedback: bool = False

    @field_validator("oversized_video_metadata_mode", mode="before")
    @classmethod
    def normalize_oversized_video_metadata_mode(cls, value: object) -> str:
        """Normalize oversized-video metadata handling mode."""

        normalized = (
            str(value or "head_only").strip().lower().replace("-", "_")
        )
        if normalized not in {"head_only", "range"}:
            raise ValueError(
                "oversized_video_metadata_mode must be one of head_only, "
                "or range"
            )
        return normalized

    @field_validator("head_preflight_host_allowlist", mode="before")
    @classmethod
    def normalize_head_preflight_host_allowlist(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        """Normalize HEAD preflight host allowlist entries."""

        normalized: list[str] = []
        seen: set[str] = set()

        for candidate in as_candidates(value):
            host = str(candidate).strip().lower()
            if not host or host in seen:
                continue

            seen.add(host)
            normalized.append(host)

        return tuple(normalized)

    @field_validator("head_preflight_task_kinds", mode="before")
    @classmethod
    def normalize_head_preflight_task_kinds(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        """Normalize task kinds eligible for HEAD preflight."""

        normalized: list[str] = []
        seen: set[str] = set()

        for candidate in as_candidates(value):
            kind = str(candidate).strip().lower()
            if not kind or kind in seen:
                continue

            seen.add(kind)
            normalized.append(kind)

        return tuple(normalized)


class ResponseBodyReaderSettings(SettingsModel):
    """Settings controlling body chunk sizes, temp files, and stalls."""

    binary_content_type_prefixes: tuple[str, ...] = (
        "image/",
        "audio/",
        "video/",
        "application/pdf",
        "application/zip",
    )
    connection_closed_error_markers: tuple[str, ...] = (
        "connection closed",
        "connection reset",
        "cannot write to closing transport",
        "server disconnected",
        "payload is not completed",
    )
    temporary_directory: str = "runtime/tmp/fetch_payloads"
    chunk_size: int = Field(default=65_536, ge=1)
    binary_chunk_size: int = Field(default=524_288, ge=1)
    large_binary_chunk_size: int = Field(default=1_048_576, ge=1)
    large_body_threshold_bytes: int = Field(default=8_000_000, ge=1)
    sniff_byte_count: int = Field(default=4096, ge=1)
    preserve_partial_files: bool = False
    max_stalled_reads: int = Field(default=2, ge=1)
    max_decompression_ratio: float = Field(default=100.0, ge=1.0, le=1_000.0)
    max_in_flight_bytes: int = Field(default=16_777_216, ge=1)
    download_bytes_per_second: int = Field(default=33_554_432, ge=1)

    @field_validator(
        "binary_content_type_prefixes",
        "connection_closed_error_markers",
        mode="before",
    )
    @classmethod
    def normalize_string_tuple(cls, value: object) -> tuple[str, ...]:
        """Normalize tuple-like string settings for body reading."""

        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in as_candidates(value):
            item = str(candidate).strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_chunk_thresholds(self) -> ResponseBodyReaderSettings:
        """Validate chunk-size ordering and large-body threshold."""

        if self.binary_chunk_size < self.chunk_size:
            raise ValueError(
                "binary_chunk_size must be greater than or equal to chunk_size"
            )

        if self.large_binary_chunk_size < self.binary_chunk_size:
            raise ValueError(
                "large_binary_chunk_size must be greater than or equal to "
                "binary_chunk_size"
            )

        if self.large_body_threshold_bytes < self.large_binary_chunk_size:
            raise ValueError(
                "large_body_threshold_bytes must be greater than or equal to "
                "large_binary_chunk_size"
            )

        if self.max_in_flight_bytes < self.large_binary_chunk_size:
            raise ValueError(
                "max_in_flight_bytes must be greater than or equal to "
                "large_binary_chunk_size"
            )

        return self


class UrlSchemeValidatorSettings(SettingsModel):
    """Settings for URL scheme admission during fetch planning."""

    allowed_schemes: tuple[str, ...] = ("http", "https")

    @field_validator("allowed_schemes", mode="before")
    @classmethod
    def normalize_allowed_schemes(cls, value: object) -> tuple[str, ...]:
        """Normalize allowed URL schemes to lowercase values."""

        normalized: list[str] = []
        seen: set[str] = set()

        for candidate in as_candidates(value):
            scheme = str(candidate).strip().lower()
            if not scheme or scheme in seen:
                continue

            seen.add(scheme)
            normalized.append(scheme)

        return tuple(normalized)

    @model_validator(mode="after")
    def validate_allowed_schemes(self) -> UrlSchemeValidatorSettings:
        """Validate that at least one URL scheme is allowed."""

        if not self.allowed_schemes:
            raise ValueError("allowed_schemes must not be empty")

        return self
