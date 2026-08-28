"""Logging settings: full parity with the legacy logging model.

Full port of ``config/logging/logging_settings.py`` (root name, base
fields, console/file levels and formats, compact context, file rotation,
propagation, rate limiting, per-component levels and the console-suppressed
event registry). The profile files only override the runtime knobs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["plain", "json"]

_DEFAULT_LOG_RATE_LIMIT_MAX_ENTRIES = 10_000


class EventRateLimitRulesSettings(SettingsModel):
    """Rate-limit rules for one structured log event."""

    min_interval_sec: float = Field(default=1.0, ge=0.0)
    field_names: tuple[str, ...] = ()


class LoggingSettings(SettingsModel):
    """Logger configuration.

    Rate-limit intervals are measured in seconds. The max-entry default caps
    the in-memory event de-duplication state.
    """

    root_name: str = "project"

    base_log_fields: dict[str, Any] = {}

    level: LogLevel = "INFO"
    console_level: LogLevel | None = None
    file_level: LogLevel | None = None
    format: LogFormat = "plain"
    console_format: LogFormat = "plain"
    file_format: LogFormat = "json"
    console_compact_context: bool = True

    enable_console: bool = True
    enable_file: bool = False

    file_path: str | None = None

    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    backup_count: int = Field(default=3, ge=0)

    propagate: bool = False

    datefmt: str = "%Y-%m-%d %H:%M:%S"
    rate_limit_enabled: bool = True
    rate_limit_min_interval_sec: float = Field(default=0.0, ge=0.0)
    rate_limit_max_entries: int = Field(
        default=_DEFAULT_LOG_RATE_LIMIT_MAX_ENTRIES,
        ge=1,
    )
    event_rate_limit_governance: dict[str, EventRateLimitRulesSettings] = {}
    component_levels: dict[str, LogLevel] = {}

    @field_validator("level", "console_level", "file_level", mode="before")
    @classmethod
    def _normalize_level_choice(cls, value: object) -> object:
        if value is None:
            return None
        return str(value).strip().upper() if isinstance(value, str) else value

    @field_validator("format", "console_format", "file_format", mode="before")
    @classmethod
    def _normalize_format_choice(cls, value: object) -> object:
        if value is None:
            return None
        return str(value).strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_file_logging(self) -> LoggingSettings:
        if self.enable_file and not str(self.file_path or "").strip():
            raise ValueError("enable_file=True requires a non-empty file_path")
        return self

    console_suppressed_events: tuple[str, ...] = (
        "artifact_inventory_resolved",
        "audio_candidate_analyzed",
        "crawler_metrics",
        "data_checker_report_assembled",
        "dataset_record_written",
        "multimodal_dataloader_built",
        "multimodal_dataset_loaded",
        "workflow_augmentation_manifest_written",
        "workflow_crawl_state_stale_reference_ignored",
        "workflow_crawl_status_written",
        "workflow_crawl_state_written",
        "workflow_crawl_manifest_written",
        "workflow_preprocessing_manifest_written",
        "workflow_state_checked",
        "workflow_training_artifacts_written",
        "source_governance_registry_loaded",
        "video_candidate_analyzed",
        "worker_runtime_cancelled",
        "page_asset_discovery_details",
    )
