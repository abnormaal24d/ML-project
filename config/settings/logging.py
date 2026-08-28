"""Canonical logging settings and event governance policy."""

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


def _default_event_rate_limit_governance() -> dict[str, EventRateLimitRulesSettings]:
    """Return the project-owned default event governance policy."""

    interval = 3.0
    return {
        "autoscaler_pressure_ratio_calculated": EventRateLimitRulesSettings(
            min_interval_sec=interval, field_names=("component_path",)
        ),
        "autoscaler_effective_max_workers_calculated": EventRateLimitRulesSettings(
            min_interval_sec=interval, field_names=("component_path",)
        ),
        "autoscaler_under_pressure_evaluated": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "decision"),
        ),
        "autoscaler_underutilized_evaluated": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "decision"),
        ),
        "autoscaler_decision_state_updated": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "pressure_state_reason"),
        ),
        "autoscaler_guards_evaluated": EventRateLimitRulesSettings(
            min_interval_sec=interval, field_names=("component_path",)
        ),
        "autoscaler_pause_guard_not_triggered": EventRateLimitRulesSettings(
            min_interval_sec=interval, field_names=("component_path",)
        ),
        "autoscaler_stop_guard_not_triggered": EventRateLimitRulesSettings(
            min_interval_sec=interval, field_names=("component_path",)
        ),
        "autoscaler_effective_cap_check": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "reason"),
        ),
        "autoscaler_scale_up_check": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "reason"),
        ),
        "autoscaler_scale_down_check": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "reason"),
        ),
        "autoscaler_tick": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "action"),
        ),
        "autoscaler_snapshot_committed": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "action"),
        ),
        "rate_limiter_slot_reserved": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "host"),
        ),
        "rate_limiter_sleep": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "host"),
        ),
        "request_user_agent_resolved": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=(
                "component_path",
                "host_profile",
                "selection_strategy",
            ),
        ),
        "request_accept_encoding_built": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=("component_path", "accept_encoding"),
        ),
        "request_headers_built": EventRateLimitRulesSettings(
            min_interval_sec=interval,
            field_names=(
                "component_path",
                "host",
                "profile_name",
                "host_profile",
            ),
        ),
    }


class LoggingSettings(SettingsModel):
    """Logger configuration and canonical structured-event policy."""

    root_name: str = "project"
    base_log_fields: dict[str, Any] = Field(default_factory=dict)

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
    event_rate_limit_governance: dict[str, EventRateLimitRulesSettings] = Field(
        default_factory=_default_event_rate_limit_governance
    )
    component_levels: dict[str, LogLevel] = Field(default_factory=dict)

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
