"""Crawler runtime settings: full parity with the legacy model.

Full port of ``config/crawler/runtime_settings.py`` (checkpoint
persistence, dead-letter bookkeeping and runtime control flags).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from config.base.settings_model import SettingsModel
from config.path_resolution.project_paths import validate_safe_relative_path

type DeadLetterCanonicalStatus = Literal[
    "retry_exhausted",
    "failed",
    "cancelled",
]


def _dead_letter_canonical_status(
    value: str,
) -> DeadLetterCanonicalStatus:
    if value == "retry_exhausted":
        return "retry_exhausted"
    if value == "failed":
        return "failed"
    if value == "cancelled":
        return "cancelled"
    raise ValueError(f"unsupported dead-letter status: {value}")


class CrawlStateStoreSettings(SettingsModel):
    """Settings for checkpoint persistence and dead-letter bookkeeping."""

    enabled: bool = True
    state_subdirectory: str = Field(default="checkpoints", min_length=1)
    run_scoped_state: bool = True
    checkpoint_filename: str = Field(
        default="crawler_runtime_checkpoint.json",
        min_length=1,
    )
    checkpoint_interval_seconds: float = Field(default=15.0, gt=0.0)
    checkpoint_max_queued_tasks: int = Field(default=5_000, ge=1)
    resume_from_checkpoint: bool = True
    resume_requires_seed_match: bool = True
    include_seen_urls_in_checkpoint: bool = True
    pretty_checkpoint_json: bool = True
    dead_letter_enabled: bool = True
    dead_letter_filename: str = Field(
        default="dead_letters.jsonl",
        min_length=1,
    )
    dead_letter_statuses: tuple[
        Literal["retry_exhausted", "failed", "cancelled"],
        ...,
    ] = (
        "retry_exhausted",
        "failed",
        "cancelled",
    )
    requeue_dead_letters_on_start: bool = False
    max_dead_letters_to_requeue: int = Field(default=200, ge=0)
    clear_dead_letters_on_requeue: bool = False

    @field_validator("state_subdirectory")
    @classmethod
    def validate_state_subdirectory(cls, value: str) -> str:
        return validate_safe_relative_path(
            value,
            field_name="state_subdirectory",
        )

    @field_validator("checkpoint_filename", "dead_letter_filename")
    @classmethod
    def validate_state_filename(cls, value: str) -> str:
        return _safe_filename(value)

    @field_validator("dead_letter_statuses")
    @classmethod
    def validate_dead_letter_statuses(
        cls,
        value: tuple[DeadLetterCanonicalStatus, ...],
    ) -> tuple[DeadLetterCanonicalStatus, ...]:
        """Deduplicate the canonical terminal dead-letter allowlist."""

        normalized: list[DeadLetterCanonicalStatus] = []
        for status in value:
            canonical = _dead_letter_canonical_status(status)
            if canonical not in normalized:
                normalized.append(canonical)
        return tuple(normalized)


class CrawlerSettings(SettingsModel):
    """Settings controlling crawler lifecycle and runtime side effects."""

    enabled: bool = True
    seed_source_type: str = Field(default="seed", min_length=1)
    control_directory: str = Field(default="runtime", min_length=1)
    pause_flag_filename: str = Field(default="pause.flag", min_length=1)
    stop_flag_filename: str = Field(default="stop.flag", min_length=1)
    progress_log_interval_seconds: float = Field(default=15.0, gt=0.0)
    shutdown_poll_interval_seconds: float = Field(default=0.05, gt=0.0)
    max_idle_delay_wait_seconds: float = Field(default=0.5, ge=0.0)
    drain_delayed_backlog_before_finish: bool = True
    drain_stall_timeout_seconds: float = Field(default=180.0, gt=0.0)
    drain_watch_interval_seconds: float = Field(default=5.0, gt=0.0)
    state: CrawlStateStoreSettings = Field(
        default_factory=CrawlStateStoreSettings
    )

    @field_validator("control_directory")
    @classmethod
    def validate_control_directory(cls, value: str) -> str:
        return validate_safe_relative_path(
            value,
            field_name="control_directory",
        )

    @field_validator("pause_flag_filename", "stop_flag_filename")
    @classmethod
    def validate_control_filename(cls, value: str) -> str:
        return _safe_filename(value)


def _safe_filename(value: str) -> str:
    """Return a portable basename suitable for a runtime control file."""

    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or "\x00" in text
        or any(character in text for character in '<>:"/\\|?*')
        or text.endswith((".", " "))
    ):
        raise ValueError("runtime filenames must be safe basenames")
    return text
