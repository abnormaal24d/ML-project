"""Strict processing-activity and DPIA configuration loading.

Configuration artefacts belong to the read-only config root. Every load
failure is classified into a typed ``ProcessingActivityConfigError`` with
safe structured context (code, setting, basename) so operators never see
raw local paths in logs while still knowing exactly what is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    model_validator,
)

PROCESSING_ACTIVITIES_CONFIG_RELATIVE_PATH: Final[str] = (
    "governance/processing_activities.json"
)

PROCESSING_ACTIVITIES_SETTING: Final[str] = (
    "governance.processing_activities_file"
)

SUPPORTED_REGISTRY_SCHEMA_VERSION: Final[str] = "1.0.0"

TRAINING_DATASET_ACTIVITY_ID: Final[str] = "training_dataset_build"


class ProcessingActivityConfigError(RuntimeError):
    """A processing-activity configuration artifact is unusable."""

    def __init__(
        self,
        *,
        code: str,
        basename: str,
        setting: str = PROCESSING_ACTIVITIES_SETTING,
    ) -> None:
        self.code = code
        self.component = "processing_activity_registry"
        self.setting = setting
        self.basename = basename

        super().__init__(
            f"{code}: "
            f"component={self.component}, "
            f"setting={self.setting}, "
            f"file={self.basename}"
        )


class ProcessingActivity(BaseModel):
    """One governed processing activity and its DPIA state."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    activity_id: str
    purpose: str

    personal_data_allowed: bool

    dpia_status: Literal[
        "approved",
        "rejected",
        "required",
        "expired",
    ]
    dpia_review_expires_at: str

    retention_days: int
    rules_version: str
    enabled: bool

    @model_validator(mode="after")
    def validate_activity(
        self,
    ) -> ProcessingActivity:
        """Validate the immutable processing-activity contract."""

        if not self.activity_id.strip():
            raise ValueError("activity_id is required")

        if self.activity_id != self.activity_id.strip():
            raise ValueError(
                "activity_id must not contain surrounding whitespace"
            )

        if not self.purpose.strip():
            raise ValueError("purpose is required")

        if not self.rules_version.strip():
            raise ValueError("rules_version is required")

        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")

        _parse_aware_datetime(
            self.dpia_review_expires_at,
            field_name="dpia_review_expires_at",
        )

        return self

    def permits_training(
        self,
        *,
        now: datetime,
    ) -> bool:
        """Return whether this activity currently permits training."""

        if now.tzinfo is None:
            raise ValueError(
                "processing activity reference time requires timezone"
            )

        reference = now.astimezone(timezone.utc)

        expires = _parse_aware_datetime(
            self.dpia_review_expires_at,
            field_name="dpia_review_expires_at",
        ).astimezone(timezone.utc)

        return (
            self.enabled
            and self.dpia_status == "approved"
            and expires > reference
        )


class ProcessingActivityRegistry(BaseModel):
    """Validated registry of governed processing activities."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: str
    activities: tuple[
        ProcessingActivity,
        ...,
    ]

    @model_validator(mode="after")
    def validate_registry(
        self,
    ) -> ProcessingActivityRegistry:
        """Validate registry-level invariants."""

        if not self.schema_version.strip():
            raise ValueError("schema_version is required")

        activity_ids: set[str] = set()

        for activity in self.activities:
            activity_id = activity.activity_id

            if activity_id in activity_ids:
                raise ValueError(
                    f"duplicate processing activity_id: {activity_id}"
                )

            activity_ids.add(activity_id)

        return self

    def require(
        self,
        *,
        activity_id: str,
    ) -> ProcessingActivity:
        """Return one required activity or fail closed."""

        if not activity_id:
            raise KeyError("processing activity_id is required")

        for activity in self.activities:
            if activity.activity_id == activity_id:
                return activity

        raise KeyError(f"unknown processing activity: {activity_id}")


def load_processing_activities(
    path: Path,
    *,
    setting: str | None = None,
) -> ProcessingActivityRegistry:
    """Load and strictly validate one processing-activity registry.

    The file read is the authoritative filesystem operation. Failures are
    converted to ``ProcessingActivityConfigError`` using only path-safe
    context: error code, logical setting and basename.
    """

    basename = path.name

    effective_setting = setting or PROCESSING_ACTIVITIES_SETTING

    raw_text = _read_registry_text(
        path,
        basename=basename,
        setting=effective_setting,
    )

    payload = _decode_registry_json(
        raw_text,
        basename=basename,
        setting=effective_setting,
    )

    registry = _validate_registry_payload(
        payload,
        basename=basename,
        setting=effective_setting,
    )

    if registry.schema_version != SUPPORTED_REGISTRY_SCHEMA_VERSION:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_unsupported_version"),
            setting=effective_setting,
            basename=basename,
        )

    return registry


def _read_registry_text(
    path: Path,
    *,
    basename: str,
    setting: str,
) -> str:
    """Read one registry file and classify filesystem failures."""

    try:
        if path.is_dir():
            raise IsADirectoryError(path)
        return path.read_text(encoding="utf-8")

    except FileNotFoundError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_missing"),
            setting=setting,
            basename=basename,
        ) from exc

    except IsADirectoryError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_not_a_file"),
            setting=setting,
            basename=basename,
        ) from exc

    except PermissionError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_not_readable"),
            setting=setting,
            basename=basename,
        ) from exc

    except UnicodeDecodeError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_invalid_encoding"),
            setting=setting,
            basename=basename,
        ) from exc

    except OSError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_io_error"),
            setting=setting,
            basename=basename,
        ) from exc


def _decode_registry_json(
    raw_text: str,
    *,
    basename: str,
    setting: str,
) -> object:
    """Decode registry JSON while preserving a typed config error."""

    try:
        return json.loads(raw_text)

    except json.JSONDecodeError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_invalid_json"),
            setting=setting,
            basename=basename,
        ) from exc


def _validate_registry_payload(
    payload: object,
    *,
    basename: str,
    setting: str,
) -> ProcessingActivityRegistry:
    """Validate decoded JSON against the registry schema."""

    try:
        return ProcessingActivityRegistry.model_validate(payload)

    except ValidationError as exc:
        raise ProcessingActivityConfigError(
            code=("processing_activity_config_invalid_schema"),
            setting=setting,
            basename=basename,
        ) from exc


def _parse_aware_datetime(
    value: str,
    *,
    field_name: str,
) -> datetime:
    """Parse one required timezone-aware ISO 8601 datetime."""

    text = value.strip()

    if not text:
        raise ValueError(f"{field_name} is required")

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid ISO 8601 timestamp"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone")

    return parsed
