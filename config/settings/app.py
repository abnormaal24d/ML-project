"""Application-level workflow and identity settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from config.base.settings_model import SettingsModel

RuntimeEnvironmentName = Literal["dev", "test", "prod"]


class AppSettings(SettingsModel):
    name: str = "multimodal-data-engine"
    environment: RuntimeEnvironmentName | None = None

    max_workflow_iterations: int = Field(default=50, ge=1)
    workflow_iteration_pause_seconds: float = Field(default=0.05, ge=0.0)
    workflow_blocking_task_limit: int = Field(default=4, ge=1)
    workflow_io_timeout_seconds: float = Field(default=300.0, gt=0.0)
    data_checker_timeout_seconds: float = Field(default=60.0, gt=0.0)

    manifest_replace_retry_attempts: int = Field(default=8, ge=1)
    manifest_replace_retry_delay_seconds: float = Field(default=0.1, ge=0.0)
    manifest_replace_retry_jitter_seconds: float = Field(default=0.1, ge=0.0)

    resolved_config_export_enabled: bool = False
    resolved_config_export_path: str = "runtime/config/resolved_settings.json"

    resource_shutdown_timeout_seconds: float = Field(default=10.0, gt=0.0)

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(
        cls, value: object
    ) -> RuntimeEnvironmentName | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized not in {
            "dev",
            "test",
            "prod",
        }:
            raise ValueError(
                "application.environment must be one of: dev, test, prod"
            )
        return normalized  # type: ignore[return-value]

    def resolved_environment(self) -> RuntimeEnvironmentName:
        """Return the validated runtime environment name."""

        if self.environment is None:
            raise ValueError("application.environment must be set")
        return self.environment
