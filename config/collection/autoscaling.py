"""Public models and helpers for config.collection.autoscaling.

Exports: AutoscalerSettings.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class AutoscalerSettings(SettingsModel):
    """Settings for worker auto-scaling."""

    enabled: bool = True

    min_workers: int = Field(default=1, ge=0)
    max_workers: int = Field(default=8, ge=1)

    check_interval_seconds: float = Field(default=0.3, gt=0.0)
    scale_up_delay_seconds: float = Field(default=0.0, ge=0.0)
    scale_down_delay_seconds: float = Field(default=0.0, ge=0.0)
    scale_up_cooldown_seconds: float = Field(default=0.3, ge=0.0)
    scale_down_cooldown_seconds: float = Field(default=2.0, ge=0.0)

    max_scale_up_step: int = Field(default=2, ge=1)
    max_scale_down_step: int = Field(default=1, ge=1)

    slow_task_seconds_threshold: float = Field(default=3.0, gt=0.0)

    failure_burst_threshold: int = Field(default=2, ge=0)

    cancel_timeout_seconds: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def validate_settings(self) -> AutoscalerSettings:
        """Validate autoscaling bounds and thresholds."""

        if self.max_workers < self.min_workers:
            raise ValueError(
                "max_workers must be greater than or equal to min_workers"
            )

        if self.max_scale_down_step > self.max_workers:
            raise ValueError(
                "max_scale_down_step must be less than or equal to max_workers"
            )

        if self.max_scale_up_step > self.max_workers:
            raise ValueError(
                "max_scale_up_step must be less than or equal to max_workers"
            )

        return self
