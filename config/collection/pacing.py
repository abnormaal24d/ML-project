"""Central pacing settings for crawl etiquette."""

from __future__ import annotations

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class PacingSettings(SettingsModel):
    """Single source for crawl-delay, RPS, jitter, Retry-After."""

    default_rps: float = Field(
        default=0.33,
        gt=0.0,
        allow_inf_nan=False,
    )
    min_rps: float = Field(default=0.1, gt=0.0, allow_inf_nan=False)
    max_rps: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    ramp_up_factor: float = Field(
        default=1.05,
        ge=1.0,
        allow_inf_nan=False,
    )
    backoff_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    error_cooldown_seconds: float = Field(
        default=30.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    burst: int = Field(default=1, ge=1)
    jitter_min_seconds: float = Field(
        default=0.1,
        ge=0.0,
        allow_inf_nan=False,
    )
    jitter_max_seconds: float = Field(
        default=0.75,
        ge=0.0,
        allow_inf_nan=False,
    )
    max_crawl_delay_seconds: float = Field(
        default=60.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    honor_retry_after: bool = True
    max_retry_after_seconds: float = Field(
        default=3600.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_pacing(self) -> PacingSettings:
        if not self.min_rps <= self.default_rps <= self.max_rps:
            raise ValueError("default_rps must be between min_rps and max_rps")
        if self.jitter_max_seconds < self.jitter_min_seconds:
            raise ValueError(
                "jitter_max_seconds must be >= jitter_min_seconds"
            )
        return self
