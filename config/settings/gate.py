"""Crawl output readiness gate: minimums before a run is acceptably ready."""

from __future__ import annotations

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class GateMinimumRecordsSettings(SettingsModel):
    page: int = Field(default=0, ge=0)
    document: int = Field(default=0, ge=0)
    image: int = Field(default=0, ge=0)
    audio: int = Field(default=0, ge=0)
    video: int = Field(default=0, ge=0)


class CrawlOutputGateSettings(SettingsModel):
    enabled: bool = True
    min_quality_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_raw_objects_total: int = Field(default=80, ge=0)
    min_successful_requests_total: int = Field(default=60, ge=0)
    minimum_records: GateMinimumRecordsSettings = GateMinimumRecordsSettings()

    @model_validator(mode="after")
    def validate_settings(self) -> CrawlOutputGateSettings:
        if self.min_successful_requests_total > self.min_raw_objects_total:
            raise ValueError(
                "min_successful_requests_total must be less than or equal to "
                "min_raw_objects_total"
            )
        return self
