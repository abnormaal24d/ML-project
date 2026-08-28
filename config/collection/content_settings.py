"""Content processing settings."""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel


class ContentProcessorSettings(SettingsModel):
    """Settings for the content classification workflow."""

    fallback_kind: str = Field(default="document", min_length=1)
    classify_topic: bool = True
    text_metadata_sample_bytes: int = Field(default=32_768, ge=1)
