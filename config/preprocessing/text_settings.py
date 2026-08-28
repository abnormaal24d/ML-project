"""Text-oriented preprocessing settings (consolidated from text.py + document.py)."""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel
from config.path_resolution.project_paths import QUARANTINE_ROOT


class TextQualityScorerSettings(SettingsModel):
    """Settings consumed by preprocessing text quality scoring."""

    min_text_chars: int = 300
    max_text_chars: int = 200000
    max_boilerplate_ratio: float = 0.65
    short_text_chars: int = 600
    medium_text_chars: int = 1500
    min_ascii_ratio: float = 0.85
    boilerplate_penalty_threshold: float = 0.4
    min_unique_terms: int = 60
    short_text_penalty: float = 0.25
    medium_text_penalty: float = 0.1
    ascii_penalty: float = 0.1
    boilerplate_penalty: float = 0.25
    unique_terms_penalty: float = 0.15
    gold_threshold: float = 0.85
    silver_threshold: float = 0.65
    bronze_threshold: float = 0.45
    rejected_score_cap: float = 0.25
    trusted_language_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    language_mismatch_confidence: float = Field(default=0.65, ge=0.0, le=1.0)


class PreprocessingInputValidationSettings(SettingsModel):
    """Settings for validating raw preprocessing inputs before parsing."""

    enabled: bool = True
    max_input_bytes: int = Field(default=25_000_000, ge=1)


class PrivacyDetectionSettings(SettingsModel):
    """Settings for local PII and multimodal privacy inspection."""

    enabled: bool = True
    quarantine_on_detection: bool = True


class PreprocessingQuarantineSettings(SettingsModel):
    """Settings for rejected preprocessing sample bookkeeping."""

    root: str = QUARANTINE_ROOT
