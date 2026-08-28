"""Canonical grouped preprocessing settings."""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel
from config.preprocessing.media_settings import (
    AudioValidationSettings,
    DiarizationSettings,
    ImageValidationSettings,
    MediaPrivacySettings,
    OcrBackendSettings,
    TranscriptionSettings,
    VideoValidationSettings,
)
from config.preprocessing.text_settings import (
    PreprocessingInputValidationSettings,
    PreprocessingQuarantineSettings,
    PrivacyDetectionSettings,
    TextQualityScorerSettings,
)


class PreprocessingSettings(SettingsModel):
    """Canonical grouped settings for preprocessing flows."""

    text_quality_scorer: TextQualityScorerSettings = Field(
        default_factory=TextQualityScorerSettings
    )
    input_validation: PreprocessingInputValidationSettings = Field(
        default_factory=PreprocessingInputValidationSettings
    )
    transcription: TranscriptionSettings = Field(
        default_factory=TranscriptionSettings
    )
    diarization: DiarizationSettings = Field(
        default_factory=DiarizationSettings
    )
    ocr: OcrBackendSettings = Field(default_factory=OcrBackendSettings)
    media_privacy: MediaPrivacySettings = Field(
        default_factory=MediaPrivacySettings
    )
    image_validation: ImageValidationSettings = Field(
        default_factory=ImageValidationSettings
    )
    audio_validation: AudioValidationSettings = Field(
        default_factory=AudioValidationSettings
    )
    video_validation: VideoValidationSettings = Field(
        default_factory=VideoValidationSettings
    )
    privacy_detection: PrivacyDetectionSettings = Field(
        default_factory=PrivacyDetectionSettings
    )
    quarantine: PreprocessingQuarantineSettings = Field(
        default_factory=PreprocessingQuarantineSettings
    )
