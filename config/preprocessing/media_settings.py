"""Media (audio/image/video) preprocessing validation settings (consolidated)."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class MediaPrivacySettings(SettingsModel):
    """Preprocessing privacy operation timeouts and limits.

    Tool identity (ffmpeg/ffprobe executable and version pins) is owned by
    config.media_toolchain.MediaToolchainSettings.
    """

    probe_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    metadata_strip_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=3600.0,
    )
    video_audio_probe_timeout_seconds: float = Field(
        default=15.0,
        gt=0.0,
        le=3600.0,
    )
    video_privacy_max_frames: int = Field(default=10_000, ge=1)


class TranscriptionSettings(SettingsModel):
    """Configuration for audio/video speech-to-text (preprocessing only).

    Backend output (transcripts) may be used as training labels/metadata.
    The actual ASR model is never part of the scratch multimodal model.
    """

    enabled: bool = Field(default=False)
    backend: Literal["whisper", "disabled"] = Field(default="disabled")
    model_name: str = Field(default="base")
    model_revision: str | None = Field(default=None, min_length=1)
    model_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    backend_version: str | None = Field(default=None, min_length=1)
    device: str = Field(default="cpu")
    compute_type: str = Field(default="int8")
    language: str | None = None
    beam_size: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    vad_filter: bool = Field(default=True)
    vad_min_silence_duration_ms: int = Field(default=500, ge=0)
    word_timestamps: bool = Field(default=False)
    condition_on_previous_text: bool = Field(default=True)
    batch_size: int = Field(default=1, ge=1)
    max_audio_duration_seconds: int = Field(default=3600, ge=1)
    cache_directory: str | None = Field(default=None)
    local_files_only: bool = Field(default=False)
    production_mode: bool = Field(default=False)
    minimum_label_quality: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_production_model_pin(self) -> TranscriptionSettings:
        if self.enabled != (self.backend == "whisper"):
            raise ValueError(
                "transcription.enabled must be true exactly when "
                "backend='whisper'"
            )
        if not self.production_mode or not self.enabled:
            return self

        pinned_fields = (
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("model_artifact_hash", self.model_artifact_hash),
            ("backend_version", self.backend_version),
        )
        missing = [
            name
            for name, value in pinned_fields
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ValueError(
                "production Whisper preprocessing requires pinned "
                + ", ".join(missing)
            )
        if not self.local_files_only:
            raise ValueError(
                "production Whisper preprocessing requires local_files_only=true"
            )
        if not (
            PurePosixPath(self.model_name).is_absolute()
            or PureWindowsPath(self.model_name).is_absolute()
        ):
            raise ValueError(
                "production Whisper model_name must be an absolute local model directory"
            )
        return self


class DiarizationSettings(SettingsModel):
    """Configuration for speaker diarization (preprocessing only)."""

    enabled: bool = Field(default=False)
    backend: Literal["pyannote", "transcript_hints", "disabled"] = "disabled"
    model_name: str = Field(default="pyannote/speaker-diarization-3.1")
    model_revision: str | None = Field(default=None, min_length=1)
    model_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    backend_version: str | None = Field(default=None, min_length=1)
    local_model_path: str | None = Field(default=None, min_length=1)
    local_files_only: bool = False
    token_environment_variable: str = Field(default="HF_TOKEN", min_length=1)
    production_mode: bool = False
    device: str = Field(default="cpu")

    @model_validator(mode="after")
    def validate_enabled_backend(self) -> DiarizationSettings:
        if self.enabled != (self.backend != "disabled"):
            raise ValueError(
                "diarization.enabled must be true exactly when "
                "backend is not 'disabled'"
            )
        return self

    @model_validator(mode="after")
    def validate_production_pyannote_pin(self) -> DiarizationSettings:
        if (
            not self.production_mode
            or not self.enabled
            or self.backend != "pyannote"
        ):
            return self
        pinned_fields = (
            ("model_revision", self.model_revision),
            ("model_artifact_hash", self.model_artifact_hash),
            ("backend_version", self.backend_version),
            ("local_model_path", self.local_model_path),
        )
        missing = [name for name, value in pinned_fields if value is None]
        if missing:
            raise ValueError(
                "production pyannote preprocessing requires pinned "
                + ", ".join(missing)
            )
        if not self.local_files_only:
            raise ValueError(
                "production pyannote preprocessing requires "
                "local_files_only=true"
            )
        return self


class OcrBackendSettings(SettingsModel):
    """Canonical OCR backend and production pinning rules.

    Sole owner of which OCR engine is used. Per-modality ``run_ocr`` toggles
    only decide *where* OCR is applied.
    """

    backend: Literal["disabled", "tesseract", "rapidocr"] = "disabled"
    backend_version: str | None = Field(default=None, min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    model_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    model_artifact_path: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    production_mode: bool = False

    @model_validator(mode="after")
    def validate_production_model_pin(self) -> OcrBackendSettings:
        if not self.production_mode or self.backend == "disabled":
            return self
        if self.backend == "rapidocr":
            raise ValueError(
                "production RapidOCR is unavailable because its detector, "
                "classifier, and recognizer artifacts cannot all be bound "
                "to the single configured model_artifact_path"
            )
        pinned_fields = (
            ("backend_version", self.backend_version),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("model_artifact_hash", self.model_artifact_hash),
            ("model_artifact_path", self.model_artifact_path),
        )
        missing = [name for name, value in pinned_fields if value is None]
        if missing:
            raise ValueError(
                "production OCR preprocessing requires pinned "
                + ", ".join(missing)
            )
        return self


class AudioValidationSettings(SettingsModel):
    """Validation thresholds for audio preprocessing inputs."""

    enabled: bool = True
    min_bytes: int = Field(default=1, ge=0)
    min_duration_seconds: float = Field(default=0.1, ge=0.0)
    min_sample_rate: int = Field(default=8_000, ge=1)
    min_channels: int = Field(default=1, ge=1)
    max_channels: int = Field(default=8, ge=1)
    max_duration_seconds: float = Field(default=3600.0, ge=0.0)
    require_transcript: bool = False
    require_transcript_for_audio_text_pair: bool = True
    require_transcript_for_speech_transcription: bool = True
    require_transcript_for_sound_classification: bool = False
    require_transcript_for_diarization: bool = False
    require_transcript_for_emotion: bool = False
    require_transcript_for_generation: bool = False
    require_transcript_for_speech_translation: bool = True
    require_audio_fingerprint: bool = True
    chromaprint_executable: str = Field(default="fpcalc", min_length=1)
    chromaprint_expected_version: str = Field(
        default="1.5.1",
        pattern=r"^\d+(?:\.\d+){2}$",
    )
    chromaprint_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=3600.0,
    )


class ImageValidationSettings(SettingsModel):
    """Validation thresholds for image preprocessing inputs."""

    enabled: bool = True
    min_width: int = Field(default=64, ge=1)
    min_height: int = Field(default=64, ge=1)
    min_bytes: int = Field(default=1, ge=0)
    require_semantic_text: bool = False
    require_semantic_text_for_alignment: bool = True
    require_semantic_text_for_generation: bool = False
    require_semantic_text_for_detection: bool = False
    require_semantic_text_for_classification: bool = False
    require_semantic_text_for_screenshot: bool = False
    require_semantic_text_for_editing: bool = False


class VideoValidationSettings(SettingsModel):
    """Validation thresholds for video preprocessing inputs."""

    enabled: bool = True
    min_width: int = Field(default=64, ge=1)
    min_height: int = Field(default=64, ge=1)
    min_bytes: int = Field(default=1, ge=0)
    min_duration_seconds: float = Field(default=0.1, ge=0.0)
    min_fps: float = Field(default=1.0, ge=0.0)
    require_keyframes: bool = False
    require_semantic_text_or_keyframes: bool = True
