"""Training-input gate settings for crawl output and datachecker readiness.

``CrawlOutputGateSettings`` comes from the new settings tree
(``config.settings.gate``). The datachecker-only models
(``TrainingInputMode``, ``DataCheckerSettings``) are collection-internal
and stay here until the collection section is ported (fase 2/5-9).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from config.base.settings_model import SettingsModel
from config.environment.default_values import DEFAULT_TRAIN_SPLIT_FILENAME
from config.settings.gate import CrawlOutputGateSettings

__all__ = [
    "CrawlOutputGateSettings",
    "DataCheckerSettings",
    "TrainingInputMode",
]


class TrainingInputMode(StrEnum):
    """Declare which dataset stage training is allowed to consume."""

    PREPROCESSED_ONLY = "preprocessed_only"
    AUGMENTED_WHEN_AVAILABLE = "augmented_when_available"
    AUGMENTED_REQUIRED = "augmented_required"


class DataCheckerSettings(SettingsModel):
    """Canonical datachecker settings for phase-aware workflow decisions."""

    crawl_state_manifest_filename: str = "crawl_state_manifest.json"
    crawl_manifest_filename: str = "crawl_manifest.json"
    preprocessing_manifest_filename: str = "preprocessing_manifest.json"
    augmentation_manifest_filename: str = "augmentation_manifest.json"
    training_manifest_filename: str = "training_manifest.json"
    train_file_glob: str = f"**/{DEFAULT_TRAIN_SPLIT_FILENAME}"
    checkpoint_glob: str = "**/model_checkpoint.pt"

    training_input_mode: TrainingInputMode = (
        TrainingInputMode.AUGMENTED_WHEN_AVAILABLE
    )

    min_crawl_output_files: int = Field(default=1, ge=1)
    min_preprocessed_documents: int = Field(default=1, ge=0)
    min_preprocessed_chunks: int = Field(default=1, ge=0)
    min_preprocessed_images: int = Field(default=0, ge=0)
    min_preprocessed_audio: int = Field(default=0, ge=0)
    min_preprocessed_video: int = Field(default=0, ge=0)
    min_cross_modal_alignments: int = Field(default=0, ge=0)
    min_transcript_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    min_ocr_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    min_keyframe_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
