"""Coverage settings used by crawler.coverage."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel


class CoverageKindSettings(SettingsModel):
    """Configured coverage kinds and mappings."""

    media_kinds: tuple[str, ...] = Field(
        default=("document", "audio", "video", "image", "feed")
    )
    focusable_media_kinds: tuple[str, ...] = Field(
        default=("document", "audio", "video", "image")
    )
    gated_fetch_kinds: tuple[str, ...] = Field(
        default=("image", "audio", "video", "document", "feed")
    )
    tracked_modalities: tuple[str, ...] = Field(
        default=(
            "text",
            "document",
            "image",
            "audio",
            "video",
            "layout",
            "mask",
            "code",
            "json",
            "table",
            "screen",
        )
    )
    task_to_media_kind: dict[str, str] = Field(
        default_factory=lambda: {
            "image_text_pair": "image",
            "audio_text_pair": "audio",
            "video_text_pair": "video",
            "document_text_pair": "document",
            "pdf_text_pair": "document",
            "ocr_parse": "document",
            "doc_qa": "document",
            "speech_translation": "audio",
            "text_to_image": "image",
            "text_to_video": "video",
            "video_captioning": "video",
            "video_qa": "video",
            "video_summarization": "video",
            "scene_understanding": "video",
            "action_recognition": "video",
            "scene_retrieval": "video",
            "video_editing": "video",
        }
    )
    modality_to_media_kind: dict[str, str] = Field(
        default_factory=lambda: {
            "image": "image",
            "audio": "audio",
            "video": "video",
            "document": "document",
            "feed": "audio",
        }
    )
    generation_tasks: tuple[str, ...] = Field(
        default=(
            "text_to_image",
            "image_editing",
            "text_to_video",
            "video_editing",
            "speech_translation",
            "speech_to_audio",
        )
    )

    @model_validator(mode="after")
    def validate_references(self) -> CoverageKindSettings:
        # point 190: no duplicates
        for fname, vals in (
            ("media_kinds", self.media_kinds),
            ("focusable_media_kinds", self.focusable_media_kinds),
            ("gated_fetch_kinds", self.gated_fetch_kinds),
            ("tracked_modalities", self.tracked_modalities),
            ("generation_tasks", self.generation_tasks),
        ):
            if len(vals) != len(set(vals)):
                dups = [x for x in vals if vals.count(x) > 1]
                raise ValueError(
                    f"coverage.kinds.{fname} contains duplicates: {dups}"
                )

        known_media = set(self.media_kinds)

        for field_name, values in (
            ("focusable_media_kinds", self.focusable_media_kinds),
            ("gated_fetch_kinds", self.gated_fetch_kinds),
        ):
            unknown = set(values) - known_media
            if unknown:
                raise ValueError(
                    f"coverage.kinds.{field_name} contains unknown media "
                    f"kinds: {sorted(unknown)}"
                )

        unknown_task_targets = (
            set(self.task_to_media_kind.values()) - known_media
        )
        if unknown_task_targets:
            raise ValueError(
                "coverage.kinds.task_to_media_kind targets unknown media "
                f"kinds: {sorted(unknown_task_targets)}"
            )

        unknown_modality_targets = (
            set(self.modality_to_media_kind.values()) - known_media
        )
        if unknown_modality_targets:
            raise ValueError(
                "coverage.kinds.modality_to_media_kind targets unknown media "
                f"kinds: {sorted(unknown_modality_targets)}"
            )

        return self


class CoverageTargetSettings(SettingsModel):
    """Configured coverage targets."""

    modality_targets: dict[str, int] = Field(default_factory=dict)
    task_targets: dict[str, int] = Field(default_factory=dict)
    raw_modality_minimums: dict[str, int] = Field(default_factory=dict)
    min_missing_to_crawl: int = Field(default=1, ge=1)

    @field_validator(
        "modality_targets",
        "task_targets",
        "raw_modality_minimums",
        mode="before",
    )
    @classmethod
    def validate_target_counts(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("coverage targets must be a mapping")
        validated: dict[str, int] = {}
        for raw_key, raw_count in value.items():
            key = str(raw_key).strip()
            if not key or key != key.lower():
                raise ValueError("coverage target keys must be lowercase")
            if isinstance(raw_count, bool):
                raise ValueError("coverage target counts must be integers")
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "coverage target counts must be integers"
                ) from exc
            if count < 0:
                raise ValueError("coverage target counts must be non-negative")
            validated[key] = count
        return validated


class CoverageFocusSettings(SettingsModel):
    """Crawler focus behavior derived from coverage gaps."""

    enabled: bool = False
    media_priority: tuple[str, ...] = Field(
        default=("document", "audio", "video", "image", "feed")
    )
    task_gap_weight: int = Field(default=1, ge=1)
    modality_gap_weight: int = Field(default=2, ge=1)
    document_priority_boost_enabled: bool = True
    exclude_feed_from_focus: bool = True
    boost_multiplier: int = Field(default=3, ge=1)
    asset_ranking_boost: float = Field(default=50.0, ge=0.0)
    non_target_slots: int = Field(default=1, ge=0)
    max_total_slots: int = Field(default=64, ge=1)
    minimum_focus_slots_by_kind: dict[str, int] = Field(
        default_factory=lambda: {
            "document": 3,
            "image": 3,
            "audio": 2,
            "video": 2,
        }
    )
    discovery_scan_multiplier: int = Field(default=3, ge=1)
    focused_discovery_scan_multiplier: int = Field(default=32, ge=1)


class CoverageProgressSettings(SettingsModel):
    """No-progress guard settings."""

    max_no_progress_attempts: int = Field(default=2, ge=1)
    duplicate_sample_error_prefix: str = "duplicate_sample_id:"
    blocked_reason: str = "coverage_gaps_without_progress"


class CoverageDiscoverySettings(SettingsModel):
    """Stable names for the crawler's coverage-driven discovery modes."""

    complete_mode_name: str = "coverage_complete"
    document_recovery_mode_name: str = "document_recovery"
    balanced_recovery_mode_name: str = "balanced_recovery"


class CoverageErrorNameSettings(SettingsModel):
    """Configurable coverage validation error names."""

    total_samples_below_min: str = "total_samples_below_min"
    train_samples_below_min: str = "train_samples_below_min"
    val_samples_below_min: str = "val_samples_below_min"
    test_samples_below_min: str = "test_samples_below_min"
    modality_coverage_below_min: str = "modality_coverage_below_min"
    split_modality_coverage_below_min: str = (
        "split_modality_coverage_below_min"
    )
    raw_modality_coverage_below_min: str = "raw_modality_coverage_below_min"
    task_coverage_below_min: str = "task_coverage_below_min"
    raw_modality_missing: str = "raw_modality_missing"
    alignment_coverage_below_min: str = "alignment_coverage_below_min"
    autonomous_missing_modalities: str = (
        "autonomous_multimodal_readiness_missing_modalities"
    )
    autonomous_missing_tasks: str = (
        "autonomous_multimodal_readiness_missing_tasks"
    )
    autonomous_generation_missing_tasks: str = (
        "autonomous_multimodal_generation_missing_tasks"
    )


class CoverageTrainingValidationSettings(SettingsModel):
    """Training coverage validation thresholds."""

    min_text_samples: int = Field(default=0, ge=0)
    min_image_samples: int = Field(default=0, ge=0)
    min_audio_samples: int = Field(default=0, ge=0)
    min_video_samples: int = Field(default=0, ge=0)
    min_document_samples: int = Field(default=0, ge=0)
    min_alignment_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    require_all_modalities_in_train: bool = False
    require_all_modalities_in_val: bool = False
    require_all_modalities_in_test: bool = False
    require_autonomous_multimodal_readiness: bool = False
    require_generation_targets: bool = False
    task_minimums: dict[str, int] = Field(default_factory=dict)
    enabled_task_types: tuple[str, ...] = Field(default_factory=tuple)


class CoverageReportSettings(SettingsModel):
    """Coverage report names and schema settings."""

    directory_name: str = "coverage"
    raw_modality_report_filename: str = "raw_modality_coverage_report.json"
    modality_report_filename: str = "modality_coverage_report.json"
    task_report_filename: str = "task_coverage_report.json"
    target_quality_report_filename: str = "target_quality_report.json"
    trend_report_filename: str = "coverage_trend_report.json"
    schema_version: int = Field(default=1, ge=1)


class CoverageSettings(SettingsModel):
    """Root coverage settings."""

    kinds: CoverageKindSettings = Field(default_factory=CoverageKindSettings)
    targets: CoverageTargetSettings = Field(
        default_factory=CoverageTargetSettings
    )
    focus: CoverageFocusSettings = Field(default_factory=CoverageFocusSettings)
    progress: CoverageProgressSettings = Field(
        default_factory=CoverageProgressSettings
    )
    discovery: CoverageDiscoverySettings = Field(
        default_factory=CoverageDiscoverySettings
    )
    error_names: CoverageErrorNameSettings = Field(
        default_factory=CoverageErrorNameSettings
    )
    training_validation: CoverageTrainingValidationSettings = Field(
        default_factory=CoverageTrainingValidationSettings
    )
    reports: CoverageReportSettings = Field(
        default_factory=CoverageReportSettings
    )
