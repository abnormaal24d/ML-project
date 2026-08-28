"""Dataset section: paths, schemas, splits, raw/curated/training settings.

Full parity model for config.datasets (legacy) so consumers can migrate
without behavioral change. Path and schema constants mirror
config/path_resolution/project_paths.py, config/environment/default_values.py
and schemas/versions.py until the shared-constant cleanup (fase 8).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.path_resolution.project_paths import validate_safe_relative_path
from schemas.multimodal_tasks import canonical_task_name

_DATASET_SPLITS = frozenset({"train", "val", "test"})
_DATASET_MODALITIES = frozenset(
    {"text", "document", "image", "audio", "video"}
)

_DATA_ROOT = "data"
_RUNTIME_ROOT = "runtime"
_REGISTRY_ROOT = f"{_DATA_ROOT}/registry"
_WORKFLOW_ARTIFACTS_ROOT = f"{_REGISTRY_ROOT}/workflow_artifacts"
_RAW_RUNS_ROOT = f"{_DATA_ROOT}/raw/runs"
_CURATED_ROOT = f"{_DATA_ROOT}/curated"
_TRAINING_SETS_ROOT = f"{_DATA_ROOT}/interim/training_sets"
_AUGMENTED_TRAINING_SETS_ROOT = f"{_DATA_ROOT}/interim/augmented_training_sets"
_TRAINING_CHECKPOINTS_ROOT = f"{_RUNTIME_ROOT}/training/checkpoints"

_RAW_SCHEMA_VERSION = "3.0"
_CURATED_SCHEMA_VERSION = "3.0"
_TRAINING_SCHEMA_VERSION = "3.0"

_DEFAULT_SPLITS_DIRECTORY = "splits"
_DEFAULT_TRAIN_SPLIT_FILENAME = "train.jsonl"
_DEFAULT_VAL_SPLIT_FILENAME = "val.jsonl"
_DEFAULT_TEST_SPLIT_FILENAME = "test.jsonl"
_DEFAULT_DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
_DEFAULT_STATS_FILENAME = "stats.json"
_DEFAULT_DATASET_CARD_FILENAME = "dataset_card.json"
_DEFAULT_VALIDATION_REPORT_FILENAME = "validation_report.json"


class DatasetPathSettings(SettingsModel):
    output_directory: str = _DATA_ROOT
    output_subdirectory: Optional[str] = None
    workflow_artifacts_directory: str = _WORKFLOW_ARTIFACTS_ROOT
    raw_output_directory: str = _RAW_RUNS_ROOT
    curated_output_directory: str = _CURATED_ROOT
    training_output_directory: str = _TRAINING_SETS_ROOT
    augmented_training_output_directory: str = _AUGMENTED_TRAINING_SETS_ROOT
    training_checkpoint_directory: str = _TRAINING_CHECKPOINTS_ROOT
    objects_directory: str = "objects"
    artifacts_directory: str = "derived_text"
    raw_sync_directory: str = "records"
    raw_sync_by_modality_directory: str = "modality_index"
    raw_sync_relationships_filename: str = "crawl_edges.jsonl"
    raw_sync_metadata_filename: str = "object_metadata.jsonl"
    raw_sync_updates_filename: str = "object_events.jsonl"
    raw_sync_errors_filename: str = "errors.jsonl"
    raw_sync_discovered_assets_filename: str = (
        "discovered_asset_manifest.jsonl"
    )
    raw_sync_rejected_assets_filename: str = "rejected_asset_manifest.jsonl"
    raw_sync_current_objects_filename: str = "current_objects.jsonl"
    raw_sync_superseded_objects_filename: str = "superseded_objects.jsonl"
    raw_sync_summary_filename: str = "run_manifest.json"
    raw_pages_filename: str = "pages.jsonl"
    raw_feeds_filename: str = "feeds.jsonl"
    raw_documents_filename: str = "documents.jsonl"
    raw_images_filename: str = "images.jsonl"
    raw_audio_filename: str = "audio.jsonl"
    raw_video_filename: str = "video.jsonl"
    manifest_filename: str = "records/objects.jsonl"
    curated_entities_directory: str = "entities"
    curated_documents_filename: str = "documents.jsonl"
    curated_chunks_filename: str = "text_segments.jsonl"
    curated_images_filename: str = "images.jsonl"
    curated_audio_filename: str = "audio_items.jsonl"
    curated_video_filename: str = "video_items.jsonl"
    curated_views_directory: str = "views"
    curated_alignments_directory: str = "alignments"
    curated_sync_links_filename: str = "cross_modal_alignments.jsonl"
    curated_text_image_alignments_filename: str = "text_image_alignments.jsonl"
    curated_text_audio_alignments_filename: str = "text_audio_alignments.jsonl"
    curated_text_video_alignments_filename: str = "text_video_alignments.jsonl"
    curated_image_audio_alignments_filename: str = (
        "image_audio_alignments.jsonl"
    )
    training_splits_directory: str = _DEFAULT_SPLITS_DIRECTORY
    training_train_filename: str = _DEFAULT_TRAIN_SPLIT_FILENAME
    training_val_filename: str = _DEFAULT_VAL_SPLIT_FILENAME
    training_test_filename: str = _DEFAULT_TEST_SPLIT_FILENAME
    training_modalities_directory: str = "modality_views"
    training_tasks_directory: str = "tasks"
    snapshot_manifest_filename: str = "snapshot_manifest.json"
    dataset_manifest_filename: str = _DEFAULT_DATASET_MANIFEST_FILENAME
    stats_filename: str = _DEFAULT_STATS_FILENAME
    dataset_card_filename: str = _DEFAULT_DATASET_CARD_FILENAME
    validation_report_filename: str = _DEFAULT_VALIDATION_REPORT_FILENAME
    training_checkpoint_filename: str = "model_checkpoint.pt"
    training_metrics_filename: str = "multimodal_training_metrics.json"


class DatasetSchemaSettings(SettingsModel):
    raw_schema_version: str = _RAW_SCHEMA_VERSION
    curated_schema_version: str = _CURATED_SCHEMA_VERSION
    training_schema_version: str = _TRAINING_SCHEMA_VERSION


class SplitAssignerSettings(SettingsModel):
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class NearDeduperSettings(SettingsModel):
    threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    shingle_width: int = Field(default=4, ge=1)
    candidate_bands: int = Field(default=4, ge=1)
    use_buckets: bool = True


class DocumentChunkerSettings(SettingsModel):
    curated_schema_version: str = _CURATED_SCHEMA_VERSION
    chunk_min_target_tokens: int = Field(default=32, ge=1)
    chunk_target_tokens: int = Field(default=768, ge=1)
    chunk_overlap_tokens: int = Field(default=96, ge=0)


class CuratedDocumentAssemblerSettings(SettingsModel):
    curated_schema_version: str = _CURATED_SCHEMA_VERSION
    require_allow_training: bool = True
    max_documents_per_domain: int = Field(default=0, ge=0)
    min_quality_score_for_inclusion: float = Field(default=0.0, ge=0.0, le=1.0)


class CuratedDatasetWriterSettings(SettingsModel):
    pass


class CuratedDatasetAssemblerSettings(SettingsModel):
    curated_schema_version: str = _CURATED_SCHEMA_VERSION
    fail_on_empty_snapshot: bool = True
    write_manifest: bool = True
    max_build_errors: int = Field(default=0, ge=0)


class RawManifestReaderSettings(SettingsModel):
    raw_schema_version: str = _RAW_SCHEMA_VERSION
    run_selection_mode: str = "coverage_combined"
    selected_run_ids: tuple[str, ...] = ()
    coverage_selection_max_runs: int = Field(default=4, ge=1)
    deletion_index_max_bytes: int = Field(default=1_073_741_824, ge=1)
    deletion_index_max_rows: int = Field(default=1_000_000, ge=1)
    deletion_index_max_row_bytes: int = Field(default=8_388_608, ge=1)
    deletion_index_filename: str = "deletion_index.jsonl"

    @field_validator("run_selection_mode", mode="before")
    @classmethod
    def normalize_run_selection_mode(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("run_selection_mode must be a string")
        if value not in {
            "all",
            "coverage_combined",
            "coverage_best",
            "latest",
        }:
            raise ValueError(
                "run_selection_mode must be all, coverage_combined, "
                "coverage_best, or latest"
            )
        return value


class RawDatasetWriterSettings(SettingsModel):
    raw_schema_version: str = _RAW_SCHEMA_VERSION
    timestamped_runs: bool = True
    deduplicate_objects: bool = True
    deduplicate_within_run_by_normalized_url: bool = True
    manifest_flush_every_records: int = Field(default=20, ge=1)
    manifest_fsync_enabled: bool = False
    manifest_fsync_every_records: int = Field(default=200, ge=1)
    raw_persist_offload_to_thread: bool = True
    raw_sync_summary_every_records: int = Field(default=100, ge=1)
    enable_record_updates: bool = True


class TrainingDatasetWriterSettings(SettingsModel):
    write_jsonl: bool = True
    write_shards: bool = True
    shard_format: Literal["webdataset_tar"] = "webdataset_tar"
    training_shards_directory: str = "shards"
    shard_max_samples: int = Field(default=1000, ge=1)
    shard_max_bytes: Optional[int] = Field(default=None, ge=1)
    shard_index_filename: str = "shard_index.json"

    @model_validator(mode="after")
    def validate_output_contract(self) -> TrainingDatasetWriterSettings:
        if not self.write_jsonl and not self.write_shards:
            raise ValueError(
                "at least one of write_jsonl or write_shards must be true"
            )
        validate_safe_relative_path(
            self.training_shards_directory,
            field_name="training_shards_directory",
        )
        validate_safe_relative_path(
            self.shard_index_filename,
            field_name="shard_index_filename",
        )
        if not self.shard_index_filename.endswith(".json"):
            raise ValueError("shard_index_filename must end with '.json'")
        return self


class TrainingSnapshotAssemblerSettings(SettingsModel):
    training_schema_version: str = _TRAINING_SCHEMA_VERSION
    dataset_version_prefix: str = "training"
    processing_version: str = "training-builder-v2"
    text_tokenizer_backend: Literal["subword"] = "subword"
    text_tokenizer_name: str = "repo_subword"
    text_tokenizer_max_tokens: int = Field(default=512, gt=0)
    language_rules: str = "english_only"
    accepted_languages: tuple[str, ...] = ("en",)
    min_language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_alignment_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_caption_quality_score: float = Field(default=0.35, ge=0.0, le=1.0)
    min_pairability_score: float = Field(default=0.3, ge=0.0, le=1.0)
    require_license_rules: bool = True
    max_samples_per_domain: int = Field(default=0, ge=0)
    max_samples_per_domain_modality: int = Field(default=0, ge=0)
    max_samples_per_topic: int = Field(default=0, ge=0)
    max_samples_per_source_type: int = Field(default=0, ge=0)


def _is_unit_interval_threshold(value: float) -> bool:
    numeric_threshold = float(value)
    return 0.0 <= numeric_threshold <= 1.0


class DatasetValidatorSettings(SettingsModel):
    workflow_profile: str = "crawler_dataset"
    enable_quality_checks: bool = True
    fail_on_quality_checks: bool = True
    strict_production_acceptance: bool = False
    require_model_accepted_in_production: bool = True
    require_allow_training: bool = True
    require_non_empty_eval_splits: bool = True
    require_known_license: bool = False
    require_license_evidence: bool = False
    require_license_url_or_terms: bool = False
    require_safety_passed: bool = False
    require_pii_passed: bool = False
    require_complete_payload: bool = True
    require_media_objects: bool = True
    require_autonomous_multimodal_readiness: bool = False
    require_generation_targets: bool = False
    min_language_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    min_total_samples: int = Field(default=0, ge=0)
    min_train_samples: int = Field(default=0, ge=0)
    min_val_samples: int = Field(default=0, ge=0)
    min_test_samples: int = Field(default=0, ge=0)
    min_text_samples: int = Field(default=0, ge=0)
    min_image_samples: int = Field(default=0, ge=0)
    min_document_samples: int = Field(default=0, ge=0)
    min_audio_samples: int = Field(default=0, ge=0)
    min_video_samples: int = Field(default=0, ge=0)
    min_task_samples: dict[str, int] = {}
    min_quality_score_by_modality: dict[str, float] = {}
    min_context_score_by_modality: dict[str, float] = {}
    min_modality_samples_by_split: dict[str, dict[str, int]] = {}
    min_alignment_score_by_modality: dict[str, float] = {}
    min_alignment_score_by_task: dict[str, float] = {}
    min_task_samples_by_workflow: dict[str, dict[str, int]] = {}
    min_alignment_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    require_all_modalities_in_train: bool = False
    require_all_modalities_in_val: bool = False
    require_all_modalities_in_test: bool = False
    require_finite_losses: bool = True
    require_supervised_metrics_when_labeled: bool = True
    fail_on_smoke_trainer: bool = True
    min_training_batches: int = Field(default=0, ge=0)
    max_test_train_loss_ratio: Optional[float] = Field(default=None, gt=0.0)
    model_min_total_samples: int = Field(default=0, ge=0)
    model_min_train_samples: int = Field(default=0, ge=0)
    model_min_val_samples: int = Field(default=0, ge=0)
    model_min_test_samples: int = Field(default=0, ge=0)
    model_min_training_batches: int = Field(default=0, ge=0)
    model_max_test_train_loss_ratio: Optional[float] = Field(
        default=None, gt=0.0
    )
    min_vqa_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_ocr_character_error_rate: Optional[float] = Field(default=None, ge=0.0)
    max_ocr_word_error_rate: Optional[float] = Field(default=None, ge=0.0)
    min_doc_qa_f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_speech_translation_bleu: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    min_emotion_f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_speaker_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_image_generation_mse: Optional[float] = Field(default=None, ge=0.0)
    min_video_token_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    max_image_editing_preservation_mse: Optional[float] = Field(
        default=None, ge=0.0
    )
    max_layout_box_mse: Optional[float] = Field(default=None, ge=0.0)
    max_visual_grounding_box_mse: Optional[float] = Field(default=None, ge=0.0)
    max_domain_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_language_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    require_validation_report_clean: bool = True
    require_evaluation_metrics: bool = False
    min_evaluation_metrics: dict[str, float] = {}
    require_coverage_report: bool = False
    require_evaluation_report: bool = False
    min_active_modalities: int = Field(default=0, ge=0)
    max_batch_latency_ms: Optional[float] = Field(default=None, gt=0.0)
    max_peak_memory_mb: Optional[float] = Field(default=None, gt=0.0)
    require_dataset_card: bool = False
    require_model_card: bool = False

    def _validate_split_totals(self) -> None:
        split_minimum_total = (
            self.min_train_samples
            + self.min_val_samples
            + self.min_test_samples
        )
        if split_minimum_total > self.min_total_samples:
            raise ValueError(
                "min_train_samples + min_val_samples + min_test_samples "
                "must be less than or equal to min_total_samples"
            )

    def _validate_non_negative_minimums(self) -> None:
        invalid_task_minimums = [
            task_name
            for task_name, minimum in self.min_task_samples.items()
            if int(minimum) < 0
        ]
        workflow_minimums = self.min_task_samples_by_workflow
        invalid_profile_task_minimums = [
            f"{profile}.{task_name}"
            for profile, task_minimums in workflow_minimums.items()
            for task_name, minimum in task_minimums.items()
            if int(minimum) < 0
        ]
        if invalid_task_minimums:
            raise ValueError(
                "min_task_samples values must be greater than or equal to zero"
            )
        if invalid_profile_task_minimums:
            raise ValueError(
                "min_task_samples_by_workflow values must be greater than "
                "or equal to zero"
            )

        split_modality_minimums = self.min_modality_samples_by_split
        invalid_split_modality_minimums = [
            f"{split_name}.{modality}"
            for split_name, modality_minimums in split_modality_minimums.items()
            for modality, minimum in modality_minimums.items()
            if int(minimum) < 0
        ]
        if invalid_split_modality_minimums:
            raise ValueError(
                "min_modality_samples_by_split values must be greater than "
                "or equal to zero"
            )

        invalid_evaluation_minimums = [
            metric_name
            for metric_name, minimum in self.min_evaluation_metrics.items()
            if float(minimum) < 0.0
        ]
        if invalid_evaluation_minimums:
            raise ValueError(
                "min_evaluation_metrics values must be greater than or equal "
                "to zero"
            )

    def _validate_quality_thresholds(self) -> None:
        invalid_quality_thresholds: dict[str, list[str]] = {}
        for setting_name, thresholds in (
            (
                "min_quality_score_by_modality",
                self.min_quality_score_by_modality,
            ),
            (
                "min_context_score_by_modality",
                self.min_context_score_by_modality,
            ),
            (
                "min_alignment_score_by_modality",
                self.min_alignment_score_by_modality,
            ),
            (
                "min_alignment_score_by_task",
                self.min_alignment_score_by_task,
            ),
        ):
            invalid = [
                key
                for key, value in thresholds.items()
                if not _is_unit_interval_threshold(value)
            ]
            if invalid:
                invalid_quality_thresholds[setting_name] = invalid

        if invalid_quality_thresholds:
            setting_name = next(iter(invalid_quality_thresholds))
            raise ValueError(
                f"{setting_name} values must be between 0.0 and 1.0"
            )

    def _validate_split_and_modality_keys(self) -> None:
        invalid_split_keys = sorted(
            split_name
            for split_name in self.min_modality_samples_by_split
            if split_name not in _DATASET_SPLITS
        )
        if invalid_split_keys:
            raise ValueError(
                "min_modality_samples_by_split keys must be train, val, or "
                f"test: {invalid_split_keys}"
            )

        invalid_modality_keys = sorted(
            f"{split_name}.{modality}"
            for split_name, modality_minimums in self.min_modality_samples_by_split.items()
            for modality in modality_minimums
            if modality not in _DATASET_MODALITIES
        )
        if invalid_modality_keys:
            raise ValueError(
                "min_modality_samples_by_split modality keys must be text, "
                f"document, image, audio, or video: {invalid_modality_keys}"
            )

    def _validate_all_modalities_requirements(self) -> None:
        for split_name, required in (
            ("train", self.require_all_modalities_in_train),
            ("val", self.require_all_modalities_in_val),
            ("test", self.require_all_modalities_in_test),
        ):
            if not required:
                continue
            present = set(
                self.min_modality_samples_by_split.get(split_name, {})
            )
            missing = sorted(_DATASET_MODALITIES - present)
            if missing:
                raise ValueError(
                    "min_modality_samples_by_split."
                    f"{split_name} must include all modalities when "
                    f"require_all_modalities_in_{split_name}=True: {missing}"
                )

    def _validate_model_vs_dataset_minimums(self) -> None:
        model_split_total = (
            self.model_min_train_samples
            + self.model_min_val_samples
            + self.model_min_test_samples
        )
        if model_split_total > self.model_min_total_samples:
            raise ValueError(
                "model_min_train_samples + model_min_val_samples + "
                "model_min_test_samples must be less than or equal to "
                "model_min_total_samples"
            )

        for field_name, model_minimum, dataset_minimum in (
            (
                "model_min_train_samples",
                self.model_min_train_samples,
                self.min_train_samples,
            ),
            (
                "model_min_val_samples",
                self.model_min_val_samples,
                self.min_val_samples,
            ),
            (
                "model_min_test_samples",
                self.model_min_test_samples,
                self.min_test_samples,
            ),
            (
                "model_min_total_samples",
                self.model_min_total_samples,
                self.min_total_samples,
            ),
        ):
            if model_minimum > dataset_minimum:
                raise ValueError(
                    f"{field_name} must be less than or equal to the "
                    f"corresponding dataset minimum"
                )

    @model_validator(mode="after")
    def validate_minimums(self) -> DatasetValidatorSettings:
        self._validate_split_totals()
        self._validate_non_negative_minimums()
        self._validate_quality_thresholds()
        self._validate_split_and_modality_keys()
        self._validate_all_modalities_requirements()
        self._validate_model_vs_dataset_minimums()
        return self

    def effective_min_task_samples(self) -> dict[str, int]:
        profile = str(self.workflow_profile).strip() or "crawler_dataset"
        profile_minimums = self.min_task_samples_by_workflow.get(profile)
        source_minimums = (
            profile_minimums
            if profile_minimums is not None
            else self.min_task_samples
        )
        normalized: dict[str, int] = {}
        for task_name, minimum in source_minimums.items():
            canonical = canonical_task_name(task_name)
            if canonical in normalized:
                raise ValueError(
                    "dataset validator min_task_samples contains duplicate "
                    f"canonical task key: {canonical!r}"
                )
            normalized[canonical] = int(minimum)
        return normalized


class RawDatasetSettings(SettingsModel):
    writer: RawDatasetWriterSettings = RawDatasetWriterSettings()
    manifest_reader: RawManifestReaderSettings = RawManifestReaderSettings()


class CuratedDatasetSettings(SettingsModel):
    near_deduper: NearDeduperSettings = NearDeduperSettings()
    document_chunker: DocumentChunkerSettings = DocumentChunkerSettings()
    document_assembler: CuratedDocumentAssemblerSettings = (
        CuratedDocumentAssemblerSettings()
    )
    writer: CuratedDatasetWriterSettings = CuratedDatasetWriterSettings()
    builder: CuratedDatasetAssemblerSettings = (
        CuratedDatasetAssemblerSettings()
    )


class TrainingDatasetSettings(SettingsModel):
    dataset_validator: DatasetValidatorSettings = DatasetValidatorSettings()
    writer: TrainingDatasetWriterSettings = TrainingDatasetWriterSettings()
    snapshot_builder: TrainingSnapshotAssemblerSettings = (
        TrainingSnapshotAssemblerSettings()
    )


class DatasetSplitSettings(SettingsModel):
    curation: SplitAssignerSettings = SplitAssignerSettings()
    training: SplitAssignerSettings = SplitAssignerSettings()


class DatasetSettings(SettingsModel):
    paths: DatasetPathSettings = DatasetPathSettings()
    schemas: DatasetSchemaSettings = DatasetSchemaSettings()
    splits: DatasetSplitSettings = DatasetSplitSettings()
    raw: RawDatasetSettings = RawDatasetSettings()
    curation: CuratedDatasetSettings = CuratedDatasetSettings()
    training: TrainingDatasetSettings = TrainingDatasetSettings()
