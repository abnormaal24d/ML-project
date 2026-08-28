"""Validate the current workflow artifacts and select one next action."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from datachecker.manifests.augmentation_manifest import AugmentationManifest
from datachecker.manifests.crawl_manifest import CrawlManifest
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.preprocessing_manifest import PreprocessingManifest
from datachecker.manifests.training_manifest import TrainingManifest
from datachecker.training_input_selection import (
    SelectedTrainingInput,
    select_training_input,
)
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
    decide_workflow_action,
)
from logger.project_logger import ProjectLogger
from schemas.versions import (
    SUPPORTED_WORKFLOW_MANIFEST_SCHEMA_VERSIONS,
    is_supported_workflow_manifest_schema_version,
    schema_version_error,
)

if TYPE_CHECKING:
    from config.collection.training_input_gate import TrainingInputMode
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import FileFingerprintCalculator
    from datachecker.inventory.curated_snapshot_inventory import (
        CuratedInventory,
        CuratedInventoryReader,
    )
    from datachecker.inventory.raw_run_inventory import (
        RawInventory,
        RawInventoryReader,
    )
    from datachecker.inventory.training_snapshot_inventory import (
        TrainingInventory,
        TrainingInventoryReader,
    )
    from datachecker.validation.augmentation_artifact_validator import (
        AugmentationArtifactValidator,
    )
    from datachecker.validation.crawl_artifact_validator import (
        CrawlArtifactValidator,
    )
    from datachecker.validation.preprocessing_artifact_validator import (
        PreprocessingArtifactValidator,
    )
    from datachecker.validation.training_artifact_validator import (
        TrainingArtifactValidator,
    )


_Manifest = TypeVar("_Manifest")
CoverageGapsResolver = Callable[[tuple[str, ...]], dict[str, int]]


@dataclass(frozen=True, slots=True)
class WorkflowFingerprints:
    """Current configuration fingerprints used to validate workflow outputs."""

    source_registry: str
    crawl: str
    preprocessing: str
    normalization: str
    deduplication: str
    splitting: str
    validation: str
    augmentation: str
    augmentation_strategy: str
    training: str
    model: str


class DataCheckerTimeoutError(TimeoutError):
    """Raised when a DataChecker execution exceeds its time budget."""

    def __init__(
        self,
        *,
        stage: str,
        timeout_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            "DataChecker exceeded its execution budget "
            f"during {stage!r}: {elapsed_seconds:.3f}s elapsed, "
            f"{timeout_seconds:.3f}s allowed."
        )


class _DataCheckerDeadline:
    """Monotonic deadline shared by every timed checker dependency."""

    def __init__(
        self,
        timeout_seconds: float,
        *,
        monotonic_seconds: Callable[[], float],
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._monotonic_seconds = monotonic_seconds
        self.started_at = monotonic_seconds()
        self.timeout_seconds = timeout_seconds
        self.expires_at = self.started_at + timeout_seconds

    def checkpoint(self, stage: str) -> None:
        now = self._monotonic_seconds()
        if now >= self.expires_at:
            raise DataCheckerTimeoutError(
                stage=stage,
                timeout_seconds=self.timeout_seconds,
                elapsed_seconds=now - self.started_at,
            )


class DataChecker:
    """Validate native artifact inventories and resolve the next workflow action."""

    def __init__(
        self,
        *,
        raw_inventory_reader: RawInventoryReader,
        curated_inventory_reader: CuratedInventoryReader,
        training_inventory_reader: TrainingInventoryReader,
        crawl_validator: CrawlArtifactValidator,
        preprocessing_validator: PreprocessingArtifactValidator,
        augmentation_validator: AugmentationArtifactValidator,
        training_validator: TrainingArtifactValidator,
        artifact_path_registry: ArtifactPathRegistry,
        file_fingerprint_calculator: FileFingerprintCalculator,
        fingerprints: WorkflowFingerprints,
        coverage_gaps_resolver: CoverageGapsResolver,
        augmentation_enabled: bool,
        training_input_mode: TrainingInputMode,
        ordered_actions: tuple[WorkflowAction, ...],
        optional_actions: tuple[WorkflowAction, ...],
        require_seed_urls: bool,
        seed_url_count: int,
        logger: ProjectLogger,
        monotonic_seconds: Callable[[], float],
    ) -> None:
        self._raw_inventory_reader = raw_inventory_reader
        self._curated_inventory_reader = curated_inventory_reader
        self._training_inventory_reader = training_inventory_reader
        self._crawl_validator = crawl_validator
        self._preprocessing_validator = preprocessing_validator
        self._augmentation_validator = augmentation_validator
        self._training_validator = training_validator
        self._artifact_path_registry = artifact_path_registry
        self._file_fingerprint_calculator = file_fingerprint_calculator
        self._fingerprints = fingerprints
        self._coverage_gaps_resolver = coverage_gaps_resolver
        self._augmentation_enabled = augmentation_enabled
        self._training_input_mode = training_input_mode
        self._ordered_actions = ordered_actions
        self._optional_actions = optional_actions
        self._require_seed_urls = require_seed_urls
        self._seed_url_count = seed_url_count
        self._logger = logger
        self._monotonic_seconds = monotonic_seconds

    def check(self, *, timeout_seconds: float) -> WorkflowExecutionPlan:
        """Evaluate one immutable workflow state within one deadline."""

        deadline = _DataCheckerDeadline(
            timeout_seconds,
            monotonic_seconds=self._monotonic_seconds,
        )
        crawl_manifest, crawl_manifest_error = self._read_manifest(
            path=self._artifact_path_registry.crawl_manifest_path(),
            parser=CrawlManifest.from_payload,
            checkpoint=deadline.checkpoint,
            stage="crawl_manifest_read",
        )
        crawl_state_manifest, crawl_state_error = self._read_manifest(
            path=self._artifact_path_registry.crawl_state_manifest_path(),
            parser=CrawlStateManifest.from_payload,
            checkpoint=deadline.checkpoint,
            stage="crawl_state_manifest_read",
        )

        raw_run_directory, run_summary_path = self._raw_inventory_location(
            crawl_manifest=crawl_manifest,
            crawl_state_manifest=crawl_state_manifest,
        )
        deadline.checkpoint("raw_inventory_read")
        raw_inventory = self._raw_inventory_reader.read(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            checkpoint=deadline.checkpoint,
        )
        deadline.checkpoint("curated_inventory_read")
        curated_inventory = self._curated_inventory_reader.read(
            checkpoint=deadline.checkpoint,
        )
        deadline.checkpoint("training_inventory_read")
        training_inventory = self._training_inventory_reader.read_training(
            checkpoint=deadline.checkpoint,
        )
        deadline.checkpoint("augmented_inventory_read")
        augmented_inventory = self._training_inventory_reader.read_augmented(
            checkpoint=deadline.checkpoint,
        )

        preprocessing_manifest, preprocessing_manifest_error = (
            self._read_manifest(
                path=self._artifact_path_registry.preprocessing_manifest_path(),
                parser=PreprocessingManifest.from_payload,
                checkpoint=deadline.checkpoint,
                stage="preprocessing_manifest_read",
            )
        )
        augmentation_manifest, augmentation_manifest_error = (
            self._read_manifest(
                path=self._artifact_path_registry.augmentation_manifest_path(),
                parser=AugmentationManifest.from_payload,
                checkpoint=deadline.checkpoint,
                stage="augmentation_manifest_read",
            )
        )
        training_manifest, training_manifest_error = self._read_manifest(
            path=self._artifact_path_registry.training_manifest_path(),
            parser=TrainingManifest.from_payload,
            checkpoint=deadline.checkpoint,
            stage="training_manifest_read",
        )

        deadline.checkpoint("crawl_validation")
        crawl_result = self._crawl_result(
            manifest=crawl_manifest,
            manifest_error=crawl_manifest_error or crawl_state_error,
            crawl_state_manifest=crawl_state_manifest,
            inventory=raw_inventory,
        )

        current_crawl_manifest_hash = self._file_hash(
            path=self._artifact_path_registry.crawl_manifest_path(),
            checkpoint=deadline.checkpoint,
        )
        deadline.checkpoint("preprocessing_validation")
        preprocessing_result = self._preprocessing_result(
            manifest=preprocessing_manifest,
            manifest_error=preprocessing_manifest_error,
            curated_inventory=curated_inventory,
            training_inventory=training_inventory,
            current_crawl_manifest_hash=current_crawl_manifest_hash,
        )

        current_preprocessing_manifest_hash = self._file_hash(
            path=self._artifact_path_registry.preprocessing_manifest_path(),
            checkpoint=deadline.checkpoint,
        )
        deadline.checkpoint("augmentation_validation")
        augmentation_result = self._augmentation_result(
            manifest=augmentation_manifest,
            manifest_error=augmentation_manifest_error,
            augmented_inventory=augmented_inventory,
            current_preprocessing_manifest_hash=current_preprocessing_manifest_hash,
        )

        deadline.checkpoint("training_input_selection")
        selected_training, selection_error = self._select_training_input(
            training_inventory=training_inventory,
            augmented_inventory=augmented_inventory,
            augmentation_is_valid=augmentation_result.is_valid,
            checkpoint=deadline.checkpoint,
        )

        deadline.checkpoint("training_validation")
        training_result = self._training_result(
            manifest=training_manifest,
            manifest_error=training_manifest_error or selection_error,
            selected_training=selected_training,
        )

        deadline.checkpoint("coverage_calculation")
        coverage_gaps = self._coverage_gaps_resolver(crawl_result.details)

        deadline.checkpoint("decision")
        plan = decide_workflow_action(
            crawl=crawl_result,
            preprocessing=preprocessing_result,
            augmentation=augmentation_result,
            training=training_result,
            ordered_actions=self._ordered_actions,
            optional_actions=self._optional_actions,
            seed_url_count=self._seed_url_count,
            require_seed_urls=self._require_seed_urls,
            augmentation_enabled=self._augmentation_enabled,
            training_input_mode_is_augmented_required=(
                self._training_input_mode.value == "augmented_required"
            ),
            raw_run_directory=raw_inventory.directory,
            raw_records_manifest_path=raw_inventory.records_path,
            training_snapshot_id=selected_training.snapshot_id,
            training_root=selected_training.dataset_root,
            dataset_manifest_hash=selected_training.dataset_manifest_hash,
            coverage_gaps=coverage_gaps,
        )

        self._logger.info(
            "workflow_state_checked",
            action=plan.action.value,
            reason=plan.reason.value,
            crawl_valid=crawl_result.is_valid,
            preprocessing_valid=preprocessing_result.is_valid,
            augmentation_valid=augmentation_result.is_valid,
            training_valid=training_result.is_valid,
        )
        deadline.checkpoint("completed")
        return plan

    def _crawl_result(
        self,
        *,
        manifest: CrawlManifest | None,
        manifest_error: str | None,
        crawl_state_manifest: CrawlStateManifest | None,
        inventory: RawInventory,
    ) -> ValidationResult:
        if manifest_error is not None:
            return self._invalid_manifest_result(
                reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
                error=manifest_error,
            )
        return self._crawl_validator.validate(
            manifest=manifest,
            crawl_state_manifest=crawl_state_manifest,
            inventory=inventory,
            current_source_registry_hash=self._fingerprints.source_registry,
            current_crawl_settings_hash=self._fingerprints.crawl,
        )

    def _preprocessing_result(
        self,
        *,
        manifest: PreprocessingManifest | None,
        manifest_error: str | None,
        curated_inventory: CuratedInventory,
        training_inventory: TrainingInventory,
        current_crawl_manifest_hash: str | None,
    ) -> ValidationResult:
        if manifest_error is not None:
            return self._invalid_manifest_result(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                error=manifest_error,
            )
        return self._preprocessing_validator.validate(
            manifest=manifest,
            curated_inventory=curated_inventory,
            training_inventory=training_inventory,
            current_crawl_manifest_hash=current_crawl_manifest_hash,
            current_preprocessing_settings_hash=self._fingerprints.preprocessing,
            current_normalization_settings_hash=self._fingerprints.normalization,
            current_deduplication_settings_hash=self._fingerprints.deduplication,
            current_splitting_settings_hash=self._fingerprints.splitting,
            current_validation_settings_hash=self._fingerprints.validation,
        )

    def _augmentation_result(
        self,
        *,
        manifest: AugmentationManifest | None,
        manifest_error: str | None,
        augmented_inventory: TrainingInventory,
        current_preprocessing_manifest_hash: str | None,
    ) -> ValidationResult:
        if manifest_error is not None:
            return self._invalid_manifest_result(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
                error=manifest_error,
            )
        return self._augmentation_validator.validate(
            manifest=manifest,
            augmented_inventory=augmented_inventory,
            current_preprocessing_manifest_hash=current_preprocessing_manifest_hash,
            current_augmentation_settings_hash=self._fingerprints.augmentation,
            current_augmentation_strategy_hash=(
                self._fingerprints.augmentation_strategy
            ),
            augmentation_enabled=self._augmentation_enabled,
        )

    def _training_result(
        self,
        *,
        manifest: TrainingManifest | None,
        manifest_error: str | None,
        selected_training: SelectedTrainingInput,
    ) -> ValidationResult:
        if manifest_error is not None:
            return self._invalid_manifest_result(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                error=manifest_error,
            )
        return self._training_validator.validate(
            manifest=manifest,
            training_input_mode=self._training_input_mode,
            current_dataset_manifest_hash=selected_training.dataset_manifest_hash,
            current_checkpoint_path=(
                None if manifest is None else manifest.checkpoint_path
            ),
            current_metrics_path=(
                None if manifest is None else manifest.metrics_path
            ),
            current_training_config_fingerprint=self._fingerprints.training,
            current_model_config_fingerprint=self._fingerprints.model,
            current_modality_counts=selected_training.modality_counts,
            current_task_counts=selected_training.task_counts,
            current_sample_count=selected_training.sample_count,
        )

    def _select_training_input(
        self,
        *,
        training_inventory: TrainingInventory,
        augmented_inventory: TrainingInventory,
        augmentation_is_valid: bool,
        checkpoint: Callable[[str], None],
    ) -> tuple[SelectedTrainingInput, str | None]:
        try:
            return (
                select_training_input(
                    training_inventory=training_inventory,
                    augmented_inventory=augmented_inventory,
                    training_input_mode=self._training_input_mode,
                    augmentation_enabled=self._augmentation_enabled,
                    augmentation_is_valid=augmentation_is_valid,
                    file_fingerprint_calculator=(
                        self._file_fingerprint_calculator
                    ),
                    checkpoint=checkpoint,
                ),
                None,
            )
        except OSError as error:
            return (
                SelectedTrainingInput(
                    snapshot_id=None,
                    dataset_root=None,
                    dataset_manifest_hash=None,
                    sample_count=0,
                    modality_counts={},
                    task_counts={},
                ),
                "selected training dataset could not be read: "
                f"{type(error).__name__}",
            )

    @staticmethod
    def _raw_inventory_location(
        *,
        crawl_manifest: CrawlManifest | None,
        crawl_state_manifest: CrawlStateManifest | None,
    ) -> tuple[Path | None, Path | None]:
        if crawl_manifest is not None:
            return (
                crawl_manifest.raw_run_directory,
                crawl_manifest.run_summary_path,
            )
        if crawl_state_manifest is not None:
            return (
                crawl_state_manifest.raw_run_directory,
                crawl_state_manifest.run_summary_path,
            )
        return None, None

    def _file_hash(
        self,
        *,
        path: Path,
        checkpoint: Callable[[str], None],
    ) -> str | None:
        if not path.is_file():
            return None
        try:
            return self._file_fingerprint_calculator.calculate(
                path=path,
                checkpoint=checkpoint,
            )
        except OSError:
            return None

    @staticmethod
    def _invalid_manifest_result(
        *,
        reason: WorkflowDecisionReason,
        error: str,
    ) -> ValidationResult:
        return ValidationResult.invalid(
            reason=reason,
            details=(f"manifest is unreadable or invalid: {error}",),
        )

    @staticmethod
    def _read_manifest(
        *,
        path: Path,
        parser: Callable[[dict[str, object]], _Manifest],
        checkpoint: Callable[[str], None],
        stage: str,
    ) -> tuple[_Manifest | None, str | None]:
        checkpoint(stage)
        if not path.is_file():
            return None, None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest root must be a JSON object")
            schema_version = payload.get("manifest_schema_version")
            resolved_schema_version = (
                schema_version if isinstance(schema_version, str) else None
            )
            if not is_supported_workflow_manifest_schema_version(
                resolved_schema_version
            ):
                raise ValueError(
                    schema_version_error(
                        artifact="workflow manifest",
                        from_version=resolved_schema_version,
                        supported=(
                            SUPPORTED_WORKFLOW_MANIFEST_SCHEMA_VERSIONS
                        ),
                    )
                )
            return parser(payload), None
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            return None, type(error).__name__
