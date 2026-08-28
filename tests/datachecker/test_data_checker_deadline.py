"""Deadline contracts for the concrete DataChecker execution path."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from config.collection.training_input_gate import TrainingInputMode
from datachecker.data_checker import (
    DataChecker,
    DataCheckerTimeoutError,
    WorkflowFingerprints,
    _DataCheckerDeadline,
)
from datachecker.inventory.curated_snapshot_inventory import CuratedInventory
from datachecker.inventory.raw_run_inventory import RawInventory
from datachecker.inventory.training_snapshot_inventory import TrainingInventory
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowAction,
    WorkflowDecisionReason,
)


class _ArtifactPathRegistry:
    """Minimal concrete registry surface used by DataChecker."""

    def crawl_manifest_path(self) -> Path:
        return Path("missing-crawl-manifest.json")

    def crawl_state_manifest_path(self) -> Path:
        return Path("missing-crawl-state-manifest.json")

    def preprocessing_manifest_path(self) -> Path:
        return Path("missing-preprocessing-manifest.json")

    def augmentation_manifest_path(self) -> Path:
        return Path("missing-augmentation-manifest.json")

    def training_manifest_path(self) -> Path:
        return Path("missing-training-manifest.json")


class _RawInventoryReader:
    def read(
        self,
        *,
        raw_run_directory: Path | None,
        run_summary_path: Path | None,
        checkpoint: Callable[[str], None],
    ) -> RawInventory:
        del raw_run_directory, run_summary_path
        checkpoint("raw_reader")
        return RawInventory(
            directory=None,
            summary_path=None,
            records_path=None,
            errors_path=None,
            fingerprint=None,
            file_count=0,
            fetched_url_count=0,
            failed_url_count=0,
            modality_counts={},
            started_at=None,
            completed_at=None,
            status=None,
            final=False,
            schema_valid=False,
        )


class _CuratedInventoryReader:
    def read(
        self,
        *,
        checkpoint: Callable[[str], None],
    ) -> CuratedInventory:
        checkpoint("curated_reader")
        return CuratedInventory(
            directory=None,
            manifest_path=None,
            fingerprint=None,
            document_count=0,
            chunk_count=0,
            image_count=0,
            audio_count=0,
            video_count=0,
            alignment_count=0,
            rejected_document_count=0,
            rejected_image_count=0,
            rejected_audio_count=0,
            rejected_video_count=0,
            image_coverage={},
            audio_coverage={},
            video_coverage={},
            schema_valid=False,
        )


class _TrainingInventoryReader:
    def read_training(
        self,
        *,
        checkpoint: Callable[[str], None],
    ) -> TrainingInventory:
        checkpoint("training_reader")
        return self._empty()

    def read_augmented(
        self,
        *,
        checkpoint: Callable[[str], None],
    ) -> TrainingInventory:
        checkpoint("augmented_reader")
        return self._empty()

    @staticmethod
    def _empty() -> TrainingInventory:
        return TrainingInventory(
            directory=None,
            manifest_path=None,
            stats_path=None,
            snapshot_id=None,
            fingerprint=None,
            sample_count=0,
            modality_counts={},
            task_counts={},
            variants_by_modality={},
            variants_by_operation={},
            rejections_by_modality={},
            media_outputs={},
            quality_checks_passed=False,
            rejected_augmented_count=0,
            schema_valid=False,
        )


class _Validator:
    def validate(self, **_kwargs: object) -> ValidationResult:
        return ValidationResult.valid(
            reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        )


class _FileFingerprintCalculator:
    def calculate(self, **_kwargs: object) -> str:
        return "fingerprint"


class _Logger:
    def info(self, _event: str, **_fields: object) -> None:
        return None


def _checker() -> DataChecker:
    return DataChecker(
        raw_inventory_reader=_RawInventoryReader(),
        curated_inventory_reader=_CuratedInventoryReader(),
        training_inventory_reader=_TrainingInventoryReader(),
        crawl_validator=_Validator(),
        preprocessing_validator=_Validator(),
        augmentation_validator=_Validator(),
        training_validator=_Validator(),
        artifact_path_registry=_ArtifactPathRegistry(),  # type: ignore[arg-type]
        file_fingerprint_calculator=_FileFingerprintCalculator(),  # type: ignore[arg-type]
        fingerprints=WorkflowFingerprints(
            source_registry="source",
            crawl="crawl",
            preprocessing="preprocessing",
            normalization="normalization",
            deduplication="deduplication",
            splitting="splitting",
            validation="validation",
            augmentation="augmentation",
            augmentation_strategy="augmentation-strategy",
            training="training",
            model="model",
        ),
        coverage_gaps_resolver=lambda _details: {},
        augmentation_enabled=False,
        training_input_mode=TrainingInputMode.PREPROCESSED_ONLY,
        ordered_actions=(
            WorkflowAction.CRAWL,
            WorkflowAction.PREPROCESS,
            WorkflowAction.TRAIN,
        ),
        optional_actions=(WorkflowAction.AUGMENT,),
        require_seed_urls=False,
        seed_url_count=0,
        logger=_Logger(),  # type: ignore[arg-type]
        monotonic_seconds=lambda: 0.0,
    )


def test_check_uses_one_deadline_across_concrete_reader_and_decision_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[str] = []

    def checkpoint(_deadline: _DataCheckerDeadline, stage: str) -> None:
        stages.append(stage)

    monkeypatch.setattr(_DataCheckerDeadline, "checkpoint", checkpoint)

    plan = _checker().check(timeout_seconds=60.0)

    assert plan.action is WorkflowAction.NOOP
    assert stages == [
        "crawl_manifest_read",
        "crawl_state_manifest_read",
        "raw_inventory_read",
        "raw_reader",
        "curated_inventory_read",
        "curated_reader",
        "training_inventory_read",
        "training_reader",
        "augmented_inventory_read",
        "augmented_reader",
        "preprocessing_manifest_read",
        "augmentation_manifest_read",
        "training_manifest_read",
        "crawl_validation",
        "preprocessing_validation",
        "augmentation_validation",
        "training_input_selection",
        "training_validation",
        "coverage_calculation",
        "decision",
        "completed",
    ]


def test_check_propagates_the_precise_deadline_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def checkpoint(deadline: _DataCheckerDeadline, stage: str) -> None:
        if stage == "preprocessing_validation":
            raise DataCheckerTimeoutError(
                stage=stage,
                timeout_seconds=deadline.timeout_seconds,
                elapsed_seconds=deadline.timeout_seconds,
            )

    monkeypatch.setattr(_DataCheckerDeadline, "checkpoint", checkpoint)

    with pytest.raises(
        DataCheckerTimeoutError,
        match="preprocessing_validation",
    ) as raised:
        _checker().check(timeout_seconds=1.0)

    assert raised.value.stage == "preprocessing_validation"
    assert raised.value.timeout_seconds == 1.0


def test_deadline_rejects_nonpositive_timeouts() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _DataCheckerDeadline(
            timeout_seconds=0.0,
            monotonic_seconds=lambda: 0.0,
        )


def test_deadline_uses_injected_monotonic_source() -> None:
    values = iter((10.0, 12.5))
    deadline = _DataCheckerDeadline(
        timeout_seconds=2.0,
        monotonic_seconds=lambda: next(values),
    )

    with pytest.raises(DataCheckerTimeoutError) as raised:
        deadline.checkpoint("deterministic-stage")

    assert raised.value.stage == "deterministic-stage"
    assert raised.value.elapsed_seconds == 2.5
