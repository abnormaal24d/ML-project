"""Preprocess phase execution for the autonomous data workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from crawler.coverage.gaps import CoverageGapAnalyzer
from crawler.coverage.progress import CoverageProgressTracker
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.snapshots.training_builder import (
    TrainingSnapshotBuildResult,
)
from orchestration.workflow.phase import PhaseOutcome, PhaseStatus, RunBlocking

if TYPE_CHECKING:
    from mmcrawler_datasets.snapshots.validation import (
        TrainingDatasetValidationError,
    )


class RunPreprocessing(Protocol):
    """Bound preprocessing pipeline; static config lives in composition."""

    def __call__(
        self,
        *,
        raw_run_directory: Path,
        raw_records_manifest_path: Path,
        training_snapshot_id: str | None,
    ) -> Awaitable[TrainingSnapshotBuildResult]: ...


PreprocessingManifestWrite = Callable[[], object]


class PreprocessPhaseRunner:
    """Execute preprocessing workflow phases and collect structured multimodal."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        gap_analyzer: CoverageGapAnalyzer,
        progress_tracker: CoverageProgressTracker,
        run_preprocessing: RunPreprocessing,
        write_preprocessing_manifest: PreprocessingManifestWrite,
        run_blocking: RunBlocking,
        io_timeout_seconds: float,
    ) -> None:
        self._logger = logger
        self._gap_analyzer = gap_analyzer
        self._progress_tracker = progress_tracker
        self._run_preprocessing = run_preprocessing
        self._write_preprocessing_manifest = write_preprocessing_manifest
        self._run_blocking = run_blocking
        self._io_timeout_seconds = io_timeout_seconds

    async def run(self, plan: WorkflowExecutionPlan) -> PhaseOutcome:
        from mmcrawler_datasets.snapshots.validation import (
            TrainingDatasetValidationError,
        )

        raw_run_directory = plan.raw_run_directory
        if raw_run_directory is None:
            raise ValueError("workflow plan is missing raw_run_directory")
        raw_records_manifest_path = plan.raw_records_manifest_path
        if raw_records_manifest_path is None:
            raise ValueError(
                "workflow plan is missing raw_records_manifest_path"
            )

        try:
            # Async multimodal preprocessing runs on the event loop; do not
            # wrap this call in run_blocking (which cannot await).
            await asyncio.wait_for(
                self._run_preprocessing(
                    raw_run_directory=raw_run_directory,
                    raw_records_manifest_path=raw_records_manifest_path,
                    training_snapshot_id=plan.training_snapshot_id,
                ),
                timeout=self._io_timeout_seconds,
            )
        except TrainingDatasetValidationError as error:
            return self._handle_validation_error(error)

        await self._run_blocking(
            self._write_preprocessing_manifest,
            timeout_seconds=self._io_timeout_seconds,
        )
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    def _handle_validation_error(
        self,
        error: TrainingDatasetValidationError,
    ) -> PhaseOutcome:
        coverage_gaps = self._gap_analyzer.gaps_from_validation_errors(
            list(error.validation_errors)
        )
        recrawl_plan = WorkflowExecutionPlan(
            action=WorkflowAction.CRAWL,
            reason=WorkflowDecisionReason.COVERAGE_TARGETS_NOT_MET,
            coverage_gaps=coverage_gaps,
            details=(
                "training snapshot validation failed after preprocessing",
                "additional crawl input is required before training can run",
                *error.validation_errors,
            ),
        )
        progress_decision = self._progress_tracker.observe_validation_failure(
            validation_payload=error.validation_payload,
            validation_errors=error.validation_errors,
            validation_report_path=error.validation_report_path,
            training_directory=error.training_directory,
            coverage_gaps=coverage_gaps,
            missing_by_kind=self._gap_analyzer.missing_by_media_kind(
                coverage_gaps
            ),
        )
        if not progress_decision.should_recrawl:
            self._logger.error(
                "data_workflow_training_snapshot_blocked",
                snapshot_id=error.snapshot_id,
                reason=progress_decision.blocked_reason,
                error_type=type(error).__name__,
                error_message=str(error) or None,
                validation_errors=error.validation_errors,
                validation_remediation=error.validation_remediation,
                validation_report_path=str(error.validation_report_path),
                training_directory=str(error.training_directory),
                coverage_gaps=coverage_gaps,
                no_progress_attempts=progress_decision.attempt_count,
                blocked_details=progress_decision.details,
            )
            return PhaseOutcome(status=PhaseStatus.BLOCKED)

        self._logger.error(
            "data_workflow_training_snapshot_not_ready_recrawling",
            snapshot_id=error.snapshot_id,
            reason="validation_failed_more_data_required",
            error_type=type(error).__name__,
            error_message=str(error) or None,
            validation_errors=error.validation_errors,
            validation_remediation=error.validation_remediation,
            validation_report_path=str(error.validation_report_path),
            training_directory=str(error.training_directory),
            coverage_gaps=coverage_gaps,
            no_progress_attempts=progress_decision.attempt_count,
        )
        return PhaseOutcome(
            status=PhaseStatus.RECRAWL_REQUESTED,
            next_plan=recrawl_plan,
        )
