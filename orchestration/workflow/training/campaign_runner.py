"""Training campaign orchestration: attempts, reproducibility, acceptance, manifests."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.releases.release_requirements import (
    ReleaseRequirements,
    ReproducibilityRequirements,
)
from evaluator.reproducibility import TrainingRunReceipt
from logger.project_logger import ProjectLogger
from orchestration.workflow.phase import PhaseOutcome, PhaseStatus
from orchestration.workflow.training.attempt_runner import (
    AttemptOutcome,
    TrainingArtifactPersistenceError,
    TrainingAttemptRunner,
    TrainingStatusPersistenceError,
)
from orchestration.workflow.training.protocols import (
    ReproducibilityEvaluator,
    ReproducibilityReportWriter,
    RunReceiptsCollectionWriter,
    TrainingAcceptanceEvaluator,
    TrainingManifestWriter,
    TrainingMetricsWriter,
)
from orchestration.workflow.training.stage_executor import (
    TrainingStageExecutor,
)
from release.acceptance_result import TrainingAcceptanceResult
from shared.runtime_primitives import IdGenerator
from training.runtime.job_status.models import (
    TrainingCampaignIdentity,
    TrainingJobIdentity,
    TrainingOperationStage,
)
from training.runtime.job_status.persistence import TrainingJobStatusError
from training.runtime.job_status.store import TrainingJobStatusStore

if TYPE_CHECKING:
    from evaluator.results import EvaluationResult


@dataclass(frozen=True, slots=True)
class CampaignInputs:
    """Resolved inputs for a training campaign."""

    snapshot_id: str
    dataset_manifest_hash: str
    training_root: Path
    seeds: tuple[int | None, ...]
    policy: ReproducibilityRequirements | None
    release_requirements: ReleaseRequirements | None
    deterministic_override: bool | None


@dataclass(frozen=True, slots=True)
class CampaignSeedRunsResult:
    primary_outcome: AttemptOutcome


@dataclass(frozen=True, slots=True)
class CampaignReproducibilityResult:
    primary_outcome: AttemptOutcome


class TrainingCampaignRunner:
    """Orchestrate a complete training campaign across multiple seeds.

    All dependencies are injected as exact values and bound capabilities;
    this object never sees the root settings tree.
    """

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        id_generator: IdGenerator,
        attempt_runner: TrainingAttemptRunner,
        status_store: TrainingJobStatusStore,
        stage_executor: TrainingStageExecutor,
        candidate_base: Path | None,
        training_metrics_filename: str,
        reproducibility_evaluator: ReproducibilityEvaluator,
        reproducibility_report_writer: ReproducibilityReportWriter,
        run_receipts_writer: RunReceiptsCollectionWriter,
        manifest_writer: TrainingManifestWriter,
        metrics_writer: TrainingMetricsWriter,
        acceptance_evaluator: TrainingAcceptanceEvaluator,
    ) -> None:
        self._logger = logger
        self._id_generator = id_generator
        self._attempt_runner = attempt_runner
        self._status_store = status_store
        self._stage_executor = stage_executor
        self._candidate_base = candidate_base
        self._training_metrics_filename = training_metrics_filename
        self._reproducibility_evaluator = reproducibility_evaluator
        self._reproducibility_report_writer = reproducibility_report_writer
        self._run_receipts_writer = run_receipts_writer
        self._manifest_writer = manifest_writer
        self._metrics_writer = metrics_writer
        self._acceptance_evaluator = acceptance_evaluator

    async def run(
        self,
        inputs: CampaignInputs,
    ) -> PhaseOutcome:
        """Execute a complete training campaign."""
        campaign_identity = TrainingCampaignIdentity(
            snapshot_id=inputs.snapshot_id,
            campaign_id=self._id_generator.generate(),
        )

        cancellation = threading.Event()

        # Candidate root only for candidate releases
        candidate_root = (
            self._candidate_base / campaign_identity.campaign_id
            if self._candidate_base is not None
            else None
        )

        receipts: list[TrainingRunReceipt] = []
        attempt_identities: list[TrainingJobIdentity] = []
        primary_outcome: AttemptOutcome | None = None

        try:
            # Campaign started
            self._status_store.write_campaign_started(
                identity=campaign_identity,
                training_root=inputs.training_root,
                dataset_manifest_hash=inputs.dataset_manifest_hash,
            )

            # SEED_RUNS stage
            seed_runs_result = await self._stage_executor.execute(
                identity=campaign_identity,
                training_root=str(inputs.training_root),
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                stage=TrainingOperationStage.SEED_RUNS,
                operation=lambda: self._run_seed_runs(
                    campaign_identity=campaign_identity,
                    inputs=inputs,
                    candidate_root=candidate_root,
                    cancellation=cancellation,
                    receipts=receipts,
                    attempt_identities=attempt_identities,
                ),
            )
            if seed_runs_result.error is not None:
                raise seed_runs_result.error

            seed_runs = seed_runs_result.result
            if not isinstance(seed_runs, CampaignSeedRunsResult):
                raise TypeError(
                    f"Expected CampaignSeedRunsResult, got {type(seed_runs).__name__}"
                )
            primary_outcome = seed_runs.primary_outcome

            # Build required stages for terminal completion
            required_campaign_stages = [
                TrainingOperationStage.SEED_RUNS,
                TrainingOperationStage.ACCEPTANCE,
                TrainingOperationStage.MANIFESTS,
            ]
            if inputs.policy is not None:
                required_campaign_stages.insert(
                    1,
                    TrainingOperationStage.REPRODUCIBILITY,
                )

            # REPRODUCIBILITY stage
            if inputs.policy is not None:
                repro_policy = inputs.policy
                repro_primary = primary_outcome
                repro_result = await self._stage_executor.execute(
                    identity=campaign_identity,
                    training_root=str(inputs.training_root),
                    dataset_manifest_hash=inputs.dataset_manifest_hash,
                    stage=TrainingOperationStage.REPRODUCIBILITY,
                    operation=lambda: self._run_reproducibility_stage(
                        primary_outcome=repro_primary,
                        receipts=receipts,
                        policy=repro_policy,
                    ),
                )
                if repro_result.error is not None:
                    raise repro_result.error

                repro = repro_result.result
                if not isinstance(repro, CampaignReproducibilityResult):
                    raise TypeError(
                        f"Expected CampaignReproducibilityResult, "
                        f"got {type(repro).__name__}"
                    )
                primary_outcome = repro.primary_outcome

            # ACCEPTANCE stage
            acceptance_result = await self._stage_executor.execute(
                identity=campaign_identity,
                training_root=str(inputs.training_root),
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                stage=TrainingOperationStage.ACCEPTANCE,
                operation=lambda: self._run_acceptance_stage(
                    primary_outcome=primary_outcome,
                    training_root=inputs.training_root,
                    dataset_manifest_hash=inputs.dataset_manifest_hash,
                    release_requirements=inputs.release_requirements,
                ),
            )
            if acceptance_result.error is not None:
                raise acceptance_result.error
            acceptance = acceptance_result.result
            if not isinstance(acceptance, TrainingAcceptanceResult):
                raise TypeError(
                    f"Expected TrainingAcceptanceResult, "
                    f"got {type(acceptance).__name__}"
                )

            # MANIFESTS stage
            manifests_evaluation = primary_outcome.evaluation
            if manifests_evaluation is None:
                raise RuntimeError(
                    "manifest persistence requires the primary evaluation"
                )
            manifests_result = await self._stage_executor.execute(
                identity=campaign_identity,
                training_root=str(inputs.training_root),
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                stage=TrainingOperationStage.MANIFESTS,
                operation=lambda: self._run_manifests_stage(
                    primary_outcome=primary_outcome,
                    evaluation=manifests_evaluation,
                    acceptance=acceptance,
                    training_root=inputs.training_root,
                    dataset_manifest_hash=inputs.dataset_manifest_hash,
                ),
            )
            if manifests_result.error is not None:
                raise manifests_result.error

        except (asyncio.CancelledError, KeyboardInterrupt) as error:
            self._write_cancelled_best_effort(
                identity=campaign_identity,
                training_root=inputs.training_root,
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                error=error,
            )
            raise
        except Exception as error:
            self._write_failed_best_effort(
                identity=campaign_identity,
                training_root=inputs.training_root,
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                error=error,
            )
            raise

        # Terminal campaign completion - outside error handling
        try:
            self._status_store.write_campaign_completed(
                identity=campaign_identity,
                training_root=inputs.training_root,
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                result=primary_outcome.execution,
                required_stages=tuple(required_campaign_stages),
            )
        except TrainingJobStatusError as exc:
            self._logger.critical(
                "training_status_persistence_failed",
                campaign_id=campaign_identity.campaign_id,
                error=str(exc),
            )
            raise TrainingStatusPersistenceError(
                "Training lifecycle status could not be persisted"
            ) from exc

        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    async def _run_seed_runs(
        self,
        *,
        campaign_identity: TrainingCampaignIdentity,
        inputs: CampaignInputs,
        candidate_root: Path | None,
        cancellation: threading.Event,
        receipts: list[TrainingRunReceipt],
        attempt_identities: list[TrainingJobIdentity],
    ) -> CampaignSeedRunsResult:
        """Execute all seed runs in the campaign."""
        # Create all attempt identities upfront
        for _seed in inputs.seeds:
            identity = TrainingJobIdentity(
                snapshot_id=inputs.snapshot_id,
                attempt_id=self._id_generator.generate(),
            )
            attempt_identities.append(identity)

        attempt_ids = tuple(
            identity.attempt_id for identity in attempt_identities
        )
        self._status_store.write_campaign_attempts(
            identity=campaign_identity,
            training_root=inputs.training_root,
            dataset_manifest_hash=inputs.dataset_manifest_hash,
            attempt_ids=attempt_ids,
            primary_attempt_id=attempt_ids[0],
        )

        primary_outcome: AttemptOutcome | None = None

        for index, seed in enumerate(inputs.seeds):
            is_primary = index == 0
            identity = attempt_identities[index]

            outcome = await self._attempt_runner.run(
                dataset_manifest_hash=inputs.dataset_manifest_hash,
                training_root=inputs.training_root,
                candidate_root=candidate_root,
                seed=seed,
                is_primary=is_primary,
                cancel_event=cancellation,
                identity=identity,
                deterministic=inputs.deterministic_override,
                snapshot_id=inputs.snapshot_id,
            )
            if is_primary:
                primary_outcome = outcome
            if outcome.receipt is not None:
                receipts.append(outcome.receipt)

        if primary_outcome is None:
            raise RuntimeError("training campaign produced no primary run")

        return CampaignSeedRunsResult(primary_outcome=primary_outcome)

    async def _run_reproducibility_stage(
        self,
        *,
        primary_outcome: AttemptOutcome,
        receipts: list[TrainingRunReceipt],
        policy: ReproducibilityRequirements,
    ) -> CampaignReproducibilityResult:
        evaluation_directory = (
            primary_outcome.execution.artifacts.evaluation_directory
        )

        report = await self._reproducibility_evaluator(
            receipts=receipts,
            policy=policy,
            release_requirements_id=policy.policy_id,
        )
        report_path = evaluation_directory / "reproducibility_report.json"
        await self._reproducibility_report_writer(
            path=report_path,
            report=report,
        )
        await self._run_receipts_writer(
            path=evaluation_directory / "run_receipts.json",
            receipts=receipts,
        )

        updated_evaluation = None
        if primary_outcome.evaluation is not None:
            updated_evaluation = (
                primary_outcome.evaluation.with_reproducibility_report(
                    report_path
                )
            )

        updated_outcome = AttemptOutcome(
            identity=primary_outcome.identity,
            execution=primary_outcome.execution,
            receipt=primary_outcome.receipt,
            evaluation=updated_evaluation,
        )

        return CampaignReproducibilityResult(primary_outcome=updated_outcome)

    async def _run_acceptance_stage(
        self,
        *,
        primary_outcome: AttemptOutcome,
        training_root: Path,
        dataset_manifest_hash: str,
        release_requirements: ReleaseRequirements | None,
    ) -> TrainingAcceptanceResult:
        if primary_outcome.evaluation is None:
            raise RuntimeError(
                "release acceptance requires a primary evaluation"
            )

        try:
            await self._metrics_writer(
                input_dataset_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                training_result=primary_outcome.execution,
                evaluation_result=primary_outcome.evaluation,
            )
        except Exception as error:
            raise TrainingArtifactPersistenceError(
                "Training succeeded, but metrics persistence failed"
            ) from error

        metrics_path = (
            primary_outcome.execution.artifacts.checkpoint_path.parent
            / self._training_metrics_filename
        )

        return await self._acceptance_evaluator(
            training_result=primary_outcome.execution,
            evaluation_result=primary_outcome.evaluation,
            input_dataset_root=training_root,
            metrics_path=metrics_path,
            release_requirements=release_requirements,
        )

    async def _run_manifests_stage(
        self,
        *,
        primary_outcome: AttemptOutcome,
        evaluation: EvaluationResult,
        acceptance: TrainingAcceptanceResult,
        training_root: Path,
        dataset_manifest_hash: str,
    ) -> None:
        try:
            await self._manifest_writer(
                input_dataset_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                training_result=primary_outcome.execution,
                evaluation_result=evaluation,
                acceptance_result=acceptance,
            )
        except Exception as error:
            raise TrainingArtifactPersistenceError(
                "Training succeeded, but manifest persistence failed"
            ) from error

    def _write_failed_best_effort(
        self,
        *,
        identity: TrainingCampaignIdentity,
        training_root: Path,
        dataset_manifest_hash: str,
        error: BaseException,
    ) -> None:
        try:
            self._status_store.write_stage_failed(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                result=None,
                error=error,
            )
        except Exception as write_error:
            self._logger.error(
                "train_status_write_failed_during_error",
                primary_error=str(error),
                write_error=str(write_error),
                campaign_id=identity.campaign_id,
            )

    def _write_cancelled_best_effort(
        self,
        *,
        identity: TrainingCampaignIdentity,
        training_root: Path,
        dataset_manifest_hash: str,
        error: BaseException,
    ) -> None:
        try:
            self._status_store.write_stage_cancelled(
                identity=identity,
                training_root=training_root,
                dataset_manifest_hash=dataset_manifest_hash,
                result=None,
                error=error,
            )
        except Exception as write_error:
            self._logger.error(
                "train_status_write_failed_during_cancel",
                primary_error=str(error),
                write_error=str(write_error),
                campaign_id=identity.campaign_id,
            )
