"""Training workflow composition: value selection, capability binding, wiring."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from config.environment.runtime_environment import container_image_digest
from config.path_resolution.project_paths import ProjectPaths
from config.releases.release_requirements import (
    release_requirements_from_settings,
)
from evaluator.reproducibility import (
    evaluate_reproducibility,
    write_reproducibility_report,
    write_run_receipts_collection,
    write_training_reproducibility_receipt,
)
from logger.factory import ProjectLoggerFactory
from logger.project_logger import ProjectLogger
from orchestration.composition.runtime.workflow_manifest_writers import (
    WorkflowManifestWriters,
)
from orchestration.workflow.training.attempt_runner import (
    TrainingAttemptRunner,
)
from orchestration.workflow.training.campaign_runner import (
    TrainingCampaignRunner,
)
from orchestration.workflow.training.phase_runner import TrainPhaseRunner
from orchestration.workflow.training.stage_executor import (
    TrainingStageExecutor,
)
from release.acceptance_evaluator import evaluate_training_release
from shared.runtime_primitives import Clock, IdGenerator
from training.runtime.checkpoint.contract import CheckpointContract
from training.runtime.checkpoint.io import safe_torch_load
from training.runtime.job_status.persistence import (
    AtomicTrainingJobStatusWriter,
)
from training.runtime.job_status.store import TrainingJobStatusStore
from training.runtime.trainer import evaluate_selected_checkpoint

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from config.settings.root import Settings
    from training.runtime.trainer import MultimodalTrainer


class SnapshotTrainingRuntimeFactory:
    """Factory that creates training runtimes bound to one snapshot."""

    def __init__(
        self,
        *,
        model_settings: "ModelSettings",
        training_settings: "TrainingSettings",
        manifest_filename: str,
        project_paths: ProjectPaths,
        logger_factory: ProjectLoggerFactory,
        clock: Clock,
        checkpoint_contract: CheckpointContract | None,
    ) -> None:
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._manifest_filename = manifest_filename
        self._project_paths = project_paths
        self._logger_factory = logger_factory
        self._clock = clock
        self._checkpoint_contract = checkpoint_contract

    def create(
        self,
        *,
        training_root: Path,
        seed: int | None,
        deterministic: bool | None,
    ) -> tuple["MultimodalTrainer", "TrainingSettings"]:
        from orchestration.composition.runtime.training import (
            build_snapshot_trainer,
        )

        return build_snapshot_trainer(
            model_settings=self._model_settings,
            training_settings=self._training_settings,
            training_root=training_root,
            manifest_filename=self._manifest_filename,
            project_paths=self._project_paths,
            logger_factory=self._logger_factory,
            generated_at=self._clock.now,
            seed=seed,
            deterministic=deterministic,
            checkpoint_contract=self._checkpoint_contract,
        )


def build_training_workflow(
    *,
    settings: "Settings",
    logger: ProjectLogger,
    logger_factory: ProjectLoggerFactory,
    manifest_writers: WorkflowManifestWriters,
    run_blocking: Callable[..., Awaitable[Any]],
    clock: Clock,
    id_generator: IdGenerator,
    checkpoint_contract: CheckpointContract | None = None,
) -> TrainPhaseRunner:
    """Build the complete training workflow graph with pure DI."""
    project_paths = ProjectPaths(project_root=Path(settings.paths.root))

    status_writer = AtomicTrainingJobStatusWriter(
        root=project_paths.resolve(
            Path(settings.datasets.paths.training_checkpoint_directory)
            / "jobs"
        ),
        generate_id=id_generator.generate,
        replace_retry_attempts=settings.training.job_status_replace_retry_attempts,
        replace_retry_delay_seconds=settings.training.job_status_replace_retry_delay_seconds,
    )
    status_store = TrainingJobStatusStore(
        now=clock.now,
        writer=status_writer,
    )

    stage_executor = TrainingStageExecutor(
        status_store=status_store,
        logger=logger,
    )

    runtime_factory = SnapshotTrainingRuntimeFactory(
        model_settings=settings.multimodal,
        training_settings=settings.training,
        manifest_filename=settings.datasets.paths.dataset_manifest_filename,
        project_paths=project_paths,
        logger_factory=logger_factory,
        clock=clock,
        checkpoint_contract=checkpoint_contract,
    )

    attempt_runner = TrainingAttemptRunner(
        logger=logger,
        run_blocking=run_blocking,
        status_store=status_store,
        stage_executor=stage_executor,
        runtime_factory=runtime_factory,
        checkpoint_filename=(
            settings.datasets.paths.training_checkpoint_filename
        ),
        checkpoint_directory=(
            settings.datasets.paths.training_checkpoint_directory
        ),
        model_settings_payload=dict(
            settings.multimodal.model_dump(mode="json")
        ),
        container_digest=container_image_digest(),
        checkpoint_loader=partial(run_blocking, safe_torch_load),
        receipt_writer=partial(
            run_blocking,
            write_training_reproducibility_receipt,
        ),
        evaluator=partial(run_blocking, evaluate_selected_checkpoint),
    )

    candidate_base = (
        project_paths.artifacts / "candidates"
        if settings.training.release_stage == "candidate"
        else None
    )

    campaign_runner = TrainingCampaignRunner(
        logger=logger,
        id_generator=id_generator,
        attempt_runner=attempt_runner,
        status_store=status_store,
        stage_executor=stage_executor,
        candidate_base=candidate_base,
        training_metrics_filename=(
            settings.datasets.paths.training_metrics_filename
        ),
        reproducibility_evaluator=partial(
            run_blocking,
            evaluate_reproducibility,
        ),
        reproducibility_report_writer=partial(
            run_blocking,
            write_reproducibility_report,
        ),
        run_receipts_writer=partial(
            run_blocking,
            write_run_receipts_collection,
        ),
        metrics_writer=partial(
            run_blocking,
            manifest_writers.training.write_training_metrics,
        ),
        manifest_writer=partial(
            run_blocking,
            manifest_writers.training.write_training_manifests,
        ),
        acceptance_evaluator=partial(
            run_blocking,
            evaluate_training_release,
            release_stage=settings.training.release_stage,
            run_mode=settings.training.run_mode,
            settings=settings.datasets.training.dataset_validator,
        ),
    )

    return TrainPhaseRunner(
        campaign_runner=campaign_runner,
        release_requirements=release_requirements_from_settings(
            settings=settings
        ),
    )
