from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from config.load import load_settings
from config.settings.root import Settings
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from evaluator.results import EvaluationResult
from orchestration.workflow.training import (
    TrainingArtifactPersistenceError,
    TrainingCampaignRunner,
    TrainingStatusPersistenceError,
    TrainPhaseRunner,
)
from release.acceptance_result import (
    AcceptanceReport,
    TrainingAcceptanceResult,
)
from release.release_decision import ReleaseDecision
from schemas.release import ReleaseStatus
from training.runtime.job_status.models import TrainingJobIdentity
from training.runtime.job_status.persistence import (
    TrainingJobStatusError,
)
from training.runtime.results import (
    TrainingArtifacts,
    TrainingMetrics,
    TrainingRunIdentity,
    TrainingRunResult,
)


def _training_run_result(tmp_path: Path) -> TrainingRunResult:
    checkpoint_path = (
        tmp_path / "training" / "snapshot-20260728" / "checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    return TrainingRunResult(
        metrics=TrainingMetrics(
            train_loss=0.12,
            validation_loss=0.05,
            test_loss=0.18,
            average_loss=0.1,
            last_epoch_loss=0.05,
            epochs=1,
            batches=1,
            samples=1,
            effective_train_sample_count=1,
        ),
        artifacts=TrainingArtifacts(
            checkpoint_path=checkpoint_path,
            last_checkpoint_path=checkpoint_path,
            export_directory=None,
        ),
        identity=TrainingRunIdentity(model_seed=7),
    )


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        validation_loss=0.05,
        test_loss=None,
        dataset_split_counts={"train": 1, "validation": 1, "test": 0},
        valid=True,
        failure_reasons=(),
    )


def _acceptance_result(tmp_path: Path) -> TrainingAcceptanceResult:
    decision = ReleaseDecision(
        status=ReleaseStatus.PIPELINE_ACCEPTED,
        reasons=(),
    )
    report = AcceptanceReport.build(
        release_stage="pipeline_smoke",
        decision=decision,
        expected_status=ReleaseStatus.PIPELINE_ACCEPTED,
        evidence_paths={},
    )
    return TrainingAcceptanceResult(
        decision=decision,
        acceptance_report=report,
        acceptance_report_path=tmp_path / "acceptance_report.json",
    )


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def critical(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


class _IdGenerator:
    def __init__(self) -> None:
        self._counter = 0

    def generate(self) -> str:
        self._counter += 1
        if self._counter == 1:
            return "attempt-fixed"
        return f"tmp-{self._counter}"


def _settings(
    tmp_path: Path,
    *,
    release_stage: str = "pipeline_smoke",
) -> Settings:
    settings = load_settings(
        "test",
        project_root=tmp_path,
        environment="test",
        fingerprint=False,
    )
    training = settings.training.model_copy(
        update={
            "release_stage": release_stage,
            "run_mode": "full",
            "export_directory": "models",
            "deterministic": False,
        }
    )
    paths = settings.datasets.paths.model_copy(
        update={
            "training_checkpoint_directory": "checkpoints",
            "training_checkpoint_filename": "model_checkpoint.pt",
            "training_metrics_filename": "training_metrics.json",
        }
    )
    return settings.model_copy(
        update={
            "training": training,
            "datasets": settings.datasets.model_copy(update={"paths": paths}),
        }
    )


def _plan(tmp_path: Path) -> WorkflowExecutionPlan:
    training_root = tmp_path / "training" / "snapshot-20260728"
    training_root.mkdir(parents=True, exist_ok=True)
    return WorkflowExecutionPlan(
        action=WorkflowAction.TRAIN,
        reason=WorkflowDecisionReason.TRAINING_OUTPUT_MISSING,
        training_snapshot_id="snapshot-20260728",
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
    )


def _runner(
    tmp_path: Path,
    *,
    train_error: BaseException | None = None,
    manifest_fail: bool = False,
    release_stage: str = "pipeline_smoke",
) -> tuple[TrainPhaseRunner, _Logger, Path]:
    logger = _Logger()
    execution = _training_run_result(tmp_path)
    acceptance = _acceptance_result(tmp_path)

    async def run_blocking(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("timeout_seconds", None)
        kwargs.pop("cancel", None)
        if func.__name__ == "train_and_collect_results":
            if train_error is not None:
                raise train_error
            return execution
        raise AssertionError(f"unexpected blocking call: {func.__name__}")

    from config.path_resolution.project_paths import ProjectPaths
    from config.releases.release_requirements import (
        release_requirements_from_settings,
    )
    from orchestration.workflow.training.attempt_runner import (
        TrainingAttemptRunner,
    )
    from orchestration.workflow.training.stage_executor import (
        TrainingStageExecutor,
    )
    from training.runtime.job_status.persistence import (
        AtomicTrainingJobStatusWriter,
    )
    from training.runtime.job_status.store import TrainingJobStatusStore

    settings = _settings(tmp_path, release_stage=release_stage)
    jobs_root = tmp_path / "checkpoints" / "jobs"

    status_writer = AtomicTrainingJobStatusWriter(
        root=ProjectPaths(project_root=tmp_path).resolve(
            Path(settings.datasets.paths.training_checkpoint_directory)
            / "jobs"
        ),
        generate_id=_IdGenerator().generate,
        replace_retry_attempts=settings.training.job_status_replace_retry_attempts,
        replace_retry_delay_seconds=settings.training.job_status_replace_retry_delay_seconds,
    )
    status_store = TrainingJobStatusStore(
        now=_Clock().now,
        writer=status_writer,
    )

    stage_executor = TrainingStageExecutor(
        status_store=status_store,
        logger=logger,
    )

    class _FakeRuntimeFactory:
        def create(self, *, training_root, seed, deterministic):  # noqa: ANN001
            return object(), settings.training

    async def fake_checkpoint_loader(checkpoint_path: Path) -> object:
        return {"model_state": {}}

    async def fake_receipt_writer(**_kwargs: object) -> None:
        return None

    async def fake_evaluator(**_kwargs: object) -> EvaluationResult:
        return _evaluation_result()

    attempt_runner = TrainingAttemptRunner(
        logger=logger,
        run_blocking=run_blocking,
        status_store=status_store,
        stage_executor=stage_executor,
        runtime_factory=_FakeRuntimeFactory(),
        checkpoint_filename=(
            settings.datasets.paths.training_checkpoint_filename
        ),
        checkpoint_directory=(
            settings.datasets.paths.training_checkpoint_directory
        ),
        model_settings_payload={},
        container_digest="container-digest",
        checkpoint_loader=fake_checkpoint_loader,
        receipt_writer=fake_receipt_writer,
        evaluator=fake_evaluator,
    )

    async def unreachable_capability(
        *_args: object, **_kwargs: object
    ) -> object:
        raise AssertionError("reproducibility must not run without a policy")

    async def fake_metrics_writer(**_kwargs: object) -> None:
        return None

    async def fake_manifest_writer(**_kwargs: object) -> None:
        if manifest_fail:
            raise OSError("manifest write failed")
        return None

    async def fake_acceptance_evaluator(**_kwargs: object) -> object:
        return acceptance

    candidate_base = (
        ProjectPaths(project_root=tmp_path).artifacts / "candidates"
        if release_stage == "candidate"
        else None
    )

    campaign_runner = TrainingCampaignRunner(
        logger=logger,
        id_generator=_IdGenerator(),
        attempt_runner=attempt_runner,
        status_store=status_store,
        stage_executor=stage_executor,
        candidate_base=candidate_base,
        training_metrics_filename=(
            settings.datasets.paths.training_metrics_filename
        ),
        reproducibility_evaluator=unreachable_capability,
        reproducibility_report_writer=unreachable_capability,
        run_receipts_writer=unreachable_capability,
        manifest_writer=fake_manifest_writer,
        metrics_writer=fake_metrics_writer,
        acceptance_evaluator=fake_acceptance_evaluator,
    )

    runner = TrainPhaseRunner(
        campaign_runner=campaign_runner,
        release_requirements=release_requirements_from_settings(
            settings=settings
        ),
    )

    return runner, logger, jobs_root


def _attempt_payload(jobs_root: Path) -> dict[str, Any]:
    status_paths = list(
        jobs_root.glob("snapshot-20260728/attempts/*/status.json")
    )
    assert len(status_paths) == 1
    return json.loads(status_paths[0].read_text(encoding="utf-8"))


def _campaign_payload(jobs_root: Path) -> dict[str, Any]:
    status_paths = list(
        jobs_root.glob("snapshot-20260728/campaigns/*/status.json")
    )
    assert len(status_paths) == 1
    return json.loads(status_paths[0].read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_success_writes_completed_status(tmp_path: Path) -> None:
    runner, _logger, jobs_root = _runner(tmp_path)

    outcome = await runner.run(_plan(tmp_path))

    assert outcome.status.value == "succeeded"
    attempt_payload = _attempt_payload(jobs_root)
    campaign_payload = _campaign_payload(jobs_root)

    # Attempt document: completed with training, receipt, evaluation
    assert attempt_payload["lifecycle"]["status"] == "completed"
    assert attempt_payload["lifecycle"]["completed_stages"] == [
        "training",
        "receipt",
        "evaluation",
    ]
    assert attempt_payload["lifecycle"]["status"] == "completed"
    assert attempt_payload["snapshot_id"] == "snapshot-20260728"
    assert attempt_payload["attempt_id"].startswith("tmp-")

    # Promotion is a separate operator action, never an automatic run stage.
    assert campaign_payload["lifecycle"]["status"] == "completed"
    assert campaign_payload["lifecycle"]["completed_stages"] == [
        "seed_runs",
        "acceptance",
        "manifests",
    ]
    assert campaign_payload["lifecycle"]["status"] == "completed"
    assert campaign_payload["snapshot_id"] == "snapshot-20260728"
    assert campaign_payload["attempt_ids"] == [attempt_payload["attempt_id"]]
    assert (
        campaign_payload["primary_attempt_id"] == attempt_payload["attempt_id"]
    )


@pytest.mark.asyncio
async def test_training_exception_writes_failed_and_reraises(
    tmp_path: Path,
) -> None:
    runner, _logger, jobs_root = _runner(
        tmp_path,
        train_error=RuntimeError("cuda died"),
    )

    with pytest.raises(RuntimeError, match="cuda died"):
        await runner.run(_plan(tmp_path))

    payload = _attempt_payload(jobs_root)
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["lifecycle"]["failed_stage"] == "training"
    assert payload["error_type"] == "RuntimeError"
    assert payload["completed_at"] is not None


@pytest.mark.asyncio
async def test_type_error_writes_failed_and_reraises(tmp_path: Path) -> None:
    runner, _logger, jobs_root = _runner(
        tmp_path,
        train_error=TypeError("bad trainer config"),
    )

    with pytest.raises(TypeError, match="bad trainer config"):
        await runner.run(_plan(tmp_path))

    payload = _attempt_payload(jobs_root)
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["error_type"] == "TypeError"


@pytest.mark.asyncio
async def test_cancelled_error_writes_cancelled_and_reraises(
    tmp_path: Path,
) -> None:
    runner, _logger, jobs_root = _runner(
        tmp_path,
        train_error=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run(_plan(tmp_path))

    payload = _attempt_payload(jobs_root)
    assert payload["lifecycle"]["status"] == "cancelled"
    assert payload["lifecycle"]["cancelled_stage"] == "training"


@pytest.mark.asyncio
async def test_manifest_failure_preserves_training_success(
    tmp_path: Path,
) -> None:
    runner, _logger, jobs_root = _runner(tmp_path, manifest_fail=True)

    with pytest.raises(TrainingArtifactPersistenceError) as raised:
        await runner.run(_plan(tmp_path))

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "manifest write failed"

    # Campaign document fails at MANIFESTS stage
    campaign_payload = _campaign_payload(jobs_root)
    assert campaign_payload["lifecycle"]["status"] == "failed"
    assert campaign_payload["lifecycle"]["failed_stage"] == "manifests"
    assert campaign_payload["lifecycle"]["completed_stages"] == [
        "seed_runs",
        "acceptance",
    ]

    # Attempt document remains completed
    attempt_payload = _attempt_payload(jobs_root)
    assert attempt_payload["lifecycle"]["status"] == "completed"
    assert attempt_payload["lifecycle"]["completed_stages"] == [
        "training",
        "receipt",
        "evaluation",
    ]


@pytest.mark.asyncio
async def test_manifest_failure_status_preserves_root_cause(
    tmp_path: Path,
) -> None:
    runner, _logger, jobs_root = _runner(tmp_path, manifest_fail=True)

    with pytest.raises(TrainingArtifactPersistenceError):
        await runner.run(_plan(tmp_path))

    campaign_payload = _campaign_payload(jobs_root)
    assert campaign_payload["error_type"] == "TrainingArtifactPersistenceError"
    assert (
        campaign_payload["error_message"]
        == "Training succeeded, but manifest persistence failed"
    )
    assert campaign_payload["lifecycle"]["status"] == "failed"
    assert campaign_payload["lifecycle"]["failed_stage"] == "manifests"

    # Attempt document remains completed with training succeeded
    attempt_payload = _attempt_payload(jobs_root)
    assert attempt_payload["lifecycle"]["status"] == "completed"


@pytest.mark.asyncio
async def test_terminal_status_write_failure_is_not_reported_as_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, logger, _jobs_root = _runner(tmp_path)

    def boom_write_campaign_completed(self, **_kwargs: object) -> Path:  # noqa: ARG001
        raise TrainingJobStatusError(
            "Training job status could not be written"
        ) from OSError("disk full")

    monkeypatch.setattr(
        "training.runtime.job_status.store.TrainingJobStatusStore.write_campaign_completed",
        boom_write_campaign_completed,
    )

    with pytest.raises(TrainingStatusPersistenceError) as raised:
        await runner.run(_plan(tmp_path))

    assert isinstance(raised.value.__cause__, TrainingJobStatusError)
    critical = [
        event
        for event, _fields in logger.events
        if "training_status_persistence_failed" in event
    ]
    assert critical
    assert not any(
        event == "train_status_write_failed_during_error"
        for event, _ in logger.events
    )


@pytest.mark.asyncio
async def test_terminal_status_type_error_uses_persistence_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, logger, _jobs_root = _runner(tmp_path)

    def boom_write_campaign_completed(self, **_kwargs: object) -> Path:  # noqa: ARG001
        raise TrainingJobStatusError(
            "Training job status could not be written"
        ) from TypeError("payload serialization broke")

    monkeypatch.setattr(
        "training.runtime.job_status.store.TrainingJobStatusStore.write_campaign_completed",
        boom_write_campaign_completed,
    )

    with pytest.raises(TrainingStatusPersistenceError) as raised:
        await runner.run(_plan(tmp_path))

    assert isinstance(raised.value.__cause__, TrainingJobStatusError)
    assert isinstance(raised.value.__cause__.__cause__, TypeError)
    assert "payload serialization broke" in str(
        raised.value.__cause__.__cause__
    )
    assert any(
        "training_status_persistence_failed" in event
        for event, _ in logger.events
    )


def test_job_identity_paths_are_attempt_scoped() -> None:
    first = TrainingJobIdentity(
        snapshot_id="snapshot-1",
        attempt_id="attempt-a",
    )
    second = TrainingJobIdentity(
        snapshot_id="snapshot-1",
        attempt_id="attempt-b",
    )
    assert first != second


@pytest.mark.asyncio
async def test_candidate_stage_does_not_promote_automatically(
    tmp_path: Path,
) -> None:
    runner, _logger, jobs_root = _runner(
        tmp_path,
        release_stage="candidate",
    )

    outcome = await runner.run(_plan(tmp_path))

    assert outcome.status.value == "succeeded"
    campaign_payload = _campaign_payload(jobs_root)
    assert "promotion" not in campaign_payload["lifecycle"]["completed_stages"]
    assert not (tmp_path / "models" / "current.json").exists()
