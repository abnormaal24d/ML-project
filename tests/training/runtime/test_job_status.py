from __future__ import annotations

import ast
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from training.runtime.job_status.models import (
    TrainingCampaignIdentity,
    TrainingJobIdentity,
    TrainingLifecycleState,
    TrainingLifecycleStatus,
    TrainingOperationStage,
)
from training.runtime.job_status.persistence import (
    AtomicTrainingJobStatusWriter,
    TrainingJobStatusError,
)
from training.runtime.job_status.store import TrainingJobStatusStore
from training.runtime.results import (
    TrainingArtifacts,
    TrainingMetrics,
    TrainingRunIdentity,
    TrainingRunResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _training_run_result(tmp_path: Path) -> TrainingRunResult:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    return TrainingRunResult(
        metrics=TrainingMetrics(
            train_loss=0.3,
            validation_loss=0.3,
            test_loss=0.3,
            average_loss=0.25,
            last_epoch_loss=0.2,
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


def _store(
    tmp_path: Path, *, now: datetime | None = None
) -> TrainingJobStatusStore:
    fixed = now or datetime(2026, 7, 28, 12, 30, tzinfo=UTC)
    writer = AtomicTrainingJobStatusWriter(
        root=tmp_path / "jobs",
        generate_id=lambda: "temporary-id",
        replace_retry_attempts=3,
        replace_retry_delay_seconds=0.05,
    )
    return TrainingJobStatusStore(
        now=lambda: fixed,
        writer=writer,
    )


def _identity(
    *,
    snapshot_id: str = "snapshot-1",
    attempt_id: str = "attempt-a",
) -> TrainingJobIdentity:
    return TrainingJobIdentity(
        snapshot_id=snapshot_id,
        attempt_id=attempt_id,
    )


def _campaign_identity(
    *,
    snapshot_id: str = "snapshot-1",
    campaign_id: str = "campaign-a",
) -> TrainingCampaignIdentity:
    return TrainingCampaignIdentity(
        snapshot_id=snapshot_id,
        campaign_id=campaign_id,
    )


def _active_training(
    store: TrainingJobStatusStore,
    *,
    identity: TrainingJobIdentity,
    tmp_path: Path,
) -> None:
    store.write_started(
        identity=identity,
        training_root=tmp_path / "training" / identity.snapshot_id,
        dataset_manifest_hash="manifest-sha256",
    )
    store.write_stage_started(
        identity=identity,
        training_root=tmp_path / "training" / identity.snapshot_id,
        dataset_manifest_hash="manifest-sha256",
        stage=TrainingOperationStage.TRAINING,
    )


def test_write_started_persists_expected_status(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 7, 28, 12, 30, tzinfo=UTC)
    store = _store(tmp_path, now=fixed_time)
    training_root = tmp_path / "training" / "snapshot-1"
    identity = _identity()

    status_path = store.write_started(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="manifest-sha256",
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["lifecycle"]["status"] == "running"
    assert payload["attempt_id"] == "attempt-a"
    assert payload["snapshot_id"] == "snapshot-1"
    assert payload["training_root"] == training_root.as_posix()
    assert payload["dataset_manifest_hash"] == "manifest-sha256"
    assert payload["started_at"] == fixed_time.isoformat()
    assert payload["updated_at"] == fixed_time.isoformat()
    assert payload["lifecycle"] == {
        "status": "running",
        "current_stage": None,
        "completed_stages": [],
        "failed_stage": None,
        "cancelled_stage": None,
    }
    assert status_path == (
        tmp_path
        / "jobs"
        / "snapshot-1"
        / "attempts"
        / "attempt-a"
        / "status.json"
    )


def test_stage_transitions_project_derived_fields(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 7, 28, 12, 30, tzinfo=UTC)
    store = _store(tmp_path, now=fixed_time)
    training_root = tmp_path / "training" / "snapshot-1"
    identity = _identity()

    store.write_started(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="manifest-sha256",
    )
    started = store.write_stage_started(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="manifest-sha256",
        stage=TrainingOperationStage.TRAINING,
    )
    payload = json.loads(started.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "running"
    assert payload["lifecycle"]["current_stage"] == "training"

    succeeded = store.write_stage_succeeded(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="manifest-sha256",
        result=_training_run_result(tmp_path),
    )
    payload = json.loads(succeeded.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["current_stage"] is None
    assert payload["lifecycle"]["completed_stages"] == ["training"]
    assert payload["evidence"]["average_loss"] == 0.25
    assert (
        payload["evidence"]["checkpoint_path"]
        == (tmp_path / "checkpoint.pt").as_posix()
    )
    assert "completed_at" not in payload


def test_write_attempt_completed_persists_terminal_status(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 7, 28, 13, 0, tzinfo=UTC)
    store = _store(tmp_path, now=fixed_time)
    training_root = tmp_path / "training" / "snapshot-success"
    identity = _identity(
        snapshot_id="snapshot-success", attempt_id="attempt-1"
    )

    _active_training(store, identity=identity, tmp_path=tmp_path)
    store.write_stage_succeeded(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        result=_training_run_result(tmp_path),
    )
    status_path = store.write_attempt_completed(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        result=_training_run_result(tmp_path),
        required_stages=(TrainingOperationStage.TRAINING,),
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "completed"
    assert payload["completed_at"] == fixed_time.isoformat()
    assert payload["lifecycle"]["completed_stages"] == ["training"]
    assert payload["evidence"]["average_loss"] == 0.25


def test_write_stage_failed_records_error_details(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 7, 28, 13, 45, tzinfo=UTC)
    store = _store(tmp_path, now=fixed_time)
    training_root = tmp_path / "training" / "snapshot-failed"
    identity = _identity(snapshot_id="snapshot-failed", attempt_id="attempt-x")

    _active_training(store, identity=identity, tmp_path=tmp_path)
    status_path = store.write_stage_failed(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        result=None,
        error=RuntimeError("training exploded"),
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["lifecycle"]["failed_stage"] == "training"
    assert payload["completed_at"] == fixed_time.isoformat()
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "training exploded"


def test_write_stage_cancelled_records_cancelled_stage(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    training_root = tmp_path / "training" / "snapshot-1"
    identity = _identity()

    _active_training(store, identity=identity, tmp_path=tmp_path)
    status_path = store.write_stage_cancelled(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash=None,
        result=None,
        error=asyncio.CancelledError("user interrupt"),
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "cancelled"
    assert payload["lifecycle"]["cancelled_stage"] == "training"
    assert payload["error_type"] == "CancelledError"


def test_read_lifecycle_round_trips_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()

    store.write_started(
        identity=identity,
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash=None,
    )
    store.write_stage_started(
        identity=identity,
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash=None,
        stage=TrainingOperationStage.RECEIPT,
    )

    state = store.read_lifecycle(identity=identity)
    assert state is not None
    assert state.status is TrainingLifecycleStatus.RUNNING
    assert state.current_stage is TrainingOperationStage.RECEIPT
    assert state.completed_stages == ()


def test_atomic_write_leaves_no_temporary_status_file(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    training_root = tmp_path / "training" / "snapshot-1"
    identity = _identity()

    status_path = store.write_started(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash=None,
    )

    temporary_path = status_path.with_name(
        f"{status_path.name}.temporary-id.tmp"
    )
    assert status_path.is_file()
    assert not temporary_path.exists()


def test_distinct_snapshot_ids_have_distinct_status_paths(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = TrainingJobIdentity(
        snapshot_id="snapshot-1",
        attempt_id="attempt-a",
    )
    second = TrainingJobIdentity(
        snapshot_id="snapshot-2",
        attempt_id="attempt-a",
    )

    assert store.path_for(identity=first) != store.path_for(identity=second)


def test_distinct_attempts_do_not_overwrite_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    training_root = tmp_path / "training" / "shared-name"
    first = TrainingJobIdentity(
        snapshot_id="snapshot-1",
        attempt_id="attempt-a",
    )
    second = TrainingJobIdentity(
        snapshot_id="snapshot-1",
        attempt_id="attempt-b",
    )

    assert store.path_for(identity=first) != store.path_for(identity=second)

    first_path = store.write_started(
        identity=first,
        training_root=training_root,
        dataset_manifest_hash="hash-a",
    )
    second_path = store.write_started(
        identity=second,
        training_root=training_root,
        dataset_manifest_hash="hash-b",
    )

    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()
    assert (
        json.loads(first_path.read_text(encoding="utf-8"))[
            "dataset_manifest_hash"
        ]
        == "hash-a"
    )
    assert (
        json.loads(second_path.read_text(encoding="utf-8"))[
            "dataset_manifest_hash"
        ]
        == "hash-b"
    )


def test_campaign_identity_uses_separate_status_document(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()

    path = store.write_campaign_started(
        identity=campaign,
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash="hash-campaign",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "campaign-a"
    assert "attempt_id" not in payload
    assert (
        payload["training_root"]
        == (tmp_path / "training" / "snapshot-1").as_posix()
    )
    assert path == (
        tmp_path
        / "jobs"
        / "snapshot-1"
        / "campaigns"
        / "campaign-a"
        / "status.json"
    )
    assert store.campaign_path_for(identity=campaign) == path


def test_campaign_completed_requires_seed_runs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()
    training_root = tmp_path / "training" / "snapshot-1"

    store.write_campaign_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )
    store.write_stage_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        stage=TrainingOperationStage.SEED_RUNS,
    )
    store.write_stage_succeeded(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )
    path = store.write_campaign_completed(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        required_stages=(TrainingOperationStage.SEED_RUNS,),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "completed"
    assert payload["lifecycle"]["completed_stages"] == ["seed_runs"]


def test_campaign_completed_rejects_incomplete_seed_runs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()
    training_root = tmp_path / "training" / "snapshot-1"

    store.write_campaign_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )
    store.write_stage_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        stage=TrainingOperationStage.SEED_RUNS,
    )

    with pytest.raises(TrainingJobStatusError, match="could not be written"):
        store.write_campaign_completed(
            identity=campaign,
            training_root=training_root,
            dataset_manifest_hash="hash-campaign",
            required_stages=(TrainingOperationStage.SEED_RUNS,),
        )


def test_campaign_failure_records_error_details(tmp_path: Path) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()
    training_root = tmp_path / "training" / "snapshot-1"

    store.write_campaign_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )
    store.write_stage_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        stage=TrainingOperationStage.SEED_RUNS,
    )
    path = store.write_stage_failed(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        result=None,
        error=RuntimeError("seed run failed"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["lifecycle"]["failed_stage"] == "seed_runs"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "seed run failed"
    assert payload["completed_at"] is not None


def test_stage_failure_without_active_stage_is_documented(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()
    training_root = tmp_path / "training" / "snapshot-1"

    store.write_campaign_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )

    with pytest.raises(TrainingJobStatusError) as exc_info:
        store.write_stage_failed(
            identity=campaign,
            training_root=training_root,
            dataset_manifest_hash="hash-campaign",
            result=None,
            error=RuntimeError("seed run failed"),
        )
    assert exc_info.value.__cause__ is not None
    assert "cannot fail a lifecycle without an active stage" in str(
        exc_info.value.__cause__
    )


def test_terminal_state_is_not_reopenable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    training_root = tmp_path / "training" / "snapshot-1"

    _active_training(store, identity=identity, tmp_path=tmp_path)
    store.write_stage_failed(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash=None,
        result=None,
        error=RuntimeError("boom"),
    )

    with pytest.raises(TrainingJobStatusError):
        store.write_stage_started(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=None,
            stage=TrainingOperationStage.EVALUATION,
        )


def test_list_attempts_and_campaigns_scoped_to_snapshot(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_started(
        identity=_identity(attempt_id="attempt-a"),
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash=None,
    )
    store.write_started(
        identity=_identity(attempt_id="attempt-b"),
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash=None,
    )
    store.write_started(
        identity=_identity(snapshot_id="snapshot-2", attempt_id="attempt-a"),
        training_root=tmp_path / "training" / "snapshot-2",
        dataset_manifest_hash=None,
    )
    store.write_campaign_started(
        identity=_campaign_identity(campaign_id="campaign-1"),
        training_root=tmp_path / "training" / "snapshot-1",
        dataset_manifest_hash=None,
    )

    assert [
        item.attempt_id
        for item in store.list_attempts(snapshot_id="snapshot-1")
    ] == ["attempt-a", "attempt-b"]
    assert [
        item.attempt_id
        for item in store.list_attempts(snapshot_id="snapshot-2")
    ] == ["attempt-a"]
    assert [
        item.campaign_id
        for item in store.list_campaigns(snapshot_id="snapshot-1")
    ] == ["campaign-1"]
    assert store.list_campaigns(snapshot_id="snapshot-2") == ()


def test_invalid_snapshot_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        TrainingJobIdentity(snapshot_id="../escape", attempt_id="attempt-a")

    with pytest.raises(ValueError, match="must not be empty"):
        TrainingJobIdentity(snapshot_id="snapshot-1", attempt_id="")

    with pytest.raises(ValueError, match="unsupported characters"):
        TrainingJobIdentity(snapshot_id="snapshot-1", attempt_id="bad/id")


def test_non_string_snapshot_id_is_rejected() -> None:
    with pytest.raises(TypeError, match="snapshot_id must be a string"):
        TrainingJobIdentity(
            snapshot_id=123,  # type: ignore[arg-type]
            attempt_id="attempt-a",
        )


def test_surrounding_whitespace_in_identity_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="snapshot_id must not contain surrounding whitespace",
    ):
        TrainingJobIdentity(
            snapshot_id=" snapshot-1 ",
            attempt_id="attempt-a",
        )

    with pytest.raises(
        ValueError,
        match="attempt_id must not contain surrounding whitespace",
    ):
        TrainingJobIdentity(
            snapshot_id="snapshot-1",
            attempt_id="attempt-a ",
        )


def test_status_payload_contains_schema_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.write_started(
        identity=_identity(),
        training_root=tmp_path / "root",
        dataset_manifest_hash=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4


def test_replace_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    identity = _identity()
    seen_tmp: list[Path] = []

    def fail_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        seen_tmp.append(self)
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(TrainingJobStatusError) as raised:
        store.write_started(
            identity=identity,
            training_root=tmp_path / "root",
            dataset_manifest_hash=None,
        )

    assert isinstance(raised.value.__cause__, PermissionError)
    assert seen_tmp
    for path in seen_tmp:
        assert not path.exists()
    assert not store.path_for(identity=identity).exists()


def test_write_stage_succeeded_wraps_serialization_errors(
    tmp_path: Path,
) -> None:
    class _BrokenResult:
        def to_payload(self) -> dict[str, object]:
            raise TypeError("payload serialization broke")

    store = _store(tmp_path)
    identity = _identity()
    _active_training(store, identity=identity, tmp_path=tmp_path)

    with pytest.raises(TrainingJobStatusError) as raised:
        store.write_stage_succeeded(
            identity=identity,
            training_root=tmp_path / "training" / "snapshot-1",
            dataset_manifest_hash="hash",
            result=_BrokenResult(),  # type: ignore[arg-type]
            acceptance=None,
        )

    assert isinstance(raised.value.__cause__, TypeError)
    assert "payload serialization broke" in str(raised.value.__cause__)


def test_stage_failure_preserves_training_success(tmp_path: Path) -> None:
    store = _store(tmp_path)
    training_root = tmp_path / "training" / "snapshot-1"
    identity = _identity()
    execution = _training_run_result(tmp_path)

    _active_training(store, identity=identity, tmp_path=tmp_path)
    store.write_stage_succeeded(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        result=execution,
    )
    store.write_stage_started(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        stage=TrainingOperationStage.RECEIPT,
    )
    path = store.write_stage_failed(
        identity=identity,
        training_root=training_root,
        dataset_manifest_hash="dataset-hash",
        result=execution,
        error=RuntimeError("manifest boom"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "failed"
    assert payload["lifecycle"]["failed_stage"] == "receipt"
    assert payload["lifecycle"]["completed_stages"] == ["training"]
    assert payload["error_type"] == "RuntimeError"
    assert payload["evidence"]["average_loss"] == 0.25


def test_campaign_cancelled_records_cancelled_seed_stage(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    campaign = _campaign_identity()
    training_root = tmp_path / "training" / "snapshot-1"

    store.write_campaign_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
    )
    store.write_stage_started(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        stage=TrainingOperationStage.SEED_RUNS,
    )
    path = store.write_stage_cancelled(
        identity=campaign,
        training_root=training_root,
        dataset_manifest_hash="hash-campaign",
        result=None,
        error=asyncio.CancelledError("user interrupt"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["lifecycle"]["status"] == "cancelled"
    assert payload["lifecycle"]["cancelled_stage"] == "seed_runs"
    assert payload["completed_at"] is not None
    assert payload["error_type"] == "CancelledError"


def test_unsupported_status_schema_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    legacy_path = store.path_for(identity=identity)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "training_status": "succeeded",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingJobStatusError, match="unsupported"):
        store.read_lifecycle(identity=identity)


def test_missing_lifecycle_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    broken_path = store.path_for(identity=identity)
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "snapshot_id": identity.snapshot_id,
                "attempt_id": identity.attempt_id,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TrainingJobStatusError, match="lifecycle is missing or invalid"
    ):
        store.read_lifecycle(identity=identity)


def test_lifecycle_requires_status_and_completed_stages() -> None:
    with pytest.raises(ValueError, match="status is required"):
        TrainingLifecycleState.from_payload({"completed_stages": []})
    with pytest.raises(ValueError, match="completed_stages is required"):
        TrainingLifecycleState.from_payload({"status": "running"})


def test_training_product_code_does_not_import_crawler_runtime() -> None:
    violations: list[str] = []
    training_root = PROJECT_ROOT / "training"

    for path in training_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module,)

            for module in imported:
                if module == "crawler.runtime" or module.startswith(
                    "crawler.runtime."
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"{module}"
                    )

    assert violations == []
