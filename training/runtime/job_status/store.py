"""Application-facing store for persistent training lifecycle transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from training.runtime.job_status.models import (
    TrainingCampaignIdentity,
    TrainingJobIdentity,
    TrainingLifecycleState,
    TrainingOperationStage,
)
from training.runtime.job_status.payloads import (
    STATUS_SCHEMA_VERSION,
    TrainingResultPayload,
    lifecycle_payload,
)
from training.runtime.job_status.persistence import (
    AtomicTrainingJobStatusWriter,
    LifecycleIdentity,
    TrainingJobStatusError,
)

_ATTEMPT_STAGES = frozenset(
    {
        TrainingOperationStage.TRAINING,
        TrainingOperationStage.RECEIPT,
        TrainingOperationStage.EVALUATION,
    }
)

_CAMPAIGN_STAGES = frozenset(
    {
        TrainingOperationStage.SEED_RUNS,
        TrainingOperationStage.REPRODUCIBILITY,
        TrainingOperationStage.ACCEPTANCE,
        TrainingOperationStage.MANIFESTS,
    }
)

Now = Callable[[], datetime]
T = TypeVar("T")


class TrainingJobStatusStore:
    """Persist domain lifecycle transitions for long-running training jobs."""

    def __init__(
        self,
        *,
        now: Now,
        writer: AtomicTrainingJobStatusWriter,
    ) -> None:
        self._now = now
        self._writer = writer

    def write_started(
        self,
        *,
        identity: TrainingJobIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
    ) -> Path:
        """Create the initial running attempt document."""
        existing = self._writer.read(identity=identity)
        started_at = _existing_started_at(existing, now=lambda: self._now())
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=TrainingLifecycleState(),
            existing=existing,
            started_at=started_at,
        )

    def write_campaign_started(
        self,
        *,
        identity: TrainingCampaignIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
    ) -> Path:
        """Write the initial running state for one reproducibility campaign."""
        existing = self._writer.read(identity=identity)
        started_at = _existing_started_at(existing, now=lambda: self._now())
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=TrainingLifecycleState(),
            existing=existing,
            started_at=started_at,
        )

    def write_stage_started(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        stage: TrainingOperationStage,
    ) -> Path:
        allowed = (
            _CAMPAIGN_STAGES
            if isinstance(identity, TrainingCampaignIdentity)
            else _ATTEMPT_STAGES
        )
        if stage not in allowed:
            raise ValueError(
                f"{type(identity).__name__} cannot execute "
                f"stage {stage.value!r}"
            )
        existing = self._require_existing(identity)
        state = self._run_write(lambda: _lifecycle_from(existing).start(stage))
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def write_stage_succeeded(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        result: TrainingResultPayload | None = None,
        acceptance: dict[str, object] | None = None,
    ) -> Path:
        existing = self._require_existing(identity)
        state = self._run_write(
            lambda: _lifecycle_from(existing).complete_stage()
        )
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            result=result,
            acceptance=acceptance,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def write_stage_failed(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        result: TrainingResultPayload | None,
        error: BaseException,
    ) -> Path:
        existing = self._require_existing(identity)
        state = self._run_write(lambda: _lifecycle_from(existing).fail())
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            result=result,
            error=error,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def write_stage_cancelled(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        result: TrainingResultPayload | None,
        error: BaseException,
    ) -> Path:
        existing = self._require_existing(identity)
        state = self._run_write(lambda: _lifecycle_from(existing).cancel())
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            result=result,
            error=error,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def write_attempt_completed(
        self,
        *,
        identity: TrainingJobIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        result: TrainingResultPayload | None = None,
        required_stages: tuple[TrainingOperationStage, ...],
    ) -> Path:
        existing = self._require_existing(identity)
        state = self._run_write(
            lambda: _lifecycle_from(existing).complete_job(
                required_stages=required_stages
            )
        )
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            result=result,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def write_campaign_attempts(
        self,
        *,
        identity: TrainingCampaignIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        attempt_ids: tuple[str, ...],
        primary_attempt_id: str | None = None,
    ) -> Path:
        """Record which attempts belong to this campaign.

        Only references are persisted; run outputs live in the attempt
        documents. The lifecycle state itself is left untouched.
        """
        existing = self._require_existing(identity)
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=_lifecycle_from(existing),
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
            attempt_ids=attempt_ids,
            primary_attempt_id=primary_attempt_id,
        )

    def write_campaign_completed(
        self,
        *,
        identity: TrainingCampaignIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        result: TrainingResultPayload | None = None,
        required_stages: tuple[TrainingOperationStage, ...],
    ) -> Path:
        existing = self._require_existing(identity)
        state = self._run_write(
            lambda: _lifecycle_from(existing).complete_job(
                required_stages=required_stages
            )
        )
        return self._persist_lifecycle(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            state=state,
            result=result,
            existing=existing,
            started_at=_existing_started_at(existing, now=lambda: self._now()),
        )

    def read_lifecycle(
        self, *, identity: LifecycleIdentity
    ) -> TrainingLifecycleState | None:
        document = self._writer.read(identity=identity)
        if document is None:
            return None
        return _lifecycle_from(document)

    def list_attempts(
        self, *, snapshot_id: str
    ) -> tuple[TrainingJobIdentity, ...]:
        return self._writer.list_attempts(snapshot_id=snapshot_id)

    def list_campaigns(
        self, *, snapshot_id: str
    ) -> tuple[TrainingCampaignIdentity, ...]:
        return self._writer.list_campaigns(snapshot_id=snapshot_id)

    def path_for(self, *, identity: TrainingJobIdentity) -> Path:
        return self._writer.path_for(identity=identity)

    def campaign_path_for(self, *, identity: TrainingCampaignIdentity) -> Path:
        return self._writer.campaign_path_for(identity=identity)

    def _require_existing(
        self, identity: LifecycleIdentity
    ) -> dict[str, object]:
        document = self._writer.read(identity=identity)
        if document is None:
            raise TrainingJobStatusError(
                "Training job status document does not exist"
            )
        return document

    def _persist_lifecycle(
        self,
        *,
        identity: LifecycleIdentity,
        training_root: str | Path,
        dataset_manifest_hash: str | None,
        state: TrainingLifecycleState,
        existing: Mapping[str, object] | None = None,
        result: TrainingResultPayload | None = None,
        acceptance: dict[str, object] | None = None,
        error: BaseException | None = None,
        started_at: str,
        attempt_ids: tuple[str, ...] | None = None,
        primary_attempt_id: str | None = None,
    ) -> Path:
        timestamp = self._now().isoformat()
        return self._run_write(
            lambda: self._writer.write(
                identity=identity,
                payload=lifecycle_payload(
                    identity=identity,
                    state=state,
                    training_root=training_root,
                    dataset_manifest_hash=dataset_manifest_hash,
                    result=result,
                    acceptance=acceptance,
                    error=error,
                    started_at=started_at,
                    timestamp=timestamp,
                    existing=existing,
                    attempt_ids=attempt_ids,
                    primary_attempt_id=primary_attempt_id,
                ),
            )
        )

    @staticmethod
    def _run_write(operation: Callable[[], T]) -> T:
        try:
            return operation()
        except TrainingJobStatusError:
            raise
        except Exception as error:
            raise TrainingJobStatusError(
                "Training job status could not be written"
            ) from error


def _lifecycle_from(document: dict[str, object]) -> TrainingLifecycleState:
    if document.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise TrainingJobStatusError("unsupported training job status schema")
    lifecycle = document.get("lifecycle")
    if not isinstance(lifecycle, dict):
        raise TrainingJobStatusError(
            "training job status lifecycle is missing or invalid"
        )
    try:
        return TrainingLifecycleState.from_payload(lifecycle)
    except (TypeError, ValueError) as error:
        raise TrainingJobStatusError(
            "training job status lifecycle is invalid"
        ) from error


def _existing_started_at(
    document: dict[str, object] | None, *, now: Callable[[], datetime]
) -> str:
    if document is not None:
        started_at = document.get("started_at")
        if isinstance(started_at, str) and started_at:
            return started_at
    return now().isoformat()


__all__ = ["TrainingJobStatusStore"]
