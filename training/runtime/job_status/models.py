"""Domain identity for one persistent training-job status stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class TrainingStage(StrEnum):
    """Persisted stages of one resumable model-development campaign."""

    DATASET_FREEZE = "DATASET_FREEZE"
    TOKENIZER_BUILD = "TOKENIZER_BUILD"
    MODALITY_PRETRAIN = "MODALITY_PRETRAIN"
    CROSS_MODAL_ALIGNMENT = "CROSS_MODAL_ALIGNMENT"
    MULTIMODAL_PRETRAIN = "MULTIMODAL_PRETRAIN"
    INSTRUCTION_TUNING = "INSTRUCTION_TUNING"
    PREFERENCE_TUNING = "PREFERENCE_TUNING"
    SAFETY_TUNING = "SAFETY_TUNING"
    BENCHMARK = "BENCHMARK"
    ACCEPTANCE = "ACCEPTANCE"
    PROMOTION = "PROMOTION"


class TrainingOperationStage(StrEnum):
    """Runtime orchestration stages of one training attempt or campaign.

    Distinct from :class:`TrainingStage`, which models resumable
    model-development stages. These stages describe the operational
    lifecycle of a single training job execution.
    """

    TRAINING = "training"
    SEED_RUNS = "seed_runs"
    RECEIPT = "receipt"
    EVALUATION = "evaluation"
    REPRODUCIBILITY = "reproducibility"
    ACCEPTANCE = "acceptance"
    MANIFESTS = "manifests"


class TrainingLifecycleStatus(StrEnum):
    """Operational status of one training lifecycle record."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingLifecycleState:
    """Single source of truth for one operational lifecycle stream.

    Attempts and reproducibility campaigns share this state machine.
    """

    status: TrainingLifecycleStatus = TrainingLifecycleStatus.RUNNING
    current_stage: TrainingOperationStage | None = None
    completed_stages: tuple[TrainingOperationStage, ...] = ()
    failed_stage: TrainingOperationStage | None = None
    cancelled_stage: TrainingOperationStage | None = None

    def start(self, stage: TrainingOperationStage) -> "TrainingLifecycleState":
        if self.status is not TrainingLifecycleStatus.RUNNING:
            raise ValueError(
                f"cannot start {stage.value!r} from {self.status.value!r}"
            )
        if self.current_stage is not None:
            raise ValueError(
                f"stage {self.current_stage.value!r} is already active"
            )
        if stage in self.completed_stages:
            raise ValueError(f"stage {stage.value!r} is already completed")
        return replace(self, current_stage=stage)

    def complete_stage(self) -> "TrainingLifecycleState":
        stage = self.current_stage
        if stage is None:
            raise ValueError("no active stage to complete")
        return replace(
            self,
            current_stage=None,
            completed_stages=(*self.completed_stages, stage),
        )

    def fail(self) -> "TrainingLifecycleState":
        stage = self.current_stage
        if stage is None:
            raise ValueError("cannot fail a lifecycle without an active stage")
        if self.status is TrainingLifecycleStatus.FAILED:
            raise ValueError("job already failed")
        return replace(
            self,
            status=TrainingLifecycleStatus.FAILED,
            current_stage=None,
            failed_stage=stage,
        )

    def cancel(self) -> "TrainingLifecycleState":
        stage = self.current_stage
        if stage is None:
            raise ValueError(
                "cannot cancel a lifecycle without an active stage"
            )
        if self.status is TrainingLifecycleStatus.CANCELLED:
            raise ValueError("job already cancelled")
        return replace(
            self,
            status=TrainingLifecycleStatus.CANCELLED,
            current_stage=None,
            cancelled_stage=stage,
        )

    def complete_job(
        self,
        *,
        required_stages: tuple[TrainingOperationStage, ...],
    ) -> "TrainingLifecycleState":
        if self.status is not TrainingLifecycleStatus.RUNNING:
            raise ValueError(
                f"cannot complete a lifecycle from {self.status.value!r}"
            )
        if self.current_stage is not None:
            raise ValueError("cannot complete with an active stage")
        missing = tuple(
            stage
            for stage in required_stages
            if stage not in self.completed_stages
        )
        if missing:
            raise ValueError(
                "job cannot complete before required stages succeed: "
                + ", ".join(stage.value for stage in missing)
            )
        return replace(self, status=TrainingLifecycleStatus.COMPLETED)

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "current_stage": (
                self.current_stage.value
                if self.current_stage is not None
                else None
            ),
            "completed_stages": [
                stage.value for stage in self.completed_stages
            ],
            "failed_stage": (
                self.failed_stage.value
                if self.failed_stage is not None
                else None
            ),
            "cancelled_stage": (
                self.cancelled_stage.value
                if self.cancelled_stage is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "TrainingLifecycleState":
        if not isinstance(payload, dict):
            raise ValueError("training lifecycle state must be a mapping")
        status_value = payload.get("status")
        if not isinstance(status_value, str) or not status_value:
            raise ValueError("training lifecycle status is required")
        completed_value = payload.get("completed_stages")
        if not isinstance(completed_value, (list, tuple)):
            raise ValueError("training lifecycle completed_stages is required")
        return cls(
            status=TrainingLifecycleStatus(status_value),
            current_stage=_optional_operation_stage(
                payload.get("current_stage")
            ),
            completed_stages=tuple(
                TrainingOperationStage(str(value)) for value in completed_value
            ),
            failed_stage=_optional_operation_stage(
                payload.get("failed_stage")
            ),
            cancelled_stage=_optional_operation_stage(
                payload.get("cancelled_stage")
            ),
        )


def _optional_operation_stage(value: object) -> TrainingOperationStage | None:
    if value is None:
        return None
    return TrainingOperationStage(str(value))


def _stage_index(stage: TrainingStage | str) -> int:
    return tuple(TrainingStage).index(TrainingStage(stage))


@dataclass(frozen=True, slots=True)
class TrainingJobIdentity:
    """Canonical identity for one training status file."""

    snapshot_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        _require_safe_identifier(self.snapshot_id, field="snapshot_id")
        _require_safe_identifier(self.attempt_id, field="attempt_id")


@dataclass(frozen=True, slots=True)
class TrainingCampaignIdentity:
    """Identity for one reproducibility campaign status document."""

    snapshot_id: str
    campaign_id: str

    def __post_init__(self) -> None:
        _require_safe_identifier(self.snapshot_id, field="snapshot_id")
        _require_safe_identifier(self.campaign_id, field="campaign_id")


@dataclass(frozen=True, slots=True)
class TrainingStageState:
    """Serializable progress state for resumable training stages."""

    current_stage: TrainingStage | None = TrainingStage.DATASET_FREEZE
    completed_stages: tuple[TrainingStage, ...] = ()
    failed_stage: TrainingStage | None = None
    attempt: int = 1
    input_fingerprint: str | None = None
    output_fingerprint: str | None = None
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("stage attempt must be positive")
        indexes = [_stage_index(stage) for stage in self.completed_stages]
        if indexes != sorted(set(indexes)):
            raise ValueError("completed stages must be unique and ordered")
        if (
            self.current_stage is not None
            and self.current_stage in self.completed_stages
        ):
            raise ValueError("current stage cannot already be completed")

    def complete(
        self,
        *,
        output_fingerprint: str,
        next_stage: TrainingStage | None = None,
        parent_checkpoint: str | None = None,
    ) -> "TrainingStageState":
        if self.current_stage is None:
            raise ValueError("all training stages are already completed")
        completed = tuple((*self.completed_stages, self.current_stage))
        if next_stage is None:
            current_index = _stage_index(self.current_stage)
            ordered = tuple(TrainingStage)
            next_stage = (
                ordered[current_index + 1]
                if current_index + 1 < len(ordered)
                else None
            )
        return TrainingStageState(
            current_stage=next_stage,
            completed_stages=completed,
            failed_stage=None,
            attempt=1,
            input_fingerprint=output_fingerprint,
            output_fingerprint=output_fingerprint,
            parent_checkpoint=parent_checkpoint or self.parent_checkpoint,
        )

    def fail(self) -> "TrainingStageState":
        if self.current_stage is None:
            raise ValueError("completed training stages cannot fail")
        return replace(self, failed_stage=self.current_stage)

    def retry(self) -> "TrainingStageState":
        return replace(
            self,
            failed_stage=None,
            attempt=self.attempt + 1,
            output_fingerprint=None,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "current_stage": (
                self.current_stage.value
                if self.current_stage is not None
                else None
            ),
            "completed_stages": [
                stage.value for stage in self.completed_stages
            ],
            "failed_stage": (
                self.failed_stage.value
                if self.failed_stage is not None
                else None
            ),
            "attempt": self.attempt,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "parent_checkpoint": self.parent_checkpoint,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "TrainingStageState":
        if not isinstance(payload, dict):
            raise ValueError("training stage state must be a mapping")
        failed = payload.get("failed_stage")
        current = payload.get("current_stage")
        return cls(
            current_stage=(TrainingStage(str(current)) if current else None),
            completed_stages=tuple(
                TrainingStage(str(value))
                for value in payload.get("completed_stages", [])
            ),
            failed_stage=(TrainingStage(str(failed)) if failed else None),
            attempt=int(payload.get("attempt", 1)),
            input_fingerprint=_optional_string(
                payload.get("input_fingerprint")
            ),
            output_fingerprint=_optional_string(
                payload.get("output_fingerprint")
            ),
            parent_checkpoint=_optional_string(
                payload.get("parent_checkpoint")
            ),
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_safe_identifier(value: object, *, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    if not value:
        raise ValueError(f"{field} must not be empty")
    if not all(
        character.isalnum() or character in {"-", "_"} for character in value
    ):
        raise ValueError(f"{field} contains unsupported characters")


__all__ = [
    "TrainingCampaignIdentity",
    "TrainingJobIdentity",
    "TrainingLifecycleState",
    "TrainingLifecycleStatus",
    "TrainingOperationStage",
    "TrainingStage",
    "TrainingStageState",
]
