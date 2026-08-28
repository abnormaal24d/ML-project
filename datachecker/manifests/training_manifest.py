"""Manifest model for model-training output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datachecker.manifests.artifact_manifest import ArtifactManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus


@dataclass(frozen=True, slots=True)
class TrainingManifest(ArtifactManifest):
    """
    Persisted proof that the trained checkpoint matches the chosen dataset.
    """

    dataset_fingerprint: str
    training_config_fingerprint: str
    model_config_fingerprint: str
    input_dataset_root: Path
    checkpoint_path: Path
    metrics_path: Path
    epoch_count: int
    sample_count: int
    training_completed_at: str
    release_stage: str
    acceptance_status: str
    sha256: str
    lifecycle_stage: str = "training-ready"
    status: WorkflowLifecycleStatus = WorkflowLifecycleStatus.COMPLETED
    final: bool = True

    def __post_init__(self) -> None:
        ArtifactManifest.__post_init__(self)
        if self.lifecycle_stage != "training-ready":
            raise ValueError("training lifecycle_stage must be training-ready")
        if (
            self.status is not WorkflowLifecycleStatus.COMPLETED
            or not self.final
        ):
            raise ValueError("training manifest must be completed and final")
        if self.epoch_count <= 0 or self.sample_count <= 0:
            raise ValueError("completed training counts must be positive")
        for name in (
            "dataset_fingerprint",
            "training_config_fingerprint",
            "model_config_fingerprint",
            "training_completed_at",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"training manifest {name} must not be empty")
        if not self.acceptance_status.strip():
            raise ValueError("training acceptance_status must not be empty")
        if self.release_stage not in {
            "pipeline_smoke",
            "learning_candidate",
            "candidate",
            "production_model",
        }:
            raise ValueError("training release_stage is invalid")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> TrainingManifest:
        """Build a manifest instance from JSON payload data."""

        return cls(
            **cls.identity_from_payload(payload),
            dataset_fingerprint=cls.as_required_str(
                payload.get("dataset_fingerprint"),
                field="dataset_fingerprint",
            ),
            training_config_fingerprint=cls.as_required_str(
                payload.get("training_config_fingerprint"),
                field="training_config_fingerprint",
            ),
            model_config_fingerprint=cls.as_required_str(
                payload.get("model_config_fingerprint"),
                field="model_config_fingerprint",
            ),
            input_dataset_root=cls.as_required_path(
                payload.get("input_dataset_root"),
                field="input_dataset_root",
            ),
            checkpoint_path=cls.as_required_path(
                payload.get("checkpoint_path"),
                field="checkpoint_path",
            ),
            metrics_path=cls.as_required_path(
                payload.get("metrics_path"),
                field="metrics_path",
            ),
            epoch_count=cls.as_int(payload.get("epoch_count")),
            sample_count=cls.as_int(payload.get("sample_count")),
            training_completed_at=cls.as_required_str(
                payload.get("training_completed_at"),
                field="training_completed_at",
            ),
            release_stage=cls.as_required_str(
                payload.get("release_stage"),
                field="release_stage",
            ),
            acceptance_status=cls.as_required_str(
                payload.get("acceptance_status"),
                field="acceptance_status",
            ),
            lifecycle_stage=cls.as_required_str(
                payload.get("lifecycle_stage"),
                field="lifecycle_stage",
            ),
            status=WorkflowLifecycleStatus.parse(payload.get("status")),
            final=cls.as_bool(payload.get("final")),
            sha256=cls.as_required_str(
                payload.get("sha256"),
                field="sha256",
            ),
        )
