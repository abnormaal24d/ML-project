"""Immutable results produced by multimodal training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Numerical optimization signal for one training run."""

    train_loss: float
    validation_loss: float
    test_loss: float
    average_loss: float
    last_epoch_loss: float
    epochs: int
    batches: int
    samples: int
    effective_train_sample_count: int
    effective_task_counts: dict[str, int] = field(default_factory=dict)
    effective_modality_counts: dict[str, int] = field(default_factory=dict)
    training_signal_by_modality: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    epoch_history: tuple[dict[str, object], ...] = ()
    per_modality_losses: dict[str, float] = field(default_factory=dict)
    per_task_losses: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "test_loss": self.test_loss,
            "average_loss": self.average_loss,
            "last_epoch_loss": self.last_epoch_loss,
            "epochs": self.epochs,
            "batches": self.batches,
            "samples": self.samples,
            "effective_train_sample_count": self.effective_train_sample_count,
            "effective_task_counts": dict(self.effective_task_counts),
            "effective_modality_counts": dict(self.effective_modality_counts),
            "training_signal_by_modality": {
                name: dict(value)
                for name, value in self.training_signal_by_modality.items()
            },
            "epoch_history": [dict(row) for row in self.epoch_history],
            "per_modality_losses": dict(self.per_modality_losses),
            "per_task_losses": dict(self.per_task_losses),
        }


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Filesystem artifacts produced by one training run."""

    checkpoint_path: Path
    last_checkpoint_path: Path
    export_directory: Path | None
    export_paths: dict[str, str] = field(default_factory=dict)

    @property
    def evaluation_directory(self) -> Path:
        """Return the canonical directory for evaluation artifacts."""

        root = self.export_directory or self.checkpoint_path.parent
        return root / "evaluation"

    def to_payload(self) -> dict[str, object]:
        return {
            "checkpoint_path": self.checkpoint_path.as_posix(),
            "last_checkpoint_path": self.last_checkpoint_path.as_posix(),
            "export_directory": (
                self.export_directory.as_posix()
                if self.export_directory is not None
                else None
            ),
            "export_paths": dict(self.export_paths),
        }


@dataclass(frozen=True, slots=True)
class TrainingRunIdentity:
    """Seed and split identity for one training run."""

    model_seed: int
    split_seed: int | None = None
    split_assignment: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "model_seed": self.model_seed,
            "split_seed": self.split_seed,
            "split_assignment": dict(self.split_assignment),
        }


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Complete immutable output of the optimizer and exporter."""

    metrics: TrainingMetrics
    artifacts: TrainingArtifacts
    identity: TrainingRunIdentity

    def to_payload(self) -> dict[str, object]:
        return {
            **self.metrics.to_payload(),
            **self.artifacts.to_payload(),
            **self.identity.to_payload(),
        }


__all__ = [
    "TrainingArtifacts",
    "TrainingMetrics",
    "TrainingRunIdentity",
    "TrainingRunResult",
]
