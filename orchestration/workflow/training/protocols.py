"""Capability contracts for the pure-DI training workflow graph.

Stateful construction/policy dependencies are object protocols; stateless
operations are callable protocols so composition can bind them directly to
bound functions without adapter classes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings
    from config.releases.release_requirements import (
        ReleaseRequirements,
        ReproducibilityRequirements,
    )
    from datachecker.manifests.training_manifest import TrainingManifest
    from evaluator.reproducibility import TrainingRunReceipt
    from evaluator.results import EvaluationResult
    from release.acceptance_result import TrainingAcceptanceResult
    from training.runtime.results import TrainingRunResult
    from training.runtime.trainer import MultimodalTrainer


class TrainingRuntimeFactory(Protocol):
    """Factory owning trainer construction policy for one snapshot."""

    def create(
        self,
        *,
        training_root: Path,
        seed: int | None,
        deterministic: bool | None,
    ) -> tuple["MultimodalTrainer", "TrainingSettings"]: ...


class CheckpointLoader(Protocol):
    """Loads one persisted checkpoint payload."""

    def __call__(self, checkpoint_path: Path) -> Awaitable[object]: ...


class TrainingReceiptWriter(Protocol):
    """Persists one immutable training run receipt."""

    def __call__(
        self,
        *,
        output_path: Path,
        run_id: str,
        seed: int,
        dataset_manifest_sha256: str,
        checkpoint_payload: Mapping[str, object],
        training_settings: Mapping[str, object],
        model_settings: Mapping[str, object],
        container_digest: str,
        evaluated_metrics: Mapping[str, float] | None,
    ) -> Awaitable["TrainingRunReceipt"]: ...


class TrainingEvaluator(Protocol):
    """Evaluates the selected checkpoint of a completed run."""

    def __call__(
        self,
        *,
        trainer: "MultimodalTrainer",
        training_result: "TrainingRunResult",
        dataset_root: Path,
        leakage_report_path: Path | None,
        reproducibility_report_path: Path | None,
    ) -> Awaitable["EvaluationResult"]: ...


class TrainingManifestWriter(Protocol):
    """Persists post-acceptance training manifests."""

    def __call__(
        self,
        *,
        input_dataset_root: Path,
        dataset_manifest_hash: str,
        training_result: "TrainingRunResult",
        evaluation_result: "EvaluationResult",
        acceptance_result: "TrainingAcceptanceResult",
    ) -> Awaitable["TrainingManifest"]: ...


class TrainingMetricsWriter(Protocol):
    """Persists immutable training metrics before acceptance reads them."""

    def __call__(
        self,
        *,
        input_dataset_root: Path,
        dataset_manifest_hash: str,
        training_result: "TrainingRunResult",
        evaluation_result: "EvaluationResult",
    ) -> Awaitable[Path]: ...


class ReproducibilityEvaluator(Protocol):
    """Evaluates a multi-seed campaign against the reproducibility policy."""

    def __call__(
        self,
        *,
        receipts: Sequence["TrainingRunReceipt"],
        policy: "ReproducibilityRequirements",
        release_requirements_id: str,
    ) -> Awaitable[Mapping[str, object]]: ...


class ReproducibilityReportWriter(Protocol):
    """Persists one multi-run reproducibility report."""

    def __call__(
        self,
        *,
        path: Path,
        report: Mapping[str, object],
    ) -> Awaitable[Path]: ...


class RunReceiptsCollectionWriter(Protocol):
    """Persists the immutable receipts collection backing one report."""

    def __call__(
        self,
        *,
        path: Path,
        receipts: Sequence["TrainingRunReceipt"],
    ) -> Awaitable[Path]: ...


class TrainingAcceptanceEvaluator(Protocol):
    """Evaluates release acceptance for one completed primary run."""

    def __call__(
        self,
        *,
        training_result: "TrainingRunResult",
        evaluation_result: "EvaluationResult",
        input_dataset_root: Path,
        metrics_path: Path,
        release_requirements: "ReleaseRequirements | None",
    ) -> Awaitable["TrainingAcceptanceResult"]: ...
