"""TrainingArtifactValidator autonomous readiness regression tests."""

from __future__ import annotations

from pathlib import Path

from config.collection.training_input_gate import TrainingInputMode
from datachecker.manifests.training_manifest import TrainingManifest
from datachecker.validation.training_artifact_validator import (
    TrainingArtifactValidator,
    _metrics_schema_errors,
)
from datachecker.workflow_decision import WorkflowDecisionReason


def _validator() -> TrainingArtifactValidator:
    return TrainingArtifactValidator(
        minimum_samples=1,
        minimum_modality_counts={
            "text": 1,
            "image": 1,
            "audio": 1,
            "video": 1,
        },
        minimum_task_counts={
            "text_pretrain": 30000,
            "image_text_pair": 25000,
            "audio_text_pair": 2000,
            "video_text_pair": 500,
        },
        require_autonomous_multimodal_readiness=True,
    )


def _production_scale_task_counts() -> dict[str, int]:
    return {
        "text_pretrain": 30000,
        "image_text_pair": 25000,
        "audio_text_pair": 2000,
        "video_text_pair": 500,
    }


def _production_scale_modality_counts() -> dict[str, int]:
    return {
        "text": 30000,
        "image": 25000,
        "audio": 2000,
        "video": 500,
    }


def test_autonomous_readiness_passes_with_canonical_tasks() -> None:
    """P0: production-scale multimodal evidence must not block dataset selection."""

    result = _validator().validate(
        manifest=None,
        training_input_mode=TrainingInputMode.AUGMENTED_WHEN_AVAILABLE,
        current_dataset_manifest_hash="dataset-hash",
        current_checkpoint_path=None,
        current_metrics_path=None,
        current_training_config_fingerprint="training-fp",
        current_model_config_fingerprint="model-fp",
        current_modality_counts=_production_scale_modality_counts(),
        current_task_counts=_production_scale_task_counts(),
        current_sample_count=500_000,
    )

    assert result.reason != (
        WorkflowDecisionReason.TRAINING_BLOCKED_BY_DATASET_SELECTION
    )
    assert (
        result.reason
        == WorkflowDecisionReason.TRAINING_OUTPUT_FOR_SELECTED_DATASET_MISSING
    )


def test_autonomous_readiness_blocks_when_required_task_missing() -> None:
    task_counts = _production_scale_task_counts()
    task_counts["audio_text_pair"] = 0

    result = _validator().validate(
        manifest=None,
        training_input_mode=TrainingInputMode.AUGMENTED_WHEN_AVAILABLE,
        current_dataset_manifest_hash="dataset-hash",
        current_checkpoint_path=None,
        current_metrics_path=None,
        current_training_config_fingerprint="training-fp",
        current_model_config_fingerprint="model-fp",
        current_modality_counts=_production_scale_modality_counts(),
        current_task_counts=task_counts,
        current_sample_count=500_000,
    )

    assert result.reason == (
        WorkflowDecisionReason.TRAINING_BLOCKED_BY_DATASET_SELECTION
    )
    assert "audio_text_pair" in str(result.details)
    assert "autonomous_action" not in str(result.details)
    assert "multimodal_reasoning" not in str(result.details)


def test_immutable_metrics_do_not_embed_post_acceptance_state() -> None:
    """Acceptance belongs to its report and manifest, not signed metrics."""

    manifest = TrainingManifest(
        generation_id="generation",
        workflow_id="workflow",
        project_fingerprint="project",
        config_fingerprint="config",
        environment_name="prod",
        environment_fingerprint="environment",
        python_version="python",
        dependency_lock_fingerprint="lock",
        dataset_fingerprint="dataset",
        training_config_fingerprint="training",
        model_config_fingerprint="model",
        input_dataset_root=Path(__file__),
        checkpoint_path=Path(__file__),
        metrics_path=Path(__file__),
        epoch_count=1,
        sample_count=1,
        training_completed_at="2026-08-20T00:00:00+00:00",
        release_stage="candidate",
        acceptance_status="model_candidate",
        sha256="checkpoint",
    )

    errors = _metrics_schema_errors(
        manifest=manifest,
        metrics_payload={
            "generation_id": "generation",
            "workflow_id": "workflow",
            "project_fingerprint": "project",
            "config_fingerprint": "config",
            "environment_name": "prod",
            "environment_fingerprint": "environment",
            "python_version": "python",
            "dependency_lock_fingerprint": "lock",
            "dataset_fingerprint": "dataset",
            "training_config_fingerprint": "training",
            "model_config_fingerprint": "model",
            "release_stage": "candidate",
            "train_loss": 0.1,
            "val_loss": 0.1,
            "test_loss": 0.1,
            "evaluation": {"valid": True},
        },
    )

    assert errors == ()
