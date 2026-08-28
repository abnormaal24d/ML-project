"""Training composition for the TRAIN phase and canonical model construction.

This module owns the real training dependencies that the training pipeline
requires. It is the single composition surface replacing the former
training/multimodal service bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.multimodal.training_settings import resolve_objective_loss_weights

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from torch import nn

    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from config.path_resolution.project_paths import ProjectPaths
    from logger.factory import ProjectLoggerFactory
    from multimodal.model.model import MultimodalModel
    from training.losses.objective import SupervisedOrSelfSupervisedLoss
    from training.runtime.checkpoint.contract import CheckpointContract
    from training.runtime.trainer import MultimodalTrainer


def build_model(
    settings: ModelSettings,
    *,
    training_backend: str = "pipeline_smoke",
) -> "MultimodalModel":
    """Build the canonical model for the selected training backend.

    Model family validation is performed by the domain layer (training/).
    This composition function only constructs the model instance.
    """
    from multimodal.model.initialization import freeze_model_components
    from multimodal.model.model import MultimodalModel

    model = MultimodalModel(settings, training_backend=training_backend)
    freeze_model_components(
        model=model,
        component_prefixes=settings.freeze_components,
    )
    return model


def build_trainer(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    project_paths: ProjectPaths,
    logger_factory: ProjectLoggerFactory,
    generated_at: Callable[[], datetime],
    checkpoint_contract: "CheckpointContract | None" = None,
) -> "MultimodalTrainer":
    """Build the fully wired trainer for the TRAIN phase.

    The model factory is a local closure bound to the resolved training
    backend; the trainer (and therefore the closure) is rebuilt per training
    snapshot.
    """

    from training.runtime.trainer import MultimodalTrainer

    logger = logger_factory.get_logger_for(MultimodalTrainer)

    from mmcrawler_datasets.dataloader import (
        load_configured_vocabulary_tokenizer,
    )

    tokenizer = load_configured_vocabulary_tokenizer(
        model_settings=model_settings,
        training_settings=training_settings,
        project_root=Path(project_paths.project_root),
    )

    from training.export.export import export_model
    from training.runtime.optimization import (
        build_lr_scheduler,
        build_optimizer,
    )
    from training.runtime.preparation import prepare_training_backend

    prepared_backend = prepare_training_backend(
        training_settings=training_settings,
    )
    logger.info(
        "training_backend_resolved",
        backend=prepared_backend.name,
        requires_distributed_runtime=(
            prepared_backend.requires_distributed_runtime
        ),
        requires_gpu=prepared_backend.requires_gpu,
    )

    def create_model(settings: ModelSettings) -> "MultimodalModel":
        return build_model(settings, training_backend=prepared_backend.name)

    def model_exporter(
        *,
        model: nn.Module,
        export_directory: Path,
        model_settings: ModelSettings,
        training_settings: TrainingSettings,
        dataset_root: Path,
    ) -> dict[str, str]:
        return export_model(
            model=model,
            export_directory=export_directory,
            model_settings=model_settings,
            training_settings=training_settings,
            dataset_root=dataset_root,
            generated_at=generated_at(),
        )

    return MultimodalTrainer(
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        model_exporter=model_exporter,
        logger=logger,
        training_backend=prepared_backend.name,
        model_factory=create_model,
        loss_factory=build_training_loss,
        optimizer_factory=build_optimizer,
        scheduler_factory=build_lr_scheduler,
        project_root=project_paths.project_root,
        checkpoint_contract=checkpoint_contract,
    )


def resolve_snapshot_training_settings(
    *,
    training_settings: TrainingSettings,
    training_root: Path,
    manifest_filename: str,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> TrainingSettings:
    """Resolve the effective training settings for one immutable snapshot."""

    from training.runtime.snapshot_tokenizer_binding import (
        bind_snapshot_tokenizer,
    )

    effective_training = bind_snapshot_tokenizer(
        training=training_settings,
        manifest_filename=manifest_filename,
        training_root=training_root,
    )

    updates: dict[str, object] = {}

    if seed is not None:
        updates["seed"] = seed

    if deterministic is not None:
        updates["deterministic"] = deterministic

    if updates:
        effective_training = effective_training.model_copy(
            update=updates,
        )

    return effective_training


def build_snapshot_trainer(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    training_root: Path,
    manifest_filename: str,
    project_paths: ProjectPaths,
    logger_factory: ProjectLoggerFactory,
    generated_at: Callable[[], datetime],
    seed: int | None = None,
    deterministic: bool | None = None,
    checkpoint_contract: "CheckpointContract | None" = None,
) -> tuple[MultimodalTrainer, TrainingSettings]:
    """Build a trainer with settings bound to one training snapshot."""

    effective_training = resolve_snapshot_training_settings(
        training_settings=training_settings,
        training_root=training_root,
        manifest_filename=manifest_filename,
        seed=seed,
        deterministic=deterministic,
    )

    trainer = build_trainer(
        model_settings=model_settings,
        training_settings=effective_training,
        project_paths=project_paths,
        logger_factory=logger_factory,
        generated_at=generated_at,
        checkpoint_contract=checkpoint_contract,
    )

    return trainer, effective_training


def build_training_loss(
    settings: "TrainingSettings",
) -> "SupervisedOrSelfSupervisedLoss":
    """Construct the multimodal loss objective from training settings.

    Configuration values are passed through explicitly; the objective
    itself owns no hidden defaults.

    Preference loss validation is performed by the domain layer (training/).
    This composition function only constructs the loss instance.
    """

    from config.environment.default_values import DEFAULT_LOSS_WEIGHTS
    from training.losses.objective import SupervisedOrSelfSupervisedLoss

    preference_mode = settings.preference_loss
    if (
        settings.preference_loss == "dpo"
        and settings.reference_free_preference
    ):
        preference_mode = "pairwise"

    loss_weights = resolve_objective_loss_weights(settings)

    # Weight map validation is performed by the domain layer
    # This composition function only constructs the loss instance

    return SupervisedOrSelfSupervisedLoss(
        contrastive_temperature=DEFAULT_LOSS_WEIGHTS.contrastive_temperature,
        alignment_score_exponent=settings.alignment_loss_power,
        hard_negative_margin=settings.hard_negative_margin,
        training_backend=settings.training_backend,
        training_stage=settings.training_stage,
        preference_mode=preference_mode,
        preference_beta=settings.preference_beta,
        loss_weights=loss_weights,
    )
