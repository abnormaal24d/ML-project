"""Coordinate construction of one complete training snapshot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.multimodal.training_settings import TrainingSettings
from config.settings.datasets import (
    DatasetPathSettings,
    DatasetValidatorSettings,
    SplitAssignerSettings,
    TrainingDatasetWriterSettings,
    TrainingSnapshotAssemblerSettings,
)
from mmcrawler_datasets.assembly.build import SampleBuildResult, build_samples
from mmcrawler_datasets.snapshots.curated import read_snapshot
from mmcrawler_datasets.snapshots.errors import SnapshotBuildError
from mmcrawler_datasets.snapshots.tokenizer_artifact import (
    train_snapshot_tokenizer,
)
from mmcrawler_datasets.snapshots.training_metadata import (
    write_training_metadata,
)
from mmcrawler_datasets.snapshots.training_outputs import (
    output_settings_from,
    write_training_outputs,
)
from mmcrawler_datasets.snapshots.validation import validate_snapshot

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger
    from mmcrawler_datasets.materialization.audio_generation import (
        AudioGenerationTargetMaterializer,
    )
    from mmcrawler_datasets.materialization.video_generation import (
        VideoGenerationTargetMaterializer,
    )
    from preprocessing.privacy.text_privacy import PiiDetector


@dataclass(frozen=True, slots=True)
class TrainingSnapshotBuildResult:
    """Result of assembling and persisting one training snapshot."""

    samples: SampleBuildResult
    snapshot_dir: Path


def build_training_snapshot(
    *,
    settings: TrainingSnapshotAssemblerSettings,
    split_settings: SplitAssignerSettings,
    validator_settings: DatasetValidatorSettings,
    dataset_paths: DatasetPathSettings,
    training_settings: TrainingSettings,
    project_root: Path,
    logger: ProjectLogger,
    pii_detector: PiiDetector,
    curated_snapshot_directory: Path,
    training_directory: Path,
    snapshot_id: str,
    output_settings: TrainingDatasetWriterSettings,
    audio_materializer_factory: (
        Callable[[Path], AudioGenerationTargetMaterializer] | None
    ) = None,
    video_materializer_factory: (
        Callable[[Path], VideoGenerationTargetMaterializer] | None
    ) = None,
    require_transcript_for_audio_text_pair: bool = False,
) -> TrainingSnapshotBuildResult:
    """Build and validate dataset-owned artifacts in an unpublished directory."""

    snapshot = read_snapshot(
        dataset_paths=dataset_paths,
        snapshot_directory=curated_snapshot_directory,
    )
    training_root = training_directory.resolve()
    resolved_outputs = output_settings_from(output_settings)

    samples = build_samples(
        snapshot,
        settings=settings,
        split_settings=split_settings,
        validator_settings=validator_settings,
        project_root=project_root,
        pii_detector=pii_detector,
        snapshot_directory=curated_snapshot_directory,
        materialization_directory=training_root,
        snapshot_id=snapshot_id,
        enabled_tasks=frozenset(training_settings.tasks),
        audio_materializer_factory=audio_materializer_factory,
        video_materializer_factory=video_materializer_factory,
        require_transcript_for_audio_text_pair=(
            require_transcript_for_audio_text_pair
        ),
    )
    write_training_outputs(
        training_directory=training_root,
        samples=samples,
        dataset_paths=dataset_paths,
        output_settings=resolved_outputs,
    )
    tokenizer_identity = train_snapshot_tokenizer(
        training_directory=training_root,
        train_samples=samples.train_samples,
        snapshot_id=snapshot_id,
        training_settings=training_settings,
    )
    write_training_metadata(
        training_root=training_root,
        snapshot_id=snapshot_id,
        curated_snapshot_directory=curated_snapshot_directory,
        settings=settings,
        dataset_paths=dataset_paths,
        samples=samples,
        tokenizer_identity=tokenizer_identity,
        output_settings=resolved_outputs,
    )
    if not validate_snapshot(
        training_directory=training_root,
        dataset_paths=dataset_paths,
        write_jsonl=resolved_outputs.write_jsonl,
        write_shards=resolved_outputs.write_shards,
        shard_index_filename=resolved_outputs.shard_index_filename,
    ):
        raise SnapshotBuildError(
            "training snapshot failed structural output validation"
        )

    logger.info(
        "training_snapshot_outputs_validated",
        snapshot_id=snapshot_id,
        write_jsonl=resolved_outputs.write_jsonl,
        write_shards=resolved_outputs.write_shards,
        shard_format=(
            resolved_outputs.shard_format
            if resolved_outputs.write_shards
            else None
        ),
    )

    return TrainingSnapshotBuildResult(
        samples=samples,
        snapshot_dir=training_root,
    )


__all__ = ["TrainingSnapshotBuildResult", "build_training_snapshot"]
