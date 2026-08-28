from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.augmentation_artifact_writer import (
    build_augmentation_summary,
    write_augmentation_artifacts,
)
from augmentation.training_dataset_augmenter import TrainingDatasetAugmenter
from config.path_resolution.project_paths import ProjectPaths
from config.settings.datasets import (
    DatasetPathSettings,
    TrainingDatasetWriterSettings,
)
from datachecker.workflow_decision import WorkflowExecutionPlan
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.snapshots.output_writer import (
    SnapshotOutputSettings,
    write_shard_index,
    write_snapshot_rows,
    write_webdataset_shards,
)
from mmcrawler_datasets.snapshots.publication import staged_snapshot
from mmcrawler_datasets.snapshots.snapshot_rows import (
    read_snapshot_rows,
    resolve_snapshot_directories,
    snapshot_row_paths,
)
from mmcrawler_datasets.snapshots.training_metadata import (
    update_augmented_training_metadata,
)
from mmcrawler_datasets.snapshots.training_outputs import rewrite_derived_views
from mmcrawler_datasets.snapshots.validation import validate_snapshot
from mmcrawler_datasets.training_samples.snapshot_mapping import (
    build_snapshot_sample,
    serialize_snapshot_sample,
)
from orchestration.workflow.phase import (
    PhaseOutcome,
    PhaseStatus,
    RunBlocking,
)
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from augmentation.outcomes.augmentation_result import AugmentationReport

AugmentationManifestWrite = Callable[[], object]

_POSTPROCESSING_THREAD_NAME = "augmentation_snapshot"
_DEFAULT_POSTPROCESSING_WORKERS = 2


def _augment_train_rows(
    *,
    workflow: TrainingDatasetAugmenter,
    output_directory: Path,
    train_rows: tuple[dict[str, object], ...],
    train_path: Path,
) -> tuple[tuple[dict[str, object], ...], AugmentationReport]:
    dataset = tuple(
        build_snapshot_sample(
            payload=row,
            dataset_root=output_directory,
            source_path=train_path,
            line_number=line_number,
        )
        for line_number, row in enumerate(train_rows, start=1)
    )

    result = workflow.augment(
        dataset=dataset,
        dataset_root=output_directory,
    )
    _require_quality_checks_passed(report=result.report)

    rows = tuple(
        serialize_snapshot_sample(
            sample=sample,
            dataset_root=output_directory,
        )
        for sample in result.dataset
    )

    return rows, result.report


def _require_quality_checks_passed(*, report: AugmentationReport) -> None:
    """Fail closed before publishing a snapshot with invalid media lineage."""

    if report.quality_checks_passed:
        return
    failures = "; ".join(report.quality_check_failures[:10])
    suffix = f": {failures}" if failures else ""
    raise ValueError(f"augmentation_quality_checks_failed{suffix}")


def _postprocess_snapshot(
    *,
    dataset_paths: DatasetPathSettings,
    writer_settings: TrainingDatasetWriterSettings,
    output_directory: Path,
    augmented_train_rows: tuple[dict[str, object], ...],
    validation_rows: tuple[dict[str, object], ...],
    test_rows: tuple[dict[str, object], ...],
    workers: int,
) -> None:
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=_POSTPROCESSING_THREAD_NAME,
    ) as executor:
        futures = (
            executor.submit(
                rewrite_derived_views,
                paths=dataset_paths,
                output_directory=output_directory,
                train_rows=augmented_train_rows,
                val_rows=validation_rows,
                test_rows=test_rows,
            ),
            executor.submit(
                update_augmented_training_metadata,
                training_root=output_directory,
                dataset_paths=dataset_paths,
                augmentation_summary=build_augmentation_summary(
                    train_rows=augmented_train_rows,
                    val_rows=validation_rows,
                    test_rows=test_rows,
                ),
            ),
        )
    for future in futures:
        future.result()

    output_settings = SnapshotOutputSettings(
        write_jsonl=writer_settings.write_jsonl,
        write_shards=writer_settings.write_shards,
        shard_format=writer_settings.shard_format,
        max_samples_per_shard=writer_settings.shard_max_samples,
        max_bytes_per_shard=writer_settings.shard_max_bytes,
        shards_directory=writer_settings.training_shards_directory,
        shard_index_filename=writer_settings.shard_index_filename,
    )
    if output_settings.write_shards:
        shard_entries = write_webdataset_shards(
            training_directory=output_directory,
            rows_by_split={
                "train": augmented_train_rows,
                "val": validation_rows,
                "test": test_rows,
            },
            output_settings=output_settings,
        )
        write_shard_index(
            path=output_directory / output_settings.shard_index_filename,
            entries_by_split=shard_entries,
        )

    if not validate_snapshot(
        training_directory=output_directory,
        dataset_paths=dataset_paths,
        write_jsonl=output_settings.write_jsonl,
        write_shards=output_settings.write_shards,
        shard_index_filename=output_settings.shard_index_filename,
    ):
        raise RuntimeError(
            "augmented training snapshot failed validation: "
            f"{output_directory}"
        )


class AugmentPhaseRunner:
    """Execute augmentation workflow phases and collect structured multimodal."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        write_augmentation_manifest: AugmentationManifestWrite,
        run_blocking: RunBlocking,
        io_timeout_seconds: float,
        augmentation_workflow: TrainingDatasetAugmenter,
        clock: Clock,
        dataset_paths: DatasetPathSettings,
        writer_settings: TrainingDatasetWriterSettings,
        project_paths: ProjectPaths,
        output_root: Path,
    ) -> None:
        self._logger = logger
        self._write_augmentation_manifest = write_augmentation_manifest
        self._run_blocking = run_blocking
        self._io_timeout_seconds = io_timeout_seconds
        self._augmentation_workflow = augmentation_workflow
        self._clock = clock
        self._dataset_paths = dataset_paths
        self._writer_settings = writer_settings
        self._project_paths = project_paths
        self._output_root = output_root

    async def run(self, plan: WorkflowExecutionPlan) -> PhaseOutcome:
        training_root = plan.training_root
        if training_root is None:
            raise ValueError("workflow plan is missing training_root")

        await self._run_blocking(
            self._build_snapshot,
            training_root=training_root,
            output_root=self._output_root,
            timeout_seconds=self._io_timeout_seconds,
        )
        await self._run_blocking(
            self._write_augmentation_manifest,
            timeout_seconds=self._io_timeout_seconds,
        )
        return PhaseOutcome(status=PhaseStatus.SUCCEEDED)

    def _build_snapshot(
        self,
        *,
        training_root: Path,
        output_root: Path,
    ) -> None:
        """Coordinate one augmented snapshot build inside the staging pipeline.

        The runner only sequences dataset services and augmentation services;
        JSON-I/O, path containment, metadata formats, rollback, shard writing
        and dataset validation all live in ``mmcrawler_datasets``.
        """
        dataset_paths = self._dataset_paths
        writer_settings = self._writer_settings
        if not writer_settings.write_jsonl:
            raise ValueError(
                "augmentation requires datasets.training.writer.write_jsonl=true"
            )

        project_paths = self._project_paths
        resolved_training_root, final_output_directory = (
            resolve_snapshot_directories(
                paths=dataset_paths,
                path_resolver=project_paths,
                training_root=training_root,
                output_root=output_root,
            )
        )

        try:
            with staged_snapshot(
                final_snapshot_root=final_output_directory,
                source_directory=resolved_training_root,
                logger=self._logger,
            ) as staging_directory:
                split_rows = read_snapshot_rows(
                    paths=dataset_paths,
                    output_directory=staging_directory,
                )
                train_rows = tuple(dict(row) for row in split_rows.train)
                validation_rows = tuple(
                    dict(row) for row in split_rows.validation
                )
                test_rows = tuple(dict(row) for row in split_rows.test)
                row_paths = snapshot_row_paths(
                    paths=dataset_paths,
                    output_directory=staging_directory,
                )
                augmented_train_rows, report = _augment_train_rows(
                    workflow=self._augmentation_workflow,
                    output_directory=staging_directory,
                    train_rows=train_rows,
                    train_path=row_paths["train"],
                )
                write_snapshot_rows(
                    path=row_paths["train"],
                    rows=augmented_train_rows,
                )
                write_augmentation_artifacts(
                    built_at=self._clock.now(),
                    output_directory=staging_directory,
                    original_train_rows=train_rows,
                    augmented_train_rows=augmented_train_rows,
                    val_rows=validation_rows,
                    test_rows=test_rows,
                    report=report,
                )
                _postprocess_snapshot(
                    dataset_paths=dataset_paths,
                    writer_settings=writer_settings,
                    output_directory=staging_directory,
                    augmented_train_rows=augmented_train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                    workers=_DEFAULT_POSTPROCESSING_WORKERS,
                )
        except (RuntimeError, OSError, ValueError) as exc:
            self._logger.exception(
                "augmented_training_snapshot_failed",
                training_root=resolved_training_root.as_posix(),
                output_directory=final_output_directory.as_posix(),
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            raise

        self._logger.info(
            "augmented_training_snapshot_built",
            training_root=resolved_training_root.as_posix(),
            augmented_training_directory=final_output_directory.as_posix(),
            train_original_samples=report.original_samples,
            train_augmented_samples=report.augmented_samples,
            train_variants_added=report.variants_added,
            val_samples_copied=len(validation_rows),
            test_samples_copied=len(test_rows),
            augmented_dataset_total_samples=(
                len(augmented_train_rows)
                + len(validation_rows)
                + len(test_rows)
            ),
        )
