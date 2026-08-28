"""Higher-level snapshot job for the canonical preprocessing phase.

Dataset preprocessing phase that builds curated and training snapshots.
All static configuration and services are bound by composition; this
module only orchestrates dynamic run inputs through typed capabilities.
"""

from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol

from config.path_resolution.project_paths import join_contained_path
from evaluator.leakage.report import write_training_snapshot_leakage
from mmcrawler_datasets.snapshots.publication import staged_snapshot
from mmcrawler_datasets.snapshots.training_builder import (
    TrainingSnapshotBuildResult,
)
from orchestration.workflow.curated_snapshot_runtime import (
    CuratedSnapshotRuntimeResult,
)


class BuildCuratedSnapshot(Protocol):
    """Builds one curated snapshot for the requested raw-run selection."""

    def __call__(
        self,
        *,
        snapshot_id: str | None,
        raw_run_selection_mode: str | None,
        selected_run_ids: tuple[str, ...] | None,
    ) -> Awaitable[CuratedSnapshotRuntimeResult]: ...


class ResolveTrainingDirectory(Protocol):
    """Resolves the training output directory for one snapshot id."""

    def __call__(
        self,
        *,
        snapshot_id: str,
    ) -> Path: ...


class BuildTrainingSnapshot(Protocol):
    """Assembles one training snapshot from curated artifacts."""

    def __call__(
        self,
        *,
        curated_snapshot_directory: Path,
        training_directory: Path,
        snapshot_id: str,
    ) -> TrainingSnapshotBuildResult: ...


_COVERAGE_AWARE_SELECTION_MODES = frozenset(
    {
        "coverage_combined",
        "coverage-aware",
        "coverage_aware",
        "coverage_best",
    }
)


async def run_preprocessing_phase(
    *,
    raw_run_directory: Path,
    raw_records_manifest_path: Path,
    training_snapshot_id: str | None,
    manifest_filename: str,
    raw_run_selection_mode: str,
    build_curated_snapshot: BuildCuratedSnapshot,
    resolve_training_directory: ResolveTrainingDirectory,
    build_training_snapshot: BuildTrainingSnapshot,
) -> TrainingSnapshotBuildResult:
    """
    Build curated and training dataset artifacts from current crawl output.

    Awaits multimodal preprocessing inside curated snapshot assembly.
    """

    run_id = _raw_run_id(
        raw_run_directory=raw_run_directory,
        raw_records_manifest_path=raw_records_manifest_path,
        manifest_relative_path=manifest_filename,
    )
    coverage_aware_selection = (
        raw_run_selection_mode.strip().lower()
        in _COVERAGE_AWARE_SELECTION_MODES
    )
    curated_result: CuratedSnapshotRuntimeResult = (
        await build_curated_snapshot(
            snapshot_id=None,
            raw_run_selection_mode=(
                raw_run_selection_mode
                if coverage_aware_selection
                else "selected"
            ),
            selected_run_ids=None if coverage_aware_selection else (run_id,),
        )
    )

    effective_snapshot_id = training_snapshot_id or curated_result.snapshot_id

    # Resolve directories using same logic as curated builder
    curated_snapshot_directory = curated_result.snapshot_directory
    training_snapshot_directory = resolve_training_directory(
        snapshot_id=effective_snapshot_id
    )

    with staged_snapshot(
        final_snapshot_root=training_snapshot_directory
    ) as staging_directory:
        result = build_training_snapshot(
            curated_snapshot_directory=curated_snapshot_directory,
            training_directory=staging_directory,
            snapshot_id=effective_snapshot_id,
        )
        write_training_snapshot_leakage(
            training_directory=staging_directory,
            samples=result.samples,
        )

    return TrainingSnapshotBuildResult(
        samples=result.samples,
        snapshot_dir=training_snapshot_directory.resolve(),
    )


def _raw_run_id(
    *,
    raw_run_directory: Path,
    raw_records_manifest_path: Path,
    manifest_relative_path: str,
) -> str:
    if raw_run_directory is None:
        raise ValueError("preprocessing plan must include raw_run_directory")

    if raw_records_manifest_path is None:
        raise ValueError(
            "preprocessing plan must include raw_records_manifest_path"
        )

    run_directory = Path(raw_run_directory).expanduser().resolve()
    manifest_path = Path(raw_records_manifest_path).expanduser().resolve()

    run_id = run_directory.name.strip()
    if not run_id or run_id in {".", ".."}:
        raise ValueError("preprocessing plan raw_run_directory has no run id")

    expected_manifest_path = join_contained_path(
        run_directory,
        manifest_relative_path,
        field_name="manifest_filename",
    ).resolve()

    if manifest_path != expected_manifest_path:
        raise ValueError(
            "preprocessing plan raw_records_manifest_path does not match "
            "the configured raw manifest path"
        )

    if not manifest_path.is_file():
        raise ValueError(
            "preprocessing plan raw_records_manifest_path does not exist"
        )

    return run_id
