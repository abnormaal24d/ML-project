"""Read canonical training snapshot split rows and resolve their paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings.datasets import DatasetPathSettings
from mmcrawler_datasets.safe_io import read_jsonl

if TYPE_CHECKING:
    from config.path_resolution.project_paths import ProjectPaths


@dataclass(frozen=True, slots=True)
class SnapshotRows:
    """Rows read from one canonical training snapshot."""

    train: tuple[Mapping[str, object], ...]
    validation: tuple[Mapping[str, object], ...]
    test: tuple[Mapping[str, object], ...]


def snapshot_row_paths(
    *,
    paths: DatasetPathSettings,
    output_directory: Path,
) -> Mapping[str, Path]:
    splits_root = output_directory / paths.training_splits_directory
    return {
        "train": splits_root / paths.training_train_filename,
        "validation": splits_root / paths.training_val_filename,
        "test": splits_root / paths.training_test_filename,
    }


def resolve_snapshot_directories(
    *,
    paths: DatasetPathSettings,
    path_resolver: ProjectPaths,
    training_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    resolved_training_root = path_resolver.resolve(
        Path(training_root),
        allow_absolute=True,
    )
    resolved_output_root = path_resolver.resolve(
        Path(output_root),
        allow_absolute=True,
    )
    training_sets_root = path_resolver.resolve(
        paths.training_output_directory,
    )
    if resolved_training_root.is_relative_to(training_sets_root):
        final_output = (
            resolved_output_root
            / resolved_training_root.relative_to(training_sets_root)
        )
    else:
        final_output = resolved_output_root / resolved_training_root.name
    return resolved_training_root, final_output


def read_snapshot_rows(
    *,
    paths: DatasetPathSettings,
    output_directory: Path,
) -> SnapshotRows:
    row_paths = snapshot_row_paths(
        paths=paths,
        output_directory=output_directory,
    )
    _require_files(*row_paths.values())
    train_rows = tuple(read_jsonl(path=row_paths["train"]))
    if not train_rows:
        raise ValueError(
            f"training split is empty and cannot be augmented: {row_paths['train']}"
        )
    return SnapshotRows(
        train=train_rows,
        validation=tuple(read_jsonl(path=row_paths["validation"])),
        test=tuple(read_jsonl(path=row_paths["test"])),
    )


def _require_files(*paths: Path) -> None:
    missing = tuple(path for path in paths if not path.is_file())
    if not missing:
        return
    formatted = ", ".join(path.as_posix() for path in missing)
    raise FileNotFoundError(f"missing training split file(s): {formatted}")


__all__ = [
    "SnapshotRows",
    "read_snapshot_rows",
    "resolve_snapshot_directories",
    "snapshot_row_paths",
]
