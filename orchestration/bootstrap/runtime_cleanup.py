"""Runtime cleanup for explicit fresh workflow starts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from config.path_resolution.workflow_artifact_paths import (
    ArtifactPathRegistry,
)
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)

if TYPE_CHECKING:
    from config.settings.root import Settings


def _absolute_without_resolving(
    *,
    project_root: Path,
    path: str | Path,
) -> Path:
    """Return an absolute path without following symbolic links."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return Path(os.path.abspath(candidate))


def _remove_project_target(
    *,
    project_root: Path,
    target: Path,
    protected_paths: tuple[Path, ...],
) -> bool:
    """Remove one project-owned target without following its final link."""
    try:
        target.lstat()
    except FileNotFoundError:
        return False

    if target.is_symlink():
        target.unlink()
        return True

    if target.is_junction():
        target.rmdir()
        return True

    resolved_parent = target.parent.resolve()
    if not resolved_parent.is_relative_to(project_root):
        raise ValueError(
            f"Cleanup target reached through a path outside project root: {target}"
        )

    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(project_root):
        raise ValueError(f"Cleanup target escapes project root: {target}")

    if resolved_target == project_root:
        raise ValueError("Project root may not be removed.")

    if resolved_target.is_mount():
        raise ValueError("mount point may not be recursively removed.")

    for protected_path in protected_paths:
        resolved_protected = protected_path.resolve()
        if (
            resolved_target == resolved_protected
            or resolved_protected.is_relative_to(resolved_target)
        ):
            raise ValueError(
                f"Cleanup target contains a protected path: {target}"
            )

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return True


def clean_runtime_state(*, settings: Settings) -> list[str]:
    """Remove stale outputs without following links outside their entries."""

    project_root = Path(settings.paths.root).expanduser().resolve()
    registry = ArtifactPathRegistry(
        settings=settings.collection.datachecker,
        dataset_paths=settings.datasets.paths,
    )
    control_directory = CrawlerControlDirectory(
        settings=settings.crawler,
        project_root=project_root,
    )

    control_root = _absolute_without_resolving(
        project_root=project_root,
        path=control_directory.path(),
    )
    configured_state_directory = Path(
        settings.crawler.state.state_subdirectory
    )
    if configured_state_directory.is_absolute():
        state_directory = configured_state_directory
    elif str(configured_state_directory).startswith("."):
        state_directory = project_root / configured_state_directory
    else:
        state_directory = control_root / configured_state_directory

    targets = (
        project_root / "runtime/cache",
        state_directory,
        registry.checkpoint_root(),
        registry.artifacts_root(),
        registry.raw_runs_root(),
        registry.curated_root(),
        registry.training_sets_root(),
        registry.augmented_training_sets_root(),
        settings.augmentation.cache_directory,
        control_directory.pause_flag_path(),
        control_directory.stop_flag_path(),
    )
    protected_paths = (
        project_root / "runtime/locks",
        project_root / "runtime/logs",
    )

    removed: list[str] = []
    seen: set[Path] = set()
    for configured_target in targets:
        target = _absolute_without_resolving(
            project_root=project_root,
            path=configured_target,
        )
        if target in seen:
            continue
        seen.add(target)

        if _remove_project_target(
            project_root=project_root,
            target=target,
            protected_paths=protected_paths,
        ):
            removed.append(str(target))

    return removed
