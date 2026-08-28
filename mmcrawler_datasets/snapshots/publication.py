"""Transactional staging and publication of immutable snapshots."""

from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from mmcrawler_datasets.snapshots.errors import SnapshotBuildError

_STAGING_PREFIX = ".staging-"
_REPLACED_PREFIX = ".replaced-"


class SnapshotRecoveryError(SnapshotBuildError):
    """Raised when a missing snapshot cannot be recovered unambiguously."""


class SnapshotReplacementRecoveryError(SnapshotBuildError):
    """Raised when snapshot promotion and backup restoration both fail."""

    def __init__(
        self,
        *,
        target: Path,
        staged: Path,
        backup: Path,
    ) -> None:
        self.target = target
        self.staged = staged
        self.backup = backup
        super().__init__(
            "snapshot promotion and backup restoration both failed: "
            f"target={target}, staged={staged}, backup={backup}"
        )


@contextmanager
def staged_snapshot(
    *,
    final_snapshot_root: Path,
    source_directory: Path | None = None,
    logger: Any | None = None,
    replace_existing: bool = True,
) -> Iterator[Path]:
    """Build outside the published directory and promote only after success.

    When ``source_directory`` is provided, its contents are copied into the
    staging directory using hard links with copy fallback before the caller
    runs. Any leftover managed directories from a previous interrupted run
    are recovered or removed first.

    If ``replace_existing`` is False and the target already exists, a
    SnapshotBuildError is raised (immutable snapshot semantics).
    """

    target = final_snapshot_root.resolve()
    if target.exists() and not target.is_dir():
        raise SnapshotBuildError(
            f"snapshot target is not a directory: {target}"
        )

    if target.exists() and not replace_existing:
        raise SnapshotBuildError(
            f"immutable snapshot already exists: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    removed = recover_or_cleanup_managed_directories(target=target)
    if removed and logger is not None:
        logger.info(
            "stale_staging_directories_removed",
            target_directory=target.as_posix(),
            removed_count=len(removed),
        )

    staging = create_staging_directory(target=target)
    if source_directory is not None:
        shutil.copytree(
            source_directory,
            staging,
            dirs_exist_ok=True,
            copy_function=partial(hardlink_then_copy, logger=logger),
        )
    try:
        yield staging
        replace_directory(target=target, staged=staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def create_staging_directory(
    *,
    target: Path,
    token: str | None = None,
) -> Path:
    """Create a sibling staging directory for one snapshot target."""

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (
        f"{_STAGING_PREFIX}{target.name}-{token or uuid.uuid4().hex}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def replace_directory(
    *,
    target: Path,
    staged: Path,
    token: str | None = None,
) -> None:
    """Atomically replace ``target`` with ``staged``, keeping a backup.

    The previous target is moved to a ``.replaced-*`` backup that is
    destroyed only after a successful commit or restoration.
    """

    target = target.resolve()
    staged = staged.resolve()
    _require_child(path=staged, parent=target.parent)
    if not staged.is_dir():
        raise FileNotFoundError(f"staged directory is missing: {staged}")

    backup: Path | None = None
    promotion_succeeded = False
    restoration_succeeded = False

    try:
        if target.exists():
            backup = target.parent / (
                f"{_REPLACED_PREFIX}{target.name}-{token or uuid.uuid4().hex}"
            )
            os.replace(target, backup)

        os.replace(staged, target)
        promotion_succeeded = True

    except OSError as promotion_error:
        if backup is not None and backup.exists() and not target.exists():
            try:
                os.replace(backup, target)
                restoration_succeeded = True
            except OSError as restoration_error:
                raise SnapshotReplacementRecoveryError(
                    target=target,
                    staged=staged,
                    backup=backup,
                ) from restoration_error

        raise SnapshotBuildError(
            f"failed to publish training snapshot: {target}"
        ) from promotion_error

    finally:
        if (
            backup is not None
            and backup.exists()
            and (promotion_succeeded or restoration_succeeded)
        ):
            shutil.rmtree(backup, ignore_errors=True)


def remove_managed_directory(*, path: Path, parent: Path) -> None:
    """Remove a staging or replaced directory under ``parent``."""

    resolved_parent = parent.resolve()
    resolved_path = path.resolve()
    _require_child(path=resolved_path, parent=resolved_parent)
    name = resolved_path.name
    if not (
        name.startswith(_STAGING_PREFIX) or name.startswith(_REPLACED_PREFIX)
    ):
        raise ValueError(f"refusing to remove unmanaged directory: {path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def recover_or_cleanup_managed_directories(
    *,
    target: Path,
) -> tuple[Path, ...]:
    """Recover or clean managed staging/backup directories for one target.

    Staging directories are always disposable leftovers and are removed.

    Backup (``.replaced-*``) directories are recovery assets:
    - removed only when the final target already exists;
    - restored into ``target`` when the target is missing and exactly one
      backup exists;
    - otherwise raise when multiple backups leave recovery ambiguous.
    """

    target = target.resolve()
    parent = target.parent
    if not parent.exists():
        return ()

    staging_directories, backup_directories = _list_managed_directories(
        target=target
    )

    removed: list[Path] = []
    for staging in staging_directories:
        remove_managed_directory(path=staging, parent=parent)
        removed.append(staging)

    if target.exists():
        for backup in backup_directories:
            remove_managed_directory(path=backup, parent=parent)
            removed.append(backup)
        return tuple(removed)

    if len(backup_directories) == 1:
        os.replace(backup_directories[0], target)
        return tuple(removed)

    if backup_directories:
        raise SnapshotRecoveryError(
            "target is missing and recovery state is ambiguous: "
            f"found {len(backup_directories)} backup directories for "
            f"{target.name}"
        )

    return tuple(removed)


def hardlink_then_copy(
    src: str,
    dst: str,
    *,
    logger: Any | None = None,
) -> str:
    """Link a file into the staging tree, falling back to a copy."""

    try:
        os.link(src, dst)
    except OSError as link_error:
        if logger is not None:
            logger.debug(
                "hardlink_fallback_to_copy",
                source_path=src,
                destination_path=dst,
                error_type=type(link_error).__name__,
                error_message=str(link_error) or None,
                errno=getattr(link_error, "errno", None),
            )
        try:
            shutil.copy2(src, dst)
        except OSError as copy_error:
            if logger is not None:
                logger.exception(
                    "file_copy_failed",
                    source_path=src,
                    destination_path=dst,
                    hardlink_error_type=type(link_error).__name__,
                    error_type=type(copy_error).__name__,
                    error_message=str(copy_error) or None,
                )
            raise
    return dst


def _list_managed_directories(
    *,
    target: Path,
) -> tuple[list[Path], list[Path]]:
    staging_prefix = f"{_STAGING_PREFIX}{target.name}-"
    backup_prefix = f"{_REPLACED_PREFIX}{target.name}-"
    staging_directories: list[Path] = []
    backup_directories: list[Path] = []

    for child in sorted(target.parent.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(staging_prefix):
            staging_directories.append(child)
        elif name.startswith(backup_prefix):
            backup_directories.append(child)

    return staging_directories, backup_directories


def _require_child(*, path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(
            f"path {path} is outside managed parent {parent}"
        ) from exc


__all__ = [
    "SnapshotRecoveryError",
    "SnapshotReplacementRecoveryError",
    "create_staging_directory",
    "hardlink_then_copy",
    "recover_or_cleanup_managed_directories",
    "remove_managed_directory",
    "replace_directory",
    "staged_snapshot",
]
