"""Recovery behavior for managed snapshot staging directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.snapshots.publication import (
    SnapshotRecoveryError,
    SnapshotReplacementRecoveryError,
    recover_or_cleanup_managed_directories,
    replace_directory,
)


def test_missing_target_restores_single_replaced_directory(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-crash"
    backup = parent / ".replaced-snapshot-crash-abc123"
    backup.mkdir()
    marker = backup / "manifest.json"
    marker.write_text('{"ok": true}', encoding="utf-8")

    assert not target.exists()
    assert backup.exists()

    removed = recover_or_cleanup_managed_directories(target=target)

    assert removed == ()
    assert target.is_dir()
    assert not backup.exists()
    assert (target / "manifest.json").read_text(encoding="utf-8") == (
        '{"ok": true}'
    )


def test_existing_target_removes_backups_and_staging(tmp_path: Path) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-a"
    target.mkdir()
    (target / "live.txt").write_text("live", encoding="utf-8")

    staging = parent / ".staging-snapshot-a-old"
    staging.mkdir()
    backup = parent / ".replaced-snapshot-a-old"
    backup.mkdir()
    (backup / "old.txt").write_text("old", encoding="utf-8")

    removed = recover_or_cleanup_managed_directories(target=target)

    assert set(removed) == {staging, backup}
    assert target.exists()
    assert (target / "live.txt").read_text(encoding="utf-8") == "live"
    assert not staging.exists()
    assert not backup.exists()


def test_missing_target_removes_staging_but_keeps_recovery_path(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-a"
    staging = parent / ".staging-snapshot-a-tmp"
    staging.mkdir()
    (staging / "partial.txt").write_text("partial", encoding="utf-8")
    backup = parent / ".replaced-snapshot-a-keep"
    backup.mkdir()
    (backup / "good.txt").write_text("good", encoding="utf-8")

    removed = recover_or_cleanup_managed_directories(target=target)

    assert removed == (staging,)
    assert not staging.exists()
    assert target.exists()
    assert (target / "good.txt").read_text(encoding="utf-8") == "good"
    assert not backup.exists()


def test_missing_target_with_ambiguous_backups_raises(tmp_path: Path) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-a"
    first = parent / ".replaced-snapshot-a-one"
    second = parent / ".replaced-snapshot-a-two"
    first.mkdir()
    second.mkdir()

    with pytest.raises(SnapshotRecoveryError, match="ambiguous"):
        recover_or_cleanup_managed_directories(target=target)

    assert first.exists()
    assert second.exists()
    assert not target.exists()


def test_failed_restoration_preserves_backup_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-a"
    target.mkdir()
    (target / "old.txt").write_text("old-valid", encoding="utf-8")

    staged = parent / ".staging-snapshot-a-new"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")

    # Fail promotion (staged → target) and restoration (backup → target).
    # Allow the initial target → backup move to succeed.
    call_count = {"n": 0}
    real_replace = __import__("os").replace

    def flaky_replace(src: object, dst: object) -> None:
        call_count["n"] += 1
        src_path = Path(src)
        dst_path = Path(dst)
        # 1st call: target → backup (must succeed)
        if call_count["n"] == 1:
            real_replace(src_path, dst_path)
            return
        # 2nd: staged → target fails
        # 3rd: backup → target fails
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "mmcrawler_datasets.snapshots.publication.os.replace",
        flaky_replace,
    )

    with pytest.raises(SnapshotReplacementRecoveryError) as raised:
        replace_directory(
            target=target,
            staged=staged,
        )

    error = raised.value
    assert not target.exists()
    assert staged.exists()
    assert error.backup.exists()
    assert (error.backup / "old.txt").read_text(encoding="utf-8") == (
        "old-valid"
    )
    assert error.target == target.resolve()
    assert error.staged == staged.resolve()
    assert str(error.backup) in str(error)


def test_successful_promotion_removes_backup(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "sets"
    parent.mkdir()
    target = parent / "snapshot-a"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    staged = parent / ".staging-snapshot-a-new"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")

    replace_directory(
        target=target,
        staged=staged,
    )

    assert target.exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not staged.exists()
    leftovers = list(parent.glob(".replaced-snapshot-a-*"))
    assert leftovers == []
