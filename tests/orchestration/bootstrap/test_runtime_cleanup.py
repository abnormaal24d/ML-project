from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.bootstrap.runtime_cleanup import (
    _remove_project_target,
    clean_runtime_state,
)


def _remove(*, project_root: Path, target: Path) -> bool:
    return _remove_project_target(
        project_root=project_root,
        target=target,
        protected_paths=(
            project_root / "runtime" / "locks",
            project_root / "runtime" / "logs",
        ),
    )


def _directory_link(*, link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if exc.winerror == 1314:
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        raise


def _settings_for_cleanup(project_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(root=project_root),
        collection=SimpleNamespace(datachecker=object()),
        datachecker=object(),
        datasets=SimpleNamespace(
            paths=SimpleNamespace(
                workflow_artifacts_directory=project_root / "artifacts",
                raw_output_directory=project_root / "data" / "raw" / "runs",
                curated_output_directory=project_root / "data" / "curated",
                training_output_directory=(
                    project_root / "data" / "interim" / "training_sets"
                ),
                augmented_training_output_directory=(
                    project_root
                    / "data"
                    / "interim"
                    / "augmented_training_sets"
                ),
                training_checkpoint_directory=(
                    project_root / "runtime" / "training" / "checkpoints"
                ),
            )
        ),
        crawler=SimpleNamespace(
            control_directory="runtime/control",
            state=SimpleNamespace(state_subdirectory="state"),
            pause_flag_filename="pause",
            stop_flag_filename="stop",
        ),
        augmentation=SimpleNamespace(
            cache_directory=project_root / "runtime" / "augmentation-cache"
        ),
    )


def test_fresh_run_removes_an_ordinary_project_directory(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    target = project_root / "runtime" / "cache"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    assert _remove(project_root=project_root, target=target)
    assert not target.exists()


def test_fresh_run_unlinks_internal_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    internal_root = project_root / "internal"
    internal_root.mkdir(parents=True)
    protected_file = internal_root / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    link = project_root / "runtime" / "cache"
    _directory_link(link=link, target=internal_root)

    assert _remove(project_root=project_root, target=link)
    assert not link.is_symlink()
    assert protected_file.read_text(encoding="utf-8") == "keep"


def test_public_cleanup_keeps_the_final_symlink_lexical(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    internal_root = project_root / "internal"
    internal_root.mkdir(parents=True)
    protected_file = internal_root / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    link = project_root / "runtime" / "cache"
    _directory_link(link=link, target=internal_root)

    removed = clean_runtime_state(
        settings=_settings_for_cleanup(project_root),
    )

    assert str(link) in removed
    assert not link.is_symlink()
    assert protected_file.read_text(encoding="utf-8") == "keep"


def test_fresh_run_unlinks_external_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    project_root.mkdir()
    external_root.mkdir()
    protected_file = external_root / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    link = project_root / "runtime" / "cache"
    _directory_link(link=link, target=external_root)

    assert _remove(project_root=project_root, target=link)
    assert not link.is_symlink()
    assert protected_file.read_text(encoding="utf-8") == "keep"


def test_fresh_run_rejects_target_reached_through_external_symlink(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    project_root.mkdir()
    target = external_root / "cache"
    target.mkdir(parents=True)
    protected_file = target / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    _directory_link(link=project_root / "runtime", target=external_root)

    with pytest.raises(ValueError, match="reached through a path outside"):
        _remove(
            project_root=project_root,
            target=project_root / "runtime" / "cache",
        )

    assert protected_file.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", ("locks", "logs"))
def test_fresh_run_never_removes_protected_runtime_paths(
    tmp_path: Path,
    name: str,
) -> None:
    project_root = tmp_path / "project"
    target = project_root / "runtime" / name
    target.mkdir(parents=True)

    with pytest.raises(ValueError, match="contains a protected path"):
        _remove(project_root=project_root, target=target)

    assert target.exists()


def test_fresh_run_rejects_mountpoint_before_recursive_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    target = project_root / "runtime" / "cache"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(Path, "is_mount", lambda _path: True)

    with pytest.raises(ValueError, match="mount point"):
        _remove(project_root=project_root, target=target)

    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_fresh_run_removes_broken_symlink_without_resolving_it(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    link = project_root / "runtime" / "cache"
    _directory_link(link=link, target=project_root / "missing")

    assert link.is_symlink()
    assert _remove(project_root=project_root, target=link)
    assert not link.is_symlink()


def test_fresh_run_treats_a_junction_as_a_non_recursive_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    target = project_root / "runtime" / "cache"
    target.mkdir(parents=True)
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == target,
    )

    assert _remove(project_root=project_root, target=target)
    assert not target.exists()
