"""Path-safety for modality/task derived snapshot views."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.snapshots.training_outputs import (
    _contained_group_root,
    _require_safe_group_name,
    _resolve_group_name,
    _rewrite_grouped_views,
)


def test_task_group_rejects_parent_directory_component() -> None:
    with pytest.raises(
        ValueError, match="not a safe directory name|canonical"
    ):
        _resolve_group_name(
            row={"task_target": {"task_type": ".."}},
            field_name="task_type",
        )

    with pytest.raises(ValueError, match="not a safe directory name"):
        _require_safe_group_name("..", field="task_type")


def test_task_group_rejects_path_separators() -> None:
    with pytest.raises(ValueError, match="unsupported characters|canonical"):
        _resolve_group_name(
            row={"task_target": {"task_type": "a/b"}},
            field_name="task_type",
        )

    with pytest.raises(ValueError, match="unsupported characters"):
        _require_safe_group_name("a\\b", field="task_type")

    with pytest.raises(ValueError, match="unsupported characters"):
        _require_safe_group_name("evil:name", field="modality")


def test_derived_group_path_remains_within_view_root(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()

    contained = _contained_group_root(root=root, group_name="classification")
    assert contained.is_relative_to(root.resolve())

    # Even if a malicious name slipped past earlier checks, containment fails.
    with pytest.raises(ValueError, match="escapes its root"):
        _contained_group_root(root=root, group_name="..")


def test_task_group_does_not_write_outside_tasks_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "views" / "tasks"
    outside_before = {
        path.name for path in tmp_path.iterdir() if path.is_file()
    }

    with pytest.raises(ValueError):
        _rewrite_grouped_views(
            root=root,
            split_rows={
                "train": (
                    {
                        "sample_id": "s1",
                        "task_target": {"task_type": ".."},
                    },
                ),
                "val": (),
                "test": (),
            },
            filenames={
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
            },
            field_name="task_type",
        )

    # No split files may appear next to the views parent (outside tasks/).
    outside_after = {
        path.name for path in tmp_path.iterdir() if path.is_file()
    }
    assert outside_after == outside_before
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "val.jsonl").exists()
    assert not (tmp_path / "test.jsonl").exists()


def test_distinct_task_names_cannot_collapse_to_same_directory() -> None:
    # Lossy replacement used to map "a/b" and "a_b" onto the same directory.
    with pytest.raises(ValueError):
        left = _resolve_group_name(
            row={"task_target": {"task_type": "a/b"}},
            field_name="task_type",
        )
        right = _resolve_group_name(
            row={"task_target": {"task_type": "a_b"}},
            field_name="task_type",
        )
        assert left != right

    # Two different valid names stay distinct after canonicalization.
    first = _resolve_group_name(
        row={"task_target": {"task_type": "text_classification"}},
        field_name="task_type",
    )
    second = _resolve_group_name(
        row={"task_target": {"task_type": "image_classification"}},
        field_name="task_type",
    )
    assert first != second


def test_modality_allowlist_rejects_unknown_and_traversal() -> None:
    with pytest.raises(ValueError, match="allowed set|safe directory"):
        _resolve_group_name(row={"modality": ".."}, field_name="modality")

    with pytest.raises(ValueError, match="allowed set"):
        _resolve_group_name(
            row={"modality": "not_a_modality"},
            field_name="modality",
        )

    assert (
        _resolve_group_name(row={"modality": "text"}, field_name="modality")
        == "text"
    )
