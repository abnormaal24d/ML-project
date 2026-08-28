from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmcrawler_datasets.snapshots.training_dataset_manifest import (
    DatasetManifestError,
    read_dataset_counts,
)


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    (root / "dataset_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "samples_total": 3,
        "splits": {"train": 1, "val": 1, "test": 1},
        "tasks": {"representation": 3},
        "tasks_by_split": {
            "train": {"representation": 1},
            "val": {"representation": 1},
            "test": {"representation": 1},
        },
        "modalities": {"text": 3},
        "valid": True,
        "errors": [],
    }


def test_manifest_is_required_and_never_reconstructed(tmp_path: Path) -> None:
    with pytest.raises(DatasetManifestError):
        read_dataset_counts(dataset_root=tmp_path)


def test_schema_3_manifest_is_the_only_acceptance_source(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, _valid_manifest())
    result = read_dataset_counts(dataset_root=tmp_path)
    assert result["total"] == 3
    assert result["splits"] == {"train": 1, "val": 1, "test": 1}
    assert result["validation_valid"] is True


def test_old_schema_and_inconsistent_total_fail_closed(tmp_path: Path) -> None:
    payload = _valid_manifest()
    payload["schema_version"] = "2.0"
    _write_manifest(tmp_path, payload)
    with pytest.raises(
        DatasetManifestError, match="Unsupported|unsupported|schema"
    ):
        read_dataset_counts(dataset_root=tmp_path)

    payload = _valid_manifest()
    payload["samples_total"] = 9
    _write_manifest(tmp_path, payload)
    with pytest.raises(DatasetManifestError, match="sum"):
        read_dataset_counts(dataset_root=tmp_path)
