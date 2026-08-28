from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.safe_io import resolve_dataset_reference


def test_dataset_reference_rejects_relative_traversal(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        resolve_dataset_reference(
            dataset_root=dataset,
            reference="../secret.pt",
            label="tensor",
        )


def test_allowed_absolute_reference_still_must_be_contained(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "secret.pt"
    outside.write_bytes(b"private")

    with pytest.raises(ValueError, match="escapes"):
        resolve_dataset_reference(
            dataset_root=dataset,
            reference=outside,
            label="tensor",
            allow_absolute=True,
        )
