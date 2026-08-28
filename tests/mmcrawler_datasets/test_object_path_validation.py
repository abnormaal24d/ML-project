"""Filesystem validation for parsed multimodal object references."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.record_components.validation import (
    assert_object_paths_exist,
)
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample


@pytest.mark.parametrize(
    "modality_name", ("document", "image", "audio", "video")
)
def test_object_path_validation_rejects_directories(
    tmp_path: Path,
    modality_name: str,
) -> None:
    object_directory = tmp_path / modality_name
    object_directory.mkdir()
    sample = MultimodalSample(
        sample_id="sample-1",
        **{modality_name: ModalityObject(path=object_directory)},
    )

    with pytest.raises(FileNotFoundError, match=f"missing {modality_name}"):
        assert_object_paths_exist(
            sample=sample,
            ref_path=tmp_path / "records.jsonl",
            line_number=1,
        )


def test_object_path_validation_accepts_regular_file(tmp_path: Path) -> None:
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image")
    sample = MultimodalSample(
        sample_id="sample-1",
        image=ModalityObject(path=image_path),
    )

    assert_object_paths_exist(
        sample=sample,
        ref_path=tmp_path / "records.jsonl",
        line_number=1,
    )
