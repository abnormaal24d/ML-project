"""Regression coverage for the concrete SplitAssigner module location."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from mmcrawler_datasets.schema import SplitAssigner


def test_split_assigner_has_one_module_location() -> None:
    assert SplitAssigner.__module__ == "mmcrawler_datasets.schema"
    assert (
        importlib.util.find_spec("crawler.curation.media.split_assigner")
        is None
    )
    assert (
        importlib.util.find_spec("crawler.curation.snapshots.composition")
        is None
    )
    assert (
        importlib.util.find_spec("crawler.curation.snapshots.runtime") is None
    )


def test_affected_modules_import() -> None:
    module_names = (
        "crawler.curation.media.cleared_timed_media_records",
        "crawler.curation.media.context.timed_media_coverage",
        "mmcrawler_datasets.curated.image",
        "crawler.curation.snapshots.dataset_assembly.curated_record_loader",
        "orchestration.composition.curated_snapshot",
        "orchestration.workflow.curated_snapshot_runtime",
        "mmcrawler_datasets.assembly.audio",
        "mmcrawler_datasets.assembly.document",
        "mmcrawler_datasets.assembly.image",
        "mmcrawler_datasets.assembly.text",
        "mmcrawler_datasets.assembly.video",
        "mmcrawler_datasets.assembly.build",
        "mmcrawler_datasets.snapshots.training_builder",
        "orchestration.workflow.dataset_preprocessing",
    )

    for module_name in module_names:
        importlib.import_module(module_name)


def test_split_assigner_is_deterministic_and_covers_small_sets() -> None:
    assigner = SplitAssigner(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    assert assigner.assign("group-a") == assigner.assign("group-a")
    assignments = assigner.assign_many(keys=("a", "b", "c", "a"))
    assert set(assignments) == {"a", "b", "c"}
    assert set(assignments.values()) == {"train", "val", "test"}


@pytest.mark.parametrize("ratio", (-0.1, float("inf"), float("nan"), True))
def test_split_assigner_rejects_invalid_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        SplitAssigner(train_ratio=ratio, val_ratio=0.1, test_ratio=0.1)
