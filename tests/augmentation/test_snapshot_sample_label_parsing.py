"""Strict label parsing for snapshot sample mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.training_samples.snapshot_mapping import (
    _optional_exact_label,
    build_snapshot_sample,
)


def test_boolean_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer or null"):
        _optional_exact_label(True)

    with pytest.raises(ValueError, match="integer or null"):
        build_snapshot_sample(
            payload=_record(label=True),
            dataset_root=Path("."),
            source_path=Path("train.jsonl"),
            line_number=1,
        )


def test_fractional_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer or null"):
        _optional_exact_label(3.8)

    with pytest.raises(ValueError, match="integer or null"):
        _optional_exact_label(7.0)


def test_decimal_string_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer or null"):
        _optional_exact_label("7.9")

    with pytest.raises(ValueError, match="integer or null"):
        _optional_exact_label("3")


def test_exact_integer_label_is_preserved() -> None:
    assert _optional_exact_label(None) is None
    assert _optional_exact_label(0) == 0
    assert _optional_exact_label(3) == 3
    assert _optional_exact_label(42) == 42

    sample = build_snapshot_sample(
        payload=_record(label=7),
        dataset_root=Path("."),
        source_path=Path("train.jsonl"),
        line_number=1,
    )
    assert sample.label == 7


def _record(*, label: object) -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "sample_id": "sample-1",
        "record_id": "record-1",
        "modality": "text",
        "objects": [],
        "text": "hello",
        "label": label,
        "task_target": {"task_type": "classification"},
    }
