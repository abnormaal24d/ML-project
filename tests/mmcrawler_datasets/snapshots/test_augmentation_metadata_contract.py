from __future__ import annotations

import pytest

from mmcrawler_datasets.snapshots.training_metadata import (
    UnsupportedSnapshotSchemaError,
    replace_summary,
)

SUMMARY = {
    "samples_total": 1,
    "splits": {"train": 1, "val": 0, "test": 0},
    "modalities": {"text": 1},
    "tasks": {"representation": 1},
    "tasks_by_split": {
        "train": {"representation": 1},
        "val": {},
        "test": {},
    },
}


def test_removed_summary_fields_fail_instead_of_coexisting() -> None:
    payload = {"sample_count": 9}
    with pytest.raises(
        UnsupportedSnapshotSchemaError, match="unsupported summary"
    ):
        replace_summary(
            payload=payload,
            summary=SUMMARY,
        )


def test_canonical_summary_is_replaced_atomically() -> None:
    payload: dict[str, object] = {"schema_version": "3.0"}
    replace_summary(
        payload=payload,
        summary=SUMMARY,
    )
    assert payload["samples_total"] == 1
    assert payload["splits"] == {"train": 1, "val": 0, "test": 0}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("samples_total", True),
        ("splits", []),
        ("modalities", {"text": -1}),
        ("tasks", {"representation": 1.5}),
        ("tasks_by_split", {"train": []}),
    ),
)
def test_malformed_canonical_summary_is_rejected(
    field: str,
    value: object,
) -> None:
    summary = dict(SUMMARY)
    summary[field] = value

    with pytest.raises(UnsupportedSnapshotSchemaError, match="summary"):
        replace_summary(payload={}, summary=summary)
