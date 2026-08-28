from __future__ import annotations

import os
from pathlib import Path

import pytest

from evaluator.leakage.contracts import CATEGORIES
from evaluator.leakage.report import generate_report


def _record(
    *,
    dataset_id: str,
    sample_id: str,
    lineage_key: str,
    modality: str = "text",
    partition: str = "train",
    content_hash: str | None = None,
    fingerprints: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "partition": partition,
        "lineage_key": lineage_key,
        "modality": modality,
        "content_fingerprints": fingerprints or {},
    }
    if content_hash is not None:
        record["content_hash"] = content_hash
    return record


def _zero_minimums() -> dict[str, float]:
    return dict.fromkeys(CATEGORIES, 0.0)


def test_invalid_fingerprints_do_not_count_as_coverage(tmp_path: Path) -> None:
    report = generate_report(
        left_records=[
            _record(
                dataset_id="left",
                sample_id="image-1",
                lineage_key="left-image-1",
                modality="image",
                content_hash="not-a-sha256",
                fingerprints={
                    "image_phash": "not-hex",
                    "image_dhash": "f" * 15,
                },
            )
        ],
        right_records=[],
        output_path=tmp_path / "leakage.json",
        minimum_coverage=_zero_minimums(),
    )

    coverage = report.coverage_by_category
    assert coverage["content_hash"].left.eligible == 1
    assert coverage["content_hash"].left.with_evidence == 0
    assert coverage["image_phash"].left.eligible == 1
    assert coverage["image_phash"].left.with_evidence == 0
    assert coverage["image_dhash"].left.eligible == 1
    assert coverage["image_dhash"].left.with_evidence == 0


def test_valid_fingerprints_count_as_coverage(tmp_path: Path) -> None:
    report = generate_report(
        left_records=[
            _record(
                dataset_id="left",
                sample_id="image-1",
                lineage_key="left-image-1",
                modality="image",
                content_hash="a" * 64,
                fingerprints={
                    "image_phash": "0" * 16,
                    "image_dhash": "f" * 16,
                },
            )
        ],
        right_records=[],
        output_path=tmp_path / "leakage.json",
        minimum_coverage=_zero_minimums(),
    )

    coverage = report.coverage_by_category
    assert coverage["content_hash"].left.with_evidence == 1
    assert coverage["image_phash"].left.with_evidence == 1
    assert coverage["image_dhash"].left.with_evidence == 1


def test_duplicate_identity_is_rejected_without_any_evidence(
    tmp_path: Path,
) -> None:
    duplicate = _record(
        dataset_id="dataset",
        sample_id="sample",
        lineage_key="lineage",
        modality="unknown",
    )

    with pytest.raises(ValueError, match="duplicate leakage identity"):
        generate_report(
            left_records=[duplicate, dict(duplicate)],
            right_records=[],
            output_path=tmp_path / "leakage.json",
            minimum_coverage=_zero_minimums(),
        )


def test_same_identity_across_sides_is_reported_as_leakage(
    tmp_path: Path,
) -> None:
    row = _record(
        dataset_id="dataset",
        sample_id="same-sample",
        lineage_key="same-lineage",
        content_hash="a" * 64,
    )

    report = generate_report(
        left_records=[row],
        right_records=[dict(row)],
        output_path=tmp_path / "leakage.json",
        minimum_coverage=_zero_minimums(),
    )

    assert report.passed is False
    assert report.overlap_count_by_category["content_hash"] == 1


def test_content_family_cross_split_is_a_release_violation(
    tmp_path: Path,
) -> None:
    report = generate_report(
        left_records=[
            _record(
                dataset_id="left",
                sample_id="train-sample",
                lineage_key="family-1",
                partition="train",
                content_hash="a" * 64,
            )
        ],
        right_records=[
            _record(
                dataset_id="right",
                sample_id="val-sample",
                lineage_key="family-1",
                partition="val",
                content_hash="b" * 64,
            )
        ],
        output_path=tmp_path / "leakage.json",
        minimum_coverage=_zero_minimums(),
    )

    assert report.passed is False
    assert report.violations == (
        "content_family_cross_split:family-1:train,val",
    )


def test_summary_write_is_atomic_when_publish_fails(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "leakage.json"
    output_path.write_text("previous-summary\n", encoding="utf-8")

    def fail_replace(
        source: os.PathLike[str], target: os.PathLike[str]
    ) -> None:
        del source, target
        raise OSError("replace failed")

    with pytest.raises(OSError, match="replace failed"):
        generate_report(
            left_records=[],
            right_records=[],
            output_path=output_path,
            minimum_coverage=_zero_minimums(),
            replace=fail_replace,
        )

    assert output_path.read_text(encoding="utf-8") == "previous-summary\n"
    assert not list(tmp_path.glob(".leakage.json.*.tmp"))


def test_detailed_evidence_write_is_atomic_when_publish_fails(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "leakage.json"
    detail_path = tmp_path / "leakage-details.jsonl"
    detail_path.write_text("previous-details\n", encoding="utf-8")
    real_replace = os.replace

    def fail_detail_replace(
        source: os.PathLike[str],
        target: os.PathLike[str],
    ) -> None:
        if Path(target) == detail_path:
            raise OSError("detail replace failed")
        real_replace(source, target)

    with pytest.raises(OSError, match="detail replace failed"):
        generate_report(
            left_records=[],
            right_records=[],
            output_path=output_path,
            detailed_evidence_path=detail_path,
            minimum_coverage=_zero_minimums(),
            replace=fail_detail_replace,
        )

    assert detail_path.read_text(encoding="utf-8") == "previous-details\n"
    assert not output_path.exists()
    assert not list(tmp_path.glob(".leakage-details.jsonl.*.tmp"))
