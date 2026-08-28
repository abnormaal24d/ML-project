from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluator.leakage.report import (
    load_report,
    write_training_snapshot_leakage,
)
from mmcrawler_datasets.assembly.build import SampleBuildResult


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    split: str
    content_family_id: str
    content_hash: str
    snapshot_id: str = "snapshot-1"

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "snapshot_id": self.snapshot_id,
            "split": self.split,
            "modality": "text",
            "content_hash": self.content_hash,
            "content_fingerprints": {},
        }


def test_val_test_overlap_fails_the_canonical_leakage_report(
    tmp_path: Path,
) -> None:
    samples = SampleBuildResult(
        train_samples=(),
        val_samples=(
            _Sample(
                sample_id="val-1",
                split="val",
                content_family_id="family-val",
                content_hash="a" * 64,
            ),
        ),
        test_samples=(
            _Sample(
                sample_id="test-1",
                split="test",
                content_family_id="family-test",
                content_hash="a" * 64,
            ),
        ),
    )
    training_directory = tmp_path / "training"

    write_training_snapshot_leakage(
        training_directory=training_directory,
        samples=samples,
    )

    canonical = load_report(
        training_directory / "evaluation" / "leakage_report.json"
    )
    val_test = load_report(
        training_directory / "evaluation" / "leakage_val_vs_test.json"
    )
    assert val_test.total_overlap_count > 0
    assert canonical.passed is False
    assert "cross_split_overlap:val:test" in canonical.violations
