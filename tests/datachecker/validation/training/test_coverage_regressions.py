from __future__ import annotations

from datachecker.validation.training.coverage import (
    ALIGNMENT,
    SPLIT,
    CoverageLimits,
    CoverageSnapshot,
    check_coverage,
    coverage_ratio,
)


def test_coverage_uses_only_the_configured_required_splits() -> None:
    snapshot = CoverageSnapshot(
        modalities={"text": 1, "image": 1},
        tasks={},
        splits={"train": {"text": 1, "image": 1}},
        aligned=2,
        total=2,
    )
    limits = CoverageLimits(
        modalities={"text": 1, "image": 1},
        tasks={},
        min_alignment=0.0,
        required_splits=("train",),
    )

    issues = check_coverage(snapshot, limits)

    assert not [issue for issue in issues if issue.code == SPLIT]


def test_alignment_gate_uses_the_unrounded_coverage_ratio() -> None:
    snapshot = CoverageSnapshot(
        modalities={},
        tasks={},
        splits={},
        aligned=99_996,
        total=100_000,
    )
    limits = CoverageLimits(
        modalities={},
        tasks={},
        min_alignment=0.99999,
    )

    issues = check_coverage(snapshot, limits)

    assert coverage_ratio(99_996, 100_000) == 0.99996
    assert [(issue.code, issue.observed) for issue in issues] == [
        (ALIGNMENT, 0.99996),
    ]
