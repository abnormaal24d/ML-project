"""Leakage-v2 report orchestration, persistence, and loading."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from evaluator.leakage.contracts import (
    CATEGORIES,
    DEFAULT_MAX_CANDIDATES_PER_RECORD,
    DEFAULT_MAX_INDEXED_RECORDS_PER_CATEGORY,
    DEFAULT_OVERLAP_SAMPLE_LIMIT,
    MAX_REPORT_BYTES,
)
from evaluator.leakage.indexing import (
    SideIndex,
    build_index,
    indexed_content_family_violations,
)
from evaluator.leakage.matching import (
    OverlapSink,
    intersect_exact,
    intersect_near_text,
    intersect_perceptual,
)
from evaluator.leakage.schema import (
    LeakageCategoryCoverage,
    LeakageCoverageSide,
    LeakageReportV2,
)

if TYPE_CHECKING:
    from mmcrawler_datasets.assembly.build import SampleBuildResult
    from mmcrawler_datasets.training_samples.models import TrainingSample

PathLike = str | os.PathLike[str]
ReplaceFunction = Callable[[PathLike, PathLike], None]


@contextmanager
def atomic_text_writer(
    path: Path | None,
    *,
    replace: ReplaceFunction = os.replace,
) -> Iterator[TextIO | None]:
    """Yield a private writer and atomically publish it after success."""

    if path is None:
        yield None
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary_path, path)
        published = True
    finally:
        if not published:
            temporary_path.unlink(missing_ok=True)


def load_report(path: Path) -> LeakageReportV2:
    """Load and structurally validate the sole leakage-v2 format."""

    try:
        size = path.stat().st_size
        if size > MAX_REPORT_BYTES:
            raise ValueError("leakage report exceeds size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid leakage report") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("leakage report root must be an object")
    return LeakageReportV2.from_mapping(payload)


def generate_report(
    *,
    left_records: Iterable[Mapping[str, object]],
    right_records: Iterable[Mapping[str, object]],
    output_path: Path,
    minimum_coverage: Mapping[str, float],
    text_similarity_threshold: float = 0.9,
    image_phash_max_distance: int = 4,
    image_dhash_max_distance: int = 4,
    overlap_sample_limit: int = DEFAULT_OVERLAP_SAMPLE_LIMIT,
    max_indexed_records_per_category: int = (
        DEFAULT_MAX_INDEXED_RECORDS_PER_CATEGORY
    ),
    max_candidates_per_record: int = DEFAULT_MAX_CANDIDATES_PER_RECORD,
    detailed_evidence_path: Path | None = None,
    replace: ReplaceFunction = os.replace,
) -> LeakageReportV2:
    """Build a bounded v2 summary from one streaming pass over each split."""

    _validate_limits(
        text_similarity_threshold=text_similarity_threshold,
        image_phash_max_distance=image_phash_max_distance,
        image_dhash_max_distance=image_dhash_max_distance,
        overlap_sample_limit=overlap_sample_limit,
        max_indexed_records_per_category=max_indexed_records_per_category,
        max_candidates_per_record=max_candidates_per_record,
    )
    coverage_minimums = _coverage_minimums(minimum_coverage)

    left = build_index(
        left_records,
        max_records=max_indexed_records_per_category,
    )
    right = build_index(
        right_records,
        max_records=max_indexed_records_per_category,
    )
    violations = _index_capacity_violations(left=left, right=right)
    violations.extend(
        indexed_content_family_violations(left=left, right=right)
    )

    with atomic_text_writer(
        detailed_evidence_path,
        replace=replace,
    ) as detail_handle:
        sink = OverlapSink(
            sample_limit=overlap_sample_limit,
            detail_handle=detail_handle,
        )
        intersect_exact(left=left, right=right, sink=sink)
        if intersect_near_text(
            left=left,
            right=right,
            sink=sink,
            threshold=text_similarity_threshold,
            max_candidates=max_candidates_per_record,
        ):
            violations.append("candidate_capacity:near_duplicate_text")
        for category, max_distance in (
            ("image_phash", image_phash_max_distance),
            ("image_dhash", image_dhash_max_distance),
        ):
            if intersect_perceptual(
                category=category,
                left=left,
                right=right,
                sink=sink,
                max_distance=max_distance,
                max_candidates=max_candidates_per_record,
            ):
                violations.append(f"candidate_capacity:{category}")

    coverage = _build_coverage(
        left=left,
        right=right,
        minimums=coverage_minimums,
        violations=violations,
    )
    overlap_counts = tuple(
        (category, sink.counts[category]) for category in CATEGORIES
    )
    unique_violations = tuple(dict.fromkeys(sorted(violations)))
    report = LeakageReportV2(
        coverage=coverage,
        overlap_counts=overlap_counts,
        overlaps=tuple(
            sorted(
                sink.sample,
                key=lambda item: (
                    item.category,
                    item.left.key,
                    item.right.key,
                    item.fingerprint.normalized_digest,
                ),
            )
        ),
        violations=unique_violations,
        passed=not unique_violations
        and not any(count for _, count in overlap_counts),
        overlap_sample_limit=overlap_sample_limit,
    )
    with atomic_text_writer(output_path, replace=replace) as report_handle:
        if report_handle is None:
            raise RuntimeError("output_path is required for report writing")
        report_handle.write(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    return report


def write_training_snapshot_leakage(
    *,
    training_directory: Path,
    samples: SampleBuildResult,
) -> None:
    """Persist evaluator-owned leakage evidence for all training splits."""

    report_path = training_directory / "evaluation" / "leakage_report.json"
    train_vs_holdout = generate_report(
        left_records=(
            _training_sample_leakage_record(sample)
            for sample in samples.train_samples
        ),
        right_records=(
            _training_sample_leakage_record(sample)
            for sample in (*samples.val_samples, *samples.test_samples)
        ),
        output_path=report_path,
        minimum_coverage={category: 1.0 for category in CATEGORIES},
    )
    val_vs_test = generate_report(
        left_records=(
            _training_sample_leakage_record(sample)
            for sample in samples.val_samples
        ),
        right_records=(
            _training_sample_leakage_record(sample)
            for sample in samples.test_samples
        ),
        output_path=(
            training_directory / "evaluation" / "leakage_val_vs_test.json"
        ),
        minimum_coverage={category: 1.0 for category in CATEGORIES},
    )
    if val_vs_test.passed:
        return

    counts = tuple(
        (
            category,
            train_vs_holdout.overlap_count_by_category[category]
            + val_vs_test.overlap_count_by_category[category],
        )
        for category in CATEGORIES
    )
    violations = list(train_vs_holdout.violations)
    violations.extend(
        f"val_test:{violation}" for violation in val_vs_test.violations
    )
    if val_vs_test.total_overlap_count:
        violations.append("cross_split_overlap:val:test")
    combined = LeakageReportV2(
        coverage=train_vs_holdout.coverage,
        overlap_counts=counts,
        overlaps=tuple(
            sorted(
                (*train_vs_holdout.overlaps, *val_vs_test.overlaps),
                key=lambda item: (
                    item.category,
                    item.left.key,
                    item.right.key,
                    item.fingerprint.normalized_digest,
                ),
            )[: train_vs_holdout.overlap_sample_limit]
        ),
        violations=tuple(dict.fromkeys(sorted(violations))),
        passed=False,
        overlap_sample_limit=train_vs_holdout.overlap_sample_limit,
    )
    with atomic_text_writer(report_path) as handle:
        if handle is None:
            raise RuntimeError("leakage report path is required")
        handle.write(
            json.dumps(combined.to_dict(), indent=2, sort_keys=True) + "\n"
        )


def _training_sample_leakage_record(
    sample: TrainingSample,
) -> dict[str, object]:
    if not sample.content_family_id:
        raise ValueError(
            f"sample lacks leakage lineage identity: {sample.sample_id}"
        )
    return {
        **sample.to_dict(),
        "dataset_id": sample.snapshot_id,
        "partition": sample.split,
        "lineage_key": sample.content_family_id,
    }


def violations_for(report: LeakageReportV2) -> tuple[str, ...]:
    """Return canonical gate violations for a validated v2 report."""

    reasons = list(report.violations)
    reasons.extend(
        f"overlap:{category}:{count}"
        for category, count in report.overlap_counts
        if count
    )
    return tuple(reasons)


def _validate_limits(
    *,
    text_similarity_threshold: float,
    image_phash_max_distance: int,
    image_dhash_max_distance: int,
    overlap_sample_limit: int,
    max_indexed_records_per_category: int,
    max_candidates_per_record: int,
) -> None:
    if not 0.0 <= text_similarity_threshold <= 1.0:
        raise ValueError("text similarity threshold is out of range")
    if image_phash_max_distance < 0 or image_dhash_max_distance < 0:
        raise ValueError("image hash distance cannot be negative")
    if overlap_sample_limit < 0:
        raise ValueError("overlap sample limit cannot be negative")
    if max_indexed_records_per_category <= 0:
        raise ValueError("fingerprint index capacity must be positive")
    if max_candidates_per_record <= 0:
        raise ValueError("candidate limit must be positive")


def _coverage_minimums(
    minimum_coverage: Mapping[str, float],
) -> dict[str, float]:
    minimums = {
        category: float(minimum_coverage.get(category, 1.0))
        for category in CATEGORIES
    }
    for category, minimum in minimums.items():
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(
                f"minimum leakage coverage is out of range: {category}"
            )
    return minimums


def _index_capacity_violations(
    *,
    left: SideIndex,
    right: SideIndex,
) -> list[str]:
    return [
        f"index_capacity:{category}"
        for category in CATEGORIES
        if category in left.overflowed_categories
        or category in right.overflowed_categories
    ]


def _coverage_side(
    *,
    index: SideIndex,
    category: str,
) -> LeakageCoverageSide:
    eligible = index.eligible[category]
    evidence = index.with_evidence[category]
    return LeakageCoverageSide(
        eligible=eligible,
        with_evidence=evidence,
        ratio=evidence / eligible if eligible else 1.0,
    )


def _build_coverage(
    *,
    left: SideIndex,
    right: SideIndex,
    minimums: Mapping[str, float],
    violations: list[str],
) -> tuple[tuple[str, LeakageCategoryCoverage], ...]:
    coverage: list[tuple[str, LeakageCategoryCoverage]] = []
    for category in CATEGORIES:
        category_coverage = LeakageCategoryCoverage(
            left=_coverage_side(index=left, category=category),
            right=_coverage_side(index=right, category=category),
            minimum=minimums[category],
        )
        coverage.append((category, category_coverage))
        if category_coverage.violated:
            violations.append(f"coverage:{category}")
    return tuple(coverage)


@dataclass(frozen=True, slots=True)
class LeakageStageSource:
    """Named lineage stage participating in contamination checks."""

    name: str
    records: Iterable[Mapping[str, object]]

    def __post_init__(self) -> None:
        allowed = {
            "raw_crawl",
            "preprocessed",
            "curated_snapshot",
            "augmentation",
            "training_samples",
            "synthetic_derivations",
            "previous_release_output",
        }
        if self.name not in allowed:
            raise ValueError(f"unsupported leakage stage: {self.name!r}")


def generate_benchmark_contamination_reports(
    *,
    benchmark_records: Iterable[Mapping[str, object]],
    stage_sources: tuple[LeakageStageSource, ...],
    output_directory: Path,
    minimum_coverage: Mapping[str, float],
) -> dict[str, LeakageReportV2]:
    """Compare one pinned benchmark against every lineage stage."""
    benchmark = tuple(benchmark_records)
    if not benchmark or not stage_sources:
        raise ValueError("contamination check requires benchmark and stages")
    output_directory.mkdir(parents=True, exist_ok=True)
    reports: dict[str, LeakageReportV2] = {}
    for source in stage_sources:
        reports[source.name] = generate_report(
            left_records=benchmark,
            right_records=source.records,
            output_path=output_directory / f"benchmark_vs_{source.name}.json",
            detailed_evidence_path=output_directory
            / f"benchmark_vs_{source.name}.overlaps.jsonl",
            minimum_coverage=minimum_coverage,
        )
    return reports
