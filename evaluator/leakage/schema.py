"""Persisted leakage-v2 schema and structural validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from evaluator.leakage.contracts import (
    ALGORITHMS,
    CATEGORIES,
    PERCEPTUAL_CATEGORIES,
    SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True)
class LeakageIdentity:
    """Stable identity of one record participating in leakage evidence."""

    dataset_id: str
    sample_id: str
    partition: str
    lineage_key: str

    def __post_init__(self) -> None:
        for name in ("dataset_id", "sample_id", "partition", "lineage_key"):
            if not getattr(self, name).strip():
                raise ValueError(f"leakage identity {name} must be non-empty")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.dataset_id,
            self.sample_id,
            self.partition,
            self.lineage_key,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "partition": self.partition,
            "lineage_key": self.lineage_key,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageIdentity:
        require_fields(
            payload,
            {"dataset_id", "sample_id", "partition", "lineage_key"},
            context="identity",
        )
        return cls(
            dataset_id=required_text(payload, "dataset_id"),
            sample_id=required_text(payload, "sample_id"),
            partition=required_text(payload, "partition"),
            lineage_key=required_text(payload, "lineage_key"),
        )


@dataclass(frozen=True, slots=True)
class LeakageFingerprintReference:
    """Category-specific, normalized evidence for one overlap."""

    category: str
    algorithm: str
    normalized_digest: str
    metric_name: str | None = None
    metric_value: float | int | None = None

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unsupported leakage category: {self.category}")
        if self.algorithm != ALGORITHMS[self.category]:
            raise ValueError(f"invalid leakage algorithm for {self.category}")
        if not self.normalized_digest.strip():
            raise ValueError("leakage fingerprint digest must be non-empty")
        if self.category != "audio_chromaprint":
            require_sha256_digest(self.normalized_digest)
        if self.category == "near_duplicate_text":
            if self.metric_name != "similarity" or not isinstance(
                self.metric_value,
                float,
            ):
                raise ValueError("near-text evidence requires similarity")
            if not 0.0 <= self.metric_value <= 1.0:
                raise ValueError("near-text similarity is out of range")
        elif self.category in PERCEPTUAL_CATEGORIES:
            if self.metric_name != "hamming_distance" or not isinstance(
                self.metric_value,
                int,
            ):
                raise ValueError("image evidence requires hamming distance")
            if self.metric_value < 0:
                raise ValueError("hamming distance cannot be negative")
        elif self.metric_name is not None or self.metric_value is not None:
            raise ValueError("exact leakage evidence cannot contain a metric")

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "algorithm": self.algorithm,
            "normalized_digest": self.normalized_digest,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageFingerprintReference:
        require_fields(
            payload,
            {
                "category",
                "algorithm",
                "normalized_digest",
                "metric_name",
                "metric_value",
            },
            context="fingerprint reference",
        )
        metric_value = payload.get("metric_value")
        if isinstance(metric_value, bool) or not isinstance(
            metric_value,
            (int, float, type(None)),
        ):
            raise ValueError("invalid leakage metric value")
        return cls(
            category=required_text(payload, "category"),
            algorithm=required_text(payload, "algorithm"),
            normalized_digest=required_text(payload, "normalized_digest"),
            metric_name=optional_text(payload.get("metric_name")),
            metric_value=metric_value,
        )


@dataclass(frozen=True, slots=True)
class LeakageOverlap:
    """One structurally validated cross-dataset overlap."""

    category: str
    left: LeakageIdentity
    right: LeakageIdentity
    fingerprint: LeakageFingerprintReference

    def __post_init__(self) -> None:
        if self.category != self.fingerprint.category:
            raise ValueError("overlap category and evidence category disagree")

    @property
    def identity(
        self,
    ) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        return (
            self.category,
            self.left.dataset_id,
            self.left.sample_id,
            self.left.partition,
            self.left.lineage_key,
            self.right.dataset_id,
            self.right.sample_id,
            self.right.partition,
            self.right.lineage_key,
            self.fingerprint.normalized_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageOverlap:
        require_fields(
            payload,
            {"category", "left", "right", "fingerprint"},
            context="overlap",
        )
        left = required_mapping(payload, "left")
        right = required_mapping(payload, "right")
        fingerprint = required_mapping(payload, "fingerprint")
        return cls(
            category=required_text(payload, "category"),
            left=LeakageIdentity.from_mapping(left),
            right=LeakageIdentity.from_mapping(right),
            fingerprint=LeakageFingerprintReference.from_mapping(fingerprint),
        )


@dataclass(frozen=True, slots=True)
class LeakageCoverageSide:
    eligible: int
    with_evidence: int
    ratio: float

    def __post_init__(self) -> None:
        if self.eligible < 0 or self.with_evidence < 0:
            raise ValueError("leakage coverage counts cannot be negative")
        if self.with_evidence > self.eligible:
            raise ValueError("leakage evidence exceeds eligible records")
        expected = self.with_evidence / self.eligible if self.eligible else 1.0
        if abs(expected - self.ratio) > 1e-12:
            raise ValueError("leakage coverage ratio is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "with_evidence": self.with_evidence,
            "ratio": self.ratio,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageCoverageSide:
        require_fields(
            payload,
            {"eligible", "with_evidence", "ratio"},
            context="coverage side",
        )
        return cls(
            eligible=required_int(payload, "eligible"),
            with_evidence=required_int(payload, "with_evidence"),
            ratio=required_float(payload, "ratio"),
        )


@dataclass(frozen=True, slots=True)
class LeakageCategoryCoverage:
    left: LeakageCoverageSide
    right: LeakageCoverageSide
    minimum: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum <= 1.0:
            raise ValueError("minimum leakage coverage is out of range")

    @property
    def violated(self) -> bool:
        return (
            self.left.ratio < self.minimum or self.right.ratio < self.minimum
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "minimum": self.minimum,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageCategoryCoverage:
        require_fields(
            payload,
            {"left", "right", "minimum"},
            context="category coverage",
        )
        return cls(
            left=LeakageCoverageSide.from_mapping(
                required_mapping(payload, "left")
            ),
            right=LeakageCoverageSide.from_mapping(
                required_mapping(payload, "right")
            ),
            minimum=required_float(payload, "minimum"),
        )


@dataclass(frozen=True, slots=True)
class LeakageReportV2:
    """The only accepted persisted leakage report."""

    coverage: tuple[tuple[str, LeakageCategoryCoverage], ...]
    overlap_counts: tuple[tuple[str, int], ...]
    overlaps: tuple[LeakageOverlap, ...]
    violations: tuple[str, ...]
    passed: bool
    overlap_sample_limit: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("only leakage report v2 is accepted")
        coverage = dict(self.coverage)
        counts = dict(self.overlap_counts)
        if tuple(coverage) != CATEGORIES:
            raise ValueError("leakage coverage categories are incomplete")
        if tuple(counts) != CATEGORIES:
            raise ValueError("leakage overlap counts are incomplete")
        if any(count < 0 for count in counts.values()):
            raise ValueError("leakage overlap counts cannot be negative")
        if self.overlap_sample_limit < 0:
            raise ValueError("overlap sample limit cannot be negative")
        if len(self.overlaps) > self.overlap_sample_limit:
            raise ValueError("leakage overlap sample exceeds configured limit")
        if len({item.identity for item in self.overlaps}) != len(
            self.overlaps
        ):
            raise ValueError("duplicate overlap identity in leakage sample")
        sampled_counts = {category: 0 for category in CATEGORIES}
        for overlap in self.overlaps:
            sampled_counts[overlap.category] += 1
        if any(
            sampled_counts[category] > counts[category]
            for category in CATEGORIES
        ):
            raise ValueError("leakage sample exceeds aggregate overlap count")
        if len(self.violations) != len(set(self.violations)):
            raise ValueError("leakage violations must be unique")
        expected_coverage_violations = {
            f"coverage:{category}"
            for category, value in self.coverage
            if value.violated
        }
        actual_coverage_violations = {
            violation
            for violation in self.violations
            if violation.startswith("coverage:")
        }
        if expected_coverage_violations != actual_coverage_violations:
            raise ValueError("leakage coverage violations are inconsistent")
        expected_passed = not self.violations and not any(counts.values())
        if self.passed is not expected_passed:
            raise ValueError("leakage passed flag is inconsistent")

    @property
    def coverage_by_category(
        self,
    ) -> dict[str, LeakageCategoryCoverage]:
        return dict(self.coverage)

    @property
    def overlap_count_by_category(self) -> dict[str, int]:
        return dict(self.overlap_counts)

    @property
    def total_overlap_count(self) -> int:
        return sum(count for _, count in self.overlap_counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coverage": {
                category: value.to_dict() for category, value in self.coverage
            },
            "overlap_counts": dict(self.overlap_counts),
            "overlaps": [item.to_dict() for item in self.overlaps],
            "violations": list(self.violations),
            "passed": self.passed,
            "overlap_sample_limit": self.overlap_sample_limit,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> LeakageReportV2:
        require_fields(
            payload,
            {
                "schema_version",
                "coverage",
                "overlap_counts",
                "overlaps",
                "violations",
                "passed",
                "overlap_sample_limit",
            },
            context="report",
        )
        raw_coverage = required_mapping(payload, "coverage")
        raw_counts = required_mapping(payload, "overlap_counts")
        if set(raw_coverage) != set(CATEGORIES):
            raise ValueError("leakage coverage categories are incomplete")
        if set(raw_counts) != set(CATEGORIES):
            raise ValueError("leakage overlap categories are incomplete")
        raw_overlaps = required_sequence(payload, "overlaps")
        raw_violations = required_sequence(payload, "violations")
        coverage = tuple(
            (
                category,
                LeakageCategoryCoverage.from_mapping(
                    required_mapping(raw_coverage, category)
                ),
            )
            for category in CATEGORIES
        )
        overlap_counts = tuple(
            (category, required_int(raw_counts, category))
            for category in CATEGORIES
        )
        overlaps = tuple(
            LeakageOverlap.from_mapping(item)
            for item in raw_overlaps
            if isinstance(item, Mapping)
        )
        if len(overlaps) != len(raw_overlaps):
            raise ValueError("invalid overlap row in leakage report")
        if any(not isinstance(value, str) for value in raw_violations):
            raise ValueError("invalid leakage violation")
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise ValueError("leakage passed must be boolean")
        return cls(
            schema_version=required_text(payload, "schema_version"),
            coverage=coverage,
            overlap_counts=overlap_counts,
            overlaps=overlaps,
            violations=tuple(str(value) for value in raw_violations),
            passed=passed,
            overlap_sample_limit=required_int(
                payload,
                "overlap_sample_limit",
            ),
        )


def require_sha256_digest(value: str) -> None:
    if len(value) != 64:
        raise ValueError("leakage fingerprint must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("leakage fingerprint must be SHA-256") from exc


def required_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"leakage {key} must be an object")
    return value


def require_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"leakage {context} fields are invalid")


def required_sequence(
    payload: Mapping[str, object],
    key: str,
) -> tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"leakage {key} must be an array")
    return tuple(value)


def required_text(payload: Mapping[str, object], key: str) -> str:
    value = optional_text(payload.get(key))
    if value is None:
        raise ValueError(f"leakage {key} must be non-empty text")
    return value


def optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"leakage {key} must be an integer")
    return value


def required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"leakage {key} must be numeric")
    return float(value)
