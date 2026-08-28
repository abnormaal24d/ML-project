"""Typed results produced by model and dataset evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BaselineReference:
    """Immutable identity of the exact baseline used for comparison."""

    provider: str
    model_id: str
    evaluation_date: date
    inference_configuration: Mapping[str, object]
    output_manifest_path: Path
    output_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError(
                "baseline provider and exact model_id are required"
            )
        if not self.output_manifest_path.is_file():
            raise FileNotFoundError(self.output_manifest_path)
        actual = hashlib.sha256(
            self.output_manifest_path.read_bytes()
        ).hexdigest()
        if actual != self.output_manifest_sha256:
            raise ValueError("baseline output manifest hash mismatch")
        try:
            json.dumps(self.inference_configuration, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "baseline inference configuration must be JSON-safe"
            ) from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "evaluation_date": self.evaluation_date.isoformat(),
            "inference_configuration": dict(self.inference_configuration),
            "output_manifest_path": self.output_manifest_path.as_posix(),
            "output_manifest_sha256": self.output_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Pinned benchmark definition with mandatory capability coverage."""

    suite_id: str
    version: str
    manifest_path: Path
    manifest_sha256: str
    required_capabilities: tuple[str, ...]
    minimum_samples_per_capability: int = 100
    seeds: tuple[int, ...] = (17, 29, 43)
    blind_human_evaluation: bool = False

    def __post_init__(self) -> None:
        if not self.suite_id.strip() or not self.version.strip():
            raise ValueError("benchmark suite identity is required")
        if self.minimum_samples_per_capability < 1:
            raise ValueError("benchmark minimum sample count must be positive")
        if not self.required_capabilities:
            raise ValueError("benchmark suite requires capabilities")
        if len(set(self.required_capabilities)) != len(
            self.required_capabilities
        ):
            raise ValueError("benchmark capabilities must be unique")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be explicit and unique")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        actual = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if actual != self.manifest_sha256:
            raise ValueError("benchmark manifest hash mismatch")


@dataclass(frozen=True, slots=True)
class PairedModelComparison:
    """Paired candidate-versus-baseline outcome for one capability."""

    capability: str
    sample_count: int
    candidate_wins: int
    ties: int
    baseline_wins: int
    win_rate: float
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_interval_low: float
    bootstrap_interval_high: float
    seeds: tuple[int, ...]
    regressed: bool

    def __post_init__(self) -> None:
        if not self.capability.strip():
            raise ValueError("paired comparison capability is required")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("paired comparison requires samples")
        counts = (self.candidate_wins, self.ties, self.baseline_wins)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise ValueError(
                "paired comparison counts must be non-negative integers"
            )
        if (
            self.candidate_wins + self.ties + self.baseline_wins
            != self.sample_count
        ):
            raise ValueError(
                "paired comparison counts do not sum to sample_count"
            )
        for value in (
            self.win_rate,
            self.confidence_interval_low,
            self.confidence_interval_high,
            self.bootstrap_interval_low,
            self.bootstrap_interval_high,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "paired comparison rates must be finite probabilities"
                )
        if self.confidence_interval_low > self.confidence_interval_high:
            raise ValueError(
                "paired comparison confidence interval is inverted"
            )
        if self.bootstrap_interval_low > self.bootstrap_interval_high:
            raise ValueError(
                "paired comparison bootstrap interval is inverted"
            )
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in self.seeds
            )
        ):
            raise ValueError(
                "paired comparison seeds must be non-empty unique integers"
            )
        expected_win_rate = (
            self.candidate_wins + 0.5 * self.ties
        ) / self.sample_count
        if not math.isclose(
            self.win_rate,
            expected_win_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "paired comparison win_rate must match outcome counts"
            )

    @classmethod
    def from_scores(
        cls,
        *,
        capability: str,
        candidate_scores: Sequence[float],
        baseline_scores: Sequence[float],
        seeds: tuple[int, ...] = (17, 29, 43),
        tie_tolerance: float = 1e-9,
        bootstrap_samples: int = 2000,
        regression_margin: float = 0.0,
    ) -> PairedModelComparison:
        if (
            len(candidate_scores) != len(baseline_scores)
            or not candidate_scores
        ):
            raise ValueError(
                "paired score vectors must be non-empty and equal length"
            )
        outcomes: list[float] = []
        candidate_wins = ties = baseline_wins = 0
        for candidate, baseline in zip(
            candidate_scores, baseline_scores, strict=True
        ):
            if not math.isfinite(float(candidate)) or not math.isfinite(
                float(baseline)
            ):
                raise ValueError("paired scores must be finite")
            delta = float(candidate) - float(baseline)
            if abs(delta) <= tie_tolerance:
                ties += 1
                outcomes.append(0.5)
            elif delta > 0:
                candidate_wins += 1
                outcomes.append(1.0)
            else:
                baseline_wins += 1
                outcomes.append(0.0)
        win_rate = sum(outcomes) / len(outcomes)
        ci_low, ci_high = _wilson_interval(
            successes=candidate_wins + 0.5 * ties,
            trials=len(outcomes),
        )
        bootstrap_low, bootstrap_high = _bootstrap_interval(
            outcomes,
            seeds=seeds,
            samples=bootstrap_samples,
        )
        return cls(
            capability=capability,
            sample_count=len(outcomes),
            candidate_wins=candidate_wins,
            ties=ties,
            baseline_wins=baseline_wins,
            win_rate=win_rate,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            bootstrap_interval_low=bootstrap_low,
            bootstrap_interval_high=bootstrap_high,
            seeds=seeds,
            regressed=bootstrap_high < 0.5 - regression_margin,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "sample_count": self.sample_count,
            "candidate_wins": self.candidate_wins,
            "ties": self.ties,
            "baseline_wins": self.baseline_wins,
            "win_rate": self.win_rate,
            "confidence_interval": [
                self.confidence_interval_low,
                self.confidence_interval_high,
            ],
            "bootstrap_interval": [
                self.bootstrap_interval_low,
                self.bootstrap_interval_high,
            ],
            "seeds": list(self.seeds),
            "regressed": self.regressed,
        }


def validate_benchmark_comparisons(
    *,
    suite: BenchmarkSuite,
    comparisons: Mapping[str, PairedModelComparison],
) -> tuple[str, ...]:
    """Fail closed on missing, undersized, or regressed mandatory capabilities."""

    reasons: list[str] = []
    for capability in suite.required_capabilities:
        comparison = comparisons.get(capability)
        if comparison is None:
            reasons.append(f"benchmark_capability_missing:{capability}")
            continue
        if comparison.sample_count < suite.minimum_samples_per_capability:
            reasons.append(f"benchmark_sample_count_low:{capability}")
        if comparison.regressed:
            reasons.append(f"benchmark_capability_regressed:{capability}")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Immutable outcome of evaluating one trained checkpoint."""

    validation_loss: float | None
    test_loss: float | None
    evaluation_mode: str = "not_evaluated"
    labeled_sample_count: int = 0
    dataset_split_counts: dict[str, int] = field(default_factory=dict)
    task_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    test_task_metrics: dict[
        str,
        dict[str, float],
    ] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    leakage_report_path: Path | None = None
    reproducibility_report_path: Path | None = None
    valid: bool = False
    failure_reasons: tuple[str, ...] = ()
    max_batch_latency_ms: float | None = None
    peak_memory_mb: float | None = None
    baseline_reference: BaselineReference | None = None
    benchmark_suite: BenchmarkSuite | None = None
    paired_comparisons: dict[str, PairedModelComparison] = field(
        default_factory=dict
    )

    def with_reproducibility_report(self, path: Path) -> EvaluationResult:
        return replace(self, reproducibility_report_path=path)

    @property
    def benchmark_failure_reasons(self) -> tuple[str, ...]:
        if self.benchmark_suite is None:
            return ()
        return validate_benchmark_comparisons(
            suite=self.benchmark_suite,
            comparisons=self.paired_comparisons,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "failure_reasons": list(self.failure_reasons),
            "benchmark_failure_reasons": list(self.benchmark_failure_reasons),
            "evaluation_mode": self.evaluation_mode,
            "labeled_sample_count": self.labeled_sample_count,
            "val_loss": self.validation_loss,
            "test_loss": self.test_loss,
            "dataset_split_counts": dict(self.dataset_split_counts),
            "task_metrics": {
                task: dict(values)
                for task, values in self.task_metrics.items()
            },
            "test_task_metrics": {
                task: dict(values)
                for task, values in self.test_task_metrics.items()
            },
            # Keep free-form metrics isolated from release-critical fields.
            # Flattening this mapping allowed callers to overwrite ``valid``
            # and the canonical loss values in the persisted evidence.
            "metrics": dict(self.metrics),
            "leakage_report_path": (
                self.leakage_report_path.as_posix()
                if self.leakage_report_path is not None
                else None
            ),
            "reproducibility_report_path": (
                self.reproducibility_report_path.as_posix()
                if self.reproducibility_report_path is not None
                else None
            ),
            "max_batch_latency_ms": self.max_batch_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "baseline_reference": (
                self.baseline_reference.to_payload()
                if self.baseline_reference is not None
                else None
            ),
            "benchmark_suite_id": (
                self.benchmark_suite.suite_id
                if self.benchmark_suite is not None
                else None
            ),
            "paired_comparisons": {
                capability: comparison.to_payload()
                for capability, comparison in sorted(
                    self.paired_comparisons.items()
                )
            },
        }


def _wilson_interval(
    *, successes: float, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires trials")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    centre = probability + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _bootstrap_interval(
    outcomes: Sequence[float], *, seeds: tuple[int, ...], samples: int
) -> tuple[float, float]:
    if samples < 100:
        raise ValueError("bootstrap comparison requires at least 100 samples")
    estimates: list[float] = []
    for seed in seeds:
        generator = random.Random(seed)
        for _ in range(samples):
            estimates.append(
                sum(generator.choice(outcomes) for _ in outcomes)
                / len(outcomes)
            )
    estimates.sort()
    low_index = max(0, int(0.025 * (len(estimates) - 1)))
    high_index = min(len(estimates) - 1, int(0.975 * (len(estimates) - 1)))
    return estimates[low_index], estimates[high_index]


__all__ = [
    "BaselineReference",
    "BenchmarkSuite",
    "EvaluationResult",
    "PairedModelComparison",
    "validate_benchmark_comparisons",
]
