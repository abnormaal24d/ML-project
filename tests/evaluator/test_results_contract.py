from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from evaluator.metric_registry import SUPPORTED_EVALUATION_METHODS
from evaluator.results import (
    BaselineReference,
    BenchmarkSuite,
    EvaluationResult,
    PairedModelComparison,
    validate_benchmark_comparisons,
)


def _manifest(path: Path, content: bytes = b"manifest") -> tuple[Path, str]:
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return path, digest


def test_baseline_reference_requires_identity(
    tmp_path: Path,
) -> None:
    path, digest = _manifest(tmp_path / "manifest.json")
    reference = BaselineReference(
        provider="p",
        model_id="m",
        evaluation_date=date(2026, 8, 17),
        inference_configuration={"top_k": 5},
        output_manifest_path=path,
        output_manifest_sha256=digest,
    )
    payload = reference.to_payload()
    assert payload["provider"] == "p"
    assert payload["evaluation_date"] == "2026-08-17"
    with pytest.raises(ValueError, match="provider and exact model_id"):
        BaselineReference(
            provider="",
            model_id="m",
            evaluation_date=date(2026, 8, 17),
            inference_configuration={},
            output_manifest_path=path,
            output_manifest_sha256=digest,
        )
    with pytest.raises(FileNotFoundError):
        BaselineReference(
            provider="p",
            model_id="m",
            evaluation_date=date(2026, 8, 17),
            inference_configuration={},
            output_manifest_path=Path("missing.json"),
            output_manifest_sha256=digest,
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        BaselineReference(
            provider="p",
            model_id="m",
            evaluation_date=date(2026, 8, 17),
            inference_configuration={},
            output_manifest_path=path,
            output_manifest_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="JSON-safe"):
        BaselineReference(
            provider="p",
            model_id="m",
            evaluation_date=date(2026, 8, 17),
            inference_configuration={object(): 1},
            output_manifest_path=path,
            output_manifest_sha256=digest,
        )


def test_benchmark_suite_validates_identity_and_manifest(
    tmp_path: Path,
) -> None:
    path, digest = _manifest(tmp_path / "suite.json")
    suite = BenchmarkSuite(
        suite_id="suite-v1",
        version="1.0",
        manifest_path=path,
        manifest_sha256=digest,
        required_capabilities=("image_quality", "recall"),
        minimum_samples_per_capability=50,
        seeds=(1, 2, 3),
    )
    assert suite.suite_id == "suite-v1"
    with pytest.raises(ValueError, match="identity is required"):
        BenchmarkSuite(
            suite_id="",
            version="",
            manifest_path=path,
            manifest_sha256=digest,
            required_capabilities=("x",),
        )
    with pytest.raises(
        ValueError, match="minimum sample count must be positive"
    ):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=path,
            manifest_sha256=digest,
            required_capabilities=("x",),
            minimum_samples_per_capability=0,
        )
    with pytest.raises(ValueError, match="requires capabilities"):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=path,
            manifest_sha256=digest,
            required_capabilities=(),
        )
    with pytest.raises(ValueError, match="must be unique"):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=path,
            manifest_sha256=digest,
            required_capabilities=("x", "x"),
        )
    with pytest.raises(ValueError, match="explicit and unique"):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=path,
            manifest_sha256=digest,
            required_capabilities=("x",),
            seeds=(),
        )
    with pytest.raises(FileNotFoundError):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=Path("missing.json"),
            manifest_sha256=digest,
            required_capabilities=("x",),
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        BenchmarkSuite(
            suite_id="s",
            version="1",
            manifest_path=path,
            manifest_sha256="0" * 64,
            required_capabilities=("x",),
        )


def test_paired_comparison_from_scores_counts_wins_and_ties() -> None:
    comparison = PairedModelComparison.from_scores(
        capability="image_quality",
        candidate_scores=[1.0, 0.5, 0.0],
        baseline_scores=[0.0, 0.5, 1.0],
        seeds=(1,),
        bootstrap_samples=100,
    )
    assert comparison.sample_count == 3
    assert comparison.candidate_wins == 1
    assert comparison.ties == 1
    assert comparison.baseline_wins == 1
    assert comparison.win_rate == 0.5
    assert comparison.regressed is False
    payload = comparison.to_payload()
    assert payload["candidate_wins"] == 1
    assert payload["regressed"] is False


def test_paired_comparison_detects_regression() -> None:
    comparison = PairedModelComparison.from_scores(
        capability="recall",
        candidate_scores=[0.0, 0.0, 0.0],
        baseline_scores=[1.0, 1.0, 1.0],
        seeds=(1,),
        bootstrap_samples=100,
    )
    assert comparison.candidate_wins == 0
    assert comparison.win_rate == 0.0
    assert comparison.regressed is True


def test_paired_comparison_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty and equal length"):
        PairedModelComparison.from_scores(
            capability="x",
            candidate_scores=[1.0],
            baseline_scores=[],
        )
    with pytest.raises(ValueError, match="must be finite"):
        PairedModelComparison.from_scores(
            capability="x",
            candidate_scores=[float("nan")],
            baseline_scores=[1.0],
        )


def test_paired_comparison_validation_checks_fields() -> None:
    with pytest.raises(ValueError, match="capability is required"):
        PairedModelComparison.from_scores(
            capability="  ",
            candidate_scores=[1.0],
            baseline_scores=[0.0],
            bootstrap_samples=100,
        )
    with pytest.raises(ValueError, match="counts do not sum"):
        PairedModelComparison(
            capability="x",
            sample_count=5,
            candidate_wins=1,
            ties=1,
            baseline_wins=1,
            win_rate=0.5,
            confidence_interval_low=0.0,
            confidence_interval_high=1.0,
            bootstrap_interval_low=0.0,
            bootstrap_interval_high=1.0,
            seeds=(1,),
            regressed=False,
        )
    with pytest.raises(ValueError, match="finite probabilities"):
        PairedModelComparison(
            capability="x",
            sample_count=2,
            candidate_wins=1,
            ties=0,
            baseline_wins=1,
            win_rate=1.5,
            confidence_interval_low=0.0,
            confidence_interval_high=1.0,
            bootstrap_interval_low=0.0,
            bootstrap_interval_high=1.0,
            seeds=(1,),
            regressed=False,
        )
    with pytest.raises(ValueError, match="non-negative integers"):
        PairedModelComparison(
            capability="x",
            sample_count=2,
            candidate_wins=-1,
            ties=1,
            baseline_wins=2,
            win_rate=0.25,
            confidence_interval_low=0.0,
            confidence_interval_high=1.0,
            bootstrap_interval_low=0.0,
            bootstrap_interval_high=1.0,
            seeds=(1,),
            regressed=False,
        )
    with pytest.raises(ValueError, match="must match outcome counts"):
        PairedModelComparison(
            capability="x",
            sample_count=2,
            candidate_wins=1,
            ties=0,
            baseline_wins=1,
            win_rate=0.25,
            confidence_interval_low=0.0,
            confidence_interval_high=1.0,
            bootstrap_interval_low=0.0,
            bootstrap_interval_high=1.0,
            seeds=(1,),
            regressed=False,
        )


def test_validate_benchmark_comparisons_fails_closed(
    tmp_path: Path,
) -> None:
    path, digest = _manifest(tmp_path / "suite.json")
    suite = BenchmarkSuite(
        suite_id="suite",
        version="1",
        manifest_path=path,
        manifest_sha256=digest,
        required_capabilities=("image_quality", "recall", "emotion"),
        minimum_samples_per_capability=4,
    )
    reasons = validate_benchmark_comparisons(suite=suite, comparisons={})
    assert reasons == (
        "benchmark_capability_missing:image_quality",
        "benchmark_capability_missing:recall",
        "benchmark_capability_missing:emotion",
    )
    healthy = PairedModelComparison.from_scores(
        capability="image_quality",
        candidate_scores=[1.0] * 10,
        baseline_scores=[0.0] * 10,
        bootstrap_samples=100,
    )
    small = PairedModelComparison.from_scores(
        capability="recall",
        candidate_scores=[1.0] * 3,
        baseline_scores=[0.0] * 3,
        bootstrap_samples=100,
    )
    regressed = PairedModelComparison.from_scores(
        capability="emotion",
        candidate_scores=[0.0] * 5,
        baseline_scores=[1.0] * 5,
        bootstrap_samples=100,
    )
    reasons = validate_benchmark_comparisons(
        suite=suite,
        comparisons={
            "image_quality": healthy,
            "recall": small,
            "emotion": regressed,
        },
    )
    assert reasons == (
        "benchmark_sample_count_low:recall",
        "benchmark_capability_regressed:emotion",
    )


def test_evaluation_result_payload_and_benchmark_reasons(
    tmp_path: Path,
) -> None:
    manifest, digest = _manifest(tmp_path / "suite.json")
    suite = BenchmarkSuite(
        suite_id="suite",
        version="1",
        manifest_path=manifest,
        manifest_sha256=digest,
        required_capabilities=("image_quality",),
        minimum_samples_per_capability=1,
    )
    comparison = PairedModelComparison.from_scores(
        capability="image_quality",
        candidate_scores=[0.0],
        baseline_scores=[1.0],
        bootstrap_samples=100,
    )
    result = EvaluationResult(
        validation_loss=0.5,
        test_loss=0.6,
        evaluation_mode="test",
        labeled_sample_count=10,
        dataset_split_counts={"test": 10},
        task_metrics={"image_text_pair": {"recall_at_1": 0.8}},
        test_task_metrics={"image_text_pair": {"recall_at_1": 0.8}},
        metrics={"custom": 1.0},
        valid=True,
        failure_reasons=(),
        max_batch_latency_ms=100.0,
        peak_memory_mb=2048.0,
        benchmark_suite=suite,
        paired_comparisons={"image_quality": comparison},
    )
    payload = result.to_payload()
    assert payload["valid"] is True
    assert payload["benchmark_suite_id"] == "suite"
    assert payload["benchmark_failure_reasons"] == [
        "benchmark_capability_regressed:image_quality"
    ]
    assert payload["max_batch_latency_ms"] == 100.0
    assert payload["metrics"] == {"custom": 1.0}
    assert result.benchmark_failure_reasons == (
        "benchmark_capability_regressed:image_quality",
    )


def test_evaluation_result_without_benchmark_has_no_reasons() -> None:
    result = EvaluationResult(validation_loss=0.5, test_loss=0.6, valid=True)
    assert result.benchmark_failure_reasons == ()
    payload = result.to_payload()
    assert payload["benchmark_suite_id"] is None
    assert payload["paired_comparisons"] == {}


def test_free_form_metrics_cannot_override_canonical_result_fields() -> None:
    result = EvaluationResult(
        validation_loss=0.5,
        test_loss=0.6,
        valid=True,
        metrics={"valid": 0.0, "val_loss": 999.0, "custom": 1.0},
    )

    payload = result.to_payload()

    assert payload["valid"] is True
    assert payload["val_loss"] == 0.5
    assert payload["metrics"] == {
        "valid": 0.0,
        "val_loss": 999.0,
        "custom": 1.0,
    }


def test_evaluation_result_with_reproducibility_report(tmp_path: Path) -> None:
    result = EvaluationResult(validation_loss=0.5, test_loss=0.6, valid=True)
    path = tmp_path / "reproducibility_report.json"
    updated = result.with_reproducibility_report(path)
    assert updated.reproducibility_report_path == path
    assert (
        updated.to_payload()["reproducibility_report_path"] == path.as_posix()
    )


def test_wilson_and_bootstrap_intervals_are_valid() -> None:
    comparison = PairedModelComparison.from_scores(
        capability="image_quality",
        candidate_scores=[1.0] * 5 + [0.0] * 5,
        baseline_scores=[0.0] * 5 + [1.0] * 5,
        bootstrap_samples=200,
    )
    assert 0.0 <= comparison.confidence_interval_low <= 1.0
    assert 0.0 <= comparison.confidence_interval_high <= 1.0
    assert (
        comparison.confidence_interval_low
        <= comparison.confidence_interval_high
    )
    assert (
        comparison.bootstrap_interval_low <= comparison.bootstrap_interval_high
    )


def test_supported_evaluation_methods_are_stable() -> None:
    assert SUPPORTED_EVALUATION_METHODS == frozenset(
        {
            "causal_language_modeling",
            "cer_wer",
            "cer_wer_layout",
            "exact_match_f1",
            "masked_language_modeling",
            "retrieval_accuracy",
            "retrieval_or_contrastive",
            "rouge_or_token_f1",
            "vqa_accuracy",
        }
    )
