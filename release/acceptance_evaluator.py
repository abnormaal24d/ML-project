"""Release decision assembly for completed training runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

from evaluator.loss_thresholds import check_losses
from evaluator.task_thresholds import (
    check_counts,
    check_supervision,
    coerce_task_metric_payload,
)
from mmcrawler_datasets.snapshots.training_dataset_manifest import (
    read_dataset_counts,
    release_counts,
)
from release.acceptance_result import (
    AcceptanceReport,
    TrainingAcceptanceResult,
)
from release.release_artifact_validation import check_artifacts
from release.release_decision import ReleaseDecision, ReleaseStatus
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
    parse_release_mode,
)
from release.release_evidence_validation import (
    check_model,
    check_production,
    check_release,
)
from release.training_acceptance_validation import check_signals
from schemas.release import ReleaseReason
from training.runtime.checkpoint.io import checkpoint_is_available

if TYPE_CHECKING:
    from config.releases.release_requirements import ReleaseRequirements
    from config.settings.datasets import DatasetValidatorSettings
    from evaluator.results import EvaluationResult
    from training.runtime.results import TrainingRunResult


def decide_release(
    *,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    checkpoint_path: Path,
    settings: DatasetValidatorSettings | None = None,
    run_mode: str | None = None,
    release_stage: str = "pipeline_smoke",
    metrics_path: Path | None = None,
    dataset_root: Path | None = None,
    release_requirements: ReleaseRequirements | None = None,
    evidence: ReleaseEvidenceBundle | None = None,
) -> ReleaseDecision:
    """Decide basic completion or the configured release stage."""

    if settings is None:
        reasons = _completion_reasons(
            training_result=training_result,
            evaluation_result=evaluation_result,
            checkpoint_path=checkpoint_path,
        )
        return ReleaseDecision(
            status=ReleaseStatus.FAILED
            if reasons
            else ReleaseStatus.PIPELINE_ACCEPTED,
            reasons=reasons,
        )

    required = {
        "run_mode": run_mode,
        "metrics_path": metrics_path,
        "dataset_root": dataset_root,
        "evidence": evidence,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "configured release decision requires: " + ", ".join(missing)
        )
    return _decide_configured(
        settings=settings,
        training_result=training_result,
        evaluation_result=evaluation_result,
        run_mode=cast(str, run_mode),
        release_stage=release_stage,
        checkpoint_path=checkpoint_path,
        metrics_path=cast(Path, metrics_path),
        dataset_root=cast(Path, dataset_root),
        release_requirements=release_requirements,
        evidence=cast("ReleaseEvidenceBundle", evidence),
    )


def build_decision(
    *,
    release_stage: str,
    reasons: tuple[str, ...] | list[str],
    model_reasons: tuple[str, ...] | list[str] = (),
    warnings: tuple[str, ...] | list[str] = (),
) -> ReleaseDecision:
    """Build a unique fail-closed decision for a release stage."""

    base = list(reasons)
    quality = list(model_reasons)
    if base:
        status = ReleaseStatus.FAILED
    elif release_stage == "pipeline_smoke":
        status = ReleaseStatus.PIPELINE_ACCEPTED
    elif release_stage == "learning_candidate":
        status = ReleaseStatus.DATASET_ACCEPTED
    elif quality:
        status = ReleaseStatus.FAILED
    elif release_stage == "candidate":
        status = ReleaseStatus.MODEL_CANDIDATE
    elif release_stage == "production_model":
        status = ReleaseStatus.MODEL_ACCEPTED
    else:
        status = ReleaseStatus.FAILED
    return ReleaseDecision(
        status=status,
        reasons=tuple(dict.fromkeys((*base, *quality))),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _decide_configured(
    *,
    settings: DatasetValidatorSettings,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    run_mode: str,
    release_stage: str,
    checkpoint_path: Path,
    metrics_path: Path,
    dataset_root: Path,
    release_requirements: ReleaseRequirements | None,
    evidence: ReleaseEvidenceBundle,
) -> ReleaseDecision:
    metrics = training_result.metrics
    evaluation = evaluation_result.to_payload()
    test_task_metrics = coerce_task_metric_payload(
        evaluation_result.test_task_metrics
    )
    dataset_counts = read_dataset_counts(dataset_root=dataset_root)
    (
        total_count,
        train_count,
        val_count,
        test_count,
        task_counts,
        task_counts_by_split,
    ) = release_counts(dataset_counts)
    reasons = [
        *_release_requirements_stage_reasons(
            release_stage=release_stage,
            release_requirements=release_requirements,
        ),
        *_completion_reasons(
            training_result=training_result,
            evaluation_result=evaluation_result,
            checkpoint_path=checkpoint_path,
        ),
        *check_artifacts(
            settings=settings,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            metrics_path=metrics_path,
            evaluation=evaluation,
            dataset_counts=dataset_counts,
        ),
        *check_counts(
            settings=settings,
            total=total_count,
            train=train_count,
            val=val_count,
            test=test_count,
            task_counts=task_counts,
        ),
    ]
    if release_requirements is not None:
        from config.releases.release_requirements import (
            ReproducibilityRequirements,
        )

        reproducibility_requirements: ReproducibilityRequirements | None = (
            release_requirements.reproducibility
        )
    else:
        reproducibility_requirements = None

    reasons.extend(
        check_release(
            evidence=evidence,
            reproducibility_requirements=reproducibility_requirements,
        ).violations
    )
    for failed, reason in (
        (
            not checkpoint_is_available(checkpoint_path),
            ReleaseReason.CHECKPOINT_MISSING,
        ),
        (metrics_path.name == "", ReleaseReason.METRICS_PATH_INVALID),
        (metrics.samples <= 0, ReleaseReason.NO_TRAINING_SAMPLES),
    ):
        if failed:
            reasons.append(reason)
    reasons.extend(
        check_losses(
            settings=settings,
            metrics=metrics,
            evaluation=evaluation_result,
        )
    )
    reasons.extend(check_signals(settings=settings, metrics=metrics))
    reasons.extend(evaluation_result.benchmark_failure_reasons)

    model_reasons = list(
        check_model(
            settings=settings,
            metrics=metrics,
            evaluation=evaluation,
            task_metrics=test_task_metrics,
            total_sample_count=total_count,
            train_sample_count=train_count,
            val_sample_count=val_count,
            test_sample_count=test_count,
            task_counts=task_counts,
            task_counts_by_split=task_counts_by_split,
        )
    )
    task_requirement_warnings: list[str] = []
    if release_requirements is not None and (
        release_requirements.task_requirements
        or release_requirements.runtime_limits is not None
    ):
        from config.releases.release_requirements import (
            check_release_runtime_limits,
            check_release_task_requirements,
        )

        task_requirement_reasons = list(
            check_release_task_requirements(
                task_requirements=release_requirements.task_requirements,
                task_counts=task_counts,
                task_metrics=test_task_metrics,
            )
        )
        observed_latency = evaluation.get("max_batch_latency_ms")
        max_batch_latency_ms = (
            float(observed_latency)
            if isinstance(observed_latency, (int, float))
            and not isinstance(observed_latency, bool)
            else None
        )
        observed_memory = evaluation.get("peak_memory_mb")
        peak_memory_mb = (
            float(observed_memory)
            if isinstance(observed_memory, (int, float))
            and not isinstance(observed_memory, bool)
            else None
        )
        task_requirement_reasons.extend(
            check_release_runtime_limits(
                runtime_limits=release_requirements.runtime_limits,
                max_batch_latency_ms=max_batch_latency_ms,
                peak_memory_mb=peak_memory_mb,
            )
        )
        task_requirement_reasons.extend(
            check_release_benchmark_requirements(
                release_requirements=release_requirements,
                evaluation_result=evaluation_result,
            )
        )
        if release_stage in {"candidate", "production_model"}:
            reasons.extend(task_requirement_reasons)
        else:
            task_requirement_warnings.extend(task_requirement_reasons)
    reasons.extend(
        check_supervision(
            settings=settings,
            evaluation=evaluation,
            task_metrics=test_task_metrics,
            effective_task_counts=metrics.effective_task_counts,
        )
    )
    if settings.fail_on_smoke_trainer and run_mode == "smoke":
        reasons.append(ReleaseReason.SMOKE_CHECKPOINT)
    reasons.extend(
        check_production(
            release_stage=release_stage,
            require_model_accepted=settings.require_model_accepted_in_production,
            model_reasons=model_reasons,
        )
    )
    return build_decision(
        release_stage=release_stage,
        reasons=reasons,
        model_reasons=model_reasons,
        warnings=task_requirement_warnings,
    )


def _release_requirements_stage_reasons(
    *,
    release_stage: str,
    release_requirements: ReleaseRequirements | None,
) -> tuple[str, ...]:
    if release_requirements is None:
        return ()
    runtime_stage = str(release_stage or "").strip().lower()
    required_stage = (
        str(release_requirements.release_stage or "").strip().lower()
    )
    if runtime_stage == required_stage:
        return ()
    return (
        f"{ReleaseReason.RELEASE_REQUIREMENTS_MISMATCH.value}:"
        f"{release_requirements.release_id}:release_stage:"
        f"{runtime_stage}:{required_stage}",
    )


def check_release_benchmark_requirements(
    *,
    release_requirements: ReleaseRequirements,
    evaluation_result: EvaluationResult,
) -> tuple[str, ...]:
    """Fail closed on mandatory production benchmark requirements.

    If *require_benchmark* is set, a benchmark suite with comparisons is required.
    If *require_baseline* is set, a baseline reference is required.
    Benchmark comparisons are validated against the suite's mandatory capabilities.
    """
    reasons: list[str] = []

    if release_requirements.require_benchmark:
        if evaluation_result.benchmark_suite is None:
            reasons.append(ReleaseReason.BENCHMARK_SUITE_MISSING.value)
        else:
            reasons.extend(evaluation_result.benchmark_failure_reasons)

    if release_requirements.require_baseline:
        if evaluation_result.baseline_reference is None:
            reasons.append(ReleaseReason.BASELINE_REFERENCE_MISSING.value)

    return tuple(reasons)


def _completion_reasons(
    *,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    metrics = training_result.metrics
    reasons: list[str] = []
    if not checkpoint_is_available(checkpoint_path):
        reasons.append(ReleaseReason.CHECKPOINT_MISSING)
    for value, reason in (
        (metrics.train_loss, ReleaseReason.TRAIN_LOSS_INVALID),
        (evaluation_result.validation_loss, ReleaseReason.VAL_LOSS_INVALID),
        (evaluation_result.test_loss, ReleaseReason.TEST_LOSS_INVALID),
    ):
        if value is None or not math.isfinite(float(value)):
            reasons.append(reason)
    if metrics.epochs <= 0:
        reasons.append(ReleaseReason.NO_COMPLETED_EPOCHS)
    if metrics.batches <= 0:
        reasons.append(ReleaseReason.NO_COMPLETED_BATCHES)
    if metrics.samples <= 0:
        reasons.append(ReleaseReason.NO_TRAINING_SAMPLES)
    if not evaluation_result.valid:
        reasons.append(ReleaseReason.FINAL_EVALUATION_INVALID)
    return tuple(reasons)


def evaluate_training_release(
    *,
    training_result: TrainingRunResult,
    evaluation_result: EvaluationResult,
    input_dataset_root: Path,
    metrics_path: Path,
    release_stage: str,
    run_mode: str,
    settings: DatasetValidatorSettings,
    release_requirements: ReleaseRequirements | None,
) -> TrainingAcceptanceResult:
    """Evaluate one completed run and persist only evaluation evidence."""

    checkpoint_path = training_result.artifacts.checkpoint_path
    export_directory = training_result.artifacts.export_directory
    report_directory = training_result.artifacts.evaluation_directory
    evidence: ReleaseEvidenceBundle | None = None
    evidence_path: Path | None = None

    if release_stage == "pipeline_smoke":
        decision = decide_release(
            training_result=training_result,
            evaluation_result=evaluation_result,
            checkpoint_path=checkpoint_path,
        )
    else:
        if run_mode not in {"smoke", "full"}:
            raise RuntimeError(
                "Training acceptance requires run_mode 'smoke' or 'full'."
            )
        if export_directory is None:
            raise RuntimeError(
                "Training acceptance requires an export directory."
            )
        artifacts = EvaluationArtifacts.from_release_outputs(
            candidate_directory=export_directory,
            dataset_directory=input_dataset_root,
            checkpoint=checkpoint_path,
            metrics=metrics_path,
        )
        release_requirements_id = (
            release_requirements.release_id
            if release_requirements is not None
            else None
        )
        release_requirements_sha256 = (
            release_requirements.reproducibility.policy_sha256
            if release_requirements is not None
            and release_requirements.reproducibility is not None
            else None
        )
        evidence = ReleaseEvidenceBundle.build(
            mode=parse_release_mode(release_stage),
            artifacts=artifacts,
            release_requirements_id=release_requirements_id,
            release_requirements_sha256=release_requirements_sha256,
        )
        evidence_path = report_directory / "release_evidence_bundle.json"
        evidence.write(evidence_path)
        decision = decide_release(
            settings=settings,
            training_result=training_result,
            evaluation_result=evaluation_result,
            run_mode=run_mode,
            release_stage=release_stage,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            dataset_root=input_dataset_root,
            release_requirements=release_requirements,
            evidence=evidence,
        )
    expected_status = {
        "pipeline_smoke": ReleaseStatus.PIPELINE_ACCEPTED,
        "learning_candidate": ReleaseStatus.DATASET_ACCEPTED,
        "candidate": ReleaseStatus.MODEL_CANDIDATE,
        "production_model": ReleaseStatus.MODEL_ACCEPTED,
    }.get(release_stage, ReleaseStatus.FAILED)
    evidence_paths = {"checkpoint": checkpoint_path}
    if evidence is not None:
        evidence_paths.update(evidence.evidence_paths)
    if evidence_path is not None:
        evidence_paths["release_evidence_bundle"] = evidence_path
    report = AcceptanceReport.build(
        release_stage=release_stage,
        decision=decision,
        expected_status=expected_status,
        evidence_paths=evidence_paths,
    )
    report_path = report_directory / "acceptance_report.json"
    report.write(report_path)

    if decision.status is ReleaseStatus.FAILED:
        raise RuntimeError(
            "Training acceptance rejected: " + ", ".join(decision.reasons)
        )
    if (
        release_stage == "production_model"
        and decision.status is not ReleaseStatus.MODEL_ACCEPTED
    ):
        raise RuntimeError(
            "Production model release requires MODEL_ACCEPTED status."
        )

    return TrainingAcceptanceResult(
        decision=decision,
        acceptance_report=report,
        acceptance_report_path=report_path,
        evidence_bundle=evidence,
        evidence_bundle_path=evidence_path,
    )


__all__ = [
    "build_decision",
    "decide_release",
    "evaluate_training_release",
]
