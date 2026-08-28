from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings.datasets import DatasetValidatorSettings
from evaluator.results import EvaluationResult
from release.acceptance_evaluator import (
    build_decision,
    decide_release,
    evaluate_training_release,
)
from release.release_artifact_validation import (
    check_artifacts,
    check_cards,
    check_checkpoint_and_metrics_target,
    check_dataset_card,
    check_model_card,
    check_reproducibility,
    check_required_reports,
    check_validation,
)
from release.training_acceptance_validation import (
    check_meets_minimums_from_report,
    check_modality_coverage,
    check_signals,
)
from schemas.release import ReleaseReason, ReleaseStatus
from training.runtime.checkpoint.io import atomic_torch_save
from training.runtime.results import (
    TrainingArtifacts,
    TrainingMetrics,
    TrainingRunIdentity,
    TrainingRunResult,
)


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload={"marker": b"checkpoint"}, checkpoint_path=path)


def _metrics(**overrides: object) -> TrainingMetrics:
    values: dict[str, object] = {
        "train_loss": 0.5,
        "validation_loss": 0.6,
        "test_loss": 0.7,
        "average_loss": 0.6,
        "last_epoch_loss": 0.6,
        "epochs": 1,
        "batches": 10,
        "samples": 80,
        "effective_train_sample_count": 80,
        "effective_task_counts": {
            "image_text_pair": 40,
            "audio_text_pair": 40,
        },
        "effective_modality_counts": {"image": 50, "audio": 50},
        "training_signal_by_modality": {
            "image": {
                "trainable_parameter_count": 8,
                "gradient_observations": 16,
                "max_gradient_l2": 1.0,
                "gradient_detected": True,
                "parameter_delta_l2": 0.1,
                "updated": True,
            },
            "audio": {
                "trainable_parameter_count": 8,
                "gradient_observations": 16,
                "max_gradient_l2": 1.0,
                "gradient_detected": True,
                "parameter_delta_l2": 0.1,
                "updated": True,
            },
        },
    }
    values.update(overrides)
    return TrainingMetrics(**values)


def _run_result(
    *,
    checkpoint_path: Path,
    export_directory: Path | None = None,
) -> TrainingRunResult:
    return TrainingRunResult(
        metrics=_metrics(),
        artifacts=TrainingArtifacts(
            checkpoint_path=checkpoint_path,
            last_checkpoint_path=checkpoint_path,
            export_directory=export_directory,
        ),
        identity=TrainingRunIdentity(model_seed=1),
    )


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={},
        test_task_metrics={},
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=True,
        failure_reasons=(),
    )


def test_decide_release_completion_rejects_invalid_checkpoint() -> None:
    decision = decide_release(
        training_result=_run_result(checkpoint_path=Path("missing.pt")),
        evaluation_result=_evaluation(),
        checkpoint_path=Path("missing.pt"),
    )
    assert decision.status is ReleaseStatus.FAILED
    assert ReleaseReason.CHECKPOINT_MISSING in decision.reasons


def test_decide_release_completion_rejects_non_finite_losses(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    metrics = _metrics(train_loss=float("nan"))
    result = _run_result(checkpoint_path=checkpoint)
    result = TrainingRunResult(
        metrics=metrics,
        artifacts=result.artifacts,
        identity=result.identity,
    )
    decision = decide_release(
        training_result=result,
        evaluation_result=_evaluation(),
        checkpoint_path=checkpoint,
    )
    assert decision.status is ReleaseStatus.FAILED
    assert ReleaseReason.TRAIN_LOSS_INVALID in decision.reasons


def test_decide_release_completion_accepts_valid_run(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    decision = decide_release(
        training_result=_run_result(checkpoint_path=checkpoint),
        evaluation_result=_evaluation(),
        checkpoint_path=checkpoint,
    )
    assert decision.status is ReleaseStatus.PIPELINE_ACCEPTED
    assert decision.reasons == ()


def test_decide_release_completion_rejects_invalid_training_run(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    metrics = _metrics(
        epochs=0,
        batches=0,
        samples=0,
    )
    evaluation = _evaluation()
    evaluation = EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={},
        test_task_metrics={},
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=False,
        failure_reasons=("exploded",),
    )
    decision = decide_release(
        training_result=TrainingRunResult(
            metrics=metrics,
            artifacts=TrainingArtifacts(
                checkpoint_path=checkpoint,
                last_checkpoint_path=checkpoint,
                export_directory=None,
            ),
            identity=TrainingRunIdentity(model_seed=1),
        ),
        evaluation_result=evaluation,
        checkpoint_path=checkpoint,
    )
    assert decision.status is ReleaseStatus.FAILED
    assert ReleaseReason.NO_COMPLETED_EPOCHS in decision.reasons
    assert ReleaseReason.NO_COMPLETED_BATCHES in decision.reasons
    assert ReleaseReason.NO_TRAINING_SAMPLES in decision.reasons
    assert ReleaseReason.FINAL_EVALUATION_INVALID in decision.reasons


def test_decide_release_configured_requires_every_input(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    with pytest.raises(
        ValueError,
        match="configured release decision requires: run_mode",
    ):
        decide_release(
            training_result=_run_result(checkpoint_path=checkpoint),
            evaluation_result=_evaluation(),
            checkpoint_path=checkpoint,
            settings=DatasetValidatorSettings(),
        )


def test_build_decision_maps_every_release_stage() -> None:
    assert (
        build_decision(release_stage="pipeline_smoke", reasons=()).status
        is ReleaseStatus.PIPELINE_ACCEPTED
    )
    assert (
        build_decision(release_stage="learning_candidate", reasons=()).status
        is ReleaseStatus.DATASET_ACCEPTED
    )
    assert (
        build_decision(release_stage="candidate", reasons=()).status
        is ReleaseStatus.MODEL_CANDIDATE
    )
    assert (
        build_decision(release_stage="production_model", reasons=()).status
        is ReleaseStatus.MODEL_ACCEPTED
    )
    assert (
        build_decision(release_stage="unknown_stage", reasons=()).status
        is ReleaseStatus.FAILED
    )
    assert (
        build_decision(release_stage="candidate", reasons=("failure",)).status
        is ReleaseStatus.FAILED
    )
    decision = build_decision(
        release_stage="candidate",
        reasons=(),
        model_reasons=("model_bad",),
    )
    assert decision.status is ReleaseStatus.FAILED
    assert decision.reasons == ("model_bad",)
    decision = build_decision(
        release_stage="candidate",
        reasons=("a", "a", "b"),
    )
    assert decision.reasons == ("a", "b")


def test_release_requirements_stage_mismatch_is_fail_closed() -> None:
    from config.releases.release_requirements import ReleaseRequirements
    from release.acceptance_evaluator import (
        _release_requirements_stage_reasons,
    )

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="candidate",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=(),
        optional_tasks=(),
        blocked_capabilities=(),
    )
    assert (
        _release_requirements_stage_reasons(
            release_stage="candidate",
            release_requirements=requirements,
        )
        == ()
    )
    reasons = _release_requirements_stage_reasons(
        release_stage="production_model",
        release_requirements=requirements,
    )
    assert reasons == (
        f"{ReleaseReason.RELEASE_REQUIREMENTS_MISMATCH.value}:"
        "production_v1:release_stage:production_model:candidate",
    )
    assert (
        _release_requirements_stage_reasons(
            release_stage="candidate",
            release_requirements=None,
        )
        == ()
    )


def test_candidate_treats_release_task_failures_as_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import release.acceptance_evaluator as evaluator
    from config.releases.release_requirements import (
        MetricRequirement,
        ReleaseRequirements,
        TaskRequirement,
    )

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="candidate",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_text_pair",),
        optional_tasks=(),
        blocked_capabilities=(),
        task_requirements=(
            TaskRequirement(
                name="image_text_pair",
                min_samples=1,
                metrics=(MetricRequirement(name="recall_at_1", minimum=0.2),),
            ),
        ),
    )
    monkeypatch.setattr(evaluator, "read_dataset_counts", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "release_counts",
        lambda _counts: (1, 1, 0, 0, {"image_text_pair": 1}, {}),
    )
    monkeypatch.setattr(evaluator, "_completion_reasons", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_artifacts", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_counts", lambda **_kwargs: ())
    monkeypatch.setattr(
        evaluator,
        "check_release",
        lambda **_kwargs: SimpleNamespace(violations=()),
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_is_available", lambda _path: True
    )
    monkeypatch.setattr(evaluator, "check_losses", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_signals", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_model", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_supervision", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_production", lambda **_kwargs: ())

    decision = evaluator._decide_configured(
        settings=DatasetValidatorSettings(),
        training_result=_run_result(
            checkpoint_path=tmp_path / "checkpoint.pt"
        ),
        evaluation_result=_evaluation(),
        run_mode="full",
        release_stage="candidate",
        checkpoint_path=tmp_path / "checkpoint.pt",
        metrics_path=tmp_path / "training_metrics.json",
        dataset_root=tmp_path / "dataset",
        release_requirements=requirements,
        evidence=object(),
    )

    assert decision.status is ReleaseStatus.FAILED
    assert decision.reasons == (
        "evaluation_metric_missing:image_text_pair:recall_at_1",
    )
    assert decision.warnings == ()


def test_checkpoint_and_metrics_target_requires_checkpoint_and_valid_path(
    tmp_path: Path,
) -> None:
    missing_checkpoint = tmp_path / "missing.pt"
    future_metrics_path = tmp_path / "training_metrics.json"
    assert check_checkpoint_and_metrics_target(
        checkpoint_path=missing_checkpoint,
        metrics_path=future_metrics_path,
    ) == (ReleaseReason.CHECKPOINT_MISSING,)

    checkpoint = tmp_path / "present.pt"
    _write_checkpoint(checkpoint)
    assert check_checkpoint_and_metrics_target(
        checkpoint_path=checkpoint,
        metrics_path=Path(""),
    ) == (ReleaseReason.METRICS_PATH_INVALID,)

    assert not future_metrics_path.exists()
    assert (
        check_checkpoint_and_metrics_target(
            checkpoint_path=checkpoint,
            metrics_path=future_metrics_path,
        )
        == ()
    )


def test_required_reports_are_checked(tmp_path: Path) -> None:
    required = ("coverage/task_coverage_report.json", "other.json")
    assert check_required_reports(
        dataset_root=tmp_path,
        required=required,
    ) == (ReleaseReason.REQUIRED_REPORTS_MISSING,)
    (tmp_path / "coverage").mkdir()
    (tmp_path / "coverage" / "task_coverage_report.json").write_text("{}")
    (tmp_path / "other.json").write_text("{}")
    assert (
        check_required_reports(dataset_root=tmp_path, required=required) == ()
    )


def test_dataset_card_requires_presence(tmp_path: Path) -> None:
    missing = tmp_path / "dataset_card.json"
    assert check_dataset_card(path=missing) == (
        ReleaseReason.DATASET_CARD_MISSING,
    )
    missing.write_text("{}")
    assert check_dataset_card(path=missing) == ()


def test_model_card_requires_all_sections(tmp_path: Path) -> None:
    card = tmp_path / "model_card.md"
    assert check_model_card(path=card) == (ReleaseReason.MODEL_CARD_MISSING,)
    card.write_text(
        "# Model\n\nintended use\n\nlimitations\n", encoding="utf-8"
    )
    # No ## headings → missing_sections
    result = check_model_card(path=card)
    assert ReleaseReason.MODEL_CARD_INCOMPLETE.value in str(result)
    assert "missing_sections" in str(result)
    # Words in body do NOT count as headings
    card.write_text(
        "# Model\n\nintended use\n\nlimitations\n\nmetrics\n", encoding="utf-8"
    )
    # Still no ## headings → still missing_sections
    result2 = check_model_card(path=card)
    assert ReleaseReason.MODEL_CARD_INCOMPLETE.value in str(result2)
    assert "missing_sections" in str(result2)
    # Valid card with all 8 ## headings and content → pass
    card.write_text(
        """# Model Card

## release identity
Identity content.

## architecture boundary
Architecture content.

## intended use
Intended use content.

## out-of-scope and disabled capabilities
Out-of-scope content.

## training data and provenance
Training data content.

## evaluation and acceptance
Evaluation content.

## limitations and risks
Limitations content.

## release decision
Release decision content.
""",
        encoding="utf-8",
    )
    assert check_model_card(path=card) == ()


def test_reproducibility_requires_report(tmp_path: Path) -> None:
    report = tmp_path / "reproducibility_report.json"
    assert check_reproducibility(path=report) == (
        ReleaseReason.REPRODUCIBILITY_MISSING,
    )
    report.write_text("{}")
    assert check_reproducibility(path=report) == ()


def test_cards_honor_requirements(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    export = tmp_path / "export"
    report = tmp_path / "report"
    dataset_root.mkdir()
    export.mkdir()
    report.mkdir()
    reasons = check_cards(
        dataset_root=dataset_root,
        export_directory=export,
        report_directory=report,
        dataset_card_filename="dataset_card.json",
        require_dataset_card=True,
        require_model_card=True,
        require_reproducibility=True,
    )
    assert set(reasons) == {
        ReleaseReason.DATASET_CARD_MISSING,
        ReleaseReason.MODEL_CARD_MISSING,
        ReleaseReason.REPRODUCIBILITY_MISSING,
    }
    assert (
        check_cards(
            dataset_root=dataset_root,
            export_directory=export,
            report_directory=report,
            dataset_card_filename="dataset_card.json",
            require_dataset_card=False,
            require_model_card=False,
            require_reproducibility=False,
        )
        == ()
    )


def test_validation_report_reasons_are_fail_closed() -> None:
    reasons = check_validation(
        dataset_counts={
            "validation_valid": False,
            "validation_errors": ["corrupt"],
        },
        required=True,
    )
    assert ReleaseReason.DATASET_VALIDATION_INVALID in reasons
    assert any(
        reason.startswith(
            f"{ReleaseReason.DATASET_VALIDATION_ERROR.value}:corrupt"
        )
        for reason in reasons
    )
    assert (
        check_validation(
            dataset_counts={"validation_valid": True, "validation_errors": []},
            required=True,
        )
        == ()
    )
    assert (
        check_validation(
            dataset_counts={},
            required=False,
        )
        == ()
    )


def test_check_artifacts_reports_missing_resources(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    settings = DatasetValidatorSettings(
        require_coverage_report=True,
        require_evaluation_report=True,
        min_active_modalities=2,
        max_batch_latency_ms=250.0,
        max_peak_memory_mb=16384.0,
    )
    reasons = check_artifacts(
        settings=settings,
        checkpoint_path=tmp_path / "checkpoint.pt",
        dataset_root=dataset_root,
        metrics_path=tmp_path / "training_metrics.json",
        evaluation={"max_batch_latency_ms": 300.0, "peak_memory_mb": 20000.0},
        dataset_counts={"modalities": {"image": 0, "audio": 0}},
    )
    assert ReleaseReason.CHECKPOINT_MISSING in reasons
    assert ReleaseReason.COVERAGE_REPORT_MISSING in reasons
    assert ReleaseReason.EVALUATION_REPORT_MISSING in reasons
    assert ReleaseReason.ACTIVE_MODALITIES_LOW not in reasons
    assert ReleaseReason.BATCH_LATENCY_HIGH in reasons
    assert ReleaseReason.PEAK_MEMORY_HIGH in reasons


def test_check_artifacts_accepts_complete_inputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    dataset_root = tmp_path / "dataset"
    (dataset_root / "coverage").mkdir(parents=True)
    for report in (
        "coverage/task_coverage_report.json",
        "coverage/modality_coverage_report.json",
        "coverage/target_quality_report.json",
        "coverage/coverage_trend_report.json",
    ):
        (dataset_root / report).write_text("{}")
    settings = DatasetValidatorSettings(
        require_coverage_report=True,
        require_evaluation_report=True,
        min_active_modalities=1,
    )
    reasons = check_artifacts(
        settings=settings,
        checkpoint_path=checkpoint,
        dataset_root=dataset_root,
        metrics_path=tmp_path / "training_metrics.json",
        evaluation={"valid": True},
        dataset_counts={
            "validation_valid": True,
            "modalities": {"image": 10, "audio": 0},
        },
    )
    assert reasons == ()


def test_check_signals_rejects_missing_effective_counts() -> None:
    metrics = _metrics(
        effective_train_sample_count=0,
        effective_task_counts={},
        effective_modality_counts={},
        training_signal_by_modality={},
    )
    reasons = check_signals(
        settings=DatasetValidatorSettings(),
        metrics=metrics,
    )
    assert ReleaseReason.EFFECTIVE_SAMPLES_MISSING in reasons
    assert ReleaseReason.EFFECTIVE_TASK_COUNTS_MISSING in reasons
    assert ReleaseReason.EFFECTIVE_MODALITY_COUNTS_MISSING in reasons
    assert ReleaseReason.SIGNALS_MISSING in reasons


def test_check_signals_rejects_sample_mismatch() -> None:
    metrics = _metrics(
        samples=50,
        effective_train_sample_count=80,
    )
    reasons = check_signals(
        settings=DatasetValidatorSettings(),
        metrics=metrics,
    )
    assert any(
        reason.startswith(
            f"{ReleaseReason.EFFECTIVE_SAMPLE_MISMATCH.value}:80/50"
        )
        for reason in reasons
    )


def test_check_signals_rejects_weak_training_signal() -> None:
    metrics = _metrics(
        effective_modality_counts={"image": 50},
        training_signal_by_modality={
            "image": {
                "trainable_parameter_count": 0,
                "gradient_observations": 0,
                "max_gradient_l2": 0.0,
                "gradient_detected": False,
                "parameter_delta_l2": 0.0,
                "updated": False,
            },
        },
    )
    reasons = check_signals(
        settings=DatasetValidatorSettings(),
        metrics=metrics,
    )
    assert any(
        reason.startswith(f"{ReleaseReason.SIGNAL_PARAMETERS_MISSING.value}:")
        for reason in reasons
    )
    assert any(
        reason.startswith(f"{ReleaseReason.SIGNAL_GRADIENT_MISSING.value}:")
        for reason in reasons
    )
    assert any(
        reason.startswith(f"{ReleaseReason.SIGNAL_UPDATE_MISSING.value}:")
        for reason in reasons
    )


def test_check_signals_reports_task_minimum_violations() -> None:
    metrics = _metrics(
        effective_task_counts={"image_text_pair": 5},
    )
    settings = DatasetValidatorSettings(
        min_task_samples={"image_text_pair": 20},
    )
    reasons = check_signals(
        settings=settings,
        metrics=metrics,
    )
    assert any(
        reason.startswith(
            f"{ReleaseReason.EFFECTIVE_TASK_MINIMUM.value}:image_text_pair"
        )
        for reason in reasons
    )


def test_check_signals_reports_autonomous_readiness() -> None:
    metrics = _metrics(
        effective_task_counts={},
        effective_modality_counts={},
    )
    settings = DatasetValidatorSettings(
        require_autonomous_multimodal_readiness=True,
    )
    reasons = check_signals(
        settings=settings,
        metrics=metrics,
    )
    assert any(
        reason.startswith(
            f"{ReleaseReason.AUTONOMOUS_MODALITIES_MISSING.value}:"
        )
        for reason in reasons
    )
    assert any(
        reason.startswith(f"{ReleaseReason.AUTONOMOUS_TASKS_MISSING.value}:")
        for reason in reasons
    )


def test_check_signals_accepts_healthy_run() -> None:
    assert (
        check_signals(
            settings=DatasetValidatorSettings(),
            metrics=_metrics(),
        )
        == ()
    )


def test_modality_coverage_flags_low_counts() -> None:
    assert check_modality_coverage(
        modality_counts={"image": 10},
        targets={"image": 20, "audio": 5},
    ) == (
        f"{ReleaseReason.MODALITY_COVERAGE_LOW.value}:image:10/20",
        f"{ReleaseReason.MODALITY_COVERAGE_LOW.value}:audio:0/5",
    )
    assert (
        check_modality_coverage(
            modality_counts={"image": 20, "audio": 5},
            targets={"image": 20, "audio": 5},
        )
        == ()
    )


def test_coverage_report_meets_minimums_paths(tmp_path: Path) -> None:
    assert check_meets_minimums_from_report(None) == (
        ReleaseReason.COVERAGE_REPORT_MISSING,
    )
    report = tmp_path / "coverage_report.json"
    report.write_text(json.dumps({"meets_minimums": True}), encoding="utf-8")
    assert check_meets_minimums_from_report(report) == ()
    report.write_text(json.dumps({"meets_minimums": False}), encoding="utf-8")
    assert check_meets_minimums_from_report(report) == (
        ReleaseReason.RAW_COVERAGE_LOW,
    )
    report.write_text("not json", encoding="utf-8")
    reasons = check_meets_minimums_from_report(report)
    assert len(reasons) == 1
    assert reasons[0].startswith(
        f"{ReleaseReason.COVERAGE_REPORT_UNREADABLE.value}:"
    )


def test_evaluate_training_release_pipeline_smoke_writes_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    result = _run_result(checkpoint_path=checkpoint)
    acceptance = evaluate_training_release(
        training_result=result,
        evaluation_result=_evaluation(),
        input_dataset_root=tmp_path / "dataset",
        metrics_path=tmp_path / "training_metrics.json",
        release_stage="pipeline_smoke",
        run_mode="smoke",
        settings=DatasetValidatorSettings(),
        release_requirements=None,
    )
    assert acceptance.decision.status is ReleaseStatus.PIPELINE_ACCEPTED
    report_path = tmp_path / "evaluation" / "acceptance_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_pipeline_smoke_writes_report_below_export_when_available(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint.pt"
    export_directory = tmp_path / "candidate"
    _write_checkpoint(checkpoint)

    acceptance = evaluate_training_release(
        training_result=_run_result(
            checkpoint_path=checkpoint,
            export_directory=export_directory,
        ),
        evaluation_result=_evaluation(),
        input_dataset_root=tmp_path / "dataset",
        metrics_path=tmp_path / "training_metrics.json",
        release_stage="pipeline_smoke",
        run_mode="smoke",
        settings=DatasetValidatorSettings(),
        release_requirements=None,
    )

    assert acceptance.decision.status is ReleaseStatus.PIPELINE_ACCEPTED
    assert (
        export_directory / "evaluation" / "acceptance_report.json"
    ).is_file()
    assert not (
        checkpoint.parent / "evaluation" / "acceptance_report.json"
    ).exists()


def test_evaluate_training_release_rejects_unknown_run_mode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="requires run_mode 'smoke' or"):
        evaluate_training_release(
            training_result=_run_result(checkpoint_path=checkpoint),
            evaluation_result=_evaluation(),
            input_dataset_root=tmp_path / "dataset",
            metrics_path=tmp_path / "training_metrics.json",
            release_stage="candidate",
            run_mode="deploy",
            settings=DatasetValidatorSettings(),
            release_requirements=None,
        )


def test_evaluate_training_release_requires_export_directory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    result = _run_result(checkpoint_path=checkpoint)
    result = TrainingRunResult(
        metrics=result.metrics,
        artifacts=TrainingArtifacts(
            checkpoint_path=checkpoint,
            last_checkpoint_path=checkpoint,
            export_directory=None,
        ),
        identity=result.identity,
    )
    with pytest.raises(RuntimeError, match="requires an export directory"):
        evaluate_training_release(
            training_result=result,
            evaluation_result=_evaluation(),
            input_dataset_root=tmp_path / "dataset",
            metrics_path=tmp_path / "training_metrics.json",
            release_stage="candidate",
            run_mode="smoke",
            settings=DatasetValidatorSettings(),
            release_requirements=None,
        )


def test_release_uses_test_task_metrics_not_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release gates must use test_task_metrics, not validation task_metrics.

    Regression: validation metric = 0.90 (pass), test metric = 0.10 (fail),
    release minimum = 0.50 → release must FAIL.
    """
    import release.acceptance_evaluator as evaluator
    from config.releases.release_requirements import (
        MetricRequirement,
        ReleaseRequirements,
        TaskRequirement,
    )

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_text_pair",),
        optional_tasks=(),
        blocked_capabilities=(),
        task_requirements=(
            TaskRequirement(
                name="image_text_pair",
                min_samples=1,
                metrics=(MetricRequirement(name="accuracy", minimum=0.5),),
            ),
        ),
    )

    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    evaluation = EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={
            "image_text_pair": {"accuracy": 0.90},  # validation: passes
        },
        test_task_metrics={
            "image_text_pair": {"accuracy": 0.10},  # test: fails
        },
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=True,
        failure_reasons=(),
    )

    monkeypatch.setattr(evaluator, "read_dataset_counts", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "release_counts",
        lambda _counts: (1, 1, 0, 0, {"image_text_pair": 1}, {}),
    )
    monkeypatch.setattr(evaluator, "_completion_reasons", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_artifacts", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_counts", lambda **_kwargs: ())
    monkeypatch.setattr(
        evaluator,
        "check_release",
        lambda **_kwargs: SimpleNamespace(violations=()),
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_is_available", lambda _path: True
    )
    monkeypatch.setattr(evaluator, "check_losses", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_signals", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_model", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_supervision", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_production", lambda **_kwargs: ())

    decision = evaluator._decide_configured(
        settings=DatasetValidatorSettings(),
        training_result=_run_result(checkpoint_path=checkpoint),
        evaluation_result=evaluation,
        run_mode="full",
        release_stage="production_model",
        checkpoint_path=checkpoint,
        metrics_path=tmp_path / "training_metrics.json",
        dataset_root=tmp_path / "dataset",
        release_requirements=requirements,
        evidence=object(),
    )

    # Must fail because test_task_metrics (0.10) < minimum (0.5)
    assert decision.status is ReleaseStatus.FAILED
    assert any(
        "evaluation_metric_below_threshold:image_text_pair:accuracy:0.1:0.5"
        == r
        for r in decision.reasons
    ), f"Expected test metric failure, got: {decision.reasons}"


def test_release_requires_benchmark_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production release with require_benchmark=True must have benchmark suite.

    Regression: require_benchmark=True but no benchmark_suite → release FAILS.
    """
    import release.acceptance_evaluator as evaluator
    from config.releases.release_requirements import (
        MetricRequirement,
        ReleaseRequirements,
        TaskRequirement,
    )

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_text_pair",),
        optional_tasks=(),
        blocked_capabilities=(),
        task_requirements=(
            TaskRequirement(
                name="image_text_pair",
                min_samples=1,
                metrics=(MetricRequirement(name="accuracy", minimum=0.5),),
            ),
        ),
        require_benchmark=True,
        require_baseline=False,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    evaluation = EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={"image_text_pair": {"accuracy": 0.90}},
        test_task_metrics={"image_text_pair": {"accuracy": 0.90}},
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=True,
        failure_reasons=(),
        benchmark_suite=None,  # Missing!
    )

    monkeypatch.setattr(evaluator, "read_dataset_counts", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "release_counts",
        lambda _counts: (1, 1, 0, 0, {"image_text_pair": 1}, {}),
    )
    monkeypatch.setattr(evaluator, "_completion_reasons", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_artifacts", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_counts", lambda **_kwargs: ())
    monkeypatch.setattr(
        evaluator,
        "check_release",
        lambda **_kwargs: SimpleNamespace(violations=()),
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_is_available", lambda _path: True
    )
    monkeypatch.setattr(evaluator, "check_losses", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_signals", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_model", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_supervision", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_production", lambda **_kwargs: ())

    decision = evaluator._decide_configured(
        settings=DatasetValidatorSettings(),
        training_result=_run_result(checkpoint_path=checkpoint),
        evaluation_result=evaluation,
        run_mode="full",
        release_stage="production_model",
        checkpoint_path=checkpoint,
        metrics_path=tmp_path / "training_metrics.json",
        dataset_root=tmp_path / "dataset",
        release_requirements=requirements,
        evidence=object(),
    )

    assert decision.status is ReleaseStatus.FAILED
    assert ReleaseReason.BENCHMARK_SUITE_MISSING.value in decision.reasons


def test_release_requires_baseline_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production release with require_baseline=True must have baseline reference.

    Regression: require_baseline=True but no baseline_reference → release FAILS.
    """
    import release.acceptance_evaluator as evaluator
    from config.releases.release_requirements import (
        MetricRequirement,
        ReleaseRequirements,
        TaskRequirement,
    )

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_text_pair",),
        optional_tasks=(),
        blocked_capabilities=(),
        task_requirements=(
            TaskRequirement(
                name="image_text_pair",
                min_samples=1,
                metrics=(MetricRequirement(name="accuracy", minimum=0.5),),
            ),
        ),
        require_benchmark=False,
        require_baseline=True,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    evaluation = EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={"image_text_pair": {"accuracy": 0.90}},
        test_task_metrics={"image_text_pair": {"accuracy": 0.90}},
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=True,
        failure_reasons=(),
        baseline_reference=None,  # Missing!
    )

    monkeypatch.setattr(evaluator, "read_dataset_counts", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "release_counts",
        lambda _counts: (1, 1, 0, 0, {"image_text_pair": 1}, {}),
    )
    monkeypatch.setattr(evaluator, "_completion_reasons", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_artifacts", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_counts", lambda **_kwargs: ())
    monkeypatch.setattr(
        evaluator,
        "check_release",
        lambda **_kwargs: SimpleNamespace(violations=()),
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_is_available", lambda _path: True
    )
    monkeypatch.setattr(evaluator, "check_losses", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_signals", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_model", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_supervision", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_production", lambda **_kwargs: ())

    decision = evaluator._decide_configured(
        settings=DatasetValidatorSettings(),
        training_result=_run_result(checkpoint_path=checkpoint),
        evaluation_result=evaluation,
        run_mode="full",
        release_stage="production_model",
        checkpoint_path=checkpoint,
        metrics_path=tmp_path / "training_metrics.json",
        dataset_root=tmp_path / "dataset",
        release_requirements=requirements,
        evidence=object(),
    )

    assert decision.status is ReleaseStatus.FAILED
    assert ReleaseReason.BASELINE_REFERENCE_MISSING.value in decision.reasons


def test_release_benchmark_validates_required_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark suite with missing required capability fails release.

    Regression: benchmark_suite has required capability but no comparison → release FAILS.
    """
    import release.acceptance_evaluator as evaluator
    from config.releases.release_requirements import (
        MetricRequirement,
        ReleaseRequirements,
        TaskRequirement,
    )
    from evaluator.results import BenchmarkSuite

    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=(),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_text_pair",),
        optional_tasks=(),
        blocked_capabilities=(),
        task_requirements=(
            TaskRequirement(
                name="image_text_pair",
                min_samples=1,
                metrics=(MetricRequirement(name="accuracy", minimum=0.5),),
            ),
        ),
        require_benchmark=True,
        require_baseline=False,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)

    # Create a benchmark suite requiring "vqa_accuracy" capability
    (tmp_path / "benchmark_manifest.json").write_text("{}")
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_sha256 = hashlib.sha256(b"{}").hexdigest()
    benchmark_suite = BenchmarkSuite(
        suite_id="test_suite",
        version="1.0",
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        required_capabilities=("vqa_accuracy",),
        minimum_samples_per_capability=100,
        seeds=(17, 29, 43),
    )

    evaluation = EvaluationResult(
        validation_loss=0.6,
        test_loss=0.7,
        evaluation_mode="test",
        labeled_sample_count=20,
        dataset_split_counts={},
        task_metrics={"image_text_pair": {"accuracy": 0.90}},
        test_task_metrics={"image_text_pair": {"accuracy": 0.90}},
        metrics={},
        leakage_report_path=None,
        reproducibility_report_path=None,
        valid=True,
        failure_reasons=(),
        benchmark_suite=benchmark_suite,
        paired_comparisons={},  # Missing required capability comparison!
    )

    monkeypatch.setattr(evaluator, "read_dataset_counts", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "release_counts",
        lambda _counts: (1, 1, 0, 0, {"image_text_pair": 1}, {}),
    )
    monkeypatch.setattr(evaluator, "_completion_reasons", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_artifacts", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_counts", lambda **_kwargs: ())
    monkeypatch.setattr(
        evaluator,
        "check_release",
        lambda **_kwargs: SimpleNamespace(violations=()),
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_is_available", lambda _path: True
    )
    monkeypatch.setattr(evaluator, "check_losses", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_signals", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_model", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_supervision", lambda **_kwargs: ())
    monkeypatch.setattr(evaluator, "check_production", lambda **_kwargs: ())

    decision = evaluator._decide_configured(
        settings=DatasetValidatorSettings(),
        training_result=_run_result(checkpoint_path=checkpoint),
        evaluation_result=evaluation,
        run_mode="full",
        release_stage="production_model",
        checkpoint_path=checkpoint,
        metrics_path=tmp_path / "training_metrics.json",
        dataset_root=tmp_path / "dataset",
        release_requirements=requirements,
        evidence=object(),
    )

    assert decision.status is ReleaseStatus.FAILED
    assert any(
        "benchmark_capability_missing:vqa_accuracy" == r
        for r in decision.reasons
    )
