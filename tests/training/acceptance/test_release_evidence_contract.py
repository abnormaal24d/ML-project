from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from release.acceptance_evaluator import (
    build_decision,
)
from release.acceptance_result import AcceptanceReport
from release.release_decision import ReleaseDecision
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    parse_release_mode,
)
from schemas.release import ReleaseStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _passing_acceptance_report(tmp_path: Path) -> AcceptanceReport:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return AcceptanceReport.build(
        release_stage="pipeline_smoke",
        decision=ReleaseDecision(
            status=ReleaseStatus.PIPELINE_ACCEPTED,
            reasons=(),
        ),
        expected_status=ReleaseStatus.PIPELINE_ACCEPTED,
        evidence_paths={"checkpoint": checkpoint},
    )


def test_acceptance_report_write_load_round_trip_supports_resume(
    tmp_path: Path,
) -> None:
    report = _passing_acceptance_report(tmp_path)
    path = tmp_path / "acceptance_report.json"
    report.write(path)

    assert AcceptanceReport.load(path) == report
    assert AcceptanceReport.load(path) == report


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.__setitem__("schema_version", "0.9"),
            "unsupported acceptance report schema",
        ),
        (
            lambda payload: payload.__setitem__(
                "decision",
                {"status": "UNKNOWN", "reasons": []},
            ),
            "acceptance report status is invalid",
        ),
        (
            lambda payload: payload.__setitem__("violations", ["unexpected"]),
            "persisted acceptance report is not passed",
        ),
    ),
)
def test_acceptance_report_load_rejects_invalid_current_contract(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    report = _passing_acceptance_report(tmp_path)
    path = tmp_path / "acceptance_report.json"
    payload = report.to_payload()
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        AcceptanceReport.load(path)


def test_candidate_evidence_has_no_external_suite_report(
    tmp_path: Path,
) -> None:
    artifacts = EvaluationArtifacts.from_release_outputs(
        candidate_directory=tmp_path / "candidate",
        dataset_directory=tmp_path / "dataset",
        checkpoint=tmp_path / "candidate" / "checkpoint.pt",
        metrics=tmp_path / "candidate" / "training_metrics.json",
    )

    assert set(artifacts.required_paths("candidate")) == {
        "checkpoint",
        "leakage",
        "dataset_card",
        "model_card",
        "reproducibility",
        "run_receipts",
        "training_metrics",
    }
    assert set(artifacts.required_paths("production_model")) == {
        "checkpoint",
        "leakage",
        "dataset_card",
        "model_card",
        "reproducibility",
        "run_receipts",
        "training_metrics",
    }


def test_candidate_stage_builds_model_candidate_status() -> None:
    decision = build_decision(release_stage="candidate", reasons=())
    assert decision.status is ReleaseStatus.MODEL_CANDIDATE


def test_production_stage_builds_model_accepted_status() -> None:
    decision = build_decision(
        release_stage="production_model",
        reasons=(),
        model_reasons=(),
    )
    assert decision.status is ReleaseStatus.MODEL_ACCEPTED


def test_candidate_stage_keeps_best_effort_warnings() -> None:
    decision = build_decision(
        release_stage="candidate",
        reasons=(),
        warnings=("evaluation_metric_low:audio_text_pair:recall_at_1",),
    )
    assert decision.status is ReleaseStatus.MODEL_CANDIDATE
    assert decision.warnings == (
        "evaluation_metric_low:audio_text_pair:recall_at_1",
    )
    payload = decision.to_payload()
    assert payload["warnings"] == decision.warnings


def test_warnings_are_not_persisted_when_empty() -> None:
    decision = build_decision(release_stage="candidate", reasons=())
    assert decision.warnings == ()
    assert set(decision.to_payload()) == {"status", "reasons"}


def test_removed_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported release mode"):
        parse_release_mode("model_candidate")


def test_namespaced_task_metric_is_extracted_from_correct_task() -> None:
    from release.release_evidence_validation import (
        _evaluation_metric_value,
    )

    task_metrics = {
        "image_text_pair": {
            "recall_at_1": 0.9,
            "embedding_similarity_mean": 0.85,
        },
        "audio_text_pair": {
            "recall_at_1": 0.3,
            "embedding_similarity_mean": 0.25,
        },
    }

    assert (
        _evaluation_metric_value(
            evaluation={},
            task_metrics=task_metrics,
            metric_name="image_text_pair.recall_at_1",
        )
        == 0.9
    )
    assert (
        _evaluation_metric_value(
            evaluation={},
            task_metrics=task_metrics,
            metric_name="audio_text_pair.embedding_similarity_mean",
        )
        == 0.25
    )


def test_namespaced_metric_returns_none_for_missing_task() -> None:
    from release.release_evidence_validation import (
        _evaluation_metric_value,
    )

    task_metrics = {
        "image_text_pair": {
            "recall_at_1": 0.9,
        },
    }

    assert (
        _evaluation_metric_value(
            evaluation={},
            task_metrics=task_metrics,
            metric_name="audio_text_pair.recall_at_1",
        )
        is None
    )


def test_evaluation_metric_value_ignores_other_tasks() -> None:
    from release.release_evidence_validation import (
        _evaluation_metric_value,
    )

    task_metrics = {
        "image_text_pair": {
            "recall_at_1": 0.9,
        },
        "audio_text_pair": {
            "recall_at_1": 0.3,
        },
    }

    result = _evaluation_metric_value(
        evaluation={},
        task_metrics=task_metrics,
        metric_name="image_text_pair.recall_at_1",
    )
    assert result == 0.9


@pytest.mark.usefixtures("production_whisper_env")
def test_candidate_requires_every_release_task_metric() -> None:
    from config.load import load_settings
    from config.releases.release_requirements import (
        check_release_task_requirements,
        release_requirements_from_settings,
    )

    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )
    contract = release_requirements_from_settings(settings)
    assert contract is not None
    task_metrics = {
        task.name: {
            metric.name: (
                float(metric.minimum)
                if metric.minimum is not None
                else float(metric.maximum)
            )
            for metric in task.metrics
        }
        for task in contract.task_requirements
    }
    task_counts = {
        task.name: task.min_samples for task in settings.release.tasks
    }

    assert (
        check_release_task_requirements(
            task_requirements=contract.task_requirements,
            task_counts=task_counts,
            task_metrics=task_metrics,
        )
        == ()
    )

    del task_metrics["audio_text_pair"]["recall_at_1"]
    reasons = check_release_task_requirements(
        task_requirements=contract.task_requirements,
        task_counts=task_counts,
        task_metrics=task_metrics,
    )

    assert reasons == (
        "evaluation_metric_missing:audio_text_pair:recall_at_1",
    )


@pytest.mark.usefixtures("production_whisper_env")
def test_strong_tasks_do_not_mask_one_task_below_threshold() -> None:
    from config.load import load_settings
    from config.releases.release_requirements import (
        check_release_task_requirements,
        release_requirements_from_settings,
    )

    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )
    contract = release_requirements_from_settings(settings)
    assert contract is not None
    task_metrics = {
        task.name: {
            metric.name: (1.0 if metric.minimum is not None else 0.0)
            for metric in task.metrics
        }
        for task in contract.task_requirements
    }
    task_metrics["image_text_pair"]["recall_at_1"] = 0.19
    task_counts = {
        task.name: task.min_samples for task in settings.release.tasks
    }

    reasons = check_release_task_requirements(
        task_requirements=contract.task_requirements,
        task_counts=task_counts,
        task_metrics=task_metrics,
    )

    assert reasons == (
        "evaluation_metric_below_threshold:"
        "image_text_pair:recall_at_1:0.19:0.2",
    )
