"""Validate model artifacts, release evidence, and production readiness."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from config.environment.default_values import (
    DEFAULT_TEST_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
)
from evaluator.leakage.report import load_report, violations_for
from evaluator.leakage.schema import LeakageReportV2
from evaluator.loss_thresholds import ratio_reasons
from evaluator.reproducibility import (
    TrainingRunReceipt,
    evaluate_reproducibility,
    load_reproducibility_report,
    load_run_receipts_collection,
)
from evaluator.task_thresholds import check_tasks
from release.release_artifact_validation import check_model_card
from schemas.multimodal_tasks import canonical_task_name
from schemas.release import ReleaseReason, detail

ReleaseMode = Literal[
    "pipeline_smoke",
    "learning_candidate",
    "candidate",
    "production_model",
]


class EvidenceReference(Protocol):
    """Structural release-evidence reference consumed by validation."""

    @property
    def name(self) -> str:
        """Return the stable evidence name."""

        ...

    @property
    def path(self) -> str:
        """Return the persisted evidence path."""

        ...

    @property
    def sha256(self) -> str:
        """Return the evidence digest."""

        ...


class ReleaseEvidence(Protocol):
    """Structural release-evidence bundle consumed by validation."""

    @property
    def release_mode(self) -> ReleaseMode:
        """Return the requested release mode."""

        ...

    @property
    def references(self) -> tuple[EvidenceReference, ...]:
        """Return the immutable release-evidence references."""

        ...

    @property
    def leakage_report(self) -> LeakageReportV2 | None:
        """Return optional leakage evidence."""

        ...


if TYPE_CHECKING:
    from config.releases.release_requirements import (
        ReproducibilityRequirements,
    )
    from config.settings.datasets import DatasetValidatorSettings
    from training.runtime.results import TrainingMetrics


def check_model(
    *,
    settings: DatasetValidatorSettings,
    metrics: TrainingMetrics,
    evaluation: dict[str, object],
    task_metrics: dict[str, dict[str, float]],
    total_sample_count: int,
    train_sample_count: int,
    val_sample_count: int,
    test_sample_count: int,
    task_counts: dict[str, int],
    task_counts_by_split: dict[str, dict[str, int]],
) -> tuple[str, ...]:
    """Return every model-quality reason used by release promotion."""

    test_loss_value = evaluation.get("test_loss")
    reasons = list(
        _check_thresholds(
            settings=settings,
            total_sample_count=total_sample_count,
            train_sample_count=train_sample_count,
            val_sample_count=val_sample_count,
            test_sample_count=test_sample_count,
            task_counts=task_counts,
            task_counts_by_split=task_counts_by_split,
            training_batches=metrics.batches,
            train_loss=metrics.train_loss,
            test_loss=(
                float(test_loss_value)
                if isinstance(test_loss_value, (int, float))
                and not isinstance(test_loss_value, bool)
                else None
            ),
        )
    )
    reasons.extend(
        check_tasks(
            settings=settings,
            task_counts=task_counts,
            task_metrics=task_metrics,
        )
    )
    reasons.extend(
        check_evaluation_metrics(
            settings=settings,
            evaluation=evaluation,
            task_metrics=task_metrics,
        )
    )
    return tuple(reasons)


def check_evaluation_metrics(
    *,
    settings: DatasetValidatorSettings,
    evaluation: dict[str, object],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    """Validate configured metrics from final evaluation outputs."""

    if not settings.require_evaluation_metrics:
        return ()

    reasons: list[str] = []
    for metric_name, minimum in settings.min_evaluation_metrics.items():
        value = _evaluation_metric_value(
            evaluation=evaluation,
            task_metrics=task_metrics,
            metric_name=metric_name,
        )
        if value is None:
            reasons.append(
                detail(ReleaseReason.EVALUATION_METRIC_MISSING, metric_name)
            )
        elif value < minimum:
            reasons.append(
                detail(ReleaseReason.EVALUATION_METRIC_LOW, metric_name)
            )
    return tuple(reasons)


def _evaluation_metric_value(
    *,
    evaluation: dict[str, object],
    task_metrics: dict[str, dict[str, float]],
    metric_name: str,
) -> float | None:
    if "." in metric_name:
        task_name, task_metric_name = metric_name.split(
            ".",
            1,
        )

        value = task_metrics.get(
            task_name,
            {},
        ).get(task_metric_name)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)

        return None

    root_value = evaluation.get(metric_name)
    if (
        isinstance(root_value, (int, float))
        and not isinstance(root_value, bool)
        and math.isfinite(float(root_value))
    ):
        return float(root_value)
    for metrics in task_metrics.values():
        value = metrics.get(metric_name)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _check_thresholds(
    *,
    settings: DatasetValidatorSettings,
    total_sample_count: int,
    train_sample_count: int,
    val_sample_count: int,
    test_sample_count: int,
    task_counts: dict[str, int],
    task_counts_by_split: dict[str, dict[str, int]],
    training_batches: int,
    train_loss: float | None,
    test_loss: float | None,
) -> tuple[str, ...]:
    """Return model-level production acceptance reasons."""

    reasons: list[str] = []

    checks = (
        (
            ReleaseReason.MODEL_MIN_TOTAL_SAMPLES,
            total_sample_count,
            settings.model_min_total_samples,
        ),
        (
            ReleaseReason.MODEL_MIN_TRAIN_SAMPLES,
            train_sample_count,
            settings.model_min_train_samples,
        ),
        (
            ReleaseReason.MODEL_MIN_VAL_SAMPLES,
            val_sample_count,
            settings.model_min_val_samples,
        ),
        (
            ReleaseReason.MODEL_MIN_TEST_SAMPLES,
            test_sample_count,
            settings.model_min_test_samples,
        ),
        (
            ReleaseReason.MODEL_MIN_TRAINING_BATCHES,
            training_batches,
            settings.model_min_training_batches,
        ),
    )
    for reason, observed, minimum in checks:
        if minimum > 0 and observed < minimum:
            reasons.append(reason)

    task_minimums = dict(settings.effective_min_task_samples())
    for task_type, minimum in task_minimums.items():
        normalized = canonical_task_name(task_type)
        if minimum <= 0:
            continue
        observed = task_counts.get(normalized, 0)
        if observed <= 0:
            reasons.append(
                detail(ReleaseReason.TASK_MISSING_SAMPLES, normalized)
            )
        if observed < minimum:
            reasons.append(detail(ReleaseReason.TASK_MIN_SAMPLES, normalized))
        if requires_eval_task_coverage(task_type=normalized):
            eval_count = eval_task_count(
                task_counts_by_split=task_counts_by_split,
                task_type=normalized,
            )
            if task_counts_by_split and eval_count <= 0:
                reasons.append(
                    detail(ReleaseReason.TASK_EVAL_SAMPLES_MISSING, normalized)
                )

    reasons.extend(
        ratio_reasons(
            train_loss=train_loss,
            test_loss=test_loss,
            ratio_limit=(
                settings.model_max_test_train_loss_ratio
                if settings.model_max_test_train_loss_ratio is not None
                else settings.max_test_train_loss_ratio
            ),
            model=True,
        )
    )
    return tuple(reasons)


def requires_eval_task_coverage(*, task_type: str) -> bool:
    """Return whether a task must appear in validation or test splits."""

    return canonical_task_name(task_type) in {
        "audio_text_pair",
        "video_text_pair",
    }


def eval_task_count(
    *,
    task_counts_by_split: dict[str, dict[str, int]],
    task_type: str,
) -> int:
    """Return task coverage across validation and test splits."""

    normalized = canonical_task_name(task_type)
    return sum(
        task_counts_by_split.get(split_name, {}).get(normalized, 0)
        for split_name in (DEFAULT_VAL_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME)
    )


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of evaluating model-release artifact gates."""

    passed: bool
    missing_artifacts: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    evidence_hashes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "violations",
            tuple(str(violation) for violation in self.violations),
        )


def check_release(
    *,
    evidence: ReleaseEvidence,
    reproducibility_requirements: ReproducibilityRequirements | None = None,
) -> GateResult:
    """Validate persisted release evidence against the active release policy.

    For candidate and production stages the reproducibility report is
    fail-closed: the trusted active policy must be provided, the report must
    be cryptographically bound to it, and the release decision is re-derived
    from the active policy tolerances. A self-declared ``passed`` flag or
    ``violations`` list in the report is untrusted metadata.

    The gate recomputes the canonical reproducibility report from the
    authoritative run receipts + active policy and compares it to the
    persisted report.  Mismatch means the report has been tampered with or
    was produced under a different policy.
    """

    violations: list[str] = []
    mode = evidence.release_mode

    reproducibility_payload: dict[str, object] | None = None
    reproducibility_relative = ""
    run_receipts_payload: dict[str, object] | None = None

    evidence_hashes: list[tuple[str, str]] = []
    for reference in evidence.references:
        evidence_hashes.append((reference.name, reference.sha256))
        path = Path(reference.path)
        if not path.is_file():
            violations.append(
                detail(
                    ReleaseReason.RELEASE_ARTIFACT_INVALID,
                    reference.path,
                    "missing",
                )
            )
            continue
        if _sha256(path) != reference.sha256:
            violations.append(
                detail(
                    ReleaseReason.RELEASE_ARTIFACT_INVALID,
                    reference.path,
                    "digest_mismatch",
                )
            )
            continue
        if reference.name == "leakage":
            persisted_report: LeakageReportV2 | None = None
            try:
                persisted_report = load_report(path)
            except ValueError as exc:
                violations.append(
                    detail(
                        ReleaseReason.RELEASE_ARTIFACT_INVALID,
                        reference.path,
                        type(exc).__name__,
                    )
                )
            if persisted_report is None:
                pass
            elif evidence.leakage_report is None:
                violations.append(ReleaseReason.LEAKAGE_EVIDENCE_MISSING)
            elif persisted_report != evidence.leakage_report:
                violations.append(
                    detail(
                        ReleaseReason.RELEASE_ARTIFACT_INVALID,
                        reference.path,
                        "leakage_bundle_mismatch",
                    )
                )
            else:
                violations.extend(
                    _violations_for(
                        report=persisted_report,
                        relative=reference.path,
                    )
                )
            continue
        if reference.name == "reproducibility":
            try:
                report = load_reproducibility_report(path=path)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                violations.append(
                    detail(
                        ReleaseReason.RELEASE_ARTIFACT_INVALID,
                        reference.path,
                        type(exc).__name__,
                    )
                )
                continue

            reproducibility_payload = report
            reproducibility_relative = reference.path
            continue

        if reference.name == "run_receipts":
            try:
                collection = load_run_receipts_collection(path=path)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                violations.append(
                    detail(
                        ReleaseReason.RELEASE_ARTIFACT_INVALID,
                        reference.path,
                        type(exc).__name__,
                    )
                )
                continue

            run_receipts_payload = collection
            continue
        violations.extend(
            _artifact_violations(
                path=path,
                relative=reference.path,
                name=reference.name,
                mode=mode,
            )
        )

    # --- Reproducibility gate: compare persisted report with canonical evaluation ---
    if (
        reproducibility_payload is not None
        and run_receipts_payload is not None
    ):
        if reproducibility_requirements is not None:
            receipts = cast(
                list[TrainingRunReceipt],
                run_receipts_payload["run_receipts"],
            )

            expected_report = evaluate_reproducibility(
                receipts=receipts,
                policy=reproducibility_requirements,
                release_requirements_id=reproducibility_requirements.policy_id,
            )

            if reproducibility_payload != expected_report:
                violations.append(
                    detail(
                        ReleaseReason.RELEASE_ARTIFACT_INVALID,
                        reproducibility_relative,
                        "reproducibility_report_mismatch",
                    )
                )
        else:
            violations.append(
                detail(
                    ReleaseReason.RELEASE_ARTIFACT_INVALID,
                    reproducibility_relative,
                    "reproducibility_policy_missing",
                )
            )

    if mode in {"candidate", "production_model"}:
        violations.extend(_serving_artifact_reasons(evidence=evidence))

    return GateResult(
        passed=not violations,
        missing_artifacts=(),
        violations=tuple(violations),
        evidence_hashes=tuple(evidence_hashes),
    )


def _serving_artifact_reasons(
    *,
    evidence: ReleaseEvidence,
) -> tuple[str, ...]:
    """Verify deployable model artifacts exist for a production release.

    The export directory is derived from the ``model_card`` evidence reference;
    a ``skipped`` exporter status file is a hard gate failure for any required
    format, and at least one deployable format must be present.
    """

    from release.serving_artifacts import (
        check_serving_artifacts,
        default_serving_policy,
    )

    export_directory: Path | None = None
    for reference in evidence.references:
        if reference.name == "model_card":
            export_directory = Path(reference.path).parent
            break

    if export_directory is None:
        return (ReleaseReason.SERVING_ARTIFACT_MISSING,)

    policy = default_serving_policy(mode=evidence.release_mode)
    return check_serving_artifacts(
        export_directory=export_directory,
        policy=policy,
    )


def check_production(
    *,
    release_stage: str,
    require_model_accepted: bool,
    model_reasons: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Return production-only promotion reasons."""

    if release_stage != "production_model" or not require_model_accepted:
        return ()
    reasons: list[str] = []
    if model_reasons:
        reasons.append(ReleaseReason.PRODUCTION_REQUIREMENTS)
    return tuple(reasons)


def _artifact_violations(
    *,
    path: Path,
    relative: str,
    name: str,
    mode: ReleaseMode,
) -> list[str]:
    if name == "checkpoint":
        return []
    if name == "model_card":
        return list(check_model_card(path=path))
    payload, reason = _read_json_artifact(path=path, relative=relative)
    if reason is not None:
        return [reason]
    if payload is None:
        return [detail(ReleaseReason.RELEASE_ARTIFACT_INVALID, relative)]
    if (
        path.name == "acceptance_report.json"
        and payload.get("passed") is not True
    ):
        return [detail(ReleaseReason.ACCEPTANCE_REPORT_FAILED, relative)]
    return []


def _read_json_artifact(
    *,
    path: Path,
    relative: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Parse one JSON evidence artifact, returning its gate violation."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            detail(
                ReleaseReason.RELEASE_ARTIFACT_INVALID,
                relative,
                type(exc).__name__,
            ),
        )
    if not isinstance(payload, dict) or not payload:
        return None, detail(ReleaseReason.RELEASE_ARTIFACT_EMPTY, relative)
    if _contains_placeholder(payload):
        return (
            None,
            detail(ReleaseReason.RELEASE_ARTIFACT_PLACEHOLDER, relative),
        )
    return payload, None


def _violations_for(*, report: LeakageReportV2, relative: str) -> list[str]:
    violations = violations_for(report)
    if not violations:
        return []
    return [
        detail(
            ReleaseReason.LEAKAGE_REPORT_FAILED,
            relative,
            violation,
        )
        for violation in violations
    ]


def _contains_placeholder(payload: object) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered_key = str(key).lower()
            if lowered_key in {
                "placeholder",
                "simulated",
                "hardcoded",
            } and bool(value):
                return True
            if _contains_placeholder(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_placeholder(value) for value in payload)
    elif isinstance(payload, str):
        return payload.strip().lower() in {
            "placeholder",
            "simulated",
            "not_evaluated",
            "to-be-computed",
        }
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
