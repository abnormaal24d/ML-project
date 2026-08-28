"""Effective training-signal and modality-coverage checks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mmcrawler_datasets.snapshots.training_dataset_manifest import (
    _nonnegative_int,
)
from schemas.autonomous_readiness import (
    missing_autonomous_modalities,
    missing_autonomous_tasks,
)
from schemas.multimodal_tasks import canonical_task_name
from schemas.release import ReleaseReason, detail

if TYPE_CHECKING:
    from config.settings.datasets import DatasetValidatorSettings
    from training.runtime.results import TrainingMetrics


def check_modality_coverage(
    *,
    modality_counts: dict[str, int],
    targets: dict[str, int],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for mod, target in (targets or {}).items():
        obs = int(modality_counts.get(mod, 0))
        if obs < target:
            reasons.append(
                detail(
                    ReleaseReason.MODALITY_COVERAGE_LOW,
                    mod,
                    f"{obs}/{target}",
                )
            )
    return tuple(reasons)


def check_meets_minimums_from_report(
    report_path: Path | None,
) -> tuple[str, ...]:
    if report_path is None or not report_path.exists():
        return (ReleaseReason.COVERAGE_REPORT_MISSING,)
    try:
        import json

        data = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("coverage report root must be an object")
        if data.get("meets_minimums") is True:
            return ()
        return (ReleaseReason.RAW_COVERAGE_LOW,)
    except Exception as exc:  # exception-rules: optional-backend-probe
        return (
            detail(
                ReleaseReason.COVERAGE_REPORT_UNREADABLE,
                type(exc).__name__,
            ),
        )


def check_signals(
    *,
    settings: DatasetValidatorSettings,
    metrics: TrainingMetrics,
) -> tuple[str, ...]:
    """Return effective data and gradient-signal acceptance reasons."""

    reasons: list[str] = []

    sample_count = _nonnegative_int(metrics.effective_train_sample_count)
    task_counts = _coerce_count_map(
        metrics.effective_task_counts,
        normalize_task_names=True,
    )
    modality_counts = _coerce_count_map(
        metrics.effective_modality_counts,
        normalize_task_names=False,
    )
    training_signal = _coerce_training_signal(
        metrics.training_signal_by_modality
    )

    if sample_count <= 0:
        reasons.append(ReleaseReason.EFFECTIVE_SAMPLES_MISSING)
    if sample_count != _nonnegative_int(metrics.samples):
        reasons.append(
            detail(
                ReleaseReason.EFFECTIVE_SAMPLE_MISMATCH,
                f"{sample_count}/{_nonnegative_int(metrics.samples)}",
            )
        )
    if not task_counts:
        reasons.append(ReleaseReason.EFFECTIVE_TASK_COUNTS_MISSING)
    if not modality_counts:
        reasons.append(ReleaseReason.EFFECTIVE_MODALITY_COUNTS_MISSING)

    _append_training_signal_reasons(
        reasons=reasons,
        modality_counts=modality_counts,
        training_signal=training_signal,
    )
    _append_required_task_reasons(
        reasons=reasons,
        settings=settings,
        task_counts=task_counts,
    )
    if settings.require_autonomous_multimodal_readiness:
        _append_autonomous_multimodal_reasons(
            reasons=reasons,
            task_counts=task_counts,
            modality_counts=modality_counts,
        )
    return tuple(reasons)


def _append_training_signal_reasons(
    *,
    reasons: list[str],
    modality_counts: dict[str, int],
    training_signal: dict[str, dict[str, object]],
) -> None:
    if not training_signal:
        reasons.append(ReleaseReason.SIGNALS_MISSING)

    for modality, count in sorted(modality_counts.items()):
        if count <= 0 or modality == "unknown":
            continue

        signal = training_signal.get(modality)
        if signal is None:
            reasons.append(detail(ReleaseReason.SIGNAL_MISSING, modality))
            continue

        if _nonnegative_int(signal.get("trainable_parameter_count")) <= 0:
            reasons.append(
                detail(ReleaseReason.SIGNAL_PARAMETERS_MISSING, modality)
            )
        if (
            _nonnegative_int(signal.get("gradient_observations")) <= 0
            or _nonnegative_float(signal.get("max_gradient_l2")) <= 0.0
            or signal.get("gradient_detected") is not True
        ):
            reasons.append(
                detail(ReleaseReason.SIGNAL_GRADIENT_MISSING, modality)
            )
        if (
            _nonnegative_float(signal.get("parameter_delta_l2")) <= 0.0
            or signal.get("updated") is not True
        ):
            reasons.append(
                detail(ReleaseReason.SIGNAL_UPDATE_MISSING, modality)
            )


def _append_required_task_reasons(
    *,
    reasons: list[str],
    settings: DatasetValidatorSettings,
    task_counts: dict[str, int],
) -> None:
    for raw_task, raw_minimum in sorted(
        settings.effective_min_task_samples().items()
    ):
        task = canonical_task_name(raw_task)
        minimum = _nonnegative_int(raw_minimum)
        if minimum <= 0:
            continue
        observed = task_counts.get(task, 0)
        if observed < minimum:
            reasons.append(
                detail(
                    ReleaseReason.EFFECTIVE_TASK_MINIMUM,
                    task,
                    f"{observed}/{minimum}",
                )
            )


def _append_autonomous_multimodal_reasons(
    *,
    reasons: list[str],
    task_counts: dict[str, int],
    modality_counts: dict[str, int],
) -> None:
    missing_modalities = missing_autonomous_modalities(modality_counts)
    if missing_modalities:
        reasons.append(
            detail(
                ReleaseReason.AUTONOMOUS_MODALITIES_MISSING,
                ",".join(missing_modalities),
            )
        )

    missing_tasks = missing_autonomous_tasks(task_counts)
    if missing_tasks:
        reasons.append(
            detail(
                ReleaseReason.AUTONOMOUS_TASKS_MISSING,
                ",".join(missing_tasks),
            )
        )


def _coerce_count_map(
    raw: dict[str, int],
    *,
    normalize_task_names: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if not name:
            continue
        if normalize_task_names:
            name = canonical_task_name(name)
        counts[name] = counts.get(name, 0) + _nonnegative_int(value)
    return dict(sorted(counts.items()))


def _coerce_training_signal(
    raw: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    signal_by_modality: dict[str, dict[str, object]] = {}
    for raw_modality, raw_signal in raw.items():
        modality = str(raw_modality).strip().lower()
        if not modality or not isinstance(raw_signal, dict):
            continue
        signal_by_modality[modality] = dict(raw_signal)
    return signal_by_modality


def _nonnegative_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return 0.0
