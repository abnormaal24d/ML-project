"""Training coverage validation: limits, snapshots, and checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mmcrawler_datasets.validation.training_preflight import task_minimums
from schemas.autonomous_readiness import (
    AUTONOMOUS_REQUIRED_MODALITIES,
    AUTONOMOUS_REQUIRED_TASKS,
    missing_autonomous_modalities,
    missing_autonomous_tasks,
)
from schemas.multimodal_tasks import canonical_task_name

if TYPE_CHECKING:
    from config.settings.datasets import DatasetValidatorSettings


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """Counts consumed by the stateless coverage checks."""

    modalities: Mapping[str, int]
    tasks: Mapping[str, int]
    splits: Mapping[str, Mapping[str, int]]
    aligned: int
    total: int


# Stable coverage failure reason codes.
MODALITY = "modality_coverage_below_min"
ALIGNMENT = "alignment_coverage_below_min"
TASK = "task_coverage_below_min"
SPLIT = "split_missing_required_modalities"
AUTO_MODALITY = "autonomous_multimodal_readiness_missing_modalities"
AUTO_TASK = "autonomous_multimodal_readiness_missing_tasks"
AUTO_GENERATION = "autonomous_multimodal_generation_missing_tasks"


@dataclass(frozen=True, slots=True)
class CoverageLimits:
    """Minimum modality, task, and alignment coverage."""

    modalities: Mapping[str, int]
    tasks: Mapping[str, int]
    min_alignment: float
    required_splits: tuple[str, ...] = ()


def from_settings(settings: DatasetValidatorSettings) -> CoverageLimits:
    """Build concrete limits from dataset-validator settings."""

    required_splits = split_names(settings)
    return CoverageLimits(
        modalities={
            "text": settings.min_text_samples,
            "document": settings.min_document_samples,
            "image": settings.min_image_samples,
            "audio": settings.min_audio_samples,
            "video": settings.min_video_samples,
        },
        tasks=task_minimums(settings),
        min_alignment=settings.min_alignment_coverage,
        required_splits=required_splits,
    )


def split_names(settings: DatasetValidatorSettings) -> tuple[str, ...]:
    """Return only splits configured to contain every modality."""

    return tuple(
        name
        for name, required in (
            ("train", settings.require_all_modalities_in_train),
            ("val", settings.require_all_modalities_in_val),
            ("test", settings.require_all_modalities_in_test),
        )
        if required
    )


_GENERATION_TASKS = (
    "text_pretrain",
    "speech_translation",
    "text_to_image",
    "text_to_video",
)


@dataclass(frozen=True, slots=True)
class CoverageIssue:
    """One observed value that does not meet its threshold."""

    code: str
    subject: str
    observed: float
    required: float


def check_coverage(
    snapshot: CoverageSnapshot,
    limits: CoverageLimits,
) -> tuple[CoverageIssue, ...]:
    """Compare one observation with concrete, non-polymorphic limits."""

    issues: list[CoverageIssue] = []
    for modality, minimum in limits.modalities.items():
        required = _count(minimum)
        observed = _count(snapshot.modalities.get(modality, 0))
        if required > 0 and observed < required:
            issues.append(
                CoverageIssue(
                    code=MODALITY,
                    subject=modality,
                    observed=float(observed),
                    required=float(required),
                )
            )

    for task, minimum in sorted(limits.tasks.items()):
        required = _count(minimum)
        observed = _count(snapshot.tasks.get(task, 0))
        if required > 0 and observed < required:
            issues.append(
                CoverageIssue(
                    code=TASK,
                    subject=task,
                    observed=float(observed),
                    required=float(required),
                )
            )

    alignment = coverage_ratio(snapshot.aligned, snapshot.total)
    if limits.min_alignment > 0 and alignment < limits.min_alignment:
        issues.append(
            CoverageIssue(
                code=ALIGNMENT,
                subject="samples",
                observed=alignment,
                required=limits.min_alignment,
            )
        )

    if limits.required_splits:
        for split in limits.required_splits:
            counts = snapshot.splits.get(split, {})
            for modality in limits.modalities:
                if _count(counts.get(modality, 0)) <= 0:
                    issues.append(
                        CoverageIssue(
                            code=SPLIT,
                            subject=f"{split}:{modality}",
                            observed=0.0,
                            required=1.0,
                        )
                    )
    return tuple(issues)


def issue_messages(issues: tuple[CoverageIssue, ...]) -> tuple[str, ...]:
    """Render issues using the stable external error format."""

    messages: list[str] = []
    split_modalities: dict[str, list[str]] = {}
    for issue in issues:
        if issue.code == SPLIT:
            split, _, modality = issue.subject.partition(":")
            split_modalities.setdefault(split, []).append(modality)
            continue
        if issue.code == ALIGNMENT:
            messages.append(
                f"{issue.code}:{issue.observed:.4f}/{issue.required:.4f}"
            )
            continue
        messages.append(
            f"{issue.code}:{issue.subject}:"
            f"{_number(issue.observed)}/{_number(issue.required)}"
        )
    messages.extend(
        f"{SPLIT}:{split}:{','.join(modalities)}"
        for split, modalities in split_modalities.items()
    )
    return tuple(messages)


def readiness_messages(
    snapshot: CoverageSnapshot,
    *,
    required: bool,
    generation_targets: bool,
    modalities: tuple[str, ...] = AUTONOMOUS_REQUIRED_MODALITIES,
    tasks: tuple[str, ...] = AUTONOMOUS_REQUIRED_TASKS,
) -> tuple[str, ...]:
    """Return autonomous-readiness failures for a coverage snapshot."""

    if not required:
        return ()
    messages: list[str] = []
    missing_modalities = [
        name
        for name in modalities
        if _count(snapshot.modalities.get(name)) <= 0
    ]
    if missing_modalities:
        messages.append(f"{AUTO_MODALITY}:{','.join(missing_modalities)}")
    missing_tasks = [
        name for name in tasks if _count(snapshot.tasks.get(name)) <= 0
    ]
    if missing_tasks:
        messages.append(f"{AUTO_TASK}:{','.join(missing_tasks)}")
    if generation_targets:
        missing_generation = [
            name
            for name in _GENERATION_TASKS
            if _count(snapshot.tasks.get(name)) <= 0
        ]
        if missing_generation:
            messages.append(
                f"{AUTO_GENERATION}:{','.join(missing_generation)}"
            )
    return tuple(messages)


def training_evidence_errors(
    metrics: Mapping[str, object],
    current_modalities: Mapping[str, int],
    task_minimums: Mapping[str, int],
    *,
    require_autonomous: bool,
) -> tuple[str, ...] | None:
    """Validate evidence that each active modality affected training."""

    effective_tasks = _counts(
        metrics.get("effective_task_counts"),
        normalize_tasks=True,
    )
    effective_modalities = _counts(
        metrics.get("effective_modality_counts"),
        normalize_tasks=False,
    )
    signals = _signals(metrics.get("training_signal_by_modality"))
    if _count(metrics.get("effective_train_sample_count")) <= 0:
        return ("effective training sample count is missing",)
    if not effective_tasks:
        return ("effective task counts are missing",)
    if not effective_modalities:
        return ("effective modality counts are missing",)
    if not signals:
        return ("training signal by modality is missing",)

    missing: list[str] = []
    for raw_task, raw_minimum in sorted(task_minimums.items()):
        task = canonical_task_name(raw_task)
        minimum = _count(raw_minimum)
        if minimum > 0 and effective_tasks.get(task, 0) < minimum:
            missing.append(f"effective_task_minimum_missing:{task}")

    if require_autonomous:
        for modality in missing_autonomous_modalities(effective_modalities):
            missing.append(f"effective_autonomous_modality_missing:{modality}")
        for task in missing_autonomous_tasks(effective_tasks):
            missing.append(f"effective_autonomous_task_missing:{task}")

    for modality, raw_count in sorted(current_modalities.items()):
        if _count(raw_count) <= 0:
            continue
        if effective_modalities.get(modality, 0) <= 0:
            missing.append(f"effective_modality_missing:{modality}")
            continue
        signal = signals.get(modality)
        if signal is None:
            missing.append(f"training_signal_missing:{modality}")
            continue
        if signal.get("updated") is not True:
            missing.append(f"training_signal_update_missing:{modality}")
        if signal.get("gradient_detected") is not True:
            missing.append(f"training_signal_gradient_missing:{modality}")
    if not missing:
        return None
    return ("training output lacks effective multimodal evidence", *missing)


def coverage_ratio(numerator: int, denominator: int) -> float:
    """Return the precise ratio, treating an empty population as zero."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _counts(raw: object, *, normalize_tasks: bool) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key or "").strip().lower()
        if normalize_tasks:
            name = canonical_task_name(name)
        if name:
            counts[name] = counts.get(name, 0) + _count(value)
    return counts


def _signals(raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, Mapping):
        return {}
    signals: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        name = str(key or "").strip().lower()
        if name and isinstance(value, Mapping):
            signals[name] = dict(value)
    return signals


def _count(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


__all__ = [
    "CoverageIssue",
    "CoverageLimits",
    "CoverageSnapshot",
    "check_coverage",
    "coverage_ratio",
    "from_settings",
    "issue_messages",
    "readiness_messages",
    "split_names",
    "training_evidence_errors",
]
