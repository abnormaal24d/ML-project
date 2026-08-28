"""Fail-fast validation for configured task thresholds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from multimodal.tasks.registry import get_task
from schemas.multimodal_tasks import canonical_task_name

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings
    from config.settings.datasets import DatasetValidatorSettings


def task_minimums(settings: DatasetValidatorSettings) -> dict[str, int]:
    """Return normalized task thresholds for the active workflow."""

    return {
        canonical_task_name(name): _count(value)
        for name, value in settings.effective_min_task_samples().items()
        if name
    }


def validate_task_threshold_preflight(
    *,
    settings: DatasetValidatorSettings,
    available_task_types: Iterable[str],
) -> None:
    """Reject task minimums that the active workflow cannot produce."""

    workflow = settings.workflow_profile.strip() or "crawler_dataset"
    available = {canonical_task_name(name) for name in available_task_types}
    required = {
        name: minimum
        for name, minimum in task_minimums(settings).items()
        if minimum > 0
    }
    external = {
        name: minimum
        for name, minimum in required.items()
        if _requires_labels(name, workflow)
    }
    if external:
        messages = (
            external_label_remediation(name, minimum, workflow)
            for name, minimum in sorted(external.items())
        )
        raise ValueError(" ".join(messages))

    impossible = {
        name: minimum
        for name, minimum in required.items()
        if name not in available
    }
    if impossible:
        messages = (
            threshold_remediation(name, 0, minimum, workflow)
            for name, minimum in sorted(impossible.items())
        )
        raise ValueError(" ".join(messages))


def threshold_remediation(
    task: str,
    observed: int,
    minimum: int,
    workflow: str,
) -> str:
    """Explain how to resolve an unmet task minimum."""

    if task == "ui_to_code":
        return (
            "Task threshold ui_to_code requires "
            f"{minimum} sample, but workflow {workflow!r} has no "
            "ui_to_code builder. Fix: set min_task_samples.ui_to_code=0 "
            "for the crawler profile or enable a ui_to_code sample source."
        )
    return (
        f"Task threshold {task} requires {minimum} sample(s), observed "
        f"{observed}. Fix: lower this threshold for workflow {workflow!r} "
        f"or enable a {task} sample source."
    )


def external_label_remediation(
    task: str,
    minimum: int,
    workflow: str,
) -> str:
    """Explain why crawler data cannot satisfy an external-label task."""

    return (
        f"Task threshold {task} requires {minimum} externally labeled "
        f"sample(s), but workflow {workflow!r} is crawler-derived. "
        f"Fix: set min_task_samples.{task}=0 for crawler data or enable "
        "an annotated/synthetic sample source."
    )


def _requires_labels(task: str, workflow: str) -> bool:
    if workflow != "crawler_dataset":
        return False
    definition = get_task(task)
    return bool(definition is not None and definition.requires_external_labels)


def _count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class EffectiveTrainingSplitReport:
    """Counts proving which configured tasks reach the trainer."""

    sample_count: int
    task_counts: dict[str, int]
    modality_counts: dict[str, int]


def validate_effective_training_split(
    *,
    dataset: Any,
    training_settings: TrainingSettings,
) -> EffectiveTrainingSplitReport:
    """Reject a loaded train split that cannot satisfy task requirements."""

    sample_count = len(dataset)
    if sample_count == 0:
        raise ValueError("training dataset is empty")

    task_counts, modality_counts = _dataset_counts(dataset=dataset)
    missing_tasks = _missing_required_tasks(
        task_counts=task_counts,
        min_task_samples=training_settings.effective_min_task_samples(),
    )
    if missing_tasks:
        details = ";".join(
            f"{task}:{observed}/{minimum}"
            for task, observed, minimum in missing_tasks
        )
        raise ValueError(f"training_required_tasks_below_min:{details}")

    return EffectiveTrainingSplitReport(
        sample_count=sample_count,
        task_counts=dict(sorted(task_counts.items())),
        modality_counts=dict(sorted(modality_counts.items())),
    )


def _dataset_counts(
    *,
    dataset: Any,
) -> tuple[Counter[str], Counter[str]]:
    task_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    for index in range(len(dataset)):
        task_type = canonical_task_name(
            _dataset_index_value(dataset, "task_type_at", index)
        )
        task_counts[task_type] += 1
        for modality in _dataset_modality_signature(
            dataset=dataset,
            index=index,
        ):
            modality_counts[modality] += 1
    return task_counts, modality_counts


def _missing_required_tasks(
    *,
    task_counts: Counter[str],
    min_task_samples: object,
) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(min_task_samples, dict):
        return ()
    missing: list[tuple[str, int, int]] = []
    for raw_task, raw_minimum in sorted(min_task_samples.items()):
        task = canonical_task_name(raw_task)
        minimum = _count(raw_minimum)
        if minimum <= 0:
            continue
        observed = int(task_counts.get(task, 0))
        if observed < minimum:
            missing.append((task, observed, minimum))
    return tuple(missing)


def _dataset_modality_signature(
    *,
    dataset: Any,
    index: int,
) -> tuple[str, ...]:
    signature = _dataset_index_value(dataset, "modality_signature_at", index)
    if isinstance(signature, str):
        value = signature.strip().lower()
        return (value,) if value else ("unknown",)
    if isinstance(signature, (tuple, list)):
        values = tuple(
            str(item).strip().lower()
            for item in signature
            if str(item).strip()
        )
        return values or ("unknown",)

    sample = dataset[index]
    values = tuple(
        modality
        for modality in ("text", "document", "image", "audio", "video")
        if bool(getattr(sample, f"has_{modality}", False))
    )
    return values or ("unknown",)


def _dataset_index_value(
    dataset: Any,
    method_name: str,
    index: int,
) -> Any:
    method = getattr(dataset, method_name, None)
    if callable(method):
        return method(index)
    return getattr(dataset[index], method_name.removesuffix("_at"), "")


__all__ = [
    "EffectiveTrainingSplitReport",
    "task_minimums",
    "validate_effective_training_split",
    "validate_task_threshold_preflight",
]
