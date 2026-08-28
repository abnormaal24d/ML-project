"""Single evaluation loop orchestrating all metric strategies."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Mapping

import torch

from evaluator.metric_contracts import (
    EvaluationPlan,
    EvaluationState,
    RuntimeObserver,
)
from multimodal.model.contracts import CollatedBatch


def _prepare_task_metric_batch(
    *,
    batch: Any,
    device: Any,
) -> CollatedBatch:
    """Validate the task-metric batch contract and move its tensors."""
    if not isinstance(batch, CollatedBatch):
        raise TypeError("task metric batches must be CollatedBatch instances")
    if not batch.sample_ids:
        raise ValueError("task metric batches must not be empty")
    batch_size = len(batch.sample_ids)
    if len(batch.task_types) != batch_size:
        raise ValueError(
            "task_types must align with sample_ids: "
            f"{len(batch.task_types)} != {batch_size}"
        )
    for field_name in (
        "text_mlm_targets",
        "decoder_labels",
        "layout_box_targets",
        "document_mask",
        "image_mask",
        "audio_mask",
        "video_mask",
    ):
        value = getattr(batch, field_name)
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] != batch_size
        ):
            raise ValueError(
                f"{field_name} rows must match the batch: "
                f"{value.shape[0]} != {batch_size}"
            )
    if device is None:
        return batch

    from mmcrawler_datasets.collation.multimodal import move_batch_to_device

    return move_batch_to_device(batch=batch, device=device)


def _evaluate_task_metric_model(
    *,
    model: Any,
    batch: CollatedBatch,
) -> Mapping[str, Any]:
    """Run the model once and enforce the mapping output contract."""
    outputs = model(batch)
    if not isinstance(outputs, Mapping):
        raise TypeError("model outputs must be a mapping of metric tensors")
    return outputs


def _evaluation_methods_for_batch(batch: CollatedBatch) -> set[str]:
    from multimodal.tasks.registry import get_task

    methods: set[str] = set()
    for task_type in batch.task_types:
        definition = get_task(task_type)
        if definition is not None:
            methods.add(definition.evaluation_method)
    return methods


def _run_evaluation(
    *,
    model: Any,
    loader: Any,
    device: Any,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: Any,
    evaluation_plans: Mapping[str, EvaluationPlan] | None,
    runtime_observer: RuntimeObserver | None,
) -> tuple[dict[str, dict[str, float]], float | None, float | None]:
    if loader is None:
        raise RuntimeError("evaluation loader is required")

    if evaluation_plans is None:
        from evaluator.metric_registry import EVALUATION_METHODS

        evaluation_plans = EVALUATION_METHODS

    state = EvaluationState()
    used_methods: set[str] = set()
    max_latency_ms: float | None = None
    peak_memory_mb = (
        runtime_observer.peak_memory_mb(device=device)
        if runtime_observer is not None
        else None
    )
    was_training = bool(model.training)
    model.eval()

    try:
        with torch.no_grad(), autocast_factory():
            for raw_batch in loader:
                batch = _prepare_task_metric_batch(
                    batch=raw_batch,
                    device=device,
                )
                if runtime_observer is not None:
                    runtime_observer.start_batch(device=device)

                outputs = _evaluate_task_metric_model(
                    model=model,
                    batch=batch,
                )

                if runtime_observer is not None:
                    elapsed_ms = runtime_observer.end_batch(device=device)
                    if max_latency_ms is None or elapsed_ms > max_latency_ms:
                        max_latency_ms = elapsed_ms
                    observed_memory_mb = runtime_observer.peak_memory_mb(
                        device=device
                    )
                    if observed_memory_mb is not None and (
                        peak_memory_mb is None
                        or observed_memory_mb > peak_memory_mb
                    ):
                        peak_memory_mb = observed_memory_mb

                needed_methods = _evaluation_methods_for_batch(batch)
                unsupported_methods = needed_methods - evaluation_plans.keys()
                if unsupported_methods:
                    raise ValueError(
                        "unsupported evaluation method(s): "
                        + ", ".join(sorted(unsupported_methods))
                    )
                used_methods.update(needed_methods)

                for method, plan in evaluation_plans.items():
                    if method not in needed_methods:
                        continue
                    plan.scorer.accumulate(
                        state=state,
                        batch=batch,
                        outputs=outputs,
                        tokenizer=tokenizer,
                        model=model,
                    )
    finally:
        model.train(was_training)

    for method, plan in evaluation_plans.items():
        if method in used_methods:
            plan.scorer.synchronize(state=state, device=device)

    task_metrics: dict[str, dict[str, float]] = {}
    for method, plan in evaluation_plans.items():
        if method in used_methods:
            task_metrics.update(plan.scorer.finalize(state=state))

    return task_metrics, max_latency_ms, peak_memory_mb


def evaluate(
    *,
    model: Any,
    loader: Any,
    device: Any,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: Any = None,
    evaluation_plans: Mapping[str, EvaluationPlan] | None = None,
    runtime_observer: RuntimeObserver | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-task metrics over one complete evaluation split."""
    metrics, _, _ = _run_evaluation(
        model=model,
        loader=loader,
        device=device,
        autocast_factory=autocast_factory,
        tokenizer=tokenizer,
        evaluation_plans=evaluation_plans,
        runtime_observer=runtime_observer,
    )
    return metrics


def evaluate_with_runtime(
    *,
    model: Any,
    loader: Any,
    device: Any,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: Any = None,
    evaluation_plans: Mapping[str, EvaluationPlan] | None = None,
) -> tuple[dict[str, dict[str, float]], float | None, float | None]:
    """Compute task metrics plus observed latency and peak memory."""
    from evaluator.runtime_metrics import (
        _peak_memory_mb,
        create_runtime_observer,
    )

    observer = create_runtime_observer(
        device=device,
        peak_memory_mb_fn=_peak_memory_mb,
    )
    observer.reset(device=device)
    metrics, max_latency_ms, peak_memory_mb = _run_evaluation(
        model=model,
        loader=loader,
        device=device,
        autocast_factory=autocast_factory,
        tokenizer=tokenizer,
        evaluation_plans=evaluation_plans,
        runtime_observer=observer,
    )
    if peak_memory_mb is None:
        peak_memory_mb = _peak_memory_mb(device=device)
    return metrics, max_latency_ms, peak_memory_mb


__all__ = [
    "evaluate",
    "evaluate_with_runtime",
]
