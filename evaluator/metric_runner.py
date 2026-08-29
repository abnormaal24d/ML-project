"""Single evaluation loop orchestrating all metric strategies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager

import torch
from torch import Tensor

from evaluator.metric_contracts import (
    EvaluationPlan,
    EvaluationState,
    RuntimeObserver,
)
from multimodal.model.contracts import CollatedBatch
from multimodal.model.model import MultimodalModel
from multimodal.tokenization.text import VocabularyTokenizer


def _prepare_task_metric_batch(
    *,
    batch: CollatedBatch,
    device: torch.device | None,
) -> CollatedBatch:
    """Validate the task-metric batch contract and move its tensors."""

    if not batch.sample_ids:
        raise ValueError("task metric batches must not be empty")
    batch_size = len(batch.sample_ids)
    if len(batch.task_types) != batch_size:
        raise ValueError(
            "task_types must align with sample_ids: "
            f"{len(batch.task_types)} != {batch_size}"
        )

    tensor_rows = (
        ("text_mlm_targets", batch.text_mlm_targets),
        ("decoder_labels", batch.decoder_labels),
        ("layout_box_targets", batch.layout_box_targets),
        ("document_mask", batch.document_mask),
        ("image_mask", batch.image_mask),
        ("audio_mask", batch.audio_mask),
        ("video_mask", batch.video_mask),
    )
    for field_name, value in tensor_rows:
        if value is not None and value.ndim > 0 and value.shape[0] != batch_size:
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
    model: MultimodalModel,
    batch: CollatedBatch,
) -> Mapping[str, Tensor]:
    """Run the model once and enforce the metric tensor output contract."""

    outputs = model(batch)
    if not isinstance(outputs, Mapping):
        raise TypeError("model outputs must be a mapping of metric tensors")
    invalid = [name for name, value in outputs.items() if not isinstance(value, Tensor)]
    if invalid:
        raise TypeError(
            "metric model outputs must contain tensors only: "
            + ", ".join(sorted(str(name) for name in invalid))
        )
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
    model: MultimodalModel,
    loader: Iterable[CollatedBatch],
    device: torch.device | None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: VocabularyTokenizer | None,
    evaluation_plans: Mapping[str, EvaluationPlan] | None,
    runtime_observer: RuntimeObserver | None,
) -> tuple[dict[str, dict[str, float]], float | None, float | None]:
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
            for batch in loader:
                prepared_batch = _prepare_task_metric_batch(
                    batch=batch,
                    device=device,
                )
                if runtime_observer is not None:
                    runtime_observer.start_batch(device=device)

                outputs = _evaluate_task_metric_model(
                    model=model,
                    batch=prepared_batch,
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

                needed_methods = _evaluation_methods_for_batch(prepared_batch)
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
                        batch=prepared_batch,
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
    model: MultimodalModel,
    loader: Iterable[CollatedBatch],
    device: torch.device | None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: VocabularyTokenizer | None = None,
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
    model: MultimodalModel,
    loader: Iterable[CollatedBatch],
    device: torch.device | None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: VocabularyTokenizer | None = None,
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
