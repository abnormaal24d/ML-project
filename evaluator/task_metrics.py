"""Public task-metric evaluation API over complete evaluation splits."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager

import torch

from evaluator.metric_contracts import PAIR_TASK_ORDER
from evaluator.metric_runner import evaluate, evaluate_with_runtime
from multimodal.model.contracts import CollatedBatch
from multimodal.model.model import MultimodalModel
from multimodal.tokenization.text import VocabularyTokenizer

__all__ = [
    "evaluate_task_metrics",
    "evaluate_task_metrics_with_runtime",
    "summarize_task_metrics",
]


def evaluate_task_metrics(
    *,
    model: MultimodalModel,
    loader: Iterable[CollatedBatch],
    device: torch.device | None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: VocabularyTokenizer | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-task metrics over one complete evaluation split."""

    return evaluate(
        model=model,
        loader=loader,
        device=device,
        autocast_factory=autocast_factory,
        tokenizer=tokenizer,
    )


def evaluate_task_metrics_with_runtime(
    *,
    model: MultimodalModel,
    loader: Iterable[CollatedBatch],
    device: torch.device | None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
    tokenizer: VocabularyTokenizer | None = None,
) -> tuple[dict[str, dict[str, float]], float | None, float | None]:
    """Compute task metrics plus observed batch latency and peak memory.

    Returns ``(task_metrics, max_batch_latency_ms, peak_memory_mb)``.
    Peak memory comes from the CUDA allocator when the device is a GPU,
    otherwise from the resident-set size of this process when readable,
    and is ``None`` when no measurement is available.
    """

    return evaluate_with_runtime(
        model=model,
        loader=loader,
        device=device,
        autocast_factory=autocast_factory,
        tokenizer=tokenizer,
    )


def summarize_task_metrics(
    task_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate canonical release-facing metrics from per-task metrics."""

    metrics: dict[str, float] = {}
    recall_at_1_values = [
        task_metrics[task]["recall_at_1"]
        for task in PAIR_TASK_ORDER
        if task in task_metrics and "recall_at_1" in task_metrics[task]
    ]
    if recall_at_1_values:
        metrics["mean_recall_at_1"] = sum(recall_at_1_values) / len(
            recall_at_1_values
        )

    similarity_values = [
        task_metrics[task]["embedding_similarity_mean"]
        for task in PAIR_TASK_ORDER
        if task in task_metrics
        and "embedding_similarity_mean" in task_metrics[task]
    ]
    if similarity_values:
        metrics["embedding_similarity_mean"] = sum(similarity_values) / len(
            similarity_values
        )

    return metrics
