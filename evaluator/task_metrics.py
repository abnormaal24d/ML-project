"""Public task-metric evaluation API over complete evaluation splits."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager

import torch

from evaluator.aggregation import summarize_task_metrics
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
