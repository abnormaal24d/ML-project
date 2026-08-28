"""Release-facing metric aggregation."""

from __future__ import annotations

from evaluator.metric_contracts import PAIR_TASK_ORDER


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


__all__ = ["summarize_task_metrics"]
