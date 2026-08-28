"""Pair retrieval metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from evaluator.metric_contracts import (
    PAIR_TASK_MODALITIES,
    PAIR_TASK_ORDER,
    PAIR_TASKS,
    EvaluationState,
)


class PairRetrievalStrategy:
    """Stateless scoring strategy for pair retrieval evaluation methods."""

    def __init__(self, *, evaluation_method: str) -> None:
        self.evaluation_method = evaluation_method

    def accumulate(
        self,
        *,
        state: Any,
        batch: Any,
        outputs: Mapping[str, Any],
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        if not isinstance(state, EvaluationState):
            return
        if not hasattr(state, "pair_text_embeddings"):
            return

        task_types = batch.task_types
        from multimodal.tasks.registry import get_task

        expected_rows = set()
        for index, task_type in enumerate(task_types):
            definition = get_task(task_type)
            if (
                task_type in PAIR_TASKS
                and definition is not None
                and definition.evaluation_method == self.evaluation_method
            ):
                expected_rows.add(index)

        if not expected_rows:
            return

        text_embedding = outputs.get("text_embedding")
        if text_embedding is None:
            raise ValueError("pair task batch is missing text_embedding")

        row_indices = outputs.get("contrastive_row_indices")
        if row_indices is None:
            raise ValueError(
                "pair task batch is missing contrastive_row_indices"
            )
        if not torch.is_tensor(text_embedding):
            raise TypeError("text_embedding must be a tensor")
        if text_embedding.ndim != 2:
            raise ValueError("text_embedding must be [rows, embedding_dim]")
        if not torch.is_tensor(row_indices):
            raise TypeError("contrastive_row_indices must be a tensor")
        if row_indices.ndim != 1:
            raise ValueError("contrastive_row_indices must be one-dimensional")
        if row_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                "contrastive_row_indices must have an integer dtype"
            )

        actual_rows = [int(index.item()) for index in row_indices]

        if len(actual_rows) != len(set(actual_rows)):
            raise ValueError("contrastive_row_indices contains duplicate rows")

        if set(actual_rows) != expected_rows:
            raise ValueError(
                "pair metric coverage does not match expected pair-task rows"
            )

        for row_index in row_indices:
            batch_index = int(row_index.item())
            if batch_index < 0 or batch_index >= len(task_types):
                raise ValueError("contrastive row index is outside task_types")
            if batch_index >= text_embedding.shape[0]:
                raise ValueError(
                    "contrastive row index is outside text_embedding"
                )
            task_type = task_types[batch_index]
            modality = PAIR_TASK_MODALITIES.get(task_type)
            if modality is None:
                modality = _resolve_pair_modality(
                    task_type=task_type,
                    batch=batch,
                    batch_index=batch_index,
                    outputs=outputs,
                )
            if modality is None:
                raise ValueError(
                    f"unable to resolve modality for pair-task row {batch_index}"
                )
            media_embedding = outputs.get(f"{modality}_embedding")
            if media_embedding is None:
                raise ValueError(
                    f"missing {modality}_embedding for pair-task row {batch_index}"
                )
            if not torch.is_tensor(media_embedding):
                raise TypeError(f"{modality}_embedding must be a tensor")
            if media_embedding.ndim != 2:
                raise ValueError(
                    f"{modality}_embedding must be [rows, embedding_dim]"
                )
            if batch_index >= media_embedding.shape[0]:
                raise ValueError(
                    "contrastive row index is outside media embedding"
                )
            state.pair_text_embeddings.setdefault(task_type, []).append(
                text_embedding[batch_index].detach()
            )
            state.pair_media_embeddings.setdefault(task_type, []).append(
                media_embedding[batch_index].detach()
            )

    def synchronize(
        self,
        *,
        state: Any,
        device: Any,
    ) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        from evaluator.distributed.collectives import gather_pair_embeddings

        if isinstance(state, EvaluationState):
            state.pair_text_embeddings, state.pair_media_embeddings = (
                gather_pair_embeddings(
                    text_embeddings_by_task=state.pair_text_embeddings,
                    media_embeddings_by_task=state.pair_media_embeddings,
                    device=device,
                )
            )

    def finalize(
        self,
        *,
        state: Any,
    ) -> dict[str, dict[str, float]]:
        task_metrics: dict[str, dict[str, float]] = {}

        from multimodal.tasks.registry import get_task

        for task_type in PAIR_TASK_ORDER:
            definition = get_task(task_type)
            if (
                definition is None
                or definition.evaluation_method != self.evaluation_method
            ):
                continue
            text_values = state.pair_text_embeddings.get(task_type, [])
            media_values = state.pair_media_embeddings.get(task_type, [])
            if not text_values or not media_values:
                continue
            task_metrics[task_type] = _retrieval_metrics(
                task_type=task_type,
                text_values=text_values,
                media_values=media_values,
            )
        return task_metrics


def _resolve_pair_modality(
    *,
    task_type: str,
    batch: Any,
    batch_index: int,
    outputs: Mapping[str, Any],
) -> str | None:
    configured = PAIR_TASK_MODALITIES.get(task_type)
    if configured is not None:
        return configured
    for modality in ("document", "image", "audio", "video"):
        if f"{modality}_embedding" not in outputs:
            continue
        mask = getattr(batch, f"{modality}_mask", None)
        if (
            mask is not None
            and hasattr(mask, "shape")
            and batch_index < mask.shape[0]
            and bool(mask[batch_index].item())
        ):
            return modality
    return None


def _retrieval_metrics(
    *,
    task_type: str,
    text_values: list[torch.Tensor],
    media_values: list[torch.Tensor],
) -> dict[str, float]:
    text_embedding = torch.cat(
        [value.reshape(-1, value.shape[-1]) for value in text_values],
        dim=0,
    ).float()
    media_embedding = torch.cat(
        [value.reshape(-1, value.shape[-1]) for value in media_values],
        dim=0,
    ).float()
    if text_embedding.shape != media_embedding.shape:
        raise ValueError(
            f"paired evaluation embeddings differ for {task_type}: "
            f"text={tuple(text_embedding.shape)}, "
            f"media={tuple(media_embedding.shape)}"
        )
    similarity = text_embedding @ media_embedding.transpose(0, 1)
    total = int(similarity.shape[0])
    if total == 0:
        raise ValueError(f"no paired embeddings for {task_type}")
    labels = torch.arange(total, device=similarity.device)
    positive_similarities = similarity.diagonal()
    _, ranked_indices = similarity.sort(dim=1, descending=True)

    metrics: dict[str, float] = {
        "embedding_similarity_mean": float(
            positive_similarities.mean().item()
        ),
        "recall_at_1": float(
            (ranked_indices[:, 0] == labels).float().mean().item()
        ),
    }
    top_five = min(5, total)
    metrics["recall_at_5"] = float(
        (ranked_indices[:, :top_five] == labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    if total > 1:
        negative_similarities = similarity.clone()
        negative_similarities.fill_diagonal_(float("-inf"))
        hardest_negative = negative_similarities.max(dim=1).values
        metrics["positive_negative_margin"] = float(
            (positive_similarities - hardest_negative).mean().item()
        )
    return metrics


__all__ = [
    "PairRetrievalStrategy",
]
