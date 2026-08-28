"""Masked Language Modeling metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

IGNORE_LABEL = -100


class MLMStrategy:
    """Stateless scoring strategy for masked language modeling evaluation."""

    evaluation_method = "masked_language_modeling"

    def accumulate(
        self,
        *,
        state: Any,
        batch: Any,
        outputs: Mapping[str, Any],
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        task_types = batch.task_types
        has_text_pretrain = "text_pretrain" in task_types

        if not has_text_pretrain:
            return

        mlm_logits = outputs.get("text_mlm_logits")
        mlm_targets = batch.text_mlm_targets

        if mlm_targets is None:
            raise ValueError(
                "text_pretrain evaluation requires text_mlm_targets"
            )

        if mlm_logits is None:
            raise ValueError(
                "text_pretrain evaluation requires text_mlm_logits"
            )
        if not torch.is_tensor(mlm_logits):
            raise TypeError("text_mlm_logits must be a tensor")
        batch_size = len(task_types)
        if mlm_logits.ndim != 3:
            raise ValueError(
                "text_mlm_logits must be [batch, tokens, vocabulary]"
            )
        if mlm_logits.shape[0] != batch_size:
            raise ValueError(
                "text_mlm_logits rows must match the batch: "
                f"{mlm_logits.shape[0]} != {batch_size}"
            )
        if mlm_targets.shape[0] != batch_size:
            raise ValueError(
                "text_mlm_targets rows must match the batch: "
                f"{mlm_targets.shape[0]} != {batch_size}"
            )

        predictions = mlm_logits.argmax(dim=-1)
        for row_index in range(predictions.shape[0]):
            if (
                row_index >= len(task_types)
                or task_types[row_index] != "text_pretrain"
            ):
                continue
            target_row = mlm_targets[row_index]
            prediction_row = predictions[row_index]
            valid = target_row != -100  # IGNORE_LABEL
            if valid.any():
                state.mlm_state.correct += int(
                    (prediction_row[valid] == target_row[valid]).sum().item()
                )
                state.mlm_state.total += int(valid.sum().item())

    def synchronize(
        self,
        *,
        state: Any,
        device: Any,
    ) -> None:
        from evaluator.distributed.reductions import reduce_mlm_state

        state.mlm_state.correct, state.mlm_state.total = reduce_mlm_state(
            correct=state.mlm_state.correct,
            total=state.mlm_state.total,
            device=device,
        )

    def finalize(
        self,
        *,
        state: Any,
    ) -> dict[str, dict[str, float]]:
        task_metrics: dict[str, dict[str, float]] = {}
        if state.mlm_state.total > 0:
            task_metrics.setdefault("text_pretrain", {})
            task_metrics["text_pretrain"]["masked_token_accuracy"] = (
                state.mlm_state.correct / state.mlm_state.total
            )
        return task_metrics


__all__ = [
    "MLMStrategy",
]
