"""Causal Language Modeling metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch.nn.functional as F

if TYPE_CHECKING:
    pass


IGNORE_LABEL = -100


class CausalLMStrategy:
    """Stateless scoring strategy for causal language modeling evaluation."""

    evaluation_method = "causal_language_modeling"

    def accumulate(
        self,
        *,
        state: Any,
        batch: Any,
        outputs: Mapping[str, Any],
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        logits = outputs.get("sequence_logits")
        labels = batch.decoder_labels

        if logits is None:
            raise ValueError("causal LM evaluation requires sequence_logits")
        if labels is None:
            raise ValueError("causal LM evaluation requires decoder_labels")

        task_types = batch.task_types

        for batch_index in range(len(task_types)):
            task_type = task_types[batch_index]
            # Check if this is a causal LM task
            from multimodal.tasks.registry import get_task

            definition = get_task(task_type)
            if definition is None:
                continue
            if definition.evaluation_method != "causal_language_modeling":
                continue

            # Get logits and labels for this row
            row_logits = logits[batch_index]  # [seq_len, vocab]
            row_labels = labels[batch_index]  # [seq_len]

            # Shift for causal LM: logits[:-1] predicts labels[1:]
            pred_logits = row_logits[:-1]  # [seq_len-1, vocab]
            target_labels = row_labels[1:]  # [seq_len-1]

            # Mask out ignore labels
            valid_mask = target_labels.ne(-100)  # IGNORE_LABEL
            if not valid_mask.any():
                continue

            # Compute next-token accuracy
            pred_tokens = pred_logits.argmax(dim=-1)
            correct = (
                (pred_tokens[valid_mask] == target_labels[valid_mask])
                .sum()
                .item()
            )
            total = valid_mask.sum().item()

            # Compute cross-entropy loss for perplexity
            ce_loss = F.cross_entropy(
                pred_logits[valid_mask],
                target_labels[valid_mask],
                reduction="mean",
            ).item()

            # Accumulate statistics
            state.causal_lm_state.samples += 1
            state.causal_lm_state.token_correct += correct
            state.causal_lm_state.token_total += total
            state.causal_lm_state.ce_loss_sum += ce_loss * total
            state.causal_lm_state.ce_loss_count += total

    def synchronize(
        self,
        *,
        state: Any,
        device: Any,
    ) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        from evaluator.distributed.reductions import reduce_causal_lm_state

        (
            state.causal_lm_state.samples,
            state.causal_lm_state.token_correct,
            state.causal_lm_state.token_total,
            state.causal_lm_state.ce_loss_sum,
            state.causal_lm_state.ce_loss_count,
        ) = reduce_causal_lm_state(
            samples=state.causal_lm_state.samples,
            token_correct=state.causal_lm_state.token_correct,
            token_total=state.causal_lm_state.token_total,
            ce_loss_sum=state.causal_lm_state.ce_loss_sum,
            ce_loss_count=state.causal_lm_state.ce_loss_count,
            device=device,
        )

    def finalize(
        self,
        *,
        state: Any,
    ) -> dict[str, dict[str, float]]:
        import math

        task_metrics: dict[str, dict[str, float]] = {}
        if state.causal_lm_state.token_total > 0:
            token_accuracy = (
                state.causal_lm_state.token_correct
                / state.causal_lm_state.token_total
            )
            ce_loss = (
                state.causal_lm_state.ce_loss_sum
                / state.causal_lm_state.token_total
            )
            perplexity = math.exp(
                min(ce_loss, 20.0)
            )  # clamp to avoid overflow

            # Find the causal LM task type
            from multimodal.tasks.registry import TASKS

            for task_type, definition in TASKS.items():
                if definition.evaluation_method == "causal_language_modeling":
                    task_metrics[task_type] = {
                        "next_token_accuracy": token_accuracy,
                        "perplexity": perplexity,
                        "causal_lm_loss": ce_loss,
                    }
                    break
        return task_metrics
