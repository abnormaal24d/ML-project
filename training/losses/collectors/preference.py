"""Preference tuning loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.components import preference_objective_loss
from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import all_rows_coverage
from training.losses.tensor_ops import sequence_log_probability


def collect_preference_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect preference loss term (only during PREFERENCE_TUNING stage)."""
    if context.training_stage != "PREFERENCE_TUNING":
        return ()
    if context.weights["preference"] <= 0.0:
        return ()

    chosen_logits = model_output.get("chosen_sequence_logits")
    chosen_labels = getattr(batch, "chosen_labels", None)
    rejected_logits = model_output.get("rejected_sequence_logits")
    rejected_labels = getattr(batch, "rejected_labels", None)
    if (
        chosen_logits is None
        or chosen_labels is None
        or rejected_logits is None
        or rejected_labels is None
    ):
        return ()

    chosen = sequence_log_probability(
        logits=chosen_logits,
        labels=chosen_labels,
    )
    rejected = sequence_log_probability(
        logits=rejected_logits,
        labels=rejected_labels,
    )
    if chosen is None or rejected is None:
        return ()

    loss = preference_objective_loss(
        chosen_log_probabilities=chosen,
        rejected_log_probabilities=rejected,
        beta=context.preference_beta,
        objective=context.preference_mode,
    )
    batch_size = (
        len(task_types)
        if isinstance(task_types := getattr(batch, "task_types", None), list)
        else 0
    )
    coverage = all_rows_coverage(batch_size)

    return (
        LossTerm(
            name="preference",
            value=loss,
            coverage=coverage,
        ),
    )
