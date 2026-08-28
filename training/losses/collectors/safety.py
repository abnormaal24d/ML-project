"""Safety loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.contracts import LossContext, LossTerm
from training.losses.tensor_ops import masked_binary_cross_entropy


def collect_safety_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect safety classification loss term."""
    if context.weights["safety"] <= 0.0:
        return ()

    safety_logits = model_output.get("safety_logits")
    safety_targets = getattr(batch, "safety_targets", None)
    if safety_logits is None or safety_targets is None:
        return ()

    safety_mask = getattr(batch, "safety_target_mask", None)
    term = masked_binary_cross_entropy(
        logits=safety_logits,
        targets=safety_targets,
        mask=safety_mask,
    )
    if term is None:
        return ()

    if safety_mask is not None:
        coverage = safety_mask.to(
            device=torch.device("cpu"), dtype=torch.bool
        ).any(dim=1)
    else:
        coverage = (
            safety_targets.to(device=safety_logits.device)
            .ne(0.0)
            .any(dim=1)
            .cpu()
        )

    return (
        LossTerm(
            name="safety",
            value=term,
            coverage=coverage,
        ),
    )
