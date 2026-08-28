"""Classification loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.components import classification_loss
from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import row_coverage_from_ignore_labels


def collect_classification(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect label classification loss term."""
    if context.weights["label"] <= 0.0:
        return ()

    label_logits = model_output.get("label_logits")
    labels = getattr(batch, "labels", None)
    if label_logits is None or labels is None:
        return ()

    ignore_label = -100
    valid_rows = labels.ne(ignore_label)
    if not valid_rows.any():
        return ()

    loss = classification_loss(logits=label_logits, labels=labels)
    coverage = row_coverage_from_ignore_labels(
        labels, ignore_label=ignore_label
    )

    return (
        LossTerm(
            name="label",
            value=loss,
            coverage=coverage,
        ),
    )
