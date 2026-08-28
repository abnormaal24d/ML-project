"""Loss components: contrastive, hard negatives, classification, preference.

All callers pass keyword arguments only. Every square-matrix consumer
validates the matrix and the optional per-row ``pair_weights`` explicitly;
a zero total weight yields ``None`` instead of silently weighted material.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_square_similarity_matrix(
    logits: torch.Tensor, /, name: str
) -> int:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(
            f"{name} requires a square 2D similarity matrix, got "
            f"{tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError(f"{name} requires finite logits")
    return logits.shape[0]


def _pair_weight_vector(
    pair_weights: torch.Tensor | None, row_count: int, /, name: str
) -> tuple[torch.Tensor | None, float]:
    if pair_weights is None:
        return None, 0.0
    if pair_weights.shape != (row_count,):
        raise ValueError(
            f"{name} requires pair_weights of shape ({row_count},), got "
            f"{tuple(pair_weights.shape)}"
        )
    if not torch.isfinite(pair_weights).all():
        raise ValueError(f"{name} requires finite pair weights")
    total = float(pair_weights.sum().item())
    return pair_weights, total


def symmetric_info_nce(
    *,
    logits: torch.Tensor,
    temperature: float = 0.07,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Weighted symmetric InfoNCE over a square similarity matrix.

    Each pair contributes ``(forward + reverse) / 2``; the final loss is
    the weighted mean divided by the total weight. Returns ``None`` for a
    batch with fewer than two rows or with total weight ``<= 0``.
    """
    row_count = _validate_square_similarity_matrix(
        logits, "symmetric_info_nce"
    )
    if temperature <= 0.0:
        raise ValueError("symmetric_info_nce requires temperature > 0")
    if row_count < 2:
        return None
    weights, total_weight = _pair_weight_vector(
        pair_weights, row_count, "symmetric_info_nce"
    )
    if weights is not None and total_weight <= 0.0:
        return None
    scaled = logits / temperature
    labels = torch.arange(row_count, device=logits.device)
    forward = F.cross_entropy(scaled, labels, reduction="none")
    reverse = F.cross_entropy(scaled.t(), labels, reduction="none")
    pair_losses: torch.Tensor = (forward + reverse) / 2
    if weights is None:
        loss = pair_losses.mean()
    else:
        loss = (pair_losses * weights.to(logits.device)).sum() / total_weight
    return loss


def hard_negative_margin_loss(
    *,
    logits: torch.Tensor,
    margin: float = 0.2,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Margin loss where the diagonal is positive and the row maximum is the
    hardest in-batch negative.

    Returns ``None`` when total pair weight is ``<= 0``.
    """
    row_count = _validate_square_similarity_matrix(
        logits, "hard_negative_margin_loss"
    )
    if row_count < 2:
        return None
    weights, total_weight = _pair_weight_vector(
        pair_weights, row_count, "hard_negative_margin_loss"
    )
    if weights is not None and total_weight <= 0.0:
        return None
    positive = logits.diagonal()
    negative_mask = torch.eye(
        row_count, device=logits.device, dtype=torch.bool
    )
    hardest_negative = (
        logits.masked_fill(negative_mask, -torch.inf).max(dim=1).values
    )
    margins = torch.clamp(margin + hardest_negative - positive, min=0.0)
    if weights is None:
        loss = margins.mean()
    else:
        loss = (margins * weights.to(logits.device)).sum() / total_weight
    return loss


def classification_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Plain cross entropy for the classifier head (weight applied upstream)."""
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"classification_loss expects 2D logits and 1D labels, got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"classification_loss requires aligned logits/labels batch, "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    labels = labels.to(device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def preference_objective_loss(
    *,
    chosen_log_probabilities: torch.Tensor,
    rejected_log_probabilities: torch.Tensor,
    beta: float,
    objective: str = "pairwise",
) -> torch.Tensor:
    """Preference objective over per-row causal response log probabilities.

    ``pairwise`` (and reference-free ``dpo``) both reduce to the same
    expression because no reference model log probabilities exist.
    """
    if chosen_log_probabilities.shape != rejected_log_probabilities.shape:
        raise ValueError("chosen and rejected preference batches must align")
    if objective not in {"pairwise", "dpo"}:
        raise ValueError(f"unsupported preference objective: {objective!r}")
    margin = chosen_log_probabilities - rejected_log_probabilities
    return -F.logsigmoid(float(beta) * margin).mean()
