"""Contrastive and hard-negative loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.components import (
    _validate_square_similarity_matrix,
    hard_negative_margin_loss,
    symmetric_info_nce,
)
from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import compute_contrastive_pair_coverage


def collect_contrastive(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect contrastive and hard_negative loss terms.

    Both terms share the same pair context (row indices, alignment scores,
    pair weights, and pair coverage). We compute the shared context once
    and then produce both terms if their weights are positive.
    """
    logits = model_output.get("contrastive_logits")
    hard_logits = model_output.get("hard_negative_logits")
    if logits is None and hard_logits is None:
        return ()

    pair_weights: torch.Tensor | None = None
    pair_coverage: torch.Tensor | None = None
    batch_size = 0

    if logits is not None:
        task_types = getattr(batch, "task_types", None)
        batch_size = len(task_types) if isinstance(task_types, list) else 0
        row_count = int(logits.shape[0])
        row_indices = model_output.get("contrastive_row_indices")
        if row_indices is not None:
            if row_indices.ndim != 1 or row_indices.numel() != row_count:
                raise ValueError(
                    f"contrastive_row_indices must be 1D with "
                    f"{row_count} entries, got shape "
                    f"{tuple(row_indices.shape)}"
                )
            rows = row_indices.long()
            if rows.numel() > 0:
                if int(rows.min()) < 0 or int(rows.max()) >= batch_size:
                    raise ValueError(
                        "contrastive_row_indices must index batch rows "
                        f"in [0, {batch_size})"
                    )
                if len(set(rows.tolist())) != rows.numel():
                    raise ValueError(
                        "contrastive_row_indices must not contain duplicates"
                    )
        else:
            if batch_size == 0:
                batch_size = row_count
            if batch_size != row_count:
                raise RuntimeError(
                    "contrastive logits rows must match batch rows; use "
                    "contrastive_row_indices when pairs form a subset"
                )
            rows = torch.arange(row_count, dtype=torch.long)
        alignment_scores = getattr(batch, "alignment_scores", None)
        if alignment_scores is None:
            raise RuntimeError(
                "contrastive loss requires batch.alignment_scores"
            )
        if (
            alignment_scores.ndim != 1
            or alignment_scores.numel() != batch_size
        ):
            raise ValueError(
                f"contrastive loss requires alignment_scores of shape "
                f"({batch_size},), got {tuple(alignment_scores.shape)}"
            )
        if not torch.isfinite(alignment_scores).all():
            raise ValueError(
                "contrastive loss requires finite alignment scores"
            )
        if bool((alignment_scores < 0.0).any()) or bool(
            (alignment_scores > 1.0).any()
        ):
            raise ValueError(
                "contrastive loss requires alignment scores in [0, 1]"
            )
        scores = alignment_scores.to(logits.device)[rows]
        pair_weights = torch.where(
            scores > 0.0,
            scores.pow(context.alignment_score_exponent),
            torch.zeros_like(scores),
        )
        pair_coverage = compute_contrastive_pair_coverage(
            rows, scores, batch_size
        )

    terms: list[LossTerm] = []

    if context.weights["contrastive"] > 0.0 and logits is not None:
        term = symmetric_info_nce(
            logits=logits,
            temperature=context.contrastive_temperature,
            pair_weights=pair_weights,
        )
        if term is not None:
            terms.append(
                LossTerm(
                    name="contrastive",
                    value=term,
                    coverage=pair_coverage
                    if pair_coverage is not None
                    else torch.ones(batch_size, dtype=torch.bool),
                )
            )

    if context.weights["hard_negative"] > 0.0 and hard_logits is not None:
        row_count = _validate_square_similarity_matrix(
            hard_logits, "hard_negative_margin_loss"
        )
        if row_count >= 2:
            term = hard_negative_margin_loss(
                logits=hard_logits,
                margin=context.hard_negative_margin,
                pair_weights=pair_weights,
            )
            if term is not None:
                terms.append(
                    LossTerm(
                        name="hard_negative",
                        value=term,
                        coverage=pair_coverage.clone()
                        if pair_coverage is not None
                        else torch.ones(row_count, dtype=torch.bool),
                    )
                )

    return tuple(terms)
