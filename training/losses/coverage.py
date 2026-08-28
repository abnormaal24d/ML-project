"""Row coverage helpers for loss collectors."""

from __future__ import annotations

import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL


def all_rows_coverage(batch_size: int) -> torch.Tensor:
    """Coverage tensor where all rows are covered."""
    return torch.ones(batch_size, dtype=torch.bool)


def row_coverage_from_mask(
    mask: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    """Compute row coverage from a boolean mask.

    Args:
        mask: Boolean tensor of shape [batch_size, ...] or broadcastable.
        batch_size: Number of rows in the batch.

    Returns:
        Boolean tensor of shape [batch_size] where True indicates the row
        has at least one valid position.
    """
    if mask.ndim == 1:
        return mask.to(dtype=torch.bool)
    return mask.reshape(batch_size, -1).any(dim=1)


def row_coverage_from_ignore_labels(
    labels: torch.Tensor, *, ignore_label: int = IGNORE_LABEL
) -> torch.Tensor:
    """Compute row coverage from labels using IGNORE_LABEL as padding marker.

    Args:
        labels: Label tensor of shape [batch_size, seq_len] or [batch_size].
        ignore_label: Label value indicating padding.

    Returns:
        Boolean tensor of shape [batch_size] where True indicates the row
        has at least one non-ignored label.
    """
    if labels.ndim == 1:
        return labels.ne(ignore_label)
    return labels.ne(ignore_label).any(dim=1)


def row_coverage_from_masked_targets(
    targets: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Compute row coverage from targets combined with optional attention mask.

    Args:
        targets: Target tensor of shape [batch_size, ...].
        mask: Optional attention mask of shape [batch_size, ...].

    Returns:
        Boolean tensor of shape [batch_size] where True indicates the row
        has at least one valid target position.
    """
    valid = targets.ne(IGNORE_LABEL)
    if mask is not None:
        mask_bool = mask.to(device=targets.device, dtype=torch.bool)
        if mask_bool.ndim < valid.ndim:
            mask_bool = mask_bool.unsqueeze(-1).expand_as(valid)
        valid &= mask_bool
    return valid.flatten(1).any(dim=1)


def row_coverage_from_sequence_mask(
    attention_mask: torch.Tensor | None, *, batch_size: int
) -> torch.Tensor:
    """Compute row coverage from sequence attention mask.

    Args:
        attention_mask: Boolean mask of shape [batch_size, seq_len] or None.
        batch_size: Number of rows in the batch.

    Returns:
        Boolean tensor of shape [batch_size].
    """
    if attention_mask is None:
        return all_rows_coverage(batch_size)
    return attention_mask.to(dtype=torch.bool).any(dim=1)


def compute_contrastive_pair_coverage(
    row_indices: torch.Tensor,
    alignment_scores: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Compute per-row coverage for contrastive pairs.

    Args:
        row_indices: 1D tensor of batch row indices for each pair.
        alignment_scores: Alignment scores for each pair (1D, same length as row_indices).
        batch_size: Total number of rows in the batch.

    Returns:
        Boolean tensor of shape [batch_size] where True indicates the row
        has at least one contrastive pair with positive alignment score.
    """
    covered_rows = alignment_scores > 0.0
    pair_coverage = torch.zeros(batch_size, dtype=torch.bool)
    pair_coverage[row_indices.cpu()] = covered_rows.cpu()
    return pair_coverage
