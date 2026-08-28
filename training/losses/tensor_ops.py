"""Canonical masked tensor loss operations.

Every operator here validates its shapes strictly and never silently
truncates sequences or substitutes token id zero for padding. Loss
surfaces return ``None`` only when no valid positions remain.

All signatures are keyword-only so call sites use explicit, self-verifying
arguments instead of positional shape assumptions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL


def masked_mse(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Mean squared error over the positions selected by a right-aligned mask.

    The mask keeps its batch dimension fixed and broadcasts over the
    remaining trailing dimensions of ``pred``. Returns ``None`` when no
    position is selected; raises instead of silently shrinking inputs.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"masked_mse requires equal pred/target shapes, got "
            f"{tuple(pred.shape)} != {tuple(target.shape)}"
        )
    if mask is not None:
        if mask.ndim > pred.ndim or mask.shape[0] != pred.shape[0]:
            raise ValueError(
                f"masked_mse expects mask with batch dim {pred.shape[0]} "
                f"and rank <= {pred.ndim}, got {tuple(mask.shape)}"
            )
        broadcast_mask = mask.reshape(
            [mask.shape[0]]
            + [1] * (pred.ndim - mask.ndim)
            + list(mask.shape[1:])
        )
        valid = broadcast_mask.expand_as(pred).bool()
    else:
        valid = torch.ones_like(pred, dtype=torch.bool)
    count = valid.sum()
    if count.item() == 0:
        return None
    diff = (pred - target) ** 2
    loss: torch.Tensor = torch.where(valid, diff, torch.zeros_like(diff)).sum()
    return loss / count


def masked_binary_cross_entropy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Binary cross entropy over the positions selected by ``mask``."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"masked_binary_cross_entropy requires equal logits/targets "
            f"shapes, got {tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    valid = (
        torch.ones_like(logits, dtype=torch.bool)
        if mask is None
        else mask.to(device=logits.device, dtype=torch.bool)
    )
    if valid.shape != logits.shape:
        raise ValueError(
            "masked_binary_cross_entropy requires mask shape to match "
            f"logits, got {tuple(valid.shape)} != {tuple(logits.shape)}"
        )
    if not valid.any():
        return None
    values = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return values[valid].mean()


def token_cross_entropy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Cross entropy over aligned ``[B, T, V]`` logits and ``[B, T]`` targets.

    Padding must already be encoded as ``IGNORE_LABEL`` by the collator or
    be removed via ``attention_mask``; token id zero is a real vocabulary
    token and is never substituted implicitly.
    """
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError(
            f"token_cross_entropy expects 3D logits and 2D targets, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    batch, seq_len, _vocab = logits.shape
    if targets.shape != (batch, seq_len):
        raise ValueError(
            f"token_cross_entropy requires aligned logits/targets, got "
            f"logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    targets = targets.to(device=logits.device, dtype=torch.long)
    if attention_mask is not None:
        if attention_mask.shape != (batch, seq_len):
            raise ValueError(
                f"token_cross_entropy requires attention_mask {tuple((batch, seq_len))}, "
                f"got {tuple(attention_mask.shape)}"
            )
        targets = targets.masked_fill(
            ~attention_mask.to(logits.device).bool(), IGNORE_LABEL
        )
    if not targets.ne(IGNORE_LABEL).any():
        return None
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_LABEL,
    )


def causal_language_modeling_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Shifted next-token-loss over full-rows selected by ``row_mask``.

    Logits and labels must be length-aligned before the causal shift
    ``logits[:, :-1]`` / ``labels[:, 1:]``.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"causal_language_modeling_loss expects 3D logits and 2D "
            f"labels, got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    batch, seq_len, _vocab = logits.shape
    if labels.shape != (batch, seq_len):
        raise ValueError(
            f"causal_language_modeling_loss requires aligned logits/labels, "
            f"got logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    if seq_len < 2:
        return None
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].to(logits.device).long()
    if row_mask is not None:
        if row_mask.shape[0] != batch:
            raise ValueError(
                f"causal_language_modeling_loss requires row_mask batch dim "
                f"{batch}, got {row_mask.shape[0]}"
            )
        if not row_mask.any():
            return None
        shifted_logits = shifted_logits[row_mask]
        shifted_labels = shifted_labels[row_mask]
    if not shifted_labels.ne(IGNORE_LABEL).any():
        return None
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_LABEL,
    )


def sequence_log_probability(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    """Mean causal response log probability per row (no mask substitutions)."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"sequence_log_probability expects 3D logits and 2D labels, "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    batch, seq_len, _vocab = logits.shape
    if labels.shape != (batch, seq_len):
        raise ValueError(
            f"sequence_log_probability requires aligned logits/labels, got "
            f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    if seq_len < 2:
        return None
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].to(logits.device).long()
    valid = shifted_labels.ne(IGNORE_LABEL)
    if not valid.any(dim=1).all():
        raise ValueError(
            "every preference response must contain a scored token"
        )
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_log_probs = (
        F.log_softmax(shifted_logits, dim=-1)
        .gather(-1, safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    token_log_probs = torch.where(
        valid, token_log_probs, torch.zeros_like(token_log_probs)
    )
    return token_log_probs.sum(dim=1) / valid.sum(dim=1)


def multi_codebook_cross_entropy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Cross entropy for canonical one-codebook audio tokens.

    ``logits: [B, 1, T, V]``, ``targets: [B, 1, T]``,
    ``attention_mask: [B, T]``.
    """
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError(
            f"multi_codebook_cross_entropy expects 4D logits [B, K, T, V] "
            f"and 3D targets [B, K, T], got {tuple(logits.shape)} and "
            f"{tuple(targets.shape)}"
        )
    batch, codebooks, frames, vocab = logits.shape
    if codebooks != 1:
        raise ValueError(
            "audio generation supports exactly one codebook axis; "
            f"got logits with shape {tuple(logits.shape)}"
        )
    if targets.shape != (batch, codebooks, frames):
        raise ValueError(
            f"multi_codebook_cross_entropy requires aligned logits/targets, "
            f"got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    targets = targets.to(logits.device)
    if attention_mask is not None:
        if attention_mask.shape != (batch, frames):
            raise ValueError(
                f"multi_codebook_cross_entropy attention_mask must match "
                f"[B, T] of logits, got {tuple(attention_mask.shape)}"
            )
        mask = (
            attention_mask.to(logits.device)
            .bool()
            .unsqueeze(1)
            .expand(-1, codebooks, -1)
        )
        targets = targets.masked_fill(~mask, IGNORE_LABEL)
    if not targets.ne(IGNORE_LABEL).any():
        return None
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=IGNORE_LABEL,
    )


def video_token_cross_entropy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Cross entropy for video grid tokens.

    ``logits: [B, T, Hq, Wq, V]``, ``targets: [B, T, Hq, Wq]``,
    ``attention_mask: [B, T]``.
    """
    if logits.ndim != 5 or targets.ndim != 4:
        raise ValueError(
            f"video_token_cross_entropy expects 5D logits [B, T, Hq, Wq, V] "
            f"and 4D targets [B, T, Hq, Wq], got {tuple(logits.shape)} and "
            f"{tuple(targets.shape)}"
        )
    batch, frames, height, width, vocab = logits.shape
    if targets.shape != (batch, frames, height, width):
        raise ValueError(
            f"video_token_cross_entropy requires aligned logits/targets, "
            f"got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    targets = targets.to(logits.device)
    if attention_mask is not None:
        if attention_mask.shape != (batch, frames):
            raise ValueError(
                f"video_token_cross_entropy attention_mask must match "
                f"[B, frames] of logits, got {tuple(attention_mask.shape)}"
            )
        mask = (
            attention_mask.to(logits.device)
            .bool()[:, :, None, None]
            .expand(-1, -1, height, width)
        )
        targets = targets.masked_fill(~mask, IGNORE_LABEL)
    if not targets.ne(IGNORE_LABEL).any():
        return None
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=IGNORE_LABEL,
    )
