"""Distributed reductions for scalar and sequence state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from evaluator.metric_contracts import SEQUENCE_STAT_FIELDS


def _collective_device(
    *,
    configured_device: Any,
    local_text: list[torch.Tensor],
    local_media: list[torch.Tensor],
) -> torch.device:
    """Return a backend-compatible device for distributed collectives."""
    if local_text:
        return local_text[0].device
    if local_media:
        return local_media[0].device
    if configured_device is not None:
        return torch.device(configured_device)
    if dist.is_available() and dist.is_initialized():
        if str(dist.get_backend()).lower() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def reduce_scalar_pair(
    *,
    value_a: float,
    value_b: float,
    device: Any,
) -> tuple[float, float]:
    """All-reduce two scalar values across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return value_a, value_b
    reduction_device = _collective_device(
        configured_device=device,
        local_text=[],
        local_media=[],
    )
    tensor = torch.tensor(
        [float(value_a), float(value_b)],
        dtype=torch.float64,
        device=reduction_device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), tensor[1].item()


def reduce_mlm_state(
    *,
    correct: int,
    total: int,
    device: Any,
) -> tuple[int, int]:
    """All-reduce MLM state across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return correct, total
    reduction_device = _collective_device(
        configured_device=device,
        local_text=[],
        local_media=[],
    )
    tensor = torch.tensor(
        [float(correct), float(total)],
        dtype=torch.float64,
        device=reduction_device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor[0].item()), int(tensor[1].item())


def reduce_causal_lm_state(
    *,
    samples: int,
    token_correct: int,
    token_total: int,
    ce_loss_sum: float,
    ce_loss_count: int,
    device: Any,
) -> tuple[int, int, int, float, int]:
    """All-reduce causal LM state across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return samples, token_correct, token_total, ce_loss_sum, ce_loss_count
    reduction_device = _collective_device(
        configured_device=device,
        local_text=[],
        local_media=[],
    )
    tensor = torch.tensor(
        [
            float(samples),
            float(token_correct),
            float(token_total),
            float(ce_loss_sum),
            float(ce_loss_count),
        ],
        dtype=torch.float64,
        device=reduction_device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (
        int(tensor[0].item()),
        int(tensor[1].item()),
        int(tensor[2].item()),
        tensor[3].item(),
        int(tensor[4].item()),
    )


def reduce_sequence_statistics(
    *,
    statistics: Mapping[str, Any],
    device: Any,
    evaluation_method: str,
) -> None:
    """All-reduce sequence statistics across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    reduction_device = _collective_device(
        configured_device=device,
        local_text=[],
        local_media=[],
    )
    from multimodal.tasks.registry import get_task

    for task_type, stats in statistics.items():
        definition = get_task(task_type)
        if (
            definition is None
            or definition.evaluation_method != evaluation_method
        ):
            continue
        if not hasattr(stats, "samples"):
            continue
        totals = torch.tensor(
            [float(getattr(stats, name)) for name in SEQUENCE_STAT_FIELDS],
            dtype=torch.float64,
            device=reduction_device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if float(totals[0].item()) > 0.0:
            for name, total in zip(SEQUENCE_STAT_FIELDS, totals, strict=True):
                setattr(stats, name, float(total.item()))


__all__ = [
    "reduce_scalar_pair",
    "reduce_mlm_state",
    "reduce_causal_lm_state",
    "reduce_sequence_statistics",
]
