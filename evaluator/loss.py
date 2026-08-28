"""Loss evaluation over required training data loaders."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.distributed as dist

from mmcrawler_datasets.collation.multimodal import move_batch_to_device
from multimodal.model.contracts import CollatedBatch


def _move_evaluation_batch(
    *,
    batch: Any,
    device: Any,
) -> Any:
    if device is None:
        return batch

    if isinstance(batch, CollatedBatch):
        return move_batch_to_device(batch=batch, device=device)

    if isinstance(batch, dict):
        return {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in batch.items()
        }

    if hasattr(batch, "to"):
        return batch.to(device)

    return batch


def _evaluate_model(
    *,
    model: Any,
    batch: Any,
) -> Any:
    if isinstance(batch, dict):
        return model(**batch)
    return model(batch)


def _resolve_evaluation_loss(
    *,
    loss_fn: Any,
    outputs: Any,
    batch: Any,
    batch_index: int,
) -> torch.Tensor:
    result = loss_fn(
        model_output=outputs,
        batch=batch,
        require_targets_for_generation=True,
    )

    if torch.is_tensor(result):
        loss = result
    elif isinstance(result, dict):
        total = result.get("total")
        if torch.is_tensor(total):
            loss = total
        else:
            values = [
                value
                for key, value in result.items()
                if key != "total" and torch.is_tensor(value)
            ]
            if not values:
                raise ValueError(
                    f"evaluation batch {batch_index} produced no tensor loss"
                )
            loss = torch.stack(values).sum()
    else:
        raise ValueError(
            f"evaluation batch {batch_index} produced no tensor loss"
        )

    if not torch.isfinite(loss).all():
        raise ValueError(
            f"evaluation batch {batch_index} produced non-finite loss"
        )

    return loss


def _evaluation_batch_size(batch: Any) -> int:
    if isinstance(batch, CollatedBatch):
        size = len(batch.sample_ids)
    elif isinstance(batch, dict):
        sizes = {
            int(value.shape[0])
            for value in batch.values()
            if torch.is_tensor(value) and value.ndim > 0
        }
        if len(sizes) > 1:
            raise ValueError(
                "evaluation dictionary tensors disagree on batch size"
            )
        size = sizes.pop() if sizes else 1
    else:
        sample_ids = getattr(batch, "sample_ids", None)
        size = len(sample_ids) if isinstance(sample_ids, list) else 1
    if size <= 0:
        raise ValueError("evaluation batch must contain at least one sample")
    return size


def evaluate_loader_loss(
    *,
    model: Any,
    loss_fn: Any,
    loader: Any,
    device: Any = None,
    autocast_factory: Callable[[], AbstractContextManager[object]],
) -> float:
    """Compute average loss over a loader without gradients."""
    if loader is None:
        raise RuntimeError("evaluation loader is required")

    was_training = bool(model.training)
    model.eval()
    weighted_loss_sum = 0.0
    sample_count = 0

    try:
        with torch.no_grad(), autocast_factory():
            for batch_index, batch in enumerate(loader):
                batch = _move_evaluation_batch(
                    batch=batch,
                    device=device,
                )
                outputs = _evaluate_model(
                    model=model,
                    batch=batch,
                )
                loss = _resolve_evaluation_loss(
                    loss_fn=loss_fn,
                    outputs=outputs,
                    batch=batch,
                    batch_index=batch_index,
                )
                batch_size = _evaluation_batch_size(batch)
                weighted_loss_sum += float(loss.detach().cpu()) * batch_size
                sample_count += batch_size
    finally:
        model.train(was_training)

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [weighted_loss_sum, float(sample_count)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        weighted_loss_sum = float(totals[0].item())
        sample_count = int(totals[1].item())
    if sample_count == 0:
        raise ValueError("evaluation loader produced no evaluable batches")
    return weighted_loss_sum / sample_count


def evaluate_final_losses(
    *,
    model: Any,
    loss_fn: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    device: Any,
    autocast_factory: Callable[[], AbstractContextManager[object]],
) -> tuple[float, float, float]:
    """Evaluate train, validation, and test losses independently."""

    return (
        evaluate_loader_loss(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            device=device,
            autocast_factory=autocast_factory,
        ),
        evaluate_loader_loss(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            autocast_factory=autocast_factory,
        ),
        evaluate_loader_loss(
            model=model,
            loss_fn=loss_fn,
            loader=test_loader,
            device=device,
            autocast_factory=autocast_factory,
        ),
    )


__all__ = ["evaluate_final_losses", "evaluate_loader_loss"]
