"""Optimizer and learning-rate scheduler builders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings


def build_optimizer(
    model: torch.nn.Module,
    settings: TrainingSettings,
) -> torch.optim.Optimizer:
    """Build the AdamW optimizer for all trainable model parameters."""

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        if any(True for _parameter in model.parameters()):
            raise ValueError(
                "cannot build optimizer: model has no trainable parameters"
            )
        raise ValueError(
            "cannot build optimizer for a model without parameters"
        )

    has_cuda_parameters = any(
        parameter.is_cuda for parameter in trainable_parameters
    )

    if has_cuda_parameters:
        try:
            return torch.optim.AdamW(
                trainable_parameters,
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                trainable_parameters,
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
            )

    try:
        return torch.optim.AdamW(
            trainable_parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            foreach=True,
        )
    except TypeError:
        return torch.optim.AdamW(
            trainable_parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )


def build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
    num_training_batches: int,
    completed_optimizer_steps: int,
) -> LRScheduler | None:
    """Build the configured scheduler in its explicitly configured interval."""

    if settings.lr_scheduler == "none":
        return None

    if settings.lr_scheduler == "cosine":
        interval = settings.scheduler_interval
        if interval == "step":
            steps_per_epoch = optimizer_steps_per_epoch(
                num_training_batches=num_training_batches,
                gradient_accumulation_steps=(
                    settings.gradient_accumulation_steps
                ),
            )
            t_max = steps_per_epoch * int(settings.epochs)
        elif interval == "epoch":
            t_max = int(settings.epochs)
        else:
            raise ValueError(f"unsupported scheduler_interval: {interval!r}")

        if completed_optimizer_steps < 0:
            raise ValueError("completed_optimizer_steps must be non-negative")
        last_epoch_for_scheduler = completed_optimizer_steps - 1
        if last_epoch_for_scheduler >= 0:
            # Ensure each param group has a stable base learning rate.
            for parameter_group in optimizer.param_groups:
                parameter_group.setdefault(
                    "initial_lr",
                    float(
                        parameter_group.get(
                            "lr", getattr(settings, "learning_rate", 0.0)
                        )
                    ),
                )

            # Compute and set the effective learning rate corresponding to the
            # recovered step (completed_optimizer_steps). Use the completed
            # optimizer step count (not the scheduler's internal last_epoch)
            # so the recovered LR matches a continuous run at the same point.
            base_tmax = max(1, t_max)
            eta_min = float(settings.min_learning_rate)
            t = completed_optimizer_steps
            # Cosine annealing formula (PyTorch uses last_epoch indexing);
            # lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * t / T_max))
            for parameter_group in optimizer.param_groups:
                base_lr = float(
                    parameter_group.get(
                        "initial_lr", parameter_group.get("lr", 0.0)
                    )
                )
                if base_lr <= eta_min:
                    effective_lr = base_lr
                else:
                    effective_lr = eta_min + (base_lr - eta_min) * 0.5 * (
                        1.0 + math.cos(math.pi * float(t) / float(base_tmax))
                    )
                parameter_group["lr"] = float(effective_lr)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=float(settings.min_learning_rate),
            last_epoch=last_epoch_for_scheduler,
        )

        return scheduler

    raise ValueError(f"unsupported lr_scheduler: {settings.lr_scheduler!r}")


def optimizer_steps_per_epoch(
    *,
    num_training_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return optimizer updates, including a final partial accumulation."""

    if num_training_batches <= 0:
        raise ValueError("num_training_batches must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return math.ceil(num_training_batches / gradient_accumulation_steps)
