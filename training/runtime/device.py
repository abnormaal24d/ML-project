"""Device resolution and distributed training runtime helpers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist

from config.environment.runtime_environment import (
    local_rank as environment_local_rank,
)
from config.environment.runtime_environment import (
    rank as environment_rank,
)
from config.environment.runtime_environment import (
    world_size as environment_world_size,
)
from multimodal.model.model import MultimodalModel

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings


def resolve_device(device: str) -> torch.device:
    """Resolve auto, cuda, or explicit device strings to a torch device."""
    requested = str(device).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was explicitly requested but is unavailable: {device!r}"
        )
    return resolved


def maybe_init_distributed(
    *,
    settings: TrainingSettings,
    device: torch.device,
) -> dict[str, object]:
    """Initialize torch.distributed when a multi-process strategy is configured."""
    strategy = settings.distributed_strategy
    world_size = environment_world_size()
    rank = environment_rank()
    local_rank = environment_local_rank()

    if world_size < 1:
        raise ValueError("WORLD_SIZE must be at least 1")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be within [0, WORLD_SIZE)")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    if strategy in {"ddp", "fsdp"} and world_size <= 1:
        raise RuntimeError(
            f"distributed_strategy={strategy!r} requires WORLD_SIZE > 1 and "
            "a distributed launcher"
        )
    if strategy == "none":
        if world_size > 1:
            raise RuntimeError(
                "WORLD_SIZE > 1 conflicts with distributed_strategy='none'"
            )
        return {
            "enabled": False,
            "strategy": "none",
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }

    if strategy == "auto" and world_size <= 1:
        return {
            "enabled": False,
            "strategy": "none",
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }

    resolved_strategy = "ddp" if strategy == "auto" else strategy
    backend = "nccl" if device.type == "cuda" else "gloo"
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    return {
        "enabled": True,
        "strategy": resolved_strategy,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def destroy_distributed(distributed_context: dict[str, object]) -> None:
    """Release the process group created for this training invocation."""

    if not distributed_context.get("enabled"):
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_rank(distributed_context: dict[str, object]) -> int:
    """Return a strictly validated rank from a distributed context."""

    return _coerce_int(distributed_context.get("rank"))


def wrap_distributed_model(
    *,
    model: MultimodalModel,
    settings: TrainingSettings,
    device: torch.device,
    distributed_context: dict[str, object],
) -> torch.nn.Module:
    """Wrap the model for DDP or FSDP when distributed training is enabled."""
    if not distributed_context.get("enabled"):
        return model

    strategy = str(distributed_context["strategy"])
    if strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        parameter_dtype: torch.dtype | None = None
        if settings.precision == "fp16":
            parameter_dtype = torch.float16
        elif settings.precision == "bf16":
            parameter_dtype = torch.bfloat16
        mixed_precision = (
            MixedPrecision(
                param_dtype=parameter_dtype,
                reduce_dtype=parameter_dtype,
                buffer_dtype=parameter_dtype,
            )
            if parameter_dtype is not None
            else None
        )
        return FullyShardedDataParallel(
            model,
            auto_wrap_policy=partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000,
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=device if device.type == "cuda" else None,
            sync_module_states=device.type == "cuda",
            use_orig_params=True,
            limit_all_gathers=True,
        )
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        if device.type == "cuda":
            local_rank = _coerce_int(
                distributed_context.get("local_rank"),
            )
            return DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        return DistributedDataParallel(model, find_unused_parameters=True)
    raise ValueError(
        f"unsupported distributed_strategy: {settings.distributed_strategy}"
    )


def is_fsdp_model(model: torch.nn.Module) -> bool:
    """Return whether ``model`` is wrapped by FullyShardedDataParallel."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(model, FullyShardedDataParallel)


def unwrap_model(model: torch.nn.Module) -> MultimodalModel:
    """Return the inner MultimodalModel from a distributed wrapper."""
    inner = getattr(model, "module", model)
    return cast(MultimodalModel, inner)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("distributed rank value must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"distributed rank value must be an integer: {value!r}"
            ) from exc
    raise ValueError("distributed rank value must be an integer")
