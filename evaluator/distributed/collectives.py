"""Distributed collectives for pair embedding gather."""

from __future__ import annotations

from typing import Any

import torch

from evaluator.metric_contracts import PAIR_TASK_ORDER


def _collective_device(
    *,
    configured_device: Any,
    local_text: list[torch.Tensor],
    local_media: list[torch.Tensor],
    dist_module: Any = None,
) -> torch.device:
    """Return a backend-compatible device for distributed collectives."""
    if local_text:
        return local_text[0].device
    if local_media:
        return local_media[0].device
    if configured_device is not None:
        return torch.device(configured_device)
    if (
        dist_module is not None
        and hasattr(dist_module, "is_available")
        and hasattr(dist_module, "is_initialized")
    ):
        if dist_module.is_available() and dist_module.is_initialized():
            if hasattr(dist_module, "get_backend"):
                if str(dist_module.get_backend()).lower() == "nccl":
                    return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def gather_pair_embeddings(
    *,
    text_embeddings_by_task: dict[str, list[torch.Tensor]],
    media_embeddings_by_task: dict[str, list[torch.Tensor]],
    device: Any,
    pair_task_order: tuple[str, ...] = PAIR_TASK_ORDER,
    dist_module: Any = None,
) -> tuple[
    dict[str, list[torch.Tensor]],
    dict[str, list[torch.Tensor]],
]:
    """Gather variable-length pair embeddings in a fixed DDP order."""
    if dist_module is None:
        import torch.distributed as dist_module

    if not hasattr(dist_module, "get_world_size"):
        return {}, {}
    world_size = dist_module.get_world_size()
    gathered_text: dict[str, list[torch.Tensor]] = {}
    gathered_media: dict[str, list[torch.Tensor]] = {}

    for task_type in pair_task_order:
        local_text = text_embeddings_by_task.get(task_type, [])
        local_media = media_embeddings_by_task.get(task_type, [])
        local_count = len(local_text)
        local_dim = 0
        local_valid = 1
        if len(local_text) != len(local_media):
            local_valid = 0
        elif local_text:
            local_dim = int(local_text[0].numel())
            if any(int(value.numel()) != local_dim for value in local_text):
                local_valid = 0
            elif any(int(value.numel()) != local_dim for value in local_media):
                local_valid = 0

        collective_device = _collective_device(
            configured_device=device,
            local_text=local_text,
            local_media=local_media,
            dist_module=dist_module,
        )
        metadata = torch.tensor(
            [local_count, local_dim, local_valid],
            dtype=torch.long,
            device=collective_device,
        )
        gathered_metadata = [
            torch.zeros_like(metadata) for _ in range(world_size)
        ]
        if hasattr(dist_module, "all_gather"):
            dist_module.all_gather(gathered_metadata, metadata)

        invalid_ranks = [
            index
            for index, value in enumerate(gathered_metadata)
            if int(value[2].item()) == 0
        ]
        if invalid_ranks:
            raise ValueError(
                "invalid local pair embedding metadata on "
                f"rank(s) {invalid_ranks}"
            )
        counts = [int(value[0].item()) for value in gathered_metadata]
        dimensions = {
            int(value[1].item())
            for value in gathered_metadata
            if int(value[0].item()) > 0
        }
        if len(dimensions) > 1:
            raise ValueError(
                f"distributed embedding dimension mismatch for {task_type}: "
                f"{sorted(dimensions)}"
            )
        if not dimensions:
            continue

        embedding_dim = dimensions.pop()
        max_count = max(counts)
        padded_text = torch.zeros(
            (max_count, embedding_dim),
            dtype=torch.float32,
            device=collective_device,
        )
        padded_media = torch.zeros_like(padded_text)
        if local_count:
            local_text_tensor = torch.stack(local_text).to(
                device=collective_device,
                dtype=torch.float32,
            )
            local_media_tensor = torch.stack(local_media).to(
                device=collective_device,
                dtype=torch.float32,
            )
            padded_text[:local_count].copy_(local_text_tensor)
            padded_media[:local_count].copy_(local_media_tensor)

        gathered_text_tensors = [
            torch.zeros_like(padded_text) for _ in range(world_size)
        ]
        gathered_media_tensors = [
            torch.zeros_like(padded_media) for _ in range(world_size)
        ]
        if hasattr(dist_module, "all_gather"):
            dist_module.all_gather(gathered_text_tensors, padded_text)
            dist_module.all_gather(gathered_media_tensors, padded_media)

        gathered_text[task_type] = [
            tensor[:count]
            for tensor, count in zip(
                gathered_text_tensors, counts, strict=True
            )
            if count > 0
        ]
        gathered_media[task_type] = [
            tensor[:count]
            for tensor, count in zip(
                gathered_media_tensors, counts, strict=True
            )
            if count > 0
        ]

    return gathered_text, gathered_media


__all__ = [
    "gather_pair_embeddings",
]
