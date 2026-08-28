"""Collate video inputs, temporal labels, and token targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mmcrawler_datasets.collation.tensor_ops import (
    IGNORE_LABEL,
    maybe_reverse_video,
    sample_generator,
    stack_feature_matrix,
    to_long_tensor,
)
from mmcrawler_datasets.tensors import (
    load_optional_tensor_batch,
    load_required_token_tensor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mmcrawler_datasets.schema import MultimodalSample
    from mmcrawler_datasets.tensors import (
        SampleTensorSource,
    )


class VideoCollator:
    """Build video tensors and optional video-token targets from samples."""

    def __init__(
        self,
        *,
        video_frames: int,
        image_size: int,
        tensor_source: SampleTensorSource,
        base_seed: int = 0,
    ) -> None:
        self._tensor_source = tensor_source
        self.raw_video_frames = int(video_frames)
        self.raw_image_size = int(image_size)
        self._base_seed = int(base_seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def collate_sample(
        self,
        sample: MultimodalSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered video frames and a temporal label for one sample."""

        video = self.video_tensor_for_sample(sample=sample)
        if sample.has_video and sample.task_type not in (
            "text_to_video",
            "video_captioning",
        ):
            generator = sample_generator(
                base_seed=self._base_seed,
                epoch=self._epoch,
                sample_id=str(sample.sample_id),
                operation="video_order",
            )
            ordered_video, temporal_label = maybe_reverse_video(
                video,
                generator=generator,
            )
        else:
            ordered_video = video
            temporal_label = to_long_tensor([IGNORE_LABEL])[0]
        return ordered_video, temporal_label

    def collate_batch(
        self,
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack video tensors and temporal labels for a batch."""

        video_inputs: list[torch.Tensor] = []
        video_labels: list[torch.Tensor] = []
        for sample in samples:
            ordered_video, temporal_label = self.collate_sample(sample)
            video_inputs.append(ordered_video)
            video_labels.append(temporal_label)
        return (
            stack_feature_matrix(video_inputs),
            stack_feature_matrix(video_labels),
        )

    def video_tensor_for_sample(
        self,
        *,
        sample: MultimodalSample,
    ) -> torch.Tensor:
        """Load the video tensor for one sample from the configured source."""

        return self._tensor_source.video_tensor(sample=sample)

    def collate_optional_videos(
        self,
        *,
        paths: Sequence[Path | None],
    ) -> torch.Tensor | None:
        """Load optional materialized RGB video tensors."""

        return load_optional_tensor_batch(
            dataset_root=self._tensor_source.dataset_root,
            paths=paths,
            expected_shape=(
                self.raw_video_frames,
                3,
                self.raw_image_size,
                self.raw_image_size,
            ),
            dtype=torch.float32,
        )

    def collate_video_token_targets(
        self,
        paths: Sequence[Path | None],
        *,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load and stack optional per-sample video token targets."""

        if not any(paths):
            return None
        return collate_video_token_targets(
            paths=list(paths),
            frame_count=frame_count,
            dataset_root=self._tensor_source.dataset_root,
        )


def collate_video_token_targets(
    paths: list[Any],
    frame_count: int,
    dataset_root: Path,
    latent_shape: tuple[int, int] | None = None,  # (Hq, Wq) for token grid
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load, pad, and mask video token targets as [B, T, Hq, Wq], mask [B, T]."""

    if not paths:
        return None

    loaded = [
        _load_video_token_tensor(path=path, dataset_root=dataset_root)
        for path in paths
    ]
    present = [tensor for tensor in loaded if tensor is not None]
    if not present:
        return None

    # Expect per-sample [T, Hq, Wq] or [T, H, W] for tokens.
    # For latents we still tolerate 3D tail, but prefer explicit grid schema.
    first = present[0]
    tail: tuple[int, ...]
    if first.dim() == 3:
        tail = (int(first.shape[1]), int(first.shape[2]))
    elif first.dim() == 4:
        # [T, C, H, W] -> treat as latent, collapse or keep last 3
        tail = (int(first.shape[1]), int(first.shape[2]), int(first.shape[3]))
    else:
        return None

    if latent_shape is not None:
        target_tail = latent_shape
    else:
        target_tail = (tail[-2], tail[-1])

    target_time = _target_time(
        tensors=present,
        frame_count=frame_count,
    )
    if target_time <= 0:
        return None

    # For token grid we want [T, Hq, Wq]
    fallback_shape = (target_time, *target_tail)
    dtype = torch.long
    padding_value = -100

    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for tensor in loaded:
        normalized = _normalize_video_token_tensor(
            tensor=tensor,
            target_time=target_time,
            fallback_shape=fallback_shape,
            dtype=dtype,
            padding_value=padding_value,
        )
        # Ensure exactly 3D [T, Hq, Wq]
        if normalized.dim() == 4:
            # collapse channel dim if present (latent case)
            normalized = (
                normalized.mean(dim=1)
                if normalized.shape[1] > 1
                else normalized[:, 0]
            )
        if normalized.dim() != 3:
            normalized = normalized.view(target_time, *target_tail)

        valid_time = min(
            target_time,
            int(tensor.shape[0])
            if tensor is not None and tensor.dim() > 0
            else 0,
        )
        mask = torch.zeros(target_time, dtype=torch.bool)
        if tensor is not None and valid_time > 0:
            mask[:valid_time] = True
        rows.append(normalized)
        masks.append(mask)

    targets = torch.stack(rows)  # [B, T, Hq, Wq]
    mask = torch.stack(masks)  # [B, T]
    return targets, mask


def _load_video_token_tensor(
    *,
    path: Any,
    dataset_root: Path,
) -> torch.Tensor | None:
    if path is None:
        return None
    value = load_required_token_tensor(
        dataset_root=dataset_root,
        path=path,
    )
    if value.dim() == 0:
        value = value.reshape(1)
    return value.detach().cpu().contiguous()


def _normalize_video_token_tensor(
    *,
    tensor: torch.Tensor | None,
    target_time: int,
    fallback_shape: tuple[int, ...],
    dtype: torch.dtype,
    padding_value: float | int,
) -> torch.Tensor:
    if tensor is None:
        return torch.full(fallback_shape, padding_value, dtype=dtype)
    value = tensor.to(dtype=dtype)
    if value.dim() == 0:
        value = value.reshape(1)
    if value.shape[0] > target_time:
        value = value[:target_time]
    if value.shape[0] >= target_time:
        return value.contiguous()

    pad_shape = (target_time - value.shape[0], *value.shape[1:])
    padding = torch.full(pad_shape, padding_value, dtype=dtype)
    return torch.cat([value, padding], dim=0).contiguous()


def _target_time(
    *,
    tensors: list[torch.Tensor],
    frame_count: int,
) -> int:
    observed = max((int(tensor.shape[0]) for tensor in tensors), default=0)
    return max(1, int(frame_count), observed)
