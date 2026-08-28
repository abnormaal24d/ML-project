"""Collate image inputs, targets, and optional edit masks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mmcrawler_datasets.collation.tensor_ops import (
    mask_image,
    sample_generator,
    stack_feature_matrix,
)
from mmcrawler_datasets.tensors import load_optional_tensor_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mmcrawler_datasets.schema import MultimodalSample
    from mmcrawler_datasets.tensors import SampleTensorSource


class ImageCollator:
    """Build masked image tensors and optional image-side targets."""

    def __init__(
        self,
        *,
        image_size: int,
        dataset_root: Path,
        tensor_source: SampleTensorSource,
        image_mask_probability: float,
        base_seed: int = 0,
    ) -> None:
        self._tensor_source = tensor_source
        self._dataset_root = Path(dataset_root)
        self.raw_image_size = int(image_size)
        self.image_mask_probability = float(image_mask_probability)
        self._base_seed = int(base_seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def collate_sample(
        self,
        sample: MultimodalSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked image input, reconstruction target, and mask."""

        image = self.image_tensor_for_sample(sample=sample)
        generator = sample_generator(
            base_seed=self._base_seed,
            epoch=self._epoch,
            sample_id=str(sample.sample_id),
            operation="image_mask",
        )
        return mask_image(
            image,
            probability=(
                self.image_mask_probability if sample.has_image else 0.0
            ),
            generator=generator,
        )

    def collate_batch(
        self,
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack masked image tensors for a batch of samples."""

        image_inputs: list[torch.Tensor] = []
        image_targets: list[torch.Tensor] = []
        image_masks: list[torch.Tensor] = []
        for sample in samples:
            masked_image, image_target, image_mask = self.collate_sample(
                sample
            )
            image_inputs.append(masked_image)
            image_targets.append(image_target)
            image_masks.append(image_mask)
        return (
            stack_feature_matrix(image_inputs),
            stack_feature_matrix(image_targets),
            stack_feature_matrix(image_masks),
        )

    def image_tensor_for_sample(
        self,
        *,
        sample: MultimodalSample,
    ) -> torch.Tensor:
        """Load the image tensor for one sample from the configured source."""

        return self._tensor_source.image_tensor(sample=sample)

    def collate_optional_images(
        self,
        *,
        paths: Sequence[Path | None],
    ) -> torch.Tensor | None:
        """Load an optional batch of materialized RGB image tensors."""

        return load_optional_tensor_batch(
            dataset_root=self._dataset_root,
            paths=paths,
            expected_shape=(3, self.raw_image_size, self.raw_image_size),
            dtype=torch.float32,
        )

    def collate_optional_edit_masks(
        self,
        *,
        paths: Sequence[Path | None],
    ) -> torch.Tensor | None:
        """Load an optional batch of materialized single-channel masks."""

        return load_optional_tensor_batch(
            dataset_root=self._dataset_root,
            paths=paths,
            expected_shape=(1, self.raw_image_size, self.raw_image_size),
            dtype=torch.float32,
        )
