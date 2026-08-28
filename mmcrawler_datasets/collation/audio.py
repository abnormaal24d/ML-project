"""Collate raw audio inputs and token targets for multimodal batches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mmcrawler_datasets.collation.tensor_ops import (
    IGNORE_LABEL,
    MASK_DTYPE,
    TEXT_TOKEN_DTYPE,
    mask_audio,
    pad_audio_frames,
    sample_generator,
    stack_feature_matrix,
)
from mmcrawler_datasets.tensors import load_required_token_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mmcrawler_datasets.schema import MultimodalSample
    from mmcrawler_datasets.tensors import (
        SampleTensorSource,
    )


class AudioCollator:
    """Build masked audio tensors and audio-token targets from samples."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        tensor_source: SampleTensorSource,
        audio_mask_probability: float,
        audio_token_codec: str,
        base_seed: int = 0,
    ) -> None:
        self._tensor_source = tensor_source
        self._dataset_root = Path(dataset_root)
        self.audio_mask_probability = float(audio_mask_probability)
        self.audio_token_codec = str(audio_token_codec).strip().lower()
        self._base_seed = int(base_seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def collate_sample(
        self,
        sample: MultimodalSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked audio input, reconstruction target, and mask."""

        audio = self.audio_tensor_for_sample(sample=sample)
        generator = sample_generator(
            base_seed=self._base_seed,
            epoch=self._epoch,
            sample_id=str(sample.sample_id),
            operation="audio_mask",
        )
        return mask_audio(
            audio,
            probability=(
                self.audio_mask_probability if sample.has_audio else 0.0
            ),
            generator=generator,
        )

    def collate_batch(
        self,
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack masked audio tensors for a batch of samples."""

        audio_inputs: list[torch.Tensor] = []
        audio_targets: list[torch.Tensor] = []
        audio_masks: list[torch.Tensor] = []
        for sample in samples:
            masked_audio, audio_target, audio_mask = self.collate_sample(
                sample
            )
            audio_inputs.append(masked_audio)
            audio_targets.append(audio_target)
            audio_masks.append(audio_mask)
        return (
            stack_feature_matrix(audio_inputs),
            stack_feature_matrix(audio_targets),
            stack_feature_matrix(audio_masks),
        )

    def audio_tensor_for_sample(
        self,
        *,
        sample: MultimodalSample,
    ) -> torch.Tensor:
        """Load the audio tensor for one sample from the configured source."""

        return self._tensor_source.audio_tensor(sample=sample)

    def collate_audio_token_targets(
        self,
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad and stack audio-token targets with attention masks."""

        token_sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        max_time = 0

        for sample in samples:
            target_requested = (
                "audio" in sample.output_modalities
                or sample.target_audio_path is not None
                or sample.target_audio_tokens_path is not None
            )
            token_tensor = (
                load_required_token_tensor(
                    dataset_root=self._dataset_root,
                    path=sample.target_audio_tokens_path,
                )
                if sample.target_audio_tokens_path is not None
                else None
            )
            if token_tensor is not None:
                if token_tensor.dim() == 1:
                    token_tensor = token_tensor.unsqueeze(0)
                if (
                    token_tensor.dim() == 2
                    and token_tensor.shape[0] > token_tensor.shape[1]
                ):
                    token_tensor = token_tensor.T
                if token_tensor.dim() == 3:
                    token_tensor = (
                        token_tensor[0]
                        if token_tensor.shape[0] == 1
                        else token_tensor.squeeze(0)
                    )
                token_sequences.append(token_tensor)
                tdim = (
                    token_tensor.shape[-1]
                    if token_tensor.dim() > 1
                    else token_tensor.shape[0]
                )
                max_time = max(max_time, tdim)
                masks.append(torch.ones(tdim, dtype=MASK_DTYPE))
                continue

            if not target_requested:
                token_sequences.append(
                    torch.full((1, 1), IGNORE_LABEL, dtype=TEXT_TOKEN_DTYPE)
                )
                masks.append(torch.zeros(1, dtype=MASK_DTYPE))
                max_time = max(max_time, 1)
                continue

            # "none" is a codec configuration sentinel, not a credential.
            if self.audio_token_codec == "none":  # nosec B105
                raise ValueError(
                    "audio output requires an enabled audio token codec"
                )
            raise FileNotFoundError(
                "materialized target_audio_tokens_path is required for "
                f"audio generation sample {sample.sample_id}"
            )

        return pad_audio_frames(
            token_sequences,
            masks,
            max_time=max_time,
            pad_value=IGNORE_LABEL,
        )

    @staticmethod
    def first_audio_token_per_sample(
        *,
        token_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return the first non-padding audio token for each batch row."""

        if token_ids is None or token_ids.numel() == 0:
            return None
        if token_ids.dim() == 3:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)
        flattened = token_ids.reshape(token_ids.shape[0], -1)
        valid = flattened.ne(IGNORE_LABEL)
        first_positions = valid.to(dtype=TEXT_TOKEN_DTYPE).argmax(dim=1)
        rows = torch.arange(flattened.shape[0], device=flattened.device)
        first_tokens = flattened[rows, first_positions]
        return torch.where(
            valid.any(dim=1),
            first_tokens,
            first_tokens.new_full(first_tokens.shape, IGNORE_LABEL),
        )
