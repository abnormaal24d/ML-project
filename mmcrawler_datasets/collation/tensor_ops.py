"""Tensor dtype helpers and materialized tensor loading utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

TEXT_TOKEN_DTYPE = torch.long
FEATURE_DTYPE = torch.float32
LABEL_DTYPE = torch.long
MASK_DTYPE = torch.bool
IGNORE_LABEL = -100


def fit_feature_dim(vector: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Trim or zero-pad a feature vector to a fixed dimension."""

    flat = vector.detach().to(dtype=torch.float32).flatten()
    if flat.numel() == dim:
        return flat
    if flat.numel() > dim:
        return flat[:dim]
    output = torch.zeros(dim, dtype=torch.float32)
    output[: flat.numel()] = flat
    return output


def normalize_feature(vector: torch.Tensor) -> torch.Tensor:
    """L2-normalize a feature vector when its norm is positive."""

    norm = vector.norm(p=2)
    if float(norm) <= 0.0:
        return vector
    return cast("torch.Tensor", vector / norm)


def to_long_tensor(values: Sequence[int]) -> torch.Tensor:
    """Convert integer values into a token-id tensor."""

    return torch.tensor(list(values), dtype=TEXT_TOKEN_DTYPE)


def to_float_tensor(values: Sequence[float]) -> torch.Tensor:
    """Convert float values into a feature tensor."""

    return torch.tensor(list(values), dtype=FEATURE_DTYPE)


def to_bool_tensor(values: Sequence[bool]) -> torch.Tensor:
    """Convert boolean values into a mask tensor."""

    return torch.tensor(list(values), dtype=MASK_DTYPE)


def stack_feature_matrix(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack feature tensors along a new batch dimension."""

    if not tensors:
        return torch.empty(0, dtype=FEATURE_DTYPE)
    return torch.stack(list(tensors))


def safe_tensor_from_numpy(array: Any) -> torch.Tensor:
    """Convert a NumPy array into a contiguous feature tensor."""

    import numpy as np

    return torch.as_tensor(np.asarray(array), dtype=FEATURE_DTYPE)


# --- padding helpers ---


def pad_sequence_values(
    sequences: Sequence[torch.Tensor],
    *,
    max_length: int,
    pad_value: int | float,
    pad_dim: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Pad each sequence to a shared maximum length."""

    padded: list[torch.Tensor] = []
    for sequence in sequences:
        current_length = sequence.shape[pad_dim]
        pad_length = max_length - current_length
        if pad_length <= 0:
            padded.append(sequence)
            continue
        pad_shape = list(sequence.shape)
        pad_shape[pad_dim] = pad_length
        pad = torch.full(
            pad_shape,
            fill_value=pad_value,
            dtype=dtype,
            device=sequence.device,
        )
        padded.append(torch.cat([sequence, pad], dim=pad_dim))
    return padded


def pad_token_ids(
    token_sequences: Sequence[torch.Tensor],
    *,
    max_time: int,
    pad_value: int = IGNORE_LABEL,
) -> list[torch.Tensor]:
    """Pad token-id sequences to a shared time length."""

    return pad_sequence_values(
        token_sequences,
        max_length=max_time,
        pad_value=pad_value,
        pad_dim=1,
        dtype=TEXT_TOKEN_DTYPE,
    )


def pad_attention_mask(
    masks: Sequence[torch.Tensor],
    *,
    max_time: int,
) -> list[torch.Tensor]:
    """Pad attention masks to a shared time length."""

    return pad_sequence_values(
        masks,
        max_length=max_time,
        pad_value=0,
        pad_dim=0,
        dtype=MASK_DTYPE,
    )


def pad_feature_frames(
    frames: Sequence[torch.Tensor],
    *,
    max_frames: int,
) -> list[torch.Tensor]:
    """Pad feature frame sequences to a shared frame count."""

    return pad_sequence_values(
        frames,
        max_length=max_frames,
        pad_value=0.0,
        pad_dim=0,
        dtype=FEATURE_DTYPE,
    )


def pad_audio_frames(
    token_sequences: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    *,
    max_time: int,
    pad_value: int = IGNORE_LABEL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad audio token ids and masks into stacked batch tensors."""

    padded_tokens = pad_token_ids(
        token_sequences,
        max_time=max_time,
        pad_value=pad_value,
    )
    padded_masks = pad_attention_mask(masks, max_time=max_time)
    token_ids = (
        stack_feature_matrix(padded_tokens)
        if padded_tokens
        else torch.empty(0, dtype=TEXT_TOKEN_DTYPE)
    )
    attention_mask = (
        stack_feature_matrix(padded_masks)
        if padded_masks
        else torch.empty(0, dtype=MASK_DTYPE)
    )
    return token_ids, attention_mask


def pad_video_frames(
    frames: Sequence[torch.Tensor],
    *,
    max_frames: int,
) -> torch.Tensor:
    """Pad video frames and stack them into one batch tensor."""

    padded = pad_feature_frames(frames, max_frames=max_frames)
    return stack_feature_matrix(padded)


# --- sample_rng.py ---


def sample_generator(
    *,
    base_seed: int,
    epoch: int,
    sample_id: str,
    operation: str,
) -> torch.Generator:
    """Build a torch Generator seeded from sample identity and epoch."""

    value = f"{base_seed}:{epoch}:{sample_id}:{operation}".encode("utf-8")
    digest = hashlib.blake2b(value, digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little")
    return torch.Generator().manual_seed(seed)


# --- text_masking.py ---

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
IGNORE_INDEX = -100


def mask_text(
    tokens: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator,
    pad_token_id: int = PAD_TOKEN_ID,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked input tokens and MLM labels."""

    masked = tokens.clone()
    labels = torch.full_like(tokens, fill_value=IGNORE_INDEX)
    if probability <= 0:
        return masked, labels

    candidate_mask = tokens.ne(int(pad_token_id))
    random_mask = torch.rand(tokens.shape, generator=generator).lt(probability)
    selected = candidate_mask & random_mask
    labels[selected] = tokens[selected]
    masked[selected] = int(mask_token_id)
    return masked, labels


# --- image_masking.py ---


def mask_image(
    image: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator,
    patch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask square image patches for reconstruction pretraining."""

    target = image.clone()
    masked = image.clone()
    mask = torch.zeros_like(image)
    if probability <= 0 or image.numel() == 0:
        return masked, target, mask

    height = int(image.shape[-2])
    width = int(image.shape[-1])
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            if float(torch.rand((), generator=generator)) >= probability:
                continue
            row_end = min(height, row + patch_size)
            col_end = min(width, col + patch_size)
            masked[:, row:row_end, col:col_end] = 0.0
            mask[:, row:row_end, col:col_end] = 1.0
    return masked, target, mask


# --- audio_masking.py ---


def mask_audio(
    audio: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator,
    span_size: int = 400,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask waveform spans for masked audio prediction."""

    target = audio.clone()
    masked = audio.clone()
    mask = torch.zeros_like(audio)
    if probability <= 0 or audio.numel() == 0:
        return masked, target, mask

    length = int(audio.shape[-1])
    for start in range(0, length, span_size):
        selected = float(torch.rand((), generator=generator)) < probability
        if not selected:
            continue
        end = min(length, start + span_size)
        masked[..., start:end] = 0.0
        mask[..., start:end] = 1.0
    return masked, target, mask


# --- video_masking.py ---


def maybe_reverse_video(
    video: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a binary temporal-order target for video pretraining."""

    if (
        int(video.shape[0]) <= 1
        or float(torch.rand((), generator=generator)) < 0.5
    ):
        return video, torch.tensor(0, dtype=torch.long)
    return torch.flip(video, dims=(0,)), torch.tensor(1, dtype=torch.long)
