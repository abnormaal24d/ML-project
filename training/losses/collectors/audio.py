"""Audio generation and reconstruction loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import (
    all_rows_coverage,
    row_coverage_from_mask,
    row_coverage_from_masked_targets,
)
from training.losses.tensor_ops import masked_mse, multi_codebook_cross_entropy


def collect_audio_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect audio generation and reconstruction loss terms."""
    terms: list[LossTerm] = []
    batch_size = (
        len(task_types)
        if isinstance(task_types := getattr(batch, "task_types", None), list)
        else 0
    )
    all_rows = all_rows_coverage(batch_size)

    # Audio generation
    if context.weights["audio_generation"] > 0.0:
        audio_logits = model_output.get("audio_token_logits")
        audio_targets = getattr(batch, "target_audio_token_ids", None)
        if audio_logits is not None and audio_targets is not None:
            audio_mask = getattr(
                batch, "target_audio_token_attention_mask", None
            )
            term = multi_codebook_cross_entropy(
                logits=audio_logits,
                targets=audio_targets.to(audio_logits.device),
                attention_mask=(
                    audio_mask.to(audio_logits.device)
                    if audio_mask is not None
                    else None
                ),
            )
            if term is not None:
                coverage = row_coverage_from_masked_targets(
                    audio_targets.to(audio_logits.device), audio_mask
                )
                terms.append(
                    LossTerm(
                        name="audio_generation",
                        value=term,
                        coverage=coverage.cpu(),
                    )
                )

    # Audio reconstruction
    if context.weights["audio_reconstruction"] > 0.0:
        audio_recon = model_output.get("audio_reconstruction")
        audio_recon_target = getattr(
            batch, "audio_reconstruction_target", None
        )
        if audio_recon is not None and audio_recon_target is not None:
            audio_recon_mask = getattr(
                batch, "audio_reconstruction_mask", None
            )
            term = masked_mse(
                pred=audio_recon,
                target=audio_recon_target.to(audio_recon.device),
                mask=(
                    audio_recon_mask.to(audio_recon.device)
                    if audio_recon_mask is not None
                    else None
                ),
            )
            if term is not None:
                if audio_recon_mask is not None:
                    coverage = row_coverage_from_mask(
                        audio_recon_mask.to(
                            device=torch.device("cpu"), dtype=torch.bool
                        ),
                        batch_size=batch_size,
                    )
                else:
                    coverage = all_rows
                terms.append(
                    LossTerm(
                        name="audio_reconstruction",
                        value=term,
                        coverage=coverage,
                    )
                )

    return tuple(terms)
