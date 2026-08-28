"""Image generation and reconstruction loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import (
    all_rows_coverage,
    row_coverage_from_mask,
)
from training.losses.tensor_ops import masked_mse


def collect_image_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect image generation and reconstruction loss terms."""
    terms: list[LossTerm] = []
    batch_size = (
        len(task_types)
        if isinstance(task_types := getattr(batch, "task_types", None), list)
        else 0
    )
    all_rows = all_rows_coverage(batch_size)

    # Image generation
    if context.weights["image_generation"] > 0.0:
        generated_image = model_output.get("generated_image")
        target_image = getattr(batch, "target_image_tensor", None)
        target_image_mask = getattr(batch, "target_image_mask", None)
        if generated_image is not None and target_image is not None:
            term = masked_mse(
                pred=generated_image,
                target=target_image.to(generated_image.device),
                mask=(
                    target_image_mask.to(generated_image.device)
                    if target_image_mask is not None
                    else None
                ),
            )
            if term is not None:
                terms.append(
                    LossTerm(
                        name="image_generation",
                        value=term,
                        coverage=row_coverage_from_mask(
                            target_image_mask
                            if target_image_mask is not None
                            else all_rows,
                            batch_size=batch_size,
                        ),
                    )
                )

    # Image reconstruction
    if context.weights["image_reconstruction"] > 0.0:
        image_recon = model_output.get("image_reconstruction")
        image_recon_target = getattr(
            batch, "image_reconstruction_target", None
        )
        if image_recon is not None and image_recon_target is not None:
            image_recon_mask = getattr(
                batch, "image_reconstruction_mask", None
            )
            term = masked_mse(
                pred=image_recon,
                target=image_recon_target.to(image_recon.device),
                mask=(
                    image_recon_mask.to(image_recon.device)
                    if image_recon_mask is not None
                    else None
                ),
            )
            if term is not None:
                if image_recon_mask is not None:
                    coverage = row_coverage_from_mask(
                        image_recon_mask.to(
                            device=torch.device("cpu"), dtype=torch.bool
                        ),
                        batch_size=batch_size,
                    )
                else:
                    coverage = all_rows
                terms.append(
                    LossTerm(
                        name="image_reconstruction",
                        value=term,
                        coverage=coverage,
                    )
                )

    return tuple(terms)
