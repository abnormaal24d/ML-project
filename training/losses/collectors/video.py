"""Video generation and temporal loss collector."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from training.losses.components import classification_loss
from training.losses.contracts import LossContext, LossTerm
from training.losses.tensor_ops import video_token_cross_entropy


def collect_video_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect video generation and temporal classification loss terms."""
    terms: list[LossTerm] = []

    # Video generation
    if context.weights["video_generation"] > 0.0:
        video_logits = model_output.get("video_token_logits")
        video_targets = getattr(batch, "video_token_targets", None)
        if video_logits is not None and video_targets is not None:
            video_mask = getattr(batch, "video_token_attention_mask", None)
            term = video_token_cross_entropy(
                logits=video_logits,
                targets=video_targets.to(video_logits.device),
                attention_mask=(
                    video_mask.to(video_logits.device)
                    if video_mask is not None
                    else None
                ),
            )
            if term is not None:
                valid = video_targets.to(video_logits.device).ne(IGNORE_LABEL)
                if video_mask is not None:
                    valid &= video_mask.to(video_logits.device).bool()[
                        :, :, None, None
                    ]
                terms.append(
                    LossTerm(
                        name="video_generation",
                        value=term,
                        coverage=valid.flatten(1).any(dim=1).cpu(),
                    )
                )

    # Video temporal classification
    if context.weights["video_temporal"] > 0.0:
        video_temporal_logits = model_output.get("video_temporal_logits")
        video_temporal_labels = getattr(batch, "video_temporal_labels", None)
        if (
            video_temporal_logits is not None
            and video_temporal_labels is not None
        ):
            labels = video_temporal_labels.to(
                device=video_temporal_logits.device, dtype=torch.long
            )
            valid = labels.ne(IGNORE_LABEL)
            if valid.any():
                loss = classification_loss(
                    logits=video_temporal_logits[valid],
                    labels=labels[valid],
                )
                terms.append(
                    LossTerm(
                        name="video_temporal",
                        value=loss,
                        coverage=valid.any(dim=1).cpu(),
                    )
                )

    return tuple(terms)
