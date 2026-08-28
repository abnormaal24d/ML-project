"""Loss objective orchestrator.

Thin composite module that coordinates loss collectors, validates training
contracts, and finalizes weighted losses. All concrete loss mathematics
live in components.py and tensor_ops.py. Collector logic lives in
collectors/*.py. Validation logic lives in validation.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .collectors.audio import collect_audio_losses
from .collectors.classification import collect_classification
from .collectors.contrastive import collect_contrastive
from .collectors.image import collect_image_losses
from .collectors.preference import collect_preference_losses
from .collectors.safety import collect_safety_losses
from .collectors.sequence import collect_sequence_losses
from .collectors.video import collect_video_losses
from .contracts import LossCollection, LossContext
from .validation import (
    SUPPORTED_LOSS_KEYS,
    NonFiniteLossError,
    validate_final_losses,
    validate_required_losses,
)


class SupervisedOrSelfSupervisedLoss(nn.Module):
    """Weighted bundle of task losses with row-level coverage validation."""

    def __init__(
        self,
        *,
        contrastive_temperature: float = 0.07,
        alignment_score_exponent: float = 1.0,
        hard_negative_margin: float = 0.2,
        training_backend: str = "pipeline_smoke",
        training_stage: str = "MULTIMODAL_PRETRAIN",
        preference_mode: str = "pairwise",
        preference_beta: float = 0.1,
        loss_weights: Mapping[str, float],
    ) -> None:
        super().__init__()
        if preference_mode not in {"pairwise", "dpo"}:
            raise ValueError(
                f"unsupported preference objective: {preference_mode!r}"
            )
        if preference_mode == "dpo":
            raise RuntimeError(
                "reference-model DPO requires explicit reference log "
                "probabilities; set reference_free_preference=true"
            )
        if contrastive_temperature <= 0.0:
            raise ValueError("contrastive temperature must be positive")
        weights = {
            str(name): float(weight) for name, weight in loss_weights.items()
        }
        if set(weights) != SUPPORTED_LOSS_KEYS:
            missing = sorted(SUPPORTED_LOSS_KEYS - weights.keys())
            unknown = sorted(weights.keys() - SUPPORTED_LOSS_KEYS)
            raise ValueError(
                "loss_weights must define the complete objective contract; "
                f"missing={missing}, unknown={unknown}"
            )
        for name, weight in weights.items():
            if not torch.isfinite(torch.tensor(weight)) or weight < 0.0:
                raise ValueError(
                    f"loss weight for {name!r} must be finite and non-negative"
                )
        self.loss_weights = weights
        self.contrastive_temperature = contrastive_temperature
        self.alignment_score_exponent = alignment_score_exponent
        self.hard_negative_margin = float(hard_negative_margin)
        self.training_backend = str(training_backend)
        self.training_stage = str(training_stage)
        self.preference_mode = preference_mode
        self.preference_beta = float(preference_beta)

    @property
    def _context(self) -> LossContext:
        return LossContext(
            weights=self.loss_weights,
            contrastive_temperature=self.contrastive_temperature,
            alignment_score_exponent=self.alignment_score_exponent,
            hard_negative_margin=self.hard_negative_margin,
            training_backend=self.training_backend,
            training_stage=self.training_stage,
            preference_mode=self.preference_mode,
            preference_beta=self.preference_beta,
        )

    def forward(
        self,
        *,
        model_output: dict[str, torch.Tensor],
        batch: Any,
        require_targets_for_generation: bool = False,
    ) -> dict[str, torch.Tensor]:
        collection = LossCollection()

        # Contrastive and hard negative (share pair context)
        collection.extend(
            collect_contrastive(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Classification
        collection.extend(
            collect_classification(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Sequence losses (LM, OCR, legacy sequence, MLM)
        collection.extend(
            collect_sequence_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Modality generation losses
        collection.extend(
            collect_image_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )
        collection.extend(
            collect_audio_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )
        collection.extend(
            collect_video_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Safety
        collection.extend(
            collect_safety_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Preference (only during PREFERENCE_TUNING)
        collection.extend(
            collect_preference_losses(
                model_output=model_output,
                batch=batch,
                context=self._context,
            )
        )

        # Build losses and coverage dicts for validation
        losses = {term.name: term.value for term in collection.values()}
        coverage = {term.name: term.coverage for term in collection.values()}

        # Validate required losses per task and generation targets
        validate_required_losses(
            losses=losses,
            coverage=coverage,
            batch=batch,
            model_output=model_output,
            require_targets_for_generation=require_targets_for_generation,
            training_stage=self.training_stage,
        )

        # Finalize weighted losses
        return validate_final_losses(
            losses=losses,
            loss_weights=self.loss_weights,
            available_outputs=model_output.keys(),
        )


# Re-export for backward compatibility
__all__ = [
    "SupervisedOrSelfSupervisedLoss",
    "SUPPORTED_LOSS_KEYS",
    "NonFiniteLossError",
]
