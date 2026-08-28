"""Sequence loss collectors: causal LM, OCR, legacy sequence, MLM."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from training.losses.contracts import LossContext, LossTerm
from training.losses.coverage import (
    row_coverage_from_ignore_labels,
    row_coverage_from_sequence_mask,
)
from training.losses.tensor_ops import (
    causal_language_modeling_loss,
    token_cross_entropy,
)
from training.losses.validation import rows_producing_loss


def _collect_causal_term(
    *,
    name: str,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
    required_loss: str,
) -> LossTerm | None:
    """Collect a causal language modeling loss term (language_modeling or ocr_sequence)."""
    if context.weights[name] <= 0.0:
        return None

    sequence_logits = model_output.get("sequence_logits")
    decoder_labels = getattr(batch, "decoder_labels", None)
    if sequence_logits is None or decoder_labels is None:
        return None

    task_types = getattr(batch, "task_types", None)
    task_rows = rows_producing_loss(
        task_types, required_loss, device=sequence_logits.device
    )

    labels = decoder_labels.to(sequence_logits.device)
    scored_rows = labels[:, 1:].ne(IGNORE_LABEL).any(dim=1)
    valid_rows = scored_rows if task_rows is None else task_rows & scored_rows
    if not valid_rows.any():
        return None

    term = causal_language_modeling_loss(
        logits=sequence_logits,
        labels=labels,
        row_mask=valid_rows,
    )
    if term is None:
        return None

    return LossTerm(
        name=name,
        value=term,
        coverage=valid_rows.cpu(),
    )


def collect_sequence_losses(
    *,
    model_output: Mapping[str, torch.Tensor],
    batch: object,
    context: LossContext,
) -> tuple[LossTerm, ...]:
    """Collect all sequence-related loss terms.

    This includes:
    - language_modeling (causal LM for instruction_following, etc.)
    - ocr_sequence (causal LM for OCR tasks)
    - sequence (legacy token_cross_entropy for pipeline_smoke)
    - text_mlm (masked language modeling)
    """
    # Skip all text losses during preference tuning stage
    if context.training_stage == "PREFERENCE_TUNING":
        return ()

    terms: list[LossTerm] = []

    # Language modeling (causal LM)
    term = _collect_causal_term(
        name="language_modeling",
        model_output=model_output,
        batch=batch,
        context=context,
        required_loss="language_modeling",
    )
    if term is not None:
        terms.append(term)

    # OCR sequence (causal LM for OCR tasks)
    term = _collect_causal_term(
        name="ocr_sequence",
        model_output=model_output,
        batch=batch,
        context=context,
        required_loss="ocr_sequence",
    )
    if term is not None:
        terms.append(term)

    # Legacy sequence loss (pipeline_smoke backend or no decoder_labels)
    if context.weights["sequence"] > 0.0:
        sequence_logits = model_output.get("sequence_logits")
        target_token_ids = getattr(batch, "target_token_ids", None)
        if (
            sequence_logits is not None
            and target_token_ids is not None
            and (
                context.training_backend == "pipeline_smoke"
                or getattr(batch, "decoder_labels", None) is None
            )
        ):
            targets = target_token_ids.to(sequence_logits.device)
            attention_mask = getattr(batch, "target_attention_mask", None)
            term = token_cross_entropy(
                logits=sequence_logits,
                targets=targets,
                attention_mask=attention_mask,
            )
            if term is not None:
                task_types = getattr(batch, "task_types", None)
                batch_size = (
                    len(task_types) if isinstance(task_types, list) else 0
                )
                coverage = row_coverage_from_sequence_mask(
                    attention_mask, batch_size=batch_size
                )
                terms.append(
                    LossTerm(
                        name="sequence",
                        value=term,
                        coverage=coverage,
                    )
                )

    # MLM loss
    if context.weights["text_mlm"] > 0.0:
        mlm_logits = model_output.get("text_mlm_logits")
        mlm_targets = getattr(batch, "text_mlm_targets", None)
        if mlm_logits is not None and mlm_targets is not None:
            term = token_cross_entropy(
                logits=mlm_logits,
                targets=mlm_targets,
            )
            if term is not None:
                coverage = row_coverage_from_ignore_labels(mlm_targets)
                terms.append(
                    LossTerm(
                        name="text_mlm",
                        value=term,
                        coverage=coverage,
                    )
                )

    return tuple(terms)
