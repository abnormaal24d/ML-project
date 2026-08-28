"""Architecture tests for the refactored loss objective.

These pin the invariants introduced by the architectural refactor:
strict tensor contracts, per-row coverage with fail-closed validation,
weighted contrastive pair semantics, and the preference/build_loss mode
translation. They complement the characterization baseline in
``test_loss_objective_characterization.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from training.losses.objective import (
    SUPPORTED_LOSS_KEYS,
    SupervisedOrSelfSupervisedLoss,
)


def _batch(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "task_types": None,
        "labels": None,
        "alignment_scores": None,
        "decoder_labels": None,
        "target_token_ids": None,
        "target_attention_mask": None,
        "text_mlm_targets": None,
        "target_image_tensor": None,
        "target_audio_token_ids": None,
        "target_audio_token_attention_mask": None,
        "video_token_targets": None,
        "video_token_attention_mask": None,
        "image_reconstruction_target": None,
        "image_reconstruction_mask": None,
        "audio_reconstruction_target": None,
        "audio_reconstruction_mask": None,
        "video_temporal_labels": None,
        "safety_targets": None,
        "safety_target_mask": None,
        "output_modalities": [],
        "chosen_labels": None,
        "rejected_labels": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


# --- components: weighted symmetric InfoNCE -------------------------------


def test_weighted_info_nce_uses_pair_counts() -> None:
    from torch.nn.functional import cross_entropy

    from training.losses.components import symmetric_info_nce

    logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    weights = torch.tensor([1.0, 0.0, 0.0])

    loss = symmetric_info_nce(
        logits=logits, temperature=0.07, pair_weights=weights
    )

    assert loss is not None
    labels = torch.arange(3)
    per_row = (
        cross_entropy(logits / 0.07, labels, reduction="none")
        + cross_entropy((logits / 0.07).t(), labels, reduction="none")
    ) / 2
    expected = (per_row * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)


def test_weighted_info_nce_zero_total_weight_returns_none() -> None:
    from training.losses.components import symmetric_info_nce

    logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )

    assert (
        symmetric_info_nce(
            logits=logits,
            temperature=0.07,
            pair_weights=torch.zeros(3),
        )
        is None
    )


def test_symmetric_info_nce_rejects_non_square() -> None:
    from training.losses.components import symmetric_info_nce

    with pytest.raises(ValueError, match="square 2D similarity matrix"):
        symmetric_info_nce(logits=torch.zeros((2, 3)))


def test_hard_negative_margin_respects_pair_weights() -> None:
    from training.losses.components import hard_negative_margin_loss

    logits = torch.tensor([[10.0, 5.0], [5.0, 3.0]])
    weights = torch.tensor([0.0, 1.0])

    loss = hard_negative_margin_loss(
        logits=logits, margin=0.0, pair_weights=weights
    )

    assert loss is not None
    # row 1: positive = 3, hardest negative = 5, margin 0 -> 2
    assert torch.allclose(loss, torch.tensor(2.0))


# --- strict tensor contracts ----------------------------------------------


def test_masked_mse_right_aligned_broadcast_and_zero_valid() -> None:
    from training.losses.tensor_ops import masked_mse

    pred = torch.zeros((2, 3, 4, 4))
    target = torch.ones((2, 3, 4, 4))
    mask = torch.zeros((2, 4, 4), dtype=torch.bool)

    assert masked_mse(pred=pred, target=target, mask=mask) is None

    mask[0, :, :] = True
    loss = masked_mse(pred=pred, target=target, mask=mask)
    assert loss is not None
    assert torch.allclose(loss, torch.tensor(1.0))

    with pytest.raises(ValueError, match="equal pred/target shapes"):
        masked_mse(pred=pred, target=torch.zeros((2, 3, 4, 5)))


def test_causal_language_modeling_rejects_length_mismatch() -> None:
    from training.losses.tensor_ops import causal_language_modeling_loss

    with pytest.raises(ValueError, match="aligned"):
        causal_language_modeling_loss(
            logits=torch.zeros((1, 4, 5)),
            labels=torch.zeros((1, 6), dtype=torch.long),
        )


def test_audio_generation_uses_one_codebook_target_axis() -> None:
    from training.losses.tensor_ops import multi_codebook_cross_entropy

    logits = torch.zeros((2, 1, 3, 5))
    targets = torch.tensor(
        [
            [[0, 1, 2]],
            [[2, 3, 4]],
        ],
        dtype=torch.long,
    )
    loss = multi_codebook_cross_entropy(logits=logits, targets=targets)

    assert loss is not None
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="4D logits.*3D targets"):
        multi_codebook_cross_entropy(logits=logits, targets=targets.squeeze(1))
    with pytest.raises(ValueError, match="exactly one codebook axis"):
        multi_codebook_cross_entropy(
            logits=torch.zeros((2, 2, 3, 5)),
            targets=torch.zeros((2, 2, 3), dtype=torch.long),
        )


# --- contrastive row coverage -------------------------------------------------


def test_contrastive_row_indices_fail_closed_for_uncovered_rows() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )
    batch = _batch(
        task_types=["image_text_pair", "image_text_pair"],
        alignment_scores=torch.tensor([1.0, 0.0]),
    )

    with pytest.raises(RuntimeError, match="requires one of"):
        objective(
            model_output={
                "contrastive_logits": torch.tensor([[10.0]]),
                "contrastive_row_indices": torch.tensor([1]),
            },
            batch=batch,
        )


def test_contrastive_row_indices_cover_selected_pair_rows() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )
    batch = _batch(
        task_types=["image_text_pair", "instruction_following"],
        alignment_scores=torch.tensor([1.0, 1.0]),
        decoder_labels=torch.tensor(
            [
                [IGNORE_LABEL, IGNORE_LABEL, 2, 3],
                [IGNORE_LABEL, IGNORE_LABEL, 2, 3],
            ],
            dtype=torch.long,
        ),
    )
    logits = torch.full((2, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    logits[1, 1, 2] = 10.0
    logits[1, 2, 3] = 10.0

    losses = objective(
        model_output={
            "contrastive_logits": torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
            "contrastive_row_indices": torch.tensor([0, 1]),
            "sequence_logits": logits,
        },
        batch=batch,
    )

    assert "contrastive" in losses
    assert "language_modeling" in losses


def test_zero_contrastive_weight_disables_and_fails_closed() -> None:
    weights = {name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    weights["contrastive"] = 0.0
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=weights)
    batch = _batch(
        task_types=["image_text_pair"],
        alignment_scores=torch.tensor([1.0]),
    )

    with pytest.raises(RuntimeError, match="requires one of"):
        objective(
            model_output={
                "contrastive_logits": torch.tensor([[10.0, 0.0]]),
            },
            batch=batch,
        )


def test_unknown_task_fails_closed() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )
    batch = _batch(
        task_types=["mystery_task"],
        decoder_labels=torch.tensor(
            [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
        ),
    )
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0

    with pytest.raises(RuntimeError, match="unknown training task"):
        objective(
            model_output={"sequence_logits": logits},
            batch=batch,
        )


# --- preference/build_loss translation ----------------------------------------


def test_build_loss_reference_free_dpo_uses_pairwise() -> None:
    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    settings = TrainingSettings(
        preference_loss="dpo",
        reference_free_preference=True,
    )
    objective = build_training_loss(settings)

    assert objective.preference_mode == "pairwise"


def test_build_loss_rejects_dpo_without_reference_model() -> None:
    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    settings = TrainingSettings(
        preference_loss="dpo",
        reference_free_preference=False,
    )
    with pytest.raises(
        RuntimeError, match="reference-model DPO requires explicit"
    ):
        build_training_loss(settings)


def test_build_loss_maps_config_weights_exactly() -> None:
    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    objective = build_training_loss(TrainingSettings())
    assert set(objective.loss_weights) == set(SUPPORTED_LOSS_KEYS)
    assert objective.loss_weights["safety"] == 1.0

    settings = TrainingSettings(safety_loss_weight=0.25)
    assert build_training_loss(settings).loss_weights["safety"] == 0.25


# --- disabled-by-zero-weight primary loss --------------------------------------


def test_zero_language_weight_disables_and_fails_closed() -> None:
    weights = {name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    weights["language_modeling"] = 0.0
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=weights)
    batch = _batch(
        task_types=["instruction_following"],
        decoder_labels=torch.tensor(
            [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
        ),
    )
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0

    with pytest.raises(RuntimeError, match="requires one of"):
        objective(
            model_output={"sequence_logits": logits},
            batch=batch,
        )


# --- build_training_loss / config binding invariants ---------------------------


def test_build_training_loss_weight_map_matches_supported_keys() -> None:
    """The loss_weights dict inside build_training_loss must cover every
    entry in SUPPORTED_LOSS_KEYS and nothing more."""

    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    objective = build_training_loss(TrainingSettings())

    assert set(objective.loss_weights) == set(SUPPORTED_LOSS_KEYS)
    assert len(objective.loss_weights) == SUPPORTED_LOSS_KEYS.__len__()


def test_build_training_loss_reads_only_existing_config_fields() -> None:
    """Every ``settings.<field>`` accessed inside build_training_loss must
    be a real attribute on ``TrainingSettings``."""

    import inspect
    import re

    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    source = inspect.getsource(build_training_loss)
    referenced_fields = set(re.findall(r"settings\.(\w+)", source))

    config_fields = set(TrainingSettings.model_fields)

    missing = referenced_fields - config_fields
    assert not missing, (
        f"build_training_loss references config fields that do not exist "
        f"on TrainingSettings: {sorted(missing)}"
    )
