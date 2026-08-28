"""Characterization tests for the loss objective contract.

Baseline that documents the CURRENT observable behavior of
``training.losses.objective.SupervisedOrSelfSupervisedLoss`` before the
architectural refactor.

- Tests marked "stable contract" must keep passing unchanged after the
  refactor (they pin behavior the architecture advice preserves).
- Tests marked "CURRENT BEHAVIOR" document behaviors the architecture
  advice intentionally corrects; the migration phase updates these
  assertions together with the production change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from training.losses.objective import (
    SUPPORTED_LOSS_KEYS,
    NonFiniteLossError,
    SupervisedOrSelfSupervisedLoss,
)


def _full_weights(**overrides: float) -> dict[str, float]:
    weights = {name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    weights.update(overrides)
    return weights


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


# --- stable contract: empty / non-finite / weighted finalization ---------


def test_characterization_empty_forward_raises_with_available_outputs() -> (
    None
):
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())

    with pytest.raises(RuntimeError, match="no trainable loss terms") as exc:
        objective(
            model_output={"unrelated": torch.zeros(1)},
            batch=_batch(),
        )
    assert "unrelated" in str(exc.value)


def test_characterization_non_finite_term_identifies_the_term() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        safety_targets=torch.tensor([[1.0, 0.0]]),
        safety_target_mask=torch.tensor([[True, True]]),
    )

    with pytest.raises(NonFiniteLossError, match="safety"):
        objective(
            model_output={
                "safety_logits": torch.tensor([[float("nan"), 0.0]])
            },
            batch=batch,
        )


def test_characterization_constructor_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        SupervisedOrSelfSupervisedLoss(
            loss_weights=_full_weights(language_modeling=-1.0)
        )


# --- CURRENT BEHAVIOR: weight mapping semantics --------------------------
# The architecture advice replaces partial overrides with a single mapping
# that must match SUPPORTED_LOSS_KEYS exactly.


def test_characterization_requires_explicit_complete_weight_mapping() -> None:
    # No hidden defaults: the objective must receive every term exactly.
    with pytest.raises(ValueError, match="complete objective contract"):
        SupervisedOrSelfSupervisedLoss(loss_weights={"label": 2.5})

    weights = _full_weights()
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=weights)

    assert objective.loss_weights == weights
    assert set(objective.loss_weights) == SUPPORTED_LOSS_KEYS


def test_characterization_label_uses_label_logits_and_full_mapping() -> None:
    from training.losses.objective import SUPPORTED_LOSS_KEYS

    weights = {name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    weights["label"] = 2.5
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=weights)
    logits = torch.tensor([[2.0, -1.0]])

    losses = objective(
        model_output={"label_logits": logits},
        batch=_batch(labels=torch.tensor([0])),
    )

    expected = (
        torch.nn.functional.cross_entropy(logits, torch.tensor([0])) * 2.5
    )
    assert torch.allclose(losses["label"], expected)
    assert torch.allclose(losses["total"], expected)


def test_characterization_override_applies_configured_weight() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights=_full_weights(label=2.5)
    )

    assert objective.loss_weights["label"] == 2.5
    assert objective.loss_weights["contrastive"] == 1.0


def test_characterization_unknown_weight_key_is_rejected() -> None:
    weights = _full_weights()
    weights["mystery_term"] = 3.0

    with pytest.raises(ValueError, match="complete objective contract"):
        SupervisedOrSelfSupervisedLoss(loss_weights=weights)


def test_characterization_label_logits_takes_precedence_over_logits() -> None:
    # Stable contract: the classifier head publishes label_logits and
    # forward prefers it over any generic logits key.
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    label_logits = torch.tensor([[10.0, -5.0]])
    logits = torch.tensor([[2.0, -1.0]])

    losses = objective(
        model_output={"label_logits": label_logits, "logits": logits},
        batch=_batch(labels=torch.tensor([0])),
    )

    expected = torch.nn.functional.cross_entropy(
        label_logits, torch.tensor([0])
    )
    assert torch.allclose(losses["label"], expected)


# --- stable contract: public key contracts -------------------------------


def test_characterization_supported_loss_keys() -> None:
    from training.losses.objective import SUPPORTED_LOSS_KEYS

    assert SUPPORTED_LOSS_KEYS == frozenset(
        {
            "audio_generation",
            "audio_reconstruction",
            "contrastive",
            "hard_negative",
            "image_generation",
            "image_reconstruction",
            "label",
            "language_modeling",
            "ocr_sequence",
            "preference",
            "safety",
            "sequence",
            "text_mlm",
            "video_generation",
            "video_temporal",
        }
    )


def test_characterization_task_producible_loss_terms_public_contract() -> None:
    # Stable task-registry contract consumed by configuration validation and
    # release-requirements tests.
    from multimodal.tasks.registry import (
        task_has_trainable_loss,
        task_producible_loss_terms,
    )

    assert task_producible_loss_terms("text_pretrain") == frozenset(
        {"text_mlm"}
    )
    assert task_producible_loss_terms("instruction_following") == frozenset(
        {"language_modeling", "sequence"}
    )
    assert task_producible_loss_terms("image_text_pair") == frozenset(
        {"contrastive", "hard_negative"}
    )
    assert task_producible_loss_terms("ocr_parse") == frozenset(
        {"ocr_sequence", "sequence"}
    )
    assert task_producible_loss_terms("unknown_task") == frozenset()
    assert task_has_trainable_loss("unknown_task") is False
    assert task_has_trainable_loss("vqa") is True


# --- stable contract: contrastive ----------------------------------------


def test_characterization_contrastive_emits_symmetric_info_nce() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        task_types=["image_text_pair", "image_text_pair"],
        alignment_scores=torch.tensor([1.0, 1.0]),
    )
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])

    losses = objective(
        model_output={"contrastive_logits": logits},
        batch=batch,
    )

    assert set(losses) == {"contrastive", "total"}
    assert torch.allclose(losses["total"], torch.tensor(0.0))


def test_characterization_contrastive_requires_alignment_scores() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(task_types=["image_text_pair", "image_text_pair"])

    with pytest.raises(RuntimeError, match="requires batch.alignment_scores"):
        objective(
            model_output={
                "contrastive_logits": torch.tensor([[10.0, 0.0], [0.0, 10.0]])
            },
            batch=batch,
        )


def test_characterization_contrastive_zero_scores_fail_row_coverage() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        task_types=["image_text_pair", "image_text_pair"],
        alignment_scores=torch.zeros(2),
    )

    with pytest.raises(RuntimeError, match="requires one of"):
        objective(
            model_output={
                "contrastive_logits": torch.tensor([[10.0, 0.0], [0.0, 10.0]])
            },
            batch=batch,
        )


def test_characterization_hard_negative_uses_square_logits() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    logits = torch.tensor([[10.0, 0.0], [0.0, 5.0]])

    losses = objective(
        model_output={"hard_negative_logits": logits},
        batch=_batch(),
    )

    assert set(losses) == {"hard_negative", "total"}
    assert losses["hard_negative"].item() == 0.0


def test_characterization_hard_negative_rejects_non_square_logits() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())

    with pytest.raises(ValueError, match="square 2D similarity matrix"):
        objective(
            model_output={"hard_negative_logits": torch.zeros((2, 3))},
            batch=_batch(),
        )


# --- stable contract: text losses ----------------------------------------


def test_characterization_causal_lm_rows_emit_language_modeling_only() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        task_types=["instruction_following"],
        decoder_labels=torch.tensor(
            [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
        ),
    )
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0

    losses = objective(
        model_output={"sequence_logits": logits},
        batch=batch,
    )

    assert set(losses) == {"language_modeling", "total"}
    assert losses["language_modeling"].item() < 1e-5


def test_characterization_ocr_rows_emit_ocr_sequence_only() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        task_types=["ocr_parse"],
        decoder_labels=torch.tensor(
            [[IGNORE_LABEL, IGNORE_LABEL, 1, 4]], dtype=torch.long
        ),
    )
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 1, 1] = 10.0
    logits[0, 2, 4] = 10.0

    losses = objective(
        model_output={"sequence_logits": logits},
        batch=batch,
    )

    assert set(losses) == {"ocr_sequence", "total"}
    assert losses["ocr_sequence"].item() < 1e-5


def test_characterization_legacy_sequence_used_when_no_decoder_labels() -> (
    None
):
    # Stable contract: pipeline_smoke backend keeps the sequence term when
    # dense decoder labels are absent.
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights=_full_weights(),
        training_backend="pipeline_smoke",
    )
    batch = _batch(
        target_token_ids=torch.tensor([[2, 3, 0, 0]], dtype=torch.long),
        target_attention_mask=torch.tensor([[True, True, False, False]]),
    )

    losses = objective(
        model_output={"sequence_logits": torch.zeros((1, 4, 8))},
        batch=batch,
    )

    assert set(losses) == {"sequence", "total"}
    expected = torch.tensor(8.0).log()
    assert torch.allclose(losses["sequence"], expected)


def test_characterization_mlm_uses_collator_ignored_labels() -> None:
    # Token id zero is a real vocabulary token; padding must already be
    # marked with IGNORE_LABEL by the collator.
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        text_mlm_targets=torch.tensor(
            [[IGNORE_LABEL, 3, IGNORE_LABEL, 4]], dtype=torch.long
        ),
    )
    logits = torch.full((1, 4, 8), -10.0)
    logits[0, 1, 3] = 10.0
    logits[0, 3, 4] = 10.0

    losses = objective(
        model_output={"text_mlm_logits": logits},
        batch=batch,
    )

    assert set(losses) == {"text_mlm", "total"}
    assert losses["text_mlm"].item() < 1e-5

    zero_target = torch.tensor([[0]], dtype=torch.long)
    zero_logits = torch.full((1, 1, 8), -10.0)
    zero_logits[0, 0, 0] = 10.0
    losses_zero = objective(
        model_output={"text_mlm_logits": zero_logits},
        batch=_batch(text_mlm_targets=zero_target),
    )
    assert losses_zero["text_mlm"].item() < 1e-5


def test_characterization_all_ignored_mlm_targets_produce_no_term() -> None:
    # CURRENT BEHAVIOR: an all-IGNORE_LABEL target set yields no term and
    # the batch fails with the generic empty-losses error.
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        text_mlm_targets=torch.full((1, 4), IGNORE_LABEL, dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="no trainable loss terms"):
        objective(
            model_output={"text_mlm_logits": torch.zeros((1, 4, 8))},
            batch=batch,
        )


# --- stable contract: preference tuning stage -----------------------------


def test_characterization_preference_tuning_replaces_lm_term() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights=_full_weights(),
        training_stage="PREFERENCE_TUNING",
    )
    labels = torch.tensor(
        [[IGNORE_LABEL, IGNORE_LABEL, 2, 3]], dtype=torch.long
    )
    chosen = torch.zeros((1, 4, 6))
    chosen[0, 1, 2] = 5.0
    chosen[0, 2, 3] = 5.0
    rejected = torch.zeros((1, 4, 6))
    rejected[0, 1, 2] = -5.0
    rejected[0, 2, 3] = -5.0
    batch = _batch(
        decoder_labels=labels,
        chosen_labels=labels,
        rejected_labels=labels,
    )

    losses = objective(
        model_output={
            "sequence_logits": torch.zeros((1, 4, 6)),
            "chosen_sequence_logits": chosen,
            "rejected_sequence_logits": rejected,
        },
        batch=batch,
    )

    assert set(losses) == {"preference", "total"}
    assert torch.isfinite(losses["preference"])


def test_characterization_reference_model_dpo_fails_closed() -> None:
    # No reference log probabilities exist in this training pipeline, so a
    # direct "dpo" preference mode fails at construction time.
    with pytest.raises(
        RuntimeError, match="reference-model DPO requires explicit"
    ):
        SupervisedOrSelfSupervisedLoss(
            loss_weights=_full_weights(),
            preference_mode="dpo",
        )


# --- stable contract: auxiliary route is dead -----------------------------


def test_characterization_auxiliary_route_produces_no_terms() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())

    with pytest.raises(RuntimeError, match="no trainable loss terms") as exc:
        objective(
            model_output={
                "audio_aux": torch.zeros(1),
                "image_aux": torch.zeros(1),
            },
            batch=_batch(),
        )
    assert "audio_aux" in str(exc.value)


# --- stable contract: generation fail-closed ------------------------------


def test_characterization_generation_targets_fail_closed() -> None:
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=_full_weights())
    batch = _batch(
        task_types=["text_to_image"],
        output_modalities=[("image",)],
    )

    with pytest.raises(RuntimeError, match="generated outputs for .*image"):
        objective(
            model_output={"sequence_logits": torch.zeros((1, 4, 8))},
            batch=batch,
            require_targets_for_generation=True,
        )


# --- stable contract: build_loss translation ------------------------------


def test_characterization_build_loss_translates_settings_into_weights() -> (
    None
):
    from config.multimodal.training_settings import TrainingSettings
    from orchestration.composition.runtime.training import build_training_loss

    settings = TrainingSettings(
        training_stage="MULTIMODAL_PRETRAIN",
        mlm_loss_weight=0.25,
        language_modeling_loss_weight=0.5,
        ocr_sequence_loss_weight=0.75,
        image_patch_loss_weight=0.1,
        image_generation_loss_weight=0.65,
        audio_masked_loss_weight=0.2,
        audio_token_loss_weight=0.3,
        video_temporal_loss_weight=0.4,
        video_generation_loss_weight=0.6,
        safety_loss_weight=0.9,
        hard_negative_margin=0.3,
    )
    objective = build_training_loss(settings)

    assert objective.training_stage == "MULTIMODAL_PRETRAIN"
    assert objective.hard_negative_margin == 0.3
    assert objective.alignment_score_exponent == 1.0
    assert objective.contrastive_temperature == 0.07
    assert objective.loss_weights["language_modeling"] == 0.5
    assert objective.loss_weights["ocr_sequence"] == 0.75
    assert objective.loss_weights["sequence"] == 0.25
    assert objective.loss_weights["text_mlm"] == 0.25
    assert objective.loss_weights["image_generation"] == 0.65
    assert objective.loss_weights["image_reconstruction"] == 0.1
    assert objective.loss_weights["audio_generation"] == 0.3
    assert objective.loss_weights["audio_reconstruction"] == 0.2
    assert objective.loss_weights["video_generation"] == 0.6
    assert objective.loss_weights["video_temporal"] == 0.4
    assert objective.loss_weights["safety"] == 0.9
