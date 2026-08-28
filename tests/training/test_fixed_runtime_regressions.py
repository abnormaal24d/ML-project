"""Regression coverage for previously production-blocking runtime issues."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_settings import TrainingSettings
from training.losses.objective import (
    SUPPORTED_LOSS_KEYS,
    NonFiniteLossError,
    SupervisedOrSelfSupervisedLoss,
)
from training.runtime.checkpoint.io import atomic_torch_save
from training.runtime.checkpoint.metadata import (
    build_checkpoint_fingerprint_schema,
)
from training.runtime.checkpoint.service import load_model_weights
from training.runtime.checkpoint.state import (
    build_training_state_payload,
    validate_epoch_resume_state,
)


class _LossBatch:
    labels = torch.tensor([0])
    output_modalities: list[tuple[str, ...]] = []

    def __getattr__(self, _name: str) -> object | None:
        return None


def test_loss_terms_are_weighted_centrally() -> None:
    weights = {name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    weights["label"] = 2.5
    objective = SupervisedOrSelfSupervisedLoss(loss_weights=weights)
    logits = torch.tensor([[2.0, -1.0]])

    losses = objective(
        model_output={"label_logits": logits},
        batch=_LossBatch(),
    )

    expected = (
        torch.nn.functional.cross_entropy(logits, _LossBatch.labels) * 2.5
    )
    assert torch.allclose(losses["label"], expected)
    assert torch.allclose(losses["total"], expected)


def test_non_finite_loss_identifies_the_term() -> None:
    objective = SupervisedOrSelfSupervisedLoss(
        loss_weights={name: 1.0 for name in SUPPORTED_LOSS_KEYS}
    )
    logits = torch.tensor([[float("nan"), 0.0]])

    with pytest.raises(NonFiniteLossError, match="label"):
        objective(
            model_output={"label_logits": logits},
            batch=_LossBatch(),
        )


def test_epoch_checkpoint_rejects_mid_epoch_positions() -> None:
    state = SimpleNamespace(
        completed_epochs=1,
        completed_optimizer_steps=2,
        total_batches=3,
        final_loss=0.5,
        cumulative_loss_sum=0.5,
        epoch_losses=[0.5],
        epoch_history=[],
        best_metric=0.5,
        best_epoch=1,
        epochs_without_improvement=0,
        stop_reason=None,
        last_gradient_norm=None,
        gradient_clip_count=0,
    )
    plan = SimpleNamespace(to_dict=lambda: {})

    with pytest.raises(ValueError, match="mid-epoch checkpoint"):
        build_training_state_payload(
            state=state,
            val_loss=0.5,
            random_state={},
            training_plan=plan,
            sampler_position=1,
        )


def test_resume_rejects_legacy_nonzero_sampler_position() -> None:
    with pytest.raises(ValueError, match="mid-epoch resume"):
        validate_epoch_resume_state(
            {
                "resume_granularity": "epoch",
                "sampler_position": 4,
                "gradient_accumulation_position": 0,
            }
        )


def test_dataset_manifest_sha256_is_required_and_validated() -> None:
    from training.runtime.checkpoint.metadata import (
        resolve_dataset_fingerprint,
    )

    with pytest.raises(ValueError, match="64-character hex digest"):
        build_checkpoint_fingerprint_schema(
            model_settings=ModelSettings(),
            training_settings=TrainingSettings(),
            dataset_root=Path("/nonexistent"),
            dataset_manifest_sha256="not-a-digest",
        )

    assert (
        resolve_dataset_fingerprint(dataset_manifest_sha256="a" * 64)
        == "a" * 64
    )
    with pytest.raises(ValueError, match="64-character hex digest"):
        resolve_dataset_fingerprint(dataset_manifest_sha256="tooshort")


def test_distributed_descriptor_loads_into_plain_inference_model(
    tmp_path: Path,
) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "distributed.pt"
    atomic_torch_save(
        payload={
            "checkpoint_payload_version": 1,
            "checkpoint_format": "distributed_sharded_v2",
            "model_family": "multimodal_model",
            "artifact_version": "test-v1",
            "model_state": source.state_dict(),
        },
        checkpoint_path=checkpoint,
    )

    load_model_weights(
        model=target,
        checkpoint_path=checkpoint,
        model_settings=ModelSettings(),
    )

    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, target_parameter)


def test_fsdp_wrapper_uses_explicit_production_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.fsdp as fsdp_module

    from training.runtime.device import wrap_distributed_model

    captured: dict[str, object] = {}

    def fake_fsdp(model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        fsdp_module,
        "FullyShardedDataParallel",
        fake_fsdp,
    )
    settings = SimpleNamespace(
        precision="bf16",
        distributed_strategy="fsdp",
    )
    model = torch.nn.Linear(2, 2)

    result = wrap_distributed_model(
        model=model,  # type: ignore[arg-type]
        settings=settings,
        device=torch.device("cpu"),
        distributed_context={"enabled": True, "strategy": "fsdp"},
    )

    assert result is model
    assert captured["use_orig_params"] is True
    assert captured["limit_all_gathers"] is True
    assert captured["sync_module_states"] is False
    assert captured["auto_wrap_policy"] is not None
    assert captured["mixed_precision"] is not None
